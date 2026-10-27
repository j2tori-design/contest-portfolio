"""운영 관점 분석: 운영점 오류분석, 규칙 베이스라인, 부트스트랩 신뢰구간, 확률보정, 사전경보, 비용, 검증 보조.

모든 함수는 본 실험과 같은 홀드아웃 분할·SEED·사출 그룹 CV fold 를 사용한다.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from .config import INDEX_COL, N_REPEATS, N_SPLITS, PRODUCTS, SEED, TARGET
from .data import feature_matrix, production_blocks
from .evaluate import assert_disjoint, recall_at_k

OP_K = 0.20   # 운영점: 제품별 불량확률 상위 20% 를 검사


# ------------------------------------------------------------------ 공통
def topk_mask(p: np.ndarray, k: float) -> np.ndarray:
    """확률 상위 ceil(k·n) 개를 True 로 표시 (검사 대상)."""
    n = max(1, int(np.ceil(len(p) * k)))
    m = np.zeros(len(p), dtype=bool)
    m[np.argsort(-np.asarray(p), kind="stable")[:n]] = True
    return m


def operating_metrics(y, p, k=OP_K) -> dict:
    """상위 k% 를 '불량 판정'으로 볼 때의 Precision · Recall · F1 (운영점 지표)."""
    y = np.asarray(y)
    flag = topk_mask(p, k)
    tp = int((flag & (y == 1)).sum())
    prec = tp / max(1, flag.sum())
    rec = tp / max(1, y.sum())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {f"Precision@{int(k * 100)}%": prec, f"F1@{int(k * 100)}%": f1}


# ------------------------------------------------------------------ 운영점 기준 오류분석
def add_operating_point(oof: pd.DataFrame, prob_col="oof_prob", k=OP_K) -> pd.DataFrame:
    """제품별로 상위 k% 를 검사(=불량 판정)한다고 보고 TP/FN/FP/TN 을 다시 정의한다."""
    d = oof.copy()
    d["op_flag"] = False
    for p in PRODUCTS:
        m = (d["product"] == p).to_numpy()
        d.loc[m, "op_flag"] = topk_mask(d.loc[m, prob_col].to_numpy(), k)
    y = d[TARGET] == 1
    d["op_type"] = np.select([y & d.op_flag, y & ~d.op_flag, ~y & d.op_flag], ["TP", "FN", "FP"], "TN")
    return d


def _rates(g: pd.DataFrame) -> pd.Series:
    fails = (g[TARGET] == 1).sum()
    passes = (g[TARGET] == 0).sum()
    return pd.Series({"n": len(g), "불량": int(fails),
                      "TP": int((g.op_type == "TP").sum()), "FN": int((g.op_type == "FN").sum()),
                      "FP": int((g.op_type == "FP").sum()),
                      "FN 비율(불량 중)": (g.op_type == "FN").sum() / fails if fails else np.nan,
                      "FP 비율(양품 중)": (g.op_type == "FP").sum() / passes if passes else np.nan})


def concentration_tables(d: pd.DataFrame, variables, regime: pd.DataFrame | None = None, q=4) -> pd.DataFrame:
    """운영점 FN·FP 가 어디에 몰리는지: 공정변수 구간, 생산순번 구간, 캐비티, 운전구간 군집별 집계."""
    out = []
    for v in variables:
        b = pd.qcut(d[v], q, duplicates="drop")
        t = d.groupby(b, observed=True).apply(_rates)
        t.index = [f"{iv.left:.2f} ~ {iv.right:.2f}" for iv in t.index]
        out.append(t.assign(축="공정변수", 항목=v))
    for p in PRODUCTS:
        s = d[d["product"] == p]
        b = pd.qcut(s[INDEX_COL].rank(method="first"), 5, labels=[f"{i * 20}~{(i + 1) * 20}%" for i in range(5)])
        t = s.groupby(b, observed=True).apply(_rates)
        rng = s.groupby(b, observed=True)[INDEX_COL].agg(["min", "max"])
        t.index = [f"{i} (순번 {a}~{z})" for i, (a, z) in zip(t.index, rng.itertuples(index=False))]
        out.append(t.assign(축="생산순번 구간", 항목=p.upper()))
    t = d.groupby(["product", "cavity"]).apply(_rates)
    t.index = [f"{p.upper()} 캐비티{c}" for p, c in t.index]
    out.append(t.assign(축="캐비티", 항목="제품·캐비티"))
    if regime is not None:
        m = d.merge(regime[[INDEX_COL, "cluster"]], on=INDEX_COL, how="left")
        t = m.groupby(["product", "cluster"]).apply(_rates)
        t.index = [f"{p.upper()} 군집{int(c)}" for p, c in t.index]
        out.append(t.assign(축="운전구간 군집", 항목="제품·군집"))
    res = pd.concat(out).reset_index(names="구간")
    return res[["축", "항목", "구간", "n", "불량", "TP", "FN", "FP", "FN 비율(불량 중)", "FP 비율(양품 중)"]]


def error_rules(d: pd.DataFrame, features, kind="FN", depth=3, min_leaf=None) -> tuple[str, pd.DataFrame]:
    """얕은 의사결정나무로 오류가 몰리는 조건을 사람이 읽는 규칙으로 추출.

    FN: 불량 중 '놓친(FN) vs 잡은(TP)', FP: 양품 중 '오경보(FP) vs 정상(TN)'.
    """
    if kind == "FN":
        s = d[d[TARGET] == 1]
        target = (s.op_type == "FN").astype(int)
        min_leaf = min_leaf or 3
    else:
        s = d[d[TARGET] == 0]
        target = (s.op_type == "FP").astype(int)
        min_leaf = min_leaf or 30
    X = s[features]
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=SEED).fit(X, target)
    text = export_text(tree, feature_names=list(features), decimals=2, show_weights=True)
    leaf = tree.apply(X)
    tbl = (pd.DataFrame({"leaf": leaf, kind: target.to_numpy()}).groupby("leaf")[kind]
           .agg(["size", "sum"]).rename(columns={"size": "n", "sum": kind}))
    tbl[f"{kind} 비율"] = tbl[kind] / tbl["n"]
    return text, tbl.sort_values(f"{kind} 비율", ascending=False)


def shap_rank_by_group(sv: np.ndarray, X: pd.DataFrame, group: pd.Series, exclude=("cavity", "is_rg3"), top=8):
    """캐비티별(또는 제품별) 평균 |SHAP| 순위. cavity 처럼 조치로 이어지지 않는 변수는 제외해 공정변수 순위를 본다."""
    cols = [c for c in X.columns if c not in exclude]
    idx = [X.columns.get_loc(c) for c in cols]
    out = {}
    for gv in sorted(pd.unique(group)):
        m = (group == gv).to_numpy()
        s = pd.Series(np.abs(sv[m][:, idx]).mean(0), index=cols).sort_values(ascending=False).head(top)
        out[gv] = s
    return pd.DataFrame({k: list(zip(v.index, v.round(4))) for k, v in out.items()})


# ------------------------------------------------------------------ 단순 규칙 베이스라인
def rule_baselines(oof: pd.DataFrame, prob_col="oof_prob", early_ns=(50, 100, 150)) -> pd.DataFrame:
    """규칙 기반 검사(캐비티0 전수, 가동 초기 N 사출, 조합)의 검사 비율·포착률과,
    같은 검사량에서 모델(확률 상위)의 포착률을 비교한다. 제품별 + 전체."""
    rows = []
    for scope in ["all"] + PRODUCTS:
        d = oof if scope == "all" else oof[oof["product"] == scope]
        y = d[TARGET].to_numpy()
        early_rank = d.groupby("product")["shot_id"].rank(method="dense")      # 제품 내 사출 순서
        rules = {"추정 캐비티0 전수검사": (d["cavity"] == 0).to_numpy()}
        for n in early_ns:
            rules[f"가동 초기 {n}사출 전수검사"] = (early_rank <= n).to_numpy()
            rules[f"가동 초기 {n}사출 + 이후 추정 캐비티0"] = ((early_rank <= n) | (d["cavity"] == 0)).to_numpy()
        for name, flag in rules.items():
            rate = flag.mean()
            rule_rec = (flag & (y == 1)).sum() / y.sum() if y.sum() else np.nan
            # 같은 검사량에서 모델: 제품별 상위 rate 를 검사
            mflag = np.zeros(len(d), dtype=bool)
            for p in d["product"].unique():
                m = (d["product"] == p).to_numpy()
                mflag[m] = topk_mask(d.loc[m, prob_col].to_numpy(), rate)
            model_rec = (mflag & (y == 1)).sum() / y.sum() if y.sum() else np.nan
            rows.append({"scope": scope, "rule": name, "검사 비율": rate, "규칙 포착률": rule_rec,
                         "모델 포착률(같은 검사량)": model_rec, "모델 추가 포착(건)": int((mflag & (y == 1)).sum() - (flag & (y == 1)).sum()),
                         "불량 수": int(y.sum())})
    return pd.DataFrame(rows)


def gain_curve(y, p, grid=None):
    """검사 비율별 누적 포착률."""
    grid = np.linspace(0, 1, 101) if grid is None else grid
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="stable")
    cum = np.concatenate([[0], np.cumsum(y[order])]) / max(1, y.sum())
    return grid, cum[np.minimum((grid * len(y)).round().astype(int), len(y))]


# ------------------------------------------------------------------ 테스트 부트스트랩 신뢰구간
def bootstrap_test_ci(pred: pd.DataFrame, prob_col="fail_prob", thr=None, n_boot=1000, seed=SEED) -> pd.DataFrame:
    """사출(shot) 단위 복원추출로 테스트 지표의 95% 신뢰구간. 불량이 없는 표본은 해당 지표에서 제외."""
    rng = np.random.default_rng(seed)
    pred = pred.copy()
    pred["_g"] = pred["product"] + "_" + pred["shot_id"].astype(str)
    rows = []
    for scope in ["all"] + PRODUCTS:
        d = pred if scope == "all" else pred[pred["product"] == scope]
        groups = d["_g"].unique()
        idx_by_g = d.groupby("_g").indices
        y0, p0 = d[TARGET].to_numpy(), d[prob_col].to_numpy()

        def stats(y, p):
            if y.sum() == 0 or y.sum() == len(y):
                return None
            s = {"PR_AUC": average_precision_score(y, p), "ROC_AUC": roc_auc_score(y, p),
                 "Recall@20%": recall_at_k(y, p, 0.20)}
            if thr is not None:
                s["F1(임계값)"] = f1_score(y, p >= thr, zero_division=0)
            return s

        point = stats(y0, p0)
        boots = []
        for _ in range(n_boot):
            gs = rng.choice(groups, size=len(groups), replace=True)
            ii = np.concatenate([idx_by_g[g] for g in gs])
            s = stats(y0[ii], p0[ii])
            if s:
                boots.append(s)
        b = pd.DataFrame(boots)
        for met in point:
            rows.append({"scope": scope, "metric": met, "point": point[met],
                         "ci_low": b[met].quantile(0.025), "ci_high": b[met].quantile(0.975),
                         "valid_boot": len(b), "불량 수": int(y0.sum()), "무작위 PR-AUC": y0.mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 확률보정
def _inner_splits(groups, y, seed):
    return list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def calibrated(model, groups, y, method, seed=SEED):
    """보정용 내부 CV 도 사출 단위로 묶은 CalibratedClassifierCV."""
    return CalibratedClassifierCV(clone(model), method=method, cv=_inner_splits(groups, y, seed))


def calibration_comparison(dev: pd.DataFrame, model, use_cavity: bool):
    """보정 없음 / sigmoid / isotonic 을 같은 25개 fold 에서 비교 (Brier·PR-AUC·Recall@20%).
    반환: fold 별 지표, 반복 평균 OOF 확률(보정 방식별)."""
    X = feature_matrix(dev, use_cavity)
    y, g = dev[TARGET].to_numpy(), dev["group"].to_numpy()
    oof = {m: np.zeros(len(y)) for m in ["raw", "sigmoid", "isotonic"]}
    rows = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for f, (tr, va) in enumerate(cv.split(X, y, g)):
            preds = {"raw": clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]}
            for m in ["sigmoid", "isotonic"]:
                preds[m] = calibrated(model, g[tr], y[tr], m, SEED + r).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            for m, p in preds.items():
                oof[m][va] += p / N_REPEATS
                rows.append({"method": m, "repeat": r, "fold": f, "Brier": brier_score_loss(y[va], p),
                             "PR_AUC": average_precision_score(y[va], p), "Recall@20%": recall_at_k(y[va], p, 0.20),
                             "mean_pred": p.mean(), "obs_rate": y[va].mean()})
    oof_df = dev[[INDEX_COL, "product", "cavity", TARGET]].copy()
    for m, v in oof.items():
        oof_df[f"prob_{m}"] = v
    return pd.DataFrame(rows), oof_df


# ------------------------------------------------------------------ 사전경보
def time_block_oof(dev: pd.DataFrame, model, use_cavity: bool, n_blocks=8, n_splits=4, seed=SEED) -> np.ndarray:
    """연속 생산 구간을 통째로 평가하는 생산순번 블록 CV 의 OOF 점수 (같은 시기 사출을 학습에서 뺀 조건, 1회 반복)."""
    X = feature_matrix(dev, use_cavity)
    y = dev[TARGET].to_numpy()
    blk = production_blocks(dev, n_blocks).astype(str)          # 추정 사출 그룹 단위로 구간 분할
    groups = (dev["product"] + "_b" + blk).to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, va in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        oof[va] = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return oof


def process_instability(dev: pd.DataFrame, variables, window=10) -> np.ndarray:
    """라벨 없이 공정값만으로 만든 '공정 흔들림' 신호: 각 사출의 공정값이 직전 window 사출 평균에서
    얼마나 벗어났는지(표준화 단위 평균 절대 편차). 과거 입력만 쓰므로 처음 보는 구간에도 실시간 적용 가능."""
    shots = dev.drop_duplicates(["product", "shot_id"]).sort_values(["product", INDEX_COL])
    parts = []
    for _, s in shots.groupby("product"):
        past = s[variables].shift(1).rolling(window, min_periods=3)
        parts.append(pd.Series((s[variables] - past.mean()).abs().mean(axis=1).values, index=s.index))
    shots = shots.assign(instab=pd.concat(parts))
    return (dev[["product", "shot_id"]].merge(shots[["product", "shot_id", "instab"]], how="left")
            ["instab"].fillna(0).to_numpy())


def early_warning(dev: pd.DataFrame, prob: np.ndarray, window=10, q=0.90, cusum_k=0.5, signal="신호"):
    """제품별 생산순서에서 예측확률의 이동평균·CUSUM 으로 '불안정 구간 진입' 알람.

    - 이동평균 알람: 직전 window 사출의 평균 확률 > 그 제품 이동평균의 q 분위수
    - CUSUM 알람: 표준화 확률(z)의 누적합 S_t = max(0, S_{t-1} + z_t - k) > h (h = 그 제품 S 의 q 분위수)
    임계 분위수는 운영 전 과거 데이터로 정하는 값이며 여기서는 개발셋 전체 분포로 설정(가정).
    """
    d = dev[[INDEX_COL, "product", "shot_id", TARGET]].copy()
    d["prob"] = prob
    shots = (d.groupby(["product", "shot_id"]).agg(idx=(INDEX_COL, "min"), prob=("prob", "max"), fails=(TARGET, "sum"))
             .reset_index().sort_values(["product", "idx"]))
    series, summ = [], []
    for p, s in shots.groupby("product"):
        s = s.copy()
        s["ma"] = s["prob"].rolling(window, min_periods=3).mean()
        z = (s["prob"] - s["prob"].mean()) / (s["prob"].std() + 1e-9)
        S, acc = [], 0.0
        for v in z:
            acc = max(0.0, acc + v - cusum_k)
            S.append(acc)
        s["cusum"] = S
        s["alarm_ma"] = s["ma"] > s["ma"].quantile(q)
        s["alarm_cusum"] = s["cusum"] > s["cusum"].quantile(q)
        s["signal"] = signal
        series.append(s)
        fail_idx = s.loc[s.fails > 0, "idx"].to_numpy()
        first_fail = fail_idx.min() if len(fail_idx) else np.nan
        for col in ["alarm_ma", "alarm_cusum"]:
            al = s.loc[s[col], "idx"].to_numpy()
            first_alarm = al.min() if len(al) else np.nan
            # 각 불량 사출 직전 window 사출 안에 알람이 있었는지 (사전 감지율)
            pos = s.reset_index(drop=True)
            detected = 0
            for i in pos.index[pos.fails > 0]:
                if pos.loc[max(0, i - window):i, col].any():
                    detected += 1
            # 알람 사출 중 직후 window 사출 안에 불량이 있는 비율 (알람 정밀도)
            prec_hits = 0
            al_pos = pos.index[pos[col]]
            for i in al_pos:
                if pos.loc[i:i + window, "fails"].sum() > 0:
                    prec_hits += 1
            summ.append({"signal": signal, "product": p, "rule": "이동평균" if col == "alarm_ma" else "CUSUM",
                         "사출 수": len(s), "알람 사출 비율": s[col].mean(),
                         "첫 알람 순번": first_alarm, "첫 불량 순번": first_fail,
                         "불량 사출 중 사전 감지 비율": detected / max(1, (pos.fails > 0).sum()),
                         "알람 후 window 내 불량 비율": prec_hits / max(1, len(al_pos))})
    return pd.concat(series, ignore_index=True), pd.DataFrame(summ)


# ------------------------------------------------------------------ 비용 기반 검사 비율
def cost_curves(oof: pd.DataFrame, prob_col="oof_prob", ratios=(5, 10, 20, 50, 100)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """검사 1건 비용 = 1, 불량 유출 1건 비용 = ratio 로 가정한 총비용 곡선과 최적 검사 비율 (제품별 상위 k% 검사)."""
    grid = np.round(np.linspace(0, 1, 101), 2)
    rows = []
    for p in PRODUCTS:
        d = oof[oof["product"] == p]
        y, prob = d[TARGET].to_numpy(), d[prob_col].to_numpy()
        _, cap = gain_curve(y, prob, grid)
        missed = (1 - cap) * y.sum()
        for r in ratios:
            for k, m in zip(grid, missed):
                rows.append({"product": p, "ratio": r, "검사 비율": k, "총비용": k * len(y) + r * m,
                             "전수검사 비용": len(y), "무검사 비용": r * y.sum()})
    curves = pd.DataFrame(rows)
    best = curves.loc[curves.groupby(["product", "ratio"])["총비용"].idxmin()].reset_index(drop=True)
    # 단순 대안 = 전수검사·무검사 중 더 싼 쪽
    best["단순 대안 대비 절감"] = 1 - best["총비용"] / best[["전수검사 비용", "무검사 비용"]].min(axis=1)
    return curves, best


# ------------------------------------------------------------------ 이상치 진단
def outlier_table(df: pd.DataFrame, variables, z=4.0) -> pd.DataFrame:
    """표준화 값 |z| > 4 인 행 수와 그 행들의 불량률 (제거하지 않는 근거 확인용)."""
    rows = []
    for v in variables:
        m = df[v].abs() > z
        rows.append({"변수": v, "|z|>4 행 수": int(m.sum()), "해당 행 불량": int(df.loc[m, TARGET].sum()),
                     "해당 행 불량률": df.loc[m, TARGET].mean() if m.any() else np.nan,
                     "최대 |z|": df[v].abs().max()})
    t = pd.DataFrame(rows).sort_values("|z|>4 행 수", ascending=False)
    return t


# ------------------------------------------------------------------ 사출 공정 단계별 변수 분류
STAGES = {
    "가소화(계량)": ["Plasticizing_Time", "Plasticizing_Position", "Max_Screw_RPM", "Average_Screw_RPM",
                 "Max_Back_Pressure", "Average_Back_Pressure", "Hopper_Temperature",
                 *[f"Barrel_Temperature_{i}" for i in range(1, 7)]],
    "충전(사출)": ["Injection_Time", "Filling_Time", "Max_Injection_Speed", "Max_Injection_Pressure"],
    "보압(V/P 전환)": ["Max_Switch_Over_Pressure", "Cushion_Position"],
    "냉각(금형온도)": ["Mold_Temperature_3", "Mold_Temperature_4"],
    "형개폐·사이클": ["Clamp_Close_Time", "Clamp_Open_Position", "Cycle_Time"],
}


def variable_stage_table(df: pd.DataFrame) -> pd.DataFrame:
    """공정변수를 사출 단계로 묶고, 상수 여부(→ 제거)와 제품별 불량-양품 평균 차이(표준화 단위)를 함께 표시."""
    rows = []
    for stage, cols in STAGES.items():
        for c in cols:
            r = {"단계": stage, "변수": c, "고유값 수": int(df[c].nunique())}
            for p in PRODUCTS:
                d = df[df["product"] == p]
                r[f"{p.upper()} 불량-양품 차이"] = d.loc[d[TARGET] == 1, c].mean() - d.loc[d[TARGET] == 0, c].mean()
            r["처리"] = "상수 → 제거" if r["고유값 수"] == 1 else "사용"
            rows.append(r)
    return pd.DataFrame(rows)



# ------------------------------------------------------------------ 과거 → 미래 대리 검증 (생산 순번 기준)
def forward_validation(dev: pd.DataFrame, model, use_cavity: bool, n_segments: int = 5) -> pd.DataFrame:
    """expanding-window: 제품별 생산 순번을 n개 구간(추정 사출 그룹 단위)으로 나누고,
    구간 0..s-1 로 학습해 구간 s 를 평가한다 (s = 1..n-1). 실제 시각이 없어 생산 순번을 시간의 대리로 쓴다.
    평가 구간·학습 구간에 불량이 없으면 지표를 N/A 로 두고 사유를 남긴다. 임계값은 학습 구간 내부 그룹 CV 로 정한다."""
    X = feature_matrix(dev, use_cavity)
    y, g, prod = dev[TARGET].to_numpy(), dev["group"].to_numpy(), dev["product"].to_numpy()
    seg = production_blocks(dev, n_segments).to_numpy()
    rows = []
    for s_ in range(1, n_segments):
        tr, va = np.flatnonzero(seg < s_), np.flatnonzero(seg == s_)
        assert_disjoint(g, tr, va, f"forward s{s_}")
        p = None
        if y[tr].sum() >= 2:
            p = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        for scope in ["all", *PRODUCTS]:
            m = np.ones(len(va), bool) if scope == "all" else prod[va] == scope
            yy = y[va][m]
            row = {"평가 구간": s_, "학습 구간": f"0~{s_ - 1}", "scope": scope, "평가 행 수": int(m.sum()),
                   "평가 사출 수": int(len(np.unique(g[va][m]))), "평가 불량 수": int(yy.sum()),
                   "학습 불량 수": int(y[tr][prod[tr] == scope].sum()) if scope != "all" else int(y[tr].sum())}
            if p is None:
                row["사유"] = "학습 구간 불량 2건 미만 → 학습 불가"
            elif yy.sum() == 0:
                row.update({"사유": "평가 구간 불량 0건 → PR-AUC·Recall 정의 불가 (N/A)",
                            "상위20% 검사 시 오경보 수": int(topk_mask(p[m], 0.20).sum())})
            else:
                pm = p[m]
                row.update({"사유": "", "PR_AUC": average_precision_score(yy, pm), "무작위 PR_AUC": yy.mean(),
                            "ROC_AUC": roc_auc_score(yy, pm) if yy.sum() < len(yy) else np.nan,
                            "Recall@20%": recall_at_k(yy, pm, 0.20)})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 분할 누수 점검
def split_audit(df: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, n_repeats: int = N_REPEATS, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """모든 분할에서 같은 추정 사출 그룹(공정값이 같은 짝 포함)이 학습·평가 양쪽에 들어가지 않는지 확인."""
    rows = []

    def add(name, n, overlap):
        rows.append({"분할": name, "검사한 분할 수": n, "겹친 그룹 수": overlap, "결과": "통과" if overlap == 0 else "위반"})

    add("개발 / 테스트 (홀드아웃)", 1, len(set(dev["group"]) & set(test["group"])))
    # 그룹 정의가 놓친 짝이 없는지: 같은 공정값 행이 개발·테스트 양쪽에 있는가
    from .data import process_cols, load_labeled
    cols = process_cols(load_labeled(PRODUCTS[0]))
    key = lambda d: set(map(tuple, np.round(d[["product"] + cols].assign(product=d["product"].map({"cn7": 0, "rg3": 1})).to_numpy(), 8)))
    add("개발 / 테스트 (같은 공정값 행 기준)", 1, len(key(dev) & key(test)))
    y, g = dev[TARGET].to_numpy(), dev["group"].to_numpy()
    n_out = n_in = ov_out = ov_in = 0
    for r in range(n_repeats):
        for tr, va in StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED + r).split(np.zeros(len(y)), y, g):
            n_out += 1
            ov_out += len(set(g[tr]) & set(g[va]))
            for itr, iva in _inner_splits(g[tr], y[tr], SEED + r):      # 임계값·보정·스태킹 내부 분할
                n_in += 1
                ov_in += len(set(g[tr][itr]) & set(g[tr][iva]))
    add(f"외부 CV ({n_repeats}회 × {n_splits}-fold)", n_out, ov_out)
    add("내부 3-fold (임계값 선정·확률 보정·스태킹 메타 학습)", n_in, ov_in)
    for nb in (5, 8, 16):
        b = production_blocks(dev, nb)
        add(f"생산순번 블록 {nb}개: 그룹이 두 블록에 걸침", 1, int((b.groupby(dev["group"]).nunique() > 1).sum()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ OOF 판정 관점 분리
def oof_threshold_views(dev: pd.DataFrame, fold_preds: pd.DataFrame, oof_mean: np.ndarray, common_thr: float) -> pd.DataFrame:
    """세 가지 판정 관점의 혼동행렬을 구분해 기록한다.
    A. 외부 fold 임계값: 각 fold 학습 데이터 내부에서 정한 임계값으로 그 fold 를 판정. CV 요약과 같은 기준
    B. 반복 평균 점수 + 공통 임계값: 개발셋 전체에서 정한 임계값을 반복 평균 점수에 적용. 진단용
    C. 제품별 상위 20% 검사: 반복 평균 점수의 제품별 상위 20%. 운영 방식과 같은 기준"""
    y = dev[TARGET].to_numpy()
    fp = fold_preds.assign(y=y[fold_preds["row"]])
    rows = []
    for r, d in fp.groupby("repeat"):
        rows.append({"관점": "A. 외부 fold 임계값 (반복별)", "repeat": r,
                     "TP": int(((d.y == 1) & (d.pred_fold_thr == 1)).sum()), "FN": int(((d.y == 1) & (d.pred_fold_thr == 0)).sum()),
                     "FP": int(((d.y == 0) & (d.pred_fold_thr == 1)).sum()), "TN": int(((d.y == 0) & (d.pred_fold_thr == 0)).sum()),
                     "임계값": f"{d.fold_threshold.min():.3f}~{d.fold_threshold.max():.3f}"})
    pb = oof_mean >= common_thr
    rows.append({"관점": "B. 반복 평균 점수 + 공통 임계값 (진단용)", "repeat": "-",
                 "TP": int((pb & (y == 1)).sum()), "FN": int((~pb & (y == 1)).sum()),
                 "FP": int((pb & (y == 0)).sum()), "TN": int((~pb & (y == 0)).sum()), "임계값": f"{common_thr:.3f}"})
    op = add_operating_point(dev.assign(oof_prob=oof_mean))
    rows.append({"관점": "C. 제품별 상위 20% 검사 (운영 기준)", "repeat": "-",
                 **{t: int((op.op_type == t).sum()) for t in ["TP", "FN", "FP", "TN"]}, "임계값": "제품별 순위"})
    out = pd.DataFrame(rows)
    out["Recall"] = out.TP / (out.TP + out.FN)
    out["Precision"] = np.where(out.TP + out.FP > 0, out.TP / (out.TP + out.FP).replace(0, np.nan), np.nan)
    out["검사(양성 판정) 비율"] = (out.TP + out.FP) / (out.TP + out.FN + out.FP + out.TN)
    return out


# ------------------------------------------------------------------ 검사 용량별 포착 (두 할당 정책)
def capacity_table(d: pd.DataFrame, prob_col="oof_prob", ks=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """검사 용량 k 별 검사 수·포착·미탐지·포착률.
    - 제품별 할당: 제품마다 상위 ceil(k·n_제품) 개 검사 (제품별 용량 보장)
    - 통합 순위  : 두 제품을 한 줄로 세워 상위 ceil(k·n) 개 검사 (점수가 높은 제품에 용량이 몰릴 수 있음)
    동점은 생산 순번이 앞선 제품을 먼저 검사한다(안정 정렬). 실제 검사 비율은 올림 때문에 k 보다 약간 클 수 있다."""
    d = d.sort_values(["product", INDEX_COL]).reset_index(drop=True)
    rows = []
    for k in ks:
        flags = {"제품별 할당": np.zeros(len(d), bool)}
        for p in PRODUCTS:
            m = (d["product"] == p).to_numpy()
            flags["제품별 할당"][m] = topk_mask(d.loc[m, prob_col].to_numpy(), k)
        order = np.lexsort((d[INDEX_COL].to_numpy(), -d[prob_col].to_numpy()))
        f = np.zeros(len(d), bool)
        f[order[:int(np.ceil(len(d) * k))]] = True
        flags["통합 순위"] = f
        for pol, fl in flags.items():
            for scope in ["all", *PRODUCTS]:
                m = np.ones(len(d), bool) if scope == "all" else (d["product"] == scope).to_numpy()
                yy, ff = d.loc[m, TARGET].to_numpy(), fl[m]
                caught = int((ff & (yy == 1)).sum())
                rows.append({"정책": pol, "검사 용량": k, "scope": scope, "대상 수": int(m.sum()), "검사 수": int(ff.sum()),
                             "실제 검사 비율": ff.mean(), "불량 수": int(yy.sum()), "포착 불량": caught,
                             "미탐지 불량": int(yy.sum()) - caught,
                             "포착률": caught / yy.sum() if yy.sum() else np.nan,
                             "검사 중 불량 비율": caught / ff.sum() if ff.sum() else np.nan})
    return pd.DataFrame(rows)


def cost_breakeven(d: pd.DataFrame, prob_col: str, ratios=(5, 10, 20, 50, 100)) -> pd.DataFrame:
    """손익분기: 검사 비용 1, 유출 비용 r 이면 '보정 확률 p × r ≥ 1', 즉 p ≥ 1/r 인 제품만 검사하는 것이 기대비용상 이득.
    보정 확률이 실제 빈도와 맞는다는 가정이 필요하다."""
    rows = []
    for r in ratios:
        for scope in ["all", *PRODUCTS]:
            s = d if scope == "all" else d[d["product"] == scope]
            flag = s[prob_col].to_numpy() >= 1 / r
            yy = s[TARGET].to_numpy()
            rows.append({"유출:검사 비용비": f"{r}:1", "손익분기 확률(1/r)": 1 / r, "scope": scope,
                         "검사 대상 수": int(flag.sum()), "검사 비율": flag.mean(),
                         "포착률": (flag & (yy == 1)).sum() / yy.sum() if yy.sum() else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ SHAP 기여 방향 (제품별)
def shap_direction(sv: np.ndarray, X: pd.DataFrame, product: pd.Series, top: int = 6) -> pd.DataFrame:
    """제품별 상위 변수의 기여 방향: 변수값과 SHAP 의 순위상관(+면 값이 클수록 위험점수↑).
    인과 관계를 뜻하지 않으며, 상관 변수끼리 기여가 나뉠 수 있다."""
    from scipy.stats import spearmanr

    rows = []
    cols = [c for c in X.columns if c not in ("is_rg3",)]
    for p in PRODUCTS:
        m = (product == p).to_numpy()
        imp = pd.Series(np.abs(sv[m][:, [X.columns.get_loc(c) for c in cols]]).mean(0), index=cols).sort_values(ascending=False)
        for c in imp.index[:top]:
            j = X.columns.get_loc(c)
            rho = spearmanr(X.loc[m, c], sv[m, j]).correlation if X.loc[m, c].nunique() > 1 else np.nan
            rows.append({"product": p, "변수": c, "평균 |SHAP|": imp[c], "값-기여 순위상관": rho,
                         "방향": "값↑ → 점수↑" if rho > 0.3 else ("값↑ → 점수↓" if rho < -0.3 else "비단조/약함")})
    return pd.DataFrame(rows)
