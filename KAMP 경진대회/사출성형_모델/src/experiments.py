"""부가 실험: 검증 누수 비교, 비라벨 데이터 진단, 운전구간 군집, 오토인코더 비교, 예측 기여 변수, 유의성, 블록 검증."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .config import INDEX_COL, N_REPEATS, N_SPLITS, PRODUCTS, SEED, SEMI_REPEATS, TARGET, TB_REPEATS
from .data import add_shot_structure, feature_matrix, load_labeled, load_unlabeled, process_cols, production_blocks
from .evaluate import recall_at_k
from .models import get_models


def _scores(y, p, products):
    """전체·제품별 PR-AUC (해당 범위에 불량이 있을 때만)."""
    out = {"PR_AUC_all": average_precision_score(y, p)}
    for prod in PRODUCTS:
        m = products == prod
        out[f"PR_AUC_{prod}"] = average_precision_score(y[m], p[m]) if y[m].sum() else np.nan
    out["Recall@20%"] = recall_at_k(y, p, 0.20)
    return out


# ------------------------------------------------------------------ 1. 검증 누수 비교
def leakage_comparison(dev: pd.DataFrame) -> pd.DataFrame:
    """행 단위 CV(같은 사출의 두 제품이 train/valid 로 갈라짐) vs 사출 단위 그룹 CV."""
    X = feature_matrix(dev, use_cavity=True)
    y, g = dev[TARGET].to_numpy(), dev["group"].to_numpy()
    rows = []
    for name in ["LR_baseline", "RandomForest", "ExtraTrees"]:
        model = get_models()[name]
        for scheme in ["row", "group"]:
            for r in range(N_REPEATS):
                if scheme == "row":
                    splits = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + r).split(X, y)
                else:
                    splits = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r).split(X, y, g)
                for f, (tr, va) in enumerate(splits):
                    p = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
                    rows.append({"model": name, "scheme": scheme, "repeat": r, "fold": f,
                                 "PR_AUC": average_precision_score(y[va], p)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. 비라벨 데이터 진단
def domain_discrimination() -> pd.DataFrame:
    """라벨/비라벨 행을 구분하는 분류기의 CV AUC. 0.5 = 같은 분포, 1.0 = 완전히 다른 분포.

    rank: 파일별 분위(순위) 변환 → 표준화 방식 차이를 제거한 뒤에도 구분되는지 확인.
    """
    rows = []
    for p in PRODUCTS:
        L, U = load_labeled(p), load_unlabeled(p)
        cols = process_cols(L)
        Us = U.sample(len(L) * 3, random_state=SEED)
        y = np.r_[np.zeros(len(L)), np.ones(len(Us))]
        for mode in ["raw", "rank"]:
            if mode == "raw":
                A = pd.concat([L[cols], Us[cols]])
            else:
                A = pd.concat([L[cols].rank(pct=True), Us[cols].rank(pct=True)])
            clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
            prob = cross_val_predict(clf, A, y, method="predict_proba",
                                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
            rows.append({"product": p, "transform": mode, "domain_AUC": roc_auc_score(y, prob)})
    return pd.DataFrame(rows)


def _unlabeled_features(columns) -> pd.DataFrame:
    parts = []
    for p in PRODUCTS:
        u = add_shot_structure(load_unlabeled(p))
        u["product"] = p
        parts.append(u)
    U = pd.concat(parts, ignore_index=True)
    return feature_matrix(U, use_cavity=True)[columns]


def self_training(dev: pd.DataFrame, neg_cap: int = 5000) -> pd.DataFrame:
    """자기학습(가짜 라벨) 실험. 라벨 데이터로 학습한 RF 가 비라벨에 매긴 확률 상·하위를
    불량·양품 가짜 라벨로 추가해 재학습하고, 같은 그룹 CV fold 에서 라벨만 쓴 모델과 비교."""
    X = feature_matrix(dev, use_cavity=True)
    y, g, prod = dev[TARGET].to_numpy(), dev["group"].to_numpy(), dev["product"].to_numpy()
    XU = _unlabeled_features(X.columns)
    rf = get_models()["RandomForest"]
    settings = {"strict": (0.995, 0.50), "loose": (0.98, 0.80)}   # (불량 분위, 양품 분위)
    rng = np.random.default_rng(SEED)
    rows = []
    for r in range(SEMI_REPEATS):
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for f, (tr, va) in enumerate(cv.split(X, y, g)):
            base = clone(rf).fit(X.iloc[tr], y[tr])
            rows.append({"method": "labeled_only", "repeat": r, "fold": f,
                         **_scores(y[va], base.predict_proba(X.iloc[va])[:, 1], prod[va])})
            pu = base.predict_proba(XU)[:, 1]
            for name, (qpos, qneg) in settings.items():
                pos = np.flatnonzero(pu >= np.quantile(pu, qpos))
                neg = np.flatnonzero(pu <= np.quantile(pu, qneg))
                neg = rng.choice(neg, size=min(neg_cap, len(neg)), replace=False)
                Xa = pd.concat([X.iloc[tr], XU.iloc[pos], XU.iloc[neg]])
                ya = np.r_[y[tr], np.ones(len(pos)), np.zeros(len(neg))]
                p2 = clone(rf).fit(Xa, ya).predict_proba(X.iloc[va])[:, 1]
                rows.append({"method": f"self_training_{name}", "repeat": r, "fold": f,
                             "pseudo_pos": len(pos), "pseudo_neg": len(neg),
                             "unlabeled_mean_prob": pu.mean(), **_scores(y[va], p2, prod[va])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. 운전구간 군집
def regime_clusters(k_range=range(2, 7)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """라벨 데이터 안에서 공정값만으로 K-평균 군집 → 군집별 불량률·생산순번 구간.
    라벨을 쓰지 않고 찾은 운전구간이 불량률과 연결되는지 확인한다."""
    summary, assign = [], []
    for p in PRODUCTS:
        L = load_labeled(p)
        cols = [c for c in process_cols(L) if L[c].nunique() > 1]
        sil = {k: silhouette_score(L[cols], KMeans(k, n_init=10, random_state=SEED).fit_predict(L[cols]))
               for k in k_range}
        k = max(sil, key=sil.get)
        L["cluster"] = KMeans(k, n_init=10, random_state=SEED).fit_predict(L[cols])
        t = L.groupby("cluster").agg(rows=(TARGET, "size"), fails=(TARGET, "sum"), fail_rate=(TARGET, "mean"),
                                     idx_start=(INDEX_COL, "min"), idx_end=(INDEX_COL, "max")).reset_index()
        t.insert(0, "product", p)
        t["k"], t["silhouette"] = k, sil[k]
        summary.append(t)
        assign.append(L[[INDEX_COL, TARGET, "cluster"]].assign(product=p))
    return pd.concat(summary, ignore_index=True), pd.concat(assign, ignore_index=True)


# ------------------------------------------------------------------ 4. 오토인코더 (비지도 이상탐지)
def autoencoder_cv(dev: pd.DataFrame) -> pd.DataFrame:
    """양품만으로 학습한 오토인코더의 복원 오차를 불량 점수로 사용. 지도학습 모델과 같은 그룹 CV fold 에서 평가."""
    X = feature_matrix(dev, use_cavity=False)
    y, g, prod = dev[TARGET].to_numpy(), dev["group"].to_numpy(), dev["product"].to_numpy()
    rows = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for f, (tr, va) in enumerate(cv.split(X, y, g)):
            normal = tr[y[tr] == 0]
            sc = StandardScaler().fit(X.iloc[normal])
            Xn, Xv = sc.transform(X.iloc[normal]), sc.transform(X.iloc[va])
            ae = MLPRegressor(hidden_layer_sizes=(16, 6, 16), activation="relu", alpha=1e-3,
                              max_iter=600, early_stopping=True, random_state=SEED + r).fit(Xn, Xn)
            err = ((ae.predict(Xv) - Xv) ** 2).mean(axis=1)
            rows.append({"model": "AutoEncoder", "repeat": r, "fold": f, **_scores(y[va], err, prod[va])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. 개별 제품 예측 기여 변수
def shap_positive(model, X: pd.DataFrame) -> np.ndarray:
    """불량(1) 클래스 기준 SHAP 값 (행 × 변수).

    트리 모델은 TreeExplainer 로 계산하고, 소프트 보팅 앙상블은 확률의 평균이므로
    구성 모델 SHAP 값의 평균이 앙상블의 SHAP 값과 같다.
    """
    import shap

    if hasattr(model, "estimators_") and hasattr(model, "voting"):
        return np.mean([shap_positive(m, X) for m in model.estimators_], axis=0)
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    elif np.ndim(sv) == 3:
        sv = sv[..., 1]
    return np.asarray(sv)


def top_factors(model, X: pd.DataFrame, k: int = 3, exclude=("is_rg3",)) -> pd.DataFrame:
    """제품별로 위험점수를 가장 크게 올린 변수 k개 (SHAP 양의 기여). 인과적 원인이 아니라 예측 기여 변수.
    is_rg3(제품 구분)는 제외하고, cavity 는 추정값임을 표시한다."""
    sv = shap_positive(model, X)
    keep = [i for i, c in enumerate(X.columns) if c not in exclude]
    cols = np.array(X.columns)[keep]
    sv = sv[:, keep]
    out = []
    for vals in sv:
        order = np.argsort(-vals)[:k]
        out.append([f"{'추정 ' if cols[j] == 'cavity' else ''}{cols[j]}(+{vals[j]:.3f})" if vals[j] > 0 else ""
                    for j in order])
    return pd.DataFrame(out, columns=[f"contrib_{i + 1}" for i in range(k)], index=X.index)


# ------------------------------------------------------------------ 6. 모델 간 유의성 검정
def corrected_t(diff, n_splits: int = N_SPLITS) -> float:
    """Nadeau-Bengio 보정 재표본 t-검정 p 값 (짝지은 fold 차이)."""
    from scipy.stats import t as tdist

    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    k = len(diff)
    if k < 2:
        return np.nan
    var = diff.var(ddof=1)
    if var == 0:
        return 1.0 if np.allclose(diff, 0) else 0.0
    t_stat = diff.mean() / np.sqrt((1 / k + 1 / (n_splits - 1)) * var)
    return float(2 * tdist.sf(abs(t_stat), df=k - 1))


def holm(p) -> np.ndarray:
    """Holm 다중비교 보정."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def significance_tests(cv: pd.DataFrame, best_model: str, best_fs: str,
                       metrics=("PR_AUC", "Recall@20%")) -> pd.DataFrame:
    """최종모델과 나머지 조합을 같은 fold 끼리 짝지어 비교한다.

    - wins/losses: 25개 fold 중 최종모델이 이긴/진 횟수
    - wilcoxon_p : Wilcoxon 부호순위 검정 (반복 CV 의 fold 들은 서로 독립이 아니어서 다소 낙관적)
    - corrected_t_p: Nadeau-Bengio 보정 재표본 t-검정 (반복 k-fold 의 겹침을 보정, 보수적)
    """
    from scipy.stats import wilcoxon

    d = cv[cv.scope == "all"]
    ref = d[(d.model == best_model) & (d.features == best_fs)].set_index(["repeat", "fold"])
    rows = []
    for (m, fs), g in d.groupby(["model", "features"]):
        if (m, fs) == (best_model, best_fs):
            continue
        g = g.set_index(["repeat", "fold"]).loc[ref.index]
        for met in metrics:
            diff = (ref[met] - g[met]).to_numpy()
            p_t = corrected_t(diff)
            p_w = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            rows.append({"compared_model": m, "features": fs, "metric": met,
                         "best_mean": ref[met].mean(), "other_mean": g[met].mean(), "mean_diff": diff.mean(),
                         "wins": int((diff > 1e-12).sum()), "losses": int((diff < -1e-12).sum()),
                         "wilcoxon_p": p_w, "corrected_t_p": p_t})
    out = pd.DataFrame(rows)
    # 13개 조합 동시 비교 → 지표별 Holm 보정
    out["corrected_t_p_holm"] = out.groupby("metric")["corrected_t_p"].transform(lambda v: holm(v.to_numpy()))
    return out.sort_values(["metric", "mean_diff"])


# ------------------------------------------------------------------ 7. 불균형 처리 방식 비교
def _without_weights(model):
    """클래스 가중치를 끈 복제본 (오버샘플링과 가중치를 중복 적용하지 않기 위해)."""
    m = clone(model)
    off = {}
    for k in m.get_params():
        if k.endswith("class_weight"):
            off[k] = None
        elif k.endswith("scale_pos_weight") or k.endswith("is_unbalance"):
            off[k] = 1 if k.endswith("scale_pos_weight") else False
        elif k.endswith("auto_class_weights"):
            # CatBoost 는 None 파라미터가 get_params 에서 빠져 clone 이 실패함 → 키 없이 재생성
            return type(m)(**{p: v for p, v in m.get_params().items() if p != "auto_class_weights"})
    return m.set_params(**off)


def imbalance_comparison(dev: pd.DataFrame, model, use_cavity: bool) -> pd.DataFrame:
    """같은 그룹 CV 에서 불균형 처리 방식만 바꿔 비교. 오버샘플링은 학습 fold 에만 적용한다.

    SMOTE-NC 는 범주형(is_rg3, cavity)을 보존하고, ADASYN·BorderlineSMOTE 는 모든 변수를 연속형으로 보간한다.
    """
    from imblearn.over_sampling import ADASYN, SMOTENC, BorderlineSMOTE
    from imblearn.pipeline import make_pipeline as imb_pipeline

    X = feature_matrix(dev, use_cavity)
    y, g, prod = dev[TARGET].to_numpy(), dev["group"].to_numpy(), dev["product"].to_numpy()
    cat_idx = [i for i, c in enumerate(X.columns) if c in ("is_rg3", "cavity")]
    plain = _without_weights(model)
    variants = {
        "클래스 가중치 (현재)": model,
        "가중치 없음": plain,
        "SMOTE-NC": imb_pipeline(SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=SEED), plain),
        "BorderlineSMOTE": imb_pipeline(BorderlineSMOTE(k_neighbors=5, random_state=SEED), plain),
        "ADASYN": imb_pipeline(ADASYN(n_neighbors=5, random_state=SEED), plain),
    }
    rows = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for f, (tr, va) in enumerate(cv.split(X, y, g)):
            for name, m in variants.items():
                try:
                    p = clone(m).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
                except ValueError:          # ADASYN 이 합성할 이웃을 못 찾는 fold
                    continue
                rows.append({"method": name, "repeat": r, "fold": f, **_scores(y[va], p, prod[va])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 8. 스태킹 앙상블 · 딥러닝(MLP) 검토
def _mlp(hidden, alpha):
    """표 데이터용 신경망. sklearn MLP 는 class_weight 가 없어 학습 fold 안에서만 불량을 무작위 복제한다."""
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import make_pipeline as imb_pipeline
    from sklearn.neural_network import MLPClassifier

    return imb_pipeline(StandardScaler(), RandomOverSampler(random_state=SEED),
                        MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=500, early_stopping=True,
                                      validation_fraction=0.15, random_state=SEED))


def _stacking(tr_groups, ytr):
    """ET·RF·XGBoost·LR 의 예측을 로지스틱 회귀가 조합. 메타 학습용 내부 CV 도 사출 단위로 묶어 누수를 막는다."""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    m = get_models()
    inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(ytr)), ytr, tr_groups))
    return StackingClassifier(
        estimators=[("et", m["ExtraTrees"]), ("rf", m["RandomForest"]), ("xgb", m["XGBoost"]), ("lr", m["LR_baseline"])],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000),
        stack_method="predict_proba", cv=inner, n_jobs=1)


def dl_stacking_comparison(dev: pd.DataFrame, best_model, use_cavity: bool) -> pd.DataFrame:
    """최종모델 vs 스태킹 앙상블 vs MLP 2종을 같은 그룹 CV fold 에서 비교."""
    X = feature_matrix(dev, use_cavity)
    y, g, prod = dev[TARGET].to_numpy(), dev["group"].to_numpy(), dev["product"].to_numpy()
    fixed = {"최종모델": best_model, "MLP 32-16": _mlp((32, 16), 1e-3), "MLP 64-32-16": _mlp((64, 32, 16), 1e-2)}
    rows = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for f, (tr, va) in enumerate(cv.split(X, y, g)):
            models = {**fixed, "스태킹 (ET·RF·XGB·LR → LR)": _stacking(g[tr], y[tr])}
            for name, m in models.items():
                p = clone(m).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
                rows.append({"model": name, "repeat": r, "fold": f, "ROC_AUC": roc_auc_score(y[va], p),
                             **_scores(y[va], p, prod[va])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 9. 생산순번 블록 CV
def time_block_validation(dev: pd.DataFrame, n_blocks=(8, 16), repeats: int = TB_REPEATS, n_splits: int = 4) -> pd.DataFrame:
    """제품별 생산 순번을 연속 구간 n개로 나누고(추정 사출 그룹 단위), 구간 전체를 하나의 그룹으로 묶어 검증한다.

    평가 구간의 앞·뒤 구간이 모두 학습에 들어갈 수 있으므로 '과거→미래' 검증이 아니다. '같은 시기의 다른 사출을
    학습에서 빼면 성능이 얼마나 떨어지는가'(시간 근접 효과)를 보는 용도다. 과거→미래 대리 검증은 insights.forward_validation.
    학습 또는 평가 fold 에 불량이 없으면 지표를 계산하지 않되, 사유를 기록한다.
    """
    y, prod = dev[TARGET].to_numpy(), dev["product"].to_numpy()
    models = get_models()
    plan = [(n, True) for n in models] + [("ExtraTrees", False)]      # 캐비티 효과 확인용으로 ET 공정변수만 포함
    X_by = {True: feature_matrix(dev, True), False: feature_matrix(dev, False)}
    rows = []
    for nb in n_blocks:
        blk = production_blocks(dev, nb).astype(str)
        groups = (dev["product"] + "_b" + blk).to_numpy()
        assert (pd.Series(groups).groupby(dev["group"].to_numpy()).nunique() == 1).all(), "사출 그룹이 두 블록에 걸침"
        for r in range(repeats):
            cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED + r)
            for f, (tr, va) in enumerate(cv.split(X_by[True], y, groups)):
                base = {"n_blocks": nb, "repeat": r, "fold": f, "n_rows_eval": len(va), "n_fail_train": int(y[tr].sum()),
                        "n_fail_eval": int(y[va].sum()),
                        **{f"n_fail_eval_{p}": int(y[va][prod[va] == p].sum()) for p in PRODUCTS}}
                if y[tr].sum() == 0 or y[va].sum() == 0:
                    rows.append({**base, "model": "-", "features": "-", "note": "학습 또는 평가 fold 에 불량 0건 → 평가 불가"})
                    continue
                for name, cav in plan:
                    X = X_by[cav]
                    p = clone(models[name]).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
                    rows.append({**base, "model": name, "features": "process+cavity" if cav else "process_only",
                                 "note": "", "ROC_AUC": roc_auc_score(y[va], p), **_scores(y[va], p, prod[va])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 10. 통합 모델 vs 제품별 모델
def product_model_comparison(dev: pd.DataFrame, model, use_cavity: bool) -> pd.DataFrame:
    """같은 25개 fold 에서 '두 제품 통합 모델(is_rg3 포함)' 과 '제품별 별도 모델' 을 제품별 지표로 비교."""
    X = feature_matrix(dev, use_cavity)
    y, g, prod = dev[TARGET].to_numpy(), dev["group"].to_numpy(), dev["product"].to_numpy()
    rows = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for f, (tr, va) in enumerate(cv.split(X, y, g)):
            p_int = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            for pr in PRODUCTS:
                trm, vam = tr[prod[tr] == pr], va[prod[va] == pr]
                yv = y[vam]
                base = {"repeat": r, "fold": f, "product": pr, "n_eval": len(vam), "n_fail_eval": int(yv.sum()),
                        "n_fail_train": int(y[trm].sum())}
                p_sep = clone(model).fit(X.iloc[trm].drop(columns="is_rg3"), y[trm]).predict_proba(
                    X.iloc[vam].drop(columns="is_rg3"))[:, 1]
                for name, pp in [("통합 모델", p_int[prod[va] == pr]), ("제품별 모델", p_sep)]:
                    ok = yv.sum() > 0
                    rows.append({**base, "approach": name,
                                 "PR_AUC": average_precision_score(yv, pp) if ok else np.nan,
                                 "Recall@20%": recall_at_k(yv, pp, 0.20), "note": "" if ok else "평가 fold 불량 0건 → N/A"})
    return pd.DataFrame(rows)
