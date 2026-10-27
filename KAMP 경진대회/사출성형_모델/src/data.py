"""데이터 적재, 추정 사출 그룹 구조 복원, 데이터 진단.

주의: 원본 데이터에는 사출 번호·캐비티 번호 컬럼이 없다. shot_id 와 cavity 는
'연속으로 공정값이 같은 행 = 한 번의 사출' 이라는 가정으로 만든 **추정값**이다.
"""
import numpy as np
import pandas as pd

from .config import DATA_DIR, INDEX_COL, PRODUCTS, RAW_INDEX_COL, TARGET


def load_labeled(product: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"moldset_labeled_{product}.csv").rename(columns={RAW_INDEX_COL: INDEX_COL})
    return df.sort_values(INDEX_COL).reset_index(drop=True)


def load_unlabeled(product: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"moldset_unlabeled_{product}.csv").rename(columns={RAW_INDEX_COL: INDEX_COL})
    return df.sort_values(INDEX_COL).reset_index(drop=True)


def process_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (INDEX_COL, TARGET)]


def add_shot_structure(df: pd.DataFrame, twin_window: int = 5) -> pd.DataFrame:
    """공정값이 동일한 행을 하나의 '추정 사출 그룹'으로 묶는다 (설비가 준 사출 번호가 아님).

    1) 연속으로 공정값이 같은 행 → 같은 그룹 (seq_id: 기존 방식)
    2) 1행 그룹끼리 twin_window 행 이내에 공정값이 완전히 같은 짝이 있으면 같은 그룹으로 병합
       (짝이 A·B·A·B 처럼 교차 기록된 경우. 병합하지 않으면 같은 공정값이 학습/검증 양쪽에 들어가 누수가 된다)

    - shot_id   : 추정 사출 그룹 번호
    - cavity    : 그룹 안의 행 순서(0,1). 2캐비티 금형의 위치일 가능성이 있으나 확인되지 않은 **추정 캐비티**.
                  끝까지 짝이 없는 1행 그룹은 항상 0 이므로, 그 cavity=0 은 위치 정보가 아니다.
    - shot_size : 그룹 행 수,  pair_type : 연속 2행 / 교차 2행(병합) / 단일
    """
    cols = process_cols(df)
    df = df.copy()
    new_seq = df[cols].diff().abs().sum(axis=1).ne(0)
    new_seq.iloc[0] = True
    df["seq_id"] = new_seq.cumsum() - 1
    seq_size = df.groupby("seq_id")["seq_id"].transform("size").to_numpy()
    key = df[cols].round(10).astype(str).agg("|".join, axis=1).to_numpy()
    merged = df["seq_id"].to_numpy().copy()
    used = set()
    singles = np.flatnonzero(seq_size == 1)
    for a in singles:
        if a in used:
            continue
        for b in singles[(singles > a) & (singles <= a + twin_window)]:
            if b not in used and key[a] == key[b]:
                merged[b] = merged[a]
                used.update((a, b))
                break
    df["shot_id"] = pd.factorize(merged)[0]
    df["cavity"] = df.groupby("shot_id").cumcount()
    df["shot_size"] = df.groupby("shot_id")["shot_id"].transform("size")
    is_twin = np.isin(np.arange(len(df)), list(used))
    df["pair_type"] = np.select([is_twin, df["shot_size"] == 2], ["교차 2행(병합)", "연속 2행"], "단일")
    return df


def load_all() -> pd.DataFrame:
    """두 제품을 합쳐 반환. 제품 간 shot_id 충돌을 막기 위해 제품 접두사를 붙인다."""
    parts = []
    for p in PRODUCTS:
        d = add_shot_structure(load_labeled(p))
        d["product"] = p
        d["group"] = p + "_" + d["shot_id"].astype(str)
        d["seq_group"] = p + "_" + (d["seq_id"] - 1).astype(str)   # 병합 전 연속 행 기준 그룹 (홀드아웃 분할용)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def diagnose(df: pd.DataFrame) -> pd.DataFrame:
    """제품별 데이터 진단표."""
    rows = []
    for p, d in df.groupby("product"):
        cols = process_cols(load_labeled(p))
        pat = d.groupby(cols)[TARGET].agg(["size", "nunique", "sum"])
        shot = d.groupby("shot_id")[TARGET].sum()
        rows.append({
            "product": p,
            "rows": len(d),
            "fail_rows": int(d[TARGET].sum()),
            "fail_rate": d[TARGET].mean(),
            "missing": int(d[cols].isna().sum().sum()),
            "constant_cols": ", ".join(c for c in cols if d[c].nunique() == 1),
            "shots": d["shot_id"].nunique(),
            "shots_2cavity": int((d.groupby("shot_id").size() == 2).sum()),
            "fail_shots": int((shot > 0).sum()),
            "fail_at_cavity0": int(d.loc[d[TARGET] == 1, "cavity"].eq(0).sum()),
            "conflicting_patterns": int((pat["nunique"] > 1).sum()),
            "fails_in_conflicting": int(pat.loc[pat["nunique"] > 1, "sum"].sum()),
            "first_fail_idx": int(d.loc[d[TARGET] == 1, INDEX_COL].min()),
            "last_fail_idx": int(d.loc[d[TARGET] == 1, INDEX_COL].max()),
        })
    return pd.DataFrame(rows)


def scaling_check() -> pd.DataFrame:
    """라벨/비라벨 각 파일의 평균·표준편차 (파일별 표준화 여부 확인용)."""
    rows = []
    for p in PRODUCTS:
        for kind, d in [("labeled", load_labeled(p)), ("unlabeled", load_unlabeled(p))]:
            cols = process_cols(d)
            rows.append({"product": p, "set": kind, "rows": len(d),
                         "max_abs_mean": d[cols].mean().abs().max(),
                         "min_std": d[cols].std().min(), "max_std": d[cols].std().max()})
    return pd.DataFrame(rows)


def feature_columns(train_df: pd.DataFrame) -> list[str]:
    """학습 데이터에서 상수가 아닌 공정변수. 평가 데이터에는 이 목록을 그대로 적용해 입력 열이 달라지지 않게 한다."""
    return [c for c in process_cols(load_labeled(PRODUCTS[0])) if train_df[c].nunique() > 1]


def feature_matrix(df: pd.DataFrame, use_cavity: bool, cols: list[str] | None = None) -> pd.DataFrame:
    """모델 입력. cols 를 주지 않으면 df 자체에서 상수 열을 뺀다 (개발셋 내부 CV 용).
    테스트·신규 데이터에는 반드시 개발셋에서 정한 cols 를 넘긴다."""
    cols = feature_columns(df) if cols is None else cols
    X = df[cols].copy()
    X["is_rg3"] = (df["product"] == "rg3").astype(int)
    if use_cavity:
        X["cavity"] = df["cavity"].astype(int)
    return X


def structure_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """추정 사출 그룹 구조의 예외 점검.

    반환 ① 제품별 요약: 연속 규칙 그룹 크기, 교차 짝 병합 결과, 생산 순번 누락, 최종 그룹 구성
         ② 연속 규칙상 1행 그룹 상세: 같은 공정값 짝까지의 거리, 병합 여부, 인접 순번 누락, 불량 여부
    """
    summ, singles = [], []
    for p, d in df.groupby("product"):
        cols = process_cols(load_labeled(p))
        d = d.sort_values(INDEX_COL).reset_index(drop=True)
        idx = d[INDEX_COL].to_numpy()
        gaps = np.diff(idx)
        seq_size = d.groupby("seq_id").size()
        size = d.groupby("shot_id").size()
        key = d[cols].round(10).astype(str).agg("|".join, axis=1)
        pos_by_key = d.groupby(key).indices
        s1 = d[d["seq_id"].map(seq_size) == 1]
        for i, r in s1.iterrows():
            others = [j for j in pos_by_key[key[i]] if j != i]
            prev_gap = idx[i] - idx[i - 1] if i > 0 else np.nan
            next_gap = idx[i + 1] - idx[i] if i < len(idx) - 1 else np.nan
            singles.append({"product": p, INDEX_COL: r[INDEX_COL], TARGET: r[TARGET],
                            "같은 공정값 짝까지 거리(행)": min(abs(j - i) for j in others) if others else np.nan,
                            "병합 결과": r["pair_type"],
                            "인접 순번 누락": bool((prev_gap > 1) or (next_gap > 1))})
        sp = pd.DataFrame([x for x in singles if x["product"] == p])
        summ.append({
            "product": p, "행 수": len(d), "순번 범위": f"{idx.min()}~{idx.max()}",
            "순번 누락 위치 수": int((gaps > 1).sum()), "누락된 순번 수": int((gaps[gaps > 1] - 1).sum()),
            "[연속 규칙] 그룹 수": len(seq_size), "[연속 규칙] 2행 그룹": int((seq_size == 2).sum()),
            "[연속 규칙] 1행 그룹": int((seq_size == 1).sum()), "[연속 규칙] 3행 이상": int((seq_size >= 3).sum()),
            "1행 그룹 중 5행 이내 같은 공정값 짝 있음": int((sp["같은 공정값 짝까지 거리(행)"] <= 5).sum()) if len(sp) else 0,
            "1행 그룹 중 인접 순번 누락": int(sp["인접 순번 누락"].sum()) if len(sp) else 0,
            "1행 그룹 불량": int(s1[TARGET].sum()),
            "[최종] 추정 사출 그룹 수": len(size), "[최종] 연속 2행": int((d.groupby("shot_id")["pair_type"].first() == "연속 2행").sum()),
            "[최종] 교차 2행(병합)": int((d.groupby("shot_id")["pair_type"].first() == "교차 2행(병합)").sum()),
            "[최종] 단일 행": int((size == 1).sum()),
            "2행 그룹 중 두 행 모두 불량": int((d[d.shot_size == 2].groupby("shot_id")[TARGET].sum() == 2).sum()),
        })
    return pd.DataFrame(summ), pd.DataFrame(singles)


def data_quality_summary(df: pd.DataFrame, z: float = 4.0) -> pd.DataFrame:
    """결측 · 중복 · 상수 · 극단값 · 라벨 충돌의 규모와 처리 방침을 한 표로."""
    rows = []
    for p, d in df.groupby("product"):
        cols = process_cols(load_labeled(p))
        const = [c for c in cols if d[c].nunique() == 1]
        var = [c for c in cols if c not in const]
        ext = (d[var].abs() > z).any(axis=1)
        pat = d.groupby(cols)[TARGET].agg(["size", "nunique", "sum"])
        conflict_rows = int(pat.loc[pat["nunique"] > 1, "size"].sum())
        dup_in = d.duplicated(cols, keep=False)
        rows += [
            {"product": p, "항목": "결측", "규모": int(d[cols].isna().sum().sum()), "불량 관련": "-", "처리": "없음 (결측 0)"},
            {"product": p, "항목": "상수 열", "규모": len(const), "불량 관련": ", ".join(const) or "-",
             "처리": "정보 없음 → 입력에서 제외 (학습 데이터 기준으로 결정, 평가 데이터에 같은 열 목록 적용)"},
            {"product": p, "항목": "공정값 중복 행(순번 제외)", "규모": int(dup_in.sum()),
             "불량 관련": f"불량 {int(d.loc[dup_in, TARGET].sum())}건",
             "처리": "삭제하지 않음: 대부분 같은 추정 사출 그룹의 짝 행이며, 그룹 CV 로 누수를 막음"},
            {"product": p, "항목": f"극단값 행 (|z|>{z:g})", "규모": int(ext.sum()),
             "불량 관련": f"불량 {int(d.loc[ext, TARGET].sum())}건 (불량률 {d.loc[ext, TARGET].mean() * 100:.1f}%)",
             "처리": ("삭제하지 않음: 불량률이 전체보다 높아 불량 신호일 수 있음" if d.loc[ext, TARGET].mean() > d[TARGET].mean()
                    else "삭제하지 않음: 불량과 무관해 보이나 측정 오류 근거가 없고, 트리 모델은 극단값에 강건")},
            {"product": p, "항목": "라벨 충돌 행 (같은 공정값에 양품·불량 공존)", "규모": conflict_rows,
             "불량 관련": f"불량 {int(pat.loc[pat['nunique'] > 1, 'sum'].sum())}/{int(d[TARGET].sum())}건",
             "처리": "삭제·수정하지 않음: 공정변수만으로 구분 불가한 불량 → 예측 상한으로 해석"},
        ]
    return pd.DataFrame(rows)


def production_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    """제품별 생산 순번을 연속 n개 구간으로 나눈다. 행이 아니라 **추정 사출 그룹 단위**로 나누므로
    한 그룹이 두 구간에 걸치지 않는다. 반환: 행별 구간 번호(0 = 가장 이른 구간)."""
    out = pd.Series(0, index=df.index, dtype="int64")
    for _, d in df.groupby("product"):
        first = d.groupby("group")[INDEX_COL].min().sort_values()
        blk = pd.Series(pd.qcut(np.arange(len(first)), n_blocks, labels=False), index=first.index)
        out.loc[d.index] = d["group"].map(blk).to_numpy()
    return out
