"""평가: 사출 그룹 CV, 임계값 선정, 검사 우선순위 지표."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from .config import N_REPEATS, N_SPLITS, SEED, TOP_K


def best_threshold(y, p) -> float:
    """F1 최대 임계값. 반드시 학습 fold 내부 예측으로만 고른다(평가 fold 누수 방지)."""
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 100)))
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def assert_disjoint(groups, tr, va, where=""):
    """같은 추정 사출 그룹이 학습·평가 양쪽에 들어가지 않았는지 확인."""
    groups = np.asarray(groups)
    both = set(groups[tr]) & set(groups[va])
    assert not both, f"{where}: 그룹 {len(both)}개가 학습/평가에 동시 존재"


def inner_oof(model, X, y, groups, seed) -> np.ndarray:
    """학습 fold 안에서 3-fold 그룹 CV 예측 → 임계값 선정용."""
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y, groups):
        assert_disjoint(groups, tr, va, "inner_oof")
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def recall_at_k(y, p, k) -> float:
    """상위 ceil(k·n) 개 검사 시 포착률. 동점은 입력 순서(생산 순번) 우선. 불량이 없으면 정의 불가 → NaN."""
    y = np.asarray(y)
    if y.sum() == 0:
        return np.nan
    n = max(1, int(np.ceil(len(p) * k)))
    top = np.argsort(-np.asarray(p), kind="stable")[:n]
    return y[top].sum() / y.sum()


def metrics(y, p, thr) -> dict:
    """불량이 없는 범위에서는 PR-AUC·Recall 류가 정의되지 않으므로 NaN(N/A) 으로 둔다."""
    y = np.asarray(y)
    if y.sum() == 0:
        keys = ["PR_AUC", "ROC_AUC", "F1", "Precision", "Recall", *[f"Recall@{int(k * 100)}%" for k in TOP_K],
                "Precision@20%", "F1@20%"]
        return {k: np.nan for k in keys}
    out = {
        "PR_AUC": average_precision_score(y, p),
        "ROC_AUC": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        "F1": f1_score(y, p >= thr, zero_division=0),
        "Precision": precision_score(y, p >= thr, zero_division=0),
        "Recall": recall_score(y, p >= thr, zero_division=0),
    }
    for k in TOP_K:
        out[f"Recall@{int(k * 100)}%"] = recall_at_k(y, p, k)
    # 운영점: 점수 상위 20% 를 검사할 때의 정밀도·F1
    n = max(1, int(np.ceil(len(p) * 0.20)))
    prec20 = y[np.argsort(-np.asarray(p), kind="stable")[:n]].sum() / n
    rec20 = out["Recall@20%"]
    out["Precision@20%"] = prec20
    out["F1@20%"] = 2 * prec20 * rec20 / (prec20 + rec20) if prec20 + rec20 > 0 else 0.0
    return out


def grouped_cv(model, X, y, groups, products, keep_preds=False):
    """반복 StratifiedGroupKFold.

    반환: fold별 지표 리스트, 반복 평균 OOF 점수, (keep_preds=True 면) fold별 예측 기록
    - 지표는 각 외부 fold 에서 '그 fold 의 학습 데이터 내부 3-fold' 로 정한 임계값으로 계산한다.
    - 반복 평균 OOF 점수는 순위·오류 진단용이며, 반복 평균은 점수 분포를 좁히므로 fold 임계값을 그대로 적용하면 안 된다.
    - 제품 범위에 불량이 없으면 지표를 NaN 으로 남기고 note 에 사유를 적는다 (조용히 제외하지 않음).
    """
    y = np.asarray(y)
    records, preds, oof_sum, oof_cnt = [], [], np.zeros(len(y)), np.zeros(len(y))
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + r)
        for f, (tr, va) in enumerate(cv.split(X, y, groups)):
            assert_disjoint(groups, tr, va, f"grouped_cv r{r} f{f}")
            m = clone(model).fit(X.iloc[tr], y[tr])
            thr = best_threshold(y[tr], inner_oof(model, X.iloc[tr], y[tr], groups[tr], SEED + r))
            p = m.predict_proba(X.iloc[va])[:, 1]
            oof_sum[va] += p
            oof_cnt[va] += 1
            if keep_preds:
                preds.append(pd.DataFrame({"row": va, "repeat": r, "fold": f, "score": p, "fold_threshold": thr,
                                           "pred_fold_thr": (p >= thr).astype(int)}))
            for scope in ["all", *np.unique(products)]:
                msk = np.ones(len(va), bool) if scope == "all" else products[va] == scope
                yy, pp = y[va][msk], p[msk]
                records.append({"repeat": r, "fold": f, "threshold": thr, "scope": scope,
                                "n_rows": int(msk.sum()), "n_shots": int(len(np.unique(groups[va][msk]))),
                                "n_fail": int(yy.sum()), "note": "" if yy.sum() else "불량 0건 → 지표 정의 불가(N/A)",
                                **metrics(yy, pp, thr)})
    oof = oof_sum / np.maximum(oof_cnt, 1)
    if keep_preds:
        return records, oof, pd.concat(preds, ignore_index=True)
    return records, oof
