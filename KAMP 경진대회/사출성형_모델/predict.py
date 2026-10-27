"""저장된 최종모델로 재학습 없이 추론한다 (run_all.py 를 한 번 실행해 outputs/model/ 이 있어야 함).

    python predict.py --input data/moldset_labeled_cn7.csv --product cn7 --out outputs/predict_cn7.csv

입력 CSV 는 제공 데이터와 같은 형식(첫 열 = 생산 순번, 공정변수 24개, PassOrFail 은 있어도 되고 없어도 됨).
같은 생산 배치 안에서 위험점수 순위를 매기고, 상위 20% 를 검사 대상으로 표시한다.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import INDEX_COL, OUT_DIR, RAW_INDEX_COL, TARGET  # noqa: E402
from src.data import add_shot_structure, feature_matrix  # noqa: E402
from src.experiments import top_factors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--product", required=True, choices=["cn7", "rg3"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=float, default=0.20, help="검사 비율 (기본 0.20)")
    a = ap.parse_args()

    mdir = OUT_DIR / "model"
    meta = json.loads((mdir / "model_meta.json").read_text(encoding="utf-8"))
    final = joblib.load(mdir / "final_model.joblib")
    cal = joblib.load(mdir / "calibrated_model.joblib")

    df = pd.read_csv(a.input).rename(columns={RAW_INDEX_COL: INDEX_COL}).sort_values(INDEX_COL).reset_index(drop=True)
    has_label = TARGET in df.columns
    label = df.pop(TARGET) if has_label else None
    df = add_shot_structure(df)
    df["product"] = a.product
    X = feature_matrix(df, meta["use_cavity"], meta["process_columns"])[meta["feature_columns"]]

    out = df[[INDEX_COL, "shot_id", "cavity", "pair_type"]].copy()
    out["risk_score"] = final.predict_proba(X)[:, 1]
    out["risk_prob_cal"] = cal.predict_proba(X)[:, 1]
    cut = meta["grade_cut"]
    out["risk_grade"] = np.select([out.risk_prob_cal >= cut["위험"], out.risk_prob_cal >= cut["주의"]], ["위험", "주의"], "정상")
    out["inspect_rank"] = out["risk_score"].rank(ascending=False, method="first").astype(int)
    out["inspect"] = (out["inspect_rank"] <= np.ceil(len(out) * a.top)).astype(int)
    out = out.join(top_factors(final, X))
    if has_label:
        out[TARGET] = label.to_numpy()
    path = Path(a.out) if a.out else OUT_DIR / f"predict_{a.product}.csv"
    out.sort_values("inspect_rank").to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(out)}행 추론 → {path}  (검사 대상 {out.inspect.sum()}개, 모델 {meta['model']}/{meta['features']})")
    if has_label and label.sum():
        print(f"  참고: 라벨이 있어 계산한 상위 {a.top:.0%} 포착률 = {out.loc[out.inspect == 1, TARGET].sum() / label.sum():.3f}"
              " (학습에 쓴 행이 포함되면 성능 지표로 쓰지 말 것)")


if __name__ == "__main__":
    main()
