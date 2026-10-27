"""머신러닝 비교 모형: 그래디언트 부스팅(LightGBM과 같은 계열, scikit-learn HistGradientBoosting)
→ outputs/cv_ml_predictions.csv, cv_ml_summary.csv
선행: 04 실행 (cv_predictions.csv의 A·B·C 예측과 같은 행·같은 폴드로 비교)

04 단계와 똑같이 호우 이벤트 하나(당일 + 이후 2일)를 통째로 빼고 학습한 뒤 그 기간을 예측한다.
회귀 모형(A·B·C)은 업종마다 따로 학습하지만, 부스팅은 한 지역의 모든 업종을 한 번에 학습해
업종 간 정보를 공유하고 날씨·달력의 비선형·상호작용을 스스로 찾는다.
  목표값: log(매출) - 업종 평균 log(매출)   (업종 규모 차이를 빼고 '평소 대비 변화'만 학습)
  입력: 강수량(mm)·전날 강수량·호우 표시와 시차·보통비·약한 비·강설·적설·폭염·최고기온·요일·월·달력 표시·업종군·업종
  하이퍼파라미터는 미리 고정(튜닝 없음): 검증 이벤트 정보가 학습에 새지 않게 하기 위해
"""
import importlib
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config as CFG

M = importlib.import_module("04_model_sensitivity")
warnings.filterwarnings("ignore")

NUM = ["rain", "rain_lag1", "tmax", *M.WEATHER, *M.CAL_FLAGS]
CAT = ["dow", "month", "grp_code", "ind_code"]
PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
              random_state=0)


def features(pr: pd.DataFrame) -> pd.DataFrame:
    d = pr[(pr.observed == 1) & (pr.sales > 0)].copy()
    d["grp_code"] = d.ind_group.astype("category").cat.codes
    d["ind_code"] = d.MCT_RY_CD.astype("category").cat.codes
    d["y"] = np.log(d.sales)
    return d


def cross_validate_ml(p: pd.DataFrame) -> pd.DataFrame:
    preds = []
    for region, pr in p.groupby("region"):
        d = features(pr)
        day = pr.drop_duplicates("date")[["date", "dow", "month", *M.WEATHER, *M.CAL_FLAGS]]
        events = M.rain_events(day)
        dates = set(day.date)
        cat_mask = [c in CAT for c in NUM + CAT]
        for ei, ev in enumerate(events):
            last = ev[-1]
            test_days = [t for t in ev + [last + pd.Timedelta(days=k) for k in range(1, M.RECOVERY_DAYS + 1)]
                         if t in dates]
            tr = d[~d.date.isin(test_days)].copy()
            te = d[d.date.isin(test_days)].copy()
            mu = tr.groupby("MCT_RY_CD").y.mean()
            tr["dev"] = tr.y - tr.MCT_RY_CD.map(mu)
            m = HistGradientBoostingRegressor(categorical_features=cat_mask, **PARAMS)
            m.fit(tr[NUM + CAT], tr.dev)
            # 로그 예측을 원 단위로 되돌릴 때의 편향 보정(업종별 잔차 평균, 회귀 모형과 같은 방식)
            smear = (tr.y - tr.MCT_RY_CD.map(mu) - m.predict(tr[NUM + CAT])).pipe(np.exp).groupby(tr.MCT_RY_CD).mean()
            te = te[te.MCT_RY_CD.isin(mu.index)]
            pred = np.exp(te.MCT_RY_CD.map(mu) + m.predict(te[NUM + CAT])) * te.MCT_RY_CD.map(smear)
            for t, ry, yhat in zip(te.date, te.MCT_RY_CD, pred):
                preds.append({"region": region, "event": ei, "date": t, "MCT_RY_CD": ry, "pred_D_boosting": yhat})
    return pd.DataFrame(preds)


def summarize(cv: pd.DataFrame) -> pd.DataFrame:
    models = ["pred_A_calendar", "pred_B_weather", "pred_C_weather_eb", "pred_D_boosting"]
    rows = []
    for (region, rain_day), g in cv.groupby(["region", "is_rain_day"]):
        r = {"region": region, "window": "호우 당일" if rain_day else "호우 후 1~2일", "n": len(g)}
        for m in models:
            r[f"WAPE_{m[5:]}"] = (g[m] - g.actual).abs().sum() / g.actual.sum()
        rows.append(r)
    # 이벤트별(지역 업종 합계) 승자
    ev = cv.groupby(["region", "event"])[["actual", *models]].sum()
    wins = ev[models].sub(ev.actual, axis=0).abs().idxmin(axis=1)
    for region, w in wins.groupby(level=0):
        print(f"  {region} 이벤트별 최소오차 모형:", w.value_counts().to_dict())
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 200)
    p = M.load_panel()
    base = pd.read_csv(CFG.OUTPUT_DIR / "cv_predictions.csv", parse_dates=["date"], encoding="utf-8-sig")
    ml = cross_validate_ml(p)
    cv = base.merge(ml, on=["region", "event", "date", "MCT_RY_CD"], how="inner")
    print(f"비교 행: {len(cv)} / 기존 {len(base)}")
    cv.to_csv(CFG.OUTPUT_DIR / "cv_ml_predictions.csv", index=False, encoding="utf-8-sig")
    s = summarize(cv)
    s.to_csv(CFG.OUTPUT_DIR / "cv_ml_summary.csv", index=False, encoding="utf-8-sig")
    show = s.copy()
    for c in [c for c in s if c.startswith("WAPE")]:
        show[c] = show[c] * 100
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
