"""호우 기준 민감도: 30mm(기본) / 40mm / 50mm / 기상청 호우주의보급(3시간 60mm 또는 12시간 110mm)
→ outputs/threshold_sensitivity.csv
선행: 04, 05, 10 실행 (패널·지하철 일별 자료 재사용)

호우 표시만 바꾸고 나머지(달력 통제, 약한 비, 보통비 구간)는 같은 방식으로 다시 만든다.
보통비 = 10mm 이상이면서 호우가 아닌 날 (호우 기준이 올라가면 보통비 구간이 넓어진다).
"""
import importlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config as CFG

M = importlib.import_module("04_model_sensitivity")
R = importlib.import_module("05_robustness_checks")
warnings.filterwarnings("ignore")

RULES = ["30mm", "40mm", "50mm", "호우주의보급"]


def day_flags() -> pd.DataFrame:
    """지역 x 일 호우 표시. 일강수는 ASOS 일자료, 주의보급은 시간자료의 3·12시간 누적 최대."""
    w = pd.read_csv(CFG.PROCESSED_DIR / "weather_daily.csv", parse_dates=["date"], encoding="utf-8-sig")
    h = pd.read_csv(CFG.PROCESSED_DIR / "weather_hourly.csv", parse_dates=["time"], encoding="utf-8-sig")
    h = h.sort_values(["region", "time"])
    h["rn"] = h.rn.fillna(0)
    roll = []
    for region, g in h.groupby("region"):
        s = g.set_index("time").rn
        roll.append(pd.DataFrame({"region": region, "max3h": s.rolling(3).sum().resample("D").max(),
                                  "max12h": s.rolling(12).sum().resample("D").max()}).rename_axis("date").reset_index())
    w = w[["region", "date", "rain"]].merge(pd.concat(roll), on=["region", "date"], how="left")
    for rule in RULES:
        if rule == "호우주의보급":
            w[rule] = ((w.max3h >= 60) | (w.max12h >= 110)).astype(int)
        else:
            w[rule] = (w.rain >= float(rule[:-2])).astype(int)
    return w


def apply_rule(p: pd.DataFrame, flags: pd.DataFrame, rule: str) -> pd.DataFrame:
    f = flags[["region", "date", "rain", rule]].rename(columns={rule: "hr"}).sort_values(["region", "date"])
    g = f.groupby("region").hr
    f["hr1"], f["hr2"] = g.shift(1).fillna(0), g.shift(2).fillna(0)
    p = p.drop(columns=["heavy_rain", "heavy_rain_lag1", "heavy_rain_lag2", "rain_mid"]).merge(
        f[["region", "date", "hr", "hr1", "hr2"]], on=["region", "date"])
    p = p.rename(columns={"hr": "heavy_rain", "hr1": "heavy_rain_lag1", "hr2": "heavy_rain_lag2"})
    p["rain_mid"] = ((p.rain >= 10) & (p.heavy_rain == 0)).astype(int)
    return p


def arrivals(flags: pd.DataFrame, rule: str) -> dict:
    """추석 제외 호우 당일 지하철 도착 인원 변화 (10단계와 같은 식)"""
    a = pd.read_csv(CFG.OUTPUT_DIR / "subway_daily.csv", parse_dates=["date"], encoding="utf-8-sig")
    a = a.drop(columns=["heavy_rain"]).merge(flags[["region", "date", rule]].rename(columns={rule: "heavy_rain"}),
                                             on=["region", "date"])
    cs, ce = pd.to_datetime(CFG.CHUSEOK_BLOCK)
    a = a[~a.date.between(cs, ce)]
    out = {}
    for region, g in a.groupby("region"):
        if g.heavy_rain.sum() == 0:
            out[region] = (np.nan, np.nan)
            continue
        m = smf.ols("np.log(off) ~ heavy_rain + C(dow) + C(month) + is_holiday", g).fit(
            cov_type="HAC", cov_kwds={"maxlags": 7})
        out[region] = (np.exp(m.params["heavy_rain"]) - 1, m.pvalues["heavy_rain"])
    return out


def main():
    pd.set_option("display.width", 220)
    base = M.load_panel()
    base = base[base.scope == "core"]
    flags = day_flags()
    in_range = flags[flags.date.between(CFG.ANALYSIS_START, CFG.ANALYSIS_END)]
    rows = []
    for rule in RULES:
        p = apply_rule(base, flags, rule)
        rt = pd.DataFrame(R.region_test(p, rule))
        rt = rt[rt.balanced]
        arr = arrivals(flags, rule)
        for _, r in rt.iterrows():
            region = r.region
            row = {"rule": rule, "region": region, "gross": r.get("gross"), "net": r.net, "net_lo": r.net_lo,
                   "net_hi": r.net_hi, "p_value": r.get("p_value")}
            if region in CFG.REGIONS:
                row["heavy_days"] = int(in_range[in_range.region == region][rule].sum())
                row["arrival_effect"], row["arrival_p"] = arr[region]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(CFG.OUTPUT_DIR / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    show = out.copy()
    for c in ["gross", "net", "net_lo", "net_hi", "arrival_effect"]:
        show[c] = show[c] * 100
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
