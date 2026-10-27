"""카드 + 기상 + 달력 결합 → data/processed/panel_daily.csv, panel_slot.csv

panel_daily: 일자 x 지역 x 업종 (분석 기본 단위, 이연소비용 lag/lead 포함)
panel_slot : 일자 x 지역 x 업종 x 시간대 (시간대별 기상과 매칭)
선행: 01_fetch_weather.py, 02_build_card_panel.py
"""
import numpy as np
import pandas as pd

import config as C


def calendar() -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(C.ANALYSIS_START, C.ANALYSIS_END, freq="D")})
    d["dow"] = d.date.dt.dayofweek  # 0=월
    d["month"] = d.date.dt.month
    d["is_weekend"] = (d.dow >= 5).astype(int)
    d["is_holiday"] = d.date.isin(C.HOLIDAYS).astype(int)
    d["is_offday"] = ((d.is_weekend == 1) | (d.is_holiday == 1)).astype(int)
    d["pre_offday"] = d.is_offday.shift(-1, fill_value=0).astype(int) * (1 - d.is_offday)
    d["chuseok_block"] = d.date.between(*C.CHUSEOK_BLOCK).astype(int)
    d["special_day"] = d.date.isin(C.SPECIAL_DAYS).astype(int)
    d["payday"] = d.date.dt.day.isin([10, 25]).astype(int)
    d["trend"] = (d.date - d.date.min()).dt.days
    return d


def read(name: str, dates=("date",)) -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED_DIR / name, parse_dates=list(dates), encoding="utf-8-sig")


def main():
    cal = calendar()
    wd = read("weather_daily.csv").rename(columns={"source": "weather_source"})
    ws = read("weather_slot.csv").rename(columns={"slot": "TIME_GB", "source": "weather_source"})
    wd = wd[wd.date.between(C.ANALYSIS_START, C.ANALYSIS_END)]
    ws = ws[ws.date.between(C.ANALYSIS_START, C.ANALYSIS_END)]

    # ------------------------------------------------ 일별 패널
    card = read("card_daily.csv")
    panel = card.merge(cal, on="date", how="left").merge(wd, on=["region", "date"], how="left")
    panel["log_sales"] = np.log(panel.sales)
    panel["visitor_ratio"] = panel.sales_visitor / panel.sales
    panel = panel.sort_values(["region", "MCT_RY_CD", "date"])
    panel.to_csv(C.PROCESSED_DIR / "panel_daily.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ 시간대 패널
    slot = read("card_slot.csv")
    day_cols = ["region", "date", "tmax", "tmin", "rain", "snow_new", "heat33", "heavy_rain", "snow",
                "rain_lag1", "heavy_rain_lag1"]
    ps = (slot.merge(cal, on="date", how="left")
              .merge(ws.add_prefix("slot_").rename(columns={"slot_region": "region", "slot_date": "date",
                                                            "slot_TIME_GB": "TIME_GB"}),
                     on=["region", "date", "TIME_GB"], how="left")
              .merge(wd[day_cols].rename(columns=lambda c: c if c in ("region", "date") else f"day_{c}"),
                     on=["region", "date"], how="left"))
    ps["log_sales"] = np.log(ps.sales)
    ps = ps.sort_values(["region", "MCT_RY_CD", "date", "TIME_GB"])
    ps.to_csv(C.PROCESSED_DIR / "panel_slot.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ 점검
    print(f"panel_daily {panel.shape}, panel_slot {ps.shape}")
    print("기상 출처:", panel.weather_source.dropna().unique())
    for name, p, key in [("daily", panel, "tmax"), ("slot", ps, "slot_rain")]:
        miss = p[key].isna().mean()
        print(f"[{name}] 기상 미결합 비율 {miss:.4f}")
        assert miss == 0, f"{name} 패널에 기상 결측 존재"
    dup = panel.duplicated(["date", "region", "MCT_RY_CD"]).sum() + ps.duplicated(
        ["date", "region", "MCT_RY_CD", "TIME_GB"]).sum()
    assert dup == 0, "키 중복"

    core = panel[(panel.scope == "core") & (panel.reliable == 1)]
    print(f"\n분석 대상(core & reliable): {core.MCT_RY_CD.nunique()}개 업종, {len(core)}행")
    tot = core.groupby(["region", "month"]).sales.sum().div(1e8).round(0).unstack()
    print("월별 매출(억원):\n", tot)


if __name__ == "__main__":
    main()
