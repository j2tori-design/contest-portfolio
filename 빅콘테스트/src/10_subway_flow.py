"""지하철 역별 일별 하차(도착) 인원으로 '발길이 끊기는가'를 측정한다
→ outputs/subway_daily.csv, subway_effects.csv, subway_mediation.csv, figures F12·F13

외부 데이터: 서울 열린데이터광장 '서울시 지하철호선별 역별 승하차 인원 정보'(OA-12914, 교통카드 기준, 공공누리 1유형)
  원본: data/external/seoul_subway/CARD_SUBWAY_MONTH_2025MM.csv (2025년 7~12월)
  춘천: 경춘선 춘천·남춘천·김유정·강촌·백양리·굴봉산 (춘천시 소재 6개 역)
  강남: 강남구 소재 지하철역 25개 (2호선·3호선·7호선·9호선·수인분당선)
한계: 교통카드 이용자만 집계(ITX-청춘·자가용 방문객 제외), 하차 인원에는 귀가하는 주민도 포함

1) 호우가 도착 인원을 줄이는가 (요일·월·공휴일·추석 통제, HAC)
2) 매출 감소 중 '도착 인원 감소'로 설명되는 몫 (도착 인원을 넣기 전후 호우 계수 비교, 주 단위 블록 부트스트랩)
"""
import csv
import glob
import importlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

import config as CFG  # noqa: E402

F = importlib.import_module("07_figures")
warnings.filterwarnings("ignore")  # 평일/휴일 하위표본에서 공휴일 더미가 상수가 되는 경고

SUB_DIR = CFG.EXTERNAL_DIR / "seoul_subway"
CHUNCHEON = ["춘천", "남춘천", "김유정", "강촌", "백양리", "굴봉산"]
GANGNAM = ["강남", "역삼", "선릉", "삼성", "신사", "압구정", "학여울", "대치", "도곡", "매봉", "일원", "대청", "수서",
           "청담", "강남구청", "학동", "논현", "언주", "선정릉", "삼성중앙", "봉은사", "압구정로데오", "한티", "구룡",
           "개포동", "대모산입구"]
LAGS = ["heavy_rain", "heavy_rain_lag1", "heavy_rain_lag2"]
CTRL = "C(dow) + C(month) + is_holiday + chuseok_block"
N_BOOT = 500


def load_arrivals() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(SUB_DIR / "CARD_SUBWAY_MONTH_2025*.csv"))):
        with open(f, encoding="utf-8-sig") as fh:
            r = csv.reader(fh)
            next(r)
            rows += [x[:5] for x in r]  # 행 끝의 빈 칸 제거
    s = pd.DataFrame(rows, columns=["ymd", "line", "stn", "on", "off"])
    s[["on", "off"]] = s[["on", "off"]].astype(int)
    s["date"] = pd.to_datetime(s.ymd, format="%Y%m%d")
    ch = (s.line == "경춘선") & s.stn.isin(CHUNCHEON)
    gn = (s.line != "경춘선") & s.stn.isin(GANGNAM)
    s = s[ch | gn].copy()
    s["region"] = np.where(s.stn.isin(CHUNCHEON), "강원 춘천시", "서울 강남구")
    found = {r: sorted(g.stn.unique()) for r, g in s.groupby("region")}
    print("포함 역:", {r: len(v) for r, v in found.items()})
    return s.groupby(["region", "date"], as_index=False)[["on", "off"]].sum()


def attach(a: pd.DataFrame) -> pd.DataFrame:
    w = pd.read_csv(CFG.PROCESSED_DIR / "weather_daily.csv", parse_dates=["date"], encoding="utf-8-sig")
    pan = pd.read_csv(CFG.PROCESSED_DIR / "panel_daily.csv", parse_dates=["date"], encoding="utf-8-sig")
    cal = pan.drop_duplicates(["region", "date"])[["region", "date", "dow", "month", "is_holiday", "chuseok_block",
                                                    "is_offday"]]
    a = a.merge(w[["region", "date", "rain", *LAGS]], on=["region", "date"]).merge(cal, on=["region", "date"])
    # 소상공인 매출(184일 완전관측 업종 합계)과 외지 고객 매출
    d = pan.drop(columns=["ind_group", "scope"]).merge(CFG.industry_table(), on="MCT_RY_CD")
    d = d[(d.scope == "core") & (d.reliable == 1)]
    d = d[d.groupby(["region", "MCT_RY_CD"])["observed"].transform("min") == 1]
    sales = d.groupby(["region", "date"], as_index=False).agg(sales=("sales", "sum"), sales_visitor=("sales_visitor", "sum"))
    a = a.merge(sales, on=["region", "date"])
    a["week"] = a.date.dt.isocalendar().week.astype(int)
    return a


def arrival_effects(a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cs, ce = pd.to_datetime(CFG.CHUSEOK_BLOCK)
    for label, df in [("전체", a), ("추석 제외", a[~a.date.between(cs, ce)])]:
        for region, g in df.groupby("region"):
            m = smf.ols(f"np.log(off) ~ {' + '.join(LAGS)} + {CTRL}", g).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
            for k, name in zip(LAGS, ["호우 당일", "+1일", "+2일"]):
                b, se = m.params[k], m.bse[k]
                rows.append({"sample": label, "region": region, "group": "전체", "term": name, "effect": np.exp(b) - 1,
                             "lo": np.exp(b - 1.96 * se) - 1, "hi": np.exp(b + 1.96 * se) - 1, "p": m.pvalues[k]})
            for off, gg in g.groupby("is_offday"):
                m = smf.ols(f"np.log(off) ~ heavy_rain + {CTRL}", gg).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
                b, se = m.params["heavy_rain"], m.bse["heavy_rain"]
                rows.append({"sample": label, "region": region, "group": "휴일" if off else "평일", "term": "호우 당일",
                             "effect": np.exp(b) - 1, "lo": np.exp(b - 1.96 * se) - 1, "hi": np.exp(b + 1.96 * se) - 1,
                             "p": m.pvalues["heavy_rain"]})
    return pd.DataFrame(rows)


def _explained(g: pd.DataFrame, y: str) -> tuple[float, float, float]:
    f0 = f"np.log({y}) ~ heavy_rain + {CTRL}"
    b0 = smf.ols(f0, g).fit().params["heavy_rain"]
    m1 = smf.ols(f0 + " + np.log(off)", g).fit()
    b1 = m1.params["heavy_rain"]
    return b0, b1, m1.params["np.log(off)"]


def _mediate(g: pd.DataFrame, y: str, rng) -> dict:
    b0, b1, el = _explained(g, y)
    weeks = g.week.unique()
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(weeks, len(weeks), replace=True)
        gb = pd.concat([g[g.week == w] for w in pick], ignore_index=True)
        if gb.heavy_rain.sum() < 3:
            continue
        c0, c1, _ = _explained(gb, y)
        boot.append(1 - c1 / c0 if c0 < 0 else np.nan)
    boot = np.array([x for x in boot if np.isfinite(x)])
    return {"rain_effect": np.exp(b0) - 1, "rain_effect_given_arrivals": np.exp(b1) - 1,
            "explained_share": 1 - b1 / b0, "explained_lo": np.percentile(boot, 2.5),
            "explained_hi": np.percentile(boot, 97.5), "elasticity": el, "n_boot": len(boot)}


def mediation(a: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """도착 인원을 통제하기 전후의 호우 계수 비교. 신뢰구간은 주 단위 블록 부트스트랩."""
    rng = np.random.default_rng(seed)
    rows = []
    cs, ce = pd.to_datetime(CFG.CHUSEOK_BLOCK)
    for label, df in [("전체", a), ("추석 제외", a[~a.date.between(cs, ce)])]:
        for region, g in df.groupby("region"):
            for y, name in [("sales", "소상공인 매출"), ("sales_visitor", "외지 고객 매출")]:
                rows.append({"sample": label, "region": region, "outcome": name, **_mediate(g, y, rng)})
    return pd.DataFrame(rows)


# ================================================================ 그래프
def fig_flow(eff: pd.DataFrame, med: pd.DataFrame):
    """F12: 호우 당일 '도착 인원'과 '매출' 변화 비교"""
    e = eff[(eff["sample"] == "추석 제외") & (eff.group == "전체") & (eff.term == "호우 당일")].set_index("region")
    sales = med[(med["sample"] == "추석 제외") & (med.outcome == "소상공인 매출")].set_index("region")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    cats = [("도착 인원\n(지하철 하차)", "arr"), ("소상공인 매출", "sales")]
    x = np.arange(len(cats))
    w = 0.36
    for i, region in enumerate(["강원 춘천시", "서울 강남구"]):
        vals = [e.loc[region, "effect"], sales.loc[region, "rain_effect"]]
        xs = x + (i - 0.5) * w
        ax.bar(xs, np.array(vals) * 100, w - 0.04, color=F.REGION_COLOR[region], label=F.SHORT[region])
        lo, hi = e.loc[region, "lo"], e.loc[region, "hi"]
        ax.errorbar(xs[0], vals[0] * 100, yerr=[[(vals[0] - lo) * 100], [(hi - vals[0]) * 100]], fmt="none",
                    ecolor=F.INK2, capsize=4, elinewidth=1.4)
        for xi, v, bottom in zip(xs, vals, [min(v, lo) for v in vals[:1]] + vals[1:]):
            ax.text(xi, bottom * 100 - 0.6, F.pct(v), ha="center", va="top", color=F.INK, fontsize=12)
    ax.axhline(0, color=F.INK2, lw=1)
    ax.set_xticks(x, [c for c, _ in cats])
    ax.set_ylabel("호우(30mm+) 당일 변화 (%)")
    ax.set_ylim(-22, 6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.set_title("춘천은 발길이 끊기고, 강남은 사람은 오는데 지갑이 닫힌다")
    ax.text(0, -0.15, "도착 인원 = 서울시 지하철 역별 일별 하차(춘천 경춘선 6개 역, 강남구 25개 역) · 추석 연휴 제외, 요일·월·공휴일 통제 · "
                      "오차막대 95% 신뢰구간", transform=ax.transAxes, color=F.MUTED, fontsize=10)
    F.save(fig, "F12_flow_vs_sales")


def fig_mediation(med: pd.DataFrame):
    """F13: 매출 감소를 '도착 감소로 설명되는 몫'과 '나머지(와서 덜 씀)'로 분해"""
    m = med[(med["sample"] == "추석 제외") & (med.outcome == "소상공인 매출")].set_index("region")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    regions = ["강원 춘천시", "서울 강남구"]
    y = np.arange(len(regions))[::-1]
    for yi, region in zip(y, regions):
        total = -m.loc[region, "rain_effect"] * 100
        part = total * m.loc[region, "explained_share"]
        ax.barh(yi, part, 0.55, color=F.REGION_COLOR[region], label="발길 감소로 설명" if yi == y[0] else None)
        ax.barh(yi, total - part, 0.55, left=part, color="#d4d3cf", label="나머지(와서 덜 씀)" if yi == y[0] else None)
        share = m.loc[region, "explained_share"]
        if share >= 0.2:
            ax.text(part / 2, yi, f"{share * 100:.0f}%", ha="center", va="center", color="white", fontsize=13,
                    fontweight="bold")
        else:  # 막대가 너무 짧으면 회색 구간에 적는다
            ax.text(part + 0.15, yi, f"발길 감소 몫 {share * 100:.0f}%", va="center", color=F.INK, fontsize=11)
        ax.text(total + 0.2, yi, f"매출 -{total:.1f}%", va="center", color=F.INK, fontsize=12)
    ax.set_yticks(y, [F.SHORT[r] for r in regions])
    ax.set_xlim(0, 11)
    ax.set_xlabel("호우 당일 소상공인 매출 감소 (%p)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    lo, hi = m.loc["강원 춘천시", "explained_lo"] * 100, m.loc["강원 춘천시", "explained_hi"] * 100
    ax.set_title("춘천 매출 감소의 절반 이상은 '사람이 오지 않아서' 생긴다")
    ax.text(0, -0.26, f"추석 연휴 제외 · 도착 인원을 통제하기 전후의 호우 효과 비교 · 춘천 설명 비율 95% 범위 [{lo:.0f}%, {hi:.0f}%] "
                      f"(주 단위 부트스트랩 {N_BOOT}회)", transform=ax.transAxes, color=F.MUTED, fontsize=10)
    F.save(fig, "F13_flow_mediation")


def main():
    pd.set_option("display.width", 200)
    a = attach(load_arrivals())
    a.to_csv(CFG.OUTPUT_DIR / "subway_daily.csv", index=False, encoding="utf-8-sig")
    print(a.groupby("region").agg(days=("date", "nunique"), arrivals_per_day=("off", "mean")).round(0))

    eff = arrival_effects(a)
    eff.to_csv(CFG.OUTPUT_DIR / "subway_effects.csv", index=False, encoding="utf-8-sig")
    print("\n[1] 호우와 도착 인원")
    print(eff.assign(effect=eff.effect * 100, lo=eff.lo * 100, hi=eff.hi * 100).round(3).to_string(index=False))

    med = mediation(a)
    med.to_csv(CFG.OUTPUT_DIR / "subway_mediation.csv", index=False, encoding="utf-8-sig")
    print("\n[2] 매출 감소 중 도착 인원 감소로 설명되는 몫")
    print(med.round(3).to_string(index=False))

    fig_flow(eff, med)
    fig_mediation(med)


if __name__ == "__main__":
    main()
