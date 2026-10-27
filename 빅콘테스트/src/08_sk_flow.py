"""SK 유동인구로 '방문형 상권' 노출도 산출 → 카드 분석의 '외지 고객이 비에 취약' 결과를 공간으로 연결

1) 격자(50m)를 250m 블록으로 묶고 월별 주말/평일 일평균 유동 비율(방문형 지수)을 계산
   - 12월 파일 중복 행 제거(운영사무국 공지)
   - 유동 상위 30% 블록(상권 지역)만 사용: 저유동 격자의 주말 0값(마스킹 추정) 영향 차단
2) 타당성: 카드 데이터에서 주말일수록 외지 고객 매출 비중이 높은가
3) 탐색적 검정: 방문형 블록일수록 호우가 잦은 달에 유동이 더 줄었는가
   log(유동_bm) = 블록 FE + 월 FE + β·(방문형 지수 z × 월 호우일수)
   월이 6개뿐이고 여름 관광 계절성과 겹치므로 참고용
4) 지도 F10
출력: outputs/sk_block_exposure.csv, sk_did.csv, figures/F10_sk_visitor_type_map.png
"""
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

import config as C  # noqa: E402

SK_DIR = C.ROOT / "2026빅콘테스트_sk통신인구_데이터_레이아웃"
REGION_CODE = {"11230": "서울 강남구", "32010": "강원 춘천시"}
BLOCK_M = 250
BUSY_QUANTILE = 0.70
WEEKDAY = ["FLOW_POP_CNT_MON", "FLOW_POP_CNT_TUS", "FLOW_POP_CNT_WED", "FLOW_POP_CNT_THU", "FLOW_POP_CNT_FRI"]
WEEKEND = ["FLOW_POP_CNT_SAT", "FLOW_POP_CNT_SUN"]


def load_weekday_flow() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(SK_DIR / "flow_wkdy_pop_*.csv"))):
        d = pd.read_csv(f, sep="|", dtype={"BLOCK_CD": str, "STD_YM": str}, encoding="utf-8-sig")
        n0 = len(d)
        d = d.drop_duplicates()
        if len(d) < n0:
            print(f"  {f[-10:-4]}: 중복 {n0 - len(d):,}행 제거")
        d["region"] = d.BLOCK_CD.str[:5].map(REGION_CODE)
        d = d[d.region.notna()]
        d["wk"] = d[WEEKDAY].mean(axis=1)
        d["we"] = d[WEEKEND].mean(axis=1)
        frames.append(d[["STD_YM", "region", "X_COORD", "Y_COORD", "wk", "we"]])
    d = pd.concat(frames, ignore_index=True)
    d["bx"] = (d.X_COORD // BLOCK_M).astype(int)
    d["by"] = (d.Y_COORD // BLOCK_M).astype(int)
    return d


def block_panel(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bm = d.groupby(["region", "bx", "by", "STD_YM"], as_index=False)[["wk", "we"]].sum()
    bm["flow"] = (5 * bm.wk + 2 * bm.we) / 7
    blk = bm.groupby(["region", "bx", "by"], as_index=False).agg(wk=("wk", "sum"), we=("we", "sum"),
                                                               flow=("flow", "mean"), n_month=("STD_YM", "nunique"))
    blk = blk[blk.n_month == 6]
    thr = blk.groupby("region").flow.transform(lambda s: s.quantile(BUSY_QUANTILE))
    blk = blk[blk.flow >= thr].copy()
    blk["visit_idx"] = blk.we / blk.wk  # >1 주말 방문형, <1 평일 직장형
    blk["visit_z"] = blk.groupby("region").visit_idx.transform(lambda s: (s - s.mean()) / s.std())
    blk["x_km"] = (blk.bx + 0.5) * BLOCK_M / 1000
    blk["y_km"] = (blk.by + 0.5) * BLOCK_M / 1000
    bm = bm.merge(blk[["region", "bx", "by", "visit_idx", "visit_z"]], on=["region", "bx", "by"])
    return blk, bm


def card_check() -> pd.DataFrame:
    p = pd.read_csv(C.PROCESSED_DIR / "panel_daily.csv", parse_dates=["date"], encoding="utf-8-sig",
                    usecols=["date", "region", "scope", "sales", "sales_visitor", "is_offday", "chuseok_block"])
    p = p[(p.scope == "core") & (p.chuseok_block == 0)]
    g = p.groupby(["region", "is_offday"])[["sales", "sales_visitor"]].sum()
    g["visitor_share"] = g.sales_visitor / g.sales
    return g.reset_index()


def did(bm: pd.DataFrame) -> pd.DataFrame:
    w = pd.read_csv(C.PROCESSED_DIR / "weather_daily.csv", parse_dates=["date"], encoding="utf-8-sig")
    w = w[w.date.between(C.ANALYSIS_START, C.ANALYSIS_END)]
    w["STD_YM"] = w.date.dt.strftime("%Y%m")
    rain = w.groupby(["region", "STD_YM"], as_index=False).heavy_rain.sum().rename(columns={"heavy_rain": "rain_days"})
    d = bm.merge(rain, on=["region", "STD_YM"])
    d = d[d.flow > 0].copy()
    d["lflow"] = np.log(d.flow)
    d["blk"] = d.bx.astype(str) + "_" + d.by.astype(str)
    d["x"] = d.visit_z * d.rain_days
    rows = []
    for region, g in d.groupby("region"):
        g = g[g.groupby("blk").STD_YM.transform("size") == 6].copy()  # 균형 패널
        # 블록·월 이원 고정효과 = 균형 패널의 이원 평균 제거
        for v in ["lflow", "x"]:
            g[f"{v}_dm"] = (g[v] - g.groupby("blk")[v].transform("mean") - g.groupby("STD_YM")[v].transform("mean")
                            + g[v].mean())
        m = smf.ols("lflow_dm ~ x_dm - 1", g).fit(cov_type="cluster", cov_kwds={"groups": g.blk})
        b, se = m.params["x_dm"], m.bse["x_dm"]
        rows.append({"region": region, "n_block": g.blk.nunique(),
                     "rain_days_by_month": dict(zip(*rain[rain.region == region][["STD_YM", "rain_days"]].values.T)),
                     "beta_pct": (np.exp(b) - 1) * 100, "se_pct": se * 100, "p": m.pvalues["x_dm"]})
    return pd.DataFrame(rows)


def fig_map(blk: pd.DataFrame):
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, region in zip(axes, ["강원 춘천시", "서울 강남구"]):
        b = blk[blk.region == region]
        # 블록을 래스터로 그려 축척과 무관하게 실제 면적을 채운다
        x0, y0 = b.bx.min(), b.by.min()
        grid = np.full((b.by.max() - y0 + 1, b.bx.max() - x0 + 1), np.nan)
        grid[b.by - y0, b.bx - x0] = np.log2(b.visit_idx).clip(-1, 1)
        km = BLOCK_M / 1000
        ext = [x0 * km, (b.bx.max() + 1) * km, y0 * km, (b.by.max() + 1) * km]
        sc = ax.imshow(grid, origin="lower", extent=ext, cmap=cmap, norm=TwoSlopeNorm(0, -1, 1),
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        share = (b.visit_idx > 1).mean() * 100
        ax.set_title(f"{region.split()[1]}: 방문형 블록 {share:.0f}% (주말/평일 중앙값 {b.visit_idx.median():.2f})",
                     loc="left", fontsize=13, fontweight="bold", color="#0b0b0b")
        ax.plot([ext[0], ext[0] + 1], [ext[2] - 0.3] * 2, color="#52514e", lw=2, clip_on=False)
        ax.text(ext[0] + 0.5, ext[2] - 0.45, "1km", ha="center", va="top", fontsize=9, color="#52514e")
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04, ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["평일 2배(직장형)", "같음", "주말 2배(방문형)"])
    cb.outline.set_visible(False)
    fig.suptitle("SK 유동인구로 본 '찾아오는 상권': 춘천은 주말 방문형, 강남은 평일 직장형", x=0.01, ha="left",
                 fontsize=16, fontweight="bold", color="#0b0b0b")
    fig.text(0.01, 0.01, f"SK텔레콤 요일별 유동인구(2025.7~12) · 50m 격자를 250m 블록으로 집계 · 유동 상위 {int((1 - BUSY_QUANTILE) * 100)}% 블록 · "
                         "좌표 UTM-K", color="#8a8984", fontsize=10)
    out = C.OUTPUT_DIR / "figures" / "F10_sk_visitor_type_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig._suptitle.set_text("")  # 슬라이드용: 메인 제목 제거
    (C.OUTPUT_DIR / "figures_slide").mkdir(parents=True, exist_ok=True)
    fig.savefig(C.OUTPUT_DIR / "figures_slide" / out.name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved", out.name)


def main():
    pd.set_option("display.width", 200)
    print("[1] SK 요일별 유동 → 250m 블록 방문형 지수")
    d = load_weekday_flow()
    blk, bm = block_panel(d)
    blk.to_csv(C.OUTPUT_DIR / "sk_block_exposure.csv", index=False, encoding="utf-8-sig")
    print(blk.groupby("region").agg(n_block=("visit_idx", "size"), median_idx=("visit_idx", "median"),
                                    visit_type_share=("visit_idx", lambda s: (s > 1).mean())).round(3))

    print("\n[2] 타당성: 휴일(주말·공휴일)에 외지 고객 비중이 높은가 (카드 데이터2, 추석 제외)")
    print(card_check().round(3).to_string(index=False))

    print("\n[3] 탐색: 방문형 블록(+1 SD)은 월 호우일 1일당 유동이 몇 % 더 변하는가")
    dd = did(bm)
    dd.to_csv(C.OUTPUT_DIR / "sk_did.csv", index=False, encoding="utf-8-sig")
    print(dd.round(3).to_string(index=False))

    print("\n[4] 지도")
    fig_map(blk)


if __name__ == "__main__":
    main()
