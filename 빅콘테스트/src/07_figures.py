"""발표용 그래프 → outputs/figures/*.png (16:9 슬라이드용)
선행: 04, 05, 06 실행
색: 지역은 고정 색(강남=파랑, 춘천=주황), 강조 외 요소는 회색. 이중 축 사용하지 않음.
"""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

import config as C  # noqa: E402

M = importlib.import_module("04_model_sensitivity")
R = importlib.import_module("05_robustness_checks")

FIG_DIR = C.OUTPUT_DIR / "figures"
SLIDE_DIR = C.OUTPUT_DIR / "figures_slide"  # 메인 제목 없는 슬라이드 삽입용
REGION_COLOR = {"서울 강남구": "#2a78d6", "강원 춘천시": "#eb6834"}
SHORT = {"서울 강남구": "강남", "강원 춘천시": "춘천"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SEQ = {"1순위 지원": "#1c5cab", "2순위 지원": "#6da7ec", "모니터링": "#d4d3cf"}

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 12,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
})


def strip_main_title(fig):
    """슬라이드용: 그림의 메인 제목만 지운다(슬라이드 제목과 중복 방지). 패널 제목·출처 주석은 유지."""
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    else:
        fig.axes[0].set_title("", loc="left")


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    strip_main_title(fig)
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SLIDE_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def pct(x, signed=True):
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


# ================================================================ 공통: 지역 합계 회귀
def region_fit(p: pd.DataFrame, region: str, value: str = "sales", extra: list[str] = ()):
    pr = p[p.region == region]
    full = pr.groupby("MCT_RY_CD")["observed"].transform("min") == 1
    agg = pr[full].groupby("date")[value].sum()
    day = pr.drop_duplicates("date")[["date", "dow", "month", *M.WEATHER, *M.CAL_FLAGS, *extra]]
    X = M.design(day.drop(columns=list(extra)))
    for c in extra:
        X[c] = day.set_index("date").sort_index()[c].astype(float)
    X = X[M.usable(X)]
    y = np.log(agg.loc[X.index])
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": M.HAC_LAGS}), X, y


def effect(m, name):
    b, se = m.params[name], m.bse[name]
    return np.exp(b) - 1, np.exp(b) * se


# ================================================================ F1 헤드라인
def fig_headline():
    rt = pd.read_csv(C.OUTPUT_DIR / "robust_region_net.csv", encoding="utf-8-sig")
    rt = rt[rt.balanced]
    diff = rt[rt.region.str.startswith("차이")].iloc[0]
    rt = rt[~rt.region.str.startswith("차이")].set_index("region")
    comps = [("gross", "호우 당일"), ("after2d", "이후 2일"), ("net", "3일 합계(순손실)")]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(comps))
    w = 0.36
    for i, region in enumerate(["강원 춘천시", "서울 강남구"]):
        r = rt.loc[region]
        vals = np.array([r[c] for c, _ in comps])
        lo = np.array([r[f"{c}_lo"] for c, _ in comps])
        hi = np.array([r[f"{c}_hi"] for c, _ in comps])
        xs = x + (i - 0.5) * w
        ax.bar(xs, vals * 100, w - 0.04, color=REGION_COLOR[region], label=SHORT[region])
        ax.errorbar(xs, vals * 100, yerr=[(vals - lo) * 100, (hi - vals) * 100], fmt="none", ecolor=INK2,
                    elinewidth=1.4, capsize=4)
        for xi, v, l in zip(xs, vals, lo):
            ax.text(xi, min(v, l) * 100 - 1.2, pct(v), ha="center", va="top", color=INK, fontsize=12,
                    fontweight="bold" if xi == xs[-1] else "normal")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, [lab for _, lab in comps])
    ax.set_ylabel("평상시 대비 매출 변화 (%)")
    ax.set_ylim(-26, 12)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("같은 비, 다른 피해: 호우(30mm+) 시 춘천 소상공인 순손실은 강남의 2배")
    ax.text(0, -0.13, f"순손실 차이(춘천-강남) {diff.net * 100:.1f}%p, 95% CI [{diff.net_lo * 100:.1f}, "
                      f"{diff.net_hi * 100:.1f}], p={diff.p_value:.3f} · 오차막대 95% 신뢰구간 · 의료 제외 소상공인 업종 합계",
            transform=ax.transAxes, color=MUTED, fontsize=10)
    save(fig, "F1_headline_same_rain_different_damage")


# ================================================================ F2 이벤트 스터디
def fig_event_study(p):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    names = ["heavy_rain_lead1", "heavy_rain", "heavy_rain_lag1", "heavy_rain_lag2", "heavy_rain_lag3"]
    days = np.arange(-1, 4)
    for k, region in enumerate(["서울 강남구", "강원 춘천시"]):
        m, *_ = region_fit(p, region, extra=["heavy_rain_lead1", "heavy_rain_lag3"])
        e = np.array([effect(m, n) for n in names])
        xs = days + (k - 0.5) * 0.08
        c = REGION_COLOR[region]
        ax.fill_between(xs, (e[:, 0] - 1.96 * e[:, 1]) * 100, (e[:, 0] + 1.96 * e[:, 1]) * 100, color=c, alpha=0.12,
                        lw=0)
        ax.plot(xs, e[:, 0] * 100, color=c, lw=2, marker="o", ms=8, mec="white", mew=2, label=SHORT[region])
    ax.axhline(0, color=INK2, lw=1)
    ax.axvspan(-0.3, 0.3, color=GRID, alpha=0.6, lw=0)
    ax.set_xticks(days, ["호우 전날", "호우 당일", "+1일", "+2일", "+3일"])
    ax.set_ylabel("평상시 대비 매출 변화 (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.set_title("비가 그친 뒤: 강남은 다음날 평소 수준으로 돌아오고, 춘천은 반등이 없다")
    ax.text(0, -0.13, "음영 95% 신뢰구간(+1·+2일 각각은 0 포함) · 요일·월·공휴일 통제 · 184일 완전관측 소상공인 업종 합계",
            transform=ax.transAxes, color=MUTED, fontsize=10)
    save(fig, "F2_event_study_recovery")


# ================================================================ F3 현지/외지 고객
def fig_visitor(p):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    rows = []
    for region in ["강원 춘천시", "서울 강남구"]:
        for seg, lab in [("sales_local", "현지 고객"), ("sales_visitor", "외지 고객")]:
            m, *_ = region_fit(p, region, value=seg)
            e, se = effect(m, "heavy_rain")
            rows.append((region, lab, e, se))
    d = pd.DataFrame(rows, columns=["region", "seg", "e", "se"])
    ylab = [f"{SHORT[r]} · {s}" for r, s in zip(d.region, d.seg)]
    y = np.arange(len(d))[::-1]
    colors = [REGION_COLOR[r] if s == "외지 고객" else "#c3c2b7" for r, s in zip(d.region, d.seg)]
    ax.barh(y, d.e * 100, 0.6, color=colors)
    ax.errorbar(d.e * 100, y, xerr=1.96 * d.se * 100, fmt="none", ecolor=INK2, capsize=4, elinewidth=1.4)
    for yi, e, se in zip(y, d.e, d.se):
        ax.text((e - 1.96 * se) * 100 - 0.8, yi, pct(e), ha="right", va="center", color=INK)
    ax.set_yticks(y, ylab)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-38, 8)
    ax.set_xlabel("호우 당일 매출 변화 (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("비는 '찾아오는 손님'을 먼저 막는다: 춘천 외지 고객 매출 급감")
    ax.text(0, -0.15, "외지 = 가맹점 소재 시도 밖 거주 고객(신한카드 데이터2) · 오차막대 95% 신뢰구간",
            transform=ax.transAxes, color=MUTED, fontsize=10)
    save(fig, "F3_visitor_vs_local")


# ================================================================ F4 업종군
def fig_groups():
    g = pd.read_csv(C.OUTPUT_DIR / "robust_group_net.csv", encoding="utf-8-sig")
    g = g[g.n_ind >= 2]
    order = (g[g.region == "강원 춘천시"].sort_values("net").ind_group.tolist())
    order += [x for x in g.ind_group.unique() if x not in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, region in zip(axes, ["강원 춘천시", "서울 강남구"]):
        d = g[g.region == region].set_index("ind_group").reindex(order).dropna(subset=["net"])
        y = np.array([order.index(i) for i in d.index])[::1]
        c = REGION_COLOR[region]
        ax.errorbar(d.net * 100, y, xerr=[(d.net - d.net_lo) * 100, (d.net_hi - d.net) * 100], fmt="o", color=c,
                    ms=9, mec="white", mew=2, ecolor=c, elinewidth=1.6, capsize=0, label="순손실(3일)")
        ax.scatter(d.gross * 100, y, s=60, facecolors="white", edgecolors=INK2, lw=1.5, zorder=3, label="호우 당일")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(SHORT[region], fontsize=14)
        ax.set_xlim(-60, 30)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("평상시 대비 매출 변화 (%)")
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.95))
    fig.suptitle("업종군별 호우 피해: 춘천 여가·숙박 -39%, 생활소매·외식은 비 그친 뒤에도 감소", x=0.01, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    fig.text(0.01, -0.02, "점 = 순손실(당일+2일), 가로선 95% 신뢰구간, 빈 원 = 호우 당일 · 업종 2개 이상 업종군",
             color=MUTED, fontsize=10)
    fig.tight_layout()
    save(fig, "F4_industry_groups")


# ================================================================ F5 축소추정
def fig_shrinkage():
    s = pd.read_csv(C.OUTPUT_DIR / "sensitivity_series.csv", encoding="utf-8-sig")
    s = s[(s.region == "강원 춘천시") & (s.ind_group != "의료")]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for _, r in s.iterrows():
        ax.plot([0, 1], [r.gross * 100, r.gross_eb * 100], color="#c3c2b7", lw=1.2)
    ax.scatter(np.zeros(len(s)), s.gross * 100, s=40, color=MUTED, zorder=3)
    ax.scatter(np.ones(len(s)), s.gross_eb * 100, s=40, color=REGION_COLOR["강원 춘천시"], zorder=3)
    ext = s.assign(move=(s.gross - s.gross_eb).abs()).nlargest(3, "move")
    for _, r in ext.iterrows():
        ax.plot([0, 1], [r.gross * 100, r.gross_eb * 100], color=INK2, lw=2)
        ax.annotate(f"{r.MCT_RY_CD} (일 {r.baseline_daily / 1e6:.1f}백만원)", (0, r.gross * 100), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=10, color=INK)
    sd_raw, sd_eb = s.gross.std() * 100, s.gross_eb.std() * 100
    ax.set_xticks([0, 1], ["업종별 개별 추정", "축소추정(업종군 평균으로 당김)"])
    ax.set_xlim(-0.7, 1.3)
    ax.set_ylabel("호우 당일 매출 변화 (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("표본이 작은 업종의 튀는 추정을 업종군 정보로 안정화")
    ax.text(0, -0.13, f"춘천 {len(s)}개 업종 · 업종 간 표준편차 {sd_raw:.1f}%p → {sd_eb:.1f}%p · 경험적 베이즈(DerSimonian-Laird)",
            transform=ax.transAxes, color=MUTED, fontsize=10)
    save(fig, "F5_shrinkage")


# ================================================================ F6 검증
def fig_validation():
    # 14단계(머신러닝 비교)를 돌렸으면 그 요약을 쓴다: A·B·C는 04와 같은 값, D(부스팅)가 추가됨
    ml_path = C.OUTPUT_DIR / "cv_ml_summary.csv"
    cv = pd.read_csv(ml_path if ml_path.exists() else C.OUTPUT_DIR / "cv_summary.csv", encoding="utf-8-sig")
    cv = cv[cv.window == "호우 당일"].set_index("region")
    models = [("WAPE_A_calendar", "달력만"), ("WAPE_B_weather", "+기상(개별)"), ("WAPE_C_weather_eb", "+기상(축소추정)")]
    if "WAPE_D_boosting" in cv:
        models.append(("WAPE_D_boosting", "머신러닝\n(부스팅)"))
    n = len(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, region in zip(axes, ["강원 춘천시", "서울 강남구"]):
        vals = [cv.loc[region, k] * 100 for k, _ in models]
        cols = ["#d4d3cf", "#a9a8a2", REGION_COLOR[region], "#6b6a64"][:n]
        ax.bar(range(n), vals, 0.6, color=cols)
        best = int(np.argmin(vals))
        for i, v in enumerate(vals):
            ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", color=INK, fontweight="bold" if i == best else "normal")
        imp = (1 - vals[2] / vals[0]) * 100
        title = f"{SHORT[region]}: 축소추정 오차 {imp:.0f}% 감소"
        if n == 4:
            title += f" · 부스팅 {(1 - vals[3] / vals[0]) * 100:.0f}%"
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(n), [lab for _, lab in models])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("호우 당일 예측오차 WAPE (%)")
    fig.suptitle("학습에 쓰지 않은 호우 이벤트로 검증: 춘천은 축소추정 회귀, 강남은 부스팅이 가장 정확" if n == 4
                 else "학습에 쓰지 않은 호우 이벤트로 검증: 축소추정 모형이 가장 정확", x=0.01, ha="left", fontsize=16,
                 fontweight="bold", color=INK)
    fig.text(0.01, -0.03, "Leave-one-event-out: 호우 이벤트(춘천 14, 강남 13개)를 하나씩 제외하고 학습 후 예측 · "
                          "WAPE = Σ|실제-예측| / Σ실제 · 부스팅 = HistGradientBoosting(업종 통합 학습)",
             color=MUTED, fontsize=10)
    fig.tight_layout()
    save(fig, "F6_validation")


# ================================================================ F7 우선지원
def fig_priority(scenario="춘천 호우 60mm", top=15):
    d = pd.read_csv(C.OUTPUT_DIR / "priority_scenarios.csv", encoding="utf-8-sig")
    d = d[d.scenario == scenario].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(d))
    ax.barh(y, -d.exp_rate * 100, 0.66, color=[SEQ[t] for t in d.tier])
    for yi, r in zip(y, d.itertuples()):
        ax.text(-r.exp_rate * 100 + 0.8, yi, f"{-r.exp_rate * 100:.0f}% · 손실 일 {r.exp_loss_won / 1e6:.0f}백만원",
                va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [f"{r.MCT_RY_CD}" for r in d.itertuples()])
    ax.set_xlim(0, max(-d.exp_rate * 100) * 1.45)
    ax.set_xlabel("예상 순손실률 (%, 호우 당일+2일)")
    ax.grid(axis="y", visible=False)
    tiers = [t for t in SEQ if t in set(d.tier)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEQ[t]) for t in tiers]
    ax.legend(handles, tiers, frameon=False, loc="lower right")
    ax.set_title(f"예보 → 우선지원 목록: '{scenario}' 시나리오 상위 {top}개 업종")
    ax.text(0, -0.16, "막대 = 예상 순손실률, 금액 = 예상 하루 손실액 · 점수 = 0.4·피해규모 + 0.4·손실률 + 0.2·회복불가도 · "
                      "1·2순위 = 손실 확실(단측 95%) 업종", transform=ax.transAxes, color=MUTED, fontsize=10)
    save(fig, "F7_priority_chuncheon_rain")


# ================================================================ F8 타임라인(춘천)
def fig_timeline(p, region="강원 춘천시"):
    pr = p[p.region == region]
    full = pr.groupby("MCT_RY_CD")["observed"].transform("min") == 1
    agg = pr[full].groupby("date").sales.sum()
    day = pr.drop_duplicates("date").set_index("date").sort_index()
    X = M.design(day.reset_index()[["date", "dow", "month", *M.WEATHER, *M.CAL_FLAGS]])
    cal = [c for c in X.columns if c not in M.WEATHER]
    normal = (day[["heavy_rain", "heavy_rain_lag1", "heavy_rain_lag2", "rain_mid", "rain_light"]].sum(axis=1) == 0) \
        if "rain_light" in day else (day[["heavy_rain", "heavy_rain_lag1", "heavy_rain_lag2"]].sum(axis=1) == 0)
    m = sm.OLS(np.log(agg[normal]), X.loc[normal, cal]).fit()
    dev = (agg / np.exp(m.predict(X[cal])) - 1) * 100
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    a1.bar(day.index, day.rain, color="#6da7ec", width=0.9)
    a1.set_ylabel("일 강수량(mm)")
    a1.axhline(C.HEAVY_RAIN_MM, color=INK2, lw=1, ls="--")
    a1.text(day.index[-1], C.HEAVY_RAIN_MM + 3, "호우 30mm", ha="right", color=INK2, fontsize=10)
    a1.grid(axis="x", visible=False)
    heavy = day.heavy_rain == 1
    a2.axhline(0, color=INK2, lw=1)
    a2.plot(dev.index, dev, color="#c3c2b7", lw=1.2)
    a2.scatter(dev.index[heavy], dev[heavy], color=REGION_COLOR[region], s=36, zorder=3, label="호우일")
    a2.set_ylabel("평상시 예측 대비 매출 (%)")
    a2.set_ylim(-45, 30)
    cs, ce = pd.to_datetime(C.CHUSEOK_BLOCK)
    a2.axvspan(cs, ce, color=GRID, alpha=0.7, lw=0)
    a2.text(cs, 26, "추석 연휴", color=INK2, fontsize=10)
    a2.legend(frameon=False, loc="lower left")
    a2.grid(axis="x", visible=False)
    a1.set_title(f"{SHORT[region]} 2025년 하반기: 호우일 {int(heavy.sum())}일 중 {int((dev[heavy] < 0).sum())}일은 "
                 f"소상공인 매출이 평소보다 낮았다")
    a2.text(0, -0.2, "기준선 = 비 오지 않은 날로 학습한 요일·월·공휴일 모형 · 기상청 ASOS 춘천(101)",
            transform=a2.transAxes, color=MUTED, fontsize=10)
    save(fig, "F8_timeline_chuncheon")


# ================================================================ F9 폭염 무효과
def fig_heat(p):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, region in zip(axes, ["강원 춘천시", "서울 강남구"]):
        m, X, y = region_fit(p, region)
        summer = X.index[(X.index.month <= 9)]
        tcols = [c for c in X.columns if c != "heat33"]
        mm = sm.OLS(y, X[tcols]).fit()
        resid = (np.exp(y - mm.predict(X[tcols])) - 1) * 100
        tmax = p[p.region == region].drop_duplicates("date").set_index("date").tmax
        dry = X.loc[summer, ["heavy_rain", "rain_mid"]].sum(axis=1) == 0
        sd = summer[dry.values]
        ax.scatter(tmax.loc[sd], resid.loc[sd], s=36, color=REGION_COLOR[region], alpha=0.75, edgecolors="white")
        ax.axhline(0, color=INK2, lw=1)
        ax.axvline(C.HEAT_TMAX, color=INK2, lw=1, ls="--")
        e, se = effect(m, "heat33")
        ax.set_title(f"{SHORT[region]}: 폭염일 효과 {pct(e)} (95% CI ±{1.96 * se * 100:.1f}%p)", fontsize=14)
        ax.set_xlabel("일 최고기온(℃), 7~9월 비 없는 날")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("요일·월 통제 후 매출 편차 (%)")
    fig.suptitle("폭염은 일 매출을 줄이지 않았다: 33℃ 이상에서도 매출 편차는 0 근처", x=0.01, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "F9_heat_null")


# ================================================================ F11 피해 규모
def fig_damage():
    s = pd.read_csv(C.OUTPUT_DIR / "sensitivity_series.csv", encoding="utf-8-sig")
    s = s[s.ind_group != "의료"]
    g = s.groupby(["ind_group", "region"]).damage_net_won.sum().div(1e8).unstack("region").fillna(0)
    g = g[(g.max(axis=1) >= 1)].sort_values("강원 춘천시")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(g))
    h = 0.38
    for k, region in enumerate(["강원 춘천시", "서울 강남구"]):
        ys = y + (0.5 - k) * h
        ax.barh(ys, g[region], h - 0.04, color=REGION_COLOR[region], label=SHORT[region])
        for yi, v in zip(ys, g[region]):
            ax.text(v + 1, yi, f"{v:.0f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, g.index)
    ax.set_xlabel("6개월 호우 순손실 (억 원, 당일+이후 2일)")
    ax.set_xlim(0, g.values.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    tot = s.groupby("region").damage_net_won.sum() / 1e8
    ax.set_title(f"호우 순손실 규모: 춘천 {tot['강원 춘천시']:.0f}억, 강남 {tot['서울 강남구']:.0f}억 (2025.7~12)")
    ax.text(0, -0.17, "업종별 축소추정 순손실률 × 평상시 일매출 × 호우일수(춘천 18일, 강남 15일) 합산 · 의료 제외",
            transform=ax.transAxes, color=MUTED, fontsize=10)
    save(fig, "F11_damage_scale")


def main():
    p = M.load_panel()
    p = p[p.scope == "core"]
    fig_headline()
    fig_event_study(p)
    fig_visitor(p)
    fig_groups()
    fig_shrinkage()
    fig_validation()
    fig_priority()
    fig_timeline(p)
    fig_heat(p)
    fig_damage()


if __name__ == "__main__":
    main()
