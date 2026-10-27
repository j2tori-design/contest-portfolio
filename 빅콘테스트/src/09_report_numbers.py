"""발표자료에 쓰인 핵심 숫자를 한 파일로 모은다 → outputs/report_numbers.csv
선행: 04, 05, 06, 08, 10, 11 실행
다른 단계에서 표·그래프 안에서만 계산되던 숫자(추석 제외 검정, 현지/외지 효과, 폭염 효과, 피해 비율 등)를
여기서 다시 계산해 저장하므로, 발표 숫자는 모두 이 파일 한 곳에서 확인할 수 있다.
"""
import importlib

import numpy as np
import pandas as pd

import config as C

M = importlib.import_module("04_model_sensitivity")
R = importlib.import_module("05_robustness_checks")
F = importlib.import_module("07_figures")

rows = []


def add(slide, item, region, value, unit, source):
    rows.append({"slide": slide, "item": item, "region": region, "value": value, "unit": unit, "source": source})


def main():
    p = M.load_panel()
    core = p[p.scope == "core"]
    s = pd.read_csv(C.OUTPUT_DIR / "sensitivity_series.csv", encoding="utf-8-sig")
    s = s[s.ind_group != "의료"]

    # ---------------------------------------------------- 지역 순손실 (전체 / 추석 제외)
    cs, ce = pd.to_datetime(C.CHUSEOK_BLOCK)
    for label, df in [("전체", core), ("추석 제외", core[~core.date.between(cs, ce)])]:
        rt = pd.DataFrame(R.region_test(df, label))
        rt = rt[rt.balanced]
        for _, r in rt.iterrows():
            if r.region.startswith("차이"):
                add("7", f"순손실 차이(춘천-강남), {label}", "", r.net * 100, "%p", "05 region_test")
                add("7", f"차이 p값, {label}", "", r.p_value, "", "05 region_test")
                add("7", f"차이 95% 하한, {label}", "", r.net_lo * 100, "%p", "05 region_test")
                add("7", f"차이 95% 상한, {label}", "", r.net_hi * 100, "%p", "05 region_test")
            else:
                for k, name in [("gross", "호우 당일"), ("after2d", "이후 2일"), ("net", "순손실(3일)")]:
                    add("1·2·7", f"{name}, {label}", r.region, r[k] * 100, "%", "05 region_test")
                    add("7", f"{name} 95% 하한, {label}", r.region, r[f"{k}_lo"] * 100, "%", "05 region_test")
                    add("7", f"{name} 95% 상한, {label}", r.region, r[f"{k}_hi"] * 100, "%", "05 region_test")

    # ---------------------------------------------------- 현지/외지, 폭염, 강설
    for region in C.REGIONS:
        for seg, name in [("sales_local", "현지 고객"), ("sales_visitor", "외지 고객")]:
            m, *_ = F.region_fit(core, region, value=seg)
            for h, hname in [("heavy_rain", "호우 당일"), ("snow", "강설일")]:
                e, se = F.effect(m, h)
                add("9", f"{name} {hname} 매출 변화", region, e * 100, "%", "07 region_fit")
                add("9", f"{name} {hname} p값", region, float(m.pvalues[h]), "", "07 region_fit")
        m, *_ = F.region_fit(core, region)
        e, se = F.effect(m, "heat33")
        add("6", "폭염일(33℃+) 매출 변화", region, e * 100, "%", "07 region_fit")
        add("6", "폭염일 95% 반폭", region, 1.96 * se * 100, "%p", "07 region_fit")

    # ---------------------------------------------------- 피해 규모
    sales6 = core.groupby("region").sales.sum()
    days = core.drop_duplicates(["region", "date"]).groupby("region").heavy_rain.sum()
    rt = pd.DataFrame(R.region_test(core, "전체"))
    rt = rt[rt.balanced & ~rt.region.str.startswith("차이")].set_index("region")
    for region, g in s.groupby("region"):
        net = g.damage_net_won.sum()
        base = g.baseline_daily.sum()
        add("15", "6개월 순손실액", region, net / 1e8, "억 원", "04 damage_net_won 합")
        add("15", "6개월 총손실액(당일 기준)", region, g.damage_gross_won.sum() / 1e8, "억 원", "04 damage_gross_won 합")
        add("15", "6개월 매출 대비 순손실", region, net / sales6[region] * 100, "%", "09")
        add("15", "호우 1일당 순손실", region, net / days[region] / 1e8, "억 원", "09")
        add("15", "호우일 수", region, days[region], "일", "01 heavy_rain")
        for k, name in [("net", "검산(지역 회귀)"), ("net_hi", "검산 하한"), ("net_lo", "검산 상한")]:
            add("15", name, region, -rt.loc[region, k] * base * days[region] / 1e8, "억 원", "05 × 평상시 일매출 × 호우일")

    # ---------------------------------------------------- 축소추정·검증·알고리즘·SK
    ch = s[s.region == "강원 춘천시"]
    add("16", "업종 간 편차(개별 추정)", "강원 춘천시", ch.gross.std() * 100, "%p", "04 gross")
    add("16", "업종 간 편차(축소추정)", "강원 춘천시", ch.gross_eb.std() * 100, "%p", "04 gross_eb")
    cv = pd.read_csv(C.OUTPUT_DIR / "cv_summary.csv", encoding="utf-8-sig")
    for _, r in cv.iterrows():
        for m_ in ["A_calendar", "B_weather", "C_weather_eb"]:
            add("17", f"WAPE {r.window} {m_}", r.region, r[f"WAPE_{m_}"] * 100, "%", "04 cv_summary")
    ml_path = C.OUTPUT_DIR / "cv_ml_summary.csv"
    if ml_path.exists():
        for _, r in pd.read_csv(ml_path, encoding="utf-8-sig").iterrows():
            add("17", f"WAPE {r.window} D_boosting", r.region, r.WAPE_D_boosting * 100, "%", "14 cv_ml_summary")
    bt = pd.read_csv(C.OUTPUT_DIR / "priority_backtest.csv", encoding="utf-8-sig")
    for region, g in bt.groupby("region"):
        add("18", "백테스트 상위10 적중률", region, g.precision_at_10.mean() * 100, "%", "06 priority_backtest")
        add("18", "무작위 기대 적중률", region, g.random_expect.mean() * 100, "%", "06 priority_backtest")
    ws = pd.read_csv(C.OUTPUT_DIR / "priority_weight_sensitivity.csv", encoding="utf-8-sig")
    for _, r in ws.iterrows():
        add("18", "가중치 변경 시 상위10 유지율", r.scenario, r.mean_top10_overlap * 100, "%", "06 weight_sensitivity")
    sk = pd.read_csv(C.OUTPUT_DIR / "sk_block_exposure.csv", encoding="utf-8-sig")
    for region, g in sk.groupby("region"):
        add("12", "방문형 블록 비율", region, (g.visit_idx > 1).mean() * 100, "%", "08 sk_block_exposure")
        add("12", "주말/평일 비율 중앙값", region, g.visit_idx.median(), "배", "08 sk_block_exposure")
    off = pd.read_csv(C.PROCESSED_DIR / "panel_daily.csv", encoding="utf-8-sig",
                      usecols=["region", "scope", "sales", "sales_visitor", "is_offday", "chuseok_block"])
    off = off[(off.scope == "core") & (off.chuseok_block == 0)]
    for (region, o), g in off.groupby(["region", "is_offday"]):
        add("12", f"외지 고객 매출 비중({'휴일' if o else '평일'})", region,
            g.sales_visitor.sum() / g.sales.sum() * 100, "%", "03 panel_daily")
    cov = pd.read_csv(C.PROCESSED_DIR / "industry_coverage.csv", encoding="utf-8-sig")
    cov = cov.drop(columns=["scope", "ind_group"]).merge(C.industry_table(), on="MCT_RY_CD")
    add("3·4", "전체 업종 수", "", cov.MCT_RY_CD.nunique(), "개", "02 industry_coverage")
    add("3·4", "분석 대상 소상공인 업종 수", "",
        cov[(cov.reliable == 1) & (cov.scope == "core")].MCT_RY_CD.nunique(), "개", "02 industry_coverage")

    # ---------------------------------------------------- 지하철 도착 인원·원인 분해 (10)
    eff = pd.read_csv(C.OUTPUT_DIR / "subway_effects.csv", encoding="utf-8-sig")
    eff = eff[eff.term == "호우 당일"]
    for _, r in eff.iterrows():
        tag = f"호우일 도착 인원 변화({r.group}), {r['sample']}"
        add("10", tag, r.region, r.effect * 100, "%", "10 subway_effects")
        add("10", tag + " 95% 하한", r.region, r.lo * 100, "%", "10 subway_effects")
        add("10", tag + " 95% 상한", r.region, r.hi * 100, "%", "10 subway_effects")
        add("10", tag + " p값", r.region, r.p, "", "10 subway_effects")
    med = pd.read_csv(C.OUTPUT_DIR / "subway_mediation.csv", encoding="utf-8-sig")
    for _, r in med.iterrows():
        tag = f"{r.outcome}, {r['sample']}"
        add("10·11", f"호우일 {tag} 변화", r.region, r.rain_effect * 100, "%", "10 subway_mediation")
        add("11", f"발길 감소로 설명되는 몫: {tag}", r.region, r.explained_share * 100, "%", "10 subway_mediation")
        add("11", f"설명 몫 95% 하한: {tag}", r.region, r.explained_lo * 100, "%", "10 subway_mediation")
        add("11", f"설명 몫 95% 상한: {tag}", r.region, r.explained_hi * 100, "%", "10 subway_mediation")

    # ---------------------------------------------------- 지수형 손실보전 (11)
    pt = pd.read_csv(C.OUTPUT_DIR / "payout_table.csv", encoding="utf-8-sig")
    pb = pd.read_csv(C.OUTPUT_DIR / "payout_budget.csv", encoding="utf-8-sig")
    for _, r in pb.iterrows():
        add("19", "지급 대상 업종 수", r.region, r.n_paid, "개", "11 payout_budget")
        add("19", "분석 업종 수(의료 제외)", r.region, r.n_industries, "개", "11 payout_budget")
        add("19", "1회 발동 지급액", r.region, r.payout_per_event / 1e8, "억 원", "11 payout_budget")
        add("19", "6개월 지급 예산", r.region, r.payout_6m / 1e8, "억 원", "11 payout_budget")
        add("19", "예산 / 6개월 순손실", r.region, r.coverage_of_total_loss * 100, "%", "11 payout_budget")
        add("19", "지급률 중앙값", r.region, r.median_payout_rate * 100, "%", "11 payout_budget")
    paid = pt[pt.paid == 1]
    for (region, grp), g in paid.groupby(["region", "ind_group"]):
        add("21", f"지급률 중앙값({grp})", region, g.payout_rate.median() * 100, "%", "11 payout_table")
    pbt = pd.read_csv(C.OUTPUT_DIR / "payout_backtest.csv", encoding="utf-8-sig")
    for region, g in pbt.groupby("region"):
        add("20", "이벤트별 지급액-실손 상관", region, np.corrcoef(g.payout, g.actual_loss)[0, 1], "", "11 payout_backtest")
        add("20", "총지급 / 총실손", region, g.payout.sum() / g.actual_loss.sum() * 100, "%", "11 payout_backtest")
        add("20", "검증 이벤트 수", region, len(g), "건", "11 payout_backtest")

    out = pd.DataFrame(rows)
    out["value"] = out.value.astype(float).round(3)
    out.to_csv(C.OUTPUT_DIR / "report_numbers.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
