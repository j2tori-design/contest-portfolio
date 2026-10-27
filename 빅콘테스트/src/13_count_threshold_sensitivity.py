"""분석 대상 업종 기준 민감도: 하루 결제 중간값 기준(MIN_MEDIAN_CNT)을 바꿔 핵심 결과를 다시 계산
→ outputs/count_threshold_sensitivity.csv
선행: 03 실행 (panel_daily.csv, industry_coverage.csv)

관측일 비율 90% 기준(MIN_COVERAGE)은 그대로 두고 중간값 기준만 바꾼다. '필터 없음'은 두 기준을 모두 끈다.
"""
import importlib
import warnings

import numpy as np
import pandas as pd

import config as CFG

M = importlib.import_module("04_model_sensitivity")
R = importlib.import_module("05_robustness_checks")
warnings.filterwarnings("ignore")

RULES = [("필터 없음", 0.0, 0), ("5건", CFG.MIN_COVERAGE, 5), ("10건", CFG.MIN_COVERAGE, 10),
         ("20건", CFG.MIN_COVERAGE, 20), ("30건(현재)", CFG.MIN_COVERAGE, 30), ("50건", CFG.MIN_COVERAGE, 50),
         ("100건", CFG.MIN_COVERAGE, 100)]
MIN_OBS_DAYS = 40  # 회귀가 돌아가려면 최소 관측일이 필요(필터 없음일 때 극소 업종 제외용)


def load_all() -> pd.DataFrame:
    p = pd.read_csv(CFG.PROCESSED_DIR / "panel_daily.csv", parse_dates=["date"], encoding="utf-8-sig")
    p = p.drop(columns=["ind_group", "scope", "reliable"]).merge(CFG.industry_table(), on="MCT_RY_CD", how="left")
    p = p[p.scope == "core"].copy()
    p["rain_light"] = ((p.rain >= 0.1) & (p.rain < 10)).astype(int)
    p["rain_mid"] = ((p.rain >= 10) & (p.rain < CFG.HEAVY_RAIN_MM)).astype(int)
    return p


def main():
    pd.set_option("display.width", 240)
    p_all = load_all()
    cov = pd.read_csv(CFG.PROCESSED_DIR / "industry_coverage.csv", encoding="utf-8-sig")[
        ["region", "MCT_RY_CD", "coverage", "median_cnt", "total_sales_100m"]]
    obs_days = p_all[p_all.observed == 1].groupby(["region", "MCT_RY_CD"]).size().rename("obs_days").reset_index()
    cov = cov.merge(obs_days, on=["region", "MCT_RY_CD"], how="inner")
    core_sales = cov.groupby("region").total_sales_100m.sum()

    rows = []
    for name, min_cov, min_cnt in RULES:
        keep = cov[(cov.coverage >= min_cov) & (cov.median_cnt >= min_cnt) & (cov.obs_days >= MIN_OBS_DAYS)]
        p = p_all.merge(keep[["region", "MCT_RY_CD"]], on=["region", "MCT_RY_CD"])
        rt = pd.DataFrame(R.region_test(p, name))
        s = M.add_shrinkage(M.fit_series(p))
        for region in CFG.REGIONS:
            k = keep[keep.region == region]
            sr = s[s.region == region]
            bal = rt[rt.balanced & (rt.region == region)].iloc[0]
            unbal = rt[~rt.balanced & (rt.region == region)].iloc[0]
            rows.append({
                "rule": name, "region": region, "n_ind": len(k),
                "sales_share": k.total_sales_100m.sum() / core_sales[region],
                "net_balanced": bal.net, "net_balanced_lo": bal.net_lo, "net_balanced_hi": bal.net_hi,
                "net_all": unbal.net, "net_all_lo": unbal.net_lo, "net_all_hi": unbal.net_hi,
                "damage_net_100m": sr.damage_net_won.sum() / 1e8,
                "gross_sd_raw": sr.gross.std(), "gross_sd_eb": sr.gross_eb.std(),
                "n_grade_high": int((sr.grade == "상").sum()),
                "n_sig_loss": int(((sr.net_eb / sr.net_eb_se) <= -1.645).sum()),
            })
        for bal_flag, tag in [(True, "diff_balanced"), (False, "diff_all")]:
            d = rt[(rt.balanced == bal_flag) & rt.region.str.startswith("차이")].iloc[0]
            rows[-1][f"{tag}"], rows[-1][f"{tag}_p"] = d.net, d.p_value
            rows[-2][f"{tag}"], rows[-2][f"{tag}_p"] = d.net, d.p_value
    out = pd.DataFrame(rows)
    out.to_csv(CFG.OUTPUT_DIR / "count_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    show = out.copy()
    for c in ["sales_share", "net_balanced", "net_balanced_lo", "net_balanced_hi", "net_all", "net_all_lo",
              "net_all_hi", "gross_sd_raw", "gross_sd_eb", "diff_balanced", "diff_all"]:
        show[c] = show[c] * 100
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
