"""강건성 점검
[1] 지역 단위 순손실 검정: 업종 합계 매출로 직접 회귀 → 총/순손실 신뢰구간, 춘천-강남 차이 검정
[2] 마스킹(5건 미만 삭제) 편향 방향과 크기
    (a) 업종-일 행 전체 누락(observed=0)이 호우일에 늘어나는가 → 누락일 제외 시 손실 과소추정
    (b) 업종-일 내부 셀 삭제가 호우일에 늘어나는가 → 관측 합계가 더 작아져 손실 과대추정
출력: outputs/robust_region_net.csv, robust_group_net.csv, robust_masking.csv
"""
import importlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C

M = importlib.import_module("04_model_sensitivity")
NET_LAGS = ["heavy_rain", "heavy_rain_lag1", "heavy_rain_lag2"]


def gross_net(params: pd.Series, cov: pd.DataFrame, names: list[str]) -> dict:
    b = params[names].values
    V = cov.loc[names, names].values
    g = np.exp(b)
    out = {"gross": g[0] - 1, "gross_se": g[0] * np.sqrt(V[0, 0]),
           "net": np.sum(g - 1), "net_se": float(np.sqrt(g @ V @ g)),
           "after2d": np.sum(g[1:] - 1), "after2d_se": float(np.sqrt(g[1:] @ V[1:, 1:] @ g[1:]))}
    for k in ["gross", "net", "after2d"]:
        out[f"{k}_lo"] = out[k] - 1.96 * out[f"{k}_se"]
        out[f"{k}_hi"] = out[k] + 1.96 * out[f"{k}_se"]
    return out


def aggregate(p: pd.DataFrame, balanced: bool) -> pd.DataFrame:
    """지역 x 일 합계. balanced=True면 184일 모두 관측된 업종만 사용(구성 변화 제거)."""
    if balanced:
        full = p.groupby(["region", "MCT_RY_CD"])["observed"].transform("min") == 1
        p = p[full]
    agg = p.groupby(["region", "date"], as_index=False).agg(sales=("sales", "sum"), n_ind=("MCT_RY_CD", "nunique"))
    return agg


def region_test(p: pd.DataFrame, label: str) -> list[dict]:
    rows, fits = [], {}
    for balanced in [True, False]:
        agg = aggregate(p, balanced)
        stacked = []
        for region, g in agg.groupby("region"):
            day = p[p.region == region].drop_duplicates("date")[["date", "dow", "month", *M.WEATHER, *M.CAL_FLAGS]]
            X = M.design(day)
            X = X[M.usable(X)]
            y = np.log(g.set_index("date").sales.loc[X.index])
            m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": M.HAC_LAGS})
            r = {"sample": label, "balanced": balanced, "region": region, "n_ind": int(g.n_ind.median()),
                 **gross_net(m.params, m.cov_params(), NET_LAGS)}
            rows.append(r)
            Xs = X.add_prefix(f"{region}:")
            Xs["week"] = X.index.isocalendar().week.values
            stacked.append((Xs, y))
        # 두 지역을 쌓아 주(week) 단위 군집 표준오차로 차이 검정(지역 간 공통 충격 상관 반영)
        Xall = pd.concat([x for x, _ in stacked]).fillna(0.0)
        yall = pd.concat([y for _, y in stacked])
        weeks = Xall.pop("week").values
        ms = sm.OLS(yall.values, Xall).fit(cov_type="cluster", cov_kwds={"groups": weeks})
        regs = list(C.REGIONS)
        est = {}
        for region in regs:
            est[region] = gross_net(ms.params, ms.cov_params(), [f"{region}:{v}" for v in NET_LAGS])
        names = [f"{r}:{v}" for r in regs for v in NET_LAGS]
        b = ms.params[names].values
        V = ms.cov_params().loc[names, names].values
        # net_춘천 - net_강남 (regs 순서: 강남, 춘천)
        grad = np.concatenate([-np.exp(b[:3]), np.exp(b[3:])])
        diff = np.sum(np.exp(b[3:]) - 1) - np.sum(np.exp(b[:3]) - 1)
        se = float(np.sqrt(grad @ V @ grad))
        rows.append({"sample": label, "balanced": balanced, "region": "차이(춘천-강남), 주 군집SE",
                     "net": diff, "net_se": se, "net_lo": diff - 1.96 * se, "net_hi": diff + 1.96 * se,
                     "p_value": 2 * (1 - __import__("scipy").stats.norm.cdf(abs(diff / se)))})
    return rows


def group_net(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, grp), g in p.groupby(["region", "ind_group"]):
        agg = aggregate(g, balanced=True)
        if agg.empty or agg.n_ind.iloc[0] == 0:
            continue
        day = p[p.region == region].drop_duplicates("date")[["date", "dow", "month", *M.WEATHER, *M.CAL_FLAGS]]
        X = M.design(day)
        X = X[M.usable(X)]
        y = np.log(agg.set_index("date").sales.reindex(X.index))
        ok = y.notna()
        m = sm.OLS(y[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": M.HAC_LAGS})
        rows.append({"region": region, "ind_group": grp, "n_ind": int(agg.n_ind.iloc[0]),
                     **gross_net(m.params, m.cov_params(), NET_LAGS)})
    return pd.DataFrame(rows)


# ================================================================ 마스킹
def masking(p_all: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(C.CARD2_PATH, sep="\t", encoding=C.CARD_ENCODING, dtype={"TA_YMD": str})
    raw["date"] = pd.to_datetime(raw.TA_YMD, format="%Y%m%d")
    raw = raw.rename(columns={"MCT_SGG_CD": "region"})
    raw["cell"] = raw.CLN_SGG_CD + "|" + raw.SEX_CCD + "|" + raw.AGE_CCD
    # 업종별 잠재 셀 = 기간 중 한 번이라도 등장한 셀 조합
    potential = raw.groupby(["region", "MCT_RY_CD"]).cell.nunique().rename("potential_cells")
    ticket = (raw.groupby(["region", "MCT_RY_CD"]).TS_AT.sum() / raw.groupby(["region", "MCT_RY_CD"]).USE_CNT.sum()
              ).rename("avg_ticket")
    d = p_all.merge(potential.reset_index(), on=["region", "MCT_RY_CD"]).merge(ticket.reset_index(),
                                                                             on=["region", "MCT_RY_CD"])
    d["missing_cells"] = d.potential_cells - d.n_cells
    # 삭제된 셀은 1~4건 → 누락 매출 상한(4건)·하한(1건)
    for k in [1, 4]:
        d[f"hidden_{k}"] = d.missing_cells * k * d.avg_ticket
    normal = (d[["heavy_rain", "heavy_rain_lag1", "heavy_rain_lag2", "snow_cover"]].sum(axis=1) == 0)
    d["day_type"] = np.where(d.heavy_rain == 1, "호우일", np.where(normal, "평상일", "기타"))
    rows = []
    for (region, rel), g in d.groupby(["region", "reliable"]):
        for t in ["평상일", "호우일"]:
            s = g[g.day_type == t]
            obs = s[s.observed == 1]
            rows.append({"region": region, "reliable": rel, "day_type": t, "n_ind": s.MCT_RY_CD.nunique(),
                         "observed_rate": s.observed.mean(),
                         "missing_cell_share": (obs.missing_cells / obs.potential_cells).mean(),
                         "hidden_share_lo": obs.hidden_1.sum() / (obs.sales.sum() + obs.hidden_1.sum()),
                         "hidden_share_hi": obs.hidden_4.sum() / (obs.sales.sum() + obs.hidden_4.sum())})
    out = pd.DataFrame(rows)
    # 업종 고정효과 LPM: 호우일에 업종-일 누락 확률이 오르는가
    for region, g in d[d.day_type.isin(["평상일", "호우일"])].groupby("region"):
        X = pd.get_dummies(g.MCT_RY_CD, drop_first=True, dtype=float)
        X["heavy_rain"] = g.heavy_rain.values
        X = sm.add_constant(X)
        m = sm.OLS(1 - g.observed.values, X).fit(cov_type="cluster", cov_kwds={"groups": g.date.values})
        print(f"  {region}: 호우일 업종-일 누락확률 변화 {m.params['heavy_rain'] * 100:+.2f}%p "
              f"(p={m.pvalues['heavy_rain']:.3f})")
    return out


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    p = M.load_panel()
    core = p[p.scope == "core"]
    print("[1] 지역 단위 총/순손실 (소상공인 core, 의료 제외)")
    rt = pd.DataFrame(region_test(core, "core"))
    rt.to_csv(C.OUTPUT_DIR / "robust_region_net.csv", index=False, encoding="utf-8-sig")
    cols = ["balanced", "region", "n_ind", "gross", "gross_lo", "gross_hi", "after2d", "after2d_lo", "after2d_hi",
            "net", "net_lo", "net_hi", "p_value"]
    print(rt[[c for c in cols if c in rt]].round(3).to_string(index=False))

    print("\n[1b] 업종군별 (184일 완전관측 업종만)")
    gn = group_net(core)
    gn.to_csv(C.OUTPUT_DIR / "robust_group_net.csv", index=False, encoding="utf-8-sig")
    print(gn[["region", "ind_group", "n_ind", "gross", "gross_lo", "gross_hi", "after2d", "after2d_lo",
              "after2d_hi", "net", "net_lo", "net_hi"]].round(3).to_string(index=False))

    print("\n[2] 마스킹 편향 (core 전 업종, reliable 여부별)")
    p_all = pd.read_csv(C.PROCESSED_DIR / "panel_daily.csv", parse_dates=["date"], encoding="utf-8-sig")
    p_all = p_all.drop(columns=["ind_group", "scope"]).merge(C.industry_table(), on="MCT_RY_CD")
    p_all = p_all[p_all.scope == "core"]
    mk = masking(p_all)
    mk.to_csv(C.OUTPUT_DIR / "robust_masking.csv", index=False, encoding="utf-8-sig")
    print(mk.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
