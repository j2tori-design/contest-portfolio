"""2단계: 업종별 기후 민감도(경험적 베이즈 축소추정) + 순손실률 + 이벤트 단위 검증

모형 (지역 x 업종 시계열마다):
    log(매출_t) = 기상_t·β + 호우_{t-1}·γ1 + 호우_{t-2}·γ2 + 달력_t·δ + ε_t
    기상: 호우(30mm+), 보통비(10~30mm), 강설, 적설지속, 폭염(33℃+)
    달력: 요일, 월, 공휴일, 추석연휴, 연휴 전날, 특수일(12/24·31), 급여일
총손실률 = exp(β_호우) - 1                      (호우 당일 매출 변화율)
순손실률 = Σ_{k=0..2} (exp(b_k) - 1)            (당일 손실 + 이후 2일 반등, 평상시 하루 매출 대비)
회복률   = 1 - 순손실/총손실                     (이연소비로 되찾은 비율)
축소추정: 같은 지역·업종군 평균으로 당김(DerSimonian-Laird τ²). 표본이 작은 업종의 튀는 값을 억제.

검증(leave-one-event-out): 호우 이벤트(연속 호우일 묶음) 하나와 그 뒤 2일을 학습에서 빼고 예측.
    A 달력만 / B 기상 포함(개별 OLS) / C 기상 포함 + 축소추정  → 이벤트 구간 WAPE 비교

출력: outputs/sensitivity_series.csv, sensitivity_by_group.csv, cv_predictions.csv, cv_summary.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C

warnings.filterwarnings("ignore", category=RuntimeWarning)

HAZARDS = ["heavy_rain", "rain_mid", "rain_light", "snow", "snow_cover", "heat33"]
LAGS = ["heavy_rain_lag1", "heavy_rain_lag2"]
WEATHER = HAZARDS + LAGS
CAL_FLAGS = ["is_holiday", "chuseok_block", "pre_offday", "special_day", "payday"]
RECOVERY_DAYS = 2
HAC_LAGS = 7


# ================================================================ 데이터·설계행렬
def load_panel() -> pd.DataFrame:
    p = pd.read_csv(C.PROCESSED_DIR / "panel_daily.csv", parse_dates=["date"], encoding="utf-8-sig")
    # 업종 범위는 패널 생성 시점이 아니라 현재 config 기준으로 다시 붙인다
    p = p.drop(columns=["ind_group", "scope"]).merge(C.industry_table(), on="MCT_RY_CD", how="left")
    p = p[p.scope.isin(["core", "essential"]) & (p.reliable == 1)].copy()
    p["rain_light"] = ((p.rain >= 0.1) & (p.rain < 10)).astype(int)
    p["rain_mid"] = ((p.rain >= 10) & (p.rain < C.HEAVY_RAIN_MM)).astype(int)
    return p


def design(day: pd.DataFrame) -> pd.DataFrame:
    """지역별 일자 단위 설계행렬(업종 공통). index = date."""
    day = day.set_index("date").sort_index()
    X = day[WEATHER + CAL_FLAGS].astype(float)
    X = X.join(pd.get_dummies(day.dow, prefix="dow", drop_first=True, dtype=float))
    X = X.join(pd.get_dummies(day.month, prefix="m", drop_first=True, dtype=float))
    X.insert(0, "const", 1.0)
    return X


def usable(X: pd.DataFrame) -> list[str]:
    """변동이 없는 열 제외(예: 강남 폭염·적설 외 결측 구간)."""
    return [c for c in X.columns if c == "const" or X[c].std() > 0]


# ================================================================ 축소추정
def eb_shrink(est: pd.Series, se: pd.Series, groups: pd.Series, min_group: int = 3) -> pd.DataFrame:
    """정규-정규 경험적 베이즈. 그룹이 작으면 상위(전체) 평균으로 당긴다."""
    out = pd.DataFrame({"est": est, "se": se, "grp": groups})
    ok = out.est.notna() & out.se.gt(0)

    def dl(e, s):
        w = 1 / s ** 2
        mu_fe = np.sum(w * e) / np.sum(w)
        q = np.sum(w * (e - mu_fe) ** 2)
        tau2 = max(0.0, (q - (len(e) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
        wr = 1 / (s ** 2 + tau2)
        return np.sum(wr * e) / np.sum(wr), tau2, 1 / np.sum(wr)

    out[["prior_mu", "tau2", "mu_var"]] = dl(out.est[ok].values, out.se[ok].values)
    for g, idx in out[ok].groupby("grp").groups.items():
        if len(idx) >= min_group:
            out.loc[idx, ["prior_mu", "tau2", "mu_var"]] = dl(out.est[idx].values, out.se[idx].values)
    w = np.where(out.tau2 > 0, out.tau2 / (out.tau2 + out.se ** 2), 0.0)
    out["shrink_w"] = w
    out["eb"] = w * out.est + (1 - w) * out.prior_mu
    # 사후분산 = 개별 정보 몫(w·se²) + 그룹 평균 추정 불확실성 몫((1-w)²·Var(μ))
    out["eb_se"] = np.sqrt(w * out.se ** 2 + (1 - w) ** 2 * out.mu_var)
    return out[["eb", "eb_se", "prior_mu", "tau2", "shrink_w"]]


# ================================================================ 1) 전체 표본 추정
def fit_series(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, pr in p.groupby("region"):
        X_all = design(pr.drop_duplicates("date")[["date", "dow", "month", *WEATHER, *CAL_FLAGS]])
        for (ry, grp, scope), g in pr.groupby(["MCT_RY_CD", "ind_group", "scope"]):
            g = g[(g.observed == 1) & (g.sales > 0)].set_index("date")
            X = X_all.loc[g.index]
            cols = usable(X)
            m = sm.OLS(np.log(g.sales), X[cols]).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            r = {"region": region, "MCT_RY_CD": ry, "ind_group": grp, "n_days": len(g), "r2": m.rsquared}
            for v in WEATHER:
                r[f"b_{v}"] = m.params.get(v, np.nan)
                r[f"se_{v}"] = m.bse.get(v, np.nan)
            # 총손실률·순손실률과 델타법 표준오차
            k = ["heavy_rain", *LAGS]
            b = m.params[k].values
            V = m.cov_params().loc[k, k].values
            r["gross"] = np.exp(b[0]) - 1
            r["gross_se"] = np.exp(b[0]) * np.sqrt(V[0, 0])
            r["net"] = np.sum(np.exp(b) - 1)
            grad = np.exp(b)
            r["net_se"] = float(np.sqrt(grad @ V @ grad))
            for h in ["snow_cover", "snow", "heat33", "rain_mid"]:
                if h in m.params:
                    r[f"rate_{h}"] = np.exp(m.params[h]) - 1
                    r[f"rate_{h}_se"] = np.exp(m.params[h]) * m.bse[h]
            normal = (X[["heavy_rain", *LAGS, "snow_cover", "snow"]].sum(axis=1) == 0)
            r["baseline_daily"] = g.sales[normal].mean()
            r["n_heavy_days"] = int(X.heavy_rain.sum())
            r["n_snow_cover_days"] = int(X.snow_cover.sum()) if "snow_cover" in X else 0
            rows.append(r)
    return pd.DataFrame(rows)


def add_shrinkage(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    grp = s.region + "|" + s.ind_group
    targets = {"gross": "gross_se", "net": "net_se", "rate_snow_cover": "rate_snow_cover_se",
               "rate_heat33": "rate_heat33_se", "rate_rain_mid": "rate_rain_mid_se"}
    for est, se in targets.items():
        for region, idx in s.groupby("region").groups.items():
            sub = s.loc[idx]
            if sub[est].notna().sum() < 3:
                continue
            sh = eb_shrink(sub[est], sub[se], grp[idx])
            s.loc[idx, f"{est}_eb"] = sh.eb
            s.loc[idx, f"{est}_eb_se"] = sh.eb_se
            s.loc[idx, f"{est}_shrink_w"] = sh.shrink_w
    # 회복률: 총손실이 음(-)일 때만 의미
    s["recovery_share"] = np.where(s.gross_eb < 0, 1 - s.net_eb / s.gross_eb, np.nan)
    # 분석기간 피해액(원): 평상시 일매출 x 손실률 x 호우일수
    s["damage_gross_won"] = -s.gross_eb.clip(upper=0) * s.baseline_daily * s.n_heavy_days
    s["damage_net_won"] = -s.net_eb.clip(upper=0) * s.baseline_daily * s.n_heavy_days
    s["damage_snow_won"] = -s.rate_snow_cover_eb.clip(upper=0) * s.baseline_daily * s.n_snow_cover_days
    # 취약 등급(호우 순손실 기준, 단측 95%)
    z = s.net_eb / s.net_eb_se.replace(0, np.nan)
    s["grade"] = np.select([(s.net_eb <= -0.10) & (z <= -1.645), (s.net_eb <= -0.05) | (z <= -1.645)],
                           ["상", "중"], "하")
    s["loss_type"] = np.select(
        [s.gross_eb > -0.03, s.recovery_share >= 0.5, s.recovery_share < -0.2, s.recovery_share < 0.2],
        ["영향미미", "이연형(대부분 회복)", "지속감소형(이후에도 감소)", "소멸형(회복 없음)"], "부분회복형")
    return s


# ================================================================ 2) 이벤트 단위 검증
def rain_events(day: pd.DataFrame) -> list[list[pd.Timestamp]]:
    d = day.set_index("date").sort_index().heavy_rain
    events, cur = [], []
    for t, v in d.items():
        if v == 1 and cur and (t - cur[-1]).days <= 1:
            cur.append(t)
        elif v == 1:
            if cur:
                events.append(cur)
            cur = [t]
    if cur:
        events.append(cur)
    return events


def ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None)), resid


def cross_validate(p: pd.DataFrame) -> pd.DataFrame:
    preds = []
    for region, pr in p.groupby("region"):
        day = pr.drop_duplicates("date")[["date", "dow", "month", *WEATHER, *CAL_FLAGS]]
        X_all = design(day)
        events = rain_events(day)
        series = {ry: g[(g.observed == 1) & (g.sales > 0)].set_index("date")
                  for ry, g in pr.groupby("MCT_RY_CD")}
        groups = pr.drop_duplicates("MCT_RY_CD").set_index("MCT_RY_CD").ind_group
        print(f"  {region}: 호우 이벤트 {len(events)}개, 업종 {len(series)}개")
        for ei, ev in enumerate(events):
            last = ev[-1]
            test_days = ev + [last + pd.Timedelta(days=k) for k in range(1, RECOVERY_DAYS + 1)]
            test_days = [t for t in test_days if t in X_all.index]
            train_mask = ~X_all.index.isin(test_days)
            cols = usable(X_all[train_mask])
            wcols = [c for c in cols if c in WEATHER]
            ccols = [c for c in cols if c not in WEATHER]
            fold = {}
            for ry, g in series.items():
                tr = g.index[g.index.isin(X_all.index[train_mask])]
                te = g.index[g.index.isin(test_days)]
                if len(te) == 0:
                    continue
                y = np.log(g.sales.loc[tr].values)
                Xtr, Xte = X_all.loc[tr], X_all.loc[te]
                bA, _, rA = ols(Xtr[ccols].values, y)
                bB, seB, rB = ols(Xtr[cols].values, y)
                fold[ry] = dict(y=y, Xtr=Xtr, Xte=Xte, te=te, bA=bA, rA=rA, bB=bB, seB=seB, rB=rB)
            # C: 기상 계수만 업종군 축소 → 기상 기여분을 뺀 뒤 달력 계수 재추정
            shrunk = {}
            for j, w in enumerate(wcols):
                k = cols.index(w)
                est = pd.Series({ry: f["bB"][k] for ry, f in fold.items()})
                se = pd.Series({ry: f["seB"][k] for ry, f in fold.items()})
                shrunk[w] = eb_shrink(est, se, groups.reindex(est.index)).eb
            for ry, f in fold.items():
                bw = np.array([shrunk[w][ry] for w in wcols])
                yadj = f["y"] - f["Xtr"][wcols].values @ bw
                bC, _, rC = ols(f["Xtr"][ccols].values, yadj)
                actual = series[ry].sales.loc[f["te"]].values
                pA = np.exp(f["Xte"][ccols].values @ f["bA"]) * np.mean(np.exp(f["rA"]))
                pB = np.exp(f["Xte"][cols].values @ f["bB"]) * np.mean(np.exp(f["rB"]))
                pC = (np.exp(f["Xte"][ccols].values @ bC + f["Xte"][wcols].values @ bw)
                      * np.mean(np.exp(rC)))
                for t, a, ya, yb, yc in zip(f["te"], actual, pA, pB, pC):
                    preds.append({"region": region, "event": ei, "event_start": ev[0], "date": t,
                                  "is_rain_day": int(t in ev), "MCT_RY_CD": ry, "ind_group": groups[ry],
                                  "actual": a, "pred_A_calendar": ya, "pred_B_weather": yb,
                                  "pred_C_weather_eb": yc})
    return pd.DataFrame(preds)


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    models = ["pred_A_calendar", "pred_B_weather", "pred_C_weather_eb"]
    rows = []
    for (region, rain_day), g in cv.groupby(["region", "is_rain_day"]):
        r = {"region": region, "window": "호우 당일" if rain_day else "호우 후 1~2일", "n": len(g)}
        for m in models:
            r[f"WAPE_{m[5:]}"] = (g[m] - g.actual).abs().sum() / g.actual.sum()
            r[f"RMSLE_{m[5:]}"] = np.sqrt(np.mean((np.log(g[m]) - np.log(g.actual)) ** 2))
        rows.append(r)
    # 이벤트별로 어떤 모형이 이겼는지 (업종 합계 기준 절대오차)
    ev = cv.groupby(["region", "event"])[["actual", *models]].sum()
    err = ev[models].sub(ev.actual, axis=0).abs()
    wins = err.idxmin(axis=1).value_counts()
    print("\n이벤트별 최소오차 모형(지역 업종합계 기준):", wins.to_dict())
    return pd.DataFrame(rows)


def main():
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = load_panel()
    print(f"분석 패널: {p.MCT_RY_CD.nunique()}개 업종, {len(p)}행")

    print("[1] 업종별 회귀(HAC) + 축소추정")
    s = add_shrinkage(fit_series(p))
    s.to_csv(C.OUTPUT_DIR / "sensitivity_series.csv", index=False, encoding="utf-8-sig")

    by_group = s.groupby(["region", "ind_group"]).apply(lambda g: pd.Series({
        "n_ind": len(g),
        "gross_eb_w": np.average(g.gross_eb, weights=g.baseline_daily),
        "net_eb_w": np.average(g.net_eb, weights=g.baseline_daily),
        "snow_cover_eb_w": np.average(g.rate_snow_cover_eb.fillna(0), weights=g.baseline_daily),
        "damage_net_100m": g.damage_net_won.sum() / 1e8,
        "damage_snow_100m": g.damage_snow_won.sum() / 1e8,
        "n_grade_high": (g.grade == "상").sum(),
    }), include_groups=False).reset_index()
    by_group.to_csv(C.OUTPUT_DIR / "sensitivity_by_group.csv", index=False, encoding="utf-8-sig")

    print("[2] 이벤트 단위 검증(leave-one-event-out)")
    cv = cross_validate(p)
    cv.to_csv(C.OUTPUT_DIR / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    summ = cv_summary(cv)
    summ.to_csv(C.OUTPUT_DIR / "cv_summary.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ 요약 출력
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    show = ["MCT_RY_CD", "ind_group", "gross", "gross_eb", "net_eb", "recovery_share", "grade", "loss_type",
            "damage_net_won"]
    for region, g in s.groupby("region"):
        print(f"\n=== {region}: 호우 순손실 상위 10 (축소추정) | 호우일 {g.n_heavy_days.iloc[0]}일")
        t = g.sort_values("net_eb")[show].head(10).copy()
        t["damage_net_won"] = (t.damage_net_won / 1e8).round(2)
        print(t.round(3).rename(columns={"damage_net_won": "순피해(억)"}).to_string(index=False))
        print("등급 분포:", g.grade.value_counts().to_dict(), "| 유형:", g.loss_type.value_counts().to_dict())
        for label, sub in [("소상공인(core)", g[g.ind_group != "의료"]), ("필수서비스(의료)", g[g.ind_group == "의료"])]:
            print(f"[{label}] 총피해 {sub.damage_gross_won.sum() / 1e8:.1f}억 → 순피해 {sub.damage_net_won.sum() / 1e8:.1f}억"
                  f" | 적설 피해 {sub.damage_snow_won.sum() / 1e8:.1f}억")
        print(f"축소 가중치 평균(1=원추정 유지): {g.net_shrink_w.mean():.2f}")
    print("\n=== 업종군별 (매출가중)")
    print(by_group.round(3).to_string(index=False))
    print("\n=== 검증 요약")
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
