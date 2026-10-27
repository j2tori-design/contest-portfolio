"""3단계: 예보 기반 소상공인 우선지원 알고리즘 + 백테스트

입력: 지역, 예보(강수 mm, 신적설 cm, 최고기온 ℃) + 04 단계 업종별 축소추정 민감도
절차:
  1) 예상 손실률: 강수 30mm+ → 순손실률(당일+2일), 10~30mm → 보통비 계수, 적설·폭염 계수 가산
  2) 예상 순손실액 = 평상시 일매출 x 예상 손실률
  3) 우선순위 점수 = 0.4·피해규모(로그 정규화) + 0.4·피해강도(손실률) + 0.2·회복불가도(1-회복률)
  4) 지원 등급: 손실이 통계적으로 확실(단측 95%)한 업종 중 점수 상위 25% → 1순위, 나머지 확실 업종 → 2순위,
     그 외 → 모니터링. 의료(필수서비스)는 소상공인 지원 대상에서 제외.
  5) 업종군 x 손실유형별 맞춤 지원 수단 매칭
검증:
  - 백테스트: 04 단계 leave-one-event-out 예측(학습에서 뺀 호우 이벤트)으로 '예측 손실률 순위'가 실제 손실 순위를 맞히는지
    (Spearman, precision@10 vs 무작위 기대값)
  - 가중치 민감도: 무작위 가중치 500회에서 상위 10개 업종이 얼마나 유지되는지
출력: outputs/priority_scenarios.csv, priority_backtest.csv, priority_weight_sensitivity.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C

WEIGHTS = {"size": 0.4, "rate": 0.4, "persist": 0.2}
MIN_LOSS = 0.03  # 3% 미만 손실은 지원 대상에서 제외
# 소상공인 손실보전 대상이 아닌 업종군: 공공 교통·연료, 구매를 미룰 뿐인 내구재
NON_SUPPORT_GROUPS = {"교통·연료", "내구재"}

# reliable=False: 추정 근거 이벤트가 적어 참고용으로만 제시
SCENARIOS = [
    {"name": "춘천 호우 60mm", "region": "강원 춘천시", "rain": 60, "snow": 0, "tmax": 27, "reliable": True},
    {"name": "강남 호우 60mm", "region": "서울 강남구", "rain": 60, "snow": 0, "tmax": 27, "reliable": True},
    {"name": "강남 대설 5cm(참고: 2회 이벤트)", "region": "서울 강남구", "rain": 0, "snow": 5, "tmax": 0,
     "reliable": False},
    {"name": "춘천 폭염 35℃", "region": "강원 춘천시", "rain": 0, "snow": 0, "tmax": 35, "reliable": True},
]

# (업종군, 손실유형) → 지원 수단. 손실유형 미지정 키는 업종군 기본값
SUPPORT = {
    ("여가·숙박", None): "우천 취소분 재방문 쿠폰(지역화폐 페이백)·예약 취소 수수료 보전",
    ("외식", "지속감소형(이후에도 감소)"): "호우 후 3일 '다시 골목으로' 할인 캠페인·배달 수수료 한시 지원",
    ("외식", None): "호우 당일 배달 수수료 한시 지원",
    ("생활소매", "지속감소형(이후에도 감소)"): "호우 후 재방문 페이백·소상공인 경영안정자금 우선 배정",
    ("생활소매", None): "온라인 판로(라이브커머스) 연계",
    ("식료품", "이연형(대부분 회복)"): "지원 불필요(다음날 회복) → 사전 재고·발주 조정 알림만",
    ("식료품", None): "사전 재고·발주 조정 알림, 신선식품 폐기 손실 보전",
    ("교통·연료", None): "지원 우선순위 낮음(공공 교통 대책과 연계)",
    ("생활서비스", None): "예약 변경 무료화 홍보·소액 손실 보전",
    ("내구재", None): "지원 불필요 수준(구매 연기형) → 모니터링",
    ("교육", None): "기상 악화일 온라인 수업 전환 지원",
    ("주점·유흥", None): "호우 후 재방문 페이백",
}


def support_for(group: str, loss_type: str) -> str:
    return SUPPORT.get((group, loss_type)) or SUPPORT.get((group, None)) or "모니터링"


def load_sensitivity() -> pd.DataFrame:
    s = pd.read_csv(C.OUTPUT_DIR / "sensitivity_series.csv", encoding="utf-8-sig")
    cov = pd.read_csv(C.PROCESSED_DIR / "industry_coverage.csv", encoding="utf-8-sig")
    s = s.merge(cov[["region", "MCT_RY_CD", "visitor_share"]], on=["region", "MCT_RY_CD"], how="left")
    return s[s.ind_group != "의료"].copy()


def expected_loss(sub: pd.DataFrame, rain: float, snow: float, tmax: float) -> pd.DataFrame:
    d = sub.copy()
    rate = pd.Series(0.0, index=d.index)
    se2 = pd.Series(0.0, index=d.index)
    persist = pd.Series(np.nan, index=d.index)
    if rain >= C.HEAVY_RAIN_MM:
        rate += d.net_eb
        se2 += d.net_eb_se ** 2
        persist = (1 - d.recovery_share).clip(0, 1.5)
    elif rain >= 10:
        rate += d.rate_rain_mid_eb
        se2 += d.rate_rain_mid_eb_se ** 2
    if snow >= C.SNOW_MIN_CM:
        rate += d.rate_snow_cover_eb.fillna(0)
        se2 += d.rate_snow_cover_eb_se.fillna(0) ** 2
    if tmax >= C.HEAT_TMAX:
        rate += d.rate_heat33_eb.fillna(0)
        se2 += d.rate_heat33_eb_se.fillna(0) ** 2
    d["exp_rate"] = rate
    d["exp_rate_se"] = np.sqrt(se2)
    d["exp_loss_won"] = -rate.clip(upper=0) * d.baseline_daily
    d["persist"] = persist.fillna(0.5)  # 회복 정보 없으면 중립
    return d


def score(d: pd.DataFrame, w: dict) -> pd.Series:
    loss = np.log1p(d.exp_loss_won)
    s_size = loss / loss.max() if loss.max() > 0 else loss * 0
    s_rate = (-d.exp_rate).clip(lower=0)
    s_rate = s_rate / s_rate.max() if s_rate.max() > 0 else s_rate
    s_persist = d.persist / 1.5
    return w["size"] * s_size + w["rate"] * s_rate + w["persist"] * s_persist


def rank_scenario(s: pd.DataFrame, sc: dict, w: dict = WEIGHTS) -> pd.DataFrame:
    d = expected_loss(s[s.region == sc["region"]], sc["rain"], sc["snow"], sc["tmax"])
    d["score"] = score(d, w)
    heavy = sc["rain"] >= C.HEAVY_RAIN_MM
    if not heavy:  # 손실유형(회복 여부)은 호우 모형에서만 추정됨
        d["loss_type"] = "해당없음"
    z = d.exp_rate / d.exp_rate_se.replace(0, np.nan)
    d["certain"] = ((z <= -1.645) & (d.exp_rate <= -MIN_LOSS)).astype(int)
    d["eligible"] = (~d.ind_group.isin(NON_SUPPORT_GROUPS) & d.loss_type.ne("이연형(대부분 회복)")).astype(int)
    ok = (d.certain == 1) & (d.eligible == 1)
    cut = d.loc[ok, "score"].quantile(0.75) if ok.any() else np.inf
    d["tier"] = np.select([ok & (d.score >= cut), ok], ["1순위 지원", "2순위 지원"], "모니터링")
    d["support"] = [support_for(g, t) if tier != "모니터링" else "모니터링"
                    for g, t, tier in zip(d.ind_group, d.loss_type, d.tier)]
    d["scenario"] = sc["name"]
    d["scenario_reliable"] = sc["reliable"]
    d["tier_order"] = d.tier.map({"1순위 지원": 0, "2순위 지원": 1, "모니터링": 2})
    d = d.sort_values(["tier_order", "score"], ascending=[True, False])
    d["rank"] = np.arange(1, len(d) + 1)
    return d


# ================================================================ 검증
def backtest() -> pd.DataFrame:
    cv = pd.read_csv(C.OUTPUT_DIR / "cv_predictions.csv", parse_dates=["date"], encoding="utf-8-sig")
    cv = cv[(cv.is_rain_day == 1) & (cv.ind_group != "의료")]
    ev = cv.groupby(["region", "event", "MCT_RY_CD"])[["actual", "pred_A_calendar", "pred_C_weather_eb"]].sum()
    ev["pred_rate"] = ev.pred_C_weather_eb / ev.pred_A_calendar - 1   # 학습에서 뺀 이벤트에 대한 예측
    ev["real_rate"] = ev.actual / ev.pred_A_calendar - 1              # 달력 기준선 대비 실제 손실
    rows = []
    for (region, e), g in ev.groupby(level=["region", "event"]):
        n = len(g)
        top_pred = set(g.pred_rate.nsmallest(10).index)
        top_real = set(g.real_rate.nsmallest(10).index)
        rows.append({"region": region, "event": e, "n_ind": n,
                     "spearman": spearmanr(g.pred_rate, g.real_rate).statistic,
                     "precision_at_10": len(top_pred & top_real) / 10, "random_expect": 10 / n})
    return pd.DataFrame(rows)


def weight_sensitivity(s: pd.DataFrame, n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sc in SCENARIOS[:2]:
        base = set(rank_scenario(s, sc).head(10).MCT_RY_CD)
        overlaps = []
        for _ in range(n):
            a = rng.dirichlet([1, 1, 1])
            r = rank_scenario(s, sc, {"size": a[0], "rate": a[1], "persist": a[2]})
            overlaps.append(len(base & set(r.head(10).MCT_RY_CD)) / 10)
        rows.append({"scenario": sc["name"], "mean_top10_overlap": np.mean(overlaps),
                     "p10_top10_overlap": np.percentile(overlaps, 10)})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 45)
    s = load_sensitivity()

    out = pd.concat([rank_scenario(s, sc) for sc in SCENARIOS], ignore_index=True)
    keep = ["scenario", "scenario_reliable", "rank", "MCT_RY_CD", "ind_group", "exp_rate", "exp_loss_won",
            "loss_type", "visitor_share", "score", "tier", "support"]
    out[keep].to_csv(C.OUTPUT_DIR / "priority_scenarios.csv", index=False, encoding="utf-8-sig")
    for sc in SCENARIOS:
        t = out[out.scenario == sc["name"]][keep].head(10).copy()
        t["exp_loss_won"] = (t.exp_loss_won / 1e6).round(1)
        print(f"\n=== 시나리오: {sc['name']} | 대상 업종 {len(out[out.scenario == sc['name']])}개 | 등급 "
              f"{out[out.scenario == sc['name']].tier.value_counts().to_dict()}")
        print(t.drop(columns="scenario").rename(columns={"exp_loss_won": "예상순손실(백만원/일)"})
              .round(3).to_string(index=False))

    print("\n=== 백테스트(학습에서 뺀 호우 이벤트, 의료 제외)")
    bt = backtest()
    bt.to_csv(C.OUTPUT_DIR / "priority_backtest.csv", index=False, encoding="utf-8-sig")
    print(bt.groupby("region")[["spearman", "precision_at_10", "random_expect"]].mean().round(3))
    print("Spearman>0 인 이벤트 비율:", bt.groupby("region").spearman.apply(lambda x: (x > 0).mean()).round(2).to_dict())

    print("\n=== 가중치 민감도(무작위 가중치 500회, 상위 10 유지율)")
    ws = weight_sensitivity(s)
    ws.to_csv(C.OUTPUT_DIR / "priority_weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(ws.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
