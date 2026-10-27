"""정책 제안: 호우 지수형 손실보전 (손해사정 없이 '비가 30mm 넘으면 자동 지급')
→ outputs/payout_table.csv, payout_budget.csv, payout_backtest.csv, figure F14

설계
  - 발동 조건(지수): 기상청 ASOS 일강수량 30mm 이상 (지점 108 서울 / 101 춘천)
  - 지급 대상: 손실이 통계적으로 확실한(단측 95%, 3% 이상) 소상공인 업종 중
    이연형(다음날 회복)·영향미미·교통·연료·내구재·의료 제외 → 06 단계 우선지원 알고리즘과 같은 기준
  - 가맹점 지급액 = 평상시 일매출 × 업종 지급률,  업종 지급률 = 3일 순손실률(축소추정) × 보전율 50%
    (보전율 50%: 손실의 절반은 자기부담으로 남겨 '비 오는 날 일부러 쉬는' 도덕적 해이를 막음)
  - 풍수해보험은 건물·시설·재고(재산) 피해만 보상하므로, 이 제도는 '매출 손실'이라는 빈칸을 메운다

검증(기초위험, basis risk): 04 단계 leave-one-event-out 예측으로
  '지수가 미리 정한 지급액'과 '실제로 발생한 손실'이 이벤트마다 얼마나 맞는지 비교
"""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as CFG  # noqa: E402

P = importlib.import_module("06_priority_algorithm")
F = importlib.import_module("07_figures")

COVER = 0.5
EXCLUDE_TYPES = {"이연형(대부분 회복)", "영향미미"}


def payout_table(s: pd.DataFrame) -> pd.DataFrame:
    out = []
    for region in CFG.REGIONS:
        d = P.rank_scenario(s, {"name": "지수 발동", "region": region, "rain": CFG.HEAVY_RAIN_MM, "snow": 0,
                                "tmax": 0, "reliable": True})
        d["paid"] = ((d.tier != "모니터링") & ~d.loss_type.isin(EXCLUDE_TYPES)).astype(int)
        d["payout_rate"] = np.where(d.paid == 1, -d.exp_rate * COVER, 0.0)
        d["payout_won"] = d.payout_rate * d.baseline_daily
        d["region"] = region
        out.append(d)
    t = pd.concat(out, ignore_index=True)
    return t[["region", "MCT_RY_CD", "ind_group", "loss_type", "exp_rate", "baseline_daily", "paid", "payout_rate",
              "payout_won", "damage_net_won", "n_heavy_days"]]


def budget(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, g in t.groupby("region"):
        days = g.n_heavy_days.max()
        p = g[g.paid == 1]
        rows.append({"region": region, "heavy_days": days, "n_industries": len(g), "n_paid": len(p),
                     "payout_per_event": p.payout_won.sum(), "payout_6m": p.payout_won.sum() * days,
                     "net_loss_6m": g.damage_net_won.sum(), "net_loss_paid_ind_6m": p.damage_net_won.sum(),
                     "median_payout_rate": p.payout_rate.median()})
    b = pd.DataFrame(rows)
    b["coverage_of_total_loss"] = b.payout_6m / b.net_loss_6m
    return b


def backtest(t: pd.DataFrame) -> pd.DataFrame:
    """학습에서 뺀 호우 이벤트마다: 지급액(모형 예측 손실 × 보전율) vs 실제 손실 × 보전율"""
    cv = pd.read_csv(CFG.OUTPUT_DIR / "cv_predictions.csv", encoding="utf-8-sig")
    paid = t[t.paid == 1][["region", "MCT_RY_CD"]]
    cv = cv.merge(paid, on=["region", "MCT_RY_CD"])
    ev = cv.groupby(["region", "event", "event_start"], as_index=False).agg(
        base=("pred_A_calendar", "sum"), pred=("pred_C_weather_eb", "sum"), actual=("actual", "sum"))
    ev["payout"] = (ev.base - ev.pred).clip(lower=0) * COVER
    ev["actual_loss"] = (ev.base - ev.actual) * COVER
    return ev


def fig_backtest(ev: pd.DataFrame):
    """F14: 이벤트별 '자동 지급액' vs '실제 손실'(둘 다 보전율 50% 적용)"""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    hi = max(ev.payout.max(), ev.actual_loss.max()) / 1e8 * 1.1
    lo = min(0, ev.actual_loss.min() / 1e8 * 1.1)
    ax.plot([lo, hi], [lo, hi], color=F.MUTED, lw=1, ls="--")
    for region, g in ev.groupby("region"):
        r = np.corrcoef(g.payout, g.actual_loss)[0, 1]
        ax.scatter(g.payout / 1e8, g.actual_loss / 1e8, s=70, color=F.REGION_COLOR[region], alpha=0.85,
                   label=f"{F.SHORT[region]} ({len(g)}회, 상관 {r:.2f})", edgecolor="white", lw=0.8)
    ax.axhline(0, color=F.INK2, lw=0.8)
    ax.set_xlim(0, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("지수로 자동 지급되는 금액 (억 원, 이벤트당)")
    ax.set_ylabel("실제 손실 × 보전율 50% (억 원)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("손해사정 없이 자동 지급해도, 지급액은 실제 손실을 따라간다")
    ax.text(0, -0.15, "각 점 = 학습에서 뺀 호우 이벤트 1회(당일+2일), 지급 대상 업종 합계 · 점선 위 = 덜 지급, 아래 = 더 지급",
            transform=ax.transAxes, color=F.MUTED, fontsize=10)
    F.save(fig, "F14_index_backtest")


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    s = P.load_sensitivity()
    t = payout_table(s)
    t.to_csv(CFG.OUTPUT_DIR / "payout_table.csv", index=False, encoding="utf-8-sig")
    b = budget(t)
    b.to_csv(CFG.OUTPUT_DIR / "payout_budget.csv", index=False, encoding="utf-8-sig")
    ev = backtest(t)
    ev.to_csv(CFG.OUTPUT_DIR / "payout_backtest.csv", index=False, encoding="utf-8-sig")

    print("[지급 업종]")
    show = t[t.paid == 1].sort_values(["region", "payout_rate"], ascending=[True, False])
    print(show.assign(payout_rate=show.payout_rate * 100, payout_won=show.payout_won / 1e6)
          [["region", "MCT_RY_CD", "ind_group", "loss_type", "payout_rate", "payout_won"]].round(2).to_string(index=False))
    print("\n[업종군별 지급]")
    g = t[t.paid == 1].groupby(["region", "ind_group"]).agg(n=("MCT_RY_CD", "size"), rate=("payout_rate", "median"),
                                                           won=("payout_won", "sum"))
    print(g.assign(rate=g.rate * 100, won=g.won / 1e6).round(2))
    print("\n[예산]")
    print(b.assign(payout_per_event=b.payout_per_event / 1e8, payout_6m=b.payout_6m / 1e8,
                   net_loss_6m=b.net_loss_6m / 1e8, net_loss_paid_ind_6m=b.net_loss_paid_ind_6m / 1e8).round(3)
          .to_string(index=False))
    print("\n[기초위험 백테스트]")
    for region, gg in ev.groupby("region"):
        print(region, "상관", round(np.corrcoef(gg.payout, gg.actual_loss)[0, 1], 3),
              "총지급/총실손", round(gg.payout.sum() / gg.actual_loss.sum(), 3),
              "실손<0 이벤트", int((gg.actual_loss < 0).sum()), "/", len(gg))
    fig_backtest(ev)


if __name__ == "__main__":
    main()
