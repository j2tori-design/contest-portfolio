"""신한카드 데이터 정제·집계 → data/processed/card_daily.csv, card_slot.csv, industry_coverage.csv

card_daily (데이터2, 법인 제외): 일자 x 지역 x 업종, 고객 거주지(현지/외지)·성·연령 세분 매출 포함
card_slot  (데이터1, 법인 분리): 일자 x 지역 x 업종 x 시간대(6시간)

주의: 원천 데이터는 결제건수 5건 미만 셀이 삭제(마스킹)되어 있다.
  - observed=0 : 그 날 해당 업종 행이 아예 없음 → 매출 NaN (0으로 채우지 않음)
  - 세분 컬럼(sales_visitor 등)의 0은 '실제 0' 또는 '마스킹'을 구분할 수 없음
"""
import numpy as np
import pandas as pd

import config as C

OLD_AGES = {"60대", "70대", "80대", "90대"}
YOUNG_AGES = {"10대", "20대", "30대"}


def read_card(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding=C.CARD_ENCODING, dtype={"TA_YMD": str})
    df["date"] = pd.to_datetime(df.TA_YMD, format="%Y%m%d")
    df["AGE_CCD"] = df.AGE_CCD.str.replace(" ", "", regex=False)
    df = df.rename(columns={"MCT_SGG_CD": "region", "TS_AT": "sales", "USE_CNT": "cnt"})
    unknown = set(df.region) - set(C.REGIONS)
    if unknown:
        raise ValueError(f"config.REGIONS에 없는 지역: {unknown}")
    return df


def attach_industry(df: pd.DataFrame) -> pd.DataFrame:
    ind = C.industry_table()
    out = df.merge(ind, on="MCT_RY_CD", how="left")
    missing = out.loc[out.ind_group.isna(), "MCT_RY_CD"].unique()
    if len(missing):
        print(f"[경고] 업종 분류 미지정 → '미분류': {list(missing)}")
        out[["ind_group", "scope"]] = out[["ind_group", "scope"]].fillna({"ind_group": "미분류", "scope": "exclude"})
    return out


def complete_index(agg: pd.DataFrame, keys_extra: list[str]) -> pd.DataFrame:
    """지역별로 한 번이라도 등장한 업종 x 전체 일자(x 시간대) 격자를 만든 뒤 결측 표시."""
    dates = pd.date_range(C.ANALYSIS_START, C.ANALYSIS_END, freq="D")
    pairs = agg[["region", "MCT_RY_CD"]].drop_duplicates()
    grid = pairs.merge(pd.DataFrame({"date": dates}), how="cross")
    if "TIME_GB" in keys_extra:
        grid = grid.merge(pd.DataFrame({"TIME_GB": list(C.TIME_SLOTS)}), how="cross")
    keys = ["date", "region", "MCT_RY_CD", *keys_extra]
    out = grid.merge(agg, on=keys, how="left")
    out["observed"] = out.n_cells.notna().astype(int)
    out["n_cells"] = out.n_cells.fillna(0).astype(int)
    return out


def build_daily() -> pd.DataFrame:
    df = read_card(C.CARD2_PATH)
    home = df.region.map({k: v["home_sido"] for k, v in C.REGIONS.items()})
    df["cust"] = np.select([df.CLN_SGG_CD.eq("정보없음"), df.CLN_SGG_CD.eq(home)], ["unknown", "local"], "visitor")

    seg = {
        "local": df.cust.eq("local"), "visitor": df.cust.eq("visitor"), "unknown": df.cust.eq("unknown"),
        "female": df.SEX_CCD.eq("여성"), "age60p": df.AGE_CCD.isin(OLD_AGES), "age10_30": df.AGE_CCD.isin(YOUNG_AGES),
    }
    for name, mask in seg.items():
        df[f"sales_{name}"] = df.sales.where(mask, 0)
        df[f"cnt_{name}"] = df.cnt.where(mask, 0)

    val_cols = ["sales", "cnt"] + [c for c in df.columns if c.startswith(("sales_", "cnt_"))]
    agg = df.groupby(["date", "region", "MCT_RY_CD"], as_index=False).agg(
        **{c: (c, "sum") for c in val_cols}, n_cells=("sales", "size"))
    out = complete_index(agg, [])
    return attach_industry(out)


def build_slot() -> pd.DataFrame:
    df = read_card(C.CARD1_PATH)
    corp = df.SEX_CCD.eq("법인")
    df["sales_corp"] = df.sales.where(corp, 0)
    df["cnt_corp"] = df.cnt.where(corp, 0)
    df["sales_personal"] = df.sales.where(~corp, 0)
    df["cnt_personal"] = df.cnt.where(~corp, 0)
    agg = df.groupby(["date", "region", "MCT_RY_CD", "TIME_GB"], as_index=False).agg(
        sales=("sales_personal", "sum"), cnt=("cnt_personal", "sum"),
        sales_corp=("sales_corp", "sum"), cnt_corp=("cnt_corp", "sum"),
        n_cells=("sales", "size"))
    out = complete_index(agg, ["TIME_GB"])
    # 개인 결제 행이 하나도 없으면 개인 매출은 미관측
    no_personal = out.observed.eq(1) & out.cnt.eq(0)
    out.loc[no_personal, ["sales", "cnt"]] = np.nan
    return attach_industry(out)


def coverage_table(daily: pd.DataFrame) -> pd.DataFrame:
    cov = daily.groupby(["region", "MCT_RY_CD", "ind_group", "scope"], as_index=False).agg(
        coverage=("observed", "mean"), median_cnt=("cnt", "median"),
        total_sales_100m=("sales", lambda s: s.sum() / 1e8), visitor_share=("sales_visitor", "sum"),
        age60p_share=("sales_age60p", "sum"))
    cov["visitor_share"] /= cov.total_sales_100m * 1e8
    cov["age60p_share"] /= cov.total_sales_100m * 1e8
    cov["reliable"] = ((cov.coverage >= C.MIN_COVERAGE) & (cov.median_cnt >= C.MIN_MEDIAN_CNT)).astype(int)
    return cov.sort_values(["region", "total_sales_100m"], ascending=[True, False])


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("[데이터2] 일별 x 고객세분 집계")
    daily = build_daily()
    cov = coverage_table(daily)
    daily = daily.merge(cov[["region", "MCT_RY_CD", "reliable"]], on=["region", "MCT_RY_CD"], how="left")
    daily.to_csv(C.PROCESSED_DIR / "card_daily.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(C.PROCESSED_DIR / "industry_coverage.csv", index=False, encoding="utf-8-sig")

    print("[데이터1] 일별 x 시간대 집계")
    slot = build_slot()
    slot = slot.merge(cov[["region", "MCT_RY_CD", "reliable"]], on=["region", "MCT_RY_CD"], how="left")
    slot.to_csv(C.PROCESSED_DIR / "card_slot.csv", index=False, encoding="utf-8-sig")

    print(f"\ncard_daily {daily.shape}, card_slot {slot.shape}")
    print("지역·범위별 업종 수(신뢰 가능/전체):")
    print(cov.groupby(["region", "scope"]).reliable.agg(["sum", "size"]))
    print("\n관측률(observed) 평균:", daily.groupby("region")["observed"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
