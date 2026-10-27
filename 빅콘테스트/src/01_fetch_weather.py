"""기상 데이터 수집 → data/processed/weather_daily.csv, weather_slot.csv

공식(제출용): 공공데이터포털 '기상청_지상(종관, ASOS) 일자료/시간자료 조회서비스'
    1) data.go.kr 에서 두 서비스 활용신청 → 일반 인증키(Decoding) 발급
    2) 환경변수 KMA_API_KEY 에 키 설정 후 실행
         PowerShell: $env:KMA_API_KEY="발급키"; python src/01_fetch_weather.py
개발용(키 없을 때): python src/01_fetch_weather.py --source openmeteo
    ERA5 재분석 기반이라 극값이 관측보다 완만함 → 최종 보고서에는 사용하지 않음.

원본 응답은 data/external/<source>/ 에 그대로 저장한다(외부 데이터 원본 제출 규정).
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import config as C

ASOS_DAILY_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
ASOS_HOURLY_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
PAGE_ROWS = 999


# ================================================================ HTTP
def _get_json(url: str, params: dict, cache_path, retries: int = 3) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    full = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=60) as r:
                text = r.read().decode("utf-8")
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            print(f"  재시도 {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 인증키 오류 등은 XML로 돌아온다
        raise RuntimeError(f"JSON이 아닌 응답(인증키·활용신청 확인 필요):\n{text[:500]}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _asos_items(url: str, key: str, extra: dict, tag: str) -> list:
    """페이지를 끝까지 넘기며 item 목록 수집."""
    items, page = [], 1
    while True:
        params = {"serviceKey": key, "pageNo": page, "numOfRows": PAGE_ROWS, "dataType": "JSON",
                  "dataCd": "ASOS", **extra}
        cache = C.EXTERNAL_DIR / "kma_asos" / f"{tag}_p{page}.json"
        res = _get_json(url, params, cache)["response"]
        if res["header"]["resultCode"] != "00":
            cache.unlink(missing_ok=True)
            raise RuntimeError(f"ASOS API 오류 {res['header']}")
        body = res["body"]
        got = body["items"]["item"] if body.get("items") else []
        items.extend(got)
        if page * PAGE_ROWS >= int(body["totalCount"]) or not got:
            return items
        page += 1


def _num(s: pd.Series, fill=np.nan) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce").fillna(fill)


# ================================================================ ASOS
def fetch_asos(key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "%" in key:  # Encoding 키를 넣은 경우 이중 인코딩 방지
        key = urllib.parse.unquote(key)
    start, end = C.WEATHER_START.replace("-", ""), C.WEATHER_END.replace("-", "")
    daily, hourly = [], []
    for region, r in C.REGIONS.items():
        stn = r["stn_id"]
        print(f"[ASOS] {region}({stn}) 일자료")
        d = pd.DataFrame(_asos_items(ASOS_DAILY_URL, key,
                                     {"dateCd": "DAY", "startDt": start, "endDt": end, "stnIds": stn},
                                     f"daily_{stn}_{start}_{end}"))
        daily.append(pd.DataFrame({
            "region": region,
            "date": pd.to_datetime(d["tm"]),
            "tavg": _num(d["avgTa"]), "tmin": _num(d["minTa"]), "tmax": _num(d["maxTa"]),
            "rain": _num(d["sumRn"], 0.0),            # 공백 = 무강수
            "rain_1h_max": _num(d["hr1MaxRn"], 0.0),
            "snow_new": _num(d["ddMefs"], 0.0),       # 일 최심신적설(cm)
            "snow_depth": _num(d["ddMes"], 0.0),      # 일 최심적설(cm)
            "rh_avg": _num(d["avgRhm"]), "ws_avg": _num(d["avgWs"]),
        }))
        print(f"[ASOS] {region}({stn}) 시간자료")
        h = pd.DataFrame(_asos_items(ASOS_HOURLY_URL, key,
                                     {"dateCd": "HR", "startDt": start, "startHh": "00",
                                      "endDt": end, "endHh": "23", "stnIds": stn},
                                     f"hourly_{stn}_{start}_{end}"))
        hourly.append(pd.DataFrame({
            "region": region,
            "time": pd.to_datetime(h["tm"]),
            "ta": _num(h["ta"]), "rn": _num(h["rn"], 0.0), "hm": _num(h["hm"]), "ws": _num(h["ws"]),
            "snow_depth": _num(h["dsnw"], 0.0),
        }))
    daily = pd.concat(daily, ignore_index=True).assign(source="KMA ASOS")
    hourly = pd.concat(hourly, ignore_index=True).assign(source="KMA ASOS")
    return daily, hourly


# ================================================================ Open-Meteo (개발용)
def fetch_openmeteo() -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = []
    for region, r in C.REGIONS.items():
        print(f"[Open-Meteo] {region}")
        params = {"latitude": r["lat"], "longitude": r["lon"], "start_date": C.WEATHER_START,
                  "end_date": C.WEATHER_END, "timezone": "Asia/Seoul", "wind_speed_unit": "ms",
                  "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,snowfall,snow_depth"}
        cache = C.EXTERNAL_DIR / "openmeteo" / f"hourly_{r['stn_id']}_{C.WEATHER_START}_{C.WEATHER_END}.json"
        h = _get_json(OPENMETEO_URL, params, cache)["hourly"]
        hourly.append(pd.DataFrame({
            "region": region, "time": pd.to_datetime(h["time"]),
            "ta": h["temperature_2m"], "rn": h["precipitation"], "hm": h["relative_humidity_2m"],
            "ws": h["wind_speed_10m"], "snowfall": h["snowfall"],
            "snow_depth": np.array(h["snow_depth"], dtype=float) * 100,  # m → cm
        }))
    hourly = pd.concat(hourly, ignore_index=True).assign(source="Open-Meteo ERA5(개발용)")
    hourly[["rn", "snowfall"]] = hourly[["rn", "snowfall"]].fillna(0.0)
    g = hourly.assign(date=hourly.time.dt.normalize()).groupby(["region", "date"])
    daily = g.agg(tavg=("ta", "mean"), tmin=("ta", "min"), tmax=("ta", "max"), rain=("rn", "sum"),
                  rain_1h_max=("rn", "max"), snow_new=("snowfall", "sum"), snow_depth=("snow_depth", "max"),
                  rh_avg=("hm", "mean"), ws_avg=("ws", "mean")).reset_index()
    daily["source"] = hourly.source.iloc[0]
    return daily, hourly.drop(columns="snowfall")


# ================================================================ 파생 변수
def feels_like(ta: pd.Series, hm: pd.Series, ws: pd.Series, month: pd.Series) -> pd.Series:
    """기상청 체감온도: 5~9월 여름철 식(습구온도 Stull 추정), 그 외 겨울철 풍속냉각 식."""
    atan = np.arctan
    tw = (ta * atan(0.151977 * np.sqrt(hm + 8.313659)) + atan(ta + hm) - atan(hm - 1.67633)
          + 0.00391838 * hm ** 1.5 * atan(0.023101 * hm) - 4.686035)
    summer = -0.2442 + 0.55399 * tw + 0.45535 * ta - 0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0
    v = ws * 3.6  # m/s → km/h
    chill = 13.12 + 0.6215 * ta - 11.37 * v ** 0.16 + 0.3965 * ta * v ** 0.16
    winter = np.where((ta <= 10) & (v >= 4.8), chill, ta)
    return pd.Series(np.where(month.between(5, 9), summer, winter), index=ta.index)


def slot_of(hour: pd.Series) -> pd.Series:
    lut = {h: s for s, hours in C.TIME_SLOTS.items() for h in hours}
    return hour.map(lut)


def add_hourly_derived(daily: pd.DataFrame, hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = hourly.copy()
    hourly["date"] = hourly.time.dt.normalize()
    hourly["feels"] = feels_like(hourly.ta, hourly.hm, hourly.ws, hourly.time.dt.month)
    hourly["slot"] = slot_of(hourly.time.dt.hour)
    hourly["rain_hour"] = (hourly.rn >= 0.1).astype(int)
    hourly["day_rain"] = np.where(hourly.time.dt.hour.between(9, 20), hourly.rn, 0.0)

    by_day = hourly.sort_values("time").groupby(["region", "date"]).agg(
        feels_max=("feels", "max"), feels_min=("feels", "min"),
        rain_hours=("rain_hour", "sum"), rain_daytime=("day_rain", "sum"), n_hours=("ta", "size"),
        depth_max_h=("snow_depth", "max"), depth_last_h=("snow_depth", "last"),
    ).reset_index()
    # 춘천(101) 등은 일자료 적설 항목이 공란 → 시간 적설심으로 신적설 추정
    # 신적설 = 당일 최대 적설심 - 전날 마지막 적설심 (서울 108 일자료 ddMefs와 일치 확인)
    prev_last = by_day.groupby("region").depth_last_h.shift(1).fillna(0)
    by_day["snow_new_h"] = (by_day.depth_max_h - prev_last).clip(lower=0)
    daily = daily.merge(by_day.drop(columns="depth_last_h"), on=["region", "date"], how="left")
    daily["snow_new"] = daily[["snow_new", "snow_new_h"]].max(axis=1)
    daily["snow_depth"] = daily[["snow_depth", "depth_max_h"]].max(axis=1)
    daily = daily.drop(columns=["snow_new_h", "depth_max_h"])

    slot = hourly.groupby(["region", "date", "slot"]).agg(
        ta_mean=("ta", "mean"), ta_max=("ta", "max"), feels_max=("feels", "max"), feels_min=("feels", "min"),
        rain=("rn", "sum"), rain_1h_max=("rn", "max"), rain_hours=("rain_hour", "sum"),
        snow_depth=("snow_depth", "max"), n_hours=("ta", "size"),
    ).reset_index()
    return daily, slot


def add_event_flags(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["region", "date"]).copy()
    d["rain_bin"] = pd.cut(d.rain, C.RAIN_BINS, labels=C.RAIN_LABELS)
    d["heat33"] = (d.tmax >= C.HEAT_TMAX).astype(int)
    d["heat35"] = (d.tmax >= C.HEAT_SEVERE_TMAX).astype(int)
    d["feels33"] = (d.feels_max >= C.HEAT_TMAX).astype(int)
    d["heavy_rain"] = (d.rain >= C.HEAVY_RAIN_MM).astype(int)
    d["snow"] = (d.snow_new >= C.SNOW_MIN_CM).astype(int)
    d["snow_cover"] = (d.snow_depth > 0).astype(int)  # 쌓인 눈 지속(결빙·보행 불편)
    d["cold"] = (d.tmin <= C.COLD_TMIN).astype(int)
    g = d.groupby("region")
    for col in ["rain", "heavy_rain", "snow", "heat33"]:
        for k in range(1, C.LAG_DAYS + 1):
            d[f"{col}_lag{k}"] = g[col].shift(k)
        d[f"{col}_lead1"] = g[col].shift(-1)  # 위약 검정(placebo)·사전 대비 소비용
    # 폭염 지속일수: 연속된 heat33 일수(누적 피로 효과)
    run = d.heat33.groupby((d.heat33 != g.heat33.shift()).cumsum()).cumsum()
    d["heat_run"] = run.where(d.heat33 == 1, 0)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["asos", "openmeteo"], default="asos")
    args = ap.parse_args()

    if args.source == "asos":
        key = os.environ.get("KMA_API_KEY", "")
        # 제출 폴더에는 원본 응답이 저장되어 있어 인증키 없이도 재현 가능(저장본이 없을 때만 키 필요)
        if not key and not any((C.EXTERNAL_DIR / "kma_asos").glob("*.json")):
            sys.exit("KMA_API_KEY 환경변수가 없습니다. 파일 상단 설명 참고 (개발용: --source openmeteo)")
        daily, hourly = fetch_asos(key)
    else:
        daily, hourly = fetch_openmeteo()

    daily, slot = add_hourly_derived(daily, hourly)
    daily = add_event_flags(daily)
    slot["source"] = daily.source.iloc[0]

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(C.PROCESSED_DIR / "weather_daily.csv", index=False, encoding="utf-8-sig")
    slot.to_csv(C.PROCESSED_DIR / "weather_slot.csv", index=False, encoding="utf-8-sig")
    hourly.to_csv(C.PROCESSED_DIR / "weather_hourly.csv", index=False, encoding="utf-8-sig")

    in_range = daily[daily.date.between(C.ANALYSIS_START, C.ANALYSIS_END)]
    print(f"\n출처: {daily.source.iloc[0]}  | 일자료 {len(daily)}행, 시간대 {len(slot)}행")
    print(in_range.groupby("region")[["heat33", "feels33", "heavy_rain", "snow", "cold"]].sum())
    short = daily[daily.n_hours < 24]
    if len(short):
        print(f"[경고] 시간자료 24개 미만인 날 {len(short)}일:", short[["region", "date", "n_hours"]].head().to_dict("records"))


if __name__ == "__main__":
    main()
