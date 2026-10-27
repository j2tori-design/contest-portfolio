# 2026 빅콘테스트 – 기후 취약 상권 분석

## 실행 순서 (프로젝트 루트에서)
```
python src/01_fetch_weather.py          # KMA_API_KEY 필요 (개발용: --source openmeteo)
python src/02_build_card_panel.py
python src/03_build_panel.py
python src/04_model_sensitivity.py      # 업종별 민감도(축소추정)·순손실률·이벤트 단위 검증 → outputs/
python src/05_robustness_checks.py      # 지역 단위 순손실·지역 차이 검정, 마스킹 편향 점검
python src/06_priority_algorithm.py     # 예보 기반 우선지원 알고리즘·백테스트·가중치 민감도
python src/14_ml_benchmark.py           # 머신러닝 비교 모형(그래디언트 부스팅), 04와 같은 교차검증 → F6에 반영 (07 전에 실행)
python src/07_figures.py                # 발표용 그래프 → outputs/figures/
python src/08_sk_flow.py                # SK 유동인구 방문형 상권 지수·지도(F10)
python src/10_subway_flow.py            # 지하철 도착 인원(발길)·매출 감소 원인 분해(F12·F13)
python src/11_index_payout.py           # 호우 지수형 손실보전 지급표·예산·기초위험 백테스트(F14)
python src/09_report_numbers.py         # 발표자료 핵심 숫자 모음 → outputs/report_numbers.csv (10·11·14 다음에 실행)
python src/12_threshold_sensitivity.py  # (점검) 호우 기준 30·40·50mm·호우주의보급 민감도
python src/13_count_threshold_sensitivity.py  # (점검) 분석 대상 업종 기준(하루 결제 중간값) 민감도
python presentation/build_deck.py       # 발표자료 PPT (PDF는 PowerPoint에서 저장)
```
필요 패키지: pandas, numpy (분석 단계: statsmodels, scikit-learn, matplotlib, seaborn)

## 산출물 (`data/processed/`)
| 파일 | 단위 | 내용 |
|---|---|---|
| `panel_daily.csv` | 일자 x 지역 x 업종 | 데이터2 매출(현지/외지/연령 세분) + 일 기상 + lag/lead + 달력 |
| `panel_slot.csv` | 일자 x 지역 x 업종 x 시간대 | 데이터1 개인 매출(법인 분리) + 시간대 기상 + 일 기상 |
| `industry_coverage.csv` | 지역 x 업종 | 관측률, 외지인·고령 비중, 분석 신뢰 여부(`reliable`) |
| `weather_*.csv` | 일/시간대/시간 | 기상 정제본 |

분석 기본 대상: `scope == "core"` & `reliable == 1`.
`observed == 0`은 마스킹(5건 미만)으로 행이 없는 날이며 매출은 NaN으로 둔다.

## 외부 데이터 출처 (제출 규정: 출처·수집방법·원본 제출)
- 기상청 지상(종관, ASOS) 일자료·시간자료 조회서비스, 공공데이터포털(data.go.kr), 지점 108(서울)·101(춘천), 2025-06-28~2026-01-01. 원본 응답: `data/external/kma_asos/`
- 서울시 지하철호선별 역별 승하차 인원 정보(교통카드 기준), 서울 열린데이터광장 OA-12914, 공공누리 제1유형.
  2025년 7~12월 월별 파일(CARD_SUBWAY_MONTH_2025MM.csv)을 데이터셋 페이지의 '파일 내려받기'로 수집.
  사용 역: 경춘선 춘천·남춘천·김유정·강촌·백양리·굴봉산(춘천시), 강남구 소재 25개 역. 원본: `data/external/seoul_subway/`
- 풍수해보험 보상 범위(건물·시설·재고): 행정안전부 소상공인 풍수해보험 안내 및 관련 보도(정책 슬라이드의 배경 설명에만 사용, 분석에는 미사용)
- (개발용, 제출 제외) Open-Meteo Historical Weather API(ERA5). 원본: `data/external/openmeteo/`
- 공휴일: 관보 고시 기준, `src/config.py`의 `HOLIDAYS`

## 정제 규칙
- 카드 원본 cp949, 연령 표기 공백 제거('20 대'→'20대')
- 업종 분류·분석 범위: `src/config.py`의 `INDUSTRY_GROUPS`
- 고객 구분: 거주 시도가 가맹점 시도와 같으면 현지, 다르면 외지, '정보없음'은 unknown
- 체감온도: 기상청 식(5~9월 여름철 식, 그 외 겨울철 풍속냉각)
