# 동물보호소 개·고양이 안락사 예측: 평가 설계가 보고 성능에 미치는 영향 — 분석 코드

| 파일·폴더 | 내용 |
|---|---|
| `Animal_Shelter_Analytics_최종_논문용.ipynb` | 주 분석 (데이터 준비 → 순차 검증 선택 → 2024 테스트 → 해석, 19-1절 논문 수정본 근거 수치) |
| `비교실험_선행연구_평가설계.ipynb` | 비교 실험 E1~E3 (+ E1 보강: 안락사 확률 순위 재현율) |
| `run_final.py`, `run_compare.py` | 두 노트북을 새 커널에서 처음부터 실행하고 출력을 저장 |
| `requirements_final.txt` | 검증 환경 (Python 3.12.7, Windows 11) |
| `데이터/` | 오스틴 동물센터 입소·퇴소 CSV (아래 참고) |
| `cache_tune/` | 하이퍼파라미터 튜닝(10-2절) 폴드 예측 저장분 |
| `outputs_final/`, `figures_final/` | 주 분석 결과 (`summary.json`, 표, 그림) |
| `outputs_compare/`, `figures_compare/` | 비교 실험 결과 |

## 실행 순서

```bash
pip install -r requirements_final.txt
python run_final.py
python run_compare.py
```

- `run_compare.py`는 주 분석이 저장한 최종 설정(`outputs_final/summary.json`)을 읽으므로 **반드시 `run_final.py` 다음에** 실행합니다. 파일이 없으면 안내를 출력하고 종료합니다.
- 두 스크립트는 실행 전에 노트북의 이전 출력을 지우고, 실행 결과를 노트북에 다시 저장합니다. 오류가 나면 0이 아닌 코드로 끝납니다.
- 대략의 소요 시간 (Windows 11, 4코어 사용 기준, `cache_tune/`이 있을 때): `run_final.py` 약 40분(2026-10-26 실행 2,283초), `run_compare.py` 약 5~10분. 대부분은 군집 부트스트랩(B=500)과 SHAP 계산입니다.

## `cache_tune/`의 역할

10-2절은 상위 3개 계열 × 불균형 전략 × 후보 9개를 3개 순차 검증 폴드에서 학습합니다. 조합마다 폴드 예측을 `cache_tune/{계열}_{md5}.pkl`로 저장하고, 같은 조합이면 다시 학습하지 않고 읽습니다(로그에 `저장분 사용`, 진행 기록은 `run_progress.log`).

- 키는 (계열, 전략, 파라미터, 특성, 시드, 폴드 경계)의 `repr`을 md5한 값입니다. 이 값들이 바뀌면 키가 달라져 튜닝을 처음부터 다시 합니다.
- 저장분이 없으면 MLP·SMOTENC 조합 때문에 **몇 시간** 걸립니다. 폴더를 지우지 마세요.

## 데이터

### 필수: `데이터/` (오스틴 동물센터 레거시 파일, 2013-10-01 ~ 2025-05-05)

- `Austin_Animal_Center_Intakes_(10_01_2013_to_05_05_2025)_20260925.csv`
- `Austin_Animal_Center_Outcomes_(10_01_2013_to_05_05_2025)_20260925.csv`

내려받는 방식에 따라 괄호가 밑줄로 바뀐 이름(`Austin_Animal_Center_Intakes__10_01_2013_to_05_05_2025__20260925.csv`)일 수 있으며, 노트북은 두 형식을 모두 찾습니다. 둘 다 없으면 2장에서 즉시 중단합니다.

2025-05 이후 신규 시스템 파일(`Austin_Animal_Center_Intakes_20260925.csv`, `Austin_Animal_Center_Outcomes_20260925.csv`)은 동물 ID를 레거시와 연결할 수 없어 이번 분석에 쓰지 않습니다.

### 선택: Kaggle 2021년판 (외부 대조용)

Kaggle 데이터셋 [jackdaoud/animal-shelter-analytics](https://www.kaggle.com/datasets/jackdaoud/animal-shelter-analytics)(2013-10 ~ 2021-03 스냅숏)의 zip을 내려받아 다음 중 한 곳에 둡니다. 노트북은 이 순서로 찾습니다.

1. 환경변수 `KAGGLE_2021_ZIP`에 zip 경로 지정
2. `데이터/` 폴더 안에 zip 그대로 두기 (예: `데이터/반려동물 데이터셋.zip`). zip 안에 `Austin_Animal_Center_Intakes.csv`, `Austin_Animal_Center_Outcomes.csv`가 있어야 합니다.
3. 기존 경로 `../../../../매트릭스/Animal Shelter 프로젝트/반려동물 데이터셋.zip` (원작성자 환경)

이 파일이 없으면 3-1절(날짜 해석 외부 대조), 3-2절(사후 수정 점검), 6-4절 자연 실험을 건너뜁니다. 이때 19-1절 표와 `summary.json`의 `논문 수정본 근거`에 **"외부 대조 생략됨"**이 표시되고, `summary.json`의 `민감도` 중 `사후 수정 입소 제외 AP`, `사후 수정 제외-주 분석 차이`는 `null`이 됩니다. 나머지 결과(모델 선택, 테스트 성능)는 이 파일과 무관합니다.

## 원칙

- 2024 테스트 기간은 어떤 선택(변수, 모델, 하이퍼파라미터, 구조, 불균형 전략, 임곗값, 기준값)에도 쓰지 않습니다. 19-1절과 E1 보강처럼 나중에 추가한 셀은 모두 보고용입니다.
- 해석 문단은 실행 결과에서 자동 생성합니다(`show_md`).
