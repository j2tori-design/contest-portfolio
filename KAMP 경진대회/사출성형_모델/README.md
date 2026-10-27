# 사출성형 공정데이터 기반 품질불량 위험 예측 및 검사 우선순위 결정

사출 완료 후 기록된 공정값으로 품질검사 전에 제품별 위험점수를 매겨, 한정된 검사 용량을 어디에 먼저 쓸지 정하는 검사 보조 모델.

## 실행
```bash
pip install -r requirements.txt
python run_all.py                 # 전체 실행 → outputs/ (CPU 약 1시간, 사양에 따라 다름)
python run_all.py --quick         # CV 반복 1회 점검용 → outputs_quick/
python run_all.py --no-extra      # 부가 실험 생략
python run_all.py --clean         # 출력 폴더를 비우고 실행
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # 노트북 실행 (약 5분)
python predict.py --input data/moldset_labeled_cn7.csv --product cn7  # 저장 모델로 추론
```
- 노트북은 핵심 결과 파일이 없을 때만 `run_all.main()` 을 실행한다. 다시 만들려면 0장의 `FORCE_RERUN = True`.
- `outputs/run_manifest.json` 에 실행 시각, 소요 시간, 패키지 버전, 입력 데이터·코드 해시가 기록된다. 노트북은 해시가 다르면 경고한다.
- 시스템 패키지 수정 권한이 없는 환경(KAMP-NOTE 등)에서는 `pip install --user -r requirements.txt`.

## 검증 환경
| 구분 | Python | numpy | pandas | scikit-learn | lightgbm | xgboost | catboost | shap |
|---|---|---|---|---|---|---|---|---|
| 결과 생성 (로컬, Windows 11) | 3.12.7 | 1.26.4 | 2.2.2 | 1.5.1 | 4.6.0 | 3.4.1 | 1.2.10 | 0.50.0 |
| KAMP-NOTE (노트북 실행 확인) | 3.11.9 | 1.26.4 | 2.1.4 | 1.4.2 | 4.5.0 | --user 설치 | --user 설치 | 0.49.1 |

- `requirements.txt`: 최저 버전만 지정. KAMP-NOTE 는 사전 설치 패키지를 바꿀 수 없어 `==` 로 고정하면 설치가 실패한다.
- `requirements-lock.txt`: 결과를 생성한 환경의 정확한 버전.
- KAMP-NOTE 에서는 `analysis.ipynb` 를 저장된 `outputs/` 로 끝까지 실행해 확인했다. `run_all.py` 전체 재학습은 로컬 환경에서 실행했다.

## 폴더 구성
| 경로 | 내용 |
|---|---|
| `data/` | 제공 데이터 원본 4개 (labeled/unlabeled × CN7/RG3) |
| `src/config.py` | 경로·시드·CV 설정, 핵심/부가 결과 파일 목록 |
| `src/data.py` | 적재, 추정 사출 그룹(교차 기록 짝 병합)·추정 캐비티, 구조·품질 진단, 생산순번 블록 |
| `src/models.py` | 비교 모델 7종 (로지스틱 베이스라인, RandomForest, ExtraTrees, LightGBM, XGBoost, CatBoost, RF+ET 앙상블) |
| `src/evaluate.py` | 반복 사출 그룹 CV(그룹 겹침 검사, fold별 예측 기록), 학습 fold 내부 임계값, 지표(정의 불가 시 NaN) |
| `src/insights.py` | 분할 누수 점검, 과거→미래 대리 검증, OOF 판정 관점 분리, 운영점 오류분석, 규칙 베이스라인, 검사 용량표, 보정, 비용·손익분기, 사전경보 |
| `src/experiments.py` | 유의성(보정 t·Holm), 불균형 처리, 통합 vs 제품별 모델, 스태킹·MLP, 행 단위 누수 비교, 비라벨 진단, 군집, 오토인코더, 생산순번 블록 CV, 예측 기여 변수 |
| `run_all.py` | 전체 파이프라인 (`--quick`, `--no-extra`, `--clean`) |
| `predict.py` | 저장 모델로 신규 CSV 추론 |
| `analysis.ipynb` | 분석 노트북 (0 실행 제어, 1 데이터, 2 모델, 3 영향요인·오류, 4 현장 활용, 5 차별성, 6 재현성) |

## 산출물 (`outputs/`)
| 파일 | 내용 |
|---|---|
| `run_manifest.json` | 실행 기록 (모드, 시각, 소요 시간, 환경, 입력·코드 해시, 완료 단계) |
| `model/final_model.joblib`, `model/calibrated_model.joblib`, `model/model_meta.json` | 최종모델·보정 모델, 입력 열 목록·공통 임계값·보정 방식·등급 경계 |
| `data_diagnosis.csv`, `structure_summary.csv`, `single_groups.csv` | 제품별 진단, 추정 사출 그룹 구조(2행/1행/교차 짝/순번 누락/패턴 재등장), 연속 규칙상 1행 그룹 상세 |
| `data_quality.csv`, `outliers.csv`, `variable_stages.csv`, `scaling_check.csv` | 결측·중복·상수·극단값·라벨 충돌과 처리 근거, \|z\|>4 이상치, 공정 단계별 변수, 파일별 평균·표준편차 |
| `split_audit.csv`, `sample_counts.csv` | 모든 분할의 그룹 겹침 점검, 개발/테스트 × 제품별 행·그룹·불량 수 |
| `cv_folds.csv`, `cv_summary.csv` | 모델 × 특성셋 × 범위별 CV 성능 (fold 평가 규모, 정의 불가 사유 포함) |
| `significance.csv` | 최종모델 대비 짝 비교 (Wilcoxon, Nadeau-Bengio 보정 t, Holm 보정) |
| `calibration_cv.csv`, `calibration_oof.csv` | 보정 없음 / sigmoid / isotonic 비교 |
| `test_metrics.json`, `test_bootstrap_ci.csv`, `test_predictions.csv` | 테스트 성능, 사출 그룹 부트스트랩 95% CI, **테스트 예측결과** (`risk_score`, `risk_prob_cal`, `risk_grade`, 순위·제품별 상위 20%, `contrib_1~3`) |
| `oof_dev.csv`, `oof_fold_preds.csv`, `oof_threshold_views.csv` | 반복 평균 OOF 점수와 판정, fold별 점수·임계값·판정 기록, 세 판정 관점의 혼동행렬 |
| `forward_cv.csv` | 과거→미래 대리 검증 (구간별 표본·불량 수, N/A 사유) |
| `op_concentration.csv`, `op_rules_FN.txt`, `op_rules_FP.txt` (+ `_leaves.csv`) | 운영점 FN·FP 집중 조건, 규칙나무 |
| `rule_baselines.csv`, `capacity_table.csv` | 규칙 베이스라인 vs 모델, 검사 용량 5·10·20·30% × 할당 정책별 포착·미탐지 |
| `cost_curves.csv`, `cost_best.csv`, `cost_breakeven.csv` | 가정 비용비별 총비용·최적 검사 비율·손익분기 |
| `early_warning_series.csv`, `early_warning_summary.csv`, `regime_clusters.csv`, `regime_cluster_assign.csv` | 사전경보, 운전구간 군집 |
| 부가 실험 (`--no-extra` 시 생략) | `imbalance_cv.csv`, `product_model_cv.csv`, `leakage_cv.csv`, `unlabeled_domain_auc.csv`, `self_training_cv.csv`, `dl_stacking_cv.csv`, `time_block_cv.csv`, `autoencoder_cv.csv` |

## 핵심 설계
- **문제 정의**: 사출 완료 후·검사 전 시점의 검사 우선순위 결정 (사이클·충전 시간, 최대 사출압 등은 사출 후에 확정되는 값으로 추정).
- **추정 사출 그룹**: 원본에 사출·캐비티 번호가 없어, 공정값이 같은 연속 행과 5행 이내에 교차 기록된 같은 공정값 짝을 한 그룹으로 묶는다. 추정 캐비티 = 그룹 내 행 순서 (실제 위치 미확인).
- **분할**: 모든 분할(홀드아웃, 외부 CV, 내부 임계값·보정·스태킹 CV, 생산순번 블록, 과거→미래)이 그룹 단위이며 `split_audit.csv` 로 겹침 0 을 확인한다. 입력 열·임계값·보정은 학습 데이터 안에서만 정한다.
- **검증 3종**: ① 무작위 사출 그룹 CV(주 검증, 낙관적일 수 있음) ② 생산순번 블록 CV ③ 과거→미래 대리 검증(생산 순번 기반).
- **테스트셋**: 사출 그룹·불량 층화 20%. 모델 선정과 설정 변경에 쓰지 않았다.
- **선정 규칙(사전 고정)**: 개발셋 CV 전체 PR-AUC 평균 최대. 상위권 모델 간 차이는 통계적으로 유의하지 않다.
- **운영 방식**: 고정 임계값 대신 제품별 위험점수 상위 k% 검사. 순위는 보정 전 위험점수, 표시용 수치는 보정 확률.
- **원본 데이터 컬럼명**: 이름 없는 첫 열(`Unnamed: 0`)은 적재 시 `prod_idx`(생산 순번)로 바꾼다.

## 한계와 확인이 필요한 정보
- 변수 의미·단위·측정 시점, 타깃 코딩(1=불량), 사출·캐비티 번호는 자료로 확인되지 않아 추정했다.
- 불량이 42건(개발 34, 테스트 8)뿐이라 모든 지표의 불확실성이 크다. CN7 불량은 가동 초기 한 구간에만 있어 과거→미래 검증으로 평가할 수 없다.
- RG3 불량은 전부 같은 공정값에 양품도 있는 라벨 충돌이라, 공정변수만으로는 예측 상한이 낮다.
- 검사·유출 비용은 실측값이 없어 가정 시나리오로만 제시했다.
