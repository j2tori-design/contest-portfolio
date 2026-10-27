"""경로·상수 설정."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

QUICK = os.environ.get("KAMP_QUICK") == "1"   # python run_all.py --quick : 반복 수 축소, 별도 폴더에 저장
OUT_DIR = ROOT / ("outputs_quick" if QUICK else "outputs")

PRODUCTS = ["cn7", "rg3"]
TARGET = "PassOrFail"          # 1 = 불량
RAW_INDEX_COL = "Unnamed: 0"   # 원본 CSV 의 이름 없는 순번 컬럼
INDEX_COL = "prod_idx"         # 원본 생산 순번 (적재 시 이름 변경)

SEED = 42
N_SPLITS = 5                       # 외부 CV fold 수
N_REPEATS = 1 if QUICK else 5      # 외부 CV 반복 수
TEST_FRAC = 0.2                    # 홀드아웃(테스트) 비율, 사출 그룹 단위 층화 분할
TOP_K = [0.05, 0.10, 0.20]         # 검사 우선순위: 상위 k% 검사 시 불량 포착률

RUN_EXTRA = True                   # 부가 실험(누수 비교·비라벨 진단·군집·오토인코더 등) 실행 여부
SEMI_REPEATS = 1 if QUICK else 3   # 자기학습 실험 반복 수 (비라벨 7만 행 예측이 포함되어 느림)
TB_REPEATS = 1 if QUICK else 3     # 생산순번 블록 CV 반복 수

RUN_EXTRA = os.environ.get("KAMP_NO_EXTRA") != "1" and RUN_EXTRA   # python run_all.py --no-extra 로도 끌 수 있음

# 노트북이 읽는 결과 파일. CORE 가 없으면 파이프라인을 실행하고, EXTRA 가 없으면 해당 셀만 건너뛴다.
CORE_OUTPUTS = [
    "run_manifest.json", "data_diagnosis.csv", "structure_summary.csv", "single_groups.csv", "data_quality.csv",
    "outliers.csv", "variable_stages.csv", "scaling_check.csv", "split_audit.csv", "sample_counts.csv",
    "cv_folds.csv", "cv_summary.csv", "significance.csv", "calibration_cv.csv", "calibration_oof.csv",
    "test_metrics.json", "test_predictions.csv", "test_bootstrap_ci.csv", "oof_dev.csv", "oof_fold_preds.csv",
    "oof_threshold_views.csv", "forward_cv.csv", "regime_clusters.csv", "regime_cluster_assign.csv",
    "op_concentration.csv", "op_rules_FN.txt", "op_rules_FP.txt", "rule_baselines.csv", "capacity_table.csv",
    "cost_curves.csv", "cost_best.csv", "cost_breakeven.csv", "early_warning_series.csv", "early_warning_summary.csv",
    "model/final_model.joblib", "model/model_meta.json",
]
EXTRA_OUTPUTS = ["imbalance_cv.csv", "leakage_cv.csv", "unlabeled_domain_auc.csv", "self_training_cv.csv",
                 "dl_stacking_cv.csv", "time_block_cv.csv", "autoencoder_cv.csv", "product_model_cv.csv"]
