"""비교 모델 정의. 베이스라인(로지스틱) + 배깅 계열 + 부스팅 계열 + 앙상블.

불량이 30여 건뿐인 데이터라 부스팅은 얕은 나무·강한 규제로, 배깅은 잎 최소 2개로 과적합을 억제했다.
"""
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import SEED

SCALE_POS_WEIGHT = 50   # 양품:불량 ≈ 55:1 → 부스팅의 불량 가중치 (XGBoost)


def _rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)


def _et():
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED)


def get_models() -> dict:
    return {
        "LR_baseline": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1),
        ),
        "RandomForest": _rf(),
        "ExtraTrees": _et(),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            is_unbalance=True, random_state=SEED, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
            min_child_weight=5, scale_pos_weight=SCALE_POS_WEIGHT, n_jobs=8, random_state=SEED, verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.03, l2_leaf_reg=10, auto_class_weights="SqrtBalanced",
            random_seed=SEED, verbose=0, thread_count=8, allow_writing_files=False,
        ),
        "RF+ET_ensemble": VotingClassifier([("rf", _rf()), ("et", _et())], voting="soft"),
    }
