"""
R/M/O hazard 분해 모델.

qR = P(reverse | 이 투구)
qM = P(middle | reverse 아님)
qO = P(outside | reverse 아님, middle 아님)

P(success) = (1-qR) x (1-qM) x (1-qO)

baseline 44피처 그대로 사용, 세 서브모델 모두 CatBoost. single-split(2019-23->24)로 먼저 확인.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from rmo_labels import add_rmo_labels  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from catboost import CatBoostClassifier, Pool


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def train_sub(train_df, target_col, label):
    y = train_df[target_col].to_numpy()
    pool = Pool(train_df[bc.FEATURES], y, cat_features=bc.CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss",
        l2_leaf_reg=bc.L2_LEAF_REG, random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(pool)
    print(f"  {label} 학습 완료 (n={len(train_df):,}, positive_rate={y.mean():.4f})")
    return model


def main():
    df = add_rmo_labels(load("train.csv"))
    train_df, valid_df = bc.time_split(df, 2024)
    train_df = train_df.dropna(subset=["reverse_label", "middle_label"])

    print("서브모델 학습:")
    qR_model = train_sub(train_df, "reverse_label", "qR (reverse)")

    not_reverse = train_df[train_df["reverse_label"] == 0]
    qM_model = train_sub(not_reverse, "middle_label", "qM (middle | not reverse)")

    not_reverse_middle = not_reverse[not_reverse["middle_label"] == 0]
    not_reverse_middle = not_reverse_middle[not_reverse_middle["outside_label"].isin([0, 1])]
    qO_model = train_sub(not_reverse_middle, "outside_label", "qO (outside | not R, not M)")

    valid_pool = Pool(valid_df[bc.FEATURES], cat_features=bc.CAT_FEATURES)
    qR = qR_model.predict_proba(valid_pool)[:, 1]
    qM = qM_model.predict_proba(valid_pool)[:, 1]
    qO = qO_model.predict_proba(valid_pool)[:, 1]

    p_success = (1 - qR) * (1 - qM) * (1 - qO)
    y_valid = valid_df[bc.TARGET].to_numpy()
    r = y_valid.mean()
    brier = float(np.mean((p_success - y_valid) ** 2))
    print(f"\nR/M/O hazard 결합: score={score(brier, r):.2f} brier={brier:.6f}")
    print("(비교 기준: baseline 734.49)")

    # 참고: 직접 CatBoost(단일 이진분류) 대비
    direct = bc.train_catboost(train_df, valid_df)
    p_direct = direct.predict_proba(valid_pool)[:, 1]
    brier_direct = float(np.mean((p_direct - y_valid) ** 2))
    print(f"(참고: 동일 train_df subset으로 학습한 단일 CatBoost: score={score(brier_direct, r):.2f})")


if __name__ == "__main__":
    main()
