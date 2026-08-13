"""
R/M/O hazard를 CatBoost 대체가 아니라 "innovation"(보정치)으로 사용.
참고: 다른 참가자 공개 레포의 E16-H1도 standalone은 기각됐지만 beta=0.25 innovation으로는 채택됨.

p_final = p_catboost + beta * (p_rmo_hazard - p_catboost)
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


def train_sub(train_df, target_col):
    y = train_df[target_col].to_numpy()
    pool = Pool(train_df[bc.FEATURES], y, cat_features=bc.CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss",
        l2_leaf_reg=bc.L2_LEAF_REG, random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(pool)
    return model


def main():
    df = add_rmo_labels(load("train.csv"))
    train_df, valid_df = bc.time_split(df, 2024)
    y_valid = valid_df[bc.TARGET].to_numpy()
    r = y_valid.mean()
    valid_pool = Pool(valid_df[bc.FEATURES], cat_features=bc.CAT_FEATURES)

    # baseline (전체 train_df, NaN 드랍 없이 = 원래 734.49와 동일 조건)
    cat_model = bc.train_catboost(train_df, valid_df)
    p_cat = cat_model.predict_proba(valid_pool)[:, 1]
    brier_cat = float(np.mean((p_cat - y_valid) ** 2))
    print(f"baseline CatBoost: score={score(brier_cat, r):.2f}")

    # R/M/O 서브모델 (라벨 있는 subset만)
    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])
    qR_model = train_sub(rmo_train, "reverse_label")
    not_reverse = rmo_train[rmo_train["reverse_label"] == 0]
    qM_model = train_sub(not_reverse, "middle_label")
    not_rm = not_reverse[not_reverse["middle_label"] == 0]
    not_rm = not_rm[not_rm["outside_label"].isin([0, 1])]
    qO_model = train_sub(not_rm, "outside_label")

    qR = qR_model.predict_proba(valid_pool)[:, 1]
    qM = qM_model.predict_proba(valid_pool)[:, 1]
    qO = qO_model.predict_proba(valid_pool)[:, 1]
    p_rmo = (1 - qR) * (1 - qM) * (1 - qO)

    print("\nbeta sweep (p_final = p_cat + beta*(p_rmo - p_cat)):")
    for beta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]:
        p_final = p_cat + beta * (p_rmo - p_cat)
        brier = float(np.mean((p_final - y_valid) ** 2))
        print(f"  beta={beta}: score={score(brier, r):.2f} brier={brier:.6f}")


if __name__ == "__main__":
    main()
