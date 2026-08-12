"""
STEP E: 2023~2024에 더 큰 학습 가중치를 줘서 2020~2022(regime 1) 정보의 영향력을 줄인다.
피처 추가 없이 CatBoost sample_weight만 바꾼다. single-split 2019-23->24.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from catboost import CatBoostClassifier, Pool


def run(train_df, valid_df, weight_2020_22):
    w = np.where(train_df["season"] < 2023, weight_2020_22, 1.0)
    train_pool = Pool(train_df[bc.FEATURES], train_df[bc.TARGET], cat_features=bc.CAT_FEATURES, weight=w)
    valid_pool = Pool(valid_df[bc.FEATURES], valid_df[bc.TARGET], cat_features=bc.CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="BrierScore",
        l2_leaf_reg=bc.L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    m = bc.evaluate(model, valid_df)
    print(f"2020~22 weight={weight_2020_22}: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    df = load("train.csv")
    train_df, valid_df = bc.time_split(df, 2024)
    for w in [1.0, 0.75, 0.5, 0.25]:
        run(train_df, valid_df, w)


if __name__ == "__main__":
    main()
