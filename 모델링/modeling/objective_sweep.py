"""
1순위: loss_function 비교 (Logloss vs RMSE).

대회 지표가 Brier(=MSE)라서 RMSE로 직접 최적화하면 이론적으로 더 유리할 수 있다는 가설을 검증.
CatBoostClassifier는 RMSE를 loss_function으로 못 받아서 CatBoostRegressor를 쓰고, 예측값을 [0,1]로
clip한 뒤 baseline과 동일한 score 공식으로 비교한다. 동일 train/valid split, 동일 iterations 조건.
"""
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def score(brier: float, r: float) -> float:
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def main():
    df = load("train.csv")
    train_df, valid_df = bc.time_split(df, 2024)

    train_pool = Pool(train_df[bc.FEATURES], train_df[bc.TARGET], cat_features=bc.CAT_FEATURES)
    valid_pool = Pool(valid_df[bc.FEATURES], valid_df[bc.TARGET], cat_features=bc.CAT_FEATURES)

    model = CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=6,
        loss_function="RMSE", eval_metric="RMSE",
        l2_leaf_reg=bc.L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    p = np.clip(model.predict(valid_pool), 0.0, 1.0)
    y = valid_df[bc.TARGET].to_numpy()
    brier = float(np.mean((p - y) ** 2))
    r = y.mean()
    print(f"RMSE objective: best_iteration={model.get_best_iteration()} brier={brier:.6f} score={score(brier, r):.2f}")
    print(f"(비교 기준: Logloss baseline brier=0.247972 score=734.49)")


if __name__ == "__main__":
    main()
