"""
season 지수감쇠 recency weighting을 여러 lambda로 제대로 스윕. 이전엔 λ=0.5 하나만 테스트해서
694.91로 기각했었는데(baseline 734.49 대비), 이번엔 λ=0(가중치 없음, 현재)부터 촘촘히 스윕해서
정말 어떤 λ도 도움 안 되는지 확인. primary(2023->2024)/stress(2022->2023) 둘 다 확인.

weight(season) = exp(-λ * (target_year - 1 - season)), 평균 1로 정규화해서 sample_weight로 사용.
"""
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

LAMBDA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def recency_weight(seasons, source_max_season, lam):
    age = source_max_season - seasons.to_numpy()
    w = np.exp(-lam * age)
    return (w / w.mean()).astype(np.float64)


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season]
    y_valid = valid_df["control_success"].to_numpy(np.float64)
    r = float(y_valid.mean())
    source_max_season = int(train_df["season"].max())

    valid_pool = Pool(valid_df[FEATURES], y_valid, cat_features=CAT_FEATURES)

    for lam in LAMBDA_GRID:
        weight = recency_weight(train_df["season"], source_max_season, lam)
        train_pool = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES, weight=weight)
        model = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
            eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
            random_seed=42, thread_count=-1, verbose=False,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        p = model.predict_proba(valid_pool)[:, 1]
        brier = float(np.mean((p - y_valid) ** 2))
        s = score(brier, r)
        tag = " (=baseline, 가중치 없음)" if lam == 0.0 else ""
        print(f"  lambda={lam}: {s:.2f}{tag}")


def main():
    df = load("train.csv")
    run_fold(df, 2024, "PRIMARY")
    run_fold(df, 2023, "STRESS")


if __name__ == "__main__":
    main()
