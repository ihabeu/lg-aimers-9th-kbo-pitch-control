"""
CatBoost monotone_constraints 테스트. 방향이 명확한 피처들에 규제를 걸어서 일반화가 좋아지는지 확인
(HANDOFF.md에 남아있던 "저비용으로 한 번 더 확인해볼 것" 항목). raw 44피처 구조/하이퍼파라미터는
baseline_catboost.py 그대로, monotone_constraints만 추가.

방향이 명확한 피처만 제약:
  + asof_pitcher_success_rate, prev1/3/5_game_success_rate (커리어/최근 성공률 높을수록 성공확률 증가)
  + asof_pitcher_strike_rate (스트라이크 비율 높을수록 제구 성공 증가)
  - asof_pitcher_ball_rate (볼 비율 높을수록 제구 성공 감소)
나머지 피처는 방향을 확신할 수 없어 제약 없음(0).
"""
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import ROLLING_FOLDS, FEATURES, CAT_FEATURES, L2_LEAF_REG, time_split  # noqa: E402

MONOTONE_UP = {
    "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_strike_rate",
}
MONOTONE_DOWN = {"asof_pitcher_ball_rate"}


def score(brier: float, r: float) -> float:
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def fit_and_score(train_df, valid_df, use_constraints: bool):
    train_pool = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES)
    valid_pool = Pool(valid_df[FEATURES], valid_df["control_success"], cat_features=CAT_FEATURES)
    kwargs = dict(
        iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
        eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=42, thread_count=-1, verbose=False,
    )
    if use_constraints:
        constraints = []
        for f in FEATURES:
            if f in MONOTONE_UP:
                constraints.append(1)
            elif f in MONOTONE_DOWN:
                constraints.append(-1)
            else:
                constraints.append(0)
        kwargs["monotone_constraints"] = constraints
    model = CatBoostClassifier(**kwargs)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    p = model.predict_proba(valid_pool)[:, 1]
    y = valid_df["control_success"].to_numpy()
    brier = float(np.mean((p - y) ** 2))
    return score(brier, float(y.mean()))


def rolling_oot(df, use_constraints: bool) -> dict:
    per_fold = {}
    weighted_score = 0.0
    for valid_season, weight in ROLLING_FOLDS:
        train_df, valid_df = time_split(df, valid_season)
        s = fit_and_score(train_df, valid_df, use_constraints)
        per_fold[valid_season] = round(s, 2)
        weighted_score += weight * s
    per_fold["weighted"] = round(weighted_score, 2)
    return per_fold


def main():
    df = load("train.csv")
    print("===== baseline (제약 없음) =====")
    base = rolling_oot(df, use_constraints=False)
    for k, v in base.items():
        print(k, v)

    print("\n===== +monotone_constraints =====")
    mono = rolling_oot(df, use_constraints=True)
    for k, v in mono.items():
        print(k, v)

    print("\n===== 비교 =====")
    print("baseline weighted:", base["weighted"], " monotone weighted:", mono["weighted"])
    print("차이:", round(mono["weighted"] - base["weighted"], 2))


if __name__ == "__main__":
    main()
