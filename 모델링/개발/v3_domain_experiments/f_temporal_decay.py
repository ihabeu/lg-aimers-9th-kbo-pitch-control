"""
F(퓨처스) 성공률이 시즌 내에서도 계속 하락한다는 패턴(자체 EDA로 재확인: 2022/2023년 4월→9월
단조 하락, 2024년은 덜 깨끗함)이 CatBoost baseline에 도움되는지 독립적으로 테스트.
아이디어 출처는 참고 자료(B0_Readme)지만 구현은 우리 baseline_catboost.py 인프라 위에서 새로
작성 -- 코드/아키텍처를 그대로 가져오지 않음.

피처: game_type='F'인 행에 한해 "이 시즌의 몇 번째 달인지"(4월=0, 5월=1, ...)를 넣는다.
R행은 0으로 둔다(원래도 이 패턴이 F에만 있었으므로).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import ROLLING_FOLDS, FEATURES, CAT_FEATURES, L2_LEAF_REG, time_split  # noqa: E402


def add_f_season_progress(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    season_start_month = out.groupby("season")["game_month"].transform("min")
    progress = (out["game_month"] - season_start_month).clip(lower=0)
    out["f_season_progress"] = (progress * (out["game_type"] == "F")).astype(int)
    return out


def score(brier: float, r: float) -> float:
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def fit_and_score(train_df, valid_df, features, cat_features):
    train_pool = Pool(train_df[features], train_df["control_success"], cat_features=cat_features)
    valid_pool = Pool(valid_df[features], valid_df["control_success"], cat_features=cat_features)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
        eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    p = model.predict_proba(valid_pool)[:, 1]
    y = valid_df["control_success"].to_numpy()
    brier = float(np.mean((p - y) ** 2))
    return score(brier, float(y.mean()))


def rolling_oot_with_feature(df: pd.DataFrame, features, cat_features) -> dict:
    per_fold = {}
    weighted_score = 0.0
    for valid_season, weight in ROLLING_FOLDS:
        train_df, valid_df = time_split(df, valid_season)
        s = fit_and_score(train_df, valid_df, features, cat_features)
        per_fold[valid_season] = round(s, 2)
        weighted_score += weight * s
    per_fold["weighted"] = round(weighted_score, 2)
    return per_fold


def main():
    df = load("train.csv")
    df_f = add_f_season_progress(df)

    print("===== baseline (44피처, f_season_progress 없음) =====")
    base_result = rolling_oot_with_feature(df, FEATURES, CAT_FEATURES)
    for k, v in base_result.items():
        print(k, v)

    print("\n===== +f_season_progress =====")
    new_features = FEATURES + ["f_season_progress"]
    new_result = rolling_oot_with_feature(df_f, new_features, CAT_FEATURES)
    for k, v in new_result.items():
        print(k, v)

    print("\n===== 비교 =====")
    print("baseline weighted:", base_result["weighted"], " +f_season_progress weighted:", new_result["weighted"])
    print("차이:", round(new_result["weighted"] - base_result["weighted"], 2))


if __name__ == "__main__":
    main()
