"""
test.csv 5행이 전부 game_type='R'(1군)이라는 사실 확인 후, 학습에서 game_type='F'(2군) 행을
아예 제거하면 R 예측이 더 좋아지는지 검증.

F는 2023부터 성공률이 약 0.70->0.47로 급락하는 별도 regime이라(EDA 확인), 같은 트리에서
R/F를 같이 학습시키는 게 R 쪽 분할을 오염시킬 가능성이 있음. 비교는 항상 R행만으로 평가
(실제 채점 대상과 동일 분포), rolling OOT(2022/23/24, 0.2/0.3/0.5)로 baseline_catboost.py와
동일한 방식.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
from eda import load, TARGET  # noqa: E402
from baseline_catboost import (  # noqa: E402
    ROLLING_FOLDS, time_split, train_catboost, evaluate,
)


def rolling_oot_r_only(df: pd.DataFrame, drop_f_from_train: bool) -> dict:
    per_fold = {}
    weighted_brier = 0.0
    weighted_score = 0.0
    for valid_season, weight in ROLLING_FOLDS:
        train_df, valid_df = time_split(df, valid_season)
        if drop_f_from_train:
            train_df = train_df[train_df["game_type"] == "R"]
        valid_r = valid_df[valid_df["game_type"] == "R"]
        model = train_catboost(train_df, valid_r)  # early stop on R-only valid (평가 대상과 동일 분포)
        m = evaluate(model, valid_r)
        m["weight"] = weight
        m["best_iteration"] = model.get_best_iteration()
        per_fold[valid_season] = m
        weighted_brier += weight * m["brier"]
        weighted_score += weight * m["score (리더보드 산식)"]
    per_fold["weighted"] = {"brier": round(weighted_brier, 6), "score": round(weighted_score, 2)}
    return per_fold


def main():
    df = load("train.csv")
    print("game_type value_counts:\n", df["game_type"].value_counts())

    print("\n===== A) baseline (R+F 같이 학습) -> R행만 평가 =====")
    a = rolling_oot_r_only(df, drop_f_from_train=False)
    for season, m in a.items():
        print(season, m)

    print("\n===== B) F 제거 (R만 학습) -> R행만 평가 =====")
    b = rolling_oot_r_only(df, drop_f_from_train=True)
    for season, m in b.items():
        print(season, m)

    print("\n===== 비교 =====")
    print(f"A (R+F 학습) weighted score: {a['weighted']['score']}")
    print(f"B (R only 학습) weighted score: {b['weighted']['score']}")
    print(f"차이 (B-A): {b['weighted']['score'] - a['weighted']['score']:.2f}")


if __name__ == "__main__":
    main()
