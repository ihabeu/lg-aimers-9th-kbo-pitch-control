"""
hand_matchup rolling OOT 재검증 (fixed-iteration 방식으로 교정).

hand_matchup_oot.py는 폴드마다 독립적으로 조기종료시켰는데, baseline_catboost.rolling_oot_evaluate()의
docstring에 이미 경고돼 있듯 이 방식은 2023 폴드를 무너뜨려서 공정한 비교가 안 된다. primary(2019-23→24)
조기종료로 iterations를 고정한 뒤 rolling_oot_evaluate_fixed()로 baseline과 나란히 비교한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)
BASE_CAT_FEATURES = list(bc.CAT_FEATURES)


def add_hand_matchup(df):
    df = df.copy()
    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    return df


def run(df, with_hand_matchup, label):
    if with_hand_matchup:
        bc.FEATURES = BASE_FEATURES + ["hand_matchup"]
        bc.CAT_FEATURES = BASE_CAT_FEATURES + ["hand_matchup"]
    else:
        bc.FEATURES, bc.CAT_FEATURES = BASE_FEATURES, BASE_CAT_FEATURES

    train_df, valid_df = bc.time_split(df, 2024)
    primary = bc.train_catboost(train_df, valid_df)
    iterations = primary.get_best_iteration() + 1
    print(f"{label}: primary(2024) best_iteration={iterations}")

    fold_results = bc.rolling_oot_evaluate_fixed(df, iterations)
    for season, m in fold_results.items():
        print(f"  {season}: {m}")


def main():
    df = add_hand_matchup(load("train.csv"))
    run(df, False, "baseline")
    run(df, True, "+hand_matchup")


if __name__ == "__main__":
    main()
