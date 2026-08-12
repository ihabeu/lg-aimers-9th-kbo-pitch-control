"""
team_matchup / count_state rolling OOT 재검증 (fixed-iteration, hand_matchup_rolling.py와 동일 방식).

기존엔 single-split(2019-23→24)만 봤는데, hand_matchup에서 방법론 버그(폴드별 독립 조기종료)를 고친 김에
나머지 matchup 피처들도 fixed-iteration rolling OOT로 다시 확인한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)
BASE_CAT_FEATURES = list(bc.CAT_FEATURES)


def add_matchup_features(df):
    df = df.copy()
    df["team_matchup"] = df["pitcher_team_id"].astype(str) + "_" + df["batter_team_id"].astype(str)
    df["count_state"] = df["balls_before"].astype(str) + "_" + df["strikes_before"].astype(str)
    return df


def run(df, extra_feature, label):
    if extra_feature:
        bc.FEATURES = BASE_FEATURES + [extra_feature]
        bc.CAT_FEATURES = BASE_CAT_FEATURES + [extra_feature]
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
    df = add_matchup_features(load("train.csv"))
    run(df, "team_matchup", "+team_matchup")
    run(df, "count_state", "+count_state")


if __name__ == "__main__":
    main()
