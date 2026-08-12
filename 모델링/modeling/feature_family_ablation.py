"""
STEP 3: 44개 feature를 의미 단위 family로 나눠서 통째로 제거 (단순 importance-N개 제거와 다름 —
이미 bottom5/10/15 제거는 해봤고 단조롭게 나빠져서 44개 유지가 확정됨, HANDOFF.md 참고).
각 family를 제거했을 때 baseline(734.49) 대비 얼마나 떨어지는지로 "이 정보 축이 실제로 얼마나
필요한가"를 본다. single-split 2019-23->24.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

FULL_FEATURES = list(bc.FEATURES)
FULL_CAT_FEATURES = list(bc.CAT_FEATURES)

FAMILIES = {
    "pitcher_asof": [
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ],
    "batter_asof": ["asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"],
    "count_situation": [
        "balls_before", "strikes_before", "outs_before", "num_runners_on",
        "run_top_before", "run_bot_before", "run_total_before",
        "score_diff_home", "score_diff_pitcher_team",
        "home_win_expectancy", "away_win_expectancy", "li",
        "runner_on_1b", "runner_on_2b", "runner_on_3b", "base_state",
    ],
    "time": ["season", "inning", "game_month", "game_dayofweek"],
    "categorical_identity": ["top_bottom", "game_type", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"],
}


def run(df, drop, label):
    features = [f for f in FULL_FEATURES if f not in drop]
    cat_features = [f for f in FULL_CAT_FEATURES if f not in drop]
    bc.FEATURES, bc.CAT_FEATURES = features, cat_features
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label} ({len(features)}개 유지): score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    df = load("train.csv")
    total = sum(len(v) for v in FAMILIES.values())
    print(f"family 총 피처수 확인: {total} (44와 일치해야 함)\n")
    for name, cols in FAMILIES.items():
        run(df, cols, f"- {name}({len(cols)}개) 제거")


if __name__ == "__main__":
    main()
