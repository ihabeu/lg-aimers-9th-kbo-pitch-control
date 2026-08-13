"""
lg aimers 제출용 추론 스크립트.

data/test.csv (평가 서버가 245,789행 실데이터로 교체) 를 읽어 각 행의 control_success 확률을 예측하고
output/submission.csv (row_id, control_success) 로 저장한다.

이 파일은 submit.zip 안에서 model/, requirements.txt와만 같이 있고 외부 모듈(eda.py 등)을 import할 수 없어서
피처 목록을 그대로 복사해서 자체완결적으로 둔다 (baseline_catboost.py의 FEATURES/CAT_FEATURES와 동일해야 함).
"""
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model" / "catboost_baseline.cbm"
OUTPUT_DIR = Path("output")

# 규정 문서상 데이터 경로 표기가 "data/"(구조도)와 "open/"(유의사항)로 서로 다르게 적혀 있어 둘 다 시도한다.
CANDIDATE_TEST_PATHS = [Path("data/test.csv"), Path("open/test.csv"), Path("test.csv")]

FEATURES = [
    "balls_before", "strikes_before", "outs_before", "num_runners_on",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "home_win_expectancy", "away_win_expectancy", "li",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "season", "inning",
    "top_bottom", "game_type", "pitcher_hand", "batter_hand",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "base_state", "pitcher_team_id", "batter_team_id",
    "game_month", "game_dayofweek",
]
CAT_FEATURES = [
    "top_bottom", "game_type", "pitcher_hand", "batter_hand",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "base_state", "pitcher_team_id", "batter_team_id",
    "game_month", "game_dayofweek",
]


def find_test_csv() -> Path:
    for p in CANDIDATE_TEST_PATHS:
        if p.exists():
            return p
    tried = ", ".join(str(p) for p in CANDIDATE_TEST_PATHS)
    raise FileNotFoundError(f"test.csv를 찾을 수 없음 (시도한 경로: {tried})")


def main() -> None:
    test_df = pd.read_csv(find_test_csv())

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    pool = Pool(test_df[FEATURES], cat_features=CAT_FEATURES)
    proba = model.predict_proba(pool)[:, 1]

    submission = pd.DataFrame({"row_id": test_df["row_id"], "control_success": proba})

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"saved {len(submission)} rows to {OUTPUT_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
