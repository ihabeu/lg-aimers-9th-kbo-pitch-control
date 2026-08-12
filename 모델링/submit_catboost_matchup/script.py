"""
lg aimers 제출용 추론 스크립트 (CatBoost + matchup 파생 피처 3종 버전).

data/test.csv (평가 서버가 245,789행 실데이터로 교체) 를 읽어 각 행의 control_success 확률을 예측하고
output/submission.csv (row_id, control_success) 로 저장한다.

이 파일은 submit.zip 안에서 model/, requirements.txt와만 같이 있고 외부 모듈을 import할 수 없어서
피처 목록/파생 로직을 그대로 복사해서 자체완결적으로 둔다 (modeling/catboost_matchup.py와 동일해야 함).

로컬 검증(2019-23→24 single-split): score 703.10 — baseline(789.23, submit/)보다 낮지만,
로컬/실제 리더보드 점수가 이번 대회에서 여러 번 어긋난 전례가 있어 별도로 제출 테스트한다.
"""
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model" / "catboost_matchup.cbm"
OUTPUT_DIR = Path("output")

# 규정 문서상 데이터 경로 표기가 "data/"(구조도)와 "open/"(유의사항)로 서로 다르게 적혀 있어 둘 다 시도한다.
CANDIDATE_TEST_PATHS = [Path("data/test.csv"), Path("open/test.csv"), Path("test.csv")]

BASE_FEATURES = [
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
BASE_CAT_FEATURES = [
    "top_bottom", "game_type", "pitcher_hand", "batter_hand",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "base_state", "pitcher_team_id", "batter_team_id",
    "game_month", "game_dayofweek",
]

MATCHUP_FEATURES = ["team_matchup", "hand_matchup", "count_state"]
FEATURES = BASE_FEATURES + MATCHUP_FEATURES
CAT_FEATURES = BASE_CAT_FEATURES + MATCHUP_FEATURES


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["team_matchup"] = df["pitcher_team_id"].astype(str) + "_" + df["batter_team_id"].astype(str)
    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    df["count_state"] = df["balls_before"].astype(str) + "_" + df["strikes_before"].astype(str)
    return df


def find_test_csv() -> Path:
    for p in CANDIDATE_TEST_PATHS:
        if p.exists():
            return p
    tried = ", ".join(str(p) for p in CANDIDATE_TEST_PATHS)
    raise FileNotFoundError(f"test.csv를 찾을 수 없음 (시도한 경로: {tried})")


def main() -> None:
    test_df = add_matchup_features(pd.read_csv(find_test_csv()))

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
