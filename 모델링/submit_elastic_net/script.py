"""
lg aimers 제출용 추론 스크립트 — Elastic Net (L1+L2) 로지스틱 회귀 버전.

주의: 로컬 검증(2019-23 학습 → 2024 검증) Brier가 베이스라인(그냥 평균 찍기)보다도 나빠서
score가 0으로 나온 모델이다. CatBoost 버전(../submit/)이 훨씬 우수하니(로컬 734.49) 실제 제출은
그쪽을 권장. 이건 비교/실험용으로 패키징한 것.

data/test.csv (평가 서버가 245,789행 실데이터로 교체) 를 읽어 각 행의 control_success 확률을 예측하고
output/submission.csv (row_id, control_success) 로 저장한다.

이 파일은 submit.zip 안에서 model/, requirements.txt와만 같이 있고 외부 모듈(eda.py, elastic_net.py 등)을
import할 수 없어서 피처 목록과 결측 플래그 로직을 그대로 복사해서 자체완결적으로 둔다.
"""
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model" / "elastic_net.joblib"
OUTPUT_DIR = Path("output")

# 규정 문서상 데이터 경로 표기가 "data/"(구조도)와 "open/"(유의사항)로 서로 다르게 적혀 있어 둘 다 시도한다.
CANDIDATE_TEST_PATHS = [Path("data/test.csv"), Path("open/test.csv"), Path("test.csv")]

NUMERIC_FEATURES = [
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "score_diff_home", "score_diff_pitcher_team", "home_win_expectancy", "li",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "season", "inning",
]
CATEGORICAL_FEATURES = [
    "top_bottom", "game_type", "pitcher_hand", "batter_hand",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "base_state", "pitcher_team_id", "batter_team_id",
    "game_month", "game_dayofweek",
]
MISSING_FLAGS = ["is_pitcher_coldstart", "is_batter_coldstart", "is_missing_recent_games"]


def find_test_csv() -> Path:
    for p in CANDIDATE_TEST_PATHS:
        if p.exists():
            return p
    tried = ", ".join(str(p) for p in CANDIDATE_TEST_PATHS)
    raise FileNotFoundError(f"test.csv를 찾을 수 없음 (시도한 경로: {tried})")


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_pitcher_coldstart"] = df["asof_pitcher_n"].eq(0).astype(int)
    df["is_batter_coldstart"] = df["asof_batter_success_rate"].isna().astype(int)
    df["is_missing_recent_games"] = df["asof_pitcher_prev1_game_success_rate"].isna().astype(int)
    return df


def main() -> None:
    test_df = add_missing_flags(pd.read_csv(find_test_csv()))

    pipe = joblib.load(MODEL_PATH)
    X = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + MISSING_FLAGS]
    proba = pipe.predict_proba(X)[:, 1]

    submission = pd.DataFrame({"row_id": test_df["row_id"], "control_success": proba})

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"saved {len(submission)} rows to {OUTPUT_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
