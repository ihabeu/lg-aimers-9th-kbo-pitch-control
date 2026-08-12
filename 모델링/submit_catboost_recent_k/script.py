"""
lg aimers 제출용 추론 스크립트 (CatBoost + recent_k_pitch_rate 버전).

data/test.csv (평가 서버가 245,789행 실데이터로 교체) 를 읽어 각 행의 control_success 확률을 예측하고
output/submission.csv (row_id, control_success) 로 저장한다.

recent_{5,10,20,50}_pitch_rate: 투수별 "최근 K구 성공률" — train.csv에서 row_id가 실제 시간순
인덱스임을 확인해서 만든 leak-safe 피처 (2019~2024 시즌 내내 이전 투구만으로 계산, 경기 단위가 아니라
투구 단위 최근성이라 기존 asof_prev*_game_success_rate와는 다른 정보).

규정상 test.csv 내부 행 순서 기반 rolling/expanding feature는 금지라서, 실제 제출 시에는 각 투수의
train.csv 마지막 시점(2024 시즌 말)까지의 값을 하나로 고정(freeze)해서 model/recent_k_snapshot.csv에
저장해두고, 그 투수의 모든 2025 test 행에 동일한 값을 적용한다. train.csv에 없던 투수(2025 신인 등)는
결측 -> CatBoost가 네이티브로 처리.

로컬 검증(rolling OOT, fixed-iteration): baseline 2024=734.49 -> +recent_k 966.18(+231.69),
weighted 818.65 -> 1000.99(+182.34). 2022 폴드도 같은 방향으로 개선(+332.46) — 이 세션 최대 개선폭.
"""
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model" / "catboost_recent_k.cbm"
SNAPSHOT_PATH = SCRIPT_DIR / "model" / "recent_k_snapshot.csv"
OUTPUT_DIR = Path("output")

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

KS = [5, 10, 20, 50]
RECENT_K_FEATURES = [f"recent_{k}_pitch_rate" for k in KS]
FEATURES = BASE_FEATURES + RECENT_K_FEATURES
CAT_FEATURES = BASE_CAT_FEATURES


def add_recent_k_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = pd.read_csv(SNAPSHOT_PATH)
    return df.merge(snapshot, on="pitcher_id", how="left")


def find_test_csv() -> Path:
    for p in CANDIDATE_TEST_PATHS:
        if p.exists():
            return p
    tried = ", ".join(str(p) for p in CANDIDATE_TEST_PATHS)
    raise FileNotFoundError(f"test.csv를 찾을 수 없음 (시도한 경로: {tried})")


def main() -> None:
    test_df = add_recent_k_snapshot(pd.read_csv(find_test_csv()))

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
