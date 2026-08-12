"""
789.23 champion CatBoost + game_type(R/F) segment별 residual corrector 제출용 추론 스크립트.

data/test.csv를 읽어 champion(model/catboost_baseline.cbm)으로 base 확률을 구하고,
model/correctors.joblib(train.csv로 오프라인 학습한 ExtraTrees 6개 + 카테고리 인코딩 맵)로
segment별 보정치를 더한다. submit.zip 규정상 이 파일은 model/, requirements.txt와만 같이 있어야
해서 baseline_catboost.py를 import하지 않고 FEATURES/CAT_FEATURES를 그대로 복사해 자체완결적으로 둔다
(submit/script.py와 동일한 이유).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = Path(__file__).resolve().parent
CHAMPION_MODEL_PATH = SCRIPT_DIR / "model" / "catboost_baseline.cbm"
CORRECTORS_PATH = SCRIPT_DIR / "model" / "correctors.joblib"
OUTPUT_DIR = Path("output")
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


def corrector_matrix(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        values = x[c].astype("string").fillna("<NA>")
        x[c] = values.map(category_maps[c]).fillna(-1).astype(np.float32)
    return x.apply(pd.to_numeric, errors="coerce")


def assign_segment(df: pd.DataFrame, hybrid_team_id: int) -> np.ndarray:
    involves_hybrid = (df["pitcher_team_id"] == hybrid_team_id) | (df["batter_team_id"] == hybrid_team_id)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def main() -> None:
    test_df = pd.read_csv(find_test_csv())

    champion = CatBoostClassifier()
    champion.load_model(str(CHAMPION_MODEL_PATH))
    base_pred = champion.predict_proba(Pool(test_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]

    artifacts = joblib.load(CORRECTORS_PATH)
    X = corrector_matrix(test_df, artifacts["category_maps"])
    segment = assign_segment(test_df, artifacts["hybrid_team_id"])

    correction = np.zeros(len(test_df), dtype=np.float64)
    for seed in artifacts["seeds"]:
        seed_correction = np.zeros(len(test_df), dtype=np.float64)
        for seg in artifacts["segments"]:
            mask = segment == seg
            if not mask.any():
                continue
            seed_correction[mask] = artifacts["correctors"][(seg, seed)].predict(X.loc[mask])
        correction += seed_correction / len(artifacts["seeds"])

    final = np.clip(base_pred + correction, 0, 1)
    assert np.isfinite(final).all() and ((final >= 0) & (final <= 1)).all()

    submission = pd.DataFrame({"row_id": test_df["row_id"], "control_success": final})
    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"saved {len(submission)} rows to {OUTPUT_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
