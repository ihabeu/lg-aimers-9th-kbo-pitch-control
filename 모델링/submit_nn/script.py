"""
lg aimers 제출용 추론 스크립트 — Entity-embedding MLP(NN) 버전.

주의: 로컬 검증(2019-23 학습 → 2024 검증) score가 CatBoost(734.49)보다 낮은 모델(약 519~601대)이다.
CatBoost 버전(../submit/)이 훨씬 우수하니 실제 제출은 그쪽을 권장. 이건 비교/블렌드 실험용 패키징.

data/test.csv 를 읽어 각 행의 control_success 확률을 예측하고 output/submission.csv로 저장한다.
외부 모듈(eda.py, nn_baseline.py 등)을 import할 수 없어서 전처리/모델 구조를 전부 복사해서 자체완결적으로 둔다.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model" / "nn_baseline.pt"
PREP_PATH = SCRIPT_DIR / "model" / "nn_preprocessor.joblib"
OUTPUT_DIR = Path("output")
CANDIDATE_TEST_PATHS = [Path("data/test.csv"), Path("open/test.csv"), Path("test.csv")]

NUMERIC_FEATURES = [
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
]
EMBED_FEATURES = [
    "top_bottom", "game_type", "pitcher_hand", "batter_hand",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "base_state", "pitcher_team_id", "batter_team_id",
    "game_month", "game_dayofweek",
]


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_pitcher_coldstart"] = df["asof_pitcher_n"].eq(0).astype(int)
    df["is_batter_coldstart"] = df["asof_batter_success_rate"].isna().astype(int)
    df["is_missing_recent_games"] = df["asof_pitcher_prev1_game_success_rate"].isna().astype(int)
    return df


class EmbeddingMLP(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, min(16, (card + 2) // 2)) for card in cat_cardinalities
        ])
        embed_dim = sum(e.embedding_dim for e in self.embeddings)
        in_dim = num_numeric + embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, num_x, cat_x):
        embeds = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([num_x] + embeds, dim=1)
        return self.net(x).squeeze(-1)


def find_test_csv() -> Path:
    for p in CANDIDATE_TEST_PATHS:
        if p.exists():
            return p
    tried = ", ".join(str(p) for p in CANDIDATE_TEST_PATHS)
    raise FileNotFoundError(f"test.csv를 찾을 수 없음 (시도한 경로: {tried})")


def main() -> None:
    test_df = add_missing_flags(pd.read_csv(find_test_csv()))

    prep = joblib.load(PREP_PATH)
    scaler, medians, cat_maps = prep["scaler"], prep["medians"], prep["cat_maps"]

    num_x = scaler.transform(test_df[NUMERIC_FEATURES].fillna(medians)).astype(np.float32)
    cat_x = np.zeros((len(test_df), len(EMBED_FEATURES)), dtype=np.int64)
    for i, c in enumerate(EMBED_FEATURES):
        cat_x[:, i] = test_df[c].astype(str).map(cat_maps[c]).fillna(0).astype(np.int64)

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = EmbeddingMLP(ckpt["num_numeric"], ckpt["cat_cardinalities"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(num_x), torch.tensor(cat_x))
        proba = torch.sigmoid(logits).numpy()

    submission = pd.DataFrame({"row_id": test_df["row_id"], "control_success": proba})

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"saved {len(submission)} rows to {OUTPUT_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
