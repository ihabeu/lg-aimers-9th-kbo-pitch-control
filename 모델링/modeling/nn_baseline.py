"""
Entity-embedding MLP. CatBoost를 대체할 목적이 아니라, 성격이 다른 세 번째 모델을 만들어서
나중에 CatBoost와 블렌드했을 때 도움이 되는지 보려는 것 (오늘 실패한 9개는 전부 "CatBoost 하나를
바꾸는" 시도였고, 이건 그것과 다른 카테고리).

범주형은 임베딩(원핫 대신), 수치형은 StandardScaler, 결측치는 중앙값 대체 + 그룹 플래그
(elastic_net.py와 동일한 3그룹 압축 방식) — NN은 결측치를 못 받아서 CatBoost처럼 네이티브 처리가 안 됨.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import TRAIN_TEST_NUMERIC, TRAIN_TEST_BINARY, TRAIN_TEST_CATEGORICAL, TARGET, load  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42  # 시드 미고정 상태로 두 번 돌렸더니 601.12 / 519.18로 편차가 커서(NN이 트리보다 불안정) 고정함
LR = 2e-3  # 학습률 스윕(1e-3=435.57, 5e-4=500.74, 2e-3=611.89)에서 채택

TIME_SERIES_NUMERIC = ["season", "inning"]
TIME_SERIES_CATEGORICAL = ["game_month", "game_dayofweek"]

NUMERIC_FEATURES = TRAIN_TEST_NUMERIC + TIME_SERIES_NUMERIC
EMBED_FEATURES = TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL + TIME_SERIES_CATEGORICAL
MISSING_FLAGS = ["is_pitcher_coldstart", "is_batter_coldstart", "is_missing_recent_games"]


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_pitcher_coldstart"] = df["asof_pitcher_n"].eq(0).astype(int)
    df["is_batter_coldstart"] = df["asof_batter_success_rate"].isna().astype(int)
    df["is_missing_recent_games"] = df["asof_pitcher_prev1_game_success_rate"].isna().astype(int)
    return df


def time_split(df, valid_season=2024):
    return df[df["season"] < valid_season], df[df["season"] == valid_season]


# rolling out-of-time 폴드: (검증 연도, 가중치). baseline_catboost.py와 동일 기준(이전 실험 방식).
ROLLING_FOLDS = [(2022, 0.2), (2023, 0.3), (2024, 0.5)]


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, num_x, cat_x, y):
        self.num_x = torch.tensor(num_x, dtype=torch.float32)
        self.cat_x = torch.tensor(cat_x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.num_x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.num_x[idx], self.cat_x[idx], self.y[idx]
        return self.num_x[idx], self.cat_x[idx]


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


class Preprocessor:
    """fit은 train에서만, transform은 아무 df에나 (누수 방지 — elastic_net.py 때 겪은 실수 반복 안 함)."""

    def fit(self, train_df):
        self.medians = train_df[NUMERIC_FEATURES].median()
        self.scaler = StandardScaler().fit(train_df[NUMERIC_FEATURES].fillna(self.medians))
        self.cat_maps = {}
        for c in EMBED_FEATURES:
            uniques = sorted(train_df[c].astype(str).unique())
            self.cat_maps[c] = {v: i + 1 for i, v in enumerate(uniques)}  # 0 = 학습때 못 본 값(unknown)
        return self

    def transform(self, df):
        num_x = self.scaler.transform(df[NUMERIC_FEATURES].fillna(self.medians)).astype(np.float32)
        cat_x = np.zeros((len(df), len(EMBED_FEATURES)), dtype=np.int64)
        for i, c in enumerate(EMBED_FEATURES):
            cat_x[:, i] = df[c].astype(str).map(self.cat_maps[c]).fillna(0).astype(np.int64)
        return num_x, cat_x

    def cardinalities(self):
        return [len(m) for m in self.cat_maps.values()]


def brier_score(y_true, p):
    return float(np.mean((p - y_true) ** 2))


def train_nn(train_df, valid_df, epochs=30, batch_size=4096, lr=LR, patience=5, seed=SEED):
    torch.manual_seed(seed)
    prep = Preprocessor().fit(train_df)
    num_tr, cat_tr = prep.transform(train_df)
    num_va, cat_va = prep.transform(valid_df)
    y_tr = train_df[TARGET].to_numpy().astype(np.float32)
    y_va = valid_df[TARGET].to_numpy().astype(np.float32)

    train_ds = TabularDataset(num_tr, cat_tr, y_tr)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = EmbeddingMLP(len(NUMERIC_FEATURES), prep.cardinalities()).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    num_va_t = torch.tensor(num_va, dtype=torch.float32).to(DEVICE)
    cat_va_t = torch.tensor(cat_va, dtype=torch.long).to(DEVICE)

    best_brier = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for num_x, cat_x, y in train_loader:
            num_x, cat_x, y = num_x.to(DEVICE), cat_x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(num_x, cat_x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(num_va_t, cat_va_t)).cpu().numpy()
        b = brier_score(y_va, p_va)
        print(f"epoch {epoch+1}: valid brier={b:.6f}")

        if b < best_brier:
            best_brier = b
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, prep, best_brier, best_epoch


def train_nn_full(df, n_epochs, batch_size=4096, lr=LR, seed=SEED):
    """실제 제출용. 검증 없이 2019~2024 전체로, 검증에서 찾은 epoch 수만큼 고정 학습.
    (baseline_catboost.py의 train_final_full과 동일한 패턴 — 최근 시즌을 반드시 포함시켜야 함)."""
    torch.manual_seed(seed)
    prep = Preprocessor().fit(df)
    num_x, cat_x = prep.transform(df)
    y = df[TARGET].to_numpy().astype(np.float32)

    ds = TabularDataset(num_x, cat_x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = EmbeddingMLP(len(NUMERIC_FEATURES), prep.cardinalities()).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        model.train()
        for num_b, cat_b, y_b in loader:
            num_b, cat_b, y_b = num_b.to(DEVICE), cat_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(num_b, cat_b), y_b)
            loss.backward()
            opt.step()
        print(f"[전체데이터 재학습] epoch {epoch+1}/{n_epochs}")

    return model, prep


def evaluate(model, prep, valid_df):
    from sklearn.metrics import roc_auc_score

    num_va, cat_va = prep.transform(valid_df)
    model.eval()
    with torch.no_grad():
        num_va_t = torch.tensor(num_va, dtype=torch.float32).to(DEVICE)
        cat_va_t = torch.tensor(cat_va, dtype=torch.long).to(DEVICE)
        p = torch.sigmoid(model(num_va_t, cat_va_t)).cpu().numpy()

    y = valid_df[TARGET].to_numpy()
    r = y.mean()
    b = brier_score(y, p)
    score = max(0.0, 100000 * (1 - b / (r * (1 - r))))
    return {"n": len(valid_df), "r": round(r, 4), "brier": round(b, 6), "score": round(score, 2),
            "auc": round(roc_auc_score(y, p), 4)}


def rolling_oot_evaluate(df, n_epochs, folds=ROLLING_FOLDS, lr=LR, seed=SEED):
    """모든 폴드를 같은 epoch 수로 고정 학습해서 공정 비교 (baseline_catboost.py의
    rolling_oot_evaluate_fixed와 동일 패턴 — 폴드별 독립 조기종료는 2023에서 CatBoost도 무너졌던 적 있음)."""
    per_fold = {}
    weighted_brier = weighted_score = 0.0
    for valid_season, weight in folds:
        train_sub, valid_sub = time_split(df, valid_season)
        model, prep = train_nn_full(train_sub, n_epochs, lr=lr, seed=seed)
        m = evaluate(model, prep, valid_sub)
        m["weight"] = weight
        per_fold[valid_season] = m
        weighted_brier += weight * m["brier"]
        weighted_score += weight * m["score"]
    per_fold["weighted"] = {"brier": round(weighted_brier, 6), "score": round(weighted_score, 2)}
    return per_fold


def main():
    import joblib

    MODEL_DIR.mkdir(exist_ok=True)
    df = add_missing_flags(load("train.csv"))
    train_df, valid_df = time_split(df, 2024)
    print(f"train={len(train_df):,} valid={len(valid_df):,} device={DEVICE}")

    model, prep, best_brier, best_epoch = train_nn(train_df, valid_df)
    metrics = evaluate(model, prep, valid_df)
    print("\n[검증 결과 — 2019~2023 학습 / 2024 검증]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  best_epoch: {best_epoch}")
    print("(비교: CatBoost baseline = 734.49)")

    # 실제 제출용: 검증에서 찾은 epoch 수로 2019~2024 전체 재학습 (2024를 빼먹으면 안 됨 — CatBoost 첫 제출 실수 반복 금지)
    print(f"\n[최종 모델] 2019~2024 전체 데이터, epochs={best_epoch}으로 재학습")
    final_model, final_prep = train_nn_full(df, best_epoch)

    torch.save({
        "state_dict": final_model.state_dict(),
        "cat_cardinalities": final_prep.cardinalities(),
        "num_numeric": len(NUMERIC_FEATURES),
    }, MODEL_DIR / "nn_baseline.pt")
    joblib.dump(
        {"scaler": final_prep.scaler, "medians": final_prep.medians, "cat_maps": final_prep.cat_maps},
        MODEL_DIR / "nn_preprocessor.joblib",
    )
    print(f"saved final (submission) model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
