"""
sequence_lstm.py 개선판: dropout, BatchNorm, L2(weight_decay 조정), L1(수동), ReduceLROnPlateau,
완전한 시드 고정(재현성 확보). 학습 끝나면 permutation importance로 어떤 입력이 실제로 기여하는지 확인
(LSTM은 CatBoost 같은 native importance가 없어서, 각 입력을 섞었을 때 Brier가 얼마나 나빠지는지로 대체).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from sequence_lstm import build_sequences, K  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load, TRAIN_TEST_NUMERIC  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
L1_LAMBDA = 1e-5


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SeqMLP(nn.Module):
    def __init__(self, n_numeric, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True, dropout=0.0)
        self.lstm_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(16 + n_numeric, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq.unsqueeze(-1))
        h = self.lstm_dropout(h.squeeze(0))
        x = torch.cat([h, static], dim=1)
        return self.head(x).squeeze(-1)


def brier_score(y, p):
    return float(np.mean((p - y) ** 2))


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def main():
    set_seed(SEED)
    df = load("train.csv")
    idx_order, seq = build_sequences(df)
    df = df.loc[idx_order].reset_index(drop=True)

    static_cols = list(TRAIN_TEST_NUMERIC)
    static_raw = df[static_cols].fillna(df[static_cols].median()).to_numpy().astype(np.float32)

    from sklearn.preprocessing import StandardScaler
    train_mask = (df["season"] < 2024).to_numpy()
    valid_mask = (df["season"] == 2024).to_numpy()
    scaler = StandardScaler().fit(static_raw[train_mask])
    static = scaler.transform(static_raw).astype(np.float32)
    y = df["control_success"].to_numpy().astype(np.float32)

    model = SeqMLP(len(static_cols), dropout=0.2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)  # L2 강화(1e-5 -> 1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()

    seq_t = torch.tensor(seq, dtype=torch.float32)
    static_t = torch.tensor(static, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)
    seq_va, static_va = seq_t[valid_idx].to(DEVICE), static_t[valid_idx].to(DEVICE)
    y_va = y[valid_idx]
    r = y_va.mean()

    batch_size = 8192
    best_brier, best_state, best_epoch, no_improve, patience = float("inf"), None, 0, 0, 4
    rng = np.random.default_rng(SEED)

    for epoch in range(20):
        model.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(seq_t[batch_idx].to(DEVICE), static_t[batch_idx].to(DEVICE))
            loss = loss_fn(logits, y_t[batch_idx].to(DEVICE))
            l1 = sum(p.abs().sum() for p in model.parameters())
            loss = loss + L1_LAMBDA * l1
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(seq_va, static_va)).cpu().numpy()
        brier = brier_score(y_va, p_va)
        sched.step(brier)
        print(f"epoch {epoch+1}: brier={brier:.6f} score={score(brier, r):.2f} lr={opt.param_groups[0]['lr']:.2e}")
        if brier < best_brier:
            best_brier, best_epoch, no_improve = brier, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print("early stop")
                break

    model.load_state_dict(best_state)
    print(f"\n최고: epoch {best_epoch}, score={score(best_brier, r):.2f}")
    print("(비교 기준: baseline 734.49, v1 LSTM 664~732, 기존 tabular NN 491.78)")

    # permutation importance: static 피처 하나씩 셔플 + 시퀀스 전체 셔플
    model.eval()
    with torch.no_grad():
        base_p = torch.sigmoid(model(seq_va, static_va)).cpu().numpy()
    base_brier = brier_score(y_va, base_p)

    print("\npermutation importance (섞었을 때 brier 악화량, 클수록 중요):")
    rng2 = np.random.default_rng(0)
    results = []
    for i, col in enumerate(static_cols):
        static_perm = static_va.clone()
        idx_shuffled = torch.tensor(rng2.permutation(static_perm.shape[0]))
        static_perm[:, i] = static_perm[idx_shuffled, i]
        with torch.no_grad():
            p_perm = torch.sigmoid(model(seq_va, static_perm)).cpu().numpy()
        results.append((col, brier_score(y_va, p_perm) - base_brier))

    seq_perm = seq_va.clone()
    idx_shuffled = torch.tensor(rng2.permutation(seq_perm.shape[0]))
    seq_perm = seq_perm[idx_shuffled]
    with torch.no_grad():
        p_perm = torch.sigmoid(model(seq_perm, static_va)).cpu().numpy()
    results.append(("[LSTM 시퀀스 전체]", brier_score(y_va, p_perm) - base_brier))

    for col, delta in sorted(results, key=lambda x: -x[1]):
        print(f"  {col}: +{delta:.6f}")


if __name__ == "__main__":
    main()
