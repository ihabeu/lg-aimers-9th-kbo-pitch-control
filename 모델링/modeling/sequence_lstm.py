"""
투수별 진짜 시간순 시퀀스(row_id로 확인됨)를 LSTM에 태워서 "최근 투구 성공/실패 흐름"이라는,
기존 asof_prev1/3/5_game_success_rate(경기 단위 최근성)와는 다른 투구 단위 최근성을 학습시킨다.

입력: 각 투구 직전까지 이 투수의 최근 K=20개 투구의 성공여부(0/1) 시퀀스 (짧으면 왼쪽 0-padding +
마스크). LSTM으로 이 시퀀스를 인코딩한 hidden state를 baseline 44피처와 concat해서 MLP로 최종 확률
예측. row_id 기준으로 leak-safe(현재 투구 이전 것만 사용).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load, TRAIN_TEST_NUMERIC, TRAIN_TEST_BINARY, TRAIN_TEST_CATEGORICAL  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
K = 20
SEED = 42


def build_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n_rows, K) 배열. 각 행은 그 투수의 직전 K개 투구 성공여부(과거->최근 순), 부족하면 앞을 -1로 패딩.
    pandas groupby().shift()로 벡터화 (K개 lag 컬럼을 한 번에 계산 -> 파이썬 반복문 없음)."""
    df = df.copy()
    df["row_num"] = df["row_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("row_num")

    g = df.groupby("pitcher_id")["control_success"]
    # lag_1 = 바로 직전 투구, lag_K = K번째 이전 투구
    lags = np.stack([g.shift(k).to_numpy() for k in range(K, 0, -1)], axis=1)  # (n, K), 왼쪽=과거, 오른쪽=최근
    seq = np.where(np.isnan(lags), -1.0, lags).astype(np.float32)
    return df.index.to_numpy(), seq


class SeqMLP(nn.Module):
    def __init__(self, n_numeric, n_binary_cat):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(16 + n_numeric, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq.unsqueeze(-1))
        h = h.squeeze(0)
        x = torch.cat([h, static], dim=1)
        return self.head(x).squeeze(-1)


def main():
    df = load("train.csv")
    idx_order, seq = build_sequences(df)
    df = df.loc[idx_order].reset_index(drop=True)

    static_cols = [c for c in TRAIN_TEST_NUMERIC]  # 수치형만 (범주형은 이 빠른 실험에서 생략)
    static = df[static_cols].fillna(df[static_cols].median()).to_numpy().astype(np.float32)
    from sklearn.preprocessing import StandardScaler

    train_mask = (df["season"] < 2024).to_numpy()
    valid_mask = (df["season"] == 2024).to_numpy()

    scaler = StandardScaler().fit(static[train_mask])
    static = scaler.transform(static).astype(np.float32)

    y = df["control_success"].to_numpy().astype(np.float32)

    torch.manual_seed(SEED)
    model = SeqMLP(len(static_cols), 0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    seq_t = torch.tensor(seq, dtype=torch.float32)
    static_t = torch.tensor(static, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)

    batch_size = 8192
    best_brier, best_epoch, no_improve, patience = float("inf"), 0, 0, 3

    seq_va = seq_t[valid_idx].to(DEVICE)
    static_va = static_t[valid_idx].to(DEVICE)
    y_va = y[valid_idx]

    for epoch in range(15):
        model.train()
        perm = np.random.permutation(train_idx)
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(seq_t[batch_idx].to(DEVICE), static_t[batch_idx].to(DEVICE))
            loss = loss_fn(logits, y_t[batch_idx].to(DEVICE))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(seq_va, static_va)).cpu().numpy()
        brier = float(np.mean((p_va - y_va) ** 2))
        r = y_va.mean()
        score = max(0.0, 100000 * (1 - brier / (r * (1 - r))))
        print(f"epoch {epoch+1}: brier={brier:.6f} score={score:.2f}")
        if brier < best_brier:
            best_brier, best_epoch, no_improve = brier, epoch + 1, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("early stop")
                break

    print(f"\n최고: epoch {best_epoch}, brier={best_brier:.6f} score={max(0.0, 100000*(1-best_brier/(r*(1-r)))):.2f}")
    print("(비교 기준: baseline 734.49, 기존 tabular NN 491.78)")

    # best_epoch 재현이 아니라 마지막 epoch 예측으로 근사 상관관계만 우선 확인(품질보다 신호유무 확인 목적)
    train_df, valid_df = bc.time_split(df, 2024)
    cat_model = bc.train_catboost(train_df, valid_df)
    p_cat = cat_model.predict_proba(bc.to_pool(valid_df, with_label=False))[:, 1]
    corr = np.corrcoef(y_va - p_va, y_va - p_cat)[0, 1]
    print(f"\ncorr(CatBoost residual, LSTM residual, 마지막 epoch 기준) = {corr:.4f}")


if __name__ == "__main__":
    main()
