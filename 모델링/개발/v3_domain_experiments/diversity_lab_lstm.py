"""
"Diversity Lab": LSTM. asof_* 피처는 투수 이력을 요약통계(rate)로 압축한 것인데, LSTM은 원시
투구 순서 자체(최근 W개 투구의 실제 시퀀스)에서 CatBoost/asof_*가 못 보는 순서-의존 패턴을 찾을
가능성이 있다 -- CatBoost와 다른 이유로 residual이 다를 수 있는 후보.

row_id가 실제 시간순 인덱스라는 이미 확인된 사실(HANDOFF.md)을 그대로 이용해, 투수별로 row_id
순 정렬 후 "이 투구 시점까지의 과거 W개"만 윈도우로 사용(시즌 경계를 넘어도 실제 과거 투구면
사용 가능 -- asof_* 피처와 같은 leak-safety 논리, 미래 데이터는 안 봄).

v2(2026-08-14): 1차 시도(EXPERIMENTS.md E017)는 loss가 3 epoch 동안 거의 안 움직여 학습 미흡으로
결론 보류였음. 원인으로 의심되는 것 -- 입력 피처가 스케일이 전혀 안 맞았다(0~100대의
home_win_expectancy와 0~1대의 rate, 수백 단위의 라벨인코딩 정수가 그대로 섞여 있었음). 이번엔
train 통계로만 fit한 StandardScaler를 추가하고 epoch/gradient clipping도 늘려서 재시도.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

WINDOW = 10
HIDDEN = 64
EPOCHS = 15
BATCH = 4096
GRAD_CLIP = 1.0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUMERIC_ISH = [c for c in FEATURES if c not in CAT_FEATURES]


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def build_matrix(df, category_maps, medians):
    x = df[NUMERIC_ISH].copy()
    for c in NUMERIC_ISH:
        x[c] = x[c].fillna(medians[c])
    cat = df[CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        cat[c] = df[c].astype("string").fillna("<NA>").map(category_maps[c]).fillna(-1).astype(np.float32)
    return pd_concat_to_numpy(x, cat)


def pd_concat_to_numpy(x, cat):
    import pandas as pd
    return pd.concat([x, cat], axis=1).to_numpy(np.float32)


def build_windows(df_sorted, X, window):
    """투수별로 row_id순 정렬됐다고 가정, sliding_window_view로 벡터화(투수 그룹 단위 -- 수백개라
    파이썬 루프 오버헤드는 무시할 만함, 행 단위 루프보다 훨씬 빠름)."""
    n, f = X.shape
    seqs = np.zeros((n, window, f), dtype=np.float32)
    for _, idx in df_sorted.groupby("pitcher_id", sort=False).indices.items():
        idx = np.sort(idx)
        L = len(idx)
        Xp = X[idx]
        padded = np.zeros((window - 1 + L, f), dtype=np.float32)
        padded[window - 1:] = Xp
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window, axis=0)
        seqs[idx] = np.transpose(windows, (0, 2, 1))
    return seqs


class LSTMClassifier(nn.Module):
    def __init__(self, in_features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def fit_catboost(train_df, valid_df):
    pool_tr = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES)
    pool_va = Pool(valid_df[FEATURES], valid_df["control_success"], cat_features=CAT_FEATURES)
    model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
                                eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
                                random_seed=42, thread_count=-1, verbose=False)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    return model.predict_proba(pool_va)[:, 1]


def train_lstm(seqs_tr, y_tr, seqs_va):
    model = LSTMClassifier(seqs_tr.shape[-1], HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(seqs_tr), torch.from_numpy(y_tr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += loss.item() * len(yb)
        epoch_loss = total / len(ds)
        sched.step(epoch_loss)
        lr_now = opt.param_groups[0]["lr"]
        print(f"    epoch {epoch + 1}: loss={epoch_loss:.5f}  lr={lr_now:.2e}")

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs_va), BATCH):
            xb = torch.from_numpy(seqs_va[i:i + BATCH]).to(DEVICE)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(preds)


def run_fold(df_sorted, seqs, valid_season, label):
    print(f"\n===== {label}: <{valid_season} -> {valid_season} =====")
    train_mask = (df_sorted["season"] < valid_season).to_numpy()
    valid_mask = (df_sorted["season"] == valid_season).to_numpy()
    y = df_sorted["control_success"].to_numpy(np.float64)

    # train 통계로만 채널별 표준화(leak-safe) -- v1은 이게 없어서 학습이 거의 안 됐음
    flat_tr = seqs[train_mask].reshape(-1, seqs.shape[-1])
    valid_rows = ~np.all(flat_tr == 0, axis=1)  # 패딩(전부 0)은 통계에서 제외
    mean = flat_tr[valid_rows].mean(axis=0)
    std = flat_tr[valid_rows].std(axis=0)
    std[std < 1e-6] = 1.0
    seqs_norm = (seqs - mean) / std

    p_lstm = train_lstm(seqs_norm[train_mask], y[train_mask], seqs_norm[valid_mask])
    y_va = y[valid_mask]
    bss = score(float(np.mean((p_lstm - y_va) ** 2)), float(y_va.mean()))

    train_df = df_sorted[train_mask]
    valid_df = df_sorted[valid_mask]
    p_cat = fit_catboost(train_df, valid_df)
    corr = float(np.corrcoef(p_cat, p_lstm)[0, 1])
    print(f"  LSTM(window={WINDOW}): BSS={bss:.2f}  corr(vs CatBoost)={corr:.4f}")
    return bss, corr


def main():
    df = load("train.csv")
    medians = {c: float(df[c].median()) for c in NUMERIC_ISH}
    category_maps = {}
    for c in CAT_FEATURES:
        values = df[c].astype("string").fillna("<NA>")
        category_maps[c] = {v: i for i, v in enumerate(sorted(values.unique()))}

    df_sorted = df.sort_values(["pitcher_id", "row_id"]).reset_index(drop=True)
    print("윈도우 빌드 중...")
    X = build_matrix(df_sorted, category_maps, medians)
    seqs = build_windows(df_sorted, X, WINDOW)
    print(f"seqs shape={seqs.shape} ({seqs.nbytes / 1e9:.2f} GB)")

    run_fold(df_sorted, seqs, 2024, "PRIMARY")
    run_fold(df_sorted, seqs, 2023, "STRESS")


if __name__ == "__main__":
    main()
