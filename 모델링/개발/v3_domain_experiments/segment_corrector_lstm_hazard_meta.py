"""
사용자 요청: "시계열적으로 잘 맞는 모델"(LSTM)도 hazard 서브모델로 시도. asof_* 피처는 투수 이력을
요약통계(rate)로 압축한 것인데, LSTM은 원시 투구 순서(최근 W개 투구의 실제 시퀀스)에서 CatBoost가
못 보는 순서-의존 패턴을 찾을 가능성이 있다 -- E020(diversity_lab_lstm.py v2)에서 이미 검증된
윈도우 구축/정규화 로직을 그대로 재사용(중복 구현 안 함), 타겟만 control_success에서 R/M/O로 교체.

E020은 이 로직으로 control_success를 직접 예측했을 때는 BSS=0.00(학습은 됐지만 신호 없음)이었다 --
그런데 이번 세션 전체 패턴(메인 타겟에서 안 통하던 대체 모델이 R/M/O 서브 타겟에서는 통하는 경우가
있었음, LightGBM/Ridge/Lasso)과 같은 맥락에서 R/M/O 타겟은 다를 수 있는지 확인.

3단계(qR -> qM|not R -> qO|not R,M) 각각을 별도 LSTM으로 학습 -- 윈도우 자체는 전체 이력 기준으로
동일하게 만들고(asof_*와 같은 leak-safety), 학습에 쓰는 행(라벨)만 각 단계 조건으로 필터링.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
    FEATURES, CAT_FEATURES, EPS,
)
from segment_corrector_joint_softmax_meta import add_joint_label, fit_joint_softmax, joint_softmax_meta  # noqa: E402
from segment_corrector_meta_source_sweep import fit_hazard_family as fit_hazard_family_v1  # noqa: E402
from diversity_lab_lstm import build_matrix, build_windows, LSTMClassifier, WINDOW, HIDDEN, BATCH, GRAD_CLIP, DEVICE  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

NUMERIC_ISH = [c for c in FEATURES if c not in CAT_FEATURES]
EPOCHS = 8  # E020의 15보다 줄임(탐색 단계, R/M/O는 표본이 더 적어 더 빨리 수렴할 가능성)
CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]
JOINT_META = ["joint_p_success", "joint_p_reverse", "joint_p_middle", "joint_p_outside"]


def train_lstm_binary(seqs_tr, y_tr, seqs_eval, epochs=EPOCHS):
    model = LSTMClassifier(seqs_tr.shape[-1], HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(seqs_tr), torch.from_numpy(y_tr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model.train()
    for epoch in range(epochs):
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
    print(f"    (마지막 epoch loss={epoch_loss:.5f})")

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs_eval), BATCH):
            xb = torch.from_numpy(seqs_eval[i:i + BATCH]).to(DEVICE)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(preds)


def fit_lstm_hazard_family(seqs_norm, df_sorted, train_mask, valid_mask):
    """전체 시즌 순서로 정렬된 df_sorted/seqs_norm 기준. train_mask/valid_mask는 해당 boolean 배열."""
    rmo_ok = df_sorted["reverse_label"].notna().to_numpy() & df_sorted["middle_label"].notna().to_numpy()

    def fit_stage(scope_mask, target_col):
        tr_idx = train_mask & scope_mask
        y_tr = df_sorted[target_col].to_numpy()[tr_idx]
        print(f"    {target_col}: train n={tr_idx.sum()}")
        return train_lstm_binary(seqs_norm[tr_idx], y_tr, seqs_norm[valid_mask])

    qR = fit_stage(rmo_ok, "reverse_label")
    not_reverse = rmo_ok & (df_sorted["reverse_label"].to_numpy() == 0)
    qM = fit_stage(not_reverse, "middle_label")
    not_rm = not_reverse & (df_sorted["middle_label"].to_numpy() == 0) & df_sorted["outside_label"].isin([0, 1]).to_numpy()
    qO = fit_stage(not_rm, "outside_label")

    return {
        "qR": qR, "qM": qM, "qO": qO,
        "mr": np.log((qM + EPS) / (qR + EPS)),
        "or": np.log((qO + EPS) / (qR + EPS)),
        "om": np.log((qO + EPS) / (qM + EPS)),
    }


def run_fold(df_sorted, seqs, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_mask = (df_sorted["season"] < target_season).to_numpy()
    valid_mask = (df_sorted["season"] == target_season).to_numpy()
    train_df = df_sorted[train_mask]
    valid_df = df_sorted[valid_mask].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)

    base_pred = fit_base(train_df, valid_df)
    residual = y - base_pred
    seg3 = assign_segment_3way(valid_df)
    X = corrector_matrix(valid_df)
    pitcher_ids = valid_df["pitcher_id"].to_numpy()

    print("  캐싱: champion 메타피처(CatBoost hazard6 + joint4 + Lasso3)...")
    cat_meta = fit_hazard_family_v1(train_df, valid_df, "catboost")
    joint_model = fit_joint_softmax(train_df)
    joint_meta = joint_softmax_meta(joint_model, valid_df)
    lasso_meta = fit_hazard_family_v1(train_df, valid_df, "lasso")

    X_champion = X.copy()
    for c in CHAMPION_META:
        X_champion[f"rmo_{c}"] = cat_meta[c]
    for c in JOINT_META:
        X_champion[c] = joint_meta[c]
    for c in ["mr", "or", "om"]:
        X_champion[f"lasso_{c}"] = lasso_meta[c]
    champion_score, champion_pred = apply_corrector(X_champion, residual, seg3, base_pred, y, valid_df)
    print(f"  champion(v14, 메타피처 13개): {champion_score:.2f}  (기준)")

    # 채널 정규화 (train 통계만 사용, E020과 동일 패턴)
    flat_tr = seqs[train_mask].reshape(-1, seqs.shape[-1])
    valid_rows = ~np.all(flat_tr == 0, axis=1)
    mean = flat_tr[valid_rows].mean(axis=0)
    std = flat_tr[valid_rows].std(axis=0)
    std[std < 1e-6] = 1.0
    seqs_norm = (seqs - mean) / std

    print("  LSTM hazard 3단계 학습...")
    lstm_meta = fit_lstm_hazard_family(seqs_norm, df_sorted, train_mask, valid_mask)

    X_new = X_champion.copy()
    for c in ["mr", "or", "om"]:
        X_new[f"lstm_{c}"] = lstm_meta[c]
    new_score, new_pred = apply_corrector(X_new, residual, seg3, base_pred, y, valid_df)
    d = (champion_pred - y) ** 2 - (new_pred - y) ** 2
    mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"  +LSTM hazard: {new_score:.2f}  차이={new_score - champion_score:+.2f}  z={z:.2f}")
    return champion_score, new_score, z


def main():
    df = add_rmo_labels(load("train.csv"))
    df = add_joint_label(df)
    df_sorted = df.sort_values(["pitcher_id", "row_id"]).reset_index(drop=True)

    medians = {c: float(df_sorted[c].median()) for c in NUMERIC_ISH}
    category_maps = {}
    for c in CAT_FEATURES:
        values = df_sorted[c].astype("string").fillna("<NA>")
        category_maps[c] = {v: i for i, v in enumerate(sorted(values.unique()))}

    print("윈도우 빌드 중...")
    Xm = build_matrix(df_sorted, category_maps, medians)
    seqs = build_windows(df_sorted, Xm, WINDOW)
    print(f"seqs shape={seqs.shape} ({seqs.nbytes / 1e9:.2f} GB)")

    r1 = run_fold(df_sorted, seqs, 2024, "PRIMARY")
    r2 = run_fold(df_sorted, seqs, 2023, "STRESS")
    print("\n===== 요약 =====")
    print(f"PRIMARY: champion {r1[0]:.2f} -> +LSTM hazard {r1[1]:.2f} (z={r1[2]:.2f})")
    print(f"STRESS:  champion {r2[0]:.2f} -> +LSTM hazard {r2[1]:.2f} (z={r2[2]:.2f})")


if __name__ == "__main__":
    main()
