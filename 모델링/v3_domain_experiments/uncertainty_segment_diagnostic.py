"""
"uncertainty(표본 신뢰도) 기반 segmentation" 후보를 먼저 진단만 한다 (segment/corrector로 바로 안 씀).
asof_pitcher_n/asof_batter_n(그 투수/타자에 대한 누적 표본 수, 이미 각 행에 제공되는 값이라 leak 없음)으로
low/high history 구간을 나눠서, 3-way(core/hybrid/dev) 각각의 내부에서 base residual이 실제로
다른지 확인한다. Brier 자체가 낮은 신뢰도(=0.5 근처로 수렴해야 하는) 표본에서 base가 과신하고 있는지
보는 것.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

HYBRID_TEAM_ID = 13
N_BUCKETS = [0, 50, 200, 1000, np.inf]
BUCKET_LABELS = ["<50", "50-200", "200-1000", ">1000"]


def assign_segment(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def fit_base(train_df, valid_df):
    train_pool = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES)
    valid_pool = Pool(valid_df[FEATURES], valid_df["control_success"], cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
        eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def diagnose(df, target_season, label):
    print(f"\n===== {label}: target={target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    residual = y - base_pred
    segment = assign_segment(valid_df)

    pitcher_n_bucket = pd.cut(valid_df["asof_pitcher_n"].fillna(0), bins=N_BUCKETS, labels=BUCKET_LABELS, right=False)

    for seg in ["core", "hybrid", "dev"]:
        print(f"\n--- {seg} ---")
        mask = segment == seg
        rows = []
        for bucket in BUCKET_LABELS:
            bmask = mask & (pitcher_n_bucket == bucket).to_numpy()
            n = int(bmask.sum())
            if n < 30:
                continue
            r = float(y[bmask].mean())
            base_bss = score(float(np.mean((base_pred[bmask] - y[bmask]) ** 2)), r) if 0 < r < 1 else float("nan")
            rows.append({
                "bucket": bucket, "n": n, "success_rate": round(r, 4),
                "pred_mean": round(float(base_pred[bmask].mean()), 4),
                "residual_mean": round(float(residual[bmask].mean()), 4),
                "residual_std": round(float(residual[bmask].std()), 4),
                "base_BSS": round(base_bss, 2),
            })
        print(pd.DataFrame(rows).to_string(index=False))


def main():
    df = load("train.csv")
    diagnose(df, 2024, "PRIMARY")
    diagnose(df, 2023, "STRESS")


if __name__ == "__main__":
    main()
