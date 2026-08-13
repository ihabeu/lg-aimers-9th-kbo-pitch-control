"""
"stable vs transition" segmentation 후보 진단. 최근에 R<->F 역할이 바뀐 투수인지(직전 시즌 대비
F 비중이 크게 변했는지)를 2023까지의 데이터로만 계산해서, 그 정보가 실제 2024 residual과 관련
있는지 확인. 진단만 하고 바로 segment로 안 씀.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402


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


def pitcher_f_ratio_by_season(history: pd.DataFrame) -> pd.DataFrame:
    g = history.groupby(["pitcher_id", "season"]).apply(
        lambda d: pd.Series({"n": len(d), "f_ratio": (d["game_type"] == "F").mean()})
    )
    return g.reset_index()


def diagnose(df, target_season, label):
    print(f"\n===== {label}: target={target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    residual = y - base_pred

    prev_season = target_season - 1
    prev2_season = target_season - 2
    by_season = pitcher_f_ratio_by_season(train_df[train_df["season"].isin([prev_season, prev2_season])])
    prev = by_season[by_season["season"] == prev_season].set_index("pitcher_id")["f_ratio"]
    prev2 = by_season[by_season["season"] == prev2_season].set_index("pitcher_id")["f_ratio"]
    role_shift = (prev - prev2).dropna()
    print(f"role_shift(직전-그전 F비중 변화) 분포: mean={role_shift.mean():.3f} std={role_shift.std():.3f} n={len(role_shift)}")

    valid_role_shift = valid_df["pitcher_id"].map(role_shift)
    is_transition = valid_role_shift.abs() > 0.3
    is_transition = is_transition.fillna(False).to_numpy()

    print(f"\ntransition 투수 행 수: {int(is_transition.sum())} / stable: {int((~is_transition).sum())}")
    for is_t, name in [(True, "transition"), (False, "stable")]:
        mask = is_transition == is_t
        if mask.sum() < 30:
            continue
        r = float(y[mask].mean())
        base_bss = score(float(np.mean((base_pred[mask] - y[mask]) ** 2)), r) if 0 < r < 1 else float("nan")
        print(f"  {name}: n={int(mask.sum())} success_rate={r:.4f} pred_mean={base_pred[mask].mean():.4f} "
              f"residual_mean={residual[mask].mean():.4f} base_BSS={base_bss:.2f}")

    valid_rows = valid_role_shift.notna().to_numpy()
    if valid_rows.sum() > 30:
        r_corr, p = stats.pearsonr(valid_role_shift[valid_rows], residual[valid_rows])
        print(f"\ncorr(role_shift, residual) 행 단위 = {r_corr:+.4f} (n={int(valid_rows.sum())}, p={p:.4f})")


def main():
    df = load("train.csv")
    diagnose(df, 2024, "PRIMARY")
    diagnose(df, 2023, "STRESS")


if __name__ == "__main__":
    main()
