"""
segment별 ExtraTrees corrector가 실제로 어떤 피처를 써서 오차를 보정하는지 확인. "더 쪼개기"가 아니라
"corrector가 뭘 이미 설명하고 뭘 놓치는지"를 보기 위한 진단.

corrector를 2024 라벨 전체로(cross-fit 아님, 진단용) 학습해서 feature_importances_를 뽑는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

HYBRID_TEAM_ID = 13
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEGMENTS = ["core", "hybrid", "dev"]


def assign_segment(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


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


def corrector_matrix(df):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category").cat.codes.astype(np.float32)
    return x.apply(pd.to_numeric, errors="coerce")


def main():
    df = load("train.csv")
    target_season = 2024
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    residual = y - base_pred

    X = corrector_matrix(valid_df)

    print("===== segment별 corrector feature importance (상위 15개) =====")
    for seg in SEGMENTS:
        mask = segment == seg
        model = ExtraTreesRegressor(n_jobs=-1, random_state=16242, **CORRECTOR_CFG)
        model.fit(X.loc[mask], residual[mask])
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"\n--- {seg} (n={int(mask.sum())}) ---")
        print(importances.head(15).round(4).to_string())

    print("\n===== base(CatBoost) feature importance 상위 15개 (비교용) =====")
    base_importances = pd.Series(base_model.get_feature_importance(), index=FEATURES).sort_values(ascending=False)
    print(base_importances.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
