"""
현재 champion 구조(789.23 CatBoost + 3-way segment ExtraTrees corrector, 실제 LB 879.80)의
2024(primary) residual을 여러 조건별로 스캔해서 "어떤 조건에서 체계적으로 과대/과소예측하는가"를 찾는다.
segmentation을 새로 만드는 게 아니라, 이미 있는 CORE/HYBRID/DEV 안에서 failure mode를 진단하는 용도.

corrector는 반드시 pitcher-disjoint cross-fit으로 만든다 -- in-sample로 하면 ExtraTrees가 이미
패턴을 암기해서 bias가 인위적으로 0에 가깝게 나오는 문제가 있었음(첫 시도에서 확인, 재작성).
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
SEGMENTS = ["core", "hybrid", "dev"]
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)


def pitcher_half(frame, seed):
    pitchers = np.array(sorted(frame["pitcher_id"].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pitchers)
    first_half = set(pitchers[: len(pitchers) // 2])
    return np.where(frame["pitcher_id"].astype(str).isin(first_half).to_numpy(), 0, 1)


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


def corrector_matrix(df):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category").cat.codes.astype(np.float32)
    return x.apply(pd.to_numeric, errors="coerce")


def bias_table(rows: pd.DataFrame, y: np.ndarray, pred: np.ndarray, group_cols) -> pd.DataFrame:
    tmp = rows.copy()
    tmp["_y"] = y
    tmp["_pred"] = pred
    tmp["_residual"] = y - pred
    g = tmp.groupby(group_cols, observed=True).agg(
        n=("_y", "size"), success_rate=("_y", "mean"), pred_mean=("_pred", "mean"),
        bias=("_residual", "mean"),
    ).reset_index()
    g = g[g["n"] >= 300].sort_values("bias", key=lambda s: s.abs(), ascending=False)
    return g


def main():
    df = load("train.csv")
    target_season = 2024
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)
    valid_df = valid_df.assign(segment=segment)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    residual = y - base_pred

    X = corrector_matrix(valid_df)
    seed_preds = []
    for seed in SEEDS:
        fold = pitcher_half(valid_df, seed)
        correction = np.zeros(len(valid_df))
        for half in (0, 1):
            tr_mask = fold != half
            ev_mask = fold == half
            for seg in SEGMENTS:
                tr = tr_mask & (segment == seg)
                ev = ev_mask & (segment == seg)
                model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **CORRECTOR_CFG)
                model.fit(X.loc[tr], residual[tr])
                correction[ev] = model.predict(X.loc[ev])
        seed_preds.append(np.clip(base_pred + correction, 0, 1))
    final_pred = np.mean(np.column_stack(seed_preds), axis=1)

    overall = score(float(np.mean((final_pred - y) ** 2)), float(y.mean()))
    print(f"전체 BSS(pitcher-disjoint cross-fit, 801.93와 같아야 함): {overall:.2f}\n")

    single_cols = ["segment", "pitcher_hand", "batter_hand", "game_month", "game_dayofweek",
                   "top_bottom", "base_state", "inning"]
    for col in single_cols:
        print(f"=== bias by {col} (|bias| 상위, n>=300) ===")
        print(bias_table(valid_df, y, final_pred, [col]).round(4).head(8).to_string(index=False))
        print()

    pair_cols = [
        ("segment", "pitcher_hand"), ("segment", "batter_hand"),
        ("pitcher_hand", "batter_hand"),
        ("segment", "game_month"),
        ("balls_before", "strikes_before"),
    ]
    for cols in pair_cols:
        print(f"=== bias by {cols} (|bias| 상위, n>=300) ===")
        print(bias_table(valid_df, y, final_pred, list(cols)).round(4).head(8).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
