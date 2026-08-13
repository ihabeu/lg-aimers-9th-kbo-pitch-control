"""
conditional_ability_feature.py에서 3개(투수x타자손/투수x카운트/투수x월)를 한꺼번에 넣었더니
primary -9.79 / stress +42.52로 트레이드오프가 나왔음. 어느 게 원인인지 하나씩 따로 넣어서 분리.
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
SEEDS = (42, 2026, 314)
SEGMENTS = ["core", "hybrid", "dev"]
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
EB_K = 50.0


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


def eb_conditional_rate(history, keys, k):
    g = history.groupby(keys)["control_success"].agg(["size", "mean"]).reset_index()
    return g.rename(columns={"size": "_n", "mean": "_raw"})


def apply_conditional_rate(rows, table, keys, league_prior, k, name):
    merged = rows[keys].merge(table, on=keys, how="left")
    n = merged["_n"].fillna(0.0).to_numpy()
    raw = merged["_raw"].fillna(league_prior).to_numpy()
    smoothed = (raw * n + league_prior * k) / (n + k)
    return pd.Series(smoothed.astype(np.float32), index=rows.index, name=name)


def corrector_matrix(df, extra=None):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category").cat.codes.astype(np.float32)
    x = x.apply(pd.to_numeric, errors="coerce")
    if extra is not None:
        x = pd.concat([x.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    return x


def pitcher_half(frame, seed):
    pitchers = np.array(sorted(frame["pitcher_id"].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pitchers)
    first_half = set(pitchers[: len(pitchers) // 2])
    return np.where(frame["pitcher_id"].astype(str).isin(first_half).to_numpy(), 0, 1)


def crossfit_score(X, residual, segment, base_pred, y, frame):
    seed_preds = []
    for seed in SEEDS:
        fold = pitcher_half(frame, seed)
        correction = np.zeros(len(frame))
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
    consensus = np.mean(np.column_stack(seed_preds), axis=1)
    return score(float(np.mean((consensus - y) ** 2)), float(y.mean()))


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    residual = y - base_pred
    league_prior = float(train_df["control_success"].mean())

    train_df2 = train_df.assign(count_state=train_df["balls_before"].astype(str) + "_" + train_df["strikes_before"].astype(str))
    valid_df2 = valid_df.assign(count_state=valid_df["balls_before"].astype(str) + "_" + valid_df["strikes_before"].astype(str))

    t1 = eb_conditional_rate(train_df, ["pitcher_id", "batter_hand"], EB_K)
    f1 = apply_conditional_rate(valid_df, t1, ["pitcher_id", "batter_hand"], league_prior, EB_K, "pitcher_x_batterhand_rate")
    t2 = eb_conditional_rate(train_df2, ["pitcher_id", "count_state"], EB_K)
    f2 = apply_conditional_rate(valid_df2, t2, ["pitcher_id", "count_state"], league_prior, EB_K, "pitcher_x_count_rate")
    t3 = eb_conditional_rate(train_df, ["pitcher_id", "game_month"], EB_K)
    f3 = apply_conditional_rate(valid_df, t3, ["pitcher_id", "game_month"], league_prior, EB_K, "pitcher_x_month_rate")

    X_base = corrector_matrix(valid_df)
    s_base = crossfit_score(X_base, residual, segment, base_pred, y, valid_df)
    print(f"  없음(현재): {s_base:.2f}")

    for name, feat in [("투수x타자손", f1), ("투수x카운트", f2), ("투수x월", f3)]:
        X = corrector_matrix(valid_df, feat.to_frame())
        s = crossfit_score(X, residual, segment, base_pred, y, valid_df)
        print(f"  +{name}: {s:.2f} (차이 {s - s_base:+.2f})")

    return s_base


def main():
    df = load("train.csv")
    run_fold(df, 2024, "PRIMARY")
    run_fold(df, 2023, "STRESS")


if __name__ == "__main__":
    main()
