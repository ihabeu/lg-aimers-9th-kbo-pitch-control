"""
현재 채택된 corrector(추가 피처 없음)와 hand+month 추가 corrector, 두 최종 예측의 residual
상관관계를 확인하고, 블렌드(가중평균)해서 primary/stress 둘 다 개선되는 지점이 있는지 찾는다.
count는 한계효용이 낮아서 제외.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

HYBRID_TEAM_ID = 13
SEEDS = (42, 2026, 314)
SEGMENTS = ["core", "hybrid", "dev"]
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
EB_K = 50.0
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def crossfit_prediction(X, residual, segment, base_pred, frame):
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
    return np.mean(np.column_stack(seed_preds), axis=1)


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

    f1 = apply_conditional_rate(valid_df, eb_conditional_rate(train_df, ["pitcher_id", "batter_hand"], EB_K), ["pitcher_id", "batter_hand"], league_prior, EB_K, "hand")
    f3 = apply_conditional_rate(valid_df, eb_conditional_rate(train_df, ["pitcher_id", "game_month"], EB_K), ["pitcher_id", "game_month"], league_prior, EB_K, "month")

    X_current = corrector_matrix(valid_df)
    X_hm = corrector_matrix(valid_df, pd.concat([f1, f3], axis=1))

    p_current = crossfit_prediction(X_current, residual, segment, base_pred, valid_df)
    p_hm = crossfit_prediction(X_hm, residual, segment, base_pred, valid_df)

    r_current = y - p_current
    r_hm = y - p_hm
    corr = np.corrcoef(r_current, r_hm)[0, 1]
    print(f"  residual correlation(현재, hand+month) = {corr:.4f}")
    print(f"  현재 단독: {score(float(np.mean(r_current**2)), float(y.mean())):.2f}")
    print(f"  hand+month 단독: {score(float(np.mean(r_hm**2)), float(y.mean())):.2f}")

    print("  --- blend alpha sweep (0=현재, 1=hand+month) ---")
    results = {}
    for alpha in ALPHA_GRID:
        blend = np.clip((1 - alpha) * p_current + alpha * p_hm, 0, 1)
        s = score(float(np.mean((blend - y) ** 2)), float(y.mean()))
        results[alpha] = s
        print(f"    alpha={alpha}: {s:.2f}")
    return results


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== alpha별 두 폴드 동시 비교 =====")
    for alpha in ALPHA_GRID:
        print(f"  alpha={alpha}: primary={r1[alpha]:.2f}  stress={r2[alpha]:.2f}")


if __name__ == "__main__":
    main()
