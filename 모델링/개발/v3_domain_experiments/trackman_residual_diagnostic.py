"""
Trackman 과거 투수 프로필 피처(기존 modeling/trackman_features.py, 우리 자체 매핑) 12개가
현재 champion 구조(789.23 + 3-way segment corrector, 실제 LB 879.80)의 잔차와 상관관계가
있는지 진단. STEP 1(상관관계) + STEP 3(기존 asof_pitcher_success_rate와 중복인지) + segment별
효과를 한 번에 본다. 아직 모델에 안 넣고 진단만.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesRegressor
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402
from trackman_features import add_trackman_history_features, HIST_FEATURES  # noqa: E402
from trackman_mapping import build_mapping  # noqa: E402

HYBRID_TEAM_ID = 13
SEGMENTS = ["core", "hybrid", "dev"]
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)


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
    final_pred = crossfit_prediction(X, residual, segment, base_pred, valid_df)
    final_residual = y - final_pred
    print("전체 champion BSS(cross-fit, 801.93와 같아야 함):", round(score(float(np.mean(final_residual**2)), float(y.mean())), 2))

    print("\nTrackman 매핑 구축 중...")
    mapping = build_mapping()
    tm_valid = add_trackman_history_features(valid_df, mapping=mapping)
    coverage = tm_valid["hist_avg_rel_speed"].notna().mean()
    print(f"2024 검증행 기준 Trackman 커버리지: {coverage:.1%}")

    print("\n===== STEP 1: Trackman 피처 12개 vs base residual / final residual 상관관계 =====")
    print(f"{'feature':30s} {'n_valid':>8s} {'corr_vs_base_resid':>20s} {'corr_vs_final_resid':>20s} {'corr_vs_asof_success':>22s}")
    asof_rate = valid_df["asof_pitcher_success_rate"].to_numpy(np.float64)
    for feat in HIST_FEATURES:
        vals = tm_valid[feat].to_numpy(np.float64)
        valid_mask = np.isfinite(vals)
        n = int(valid_mask.sum())
        if n < 100:
            print(f"{feat:30s} {n:8d}  (표본 부족)")
            continue
        r_base, _ = stats.pearsonr(vals[valid_mask], residual[valid_mask])
        r_final, _ = stats.pearsonr(vals[valid_mask], final_residual[valid_mask])
        r_asof, _ = stats.pearsonr(vals[valid_mask], asof_rate[valid_mask])  # STEP 3: 기존 능력 피처와 중복인지
        print(f"{feat:30s} {n:8d} {r_base:20.4f} {r_final:20.4f} {r_asof:22.4f}")

    print("\n===== segment별 커버리지 =====")
    for seg in SEGMENTS:
        mask = segment == seg
        cov = tm_valid.loc[mask, "hist_avg_rel_speed"].notna().mean()
        print(f"  {seg}: n={int(mask.sum())} coverage={cov:.1%}")


if __name__ == "__main__":
    main()
