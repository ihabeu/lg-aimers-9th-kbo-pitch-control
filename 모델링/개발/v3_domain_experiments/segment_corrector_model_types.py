"""
segment residual corrector 모델 종류를 ExtraTrees 외에 XGBoost로도 시도, 그리고 둘을 블렌드했을 때
도움되는지 확인. 3-way segment(core/hybrid/dev) 구조와 pitcher-disjoint cross-fit 검증 방식은
segment_residual_corrector_3way.py와 동일하게 유지하고, corrector 모델 종류만 바꾼다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

HYBRID_TEAM_ID = 13
SEEDS = (42, 2026, 314)
SEGMENTS = ["core", "hybrid", "dev"]

ET_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
XGB_CFG = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=5.0, min_child_weight=50, tree_method="hist")


def assign_segment(df: pd.DataFrame) -> np.ndarray:
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def brier_score(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


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


def crossfit_correction(model_maker, X, residual, segment, frame, seed):
    fold = pitcher_half(frame, seed)
    correction = np.zeros(len(frame))
    for half in (0, 1):
        tr_mask = fold != half
        ev_mask = fold == half
        for seg in SEGMENTS:
            tr = tr_mask & (segment == seg)
            ev = ev_mask & (segment == seg)
            model = model_maker(seed)
            model.fit(X.loc[tr], residual[tr])
            correction[ev] = model.predict(X.loc[ev])
    return correction


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    base_score = score(brier_score(y, base_pred), y.mean())
    print("base BSS:", round(base_score, 2))

    X = corrector_matrix(valid_df)
    residual = y - base_pred

    et_preds, xgb_preds = [], []
    for seed in SEEDS:
        et_corr = crossfit_correction(lambda s: ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(s), **ET_CFG), X, residual, segment, valid_df, seed)
        xgb_corr = crossfit_correction(lambda s: XGBRegressor(random_state=16200 + int(s), n_jobs=-1, verbosity=0, **XGB_CFG), X, residual, segment, valid_df, seed)
        et_preds.append(np.clip(base_pred + et_corr, 0, 1))
        xgb_preds.append(np.clip(base_pred + xgb_corr, 0, 1))

    et_consensus = np.mean(np.column_stack(et_preds), axis=1)
    xgb_consensus = np.mean(np.column_stack(xgb_preds), axis=1)
    blend = np.clip((et_consensus + xgb_consensus) / 2, 0, 1)

    et_score = score(brier_score(y, et_consensus), y.mean())
    xgb_score = score(brier_score(y, xgb_consensus), y.mean())
    blend_score = score(brier_score(y, blend), y.mean())
    print(f"  ExtraTrees corrected: {et_score:.2f} (gain {et_score - base_score:+.2f})")
    print(f"  XGBoost corrected:    {xgb_score:.2f} (gain {xgb_score - base_score:+.2f})")
    print(f"  ET+XGB blend:         {blend_score:.2f} (gain {blend_score - base_score:+.2f})")
    print(f"  corr(ET_correction, XGB_correction) on full consensus preds: {np.corrcoef(et_consensus, xgb_consensus)[0,1]:.4f}")
    return base_score, et_score, xgb_score, blend_score


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n요약 (base, ET, XGB, blend):")
    print("2023->2024:", r1)
    print("2022->2023:", r2)
    print("(비교: 3-way ExtraTrees corrected -- 2023->2024 801.93, 2022->2023 755.63)")


if __name__ == "__main__":
    main()
