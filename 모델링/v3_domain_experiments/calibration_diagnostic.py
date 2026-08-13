"""
오늘 멀티모델 블렌드(로컬 두 폴드 다 이겼는데 실제 LB는 champion보다 -10.09 낮음, E011)가
calibration(과신/과소신) 문제였는지 진단. "blend 이후 calibration을 별도로 봐야 한다"는 아이디어를 우리 데이터로 직접 검증.

champion(CatBoost 단독+corrector) vs 기각된 블렌드(0.6/0.2/0.2+corrector)의 calibration slope
(cov(p,y)/var(p), 1에서 멀수록 과/과소신) 및 mean bias(예측평균-실제평균)를 두 폴드에서 비교한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

HYBRID_TEAM_ID = 13
SEGMENTS = ["core", "hybrid", "dev"]
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)
CANDIDATES = {"champion(CatBoost단독)": (1.0, 0.0, 0.0), "기각된 블렌드(0.6/0.2/0.2)": (0.6, 0.2, 0.2)}


def assign_segment(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def label_encode(train_df, valid_df, cat_cols):
    train_x = train_df[FEATURES].copy()
    valid_x = valid_df[FEATURES].copy()
    for c in cat_cols:
        s = train_df[c].astype("string").fillna("<NA>")
        levels = {v: i for i, v in enumerate(pd.Index(s.unique()))}
        train_x[c] = s.map(levels).astype(np.float32)
        valid_x[c] = valid_df[c].astype("string").fillna("<NA>").map(levels).fillna(-1).astype(np.float32)
    return train_x.apply(pd.to_numeric, errors="coerce"), valid_x.apply(pd.to_numeric, errors="coerce")


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


def crossfit_predict(X, residual, segment, base_pred, frame):
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


def calibration_stats(p, y):
    r = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    bias = float(p.mean() - r)
    slope = float(np.cov(p, y, ddof=0)[0, 1] / np.var(p))
    return {"score": score(brier, r), "bias": bias, "slope": slope}


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y_train = train_df["control_success"].to_numpy(np.float64)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)

    cat_pool_tr = Pool(train_df[FEATURES], y_train, cat_features=CAT_FEATURES)
    cat_pool_va = Pool(valid_df[FEATURES], y, cat_features=CAT_FEATURES)
    cat_model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
                                    eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
                                    random_seed=42, thread_count=-1, verbose=False)
    cat_model.fit(cat_pool_tr, eval_set=cat_pool_va, use_best_model=True)
    p_cat = cat_model.predict_proba(cat_pool_va)[:, 1]

    X_tr, X_va = label_encode(train_df, valid_df, CAT_FEATURES)
    lgb_model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                colsample_bytree=0.8, reg_lambda=5.0, min_child_samples=100,
                                random_state=42, n_jobs=-1, verbosity=-1)
    lgb_model.fit(X_tr, y_train)
    p_lgb = lgb_model.predict_proba(X_va)[:, 1]

    xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=50,
                               random_state=42, n_jobs=-1, tree_method="hist", verbosity=0)
    xgb_model.fit(X_tr, y_train)
    p_xgb = xgb_model.predict_proba(X_va)[:, 1]

    X_corr = corrector_matrix(valid_df)
    for name, w in CANDIDATES.items():
        p_blend = np.clip(w[0] * p_cat + w[1] * p_lgb + w[2] * p_xgb, 0, 1)
        base_stats = calibration_stats(p_blend, y)
        p_final = crossfit_predict(X_corr, y - p_blend, segment, p_blend, valid_df)
        final_stats = calibration_stats(p_final, y)
        print(f"  {name}")
        print(f"    base(corrector 전):  score={base_stats['score']:.2f}  bias={base_stats['bias']:+.5f}  slope={base_stats['slope']:.4f}")
        print(f"    +corrector(최종):    score={final_stats['score']:.2f}  bias={final_stats['bias']:+.5f}  slope={final_stats['slope']:.4f}")


def main():
    df = load("train.csv")
    run_fold(df, 2024, "PRIMARY")
    run_fold(df, 2023, "STRESS")


if __name__ == "__main__":
    main()
