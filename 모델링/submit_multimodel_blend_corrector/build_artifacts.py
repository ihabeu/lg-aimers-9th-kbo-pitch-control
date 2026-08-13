"""
멀티모델(CatBoost+LightGBM+XGBoost) 가중 블렌드 + 3-way segment residual corrector.
v3_domain_experiments/multimodel_weighted_blend.py에서 검증: weight(cat=.6,lgb=.2,xgb=.2)가
기존 champion(CatBoost 단독+corrector, primary 801.93/stress 755.63)보다 두 폴드 모두 개선
(primary 815.15/stress 833.05). 가중치 그리드, LightGBM/XGBoost 하이퍼파라미터 모두 자체 설정.

base 세 모델을 전체(2019~2024)로 학습해 실제 배포용으로 저장하고, 별도로 2019~2023만 학습한
residual-source 블렌드로 2024 잔차를 만들어 3-way corrector(ExtraTrees)를 학습한다
(submit_segment_residual_corrector/build_correctors.py와 같은 구조).
LightGBM/XGBoost는 CatBoost와 카테고리 처리 방식이 달라 고정 라벨인코딩 맵을 써서
학습/추론 정합성을 맞춘다.
"""
import sys
from pathlib import Path

import joblib
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

ROOT = Path(__file__).resolve().parent
WEIGHT = (0.6, 0.2, 0.2)  # cat, lgb, xgb
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)
SEGMENTS = ["core", "hybrid", "dev"]
HYBRID_TEAM_ID = 13


def assign_segment(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def fit_label_maps(df):
    maps = {}
    for c in CAT_FEATURES:
        values = df[c].astype("string").fillna("<NA>")
        maps[c] = {v: i for i, v in enumerate(sorted(values.unique()))}
    return maps


def label_matrix(df, maps):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = df[c].astype("string").fillna("<NA>").map(maps[c]).fillna(-1).astype(np.float32)
    return x.apply(pd.to_numeric, errors="coerce")


def fit_cat(train_df, eval_df=None, eval_y=None):
    pool_tr = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES)
    model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
                                eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
                                random_seed=42, thread_count=-1, verbose=False)
    if eval_df is not None:
        pool_va = Pool(eval_df[FEATURES], eval_y, cat_features=CAT_FEATURES)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    else:
        model.fit(pool_tr)
    return model


def fit_lgb(X_tr, y_tr):
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=5.0, min_child_samples=100,
                            random_state=42, n_jobs=-1, verbosity=-1)
    model.fit(X_tr, y_tr)
    return model


def fit_xgb(X_tr, y_tr):
    model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=50,
                           random_state=42, n_jobs=-1, tree_method="hist", verbosity=0)
    model.fit(X_tr, y_tr)
    return model


def blend_predict(cat, lgb, xgb, df, X):
    p_cat = cat.predict_proba(Pool(df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    p_lgb = lgb.predict_proba(X)[:, 1]
    p_xgb = xgb.predict_proba(X)[:, 1]
    return np.clip(WEIGHT[0] * p_cat + WEIGHT[1] * p_lgb + WEIGHT[2] * p_xgb, 0, 1)


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def main():
    df = load("train.csv")
    label_maps = fit_label_maps(df)
    X_all = label_matrix(df, label_maps)

    print("최종 블렌드 학습 (2019~2024 전체, 실제 배포용)")
    cat_final = fit_cat(df)
    lgb_final = fit_lgb(X_all, df["control_success"])
    xgb_final = fit_xgb(X_all, df["control_success"])

    src_train = df[df["season"] < 2024]
    residual_target = df[df["season"] == 2024].reset_index(drop=True)
    y_target = residual_target["control_success"].to_numpy(np.float64)
    X_src_tr = label_matrix(src_train, label_maps)
    X_tgt = label_matrix(residual_target, label_maps)

    print("residual-source 블렌드 학습 (2019~2023 -> 2024, 배포에는 안 씀)")
    cat_src = fit_cat(src_train, residual_target, y_target)
    lgb_src = fit_lgb(X_src_tr, src_train["control_success"])
    xgb_src = fit_xgb(X_src_tr, src_train["control_success"])
    residual_source_pred = blend_predict(cat_src, lgb_src, xgb_src, residual_target, X_tgt)

    brier = float(np.mean((residual_source_pred - y_target) ** 2))
    r = float(y_target.mean())
    print("residual-source 2024 BSS (sanity, multimodel_weighted_blend.py (.6,.2,.2) base=724.98와 같아야 함):",
          round(score(brier, r), 2))

    residual = y_target - residual_source_pred
    segment = assign_segment(residual_target)
    print("segment 분포:", pd.Series(segment).value_counts().to_dict())

    correctors = {}
    for seed in SEEDS:
        for seg in SEGMENTS:
            mask = segment == seg
            model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **CORRECTOR_CFG)
            model.fit(X_tgt.loc[mask], residual[mask])
            correctors[(seg, seed)] = model
            print(f"  corrector fit segment={seg} seed={seed} n={int(mask.sum())}")

    test = load("test.csv")
    X_test = label_matrix(test, label_maps)
    final_base_pred = blend_predict(cat_final, lgb_final, xgb_final, test, X_test)
    test_segment = assign_segment(test)
    correction = np.zeros(len(test))
    for seed in SEEDS:
        seed_corr = np.zeros(len(test))
        for seg in SEGMENTS:
            mask = test_segment == seg
            if not mask.any():
                continue
            seed_corr[mask] = correctors[(seg, seed)].predict(X_test.loc[mask])
        correction += seed_corr / len(SEEDS)
    final = np.clip(final_base_pred + correction, 0, 1)
    print("SANITY (로컬 test.csv) base:", final_base_pred, "final:", final)

    cat_final.save_model(str(ROOT / "model" / "cat_final.cbm"))
    joblib.dump(
        {
            "lgb_final": lgb_final, "xgb_final": xgb_final, "weight": WEIGHT,
            "correctors": correctors, "seeds": list(SEEDS), "segments": SEGMENTS,
            "label_maps": label_maps, "hybrid_team_id": HYBRID_TEAM_ID,
        },
        ROOT / "model" / "artifacts.joblib", compress=3,
    )
    print("saved:", ROOT / "model")


if __name__ == "__main__":
    main()
