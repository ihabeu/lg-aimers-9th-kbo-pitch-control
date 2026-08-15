"""
lightgbm_baseline_sweep.py에서 찾은 최적 설정(num_leaves=15, min_child_samples=1000, reg_lambda=1.0)을
dual-fold(primary 2023->2024, stress 2022->2023)로 재검증. single-split 하나만으로는 이 세션 내내
"폴드 하나로는 못 믿는다"는 원칙을 지켜왔으므로, LightGBM이 구조적으로 CatBoost보다 못한 게 맞는지
확인하려면 최소 두 폴드는 봐야 한다.

sweep에서 이미 확인된 것: bias-correction(post-hoc 평균 이동)도, 진단용 in-sample Platt(상한선)도
CatBoost(734.49)를 못 따라잡음(각각 654.53 / 657.04) -- calibration 문제가 아니라 정보 추출 자체의
격차로 보임.
"""
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, time_split, train_catboost, evaluate as cat_evaluate  # noqa: E402

BEST_PARAMS = dict(num_leaves=15, min_child_samples=1000, reg_lambda=1.0)
FIXED = dict(n_estimators=3000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
             random_state=42, n_jobs=-1, verbosity=-1)


def to_lgbm_frame(df):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category")
    return x


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def score(brier_val, r):
    return max(0.0, 100000 * (1 - brier_val / (r * (1 - r))))


def run_fold(df, target_season, label):
    train_df, valid_df = time_split(df, target_season)
    X_tr, X_va = to_lgbm_frame(train_df), to_lgbm_frame(valid_df)
    y_tr, y_va = train_df["control_success"], valid_df["control_success"]

    m = LGBMClassifier(**FIXED, **BEST_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
          categorical_feature=CAT_FEATURES, callbacks=[early_stopping(100, verbose=False)])
    p_lgb = m.predict_proba(X_va)[:, 1]
    y = y_va.to_numpy()
    r = float(y_va.mean())
    lgb_score = score(brier(y, p_lgb), r)

    cat_model = train_catboost(train_df, valid_df)
    cat_metrics = cat_evaluate(cat_model, valid_df)

    print(f"{label}: LightGBM={lgb_score:.2f}  CatBoost={cat_metrics['score (리더보드 산식)']:.2f}  차이={lgb_score - cat_metrics['score (리더보드 산식)']:+.2f}")
    return lgb_score, cat_metrics["score (리더보드 산식)"], p_lgb, y


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    print(f"PRIMARY: LightGBM={r1[0]:.2f} vs CatBoost={r1[1]:.2f} (차이 {r1[0]-r1[1]:+.2f})")
    print(f"STRESS:  LightGBM={r2[0]:.2f} vs CatBoost={r2[1]:.2f} (차이 {r2[0]-r2[1]:+.2f})")


if __name__ == "__main__":
    main()
