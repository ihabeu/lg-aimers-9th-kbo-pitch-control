"""
사용자 요청 "모델을 바꾸든 섞든 다 해봐"에 대한 첫 단계: E026에서 LightGBM이 메인 타겟
(control_success) 기준으로는 CatBoost와 residual 상관 0.9996으로 사실상 같은 판단을 내린다는 걸
확인했었다. 그런데 그건 "control_success"라는 타겟에 대한 얘기고, R/M/O hazard 서브모델은 완전히
다른 타겟(reverse/middle/outside)을 예측한다 -- 같은 정도로 겹칠지는 별개 질문이라 먼저 싸게
진단만 해본다(상관관계만, corrector까지 안 붙임).

LightGBM 설정은 lightgbm_baseline_sweep.py에서 이미 찾은 최적값(num_leaves=15,
min_child_samples=1000, reg_lambda=1.0) 재사용.
"""
import sys
from pathlib import Path

import numpy as np
from catboost import Pool
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import fit_hazard_sub, FEATURES, CAT_FEATURES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

LGB_PARAMS = dict(n_estimators=3000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=-1, verbosity=-1,
                   num_leaves=15, min_child_samples=1000, reg_lambda=1.0)


def to_lgbm_frame(df):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category")
    return x


def fit_lgb_hazard(train_df, target_col):
    X = to_lgbm_frame(train_df)
    y = train_df[target_col].to_numpy()
    m = LGBMClassifier(**LGB_PARAMS)
    m.fit(X, y, categorical_feature=CAT_FEATURES)
    return m


def main():
    df = add_rmo_labels(load("train.csv"))
    train_df = df[df["season"] < 2024]
    valid_df = df[df["season"] == 2024].reset_index(drop=True)

    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])

    print("qR (reverse) 학습 중...")
    qR_cat = fit_hazard_sub(rmo_train, "reverse_label")
    qR_lgb = fit_lgb_hazard(rmo_train, "reverse_label")

    not_reverse = rmo_train[rmo_train["reverse_label"] == 0]
    print("qM (middle | not reverse) 학습 중...")
    qM_cat = fit_hazard_sub(not_reverse, "middle_label")
    qM_lgb = fit_lgb_hazard(not_reverse, "middle_label")

    not_rm = not_reverse[not_reverse["middle_label"] == 0]
    not_rm = not_rm[not_rm["outside_label"].isin([0, 1])]
    print("qO (outside | not reverse, not middle) 학습 중...")
    qO_cat = fit_hazard_sub(not_rm, "outside_label")
    qO_lgb = fit_lgb_hazard(not_rm, "outside_label")

    valid_pool = Pool(valid_df[FEATURES], cat_features=CAT_FEATURES)
    valid_lgb = to_lgbm_frame(valid_df)

    for name, cat_model, lgb_model in [("qR", qR_cat, qR_lgb), ("qM", qM_cat, qM_lgb), ("qO", qO_cat, qO_lgb)]:
        p_cat = cat_model.predict_proba(valid_pool)[:, 1]
        p_lgb = lgb_model.predict_proba(valid_lgb)[:, 1]
        corr = np.corrcoef(p_cat, p_lgb)[0, 1]
        print(f"  {name}: corr(CatBoost, LightGBM) = {corr:.4f}  (참고: control_success 기준 E026은 0.9996)")


if __name__ == "__main__":
    main()
