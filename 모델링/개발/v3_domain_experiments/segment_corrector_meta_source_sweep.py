"""
사용자 제안 방법론 적용: 비싼 fit(hazard 서브모델 계열 - CatBoost/LightGBM/Ridge/Lasso, joint softmax)은
폴드당 딱 한 번씩만 하고 예측값을 캐싱해서, 그 위에서 "corrector에 어떤 조합을 넣을지"는 재학습 없이
싸게 스윕한다(이전 스크립트들처럼 조합마다 서브모델을 다시 fit하는 낭비를 없앰).

"모델을 바꾸든 섞든 다 해봐" 요청: hazard 서브모델(reverse/middle/outside 예측)을 CatBoost 외에
LightGBM, Ridge(L2 로지스틱), Lasso(L1 로지스틱)로도 만들어서, E026처럼 진짜 다른 정보를 주는지
(상관관계) + corrector 메타피처로 추가했을 때 실제로 도움되는지 둘 다 확인.

Ridge/Lasso는 트리가 아니라 선형모델이라 elastic_net.py와 같은 전처리(원핫, 스케일링, 결측 대체)가
필요하다 -- 그 전처리 로직을 재사용.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, fit_hazard_sub, corrector_matrix, assign_segment_3way,
    apply_corrector, pitcher_bootstrap_z, FEATURES, CAT_FEATURES, EPS,
)
from segment_corrector_joint_softmax_meta import add_joint_label, fit_joint_softmax, joint_softmax_meta  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

NUMERIC_ISH = [c for c in FEATURES if c not in CAT_FEATURES]
LGB_PARAMS = dict(n_estimators=3000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=-1, verbosity=-1,
                   num_leaves=15, min_child_samples=1000, reg_lambda=1.0)

CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]
JOINT_META = ["joint_p_success", "joint_p_reverse", "joint_p_middle", "joint_p_outside"]


def to_lgbm_frame(df):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category")
    return x


def fit_lgb_hazard(train_df, valid_df, target_col):
    """early stopping 사용(valid_df 라벨 없이는 못 하니 train 내부에서 10% 홀드아웃으로 대체)."""
    labeled = train_df
    holdout_n = max(1000, int(len(labeled) * 0.1))
    holdout = labeled.iloc[-holdout_n:]
    fit_part = labeled.iloc[:-holdout_n]

    X_fit, X_hold = to_lgbm_frame(fit_part), to_lgbm_frame(holdout)
    y_fit, y_hold = fit_part[target_col].to_numpy(), holdout[target_col].to_numpy()

    m = LGBMClassifier(**LGB_PARAMS)
    m.fit(X_fit, y_fit, eval_set=[(X_hold, y_hold)], eval_metric="binary_logloss",
          categorical_feature=CAT_FEATURES, callbacks=[early_stopping(100, verbose=False)])
    X_valid = to_lgbm_frame(valid_df)
    return m.predict_proba(X_valid)[:, 1]


def build_linear_pipeline(penalty):
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="<NA>")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric_pipe, NUMERIC_ISH), ("cat", cat_pipe, CAT_FEATURES)])
    if penalty == "l2":
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs")
    else:
        clf = LogisticRegression(penalty="l1", C=1.0, max_iter=2000, solver="liblinear")
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_linear_hazard(train_df, valid_df, target_col, penalty):
    pipe = build_linear_pipeline(penalty)
    pipe.fit(train_df[FEATURES], train_df[target_col])
    return pipe.predict_proba(valid_df[FEATURES])[:, 1]


def fit_hazard_family(train_df, valid_df, algo):
    """algo별로 qR/qM/qO를 순차적으로 학습(reverse -> not-reverse의 middle -> not-r,m의 outside)."""
    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])

    def fit_one(sub_train_df, target_col):
        if algo == "catboost":
            model = fit_hazard_sub(sub_train_df, target_col)
            return model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
        elif algo == "lightgbm":
            return fit_lgb_hazard(sub_train_df, valid_df, target_col)
        elif algo in ("ridge", "lasso"):
            return fit_linear_hazard(sub_train_df, valid_df, target_col, "l2" if algo == "ridge" else "l1")
        raise ValueError(algo)

    qR = fit_one(rmo_train, "reverse_label")
    not_reverse = rmo_train[rmo_train["reverse_label"] == 0]
    qM = fit_one(not_reverse, "middle_label")
    not_rm = not_reverse[not_reverse["middle_label"] == 0]
    not_rm = not_rm[not_rm["outside_label"].isin([0, 1])]
    qO = fit_one(not_rm, "outside_label")

    return {
        "qR": qR, "qM": qM, "qO": qO,
        "mr": np.log((qM + EPS) / (qR + EPS)),
        "or": np.log((qO + EPS) / (qR + EPS)),
        "om": np.log((qO + EPS) / (qM + EPS)),
    }


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)

    base_pred = fit_base(train_df, valid_df)
    residual = y - base_pred
    seg3 = assign_segment_3way(valid_df)
    X = corrector_matrix(valid_df)
    pitcher_ids = valid_df["pitcher_id"].to_numpy()

    print("  캐싱 단계: hazard 계열별 예측값 한 번씩만 fit...")
    cat_meta = fit_hazard_family(train_df, valid_df, "catboost")
    joint_model = fit_joint_softmax(train_df)
    joint_meta = joint_softmax_meta(joint_model, valid_df)
    lgb_meta = fit_hazard_family(train_df, valid_df, "lightgbm")
    ridge_meta = fit_hazard_family(train_df, valid_df, "ridge")
    lasso_meta = fit_hazard_family(train_df, valid_df, "lasso")

    print("  다양성 진단 (corr with CatBoost qR/qM/qO, 참고: control_success 기준 LightGBM은 0.9996):")
    for name, meta in [("LightGBM", lgb_meta), ("Ridge", ridge_meta), ("Lasso", lasso_meta)]:
        corrs = [np.corrcoef(cat_meta[c], meta[c])[0, 1] for c in ("qR", "qM", "qO")]
        print(f"    {name}: qR={corrs[0]:.4f}  qM={corrs[1]:.4f}  qO={corrs[2]:.4f}")

    print("\n  조합 스윕 (재학습 없이 corrector만 다시 fit):")
    champion_score = None
    champion_pred = None
    combos = {
        "champion(CatBoost hazard6 + joint4)": (CHAMPION_META, cat_meta, JOINT_META, joint_meta, None, None),
        "+LightGBM hazard(mr,or,om)": (CHAMPION_META, cat_meta, JOINT_META, joint_meta, ["mr", "or", "om"], lgb_meta),
        "+Ridge hazard(mr,or,om)": (CHAMPION_META, cat_meta, JOINT_META, joint_meta, ["mr", "or", "om"], ridge_meta),
        "+Lasso hazard(mr,or,om)": (CHAMPION_META, cat_meta, JOINT_META, joint_meta, ["mr", "or", "om"], lasso_meta),
    }
    results = {}
    for name, (cat_cols, cat_src, joint_cols, joint_src, extra_cols, extra_src) in combos.items():
        X_c = X.copy()
        for c in cat_cols:
            X_c[f"rmo_{c}"] = cat_src[c]
        for c in joint_cols:
            X_c[c] = joint_src[c]
        if extra_cols:
            for c in extra_cols:
                X_c[f"extra_{c}"] = extra_src[c]
        s, pred = apply_corrector(X_c, residual, seg3, base_pred, y, valid_df)
        if champion_score is None:
            champion_score, champion_pred = s, pred
            print(f"    [{name}] {s:.2f}  (기준)")
        else:
            d = (champion_pred - y) ** 2 - (pred - y) ** 2
            mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
            print(f"    [{name}] {s:.2f}  차이={s - champion_score:+.2f}  z={z:.2f}")
        results[name] = s
    return results


def main():
    df = add_rmo_labels(load("train.csv"))
    df = add_joint_label(df)
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    for name in r1:
        print(f"  [{name}] PRIMARY {r1[name]:.2f}  STRESS {r2[name]:.2f}")


if __name__ == "__main__":
    main()
