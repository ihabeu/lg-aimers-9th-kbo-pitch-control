"""
E033 후속: hazard 서브모델(reverse/middle/outside)을 XGBoost, 순수 로지스틱(정규화 없음), MLP로도
만들어서 champion(v14: CatBoost hazard 6 + joint softmax 4 + Lasso hazard 3)에 추가 도움이 되는지
확인. 여전히 "비싼 fit은 폴드당 한 번" 원칙 유지.

이 머신에 다른 무거운 작업(사용자 별도 연구)이 같이 돌고 있어서, CatBoost/XGBoost/LightGBM의
thread_count/n_jobs를 -1(전체 코어) 대신 4로 낮춰서 자원을 나눠 쓴다 -- 개별 fit은 조금 느려지지만
전체적으로는 더 안정적으로 진행됨.

MLP는 sklearn이 L2(alpha)는 기본 지원하지만 L1은 없다 -- 정직하게 L2만 제공.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
    FEATURES, CAT_FEATURES, EPS,
)
from segment_corrector_joint_softmax_meta import add_joint_label, fit_joint_softmax, joint_softmax_meta  # noqa: E402
from segment_corrector_meta_source_sweep import fit_hazard_family as fit_hazard_family_v1  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

NUMERIC_ISH = [c for c in FEATURES if c not in CAT_FEATURES]
N_JOBS = 4  # 다른 무거운 작업과 자원을 나눠 씀 (-1 대신 고정값)


def label_encode_for_xgb(train_df, valid_df, cat_levels):
    def enc(df):
        x = df[FEATURES].copy()
        for c in CAT_FEATURES:
            x[c] = pd.Categorical(x[c].astype(str), categories=cat_levels[c]).codes
        return x
    return enc(train_df), enc(valid_df)


def fit_xgb_hazard(train_df, valid_df, target_col):
    cat_levels = {c: sorted(train_df[c].astype(str).unique()) for c in CAT_FEATURES}
    X_tr, X_va = label_encode_for_xgb(train_df, valid_df, cat_levels)
    y_tr = train_df[target_col].to_numpy()
    model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=50,
                           random_state=42, n_jobs=N_JOBS, tree_method="hist", verbosity=0)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]


def build_pure_logistic_pipeline():
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="<NA>")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric_pipe, NUMERIC_ISH), ("cat", cat_pipe, CAT_FEATURES)])
    clf = LogisticRegression(penalty=None, max_iter=2000, solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def build_mlp_pipeline():
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="<NA>")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric_pipe, NUMERIC_ISH), ("cat", cat_pipe, CAT_FEATURES)])
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300, early_stopping=True,
                         n_iter_no_change=10, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_sklearn_hazard(train_df, valid_df, target_col, builder):
    pipe = builder()
    pipe.fit(train_df[FEATURES], train_df[target_col])
    return pipe.predict_proba(valid_df[FEATURES])[:, 1]


def fit_hazard_family_v2(train_df, valid_df, algo):
    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])

    def fit_one(sub_df, target_col):
        if algo == "xgboost":
            return fit_xgb_hazard(sub_df, valid_df, target_col)
        elif algo == "logistic":
            return fit_sklearn_hazard(sub_df, valid_df, target_col, build_pure_logistic_pipeline)
        elif algo == "mlp":
            return fit_sklearn_hazard(sub_df, valid_df, target_col, build_mlp_pipeline)
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

    print("  캐싱: champion 메타피처(CatBoost hazard6 + joint4 + Lasso3)...")
    cat_meta = fit_hazard_family_v1(train_df, valid_df, "catboost")
    joint_model = fit_joint_softmax(train_df)
    joint_meta = joint_softmax_meta(joint_model, valid_df)
    lasso_meta = fit_hazard_family_v1(train_df, valid_df, "lasso")

    X_champion = X.copy()
    for c in ["qR", "qM", "qO", "mr", "or", "om"]:
        X_champion[f"rmo_{c}"] = cat_meta[c]
    for c in ["joint_p_success", "joint_p_reverse", "joint_p_middle", "joint_p_outside"]:
        X_champion[c] = joint_meta[c]
    for c in ["mr", "or", "om"]:
        X_champion[f"lasso_{c}"] = lasso_meta[c]
    champion_score, champion_pred = apply_corrector(X_champion, residual, seg3, base_pred, y, valid_df)
    print(f"  champion(v14, 메타피처 13개): {champion_score:.2f}  (기준)")

    print("  캐싱: XGBoost/순수로지스틱/MLP hazard...")
    xgb_meta = fit_hazard_family_v2(train_df, valid_df, "xgboost")
    logistic_meta = fit_hazard_family_v2(train_df, valid_df, "logistic")
    mlp_meta = fit_hazard_family_v2(train_df, valid_df, "mlp")

    print("\n  조합 스윕:")
    results = {}
    for name, meta in [("XGBoost", xgb_meta), ("순수 로지스틱", logistic_meta), ("MLP", mlp_meta)]:
        X_c = X_champion.copy()
        for c in ["mr", "or", "om"]:
            X_c[f"new_{c}"] = meta[c]
        s, pred = apply_corrector(X_c, residual, seg3, base_pred, y, valid_df)
        d = (champion_pred - y) ** 2 - (pred - y) ** 2
        mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
        print(f"    [+{name} hazard] {s:.2f}  차이={s - champion_score:+.2f}  z={z:.2f}")
        results[name] = (s, z)
    return champion_score, results


def main():
    df = add_rmo_labels(load("train.csv"))
    df = add_joint_label(df)
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    for name in r1[1]:
        s1, z1 = r1[1][name]
        s2, z2 = r2[1][name]
        print(f"  [+{name}] PRIMARY {s1:.2f}(z={z1:.2f})  STRESS {s2:.2f}(z={z2:.2f})")


if __name__ == "__main__":
    main()
