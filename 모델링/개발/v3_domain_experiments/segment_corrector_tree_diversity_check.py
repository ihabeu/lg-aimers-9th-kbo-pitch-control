"""
E033(LightGBM/Ridge/Lasso)/E034(XGBoost/순수로지스틱/MLP)에 이어 hazard 서브모델(R/M/O) 다양성 축을
계속 탐색 -- 아직 안 써본 트리 계열 3종(ExtraTrees/RandomForest/HistGradientBoosting)을 추가.
champion(v14: CatBoost hazard6 + joint4 + Lasso3, 메타피처 13개) 대비 검증.

ExtraTrees/RandomForest는 categorical을 못 받아서 XGBoost 실험(segment_corrector_meta_source_sweep_v2.py)
때 쓴 것과 같은 ordinal 인코딩(label_encode_for_xgb)을 재사용. HistGradientBoosting은 sklearn이
categorical_features='from_dtype'로 pandas category dtype을 네이티브 지원해서 별도 인코딩 없이 그대로 씀.

다른 무거운 작업과 자원을 나눠 쓰기 위해 n_jobs=4로 고정(E034와 동일 관례).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
    FEATURES, CAT_FEATURES, EPS,
)
from segment_corrector_joint_softmax_meta import add_joint_label, fit_joint_softmax, joint_softmax_meta  # noqa: E402
from segment_corrector_meta_source_sweep import fit_hazard_family as fit_hazard_family_v1  # noqa: E402
from segment_corrector_meta_source_sweep_v2 import label_encode_for_xgb  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

N_JOBS = 4


def to_category_frame(df):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category")
    return x


def fit_extra_trees_hazard(train_df, valid_df, target_col):
    cat_levels = {c: sorted(train_df[c].astype(str).unique()) for c in CAT_FEATURES}
    X_tr, X_va = label_encode_for_xgb(train_df, valid_df, cat_levels)
    model = ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=200,
                                  max_features=0.7, n_jobs=N_JOBS, random_state=42)
    model.fit(X_tr, train_df[target_col].to_numpy())
    return model.predict_proba(X_va)[:, 1]


def fit_random_forest_hazard(train_df, valid_df, target_col):
    cat_levels = {c: sorted(train_df[c].astype(str).unique()) for c in CAT_FEATURES}
    X_tr, X_va = label_encode_for_xgb(train_df, valid_df, cat_levels)
    model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=200,
                                    max_features=0.7, n_jobs=N_JOBS, random_state=42)
    model.fit(X_tr, train_df[target_col].to_numpy())
    return model.predict_proba(X_va)[:, 1]


def fit_hgb_hazard(train_df, valid_df, target_col):
    X_tr, X_va = to_category_frame(train_df), to_category_frame(valid_df)
    model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=6,
                                            l2_regularization=1.0, categorical_features="from_dtype",
                                            random_state=42)
    model.fit(X_tr, train_df[target_col].to_numpy())
    return model.predict_proba(X_va)[:, 1]


def fit_hazard_family_tree(train_df, valid_df, algo):
    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])

    def fit_one(sub_df, target_col):
        if algo == "extratrees":
            return fit_extra_trees_hazard(sub_df, valid_df, target_col)
        elif algo == "randomforest":
            return fit_random_forest_hazard(sub_df, valid_df, target_col)
        elif algo == "histgb":
            return fit_hgb_hazard(sub_df, valid_df, target_col)
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

    print("  캐싱: ExtraTrees/RandomForest/HistGradientBoosting hazard...")
    et_meta = fit_hazard_family_tree(train_df, valid_df, "extratrees")
    rf_meta = fit_hazard_family_tree(train_df, valid_df, "randomforest")
    hgb_meta = fit_hazard_family_tree(train_df, valid_df, "histgb")

    print("  다양성 진단 (corr with CatBoost qR/qM/qO):")
    for name, meta in [("ExtraTrees", et_meta), ("RandomForest", rf_meta), ("HistGB", hgb_meta)]:
        corrs = [np.corrcoef(cat_meta[c], meta[c])[0, 1] for c in ("qR", "qM", "qO")]
        print(f"    {name}: qR={corrs[0]:.4f}  qM={corrs[1]:.4f}  qO={corrs[2]:.4f}")

    print("\n  조합 스윕:")
    results = {}
    for name, meta in [("ExtraTrees", et_meta), ("RandomForest", rf_meta), ("HistGB", hgb_meta)]:
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
    folds = [(2024, "2024est"), (2023, "2023est"), (2022, "2022est")]
    all_results = {}
    for season, label in folds:
        _, results = run_fold(df, season, label)
        all_results[label] = results
    print("\n===== 요약 =====")
    for name in next(iter(all_results.values())):
        row = "  ".join(f"{label} {all_results[label][name][0]:.2f}(z={all_results[label][name][1]:.2f})" for _, label in folds)
        print(f"  [+{name}] {row}")


if __name__ == "__main__":
    main()
