"""
팀원 P997("diversity stack": 서로 다른 피처 부분집합 x 서로 다른 알고리즘으로 같은 잔차 타겟을
5번 따로 맞혀서 섞는 구조)을 우리 프로젝트에 이식. 코드는 안 가져오고 구조만 참고해서 독립 재구현
(대회 규정 §11). 원본은 D 성분이 LightGBM인데, 우리는 E026("잘 튜닝한 LightGBM은 CatBoost와
거의 같은 실수를 한다")을 이미 확인했으니 LightGBM 대신 CatBoost(l2_leaf_reg=L2_LEAF_REG, 우리
champion과 동일 정규화)를 씀.

target은 원본처럼 "도메인평균 대비 잔차"가 아니라, 우리가 이미 corrector 전체에서 쓰는 잔차
(champion base 예측 대비 잔차)를 그대로 씀 -- 우리 프로젝트엔 corrector가 이미 그 자리를 맡고
있어서, 이 5개 성분은 corrector "안"이 아니라 corrector에 들어가는 "메타피처"로 위치시킴(기존
hazard/joint/lasso 메타피처와 동급).

피처 부분집합은 팀원 P997_FROZEN_CONFIG.json의 FAILURE/PITCHMIX/RFSTATE/BASE_STATE 그룹을
우리 44피처 이름으로 매핑해서 재구성(팀원 원본 피처명과 안 겹치게 변수/함수 이름은 새로 지음).

폴드 이름: PRIMARY/STRESS 대신 검증 대상 연도로 직접 부름(2024est/2023est/2022est) -- 사용자
피드백 반영, "est"=estimator.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
    FEATURES, CAT_FEATURES, L2_LEAF_REG,
)
from segment_corrector_joint_softmax_meta import add_joint_label, fit_joint_softmax, joint_softmax_meta  # noqa: E402
from segment_corrector_meta_source_sweep import fit_hazard_family as fit_hazard_family_v1  # noqa: E402
from segment_corrector_meta_source_sweep_v2 import label_encode_for_xgb  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

N_JOBS = 4
CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]
JOINT_META = ["joint_p_success", "joint_p_reverse", "joint_p_middle", "joint_p_outside"]

# 팀원 P997 feature_subsets(FAILURE/PITCHMIX/RFSTATE/BASE_STATE)를 우리 44피처 이름으로 매핑.
SUBSET_FAILURE = [
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_middle_rate", "pitcher_hand", "batter_hand",
    "balls_before", "strikes_before", "outs_before", "li", "game_type",
]
SUBSET_PITCHMIX = [
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "pitcher_hand", "batter_hand", "balls_before", "strikes_before",
]
SUBSET_RFSTATE = [
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "pitcher_hand", "batter_hand",
]
SUBSET_BASE_STATE = [
    "base_state", "num_runners_on", "outs_before", "balls_before", "strikes_before",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "pitcher_hand", "batter_hand",
]


def fit_catboost_d(train_df, valid_df, target):
    pool_cat = [c for c in CAT_FEATURES]
    model = CatBoostRegressor(
        iterations=600, learning_rate=0.04, depth=6, loss_function="RMSE",
        l2_leaf_reg=L2_LEAF_REG, random_seed=918, thread_count=N_JOBS, verbose=False,
    )
    from catboost import Pool
    model.fit(Pool(train_df[FEATURES], target, cat_features=pool_cat))
    return model.predict(Pool(valid_df[FEATURES], cat_features=pool_cat))


def fit_subset_tree(train_df, valid_df, subset, target, builder):
    cat_in_subset = [c for c in subset if c in CAT_FEATURES]
    cat_levels = {c: sorted(train_df[c].astype(str).unique()) for c in cat_in_subset}

    def enc(df):
        x = df[subset].copy()
        for c in cat_in_subset:
            x[c] = pd.Categorical(x[c].astype(str), categories=cat_levels[c]).codes
        return x.apply(pd.to_numeric, errors="coerce")

    model = builder()
    model.fit(enc(train_df), target)
    return model.predict(enc(valid_df))


def build_diversity_stack_meta(train_df, valid_df, residual_train):
    """5개 성분(D=CatBoost, FAILURE/PITCHMIX/BASE_STATE=ExtraTrees, RFSTATE=RandomForest)을
    residual_train(train_df에 대응하는 잔차)에 맞춰 학습하고, valid_df에 대한 예측 5개를 반환."""
    d_pred = fit_catboost_d(train_df, valid_df, residual_train)
    failure_pred = fit_subset_tree(
        train_df, valid_df, SUBSET_FAILURE, residual_train,
        lambda: ExtraTreesRegressor(n_estimators=60, max_depth=10, min_samples_leaf=300,
                                     max_features=0.8, random_state=918, n_jobs=N_JOBS))
    pitchmix_pred = fit_subset_tree(
        train_df, valid_df, SUBSET_PITCHMIX, residual_train,
        lambda: ExtraTreesRegressor(n_estimators=50, max_depth=10, min_samples_leaf=300,
                                     max_features=0.8, random_state=918, n_jobs=N_JOBS))
    rfstate_pred = fit_subset_tree(
        train_df, valid_df, SUBSET_RFSTATE, residual_train,
        lambda: RandomForestRegressor(n_estimators=40, max_depth=10, min_samples_leaf=300,
                                       max_features=0.8, bootstrap=True, max_samples=0.8,
                                       random_state=918, n_jobs=N_JOBS))
    basestate_pred = fit_subset_tree(
        train_df, valid_df, SUBSET_BASE_STATE, residual_train,
        lambda: ExtraTreesRegressor(n_estimators=50, max_depth=10, min_samples_leaf=300,
                                     max_features=0.8, random_state=918, n_jobs=N_JOBS))
    return {
        "dstack_d": d_pred, "dstack_failure": failure_pred, "dstack_pitchmix": pitchmix_pred,
        "dstack_rfstate": rfstate_pred, "dstack_basestate": basestate_pred,
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
    for c in JOINT_META:
        X_champion[c] = joint_meta[c]
    for c in ["mr", "or", "om"]:
        X_champion[f"lasso_{c}"] = lasso_meta[c]
    champion_score, champion_pred = apply_corrector(X_champion, residual, seg3, base_pred, y, valid_df)
    print(f"  champion(v14, 메타피처 13개): {champion_score:.2f}  (기준)")

    # diversity stack 학습용 잔차: train_df 자체를 한 번 더 안쪽에서 쪼개 leak 방지
    # (train_df의 마지막 연도를 inner target으로, residual-source 패턴과 동일)
    inner_target_season = int(train_df["season"].max())
    inner_source = train_df[train_df["season"] < inner_target_season]
    inner_target = train_df[train_df["season"] == inner_target_season].reset_index(drop=True)
    inner_base_pred = fit_base(inner_source, inner_target)
    inner_residual = inner_target["control_success"].to_numpy(np.float64) - inner_base_pred

    print("  캐싱: diversity stack(D=CatBoost, FAILURE/PITCHMIX/BASE_STATE=ExtraTrees, RFSTATE=RandomForest)...")
    dstack_meta = build_diversity_stack_meta(inner_target, valid_df, inner_residual)

    print("\n  조합 스윕:")
    results = {}
    for name in ["dstack_d", "dstack_failure", "dstack_pitchmix", "dstack_rfstate", "dstack_basestate"]:
        X_c = X_champion.copy()
        X_c[name] = dstack_meta[name]
        s, pred = apply_corrector(X_c, residual, seg3, base_pred, y, valid_df)
        d = (champion_pred - y) ** 2 - (pred - y) ** 2
        mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
        print(f"    [+{name}] {s:.2f}  차이={s - champion_score:+.2f}  z={z:.2f}")
        results[name] = (s, z)

    X_all = X_champion.copy()
    for name in ["dstack_d", "dstack_failure", "dstack_pitchmix", "dstack_rfstate", "dstack_basestate"]:
        X_all[name] = dstack_meta[name]
    s, pred = apply_corrector(X_all, residual, seg3, base_pred, y, valid_df)
    d = (champion_pred - y) ** 2 - (pred - y) ** 2
    mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"    [+전부(5개)] {s:.2f}  차이={s - champion_score:+.2f}  z={z:.2f}")
    results["전부(5개)"] = (s, z)
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
    combo_names = list(next(iter(all_results.values())).keys())
    for name in combo_names:
        row = "  ".join(f"{label} {all_results[label][name][0]:.2f}(z={all_results[label][name][1]:.2f})" for _, label in folds)
        print(f"  [+{name}] {row}")


if __name__ == "__main__":
    main()
