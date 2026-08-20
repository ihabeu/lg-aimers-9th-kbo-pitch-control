"""
사용자 지적: 예전 Trackman 실험(12개 파생변수를 champion residual과 직접 상관관계로 검증, E024에서
완전 기각)은 "파생변수를 직접 얹는" 방식이었다 -- 지금 통하고 있는 패턴(타겟을 R/M/O로 분해해서
서브모델에 태우고 그 출력을 메타피처로 쓰는 것)과는 다른 카테고리다. 그래서 Trackman 12개 피처를
"hazard 서브모델의 입력"으로 다시 시도(base/corrector에 직접 넣는 게 아님).

가설: Trackman 물리 데이터(구속/무브먼트/릴리스포인트 등)가 "제구 성공 여부" 자체는 못 바꿔도(E024
결론과 일치), "실패한다면 어떤 유형인가"는 바꿀 수 있다 -- 예를 들어 무브먼트가 큰 투수는 존을 크게
벗어나는(outside) 쪽으로, 커맨드가 약한 투수는 가운데로 몰리는(middle) 쪽으로 치우칠 수 있다는
도메인 가설.

`trackman_features.py`(우리 자체 매핑, 신뢰 332명)의 add_trackman_history_features()를 그대로
재사용(중복 구현 안 함) -- FEATURES에 12개를 추가한 확장판으로 hazard 서브모델만 다시 학습.
"""
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

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
from trackman_features import add_trackman_history_features, HIST_FEATURES  # noqa: E402
from baseline_catboost import L2_LEAF_REG  # noqa: E402
from eda import load  # noqa: E402

TM_FEATURES = FEATURES + HIST_FEATURES
CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]
JOINT_META = ["joint_p_success", "joint_p_reverse", "joint_p_middle", "joint_p_outside"]


def fit_tm_hazard_sub(train_df, target_col):
    y = train_df[target_col].to_numpy()
    pool = Pool(train_df[TM_FEATURES], y, cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss",
        l2_leaf_reg=L2_LEAF_REG, random_seed=42, thread_count=4, verbose=False,
    )
    model.fit(pool)
    return model


def fit_tm_hazard_family(train_df, valid_df):
    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])
    valid_pool = Pool(valid_df[TM_FEATURES], cat_features=CAT_FEATURES)

    qR_model = fit_tm_hazard_sub(rmo_train, "reverse_label")
    qR = qR_model.predict_proba(valid_pool)[:, 1]
    not_reverse = rmo_train[rmo_train["reverse_label"] == 0]
    qM_model = fit_tm_hazard_sub(not_reverse, "middle_label")
    qM = qM_model.predict_proba(valid_pool)[:, 1]
    not_rm = not_reverse[not_reverse["middle_label"] == 0]
    not_rm = not_rm[not_rm["outside_label"].isin([0, 1])]
    qO_model = fit_tm_hazard_sub(not_rm, "outside_label")
    qO = qO_model.predict_proba(valid_pool)[:, 1]

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
    for c in CHAMPION_META:
        X_champion[f"rmo_{c}"] = cat_meta[c]
    for c in JOINT_META:
        X_champion[c] = joint_meta[c]
    for c in ["mr", "or", "om"]:
        X_champion[f"lasso_{c}"] = lasso_meta[c]
    champion_score, champion_pred = apply_corrector(X_champion, residual, seg3, base_pred, y, valid_df)
    print(f"  champion(v14, 메타피처 13개): {champion_score:.2f}  (기준)")

    print("  Trackman 12개를 hazard 서브모델 입력에 추가해서 재학습...")
    tm_meta = fit_tm_hazard_family(train_df, valid_df)
    X_new = X_champion.copy()
    for c in ["mr", "or", "om"]:
        X_new[f"tm_{c}"] = tm_meta[c]
    new_score, new_pred = apply_corrector(X_new, residual, seg3, base_pred, y, valid_df)
    d = (champion_pred - y) ** 2 - (new_pred - y) ** 2
    mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"  +Trackman hazard: {new_score:.2f}  차이={new_score - champion_score:+.2f}  z={z:.2f}")
    return champion_score, new_score, z


def main():
    df = add_rmo_labels(load("train.csv"))
    df = add_joint_label(df)
    df = add_trackman_history_features(df)
    coverage = df["hist_avg_rel_speed"].notna().mean()
    print(f"Trackman 커버리지: {coverage:.1%}")

    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    print(f"PRIMARY: champion {r1[0]:.2f} -> +Trackman hazard {r1[1]:.2f} (z={r1[2]:.2f})")
    print(f"STRESS:  champion {r2[0]:.2f} -> +Trackman hazard {r2[1]:.2f} (z={r2[2]:.2f})")


if __name__ == "__main__":
    main()
