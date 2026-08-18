"""
팀 깃헙(iamdbstjd/LGAIMERS)의 M0(4-class joint softmax: success/reverse/middle/outside를 한 번에
모델링)를 코드는 가져오지 않고 아이디어만 참고해서 독립 재구현(대회 규정 §11).

E029/E030은 R/M/O를 **순차적 hazard**(qR -> qM|not R -> qO|not R,M)로 모델링했다. 이건 다른 구조 --
4개 클래스를 하나의 multiclass 모델로 동시에 예측하면, 순차 체인이 "reverse가 아니라는 조건 하에"만
middle을 보는 것과 달리 네 유형 간 상관관계를 한 번에 공유된 표현으로 잡을 수 있어서 다른 정보를
줄 가능성이 있다. 기존 6개 hazard 메타피처는 그대로 두고, joint softmax의 확률 4개를 추가로 얹어서
"보완적인 정보"인지 확인한다.

라벨: rmo_labels.py가 이미 만든 reverse_label/middle_label/outside_label(leak-safe)로 4-class
정수 라벨(0=success, 1=reverse, 2=middle, 3=outside)을 구성.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
    FEATURES, CAT_FEATURES,
)
from segment_corrector_rmo_extended_meta import rmo_all_meta  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from baseline_catboost import L2_LEAF_REG  # noqa: E402
from eda import load  # noqa: E402

CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]
CLASS_NAMES = ["success", "reverse", "middle", "outside"]


def add_joint_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    label = np.where(df["control_success"] == 1, 0,
             np.where(df["reverse_label"] == 1, 1,
             np.where(df["middle_label"] == 1, 2,
             np.where(df["outside_label"] == 1, 3, np.nan))))
    df["joint_label"] = label
    return df


def fit_joint_softmax(train_df: pd.DataFrame) -> CatBoostClassifier:
    labeled = train_df.dropna(subset=["joint_label"])
    y = labeled["joint_label"].astype(int).to_numpy()
    pool = Pool(labeled[FEATURES], y, cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, loss_function="MultiClass",
        l2_leaf_reg=L2_LEAF_REG, random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(pool)
    return model


def joint_softmax_meta(model: CatBoostClassifier, df: pd.DataFrame) -> dict:
    pool = Pool(df[FEATURES], cat_features=CAT_FEATURES)
    proba = model.predict_proba(pool)  # 열 순서 = 클래스 정수값 오름차순 (0,1,2,3)
    return {f"joint_p_{name}": proba[:, i] for i, name in enumerate(CLASS_NAMES)}


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

    hazard_meta = rmo_all_meta(train_df, valid_df)
    X_champion = X.copy()
    for c in CHAMPION_META:
        X_champion[f"rmo_{c}"] = hazard_meta[c]
    champion_score, champion_pred = apply_corrector(X_champion, residual, seg3, base_pred, y, valid_df)
    print(f"  champion(hazard 6개): {champion_score:.2f}  (기준)")

    joint_model = fit_joint_softmax(train_df)
    joint_meta = joint_softmax_meta(joint_model, valid_df)
    X_new = X_champion.copy()
    for name, vals in joint_meta.items():
        X_new[name] = vals
    new_score, new_pred = apply_corrector(X_new, residual, seg3, base_pred, y, valid_df)
    d = (champion_pred - y) ** 2 - (new_pred - y) ** 2
    mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"  +joint softmax 4개: {new_score:.2f}  차이={new_score - champion_score:+.2f}  z={z:.2f}")
    return champion_score, new_score, z


def main():
    df = add_rmo_labels(load("train.csv"))
    df = add_joint_label(df)
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    print(f"PRIMARY: champion {r1[0]:.2f} -> +joint softmax {r1[1]:.2f} (z={r1[2]:.2f})")
    print(f"STRESS:  champion {r2[0]:.2f} -> +joint softmax {r2[1]:.2f} (z={r2[2]:.2f})")


if __name__ == "__main__":
    main()
