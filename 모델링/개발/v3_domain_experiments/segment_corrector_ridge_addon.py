"""
E033에서 확인한 두 번째 근접 후보(Ridge, STRESS z=1.82) 재검증 -- 이번엔 이미 champion에 들어간
Lasso(v14, 메타피처 13개) 위에 "추가로" 얹어서, Ridge가 Lasso와 겹치지 않는 정보를 더 주는지 확인.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
)
from segment_corrector_joint_softmax_meta import add_joint_label, fit_joint_softmax, joint_softmax_meta  # noqa: E402
from segment_corrector_meta_source_sweep import fit_hazard_family as fit_hazard_family_v1  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]
JOINT_META = ["joint_p_success", "joint_p_reverse", "joint_p_middle", "joint_p_outside"]


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

    print("  Ridge hazard 추가...")
    ridge_meta = fit_hazard_family_v1(train_df, valid_df, "ridge")
    X_new = X_champion.copy()
    for c in ["mr", "or", "om"]:
        X_new[f"ridge_{c}"] = ridge_meta[c]
    new_score, new_pred = apply_corrector(X_new, residual, seg3, base_pred, y, valid_df)
    d = (champion_pred - y) ** 2 - (new_pred - y) ** 2
    mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"  +Ridge hazard: {new_score:.2f}  차이={new_score - champion_score:+.2f}  z={z:.2f}")
    return champion_score, new_score, z


def main():
    df = add_rmo_labels(load("train.csv"))
    df = add_joint_label(df)
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    print(f"PRIMARY: champion {r1[0]:.2f} -> +Ridge hazard {r1[1]:.2f} (z={r1[2]:.2f})")
    print(f"STRESS:  champion {r2[0]:.2f} -> +Ridge hazard {r2[1]:.2f} (z={r2[2]:.2f})")


if __name__ == "__main__":
    main()
