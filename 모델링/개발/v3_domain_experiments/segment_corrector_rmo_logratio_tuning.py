"""
E029(segment_corrector_rmo_logratio_feature.py)가 STRESS는 유의(z=2.01)했지만 PRIMARY(z=1.71)가
근소하게 기준(z~1.96) 미달이었던 걸 개선할 수 있는지 확인.

두 메타피처(log_ratio_mr, log_ratio_or)가 qR을 공유해서 서로 상관이 있을 수 있다 -- 하나만 넣으면
노이즈가 줄어 z가 오를 수도 있다. 또한 corrector capacity(min_samples_leaf)를 키우면(=더 단순한
모델) 메타피처가 만드는 미세한 노이즈에 덜 휘둘릴 수 있다. 둘을 조합해서 스윕.

기존 champion corrector와 base_pred/segment 계산은 재사용(segment_corrector_rmo_logratio_feature.py
그대로 import) -- 새로 안 짬.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, rmo_logratio_features, corrector_matrix, assign_segment_3way,
    apply_corrector, pitcher_bootstrap_z, brier_score, score, CORRECTOR_CFG,
)
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

FEATURE_VARIANTS = {
    "both(E029 그대로)": ["mr", "or"],
    "mr만": ["mr"],
    "or만": ["or"],
}
CAPACITY_VARIANTS = {
    "기본(min_leaf=200)": dict(CORRECTOR_CFG),
    "정규화강화(min_leaf=400)": {**CORRECTOR_CFG, "min_samples_leaf": 400},
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

    score_base, pred_base = apply_corrector(X, residual, seg3, base_pred, y, valid_df)
    print(f"  기존 3-way(메타피처 없음): {score_base:.2f}")

    log_ratio_mr, log_ratio_or = rmo_logratio_features(train_df, valid_df)
    meta_cols = {"mr": log_ratio_mr, "or": log_ratio_or}

    results = {}
    for feat_name, feats in FEATURE_VARIANTS.items():
        X_meta = X.copy()
        for f in feats:
            X_meta[f"rmo_log_ratio_{f}"] = meta_cols[f]
        for cap_name, cfg in CAPACITY_VARIANTS.items():
            s, pred = apply_corrector(X_meta, residual, seg3, base_pred, y, valid_df, corrector_cfg=cfg)
            d = (pred_base - y) ** 2 - (pred - y) ** 2
            mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
            key = f"{feat_name} / {cap_name}"
            results[key] = (s, z)
            print(f"  [{key}] {s:.2f}  차이={s - score_base:+.2f}  z={z:.2f}")
    return score_base, results


def main():
    df = add_rmo_labels(load("train.csv"))
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")

    print("\n===== 요약 (PRIMARY / STRESS z 둘 다 필요) =====")
    for key in r1[1]:
        s1, z1 = r1[1][key]
        s2, z2 = r2[1][key]
        flag = "★둘다유의" if z1 >= 1.96 and z2 >= 1.96 else ""
        print(f"  [{key}] PRIMARY {s1:.2f}(z={z1:.2f})  STRESS {s2:.2f}(z={z2:.2f})  {flag}")


if __name__ == "__main__":
    main()
