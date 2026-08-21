"""
사용자 요청: "base는 우리꺼 쓰고 2단 이후로 3단부터는 팀원의 논리대로" — 지금까지(E038/E040)는
pressure_backoff/diversity stack을 전부 "base 잔차(residual1 = y - base_pred)"를 목표로 따로
학습해서 하나의 flat corrector에 메타피처로 욱여넣었다. 이번엔 pressure_backoff/diversity stack의
학습 타겟을 base 잔차가 아니라 **2단(hazard+joint+lasso corrector)이 설명하고 남긴 잔차
(residual2 = y - pred2)**로 바꾼다 — "이전 단의 잔차를 다음 단이 교정한다"는 팀원 pipeline의
캐스케이드 철학을 반영.

leak-safety: residual2는 corrector가 pitcher-half cross-fit으로 valid_df 안에서 이미 OOF로
계산되지만, pressure_backoff/diversity stack의 학습 테이블은 valid_df 밖(train_df의 마지막 연도,
"table_target")에서 만들어야 한다. 그러려면 table_target을 "가상의 valid_df"로 놓고 base+corrector
전체를 한 번 더(1년 더 이전 데이터로) 재현해서 table_target 자체의 stage-2 잔차를 leak-safe하게
구해야 한다 -- 무한 재귀를 피하기 위해 이 재귀는 1단계로 한정(3/4단은 같은 table_residual2를 공유).

최종 결합은 지금처럼 하나의 flat corrector(전부 residual1을 목표로 cross-fit)로 유지 -- 바뀌는 건
pressure_backoff/dstack 서브모델 자체가 "무엇을 맞히도록 학습됐는가"뿐.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
)
from segment_corrector_joint_softmax_meta import add_joint_label, fit_joint_softmax, joint_softmax_meta  # noqa: E402
from segment_corrector_meta_source_sweep import fit_hazard_family as fit_hazard_family_v1  # noqa: E402
from segment_corrector_eb_lookup_meta import fit_pressure_backoff_correction, apply_pressure_backoff_correction  # noqa: E402
from segment_corrector_diversity_stack_meta import build_diversity_stack_meta  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]
JOINT_META = ["joint_p_success", "joint_p_reverse", "joint_p_middle", "joint_p_outside"]
DSTACK_COLS = ["dstack_d", "dstack_failure", "dstack_pitchmix", "dstack_rfstate", "dstack_basestate"]


def build_champion_meta(train_df, valid_df):
    cat_meta = fit_hazard_family_v1(train_df, valid_df, "catboost")
    joint_model = fit_joint_softmax(train_df)
    joint_meta = joint_softmax_meta(joint_model, valid_df)
    lasso_meta = fit_hazard_family_v1(train_df, valid_df, "lasso")
    X = corrector_matrix(valid_df)
    for c in ["qR", "qM", "qO", "mr", "or", "om"]:
        X[f"rmo_{c}"] = cat_meta[c]
    for c in JOINT_META:
        X[c] = joint_meta[c]
    for c in ["mr", "or", "om"]:
        X[f"lasso_{c}"] = lasso_meta[c]
    return X


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    pitcher_ids = valid_df["pitcher_id"].to_numpy()

    base_pred = fit_base(train_df, valid_df)
    residual1 = y - base_pred
    seg3 = assign_segment_3way(valid_df)

    print("  1+2단: base + champion corrector(hazard6+joint4+lasso3)...")
    X_champion = build_champion_meta(train_df, valid_df)
    score2, pred2 = apply_corrector(X_champion, residual1, seg3, base_pred, y, valid_df)
    print(f"    champion(2단, 기준): {score2:.2f}")

    print("  table_target(1년 더 이전)에서 2단 잔차를 leak-safe하게 재현 중...")
    table_target_season = int(train_df["season"].max())
    table_source = train_df[train_df["season"] < table_target_season]
    table_target = train_df[train_df["season"] == table_target_season].reset_index(drop=True)
    table_y = table_target["control_success"].to_numpy(np.float64)

    table_base_pred = fit_base(table_source, table_target)
    table_residual1 = table_y - table_base_pred
    table_seg3 = assign_segment_3way(table_target)
    table_X_champion = build_champion_meta(table_source, table_target)
    _, table_pred2 = apply_corrector(table_X_champion, table_residual1, table_seg3, table_base_pred, table_y, table_target)
    table_residual2 = table_y - table_pred2
    print(f"    table_target({table_target_season}) 2단 잔차 확보 (|residual1| 평균={np.abs(table_residual1).mean():.4f} -> |residual2| 평균={np.abs(table_residual2).mean():.4f})")

    print("  3단(pressure_backoff)+4단(diversity stack)을 '2단 잔차(table_residual2)' 목표로 재학습...")
    backoff_table_v1 = fit_pressure_backoff_correction(table_target, table_residual1)  # 기존(E038) 비교용
    backoff_table_v2 = fit_pressure_backoff_correction(table_target, table_residual2)  # 신규(캐스케이드)
    eb_v1 = apply_pressure_backoff_correction(valid_df, backoff_table_v1)
    eb_v2 = apply_pressure_backoff_correction(valid_df, backoff_table_v2)

    dstack_v1 = build_diversity_stack_meta(table_target, valid_df, table_residual1)  # 기존(E040) 비교용
    dstack_v2 = build_diversity_stack_meta(table_target, valid_df, table_residual2)  # 신규(캐스케이드)

    print("\n  조합 비교 (전부 최종 결합은 residual1 목표 flat corrector, 서브모델 학습 타겟만 다름):")
    results = {}

    X_flat_v1 = X_champion.copy()
    X_flat_v1["eb_pressure_backoff"] = eb_v1
    for c in DSTACK_COLS:
        X_flat_v1[c] = dstack_v1[c]
    s, pred = apply_corrector(X_flat_v1, residual1, seg3, base_pred, y, valid_df)
    d = (pred2 - y) ** 2 - (pred - y) ** 2
    _, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"    [기존: 서브모델이 base잔차(residual1) 목표, v15/v16 재현] {s:.2f}  차이={s - score2:+.2f}  z={z:.2f}")
    results["기존(residual1 목표)"] = (s, z)

    X_flat_v2 = X_champion.copy()
    X_flat_v2["eb_pressure_backoff"] = eb_v2
    for c in DSTACK_COLS:
        X_flat_v2[c] = dstack_v2[c]
    s, pred = apply_corrector(X_flat_v2, residual1, seg3, base_pred, y, valid_df)
    d = (pred2 - y) ** 2 - (pred - y) ** 2
    _, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"    [신규: 서브모델이 2단잔차(residual2) 목표, 진짜 캐스케이드] {s:.2f}  차이={s - score2:+.2f}  z={z:.2f}")
    results["신규(residual2 목표, 캐스케이드)"] = (s, z)

    return score2, results


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
        print(f"  [{name}] {row}")


if __name__ == "__main__":
    main()
