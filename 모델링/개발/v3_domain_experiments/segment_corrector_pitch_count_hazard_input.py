"""
사용자 요청: "예전에 기각했던 코드"(우리 프라이빗 저장소, 팀원 코드 아님)를 재사용해서 새 구조로
재시도. `within_game_state.py`(E024, pitch_count_before -- 경기 내 누적 투구수)는 "CatBoost 44피처에
직접 추가"하는 방식으로는 기각됐다(-22.71). 그런데 E029/E030이 보여준 패턴은: 라우팅이나 base 모델
직접 추가가 아니라 "별도 서브모델의 입력으로 넣고, 그 서브모델의 출력을 corrector 메타피처로 쓴다"는
구조가 통한다는 것이었다.

가설: pitch_count_before(피로도)가 "성공하냐 실패하냐"는 못 바꿔도(E024 결론과 일치), "실패했을 때
어떤 유형(reverse/middle/outside)으로 실패하는가"는 바꿀 수 있다 -- 피로한 투수는 아예 존을 크게
벗어나는 것과 가운데로 몰리는 것 중 하나에 더 치우칠 수 있다는 도메인 가설.

검증: R/M/O hazard 서브모델(qR/qM/qO)의 입력 피처에만 pitch_count_before를 추가(base CatBoost와
corrector 직접 입력은 그대로 44피처 유지) -- 서브모델이 더 정확해지면 거기서 나오는 메타피처(원시
확률+log-ratio 6개)의 품질도 같이 좋아질 수 있다는 논리.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from within_game_state import add_within_game_features  # noqa: E402
from baseline_catboost import L2_LEAF_REG  # noqa: E402
from eda import load  # noqa: E402

HAZARD_FEATURES = FEATURES + ["pitch_count_before"]
CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]


def fit_hazard_sub_pcb(train_df, target_col):
    y = train_df[target_col].to_numpy()
    pool = Pool(train_df[HAZARD_FEATURES], y, cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss",
        l2_leaf_reg=L2_LEAF_REG, random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(pool)
    return model


def rmo_meta_with_pcb(train_df, valid_df):
    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])
    qR_model = fit_hazard_sub_pcb(rmo_train, "reverse_label")
    not_reverse = rmo_train[rmo_train["reverse_label"] == 0]
    qM_model = fit_hazard_sub_pcb(not_reverse, "middle_label")
    not_rm = not_reverse[not_reverse["middle_label"] == 0]
    not_rm = not_rm[not_rm["outside_label"].isin([0, 1])]
    qO_model = fit_hazard_sub_pcb(not_rm, "outside_label")

    valid_pool = Pool(valid_df[HAZARD_FEATURES], cat_features=CAT_FEATURES)
    qR = qR_model.predict_proba(valid_pool)[:, 1]
    qM = qM_model.predict_proba(valid_pool)[:, 1]
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

    meta_pcb = rmo_meta_with_pcb(train_df, valid_df)
    X_new = X.copy()
    for c in CHAMPION_META:
        X_new[f"rmo_{c}"] = meta_pcb[c]
    new_score, new_pred = apply_corrector(X_new, residual, seg3, base_pred, y, valid_df)
    print(f"  champion(hazard 입력에 pitch_count_before 추가): {new_score:.2f}")

    return new_score


def main():
    df = add_rmo_labels(load("train.csv"))
    df = add_within_game_features(df)
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 (비교 기준: E030 champion PRIMARY~809-818/STRESS~801-810, 실행마다 미세 변동) =====")
    print(f"PRIMARY: {r1:.2f}")
    print(f"STRESS:  {r2:.2f}")


if __name__ == "__main__":
    main()
