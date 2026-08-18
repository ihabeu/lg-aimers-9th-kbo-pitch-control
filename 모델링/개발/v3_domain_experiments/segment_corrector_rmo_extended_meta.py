"""
E029(실제 champion, 911.19)의 R/M/O log-ratio 메타피처(mr, or 2개)를 확장해볼 수 있는지 확인.
E029는 "비율(모양)"만 줬는데, hazard 원시 확률 qR/qM/qO 자체("크기/확신도")는 안 줬다 -- 세 번째
log-ratio(log(qO/qM), 대수적으로는 or-mr과 같은 정보지만 트리는 축-평행이라 명시적으로 주는 게
다를 수 있다 -- E028에서 이미 확인된 패턴)도 마찬가지.

E029의 hazard 서브모델 학습 로직은 그대로 재사용(중복 방지), 반환값만 qR/qM/qO까지 포함하도록
E029의 rmo_logratio_features를 확장한 버전을 여기서 별도로 만든다(원본은 실제 배포 코드라 안 건드림).
비교 기준은 baseline(메타피처 없음)이 아니라 **현재 champion(E029, mr+or 2개)**이다.
"""
import sys
from pathlib import Path

import numpy as np
from catboost import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, fit_hazard_sub, corrector_matrix, assign_segment_3way,
    apply_corrector, pitcher_bootstrap_z, FEATURES, CAT_FEATURES, EPS,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

VARIANTS = {
    "champion(mr+or)": ["mr", "or"],
    "+raw(qR,qM,qO)": ["mr", "or", "qR", "qM", "qO"],
    "+3rd log-ratio(om)": ["mr", "or", "om"],
    "전부(mr,or,om,qR,qM,qO)": ["mr", "or", "om", "qR", "qM", "qO"],
}


def rmo_all_meta(train_df, valid_df):
    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])
    qR_model = fit_hazard_sub(rmo_train, "reverse_label")
    not_reverse = rmo_train[rmo_train["reverse_label"] == 0]
    qM_model = fit_hazard_sub(not_reverse, "middle_label")
    not_rm = not_reverse[not_reverse["middle_label"] == 0]
    not_rm = not_rm[not_rm["outside_label"].isin([0, 1])]
    qO_model = fit_hazard_sub(not_rm, "outside_label")

    valid_pool = Pool(valid_df[FEATURES], cat_features=CAT_FEATURES)
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
    meta = rmo_all_meta(train_df, valid_df)

    champion_score = None
    champion_pred = None
    results = {}
    for name, cols in VARIANTS.items():
        X_meta = X.copy()
        for c in cols:
            X_meta[f"rmo_{c}"] = meta[c]
        s, pred = apply_corrector(X_meta, residual, seg3, base_pred, y, valid_df)
        if champion_score is None:
            champion_score, champion_pred = s, pred
            print(f"  [{name}] {s:.2f}  (기준)")
        else:
            d = (champion_pred - y) ** 2 - (pred - y) ** 2
            mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
            print(f"  [{name}] {s:.2f}  차이(vs champion)={s - champion_score:+.2f}  z={z:.2f}")
        results[name] = s
    return results


def main():
    df = add_rmo_labels(load("train.csv"))
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    for name in VARIANTS:
        print(f"  [{name}] PRIMARY {r1[name]:.2f}  STRESS {r2[name]:.2f}")


if __name__ == "__main__":
    main()
