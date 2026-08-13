"""
단일 모델(기존 baseline과 동일 구조)은 유지하고, 학습 행만 R은 전부 / F는 2023년 이후(post-break)만 남긴다.

기존 "R/F 비대칭 윈도우(635.96)" 실험은 R/F를 별도 모델 두 개로 쪼갠 상태에서 쟀기 때문에, 그 손해가
"오래된 F를 뺀 것" 때문인지 "모델을 쪼갠 것" 때문인지 분리가 안 됐다. 여기서는 모델 구조는 그대로 두고
학습 행 구성만 바꿔서 그 둘을 분리한다. 평가는 항상 R만(2025 test가 R뿐이므로).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

F_CUTOFF = 2023  # F는 이 시즌부터(포함)만 학습에 남긴다


def filter_rows(df, drop_old_f: bool):
    if not drop_old_f:
        return df
    keep = (df["game_type"] == "R") | (df["season"] >= F_CUTOFF)
    return df[keep]


def run(df, drop_old_f: bool, label: str) -> dict:
    train_df, valid_df = bc.time_split(df, 2024)
    train_df = filter_rows(train_df, drop_old_f)
    valid_r = valid_df[valid_df["game_type"] == "R"]

    primary = bc.train_catboost(train_df, valid_r)
    iterations = primary.get_best_iteration() + 1
    print(f"\n{label}: primary(2024, R만 평가) best_iteration={iterations}, n_train={len(train_df)}")

    per_fold = {}
    weighted_brier = 0.0
    weighted_score = 0.0
    for valid_season, weight in bc.ROLLING_FOLDS:
        tr, va = bc.time_split(df, valid_season)
        tr = filter_rows(tr, drop_old_f)
        va_r = va[va["game_type"] == "R"]
        model = bc.train_catboost_fixed(tr, iterations)
        m = bc.evaluate(model, va_r)
        m["weight"] = weight
        m["n_train"] = len(tr)
        per_fold[valid_season] = m
        weighted_brier += weight * m["brier"]
        weighted_score += weight * m["score (리더보드 산식)"]
    per_fold["weighted"] = {"brier": round(weighted_brier, 6), "score": round(weighted_score, 2)}
    for season, m in per_fold.items():
        print(f"  {season}: {m}")
    return per_fold


def main():
    df = load("train.csv")
    run(df, False, "baseline (R+F 전체 학습) / R만 평가")
    run(df, True, f"R 전체 + F는 {F_CUTOFF}년 이후만 학습 (단일 모델) / R만 평가")


if __name__ == "__main__":
    main()
