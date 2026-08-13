"""
1군(R)만 학습 vs 1군+2군(R+F) 학습, 1군만 평가 (2025 test가 전부 1군이므로).

다른 참가자 공개 레포 EDA 노트북은 LightGBM+부분 피처로 "R만 학습"이 유의하게 낫다고 봤는데(z=+2.67),
우리 INSIGHTS.md의 "R/F 완전 분리 모델" 실험(625.68)은 반대로 baseline(734.49)보다 크게 나빴다.
다만 그 실험은 검증 자체가 R+F 섞인 holdout 기준이었을 수 있어 재현성이 불확실하다 — 이 스크립트는
평가를 R만으로 통일해서 "학습 데이터 구성"이라는 변수 하나만 격리해 재확인한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def run(df, r_only_train: bool, label: str) -> dict:
    train_df, valid_df = bc.time_split(df, 2024)
    if r_only_train:
        train_df = train_df[train_df["game_type"] == "R"]
    valid_r = valid_df[valid_df["game_type"] == "R"]

    primary = bc.train_catboost(train_df, valid_r)
    iterations = primary.get_best_iteration() + 1
    print(f"\n{label}: primary(2024, R만 평가) best_iteration={iterations}")

    per_fold = {}
    weighted_brier = 0.0
    weighted_score = 0.0
    for valid_season, weight in bc.ROLLING_FOLDS:
        tr, va = bc.time_split(df, valid_season)
        if r_only_train:
            tr = tr[tr["game_type"] == "R"]
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
    run(df, False, "R+F 학습 (baseline) / R만 평가")
    run(df, True, "R만 학습 / R만 평가")


if __name__ == "__main__":
    main()
