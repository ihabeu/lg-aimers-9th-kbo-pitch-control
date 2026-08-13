"""
다년도 선형 드리프트 외삽으로 예측 평균만 옮기는 calibration (레벨시프트).

기존 calibration.py는 직전 시즌들(2022+2023) OOF에 Platt/Isotonic을 fit해서 다음 해에 적용했는데,
다른 참가자 공개 레포 EDA 노트북 13a장이 이 방식(단일/인접 시즌에 fit한 보정기)은 방향이 해마다
뒤집혀서 안 옮겨간다는 걸 보였다. 여기서는 그 대신 "그 시즌 값 자체"가 아니라 **여러 시즌에 걸친
선형 추세**로 다음 시즌 수준을 외삽하고, 예측 확률에 상수 하나만 더한다(모양은 안 바꾸고 평균만 이동).

shift = extrapolated_rate(target_year) - actual_rate(target_year - 1)
     (전부 target_year 이전 라벨만 사용 — 대회 규칙상 test 행 순서/분포를 쓰면 안 되므로
      실제 제출에서도 이 상수는 test.csv를 전혀 안 보고 학습 데이터만으로 미리 계산된다)

즉 "추세가 말하는 다음 시즌 수준이 직전 시즌 실측치보다 얼마나 다른가"만큼 모든 예측에 똑같이 더한다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load, TARGET  # noqa: E402


def shift_amount(df, target_year: int) -> float:
    hist = df[(df["game_type"] == "R") & (df["season"] < target_year)]
    seasons = sorted(hist["season"].unique())
    rates = hist.groupby("season")[TARGET].mean()
    slope, intercept = np.polyfit(seasons, [rates[s] for s in seasons], 1)
    extrapolated = slope * target_year + intercept
    last_actual = rates[seasons[-1]]
    return float(extrapolated - last_actual)


def run(df, with_shift: bool, label: str) -> dict:
    train_df, valid_df = bc.time_split(df, 2024)
    valid_r = valid_df[valid_df["game_type"] == "R"]
    primary = bc.train_catboost(train_df, valid_r)
    iterations = primary.get_best_iteration() + 1
    print(f"\n{label}: primary(2024, R만 평가) best_iteration={iterations}")

    per_fold = {}
    weighted_brier = 0.0
    weighted_score = 0.0
    for valid_season, weight in bc.ROLLING_FOLDS:
        tr, va = bc.time_split(df, valid_season)
        va_r = va[va["game_type"] == "R"]
        model = bc.train_catboost_fixed(tr, iterations)
        p = model.predict_proba(bc.to_pool(va_r, with_label=False))[:, 1]
        if with_shift:
            s = shift_amount(df, valid_season)
            p = np.clip(p + s, 0, 1)
        else:
            s = 0.0
        y = va_r[TARGET].to_numpy()
        brier = bc.brier_score(y, p)
        r = y.mean()
        base_brier = r * (1 - r)
        m = {
            "n": len(va_r), "shift": round(s, 5), "brier": round(brier, 6),
            "score (리더보드 산식)": round(max(0.0, 100000 * (1 - brier / base_brier)), 2),
            "pred_mean": round(float(p.mean()), 5), "actual_mean": round(float(r), 5),
            "weight": weight,
        }
        per_fold[valid_season] = m
        weighted_brier += weight * brier
        weighted_score += weight * m["score (리더보드 산식)"]
    per_fold["weighted"] = {"brier": round(weighted_brier, 6), "score": round(weighted_score, 2)}
    for season, m in per_fold.items():
        print(f"  {season}: {m}")
    return per_fold


def main():
    df = load("train.csv")
    run(df, False, "baseline (보정 없음)")
    run(df, True, "+레벨시프트 (선형추세 외삽)")


if __name__ == "__main__":
    main()
