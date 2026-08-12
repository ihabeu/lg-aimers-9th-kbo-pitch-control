"""
STEP 1: 2023 regime을 feature가 아니라 "학습 데이터 윈도우가 다른 모델"로 분리.

A: 2020~2022만 학습 -> 2024 예측
B: 2021~2023만 학습 -> 2024 예측
C: 2020~2023(기존 baseline과 동일) -> 2024 예측 (기준선)

그리고 p_blend = w*p_B(최근 위주) + (1-w)*p_C(전체) 를 w 스윕. 전부 동일 iterations(기존 primary
204)로 고정 학습해서 공정 비교 (독립 조기종료는 폴드마다 다른 결과를 주는 문제가 이미 확인됨).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def score(brier: float, r: float) -> float:
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def train_window(df, season_min, season_max, iterations):
    train_df = df[(df["season"] >= season_min) & (df["season"] <= season_max)]
    model = bc.train_catboost_fixed(train_df, iterations)
    return model


def main():
    df = load("train.csv")
    train_df_full, valid_df = bc.time_split(df, 2024)
    y = valid_df[bc.TARGET].to_numpy()
    r = y.mean()

    primary = bc.train_catboost(train_df_full, valid_df)
    iterations = primary.get_best_iteration() + 1
    print(f"iterations={iterations}\n")

    windows = {"A(2020-2022)": (2020, 2022), "B(2021-2023)": (2021, 2023), "C(2020-2023,기존)": (2019, 2023)}
    preds = {}
    for name, (lo, hi) in windows.items():
        model = train_window(df, lo, hi, iterations)
        p = model.predict_proba(bc.to_pool(valid_df, with_label=False))[:, 1]
        b = float(np.mean((p - y) ** 2))
        preds[name] = p
        print(f"{name}: n_train={len(df[(df.season>=lo)&(df.season<=hi)])} brier={b:.6f} score={score(b, r):.2f}")

    print("\nblend: w*p_B(최근위주) + (1-w)*p_C(전체)")
    for w in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        p_blend = w * preds["B(2021-2023)"] + (1 - w) * preds["C(2020-2023,기존)"]
        b = float(np.mean((p_blend - y) ** 2))
        print(f"  w={w}: brier={b:.6f} score={score(b, r):.2f}")


if __name__ == "__main__":
    main()
