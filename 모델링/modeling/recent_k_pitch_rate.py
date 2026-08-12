"""
LSTM permutation importance에서 [시퀀스 전체]가 압도적 1위였던 신호를, LSTM 없이 가장 단순한 형태
(최근 K개 투구 성공률, leak-safe)로 재현해서 CatBoost baseline에 직접 넣어본다. row_id가 진짜
시간순임을 확인했으므로 가능 (이전엔 (season,month) 단위라 이런 세밀한 최근성 피처를 못 만들었음).

기존 asof_prev1/3/5_game_success_rate는 "경기" 단위 최근성인데, 이건 "투구" 단위 최근성이라 다른 정보.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def add_recent_k_features(df, ks=(5, 10, 20, 50)):
    df = df.copy()
    df["row_num"] = df["row_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("row_num")
    g = df.groupby("pitcher_id")["control_success"]
    for k in ks:
        # shift(1)로 현재 투구 제외, rolling(k)로 직전 k개 평균 (leak-safe)
        df[f"recent_{k}_pitch_rate"] = g.transform(lambda s: s.shift(1).rolling(k, min_periods=max(1, k // 2)).mean())
    return df.sort_index()


def main():
    df = add_recent_k_features(load("train.csv"))
    ks = [5, 10, 20, 50]
    cols = [f"recent_{k}_pitch_rate" for k in ks]
    print(df[cols].describe())
    print("\nNaN 비율:", df[cols].isna().mean().to_dict())

    BASE = list(bc.FEATURES)
    for k in ks:
        bc.FEATURES = BASE + [f"recent_{k}_pitch_rate"]
        train_df, valid_df = bc.time_split(df, 2024)
        model = bc.train_catboost(train_df, valid_df)
        m = bc.evaluate(model, valid_df)
        print(f"+ recent_{k}_pitch_rate: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")

    bc.FEATURES = BASE + cols
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"+ 전부(4개): score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")
    print("(비교 기준: baseline 734.49)")


if __name__ == "__main__":
    main()
