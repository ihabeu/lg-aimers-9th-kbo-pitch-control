"""
within-game state (pitch_count_before) 복원.

row_id가 사실상 완벽한 시간순 인덱스임을 확인함(같은 시즌 내 season 100%, game_month 99.999%
non-decreasing). 투수별로 row_id 순 정렬 후 inning이 이전 행보다 감소하는 지점을 게임 경계로 잡으면
경기당 투구수 분포가 실제 KBO 선발투수 평균과 일치(표본 확인: 평균 93.6구, 범위 49~120).

pitch_count_before = 현재 경기에서 이 투구 이전까지 이 투수가 던진 투구 수 (게임 경계 이후 리셋).
불펜 투수(중간 이닝 등판)는 이닝-감소 휴리스틱이 완벽하지 않을 수 있지만, 대부분 선발 위주로도
표본이 충분해서 우선 이 단순 버전으로 검증한다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def add_within_game_features(df):
    df = df.copy()
    df["row_num"] = df["row_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("row_num")

    g = df.groupby("pitcher_id")
    inning_drop = df["inning"] < g["inning"].shift(1)
    df["game_idx"] = inning_drop.groupby(df["pitcher_id"]).cumsum()
    # 게임 내 몇 번째 투구인지 (0-indexed, 현재 투구 이전까지의 카운트)
    df["pitch_count_before"] = df.groupby(["pitcher_id", "game_idx"]).cumcount()

    return df.sort_index()


def main():
    df = add_within_game_features(load("train.csv"))
    print(df["pitch_count_before"].describe())

    bc.FEATURES = list(bc.FEATURES) + ["pitch_count_before"]
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"\n+ pitch_count_before: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")
    print("(비교 기준: baseline 734.49)")


if __name__ == "__main__":
    main()
