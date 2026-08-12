"""
Platoon(상대 손잡이별 과거 성적) 피처.

pitcher_vs_current_batter_hand_rate: 현재 타자의 손(batter_hand)을 상대로 한 해당 투수의 과거 성공률
batter_vs_current_pitcher_hand_rate: 현재 투수의 손(pitcher_hand)을 상대로 한 해당 타자의 과거 성공률
pitcher_platoon_advantage = pitcher_vs_current_batter_hand_rate - asof_pitcher_success_rate
batter_platoon_advantage  = batter_vs_current_pitcher_hand_rate - asof_batter_success_rate

train.csv엔 game_date가 없어 (season, game_month) 단위로만 leak-safe cutoff을 만들 수 있다
(trackman_features.py와 동일한 패턴: 현재 행의 달은 제외, 그 이전 달까지만 누적).
표본 5개 미만이면 이미 leak-safe하게 제공되는 asof_pitcher_success_rate/asof_batter_success_rate로
fallback한다 (직접 만든 overall rate보다 더 정밀한 공식 피처를 재사용).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

MIN_N = 5


def _period(season: pd.Series, month: pd.Series) -> pd.Series:
    return season.astype(int) * 12 + month.astype(int)


def _vs_hand_rate(df: pd.DataFrame, id_col: str, hand_col: str, overall_col: str, out_col: str) -> pd.DataFrame:
    """id_col 선수가 hand_col 상대 손잡이를 만났을 때의 leak-safe 누적 성공률. 표본 부족 시 overall_col로 fallback."""
    df = df.copy()
    df["period"] = _period(df["season"], df["game_month"])

    monthly = (
        df.groupby([id_col, hand_col, "season", "game_month"])
        .agg(n=("control_success", "size"), success=("control_success", "sum"))
        .reset_index()
    )
    monthly["period"] = _period(monthly["season"], monthly["game_month"])
    monthly = monthly.sort_values([id_col, hand_col, "period"])
    monthly[["n_cum", "success_cum"]] = monthly.groupby([id_col, hand_col])[["n", "success"]].cumsum()

    df_sorted = df.sort_values("period")
    monthly_sorted = monthly.sort_values("period")
    merged = pd.merge_asof(
        df_sorted, monthly_sorted[[id_col, hand_col, "period", "n_cum", "success_cum"]],
        on="period", by=[id_col, hand_col], direction="backward", allow_exact_matches=False,
    )
    rate = merged["success_cum"] / merged["n_cum"]
    rate = rate.where(merged["n_cum"] >= MIN_N, merged[overall_col])
    merged[out_col] = rate
    # n_cum/success_cum/period는 중간 산출물이라, 다음 _vs_hand_rate 호출에서 merge_asof 컬럼명이
    # 충돌하지 않도록 여기서 정리하고 최종 컬럼만 남긴다.
    merged = merged.drop(columns=["period", "n_cum", "success_cum"])
    return merged.sort_index()


def add_platoon_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _vs_hand_rate(df, "pitcher_id", "batter_hand", "asof_pitcher_success_rate", "pitcher_vs_current_batter_hand_rate")
    df = _vs_hand_rate(df, "batter_id", "pitcher_hand", "asof_batter_success_rate", "batter_vs_current_pitcher_hand_rate")
    df["pitcher_platoon_advantage"] = df["pitcher_vs_current_batter_hand_rate"] - df["asof_pitcher_success_rate"]
    df["batter_platoon_advantage"] = df["batter_vs_current_pitcher_hand_rate"] - df["asof_batter_success_rate"]
    return df


PLATOON_FEATURES = [
    "pitcher_vs_current_batter_hand_rate",
    "batter_vs_current_pitcher_hand_rate",
    "pitcher_platoon_advantage",
    "batter_platoon_advantage",
]

if __name__ == "__main__":
    df = add_platoon_features(load("train.csv"))
    print(df[PLATOON_FEATURES].describe())
    print("\nNaN 비율:")
    print(df[PLATOON_FEATURES].isna().mean())
