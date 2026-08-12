"""
734.49 baseline이 2024 validation에서 어디서 체계적으로 틀리는지 분석.

재학습 없이 기존 primary 모델(2019-23→24) 예측값만 써서 residual = y - p를 여러 상황/선수 축으로
그룹핑, 평균 residual과 표본수를 본다. |평균 residual|이 크고 표본수가 충분한 구간이 baseline이
지속적으로 과대/과소예측하는 지점 -> feature/residual 보정 후보.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

MIN_N = 500  # 이보다 표본 적은 그룹은 노이즈로 보고 제외


def bin_series(s: pd.Series, q: int = 5) -> pd.Series:
    try:
        return pd.qcut(s, q, duplicates="drop").astype(str)
    except ValueError:
        return s.astype(str)


def group_residual(df: pd.DataFrame, by, label: str, top_n: int = 8):
    g = df.groupby(by, observed=True)["residual"].agg(["mean", "count"])
    g = g[g["count"] >= MIN_N].sort_values("mean")
    print(f"\n=== {label} ===")
    print(pd.concat([g.head(top_n), g.tail(top_n)]).to_string())


def main():
    df = load("train.csv")
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)

    p = model.predict_proba(bc.to_pool(valid_df, with_label=False))[:, 1]
    y = valid_df[bc.TARGET].to_numpy()
    v = valid_df.copy()
    v["p"] = p
    v["residual"] = y - p  # 양수: 모델이 과소예측(실제보다 성공확률을 낮게 봄), 음수: 과대예측

    print(f"전체 평균 residual (calibration bias): {v['residual'].mean():.5f}")

    group_residual(v, "game_type", "game_type")
    v["hand_matchup"] = v["pitcher_hand"].astype(str) + "_" + v["batter_hand"].astype(str)
    group_residual(v, "hand_matchup", "pitcher_hand x batter_hand")
    v["count_state"] = v["balls_before"].astype(str) + "_" + v["strikes_before"].astype(str)
    group_residual(v, "count_state", "count (balls_strikes)", top_n=6)
    group_residual(v, "outs_before", "outs_before")
    group_residual(v, "base_state", "base_state")
    v["li_bin"] = bin_series(v["li"])
    group_residual(v, "li_bin", "li (5분위)")
    v["pitcher_recent_diff"] = v["asof_pitcher_prev3_game_success_rate"] - v["asof_pitcher_success_rate"]
    v["pitcher_recent_diff_bin"] = bin_series(v["pitcher_recent_diff"])
    group_residual(v, "pitcher_recent_diff_bin", "최근3경기 - 시즌 성공률 차 (5분위)")
    group_residual(v, "pitcher_team_id", "pitcher_team_id", top_n=5)
    group_residual(v, "batter_team_id", "batter_team_id", top_n=5)
    group_residual(v, "inning", "inning", top_n=6)
    v["score_diff_bin"] = bin_series(v["score_diff_pitcher_team"])
    group_residual(v, "score_diff_bin", "score_diff_pitcher_team (5분위)")


if __name__ == "__main__":
    main()
