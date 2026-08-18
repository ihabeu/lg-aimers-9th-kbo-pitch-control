"""
사용자 요청: 팀 깃헙 검토 때 확인한 "정수 count 재구성" 트릭(rate*n을 그냥 빼면 반올림 오차로
불가능한 값이 나올 수 있어서, round(rate*n)으로 정수 event count를 복원한 뒤 차분한다)을 실제로
적용해서 새 파생변수를 만들어본다.

asof_pitcher_success_rate/middle_rate는 공식 asof_pitcher_prev1/3/5_game_*_rate(최근 N경기)
버전이 이미 있는데, **asof_pitcher_reverse_rate만 커리어 누적치뿐이고 최근 추세 버전이 없다**
(data_description.md 확인). 이 공백을 우리가 직접 채운다 -- 정수 count 재구성으로 leak-safe
"최근 K구 reverse rate"를 만들어서 champion(v12, E030)의 corrector 메타피처에 하나 더 추가.

정수 재구성: reverse_count(i) = round(asof_pitcher_reverse_rate(i) * asof_pitcher_n(i))
(rmo_labels.py가 이미 인접 행 차분에 이 방식을 쓰고 있음 -- 여기선 "최근 K구" 윈도우로 확장).
recent_reverse_rate(i) = (reverse_count(i) - reverse_count(i-K)) / (asof_pitcher_n(i) - asof_pitcher_n(i-K))
표본이 K구 미만이면(투수 이력 초반) 전체 누적 asof_pitcher_reverse_rate로 fallback.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
)
from segment_corrector_rmo_extended_meta import rmo_all_meta  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

K = 50
CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]


def add_recent_reverse_rate(df: pd.DataFrame, k: int = K) -> pd.DataFrame:
    df = df.copy()
    df["row_num"] = df["row_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["pitcher_id", "row_num"])

    df["_reverse_count"] = (df["asof_pitcher_reverse_rate"] * df["asof_pitcher_n"]).round()

    g = df.groupby("pitcher_id")
    count_k_ago = g["_reverse_count"].shift(k)
    n_k_ago = g["asof_pitcher_n"].shift(k)

    window_events = df["_reverse_count"] - count_k_ago
    window_n = df["asof_pitcher_n"] - n_k_ago
    recent_rate = window_events / window_n.replace(0, np.nan)

    # K구 미만 이력(윈도우 자체가 없음)이면 커리어 누적 reverse_rate로 fallback
    df["recent_reverse_rate"] = recent_rate.where(window_n.notna() & (window_n > 0), df["asof_pitcher_reverse_rate"])

    return df.drop(columns=["row_num", "_reverse_count"]).sort_index()


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
    X_champion = X.copy()
    for c in CHAMPION_META:
        X_champion[f"rmo_{c}"] = meta[c]
    champion_score, champion_pred = apply_corrector(X_champion, residual, seg3, base_pred, y, valid_df)
    print(f"  champion(6개 메타피처): {champion_score:.2f}  (기준)")

    recent_rate_full = add_recent_reverse_rate(df)
    recent_rate_valid = recent_rate_full.loc[valid_df.index, "recent_reverse_rate"].to_numpy()
    print(f"  recent_reverse_rate(K={K}) 커버리지: fallback 비율={float((recent_rate_valid == valid_df['asof_pitcher_reverse_rate'].to_numpy()).mean()):.4f}")

    X_new = X_champion.copy()
    X_new["recent_reverse_rate"] = recent_rate_valid
    new_score, new_pred = apply_corrector(X_new, residual, seg3, base_pred, y, valid_df)
    d = (champion_pred - y) ** 2 - (new_pred - y) ** 2
    mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"  +recent_reverse_rate(K={K}): {new_score:.2f}  차이={new_score - champion_score:+.2f}  z={z:.2f}")
    return champion_score, new_score, z


def main():
    df = add_rmo_labels(load("train.csv"))
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    print(f"PRIMARY: champion {r1[0]:.2f} -> +recent_reverse_rate {r1[1]:.2f} (z={r1[2]:.2f})")
    print(f"STRESS:  champion {r2[0]:.2f} -> +recent_reverse_rate {r2[1]:.2f} (z={r2[2]:.2f})")


if __name__ == "__main__":
    main()
