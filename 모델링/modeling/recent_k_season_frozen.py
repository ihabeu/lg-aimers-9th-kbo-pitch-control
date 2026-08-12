"""
recent_k_pitch_rate v2 — train/test에 완전히 동일한 방식으로 적용 가능하게 재설계.

문제였던 점: 기존 버전은 검증(2024) 때 2024 안에서 계속 갱신했지만, 실제 test(2025)에서는 그게
금지라 "작년 말 스냅샷 고정"만 가능했다. 이 불일치 때문에 결측률이 학습(0.1~1.3%) vs 실전(~30%
추정)에서 완전히 달라졌고, CatBoost가 결측 분기를 제대로 학습 못 해서 실전 점수가 폭락했다(252.51).

수정: 처음부터 "직전 시즌 말까지의 최근 K구 성공률"을 기본 단위로 쓴다.
- season S의 모든 행 -> season S 이전(<=S-1)의 마지막 K구로 계산한 스냅샷을 그대로 사용 (그 시즌 안에서는
  갱신 안 함, 실전과 완전히 동일한 방식).
- 그러면 학습 시점에도 실전과 똑같은 수준의 결측(신인/이전 시즌 기록 없음)이 발생 -> CatBoost가 결측
  분기를 제대로 학습하게 됨.
- 그래도 결측인 경우는 asof_pitcher_success_rate(이미 존재하는, 훨씬 촘촘한 leak-safe 피처)로 fallback
  -> 완전한 NaN을 최대한 줄임.

검증은 실전과 동일한 조건으로: 2019~2023 학습 -> 2024 예측 시 "2023년 말 스냅샷"만 사용(2024 안에서
갱신 없음).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

KS = [5, 10, 20, 50]
RECENT_K_FEATURES = [f"recent_{k}_pitch_rate_frozen" for k in KS]


def build_season_end_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """투수별 x 시즌별 "그 시즌 말까지의" 최근 K구 성공률. row_id로 시간순 정렬 후 시즌 경계에서 스냅샷."""
    df = df.copy()
    df["row_num"] = df["row_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("row_num")

    rows = []
    for pid, g in df.groupby("pitcher_id"):
        y = g["control_success"].to_numpy()
        seasons = g["season"].to_numpy()
        for s in np.unique(seasons):
            # 이 시즌 s가 끝나는 시점까지의(= s 시즌 포함) 마지막 K구
            up_to_end_of_s = y[seasons <= s]
            rec = {"pitcher_id": pid, "season_end": s}
            for k in KS:
                tail = up_to_end_of_s[-k:]
                rec[f"recent_{k}_pitch_rate_frozen"] = tail.mean() if len(tail) > 0 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def add_season_frozen_features(df: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """각 행(season S)에는 season_end == S-1 스냅샷을 붙인다 (그 시즌 시작 전 상태, 시즌 내 갱신 없음)."""
    df = df.copy()
    df["season_prev"] = df["season"] - 1
    snap = snapshots.rename(columns={"season_end": "season_prev"})
    merged = df.merge(snap, on=["pitcher_id", "season_prev"], how="left")

    # fallback: 결측이면 asof_pitcher_success_rate로 채움
    for k in KS:
        col = f"recent_{k}_pitch_rate_frozen"
        merged[col] = merged[col].fillna(merged["asof_pitcher_success_rate"])
    return merged


def main():
    df = load("train.csv")
    snapshots = build_season_end_snapshots(df)

    raw_merge = df.copy()
    raw_merge["season_prev"] = raw_merge["season"] - 1
    raw_merge = raw_merge.merge(snapshots.rename(columns={"season_end": "season_prev"}), on=["pitcher_id", "season_prev"], how="left")
    print("fallback 전 원본 결측 비율(실전 상황 재현):", raw_merge[RECENT_K_FEATURES].isna().mean().to_dict())

    df2 = add_season_frozen_features(df, snapshots)
    print("fallback 후 결측 비율(0이어야 함):", df2[RECENT_K_FEATURES].isna().mean().to_dict())

    bc.FEATURES = list(bc.FEATURES) + RECENT_K_FEATURES
    train_df, valid_df = bc.time_split(df2, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"\n실전과 동일 조건(2023년 말 스냅샷 고정) 검증: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")
    print("(비교 기준: baseline 734.49)")


if __name__ == "__main__":
    main()
