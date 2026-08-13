"""
F(dev) 안에서 팀마다 residual 행동이 다른지 진단. 바로 segment로 안 쓰고, 팀 프로필(2023까지의
데이터로만 계산 -- 2024 라벨은 안 봄)이 실제로 2024 residual을 설명하는지부터 확인한다.
설명력이 없으면 여기서 멈추고, 있으면 그 다음에 segment/corrector 반영을 검토한다.

팀 프로필(2023까지만 사용, leak 없음):
  F_ratio, F_n, R_n, F_success_rate, R_success_rate, F_minus_R_gap,
  F_recent_vs_longterm(2023 F rate - 2019~2022 F rate)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def fit_base(train_df, valid_df):
    train_pool = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES)
    valid_pool = Pool(valid_df[FEATURES], valid_df["control_success"], cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
        eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def build_team_profile(history: pd.DataFrame, up_to_season: int) -> pd.DataFrame:
    """history는 season < up_to_season(예: 2024 검증이면 <=2023)까지만. 각 팀이 pitcher/batter로
    관여한 모든 행을 팀 기준으로 모아서 프로필을 만든다."""
    long = pd.concat([
        history.assign(team_id=history["pitcher_team_id"]),
        history.assign(team_id=history["batter_team_id"]),
    ], ignore_index=True)
    recent = long[long["season"] == up_to_season - 1]  # 직전 시즌(예: 2023)
    longterm = long[long["season"] < up_to_season - 1]  # 그 이전(2019~2022)

    rows = []
    for team_id, grp in long.groupby("team_id"):
        f_grp = grp[grp["game_type"] == "F"]
        r_grp = grp[grp["game_type"] == "R"]
        f_n, r_n = len(f_grp), len(r_grp)
        f_rate = f_grp["control_success"].mean() if f_n > 0 else np.nan
        r_rate = r_grp["control_success"].mean() if r_n > 0 else np.nan

        recent_f = recent[(recent["team_id"] == team_id) & (recent["game_type"] == "F")]
        longterm_f = longterm[(longterm["team_id"] == team_id) & (longterm["game_type"] == "F")]
        recent_f_rate = recent_f["control_success"].mean() if len(recent_f) > 0 else np.nan
        longterm_f_rate = longterm_f["control_success"].mean() if len(longterm_f) > 0 else np.nan

        rows.append({
            "team_id": team_id, "F_n": f_n, "R_n": r_n,
            "F_ratio": f_n / max(f_n + r_n, 1),
            "F_success_rate": f_rate, "R_success_rate": r_rate,
            "F_minus_R_gap": f_rate - r_rate if pd.notna(f_rate) and pd.notna(r_rate) else np.nan,
            "F_recent_vs_longterm": recent_f_rate - longterm_f_rate if pd.notna(recent_f_rate) and pd.notna(longterm_f_rate) else np.nan,
        })
    return pd.DataFrame(rows).set_index("team_id")


def diagnose(df, target_season, label):
    print(f"\n===== {label}: target={target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    residual = y - base_pred

    profile = build_team_profile(train_df, target_season)

    f_rows = valid_df[valid_df["game_type"] == "F"].copy()
    f_residual = residual[valid_df["game_type"] == "F"]
    # F 행은 pitcher_team_id가 F를 던지는 쪽 팀 (원래 스키마상 F는 보통 특정 팀 소속 표시)
    f_rows = f_rows.assign(row_residual=f_residual)

    print("팀 프로필 (2023까지):")
    print(profile.round(4).to_string())

    print("\nF 행의 팀별 평균 residual (2024 라벨, 진단용으로만 봄):")
    team_mean_residual = f_rows.groupby("pitcher_team_id")["row_residual"].agg(["size", "mean"])
    print(team_mean_residual.round(4).to_string())

    merged = team_mean_residual.join(profile, how="left")
    merged = merged.dropna(subset=["F_ratio", "mean"])
    print(f"\n프로필 피처와 팀별 평균 residual의 상관관계 (팀 {len(merged)}개, 진단용):")
    for col in ["F_ratio", "F_n", "F_success_rate", "F_minus_R_gap", "F_recent_vs_longterm"]:
        if merged[col].notna().sum() < 4:
            continue
        sub = merged.dropna(subset=[col])
        r, p = stats.pearsonr(sub[col], sub["mean"])
        print(f"  corr({col}, mean_residual) = {r:+.3f} (n={len(sub)}, p={p:.3f})")

    overall_f_score = score(float(np.mean(f_residual ** 2)), float(y[valid_df["game_type"] == "F"].mean()))
    print(f"\nF 전체 base BSS (참고): {overall_f_score:.2f}")


def main():
    df = load("train.csv")
    diagnose(df, 2024, "PRIMARY")
    diagnose(df, 2023, "STRESS")


if __name__ == "__main__":
    main()
