"""
STEP 1: Historical rate 고도화 (uncertainty / smoothed rate / recent-career drift).

asof_pitcher_n, asof_batter_n(표본수)은 이미 baseline 44피처에 있어서 중복 생성하지 않는다.
새로 만드는 건 rate 자체를 넘어선 "이 rate를 얼마나 믿을 수 있는가" + "최근이 시즌 평균과 얼마나 다른가":

- pitcher_smoothed_rate / batter_smoothed_rate: empirical Bayes 스무딩 (전체 평균을 prior로, n이 적을수록
  prior 쪽으로 당김). raw success count는 직접 안 주어져서 rate*n으로 추정.
- pitcher_uncertainty / batter_uncertainty: 이항분포 표준오차 sqrt(rate*(1-rate)/n) — n이 작을수록 큼.
- pitcher_recent_drift / batter_recent_drift: 최근 3경기 rate - 시즌 전체 rate.

전부 이미 leak-safe하게 제공되는 asof_* 컬럼으로부터의 결정론적 변환이라 새로 leak-safety를 신경 쓸 필요는
없다 (asof_* 자체가 이미 투구 직전 시점까지의 과거 기록).
"""
import numpy as np
import pandas as pd

GLOBAL_PRIOR_STRENGTH = 20.0  # 의사관측치(pseudo-count) 개수


def add_uncertainty_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    global_pitcher_rate = df["asof_pitcher_success_rate"].mean()
    global_batter_rate = df["asof_batter_success_rate"].mean()

    for prefix, rate_col, n_col, global_rate in [
        ("pitcher", "asof_pitcher_success_rate", "asof_pitcher_n", global_pitcher_rate),
        ("batter", "asof_batter_success_rate", "asof_batter_n", global_batter_rate),
    ]:
        rate, n = df[rate_col], df[n_col]
        success_est = rate * n
        alpha, beta = GLOBAL_PRIOR_STRENGTH * global_rate, GLOBAL_PRIOR_STRENGTH * (1 - global_rate)
        df[f"{prefix}_smoothed_rate"] = (success_est + alpha) / (n + alpha + beta)
        df[f"{prefix}_uncertainty"] = np.sqrt((rate * (1 - rate) / n.replace(0, np.nan)).fillna(1.0))

    df["pitcher_recent_drift"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    return df


UNCERTAINTY_FEATURES = {
    "pitcher": ["pitcher_smoothed_rate", "pitcher_uncertainty"],
    "batter": ["batter_smoothed_rate", "batter_uncertainty"],
    "recent_drift": ["pitcher_recent_drift"],  # batter는 asof_batter_prev*_rate 컬럼이 없어서 스킵 (아래 NOTE)
}

# NOTE: asof_batter_prev1/3/5_game_success_rate 컬럼이 원본 데이터에 없음 (asof_pitcher_*만 제공됨,
# eda/COLUMNS.md 확인 결과). 그래서 batter_recent_drift는 만들 수 없어 이 실험에서 제외.

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
    from eda import load

    df = add_uncertainty_features(load("train.csv"))
    cols = UNCERTAINTY_FEATURES["pitcher"] + UNCERTAINTY_FEATURES["batter"] + UNCERTAINTY_FEATURES["recent_drift"]
    print(df[cols].describe())
