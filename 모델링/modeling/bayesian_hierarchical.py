"""
STEP 5: Beta-Binomial 계층 스무딩을 CatBoost feature가 아니라 그 자체로 확률을 만드는 별도 모델로.

uncertainty feature(smoothed_rate를 CatBoost 입력으로)는 이미 실패했지만, 이건 메커니즘이 다르다 —
CatBoost가 이 정보를 "어떻게 쓸지" 학습하는 게 아니라, 사람이 직접 log-odds 가산 모델(global +
pitcher effect + batter effect + game_type effect + hand_matchup effect, 전부 Beta-Binomial로
표본수에 따라 스무딩)로 확률을 만든다. asof_pitcher_n/success_rate 등 이미 leak-safe하게 제공되는
컬럼을 그대로 재사용.

evaluate: 2024 holdout standalone Brier + CatBoost와의 residual correlation(블렌드 가치 확인).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

K_PITCHER = 20.0
K_BATTER = 20.0
K_GAME_TYPE = 50.0
K_HAND = 50.0


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def group_effect(train_df, valid_df, group_col, k, global_rate):
    """train_df로 그룹별 스무딩된 rate를 학습하고, valid_df에 매핑(못 본 그룹은 global_rate)."""
    g = train_df.groupby(group_col)[bc.TARGET].agg(["mean", "count"])
    smoothed = (g["mean"] * g["count"] + k * global_rate) / (g["count"] + k)
    train_map = train_df[group_col].map(smoothed).fillna(global_rate)
    valid_map = valid_df[group_col].map(smoothed).fillna(global_rate)
    return train_map, valid_map


def main():
    df = load("train.csv")
    train_df, valid_df = bc.time_split(df, 2024)
    y_train, y_valid = train_df[bc.TARGET].to_numpy(), valid_df[bc.TARGET].to_numpy()
    global_rate = y_train.mean()

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["hand_matchup"] = train_df["pitcher_hand"].astype(str) + "_" + train_df["batter_hand"].astype(str)
    valid_df["hand_matchup"] = valid_df["pitcher_hand"].astype(str) + "_" + valid_df["batter_hand"].astype(str)

    logit_pred_train = np.full(len(train_df), logit(global_rate))
    logit_pred_valid = np.full(len(valid_df), logit(global_rate))

    for col, k in [("pitcher_id", K_PITCHER), ("batter_id", K_BATTER), ("game_type", K_GAME_TYPE), ("hand_matchup", K_HAND)]:
        tr_map, va_map = group_effect(train_df, valid_df, col, k, global_rate)
        logit_pred_train = logit_pred_train + (logit(tr_map) - logit(global_rate))
        logit_pred_valid = logit_pred_valid + (logit(va_map) - logit(global_rate))

    p_valid = sigmoid(logit_pred_valid)
    brier = float(np.mean((p_valid - y_valid) ** 2))
    r = y_valid.mean()
    score = max(0.0, 100000 * (1 - brier / (r * (1 - r))))
    print(f"Bayesian hierarchical standalone: brier={brier:.6f} score={score:.2f}")

    # CatBoost와 residual correlation
    cat_model = bc.train_catboost(train_df, valid_df)
    p_cat = cat_model.predict_proba(bc.to_pool(valid_df, with_label=False))[:, 1]
    corr = np.corrcoef(y_valid - p_valid, y_valid - p_cat)[0, 1]
    print(f"corr(CatBoost residual, Bayesian residual) = {corr:.4f}")

    print("\nalpha blend (p_final = (1-a)*p_cat + a*p_bayes):")
    for a in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        p_final = (1 - a) * p_cat + a * p_valid
        b = float(np.mean((p_final - y_valid) ** 2))
        print(f"  a={a}: brier={b:.6f} score={max(0.0, 100000*(1-b/(r*(1-r)))):.2f}")


if __name__ == "__main__":
    main()
