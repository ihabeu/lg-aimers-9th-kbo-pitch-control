"""
asof_pitcher_success_rate / asof_batter_success_rate 상대값 3개 실험.

pitcher_rate_diff = pitcher_rate - batter_rate
pitcher_rate_mean = (pitcher_rate + batter_rate) / 2
pitcher_rate_product = pitcher_rate * batter_rate

Trackman historical profile은 pitcher_id/trackman_id 간 매핑이 아예 안 돼서(값 교집합 0) 중단하고
그 다음 우선순위인 이 실험으로 넘어간다. single-split(2019-23→24) 기준.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)
NEW_FEATURES = ["pitcher_rate_diff", "pitcher_rate_mean", "pitcher_rate_product"]


def add_relative_rate_features(df):
    df = df.copy()
    pr, br = df["asof_pitcher_success_rate"], df["asof_batter_success_rate"]
    df["pitcher_rate_diff"] = pr - br
    df["pitcher_rate_mean"] = (pr + br) / 2
    df["pitcher_rate_product"] = pr * br
    return df


def main():
    df = add_relative_rate_features(load("train.csv"))
    bc.FEATURES = BASE_FEATURES + NEW_FEATURES  # 전부 수치형, cat_features 추가 없음
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    metrics = bc.evaluate(model, valid_df)
    print(f"baseline + rate_diff/mean/product: {metrics}")


if __name__ == "__main__":
    main()
