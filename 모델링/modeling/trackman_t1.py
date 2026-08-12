"""T1: Baseline + Trackman historical profile (12개, 전부 수치형). single-split(2019-23->24)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from trackman_features import add_trackman_history_features, HIST_FEATURES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)


def main():
    df = add_trackman_history_features(load("train.csv"))
    bc.FEATURES = BASE_FEATURES + HIST_FEATURES  # 전부 수치형, CAT_FEATURES 추가 없음
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    metrics = bc.evaluate(model, valid_df)
    print(f"T1 baseline + Trackman historical profile: {metrics}")


if __name__ == "__main__":
    main()
