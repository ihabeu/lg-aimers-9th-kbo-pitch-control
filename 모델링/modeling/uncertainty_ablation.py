"""
STEP 1 ablation: Baseline -> +pitcher uncertainty -> +batter uncertainty -> +recent drift -> +all.
single-split 2019-23->24.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from uncertainty_features import add_uncertainty_features, UNCERTAINTY_FEATURES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)


def run(df, extra_features, label):
    bc.FEATURES = BASE_FEATURES + extra_features
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label}: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    df = add_uncertainty_features(load("train.csv"))
    run(df, UNCERTAINTY_FEATURES["pitcher"], "+ pitcher uncertainty (smoothed_rate, uncertainty)")
    run(df, UNCERTAINTY_FEATURES["batter"], "+ batter uncertainty (smoothed_rate, uncertainty)")
    run(df, UNCERTAINTY_FEATURES["recent_drift"], "+ pitcher recent drift")
    all_features = UNCERTAINTY_FEATURES["pitcher"] + UNCERTAINTY_FEATURES["batter"] + UNCERTAINTY_FEATURES["recent_drift"]
    run(df, all_features, "+ all")


if __name__ == "__main__":
    main()
