"""Trackman T4 drift 피처 단독 ablation (single-split 2019-23->24)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from trackman_mapping_v2 import build_mapping_v2  # noqa: E402
from trackman_drift_features import add_drift_features, DRIFT_FEATURES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)


def run(df, extra, label):
    bc.FEATURES = BASE_FEATURES + extra
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label}: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    mapping = build_mapping_v2()
    df = add_drift_features(load("train.csv"), mapping)
    run(df, ["rel_speed_drift"], "+ rel_speed_drift")
    run(df, ["spin_rate_drift"], "+ spin_rate_drift")
    run(df, ["induced_vert_break_drift"], "+ induced_vert_break_drift")
    run(df, DRIFT_FEATURES, "+ all drift")


if __name__ == "__main__":
    main()
