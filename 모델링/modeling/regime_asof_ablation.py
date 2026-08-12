"""STEP 1~3 ablation: baseline + post2023 -> + R/F 분리 -> + post2023 x R/F. single-split 2019-23->24."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from regime_asof_features import (  # noqa: E402
    add_regime_asof_features, REGIME_FEATURES_1, REGIME_FEATURES_2, REGIME_FEATURES_3,
)
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
    df = add_regime_asof_features(load("train.csv"))
    run(df, REGIME_FEATURES_1, "① + post2023 rate/diff")
    run(df, REGIME_FEATURES_2, "② + R/F 분리 rate")
    run(df, REGIME_FEATURES_3, "③ + post2023 x R/F")
    run(df, REGIME_FEATURES_1 + REGIME_FEATURES_2 + REGIME_FEATURES_3, "① + ② + ③ 전부")


if __name__ == "__main__":
    main()
