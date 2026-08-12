"""recent_k_pitch_rate(4개 결합) rolling OOT(fixed-iteration) 재검증. baseline과 나란히 비교."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from recent_k_pitch_rate import add_recent_k_features  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)
KS = [5, 10, 20, 50]
RECENT_FEATURES = [f"recent_{k}_pitch_rate" for k in KS]


def run(df, with_recent, label):
    bc.FEATURES = BASE_FEATURES + RECENT_FEATURES if with_recent else BASE_FEATURES
    train_df, valid_df = bc.time_split(df, 2024)
    primary = bc.train_catboost(train_df, valid_df)
    iterations = primary.get_best_iteration() + 1
    print(f"{label}: primary(2024) best_iteration={iterations}")
    fold_results = bc.rolling_oot_evaluate_fixed(df, iterations)
    for season, m in fold_results.items():
        print(f"  {season}: {m}")


def main():
    df = add_recent_k_features(load("train.csv"))
    run(df, False, "baseline")
    run(df, True, "+recent_k_pitch_rate(4개)")


if __name__ == "__main__":
    main()
