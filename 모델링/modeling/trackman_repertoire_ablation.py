"""STEP 2 ablation: usage / physical(velocity+spin) / separation / entropy 단독 -> 전부 결합. single-split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from trackman_mapping_v2 import build_mapping_v2  # noqa: E402
from trackman_repertoire import add_repertoire_features, PITCH_ABBR, PITCH_GROUPS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)
USAGE = [f"{PITCH_ABBR[g]}_usage" for g in PITCH_GROUPS]
PHYSICAL = [f"{PITCH_ABBR[g]}_velocity" for g in PITCH_GROUPS] + [f"{PITCH_ABBR[g]}_spin" for g in PITCH_GROUPS]
SEPARATION = ["fb_brk_velocity_sep", "fb_off_velocity_sep", "fb_brk_spin_sep"]
ENTROPY = ["repertoire_entropy"]


def run(df, extra, label):
    bc.FEATURES = BASE_FEATURES + extra
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label}: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    mapping = build_mapping_v2()
    df = add_repertoire_features(load("train.csv"), mapping)
    run(df, USAGE, "+ usage(3)")
    run(df, PHYSICAL, "+ physical(velocity+spin, 6)")
    run(df, SEPARATION, "+ separation(3)")
    run(df, ENTROPY, "+ entropy(1)")
    run(df, USAGE + PHYSICAL + SEPARATION + ENTROPY, "+ 전부(13)")


if __name__ == "__main__":
    main()
