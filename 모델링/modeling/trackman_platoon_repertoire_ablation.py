"""Trackman 마지막 실험 실행: baseline + pitcher_vs_current_hand_{fastball,breaking,offspeed}_rate (single-split 2019-23->24)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from trackman_mapping_v2 import build_mapping_v2  # noqa: E402
from trackman_platoon_repertoire import add_platoon_repertoire_features, PLATOON_REPERTOIRE_FEATURES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def main():
    mapping = build_mapping_v2()
    df = add_platoon_repertoire_features(load("train.csv"), mapping)
    bc.FEATURES = list(bc.FEATURES) + PLATOON_REPERTOIRE_FEATURES
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"+ pitcher_vs_current_hand pitch-mix (3개): score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")
    print("(비교 기준: baseline 734.49)")


if __name__ == "__main__":
    main()
