"""
T1의 -42.74가 Trackman 정보 자체 때문인지 매핑 품질 때문인지 분리.

HIGH(332명, hand 일치+margin 높음) vs 전체(792명, hand 일치 조건만) 매핑으로 동일한 12개
historical profile 피처를 만들어 같은 조건(single-split 2019-23->24)에서 비교한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from trackman_mapping import build_mapping  # noqa: E402
from trackman_features import add_trackman_history_features, HIST_FEATURES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)


def run(mapping, label):
    df = add_trackman_history_features(load("train.csv"), mapping=mapping)
    bc.FEATURES = BASE_FEATURES + HIST_FEATURES
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label} (매핑 {len(mapping)}명): score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    high = build_mapping(rel_gap_threshold=0.222)
    full = build_mapping(rel_gap_threshold=0.0)  # hand 일치 조건만, margin 무시
    run(high, "HIGH")
    run(full, "전체(hand 일치만)")


if __name__ == "__main__":
    main()
