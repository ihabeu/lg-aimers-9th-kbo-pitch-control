"""
STEP 1: platoon historical feature A/B/C/D 단독 ablation (single-split 2019-23->24).

A: pitcher_vs_current_batter_hand_rate
B: batter_vs_current_pitcher_hand_rate
C: pitcher_platoon_advantage
D: batter_platoon_advantage

hand_matchup(기존 pitcher_hand/batter_hand 관계의 명시적 표현)이 로컬에선 개선, 실제 LB에선 -1.83으로
실패한 사례가 있어서, 이번엔 "새로운 선수별 조건부 이력 정보"라는 다른 가설로 독립적으로 검증한다.
로컬에서 좋아도 바로 제출하지 않고 반복적으로 큰 개선일 때만 rolling OOT까지 확인하기로 함 (HANDOFF.md 참고).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from platoon_features import add_platoon_features  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)


def run(df, feature, label):
    bc.FEATURES = BASE_FEATURES + [feature]  # 전부 수치형, CAT_FEATURES 추가 없음
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label}: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    df = add_platoon_features(load("train.csv"))
    run(df, "pitcher_vs_current_batter_hand_rate", "A: pitcher_vs_current_batter_hand_rate")
    run(df, "batter_vs_current_pitcher_hand_rate", "B: batter_vs_current_pitcher_hand_rate")
    run(df, "pitcher_platoon_advantage", "C: pitcher_platoon_advantage")
    run(df, "batter_platoon_advantage", "D: batter_platoon_advantage")


if __name__ == "__main__":
    main()
