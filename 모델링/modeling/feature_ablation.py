"""
matchup 3종 개별 ablation + season categorical 단독 테스트.

catboost_matchup.py에서 3개를 한꺼번에 넣은 결과(703.10)가 baseline(734.49) 대비 미달이었는데,
셋 중 어느 게 원인인지 분리해서 봐야 한다는 피드백에 따라 단독으로 하나씩 추가해서 비교한다.
season categorical은 pitcher_id/batter_id(C)가 아니라 baseline 위에 독립적으로 얹어서 테스트한다
(season은 player identity와 무관한 별개 가설이라 밑바탕을 C로 두면 안 된다는 지적 반영).

전부 single-split(2019-23→24) 기준, baseline_catboost.py의 학습 함수를 재사용한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402

# catboost_matchup import 시 모듈 전역 bc.FEATURES/CAT_FEATURES가 matchup 버전으로 바뀌므로
# 원본을 미리 캡처해둔다.
BASE_FEATURES = list(bc.FEATURES)
BASE_CAT_FEATURES = list(bc.CAT_FEATURES)

from catboost_matchup import add_matchup_features  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def run(df, extra_features, extra_cat, label):
    bc.FEATURES = BASE_FEATURES + extra_features
    bc.CAT_FEATURES = BASE_CAT_FEATURES + extra_cat
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    metrics = bc.evaluate(model, valid_df)
    print(f"{label}: score={metrics['score (리더보드 산식)']:.2f} brier={metrics['brier']:.6f}")


def main():
    df = add_matchup_features(load("train.csv"))

    run(df, ["team_matchup"], ["team_matchup"], "baseline + team_matchup")
    run(df, ["hand_matchup"], ["hand_matchup"], "baseline + hand_matchup")
    run(df, ["count_state"], ["count_state"], "baseline + count_state")
    run(df, [], ["season"], "baseline + season(categorical)")


if __name__ == "__main__":
    main()
