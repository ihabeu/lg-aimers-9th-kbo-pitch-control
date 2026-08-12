"""
CatBoost + matchup 파생 피처 3종 실험.

- team_matchup: pitcher_team_id + "_" + batter_team_id (투수팀 x 타자팀 조합)
- hand_matchup: pitcher_hand + "_" + batter_hand (좌우 상성)
- count_state: balls_before + "_" + strikes_before (볼카운트 조합, base_state와 같은 방식)

베이스라인(734.49) 대비 미달(single-split 기준)이었지만, 로컬/실제 리더보드 점수가 여러 번 어긋난
전례가 있어 사용자 요청으로 실제 제출용으로도 패키징한다. baseline_catboost.py의 학습 함수를 그대로
재사용하고, FEATURES/CAT_FEATURES에 matchup 3종만 추가한다.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

MATCHUP_FEATURES = ["team_matchup", "hand_matchup", "count_state"]
FEATURES_MATCHUP = bc.FEATURES + MATCHUP_FEATURES
CAT_FEATURES_MATCHUP = bc.CAT_FEATURES + MATCHUP_FEATURES
# train_catboost/to_pool/train_final_full는 모듈 전역 FEATURES/CAT_FEATURES를 참조하므로
# matchup 피처를 포함시키려면 호출 전에 그 전역을 바꿔치기한다.
bc.FEATURES, bc.CAT_FEATURES = FEATURES_MATCHUP, CAT_FEATURES_MATCHUP


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["team_matchup"] = df["pitcher_team_id"].astype(str) + "_" + df["batter_team_id"].astype(str)
    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    df["count_state"] = df["balls_before"].astype(str) + "_" + df["strikes_before"].astype(str)
    return df


def main() -> None:
    df = add_matchup_features(load("train.csv"))
    train_df, valid_df = bc.time_split(df, 2024)

    model = bc.train_catboost(train_df, valid_df)
    metrics = bc.evaluate(model, valid_df)
    print(f"single-split 2024 검증: {metrics}")

    full_model = bc.train_final_full(df, model.get_best_iteration())
    bc.MODEL_DIR.mkdir(exist_ok=True)
    full_model.save_model(str(bc.MODEL_DIR / "catboost_matchup.cbm"))
    print(f"saved {bc.MODEL_DIR / 'catboost_matchup.cbm'} (best_iteration={model.get_best_iteration()}, 전체데이터 재학습)")


if __name__ == "__main__":
    main()
