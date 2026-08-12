"""
CatBoost + hand_matchup(pitcher_hand x batter_hand) 최종 모델.

matchup 3종(team_matchup/hand_matchup/count_state) 중 hand_matchup만 rolling OOT(fixed-iteration)에서
baseline을 일관되게 앞섬: 2024(primary) 734.49→752.89(+18.40), weighted 818.65→839.71(+21.06),
2022도 같은 방향(2257.03→2316.31). residual_analysis.py로 확인한 결과 pitcher_hand x batter_hand 4개
조합의 예측 편향이 뚜렷한 단조 패턴(-0.022 ~ +0.005)을 보여 메커니즘도 설명됨 (HANDOFF.md 참고).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

HAND_MATCHUP_FEATURES = list(bc.FEATURES) + ["hand_matchup"]
HAND_MATCHUP_CAT_FEATURES = list(bc.CAT_FEATURES) + ["hand_matchup"]


def add_hand_matchup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    return df


def main() -> None:
    df = add_hand_matchup(load("train.csv"))
    bc.FEATURES, bc.CAT_FEATURES = HAND_MATCHUP_FEATURES, HAND_MATCHUP_CAT_FEATURES

    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    metrics = bc.evaluate(model, valid_df)
    print(f"single-split 2024 검증: {metrics}")

    fold_results = bc.rolling_oot_evaluate_fixed(df, model.get_best_iteration() + 1)
    print("\nrolling OOT (fixed-iteration):")
    for season, m in fold_results.items():
        print(f"  {season}: {m}")

    full_model = bc.train_final_full(df, model.get_best_iteration() + 1)
    bc.MODEL_DIR.mkdir(exist_ok=True)
    full_model.save_model(str(bc.MODEL_DIR / "catboost_handmatchup.cbm"))
    print(f"\nsaved {bc.MODEL_DIR / 'catboost_handmatchup.cbm'} (전체데이터 재학습)")


if __name__ == "__main__":
    main()
