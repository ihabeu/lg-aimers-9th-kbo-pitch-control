"""
⑦ Feature selection: baseline 44개에서 하위 importance 5/10/15개 순차 제거.

ranking은 baseline 모델(2019-23->24 single-split) 하나에서 고정하고, 그 ranking 기준으로
39/34/29개 변수 조합을 다시 처음부터 학습해서 비교한다 (importance 낮다고 정보가 없는 게 아니라
interaction에서만 쓰일 수도 있어서, 낮은 importance != 제거해도 안전 이라는 점에 주의).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

FULL_FEATURES = list(bc.FEATURES)
FULL_CAT_FEATURES = list(bc.CAT_FEATURES)


def run(df, features, label):
    bc.FEATURES = features
    bc.CAT_FEATURES = [c for c in FULL_CAT_FEATURES if c in features]
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label} ({len(features)}개): score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")
    return model


def main():
    df = load("train.csv")

    # 1) baseline(44개)에서 importance ranking 고정
    baseline_model = run(df, FULL_FEATURES, "baseline 44개")
    bc.FEATURES, bc.CAT_FEATURES = FULL_FEATURES, FULL_CAT_FEATURES
    imp = pd.Series(
        baseline_model.get_feature_importance(bc.to_pool(df)), index=FULL_FEATURES
    ).sort_values()
    print("\n[importance 하위 15개]")
    print(imp.head(15))

    # 2) 하위 5/10/15 제거
    for k in [5, 10, 15]:
        drop = set(imp.head(k).index)
        features = [f for f in FULL_FEATURES if f not in drop]
        run(df, features, f"\nbottom {k}개 제거")


if __name__ == "__main__":
    main()
