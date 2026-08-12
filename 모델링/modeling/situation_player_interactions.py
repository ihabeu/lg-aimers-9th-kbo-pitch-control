"""
상황 x 선수상태 교호작용 3개. baseline 위에 각각 단독으로 얹어서 본다 (single-split 2019-23->24).

pitcher_li_interaction = asof_pitcher_success_rate * li
pitcher_recent_li_interaction = asof_pitcher_prev5_game_success_rate * li
pitcher_inning_interaction = asof_pitcher_success_rate * inning

CatBoost가 이미 native interaction으로 이런 조합을 상당 부분 잡고 있다는 게 이전 분석(type="Interaction")
결론이라 큰 기대는 안 하지만, 사람이 만들 이유가 명확한 것만 3개로 제한해서 확인한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)


def add_interactions(df):
    df = df.copy()
    df["pitcher_li_interaction"] = df["asof_pitcher_success_rate"] * df["li"]
    df["pitcher_recent_li_interaction"] = df["asof_pitcher_prev5_game_success_rate"] * df["li"]
    df["pitcher_inning_interaction"] = df["asof_pitcher_success_rate"] * df["inning"]
    return df


def run(df, extra_feature, label):
    bc.FEATURES = BASE_FEATURES + [extra_feature]
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label}: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    df = add_interactions(load("train.csv"))
    run(df, "pitcher_li_interaction", "baseline + pitcher_success_rate x li")
    run(df, "pitcher_recent_li_interaction", "baseline + pitcher_recent_success_rate x li")
    run(df, "pitcher_inning_interaction", "baseline + pitcher_success_rate x inning")


if __name__ == "__main__":
    main()
