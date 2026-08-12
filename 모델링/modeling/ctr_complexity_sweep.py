"""
2순위: max_ctr_complexity 스윕. CatBoost가 categorical feature 조합(CTR)을 자동 탐색하는 복잡도를
사람이 수동으로 만든 team_matchup/count_state(둘 다 실패) 대신 얼마나 넓게/좁게 볼지 결정하는 값.
CPU 기본값은 4. 1/2/4(기본)/6/8로 하나씩 (동시에 다른 것 안 바꿈), single-split 2019-23->24.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from catboost import CatBoostClassifier


def run(train_df, valid_df, ctr_complexity):
    train_pool = bc.to_pool(train_df)
    valid_pool = bc.to_pool(valid_df)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="BrierScore",
        l2_leaf_reg=bc.L2_LEAF_REG, early_stopping_rounds=100,
        max_ctr_complexity=ctr_complexity,
        random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    m = bc.evaluate(model, valid_df)
    print(f"max_ctr_complexity={ctr_complexity}: best_iteration={model.get_best_iteration()} "
          f"brier={m['brier']:.6f} score={m['score (리더보드 산식)']:.2f}")


def main():
    df = load("train.csv")
    train_df, valid_df = bc.time_split(df, 2024)
    for c in [1, 2, 4, 6, 8]:
        run(train_df, valid_df, c)


if __name__ == "__main__":
    main()
