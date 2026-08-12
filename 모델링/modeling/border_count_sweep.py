"""
3순위: border_count 스윕 (수치형 split 후보 정밀도). asof_* 연속형 rate 피처가 많아서 이론적으로
의미가 있을 수 있다는 가설. 128/254(CPU 기본값)/512로, single-split 2019-23->24.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from catboost import CatBoostClassifier


def run(train_df, valid_df, border_count):
    train_pool = bc.to_pool(train_df)
    valid_pool = bc.to_pool(valid_df)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="BrierScore",
        l2_leaf_reg=bc.L2_LEAF_REG, early_stopping_rounds=100,
        border_count=border_count,
        random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    m = bc.evaluate(model, valid_df)
    print(f"border_count={border_count}: best_iteration={model.get_best_iteration()} "
          f"brier={m['brier']:.6f} score={m['score (리더보드 산식)']:.2f}")


def main():
    df = load("train.csv")
    train_df, valid_df = bc.time_split(df, 2024)
    for b in [128, 254, 512]:
        run(train_df, valid_df, b)


if __name__ == "__main__":
    main()
