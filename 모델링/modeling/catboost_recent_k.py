"""
CatBoost + recent_k_pitch_rate(투구 단위 최근성, K=5/10/20/50) 최종 모델.

row_id가 실제 시간순 인덱스임을 확인해서 만든 leak-safe 피처. rolling OOT에서 baseline 대비
2022(+332.46), 2024 primary(+231.69), weighted(+182.34) 전부 큰 폭으로 일관되게 개선 — 이 세션에서
나온 가장 강력한 결과. HANDOFF.md 참고.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from recent_k_pitch_rate import add_recent_k_features  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

KS = [5, 10, 20, 50]
RECENT_K_FEATURES = [f"recent_{k}_pitch_rate" for k in KS]


def main() -> None:
    df = add_recent_k_features(load("train.csv"))
    bc.FEATURES = list(bc.FEATURES) + RECENT_K_FEATURES

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
    full_model.save_model(str(bc.MODEL_DIR / "catboost_recent_k.cbm"))
    print(f"\nsaved {bc.MODEL_DIR / 'catboost_recent_k.cbm'} (전체데이터 재학습)")


if __name__ == "__main__":
    main()
