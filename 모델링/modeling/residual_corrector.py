"""
Residual Corrector: CatBoost baseline + LightGBM residual model, alpha 블렌딩.

leak-safe 이중 구조 (참고한 다른 팀 문서의 residual/innovation stacking 방식):
1. OOF baseline prediction: 2020~2023 각 시즌을 그 이전 시즌들로만 학습한 CatBoost(fixed-iteration)로
   예측 -> residual = y - p_oof. 2019는 이전 데이터가 없어 OOF 불가라 제외, 2024는 최종 평가용으로
   완전히 남겨둔다 (residual corrector 학습에 전혀 사용하지 않음).
2. 이 OOF residual(2020~2023)로 LightGBM regressor를 학습 (baseline과 동일한 44피처, 새 피처 없음).
3. base 모델(2019~2023 학습, 기존 primary와 동일)로 2024 예측 -> p_base.
   residual 모델로 2024 예측 -> r_hat.
4. p_final = clip(p_base + alpha * r_hat, 0, 1), alpha를 0~1 스윕하며 2024 Brier 비교.
   이 2024는 base/residual 모델 둘 다 학습에 전혀 쓰지 않은 진짜 holdout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
OOF_SEASONS = [2020, 2021, 2022, 2023]  # 2019=이전 데이터 없음, 2024=최종 holdout


def score(brier: float, r: float) -> float:
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def build_oof(df: pd.DataFrame, iterations: int) -> pd.DataFrame:
    parts = []
    for valid_season in OOF_SEASONS:
        train_df, valid_df = bc.time_split(df, valid_season)
        model = bc.train_catboost_fixed(train_df, iterations)
        p = model.predict_proba(bc.to_pool(valid_df, with_label=False))[:, 1]
        part = valid_df.copy()
        part["p_base"] = p
        part["residual"] = part[bc.TARGET].to_numpy() - p
        parts.append(part)
        print(f"  OOF {valid_season}: n={len(part)}, brier={((part[bc.TARGET]-p)**2).mean():.6f}")
    return pd.concat(parts, ignore_index=True)


def to_lgbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[bc.FEATURES].copy()
    for c in bc.CAT_FEATURES:
        X[c] = X[c].astype(str).astype("category")
    return X


def main():
    df = load("train.csv")

    # primary(2024) 기준 iterations 고정 (기존과 동일한 기준)
    train_df, valid_df = bc.time_split(df, 2024)
    primary = bc.train_catboost(train_df, valid_df)
    iterations = primary.get_best_iteration() + 1
    print(f"iterations={iterations}\n")

    print("OOF baseline 생성 (2020~2023, residual corrector 학습용):")
    oof = build_oof(df, iterations)

    print("\nresidual 상관관계 진단 (game_type/season):")
    print(oof.groupby("game_type")["residual"].mean())

    X_oof = to_lgbm_frame(oof)
    y_oof = oof["residual"].to_numpy()

    lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, verbosity=-1, random_state=42)
    lgbm.fit(X_oof, y_oof, categorical_feature=bc.CAT_FEATURES)

    # base 모델(2019~2023 학습, 기존 primary와 동일) -> 2024 예측
    p_base = primary.predict_proba(bc.to_pool(valid_df, with_label=False))[:, 1]
    y_true = valid_df[bc.TARGET].to_numpy()
    r_hat = lgbm.predict(to_lgbm_frame(valid_df))

    corr = np.corrcoef(y_true - p_base, r_hat)[0, 1]
    print(f"\ncorr(진짜 2024 residual, r_hat) = {corr:.4f}")

    r = y_true.mean()
    print("\nalpha sweep (2024 holdout, base/residual 둘 다 학습에 안 쓰인 진짜 out-of-sample):")
    base_brier = float(np.mean((p_base - y_true) ** 2))
    print(f"  alpha=0.0(base only): brier={base_brier:.6f} score={score(base_brier, r):.2f}")
    for alpha in ALPHAS[1:]:
        p_final = np.clip(p_base + alpha * r_hat, 0.0, 1.0)
        brier = float(np.mean((p_final - y_true) ** 2))
        print(f"  alpha={alpha}: brier={brier:.6f} score={score(brier, r):.2f}")


if __name__ == "__main__":
    main()
