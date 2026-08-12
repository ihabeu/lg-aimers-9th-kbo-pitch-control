"""
CatBoost 확률 보정(calibration): Platt(logistic) vs Isotonic.

Brier는 (p-y)^2를 직접 벌점으로 주기 때문에 순위가 아니라 확률값 자체의 정확도가 중요하다.
baseline_catboost의 calibration_slope가 1을 벗어나 있어(과소/과대보정 신호) 보정으로 개선 여지가 있는지 확인한다.

Leak-safe 절차: rolling OOT 폴드(2019-21→22, 2019-22→23)의 out-of-fold 예측을 모아 calibrator를 학습하고,
가장 최근 폴드(2019-23→24, 최종 검증 기준과 동일)에 적용해서 보정 전후 Brier/score를 비교한다.
학습에 쓴 적 없는 연도에만 calibrator를 적용하므로 미래 시즌(2025 test) 적용과 동일한 조건이다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_catboost import (  # noqa: E402
    FEATURES, TARGET, time_split, train_catboost_fixed, brier_score, to_pool,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def score(brier: float, r: float) -> float:
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def oof_predict(df: pd.DataFrame, valid_season: int, iterations: int) -> pd.DataFrame:
    train_df, valid_df = time_split(df, valid_season)
    model = train_catboost_fixed(train_df, iterations)
    p = model.predict_proba(to_pool(valid_df, with_label=False))[:, 1]
    return pd.DataFrame({"p": p, "y": valid_df[TARGET].to_numpy(), "season": valid_season})


def main():
    df = load("train.csv")

    # main()과 동일한 기준으로 iterations 고정 (2019-23→24 조기종료 결과, HANDOFF.md 기준 204).
    train_df, valid_df = time_split(df, 2024)
    from baseline_catboost import train_catboost
    primary = train_catboost(train_df, valid_df)
    iterations = primary.get_best_iteration() + 1
    print(f"iterations={iterations}")

    # calibrator 학습용 OOF (2022, 2023) — 2024는 미래 취급, 순수 holdout으로 남긴다.
    oof_fit = pd.concat([oof_predict(df, s, iterations) for s in [2022, 2023]], ignore_index=True)
    # holdout: 2024 (calibrator가 한 번도 못 본 연도, 우리 메인 지표와 동일 조건)
    oof_holdout = oof_predict(df, 2024, iterations)

    p_fit, y_fit = oof_fit["p"].to_numpy(), oof_fit["y"].to_numpy()
    p_hold, y_hold = oof_holdout["p"].to_numpy(), oof_holdout["y"].to_numpy()

    raw_brier = brier_score(y_hold, p_hold)
    r_hold = y_hold.mean()
    print(f"\n[2024 holdout, 보정 전] brier={raw_brier:.6f} score={score(raw_brier, r_hold):.2f}")

    # Platt/logistic: logit(p) -> y 로지스틱 회귀
    logit_fit = np.log(np.clip(p_fit, 1e-6, 1 - 1e-6) / (1 - np.clip(p_fit, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    platt = LogisticRegression().fit(logit_fit, y_fit)
    logit_hold = np.log(np.clip(p_hold, 1e-6, 1 - 1e-6) / (1 - np.clip(p_hold, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    p_platt = platt.predict_proba(logit_hold)[:, 1]
    brier_platt = brier_score(y_hold, p_platt)
    print(f"[2024 holdout, Platt 보정] brier={brier_platt:.6f} score={score(brier_platt, r_hold):.2f}")

    # Isotonic
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_fit, y_fit)
    p_iso = iso.predict(p_hold)
    brier_iso = brier_score(y_hold, p_iso)
    print(f"[2024 holdout, Isotonic 보정] brier={brier_iso:.6f} score={score(brier_iso, r_hold):.2f}")


if __name__ == "__main__":
    main()
