"""
CatBoost 베이스라인 + 중복 변수 제거 + rolling OOT 검증.

결측치 전처리를 하지 않는다 — CatBoost가 수치형 NaN(내부적으로 최솟값보다 작은 값으로 취급해 분기 학습)과
범주형(내부적으로 결측을 하나의 카테고리로 처리)을 둘 다 네이티브로 처리하기 때문.

검증은 rolling out-of-time: 2019-21→22, 2019-22→23, 2019-23→24 세 폴드를 0.2/0.3/0.5로 가중평균한다
단일 폴드(2024만)로는 검증 모델과 제출 모델(전체 데이터
재학습)이 서로 다른 데이터로 학습돼서 로컬 점수를 못 믿는 문제가 있었는데, rolling 평균은 그 문제를 줄여준다.

regime 파생 피처(is_extra_inning 등 4개), recency weighting, 완전 중복 변수 제거(run_total_before 등)
셋 다 실험해봤지만 2024 폴드 기준으로 전부 오히려 떨어졌다 (734.49 → 719.5 / 694.91 / 709.35).
중복 변수 제거가 안 먹힌 이유: 트리는 계수를 안 써서 다중공선성에 안 불안정해지고, 오히려
run_total_before 같은 미리 계산된 합계가 depth로 제한된 트리한테는 "지름길" 역할을 한다.
그래서 CatBoost는 원래 44개 피처 그대로 쓴다. 중복 변수 제거는 선형모델(elastic_net.py)에 적용한다 —
거기서는 다중공선성이 실제로 계수 추정을 불안정하게 만드는 문제라 의미가 있다.

피처/스키마는 eda.py의 것을 그대로 재사용한다 (중복 정의 방지).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import (  # noqa: E402
    TRAIN_TEST_NUMERIC, TRAIN_TEST_BINARY, TRAIN_TEST_CATEGORICAL, TARGET, load,
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
VALID_SEASON = 2024  # 단일 폴드 검증(기존 방식, 하위 호환용)에서 쓰는 기준 연도

# rolling out-of-time 폴드: (검증 연도, 가중치). 그 이전 연도 전부가 각 폴드의 학습 데이터.
ROLLING_FOLDS = [(2022, 0.2), (2023, 0.3), (2024, 0.5)]

# CatBoost는 리프 값에 대한 L2만 지원 (L1/L0는 옵션 자체가 없음).
# l2_leaf_reg 3.0(기본값)~100.0 단일 OOT 분할로 스윕한 결과: 3.0=698.78, 10=714.9, 15=734.49,
# 30=715.21, 50=715.44, 100=711.75점. 15 부근이 국지 최적점이라 채택 (단일 분할 기준이라 노이즈 여지 있음).
L2_LEAF_REG = 15.0

# 시계열형 중 순환 없는 것(수치로 사용) / 순환하는 것(범주로 사용) — eda.py EDA 결론과 동일
TIME_SERIES_NUMERIC = ["season", "inning"]
TIME_SERIES_CATEGORICAL = ["game_month", "game_dayofweek"]

# CatBoost는 트리 기반이라 다중공선성에 안 불안정해지고, run_total_before 같은 파생 합계가
# depth로 제한된 트리한테는 "지름길" 역할을 해서 오히려 도움이 된다 (직접 확인: 제거하니 734.49→709.35).
# 그래서 완전 중복 변수도 그대로 다 둔다. 선형모델(elastic_net.py)에서는 반대로 제거한다.
FEATURES = TRAIN_TEST_NUMERIC + TIME_SERIES_NUMERIC + TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL + TIME_SERIES_CATEGORICAL
CAT_FEATURES = TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL + TIME_SERIES_CATEGORICAL


def time_split(df: pd.DataFrame, valid_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"] < valid_season]
    valid = df[df["season"] == valid_season]
    return train, valid


def to_pool(df: pd.DataFrame, with_label: bool = True) -> Pool:
    X = df[FEATURES]
    y = df[TARGET] if with_label else None
    return Pool(X, y, cat_features=CAT_FEATURES)


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def calibration_slope(y_true: np.ndarray, p: np.ndarray) -> float:
    """예측확률 p로 y_true를 로지스틱 회귀했을 때의 기울기. 1에 가까울수록 보정이 잘 된 것."""
    from sklearn.linear_model import LogisticRegression

    logit_p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(logit_p / (1 - logit_p)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y_true)
    return float(lr.coef_[0][0])


def train_catboost(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> CatBoostClassifier:
    train_pool = to_pool(train_df)
    valid_pool = to_pool(valid_df)

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="BrierScore",
        l2_leaf_reg=L2_LEAF_REG,
        early_stopping_rounds=100,
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def train_final_full(df: pd.DataFrame, best_iteration: int) -> CatBoostClassifier:
    """실제 제출용 최종 모델. 검증 없이 전체 데이터로, 검증에서 찾은 iteration 수만큼 고정 학습."""
    full_pool = to_pool(df)
    model = CatBoostClassifier(
        iterations=best_iteration,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="BrierScore",
        l2_leaf_reg=L2_LEAF_REG,
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )
    model.fit(full_pool)
    return model


def evaluate(model: CatBoostClassifier, valid_df: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    valid_pool = to_pool(valid_df, with_label=False)
    p = model.predict_proba(valid_pool)[:, 1]
    y = valid_df[TARGET].to_numpy()

    r = y.mean()
    brier = brier_score(y, p)
    baseline_brier = r * (1 - r)
    score = max(0.0, 100000 * (1 - brier / baseline_brier))

    return {
        "n": len(valid_df),
        "r (실제 성공률)": round(r, 4),
        "brier": round(brier, 6),
        "baseline_brier (r(1-r))": round(baseline_brier, 6),
        "score (리더보드 산식)": round(score, 2),
        "auc": round(roc_auc_score(y, p), 4),
        "calibration_slope": round(calibration_slope(y, p), 4),
    }


def feature_interaction_ranking(model: CatBoostClassifier, top_n: int = 25) -> pd.DataFrame:
    """CatBoost 네이티브 pairwise interaction 강도 (type='Interaction').
    SHAP interaction values보다 훨씬 빠르다 — 트리 분할 구조에서 바로 계산되고 데이터 샘플이 필요 없다.
    선형모델(elastic_net.py)이 놓치고 있는 교호작용이 뭔지 찾는 용도."""
    inter = model.get_feature_importance(type="Interaction")
    rows = [
        {"feature_1": FEATURES[int(f1)], "feature_2": FEATURES[int(f2)], "strength": score}
        for f1, f2, score in inter
    ]
    return pd.DataFrame(rows).sort_values("strength", ascending=False).head(top_n).reset_index(drop=True)


def rolling_oot_evaluate(df: pd.DataFrame, folds: list[tuple[int, float]] = ROLLING_FOLDS) -> dict:
    """폴드별로 (그 이전 연도 전부로 학습 → 해당 연도 검증) 하고 가중평균.

    주의: 폴드마다 조기종료를 독립적으로 시키면 2023 폴드가 무너진다 — game_type=F의 성공률이
    2019~2022엔 높다가 2023에 정확히 반전되는데(EDA2.ipynb에서 직접 확인),
    2019~2022 데이터만으로 학습한 모델은 이 반전을 예측할 근거가 전혀 없다(out-of-distribution).
    그래서 이 함수는 진단용으로만 쓰고, 실제 비교는 rolling_oot_evaluate_fixed()로 한다.
    """
    per_fold = {}
    weighted_brier = 0.0
    weighted_score = 0.0
    for valid_season, weight in folds:
        train_df, valid_df = time_split(df, valid_season)
        model = train_catboost(train_df, valid_df)
        m = evaluate(model, valid_df)
        m["weight"] = weight
        m["best_iteration"] = model.get_best_iteration()
        per_fold[valid_season] = m
        weighted_brier += weight * m["brier"]
        weighted_score += weight * m["score (리더보드 산식)"]

    per_fold["weighted"] = {"brier": round(weighted_brier, 6), "score": round(weighted_score, 2)}
    return per_fold


def train_catboost_fixed(train_df: pd.DataFrame, iterations: int) -> CatBoostClassifier:
    """조기종료 없이 고정 라운드 수로 학습. rolling 폴드 간 공정 비교 + 최종 전체 재학습에 사용."""
    pool = to_pool(train_df)
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="BrierScore",
        l2_leaf_reg=L2_LEAF_REG,
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )
    model.fit(pool)
    return model


def rolling_oot_evaluate_fixed(
    df: pd.DataFrame, iterations: int, folds: list[tuple[int, float]] = ROLLING_FOLDS
) -> dict:
    """모든 폴드를 같은 iterations로 고정 학습해서 공정 비교 (폴드별 조기종료 self-selection 문제 회피)."""
    per_fold = {}
    weighted_brier = 0.0
    weighted_score = 0.0
    for valid_season, weight in folds:
        train_df, valid_df = time_split(df, valid_season)
        model = train_catboost_fixed(train_df, iterations)
        m = evaluate(model, valid_df)
        m["weight"] = weight
        per_fold[valid_season] = m
        weighted_brier += weight * m["brier"]
        weighted_score += weight * m["score (리더보드 산식)"]

    per_fold["weighted"] = {"brier": round(weighted_brier, 6), "score": round(weighted_score, 2)}
    return per_fold


def fit_numeric_scaler(train_df: pd.DataFrame, cols: list[str] | None = None) -> StandardScaler:
    """수치형 StandardScaler. CatBoost 파이프라인에는 안 씀(트리는 스케일링 불필요) —
    나중에 NN 계열을 앙상블에 추가할 때 쓰려고 준비해두는 유틸. NaN은 중앙값으로 임시 대체 후 fit
    (NN 트랙에서는 결측 처리를 별도로 설계해야 함 — median이 최종 답은 아님, EDA.md 부록C 참고).
    """
    cols = cols or (TRAIN_TEST_NUMERIC + TIME_SERIES_NUMERIC)
    scaler = StandardScaler()
    scaler.fit(train_df[cols].fillna(train_df[cols].median()))
    return scaler


def apply_numeric_scaler(df: pd.DataFrame, scaler: StandardScaler, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or (TRAIN_TEST_NUMERIC + TIME_SERIES_NUMERIC)
    out = df.copy()
    out[cols] = scaler.transform(df[cols].fillna(df[cols].median()))
    return out


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    df = load("train.csv")
    print(f"피처 {len(FEATURES)}개, 그중 범주형 {len(CAT_FEATURES)}개")

    # 1) 2024를 주 기준(가장 최신 시즌이라 가장 신뢰)으로 조기종료 라운드 수를 정한다.
    train_df, valid_df = time_split(df, VALID_SEASON)
    primary_model = train_catboost(train_df, valid_df)
    best_iteration = primary_model.get_best_iteration() + 1
    print(f"\n[기준 라운드 수] 2019-23→24 조기종료 결과: best_iteration={best_iteration}")

    # 2) 그 라운드 수를 고정해서 세 폴드를 공정 비교 (폴드별 독립 조기종료는 2023에서 붕괴함 — 직접 확인함).
    print(f"\n[Rolling OOT 검증, iterations={best_iteration} 고정] 2019-21→22, 2019-22→23, 2019-23→24, 가중치 0.2/0.3/0.5")
    fold_results = rolling_oot_evaluate_fixed(df, best_iteration)
    for season, m in fold_results.items():
        print(f"  {season}: {m}")

    print(f"\n[최종 모델] 2019~2024 전체 데이터, iterations={best_iteration}으로 재학습")
    final_model = train_final_full(df, best_iteration)

    final_model_path = MODEL_DIR / "catboost_baseline.cbm"
    final_model.save_model(str(final_model_path))
    print(f"saved final (submission) model to {final_model_path}")

    imp = pd.Series(
        final_model.get_feature_importance(to_pool(df)), index=FEATURES
    ).sort_values(ascending=False)
    print("\n[피처 중요도 상위 20]")
    print(imp.head(20))


if __name__ == "__main__":
    main()
