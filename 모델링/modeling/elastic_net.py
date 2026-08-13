"""
Elastic Net (L1+L2) 로지스틱 회귀.

CatBoost(트리)에서는 다중공선성 제거가 오히려 손해였다 (트리는 계수를 안 써서 안 불안정해지고,
run_total_before 같은 파생 합계가 depth-limited 트리한테는 지름길 역할을 했음 — baseline_catboost.py 참고).
선형모델은 정반대다. 상관 높은 변수를 그대로 넣으면 계수 추정이 불안정해지므로, 완전 중복
(|상관계수|=1) 변수는 여기서 제거한다.

선형모델은 CatBoost와 달리 결측치/범주형을 네이티브로 처리 못 해서 준비가 더 필요하다.
- 결측치: asof_* 결측은 전부 "이력 없음"이라는 구조적 원인 3그룹으로 나뉜다 (EDA.md 2절).
  중앙값으로 채우되, 그룹별 결측 플래그(is_pitcher_coldstart 등)를 따로 추가해서 "이력이 없다"는
  정보 자체는 안 잃게 한다. 개별 컬럼마다 결측 플래그를 16개 따로 만들면 그것끼리 또 다중공선성이
  생기니(같은 그룹은 항상 같이 결측) 그룹당 1개씩 3개로 압축한다.
- 범주형: 원핫 인코딩. 이진(값 2개)은 `drop="if_binary"`로 더미변수 함정(완전공선성) 방지.
- 스케일링: StandardScaler. 계수 크기를 비교 가능하게 하고 saga solver 수렴을 빠르게 한다.

참고로 andrew-cui/mlb-game-prediction 레포도 "elastic net"을 표방하지만 실제 코드는
`LogisticRegressionCV()`로 penalty/l1_ratio를 안 넣어서 사실상 기본 L2(릿지)였고 스케일링도
안 했다 — 그대로 참고하지 않고 여기서는 제대로 (penalty="elasticnet", solver="saga", 스케일링 포함) 한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import TRAIN_TEST_NUMERIC, TRAIN_TEST_BINARY, TRAIN_TEST_CATEGORICAL, TARGET, load  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "models"
ROLLING_FOLDS = [(2022, 0.2), (2023, 0.3), (2024, 0.5)]

TIME_SERIES_NUMERIC = ["season", "inning"]
TIME_SERIES_CATEGORICAL = ["game_month", "game_dayofweek"]

# 트리와 반대로 여기서는 완전 중복 변수를 제거한다 (baseline_catboost.py의 판단과 대비됨).
REDUNDANT_DROP = ["run_total_before", "num_runners_on", "away_win_expectancy"]

# game_type의 효과가 2023을 기점으로 부호 자체가 뒤집힌다(EDA2.ipynb) — 선형모델은 변수 하나에
# 계수 하나뿐이라 이 반전을 표현 못 한다. game_type을 단독으로 넣는 대신, season_regime과 합친
# 조합 카테고리로 바꿔서 "R_pre2023"/"R_post2023"/"F_pre2023"/"F_post2023" 4개 값 중 하나로 넣는다
# (따로 두 피처로 넣는 것과 다르다).
NUMERIC_FEATURES = [c for c in TRAIN_TEST_NUMERIC + TIME_SERIES_NUMERIC if c not in REDUNDANT_DROP]
CATEGORICAL_FEATURES = [c for c in (TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL + TIME_SERIES_CATEGORICAL) if c != "game_type"] + ["game_type_regime"]
MISSING_FLAGS = ["is_pitcher_coldstart", "is_batter_coldstart", "is_missing_recent_games"]


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """16개 결측 컬럼을 원인 3그룹으로 압축한 플래그. 같은 그룹은 항상 같이 결측이라
    컬럼별로 16개 다 만들면 서로 100% 겹쳐서(다중공선성) 무의미하다."""
    df = df.copy()
    df["is_pitcher_coldstart"] = df["asof_pitcher_n"].eq(0).astype(int)
    df["is_batter_coldstart"] = df["asof_batter_success_rate"].isna().astype(int)
    df["is_missing_recent_games"] = df["asof_pitcher_prev1_game_success_rate"].isna().astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """game_type과 season_regime을 합친 조합 카테고리. game_type 단독은 이걸로 대체한다."""
    df = df.copy()
    season_regime = np.where(df["season"] >= 2023, "post2023", "pre2023")
    df["game_type_regime"] = df["game_type"].astype(str) + "_" + season_regime
    return df


def time_split(df: pd.DataFrame, valid_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["season"] < valid_season], df[df["season"] == valid_season]


# sklearn LogisticRegression은 L0(0이 아닌 계수 개수 자체에 페널티)는 지원 안 한다 — NP-hard라
# 직접 최적화가 안 되고, 보통 penalty=None(무규제)을 "규제 없음" 기준선으로 놓고 비교한다.
# 여기서는 그 셋을 비교한다: none(L0 대용, 규제 전혀 없음) / l1(Lasso, 순수 희소화) / elasticnet(L1+L2).
PENALTY_VARIANTS = {
    "none": {"penalty": None, "l1_ratio": None},
    "l1": {"penalty": "l1", "l1_ratio": None},
    "elasticnet": {"penalty": "elasticnet", "l1_ratio": 0.5},
}


def build_pipeline(penalty: str = "elasticnet", C: float = 1.0, l1_ratio: float | None = 0.5) -> Pipeline:
    """중앙값 대체를 Pipeline 안(SimpleImputer)에 넣는다 — 밖에서 미리 fillna하면 train/valid를
    합친 뒤 중앙값을 계산하는 꼴이 돼서(prepare()가 분할 전에 불리므로) 검증 데이터 정보가
    학습에 새는 문제가 생긴다. Pipeline 안에 두면 fit()할 때 train 쪽 중앙값만 학습되고,
    predict()할 때는 그 값을 그대로 재사용해서 이 문제가 없다."""
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES + MISSING_FLAGS),
    ])
    clf = LogisticRegression(
        penalty=penalty,
        solver="saga",  # none/l1/elasticnet 전부 지원하는 solver라 세 버전을 동일 조건으로 비교 가능
        l1_ratio=l1_ratio,
        C=C,
        max_iter=1000,  # 300은 ConvergenceWarning 발생 (첫 시도에서 확인)
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocess), ("clf", clf)])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """결측 플래그 + game_type_regime 교호작용 추가 (전부 행 단위 계산이라 누수 없음).
    수치형 결측 대체는 Pipeline 안에서."""
    return add_interaction_features(add_missing_flags(df))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def evaluate(pipe: Pipeline, valid_df: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    X = valid_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + MISSING_FLAGS]
    p = pipe.predict_proba(X)[:, 1]
    y = valid_df[TARGET].to_numpy()

    r = y.mean()
    brier = brier_score(y, p)
    baseline_brier = r * (1 - r)
    score = max(0.0, 100000 * (1 - brier / baseline_brier))

    return {
        "n": len(valid_df), "r": round(r, 4), "brier": round(brier, 6),
        "score": round(score, 2), "auc": round(roc_auc_score(y, p), 4),
    }


def train(train_df: pd.DataFrame, penalty: str = "elasticnet", C: float = 1.0, l1_ratio: float | None = 0.5) -> Pipeline:
    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + MISSING_FLAGS]
    y = train_df[TARGET]
    pipe = build_pipeline(penalty=penalty, C=C, l1_ratio=l1_ratio)
    pipe.fit(X, y)
    return pipe


def compare_penalties(df: pd.DataFrame, valid_season: int = 2024, variants: dict = PENALTY_VARIANTS) -> pd.DataFrame:
    """같은 학습/검증 분할로 none(L0 대용)/l1/elasticnet 세 버전을 비교."""
    train_df, valid_df = time_split(df, valid_season)
    rows = []
    for name, params in variants.items():
        pipe = train(train_df, penalty=params["penalty"], l1_ratio=params["l1_ratio"])
        m = evaluate(pipe, valid_df)
        m["variant"] = name
        n_nonzero = int((pipe.named_steps["clf"].coef_ != 0).sum())
        n_total = pipe.named_steps["clf"].coef_.size
        m["nonzero_coef"] = f"{n_nonzero}/{n_total}"
        rows.append(m)
    return pd.DataFrame(rows).set_index("variant")


def rolling_evaluate(df: pd.DataFrame, penalty: str = "elasticnet", C: float = 1.0, l1_ratio: float | None = 0.5, folds=ROLLING_FOLDS) -> dict:
    per_fold = {}
    weighted_brier = weighted_score = 0.0
    for valid_season, weight in folds:
        train_df, valid_df = time_split(df, valid_season)
        pipe = train(train_df, penalty=penalty, C=C, l1_ratio=l1_ratio)
        m = evaluate(pipe, valid_df)
        m["weight"] = weight
        per_fold[valid_season] = m
        weighted_brier += weight * m["brier"]
        weighted_score += weight * m["score"]
    per_fold["weighted"] = {"brier": round(weighted_brier, 6), "score": round(weighted_score, 2)}
    return per_fold


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    df = prepare(load("train.csv"))
    print(f"수치형 {len(NUMERIC_FEATURES)}개(중복 3개 제거) + 범주형/이진/결측플래그 {len(CATEGORICAL_FEATURES) + len(MISSING_FLAGS)}개")

    print("\n[penalty 비교] none(L0 대용) / l1 / elasticnet, 2019-23 학습 → 2024 검증")
    cmp = compare_penalties(df)
    print(cmp)

    final_pipe = train(df)
    import joblib
    model_path = MODEL_DIR / "elastic_net.joblib"
    joblib.dump(final_pipe, model_path)
    print(f"\nsaved final (full data, elasticnet) model to {model_path}")


if __name__ == "__main__":
    main()
