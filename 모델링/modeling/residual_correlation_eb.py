"""
baseline(raw 44피처) CatBoost vs +계층형 EB CatBoost, 2024 primary 폴드에서 residual 상관 확인.

기존 model_diversity.py는 "같은 44피처"를 여러 알고리즘(CatBoost/LightGBM/XGBoost 등)에 먹여서
0.998+ 상관을 확인했다 — 이건 "모델 종류"만 바꾼 비교였다. 여기서는 "피처 표현 자체"가 다른 두 모델
(raw 44피처 vs +계층형 EB 피처, 오늘 hierarchical_eb_features.py로 우리가 직접 만든 것)의 오차가
실제로 덜 겹치는지를 본다 — 덜 겹치면 블렌드/2층 보정으로 벌 여지가 있다는 뜻이고, 여전히 0.99+면
피처 표현을 바꿔도 같은 오차가 난다는 뜻이라 V14류 2층 구조 자체가 이 데이터엔 안 맞는다는 근거가 된다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from hierarchical_eb_features import add_hier_features, EB_FEATURES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load, TARGET  # noqa: E402


def main():
    raw = load("train.csv")
    target_year = 2024

    # baseline: raw 44피처
    train_df, valid_df = bc.time_split(raw, target_year)
    valid_r = valid_df[valid_df["game_type"] == "R"]
    m_base = bc.train_catboost(train_df, valid_r)
    it_base = m_base.get_best_iteration() + 1
    tr_b, va_b = bc.time_split(raw, target_year)
    m_base_fixed = bc.train_catboost_fixed(tr_b, it_base)
    p_base = m_base_fixed.predict_proba(bc.to_pool(valid_r, with_label=False))[:, 1]

    # +계층형 EB
    base_features = list(bc.FEATURES)
    bc.FEATURES = base_features + EB_FEATURES
    frame_eb = add_hier_features(raw, target_year)
    tr_e, va_e = bc.time_split(frame_eb, target_year)
    va_e_r = va_e[va_e["game_type"] == "R"]
    m_eb = bc.train_catboost(tr_e, va_e_r)
    it_eb = m_eb.get_best_iteration() + 1
    m_eb_fixed = bc.train_catboost_fixed(tr_e, it_eb)
    p_eb = m_eb_fixed.predict_proba(bc.to_pool(va_e_r, with_label=False))[:, 1]
    bc.FEATURES = base_features

    y = valid_r[TARGET].to_numpy()
    resid_base = p_base - y
    resid_eb = p_eb - y
    corr = np.corrcoef(resid_base, resid_eb)[0, 1]

    print(f"\nbaseline iterations={it_base}, +EB iterations={it_eb}")
    print(f"baseline brier={bc.brier_score(y, p_base):.6f}, +EB brier={bc.brier_score(y, p_eb):.6f}")
    print(f"pred correlation: {np.corrcoef(p_base, p_eb)[0, 1]:.5f}")
    print(f"residual correlation: {corr:.5f}  (model_diversity.py의 0.998과 비교)")

    # 단순 평균 블렌드가 둘 중 나은 쪽보다 나은지 참고로 확인
    for w in [0.3, 0.5, 0.7]:
        p_blend = (1 - w) * p_base + w * p_eb
        b = bc.brier_score(y, p_blend)
        r = y.mean()
        score = max(0.0, 100000 * (1 - b / (r * (1 - r))))
        print(f"  blend w_eb={w}: brier={b:.6f} score={score:.2f}")


if __name__ == "__main__":
    main()
