"""
baseline + 계층형 EB 블렌드, discovery(2022+2023)에서 가중치 잠그고 confirmation(2024)은 안 건드린 채 검증.

residual_correlation_eb.py에서 2024를 보고 고른 w=0.3이 +9.16(742.14→751.30)이었는데, 이건 confirmation
연도를 가중치 선택에 써버린 것이라 못 믿는다(다른 참가자 README가 강조한, 그리고 우리가 하루 종일 지킨
"discovery에서 잠그고 confirmation은 순수 확인만" 원칙 위반). 여기서는 그 원칙대로 다시 잰다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from hierarchical_eb_features import add_hier_features, EB_FEATURES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load, TARGET  # noqa: E402

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def get_preds(raw, target_year):
    """(baseline 예측, EB 예측, 정답) — target_year를 primary로 조기종료해서 라운드수 고정, 그 라운드수로 재학습."""
    train_df, valid_df = bc.time_split(raw, target_year)
    valid_r = valid_df[valid_df["game_type"] == "R"]

    m_base = bc.train_catboost(train_df, valid_r)
    it_base = m_base.get_best_iteration() + 1
    m_base_fixed = bc.train_catboost_fixed(train_df, it_base)
    p_base = m_base_fixed.predict_proba(bc.to_pool(valid_r, with_label=False))[:, 1]

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
    return p_base, p_eb, y


def score_of(p, y):
    b = bc.brier_score(y, p)
    r = y.mean()
    return max(0.0, 100000 * (1 - b / (r * (1 - r))))


def main():
    raw = load("train.csv")
    preds = {y: get_preds(raw, y) for y in (2022, 2023, 2024)}

    print("discovery grid (2022, 2023) — 여기서만 보고 가중치 고른다")
    rows = []
    for w in WEIGHTS:
        s22 = score_of((1 - w) * preds[2022][0] + w * preds[2022][1], preds[2022][2])
        s23 = score_of((1 - w) * preds[2023][0] + w * preds[2023][1], preds[2023][2])
        rows.append((w, s22, s23, min(s22, s23), (s22 + s23) / 2))
        print(f"  w_eb={w}: 2022={s22:.2f} 2023={s23:.2f} min={min(s22, s23):.2f} mean={(s22 + s23) / 2:.2f}")

    best = max(rows, key=lambda r: r[3])  # worst-case(min) 기준으로 고른다 (다른 참가자식 robust rule)
    locked_w = best[0]
    print(f"\nLOCKED_W (discovery worst-case 기준) = {locked_w}")

    p24 = (1 - locked_w) * preds[2024][0] + locked_w * preds[2024][1]
    s24 = score_of(p24, preds[2024][2])
    s24_base = score_of(preds[2024][0], preds[2024][2])
    print(f"\n===== CONFIRMATION (2024, 한 번도 안 본 값) =====")
    print(f"baseline(w=0): {s24_base:.2f}")
    print(f"locked blend(w={locked_w}): {s24:.2f}")
    print(f"차이: {s24 - s24_base:+.2f}")


if __name__ == "__main__":
    main()
