"""
사용자 질문: "지금까지 파생변수 실험은 전부 원본 44개 + 파생변수를 얹은 것뿐이었는데,
원본을 지우고 파생변수로 대체하면 어떨까?"

E022(relative_rate_features.py)와 E023(uncertainty_features.py)은 둘 다 이미 만들어진 함수를
그대로 재사용하고, FEATURES 조합 방식만 "추가"에서 "교체"로 바꿔서 재검증한다.

- relative_rate: pitcher_rate_diff = p - b, pitcher_rate_mean = (p+b)/2 는 (asof_pitcher_success_rate,
  asof_batter_success_rate) 2차원의 순수 선형 재매개변수화(45도 회전)라 산술적으로는 정보 손실이
  없다. 그런데 트리는 축에 평행한 분기만 하므로, (diff, mean) 축으로 주면 원본 축으로는 못 찾던
  대각선 경계를 더 쉽게 찾을 수도 있다 -- 그래서 "교체"가 "추가"와 다른 결과를 낼 수 있는 지점이다.
- uncertainty: pitcher_smoothed_rate(EB 스무딩)로 asof_pitcher_success_rate(raw rate)를 대체하면,
  "표본이 적을 때 신뢰도가 낮다"는 정보를 모델이 직접 안 배우고 이미 반영된 값을 받는 셈이다.

전부 single-split(2019-23->24), 기존 실험들과 비교 가능하게.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
import baseline_catboost as bc  # noqa: E402
from eda import load  # noqa: E402
from relative_rate_features import add_relative_rate_features  # noqa: E402
from uncertainty_features import add_uncertainty_features  # noqa: E402

BASE_FEATURES = list(bc.FEATURES)


def run(df, features, label):
    bc.FEATURES = features
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label}: score={m['score (리더보드 산식)']:.2f}  brier={m['brier']:.6f}  (n피처={len(features)})")
    return m["score (리더보드 산식)"]


def main():
    df_rel = add_relative_rate_features(load("train.csv"))
    df_unc = add_uncertainty_features(load("train.csv"))

    print("===== relative_rate: 원본(asof_pitcher/batter_success_rate) 유지 vs 교체 =====")
    run(df_rel, BASE_FEATURES, "baseline(원본 44개)")
    kept_minus2 = [f for f in BASE_FEATURES if f not in ("asof_pitcher_success_rate", "asof_batter_success_rate")]
    run(df_rel, BASE_FEATURES + ["pitcher_rate_diff", "pitcher_rate_mean", "pitcher_rate_product"],
        "추가(44+3, E022 재확인)")
    run(df_rel, kept_minus2 + ["pitcher_rate_diff", "pitcher_rate_mean"],
        "교체(42+diff+mean, 순수 재매개변수화)")
    run(df_rel, kept_minus2 + ["pitcher_rate_diff", "pitcher_rate_mean", "pitcher_rate_product"],
        "교체(42+diff+mean+product)")

    print("\n===== uncertainty: raw rate 유지 vs smoothed_rate로 교체 =====")
    run(df_unc, BASE_FEATURES, "baseline(원본 44개)")
    run(df_unc, BASE_FEATURES + ["pitcher_smoothed_rate", "batter_smoothed_rate"],
        "추가(44+2, E023 스타일)")
    kept_minus_raw = [f for f in BASE_FEATURES if f not in ("asof_pitcher_success_rate", "asof_batter_success_rate")]
    run(df_unc, kept_minus_raw + ["pitcher_smoothed_rate", "batter_smoothed_rate"],
        "교체(42+smoothed, raw rate 삭제)")


if __name__ == "__main__":
    main()
