"""
Trackman 파생 피처 28개 전체(701.15, 실패) importance 순위에서 상위 N개만 골라 baseline 44개에 추가.
baseline 44개는 절대 건드리지 않음(bottom-N 제거가 항상 악화됐던 것과 대칭적으로, Trackman 쪽만 선별).

importance 순위(이미 확인됨, trackman_everything.py 결과):
1 tm_std_horz_break, 2 brk_usage, 3 tm_std_rel_height, 4 tm_std_spin_rate,
5 tm_std_induced_vert_break, 6 brk_spin, 7 fb_brk_velocity_sep, 8 tm_avg_rel_speed,
9 tm_std_rel_speed, 10 fb_off_velocity_sep, 11 repertoire_entropy, 12 tm_std_extension,
13 off_velocity, 14 off_spin, 15 tm_avg_rel_side, ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from trackman_mapping_v3 import get_confident_mapping  # noqa: E402
from trackman_everything import build_everything  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

IMPORTANCE_ORDER = [
    "tm_std_horz_break", "brk_usage", "tm_std_rel_height", "tm_std_spin_rate",
    "tm_std_induced_vert_break", "brk_spin", "fb_brk_velocity_sep", "tm_avg_rel_speed",
    "tm_std_rel_speed", "fb_off_velocity_sep", "repertoire_entropy", "tm_std_extension",
    "off_velocity", "off_spin", "tm_avg_rel_side",
]

BASE_FEATURES = list(bc.FEATURES)


def run(df, extra, label):
    bc.FEATURES = BASE_FEATURES + extra
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"{label} ({len(extra)}개): score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")


def main():
    mapping = get_confident_mapping()
    df, _ = build_everything(load("train.csv"), mapping)

    for n in [3, 5, 10, 15]:
        run(df, IMPORTANCE_ORDER[:n], f"top {n}")


if __name__ == "__main__":
    main()
