"""Failure-mode gated ensemble.

직접 성공확률 모델과 R/M/O 실패유형 모델은 '전문가'일 뿐이며, 어느 쪽을 믿을지는
투수의 표본수·시즌 변화·최근 폼으로 학습한 게이트가 결정한다. 단순 고정 beta innovation과 다르다.
2024에서 투수 단위 2-fold cross-fit으로 검증한다.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modeling'))
sys.path.insert(0, str(ROOT / 'eda'))
from baseline_catboost import FEATURES, CAT_FEATURES  # noqa: E402
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

TARGET = 'control_success'


def bss(y, p):
    r = y.mean()
    return 100000 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1 - r)))


def fit_classifier(frame, label, iterations=350):
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=.04, depth=6, loss_function='Logloss',
        l2_leaf_reg=18., random_seed=2026, thread_count=-1, verbose=False,
        train_dir='/tmp/lg_aimers_catboost',
    )
    model.fit(Pool(frame[FEATURES], frame[label], cat_features=CAT_FEATURES))
    return model


def gate_features(frame, p_direct, p_rmo):
    x = pd.DataFrame()
    for c in [
        'asof_pitcher_n', 'asof_batter_n', 'asof_pitcher_success_rate',
        'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
        'asof_pitcher_prev5_game_success_rate', 'asof_pitcher_reverse_rate',
        'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate',
        'balls_before', 'strikes_before', 'outs_before', 'li', 'inning',
    ]:
        x[c] = pd.to_numeric(frame[c], errors='coerce')
    x['expert_direct'] = p_direct
    x['expert_rmo'] = p_rmo
    x['expert_gap'] = p_direct - p_rmo
    x['recent35'] = (.5*x['asof_pitcher_prev1_game_success_rate'] + .3*x['asof_pitcher_prev3_game_success_rate'] + .2*x['asof_pitcher_prev5_game_success_rate'])
    x['recent_minus_career'] = x['recent35'] - x['asof_pitcher_success_rate']
    x['log_pitcher_n'] = np.log1p(x['asof_pitcher_n'].clip(lower=0))
    return x.replace([np.inf, -np.inf], np.nan).fillna(x.median()).astype('float32')


def main():
    df = add_rmo_labels(load('train.csv'))
    train = df.loc[df.season.lt(2024)].copy()
    valid = df.loc[df.season.eq(2024)].copy()
    pool_valid = Pool(valid[FEATURES], cat_features=CAT_FEATURES)
    y = valid[TARGET].to_numpy(float)

    direct = fit_classifier(train, TARGET, iterations=204)
    p_direct = direct.predict_proba(pool_valid)[:, 1]

    rmo_train = train.dropna(subset=['reverse_label', 'middle_label']).copy()
    q_r = fit_classifier(rmo_train, 'reverse_label')
    nr = rmo_train.loc[rmo_train.reverse_label.eq(0)]
    q_m = fit_classifier(nr, 'middle_label')
    nrm = nr.loc[nr.middle_label.eq(0) & nr.outside_label.isin([0, 1])]
    q_o = fit_classifier(nrm, 'outside_label')
    p_rmo = (1 - q_r.predict_proba(pool_valid)[:, 1]) * (1 - q_m.predict_proba(pool_valid)[:, 1]) * (1 - q_o.predict_proba(pool_valid)[:, 1])

    print(f'direct expert: {bss(y, p_direct):.2f}')
    print(f'R/M/O expert: {bss(y, p_rmo):.2f}')

    X = gate_features(valid, p_direct, p_rmo)
    p_gate = np.zeros(len(valid))
    pitchers = np.array(sorted(valid.pitcher_id.astype(str).unique()))
    rng = np.random.default_rng(2026); rng.shuffle(pitchers)
    left = set(pitchers[:len(pitchers)//2])
    fold = valid.pitcher_id.astype(str).isin(left).to_numpy()
    for held_out in (False, True):
        tr, te = fold != held_out, fold == held_out
        gate = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=.035, max_leaf_nodes=7,
            min_samples_leaf=1200, l2_regularization=25., random_state=2026,
        ).fit(X.loc[tr], y[tr])
        p_gate[te] = gate.predict(X.loc[te])

    print(f'gated ensemble (pitcher-CF): {bss(y, p_gate):.2f}')
    print(f'fixed 10% RMO blend: {bss(y, .9*p_direct + .1*p_rmo):.2f}')


if __name__ == '__main__':
    main()
