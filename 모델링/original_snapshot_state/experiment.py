"""Snapshot State Ensemble — CatBoost 없이, 2023 -> 2024 OOT로만 검증한다.

각 행이 제공하는 asof 누적값에서 직전 시즌 종료 시점 상태를 빼 당해 시즌 상태를 복원한다.
test의 다른 행을 보지 않으므로 제출 규정을 지킨다.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'data' / 'train.csv'
TARGET = 'control_success'
NUM = [
    'game_month', 'inning', 'balls_before', 'strikes_before', 'outs_before',
    'score_diff_pitcher_team', 'num_runners_on', 'home_win_expectancy', 'li',
    'asof_pitcher_n', 'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate',
    'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate',
    'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
    'asof_pitcher_prev5_game_success_rate', 'asof_pitcher_prev1_game_middle_rate',
    'asof_pitcher_prev3_game_middle_rate', 'asof_pitcher_prev5_game_middle_rate',
    'asof_batter_n', 'asof_batter_success_rate', 'asof_batter_middle_rate',
    'asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
]
CATS = ['game_dayofweek', 'top_bottom', 'game_type', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id']
RATE_COLS = [
    'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate',
    'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate', 'asof_pitcher_fastball_rate',
    'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
]


def score(y, p):
    r = y.mean()
    return 100000 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1 - r)))


def category_maps(source, target):
    out = pd.DataFrame(index=target.index)
    for c in CATS:
        levels = {v: i for i, v in enumerate(source[c].astype('string').fillna('<NA>').unique())}
        out[f'cat__{c}'] = target[c].astype('string').fillna('<NA>').map(levels).fillna(-1).astype('float32')
    return out.reset_index(drop=True)


def season_bank(history):
    """직전 시즌 마지막 asof 스냅샷. target 행 하나만으로 당해 시즌 통계를 복원하기 위한 bank."""
    last = history.sort_values('row_id').groupby('pitcher_id', sort=False).tail(1).set_index('pitcher_id')
    bank = {'n': last['asof_pitcher_n'].to_dict()}
    for c in RATE_COLS:
        bank[c] = (pd.to_numeric(last['asof_pitcher_n'], errors='coerce') * pd.to_numeric(last[c], errors='coerce')).round().to_dict()
    bank['prior'] = float(history[TARGET].mean())
    return bank


def state_features(rows, bank):
    """장기-당해시즌-최근경기 세 시간축을 신뢰도와 함께 만든다."""
    x = rows.reset_index(drop=True)
    pn = pd.to_numeric(x['asof_pitcher_n'], errors='coerce').fillna(0).clip(lower=0).to_numpy(float)
    prev_n = x['pitcher_id'].map(bank['n']).fillna(0).to_numpy(float)
    season_n = np.maximum(pn - prev_n, 0)
    z = pd.DataFrame(index=np.arange(len(x)))
    z['state__season_log_n'] = np.log1p(season_n).astype('float32')
    z['state__season_reliability_30'] = (season_n / (season_n + 30)).astype('float32')
    z['state__season_reliability_100'] = (season_n / (season_n + 100)).astype('float32')
    z['state__career_reliability'] = (pn / (pn + 150)).astype('float32')
    prior = bank['prior']
    for c in RATE_COLS:
        rate = pd.to_numeric(x[c], errors='coerce').fillna(prior).to_numpy(float)
        total = np.rint(pn * rate)
        before = x['pitcher_id'].map(bank[c]).fillna(0).to_numpy(float)
        season_success = np.clip(total - before, 0, season_n)
        raw = np.divide(season_success, season_n, out=np.full(len(x), prior), where=season_n > 0)
        smooth = (season_success + 50 * prior) / (season_n + 50)
        short = (season_success + 15 * prior) / (season_n + 15)
        name = c.removeprefix('asof_pitcher_').removesuffix('_rate')
        z[f'state__season_{name}_raw'] = raw.astype('float32')
        z[f'state__season_{name}_eb50'] = smooth.astype('float32')
        z[f'state__season_{name}_eb15'] = short.astype('float32')
        z[f'state__season_minus_career_{name}'] = (raw - rate).astype('float32')
    recent = (.5 * pd.to_numeric(x['asof_pitcher_prev1_game_success_rate'], errors='coerce').fillna(prior)
              + .3 * pd.to_numeric(x['asof_pitcher_prev3_game_success_rate'], errors='coerce').fillna(prior)
              + .2 * pd.to_numeric(x['asof_pitcher_prev5_game_success_rate'], errors='coerce').fillna(prior))
    z['state__recent35'] = recent.astype('float32')
    z['state__recent_minus_season'] = (recent.to_numpy() - z['state__season_success_eb50'].to_numpy()).astype('float32')
    z['state__recent_minus_career'] = (recent.to_numpy() - pd.to_numeric(x['asof_pitcher_success_rate'], errors='coerce').fillna(prior).to_numpy()).astype('float32')
    return z


def matrix(source, target, bank):
    raw = target[NUM].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    return pd.concat([raw, state_features(target, bank), category_maps(source, target)], axis=1)


def run_fold(df, source_year=2023, target_year=2024):
    source = df.loc[df.season.eq(source_year)].copy()
    target = df.loc[df.season.eq(target_year)].copy()
    history = df.loc[df.season.lt(source_year)].copy()
    bank_source = season_bank(history)
    bank_target = season_bank(df.loc[df.season.lt(target_year)])
    Xs, Xt = matrix(source, source, bank_source), matrix(source, target, bank_target)
    y, yt = source[TARGET].to_numpy(float), target[TARGET].to_numpy(float)

    # Expert A: 최근 시즌의 비선형 상태 모델.
    lgbm = lgb.LGBMRegressor(
        objective='regression_l2', n_estimators=350, learning_rate=.03, num_leaves=15,
        min_child_samples=500, colsample_bytree=.85, reg_alpha=3., reg_lambda=12.,
        max_bin=127, random_state=2026, verbosity=-1, n_jobs=-1,
    ).fit(Xs, y)
    p_lgb = np.clip(lgbm.predict(Xt), .001, .999)

    # Expert B: 같은 상태를 다른 분할 방식으로 학습해 앙상블 다양성을 만든다.
    hgb = make_pipeline(
        SimpleImputer(strategy='median'),
        HistGradientBoostingRegressor(max_iter=220, learning_rate=.045, max_leaf_nodes=15,
                                     min_samples_leaf=500, l2_regularization=8., random_state=314),
    ).fit(Xs, y)
    p_hgb = np.clip(hgb.predict(Xt), .001, .999)

    print(f'LGB state expert: {score(yt, p_lgb):.2f}')
    print(f'HGB state expert: {score(yt, p_hgb):.2f}')
    best = (-np.inf, None)
    for w in np.arange(0, 1.01, .1):
        p = w * p_lgb + (1 - w) * p_hgb
        s = score(yt, p)
        print(f'LGB weight={w:.1f}: {s:.2f}')
        best = max(best, (s, w))
    print(f'BEST: {best[0]:.2f} at LGB={best[1]:.1f}; CatBoost baseline=734.49, V14 benchmark=936.93')

    # 독립 설계: 도메인(R/F)이 아니라 "시즌 상태 변화"별 residual adapter.
    # target의 투수 집합을 반으로 나눠 교차 예측하므로 자기 라벨을 자기 예측에 쓰지 않는다.
    base = .5 * p_lgb + .5 * p_hgb
    Xad = Xt.copy()
    Xad['base_prediction'] = base
    shift = Xt['state__season_minus_career_success'].to_numpy()
    sn = Xt['state__season_log_n'].to_numpy()
    regime = np.where(sn < np.log1p(30), 'cold', np.where(shift > .04, 'improving', np.where(shift < -.04, 'declining', 'stable')))
    Xad['state_regime'] = pd.Categorical(regime, categories=['cold', 'improving', 'stable', 'declining']).codes.astype('float32')
    pitchers = np.array(sorted(target.pitcher_id.astype(str).unique()))
    rng = np.random.default_rng(2026); rng.shuffle(pitchers)
    left = set(pitchers[:len(pitchers)//2])
    fold = target.pitcher_id.astype(str).isin(left).to_numpy()
    correction = np.zeros(len(target))
    for holdout in (False, True):
        tr_fold, te_fold = fold != holdout, fold == holdout
        for state in ('cold', 'improving', 'stable', 'declining'):
            tr = tr_fold & (regime == state)
            te = te_fold & (regime == state)
            if te.any() and tr.sum() >= 500:
                model = ExtraTreesRegressor(n_estimators=160, max_depth=14, min_samples_leaf=250,
                                            max_features=.7, random_state=2026, n_jobs=-1)
                model.fit(Xad.loc[tr], (yt - base)[tr])
                correction[te] = model.predict(Xad.loc[te])
    print('state-regime residual adapter:')
    for shrink in (.2, .4, .6, .8, 1.0):
        p = np.clip(base + shrink * correction, 0, 1)
        print(f'  shrink={shrink:.1f}: {score(yt, p):.2f} ({score(yt,p)-score(yt,base):+.2f})')


if __name__ == '__main__':
    frame = pd.read_csv(DATA, encoding='utf-8-sig', low_memory=False)
    frame['season'] = pd.to_numeric(frame['season'], errors='raise').astype(int)
    run_fold(frame)
