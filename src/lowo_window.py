"""LOWO baseline + fixed-window prototype for pore pressure (4 Murree wells).

Usage: python lowo_window.py <data_dir> <out_dir>
"""
import sys, glob, os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

ESSENTIAL = ['tvd', 'dt', 'dt_nct', 'gr', 'sphi', 'hp', 'ob']
OPTIONAL = ['rhob_combined', 'res_deep']
RANGES = {'MISSA KESWAL-01': (1136.8, 1802.3), 'MISSA KESWAL-02': (1115, 1792.9),
          'MISSA KESWAL-03': (1124.8, 1870.9), 'QAZIAN -1X': (1198.8, 2062.9)}


def load(path):
    name = os.path.basename(path).rsplit('.', 1)[0]
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
    df = df.replace([-999.25, -999, -999.0], np.nan)
    lo, hi = RANGES[name]
    df = df[(df['depth'] >= lo) & (df['depth'] <= hi)]
    df = df.dropna(subset=['ppp'] + ESSENTIAL + OPTIONAL)
    df = df[(df.ppp > 100) & (df.ppp < 30000) & (df.hp > 100) & (df.hp < 20000)
            & (df.ob > 100) & (df.ob < 30000) & (df.tvd > 0) & (df.tvd < 6000)]
    df = df.sort_values('depth').reset_index(drop=True)
    df['well'] = name
    df['eaton_ratio'] = (df.dt / df.dt_nct) ** 3
    tvd_ft = df.tvd * 3.28084
    df['hp_gradient'] = df.hp / tvd_ft
    df['ob_gradient'] = df.ob / tvd_ft
    df['tvd_normalized'] = df.tvd / df.tvd.max()
    # continuous run id: new run when depth jumps > 1.5x median step
    step = df.depth.diff().median()
    df['run'] = (df.depth.diff() > 1.5 * step).cumsum()
    df.attrs['step'] = step
    return df


wells = {}
for p in sorted(glob.glob(os.path.join(DATA, '*.CSV')) + glob.glob(os.path.join(DATA, '*.csv'))):
    d = load(p)
    wells[d['well'].iloc[0]] = d
    print(d['well'].iloc[0], len(d), 'step', round(d.attrs['step'], 4), 'runs', d.run.nunique())

FULL = ['tvd', 'dt', 'dt_nct', 'gr', 'sphi', 'hp', 'ob', 'rhob_combined', 'res_deep',
        'eaton_ratio', 'hp_gradient', 'ob_gradient', 'tvd_normalized']
RAW = ['gr', 'dt', 'sphi', 'rhob_combined', 'res_deep']
CTX = ['gr', 'dt', 'sphi', 'rhob_combined', 'res_deep']


def make_windows(df, win_m):
    """Return list of index arrays; 50% overlap; never spans a gap."""
    step = df.attrs['step']
    W = max(8, int(round(win_m / step)))
    S = max(1, W // 2)
    wins = []
    for _, g in df.groupby('run'):
        idx = g.index.values
        n = len(idx)
        if n <= W:
            wins.append(idx)
            continue
        starts = list(range(0, n - W + 1, S))
        if starts[-1] != n - W:
            starts.append(n - W)
        for s in starts:
            wins.append(idx[s:s + W])
    return wins


def window_rows(df, win_m, base_feats):
    """One row per (window, sample) with sample features + window-context features."""
    wins = make_windows(df, win_m)
    parts = []
    for wid, idx in enumerate(wins):
        w = df.loc[idx]
        row = w[base_feats].copy()
        z = (w.depth.values - w.depth.values.mean())
        for c in CTX:
            v = w[c].values
            row[f'{c}_wmean'] = v.mean()
            row[f'{c}_wstd'] = v.std()
            row[f'{c}_wslope'] = (np.polyfit(z, v, 1)[0] if len(v) > 2 and z.std() > 0 else 0.0)
            if c in ('gr', 'dt'):
                row[f'{c}_anom'] = v - v.mean()
        row['src_index'] = idx
        row['wid'] = wid
        parts.append(row)
    return pd.concat(parts, ignore_index=True)


def fit_predict(Xtr, ytr, Xte, seed=int(os.environ.get("SEED", 0))):
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, random_state=seed, n_jobs=4, verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def metrics(y, p):
    return dict(r2=r2_score(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=mean_absolute_error(y, p))


rows = []
names = list(wells)
configs = [('point', None)] + [('window', w) for w in (10, 25, 50, 100)]
for fs_name, fs in (('full', FULL), ('raw', RAW)):
    for kind, win in configs:
        for test in names:
            train = [n for n in names if n != test]
            te = wells[test]
            if kind == 'point':
                Xtr = pd.concat([wells[n][fs] for n in train])
                ytr = pd.concat([wells[n].ppp for n in train])
                pred = fit_predict(Xtr, ytr, te[fs])
                y = te.ppp.values
            else:
                tr_parts = []
                for n in train:
                    r = window_rows(wells[n], win, fs)
                    r['y'] = wells[n].loc[r.src_index, 'ppp'].values
                    tr_parts.append(r)
                tr = pd.concat(tr_parts, ignore_index=True)
                tew = window_rows(te, win, fs)
                featcols = [c for c in tr.columns if c not in ('src_index', 'wid', 'y')]
                p = fit_predict(tr[featcols], tr.y, tew[featcols])
                # merge overlapping window predictions by averaging per source sample
                merged = pd.Series(p).groupby(tew.src_index.values).mean()
                pred = merged.reindex(te.index).values
                y = te.ppp.values
            m = metrics(y, pred)
            m.update(dict(features=fs_name, model=kind, window_m=win, test_well=test, n=len(y)))
            rows.append(m)
            print(f"{fs_name:4s} {kind:6s} {str(win):>4s} {test:16s} R2={m['r2']:.3f} RMSE={m['rmse']:.0f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'lowo_results.csv'), index=False)
summ = res.groupby(['features', 'model', 'window_m'], dropna=False).agg(
    mean_r2=('r2', 'mean'), mean_rmse=('rmse', 'mean'), mean_mae=('mae', 'mean')).reset_index()
print(summ.to_string())
summ.to_csv(os.path.join(OUT, 'lowo_summary.csv'), index=False)
