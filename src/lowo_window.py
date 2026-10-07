"""Pointwise vs fixed-window pore-pressure models on the 21 Potwar wells.

Usage:
  python lowo_window.py <data_dir> <out_dir> --mode profile
  python lowo_window.py <data_dir> <out_dir> --mode fixed [--seed 0]
  python lowo_window.py <data_dir> <out_dir> --mode lowo  [--features full] [--seed 0]

Notes (from data/README.md):
- PPP is a predicted pore-pressure curve, so results are agreement with PPP, not error vs measured pressure.
- PINDORI-1/2 are sampled ~4x finer than most wells; every well is decimated to ~0.15 m so wells
  contribute in proportion to their depth interval, not their sample count.
- Always split by well, never by row.
"""
import argparse, glob, os, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ap = argparse.ArgumentParser()
ap.add_argument('data'); ap.add_argument('out')
ap.add_argument('--mode', choices=['profile', 'fixed', 'lowo', 'smooth'], default='fixed')
ap.add_argument('--features', default='full,raw')
ap.add_argument('--windows', default='10,25,50,100')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--trees', type=int, default=200)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

TEST = ['RAJIAN-03A', 'PINDORI-2', 'TURKWAL DEEP X 2', 'Balkassar POL 01']
VAL = ['MINWAL-X-1', 'MINWAL-2', 'MISSA KESWAL-02', 'Balkassar OXY 01']
ESSENTIAL = ['tvd', 'dt', 'dt_nct', 'gr', 'sphi', 'hp', 'ob']
OPTIONAL = ['rhob_combined', 'res_deep']
MURREE = {'MISSA KESWAL-01': (1136.8, 1802.3), 'MISSA KESWAL-02': (1115, 1792.9),
          'MISSA KESWAL-03': (1124.8, 1870.9), 'QAZIAN -1X': (1198.8, 2062.9)}
TARGET_STEP = 0.1524
FULL = ['tvd', 'dt', 'dt_nct', 'gr', 'sphi', 'hp', 'ob', 'rhob_combined', 'res_deep',
        'eaton_ratio', 'hp_gradient', 'ob_gradient', 'tvd_normalized']
RAW = ['gr', 'dt', 'sphi', 'rhob_combined', 'res_deep']
FEATSETS = {'full': FULL, 'raw': RAW}
CTX = ['gr', 'dt', 'sphi', 'rhob_combined', 'res_deep']
ANOM = [CTX.index('gr'), CTX.index('dt')]


def load(path):
    name = os.path.basename(path).rsplit('.', 1)[0]
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
    n_raw = len(df)
    df = df.replace([-999.25, -999, -999.0], np.nan)
    info = dict(well=name, raw_rows=n_raw)
    missing = [c for c in ESSENTIAL + OPTIONAL + ['ppp', 'depth'] if c not in df.columns]
    if missing:
        info.update(status=f'skipped: missing {missing}')
        return None, info
    if name in MURREE:
        lo, hi = MURREE[name]
        df = df[(df.depth >= lo) & (df.depth <= hi)]
    df = df.dropna(subset=['ppp'] + ESSENTIAL + OPTIONAL)
    df = df[(df.ppp > 100) & (df.ppp < 30000) & (df.hp > 100) & (df.hp < 20000)
            & (df.ob > 100) & (df.ob < 30000) & (df.tvd > 0) & (df.tvd < 6000)
            & (df.dt > 0) & (df.dt_nct > 0)]
    df = df.sort_values('depth').drop_duplicates('depth').reset_index(drop=True)
    if len(df) < 600:
        info.update(status=f'skipped: only {len(df)} clean rows', clean_rows=len(df))
        return None, info
    raw_step = float(df.depth.diff().median())
    k = max(1, int(round(TARGET_STEP / raw_step)))
    df = df.iloc[::k].reset_index(drop=True)
    step = raw_step * k
    df['well'] = name
    df['eaton_ratio'] = (df.dt / df.dt_nct) ** 3
    tvd_ft = df.tvd * 3.28084
    df['hp_gradient'] = df.hp / tvd_ft
    df['ob_gradient'] = df.ob / tvd_ft
    df['tvd_normalized'] = df.tvd / df.tvd.max()
    df['run'] = (df.depth.diff() > 1.5 * step).cumsum()
    df.attrs['step'] = step
    info.update(status='ok', clean_rows=len(df) * k, used_rows=len(df), decimate_k=k,
                step_m=round(step, 4), runs=int(df.run.nunique()),
                depth_min=round(df.depth.min(), 1), depth_max=round(df.depth.max(), 1),
                ppp_min=round(df.ppp.min()), ppp_max=round(df.ppp.max()))
    return df, info


wells, infos = {}, []
for p in sorted(glob.glob(os.path.join(args.data, '*.CSV')) + glob.glob(os.path.join(args.data, '*.csv'))):
    d, info = load(p)
    infos.append(info)
    if d is not None:
        wells[info['well']] = d
prof = pd.DataFrame(infos)
prof.to_csv(os.path.join(args.out, 'well_profile.csv'), index=False)
print(prof.to_string(), flush=True)
print('wells used:', len(wells), 'rows:', sum(len(d) for d in wells.values()), flush=True)
if args.mode == 'profile':
    raise SystemExit


def make_windows(df, win_m):
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
        wins.extend(idx[s:s + W] for s in starts)
    return wins


_cache = {}


def window_block(well, win_m):
    """src row positions + window-context features (independent of the feature set)."""
    key = (well, win_m)
    if key in _cache:
        return _cache[key]
    df = wells[well]
    ctx = df[CTX].to_numpy(float)
    depth = df.depth.to_numpy(float)
    src, blocks = [], []
    for idx in make_windows(df, win_m):
        c = ctx[idx]
        mean, std = c.mean(0), c.std(0)
        z = depth[idx] - depth[idx].mean()
        zz = (z ** 2).sum()
        slope = (z[:, None] * (c - mean)).sum(0) / zz if zz > 0 else np.zeros(c.shape[1])
        stats = np.concatenate([mean, std, slope])
        anom = c[:, ANOM] - mean[ANOM]
        blocks.append(np.hstack([np.tile(stats, (len(idx), 1)), anom]))
        src.append(idx)
    out = (np.concatenate(src), np.vstack(blocks))
    _cache[key] = out
    return out


def window_xy(well, win_m, fs):
    src, blk = window_block(well, win_m)
    df = wells[well]
    X = np.hstack([df[FEATSETS[fs]].to_numpy(float)[src], blk])
    return X, df.ppp.to_numpy(float)[src], src


def point_xy(well, fs):
    df = wells[well]
    return df[FEATSETS[fs]].to_numpy(float), df.ppp.to_numpy(float)


def fit(X, y):
    m = xgb.XGBRegressor(n_estimators=args.trees, learning_rate=0.06, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, tree_method='hist', random_state=args.seed,
                         n_jobs=2, verbosity=0)
    m.fit(X, y)
    return m


def run_split(train_wells, eval_wells, fs, win, tag, writer):
    t0 = time.time()
    if win is None:
        Xs, ys = zip(*[point_xy(w, fs) for w in train_wells])
        model = fit(np.vstack(Xs), np.concatenate(ys))
        for w in eval_wells:
            X, y = point_xy(w, fs)
            writer(tag, fs, 'point', 0, w, y, model.predict(X))
    else:
        parts = [window_xy(w, win, fs) for w in train_wells]
        model = fit(np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        for w in eval_wells:
            X, y, src = window_xy(w, win, fs)
            p = model.predict(X)
            n = len(wells[w])
            merged = np.bincount(src, weights=p, minlength=n) / np.maximum(np.bincount(src, minlength=n), 1)
            writer(tag, fs, 'window', win, w, wells[w].ppp.to_numpy(float), merged)
    print(f'  [{tag} {fs} win={win}] {time.time() - t0:.0f}s', flush=True)


rows = []
fn = os.path.join(args.out, f'{args.mode}_results_{args.features.replace(",", "-")}_seed{args.seed}.csv')


def writer(tag, fs, kind, win, well, y, p):
    if tag == 'fixed':
        tag = 'val' if well in VAL else 'test'
    r = dict(split=tag, features=fs, model=kind, window_m=win, well=well, n=len(y),
             r2=r2_score(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))),
             mae=mean_absolute_error(y, p), seed=args.seed)
    rows.append(r)
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(f"{tag:5s} {fs:4s} {kind:6s} {win:>4} {well:18s} R2={r['r2']:.3f} RMSE={r['rmse']:.0f}", flush=True)


def smooth_control(fs):
    """Control: pointwise model + centred moving average of the same length (no window context)."""
    names_ = list(wells)
    for te in names_:
        tr = [w for w in names_ if w != te]
        Xs, ys = zip(*[point_xy(w, fs) for w in tr])
        model = fit(np.vstack(Xs), np.concatenate(ys))
        X, y = point_xy(te, fs)
        p = pd.Series(model.predict(X))
        df = wells[te]
        for win in [int(w) for w in args.windows.split(',')]:
            Wn = max(8, int(round(win / df.attrs['step'])))
            ps = p.groupby(df.run.values).transform(lambda v: v.rolling(Wn, center=True, min_periods=1).mean())
            writer('lowo', fs, 'point_smooth', win, te, y, ps.to_numpy())


names = list(wells)
if args.mode == 'smooth':
    for fs in args.features.split(','):
        smooth_control(fs)
    print('done', flush=True)
    raise SystemExit
configs = [None] + [int(w) for w in args.windows.split(',')]
for fs in args.features.split(','):
    for win in configs:
        if args.mode == 'fixed':
            tr = [w for w in names if w not in TEST + VAL]
            run_split(tr, [w for w in VAL + TEST if w in wells], fs, win, 'fixed', writer)
        else:
            for te in names:
                run_split([w for w in names if w != te], [te], fs, win, 'lowo', writer)
print('done', flush=True)
