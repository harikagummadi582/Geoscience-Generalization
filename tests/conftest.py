"""Load the functions of src/lowo_window.py for testing.

The script runs at import (argparse + data loading), so we exec everything up to the experiment
loop with an empty data folder and return its globals. No real data is needed.
"""
import os, sys, pathlib, tempfile
import numpy as np
import pandas as pd
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'lowo_window.py'


@pytest.fixture(scope='module')
def lw():
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, 'data')
    os.makedirs(data)
    argv = sys.argv
    sys.argv = ['lowo_window.py', data, os.path.join(tmp, 'out'), '--mode', 'fixed', '--trees', '5']
    g = {'__name__': 'lowo_window_under_test'}
    try:
        exec(compile(SRC.read_text().split('rows = []')[0], str(SRC), 'exec'), g)
    finally:
        sys.argv = argv
    return g


def make_well(g, n=3000, step=0.1524, gaps=((800, 830), (2000, 2100)), seed=0):
    """Synthetic well with depth gaps, shaped like the output of load()."""
    rng = np.random.default_rng(seed)
    keep = np.ones(n, bool)
    for a, b in gaps:
        keep[a:b] = False
    df = pd.DataFrame({'depth': (1000 + np.arange(n) * step)[keep]})
    cols = set(g['FULL']) | set(g['RAW']) | {'ppp'}
    for c in cols:
        df[c] = rng.normal(size=len(df)).cumsum() / 20 + 100
    df['run'] = (df.depth.diff() > 1.5 * step).cumsum()
    df.attrs['step'] = step
    return df
