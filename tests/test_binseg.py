import numpy as np
import pytest
import ruptures as rpt


@pytest.mark.parametrize('seed,n,k', [(0, 600, 5), (1, 900, 12), (2, 400, 3), (3, 1500, 25)])
def test_binseg_matches_ruptures(lw, seed, n, k):
    rng = np.random.default_rng(seed)
    sig = np.cumsum(rng.normal(size=(n, 3)), 0) / 5 + np.repeat(rng.normal(size=(n // 100 + 1, 3)), 100, 0)[:n] * 3
    mine = lw['binseg_l2'](sig, k, min_size=8, jump=2)
    ref = rpt.Binseg(model='l2', min_size=8, jump=2).fit(sig).predict(n_bkps=k)
    assert mine == ref
