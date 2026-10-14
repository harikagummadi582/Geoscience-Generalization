import itertools
import numpy as np
import pytest
from conftest import make_well

MODES = list(itertools.product(['fixed', 'equal', 'cpd'], [0, 25, 50, 75], [10, 25, 50]))


@pytest.mark.parametrize('seg,ov,win', MODES)
def test_windows_never_span_a_depth_gap(lw, seg, ov, win):
    df = make_well(lw)
    lw['SEG'], lw['OV'] = seg, ov
    step = df.attrs['step']
    depth, run = df.depth.to_numpy(), df.run.to_numpy()
    for idx in lw['make_windows'](df, win):
        assert len(set(run[idx])) == 1, 'window crosses two continuous runs'
        assert (np.diff(depth[idx]) <= 1.5 * step).all(), 'depth gap inside a window'


@pytest.mark.parametrize('seg,ov,win', MODES)
def test_every_sample_is_covered(lw, seg, ov, win):
    df = make_well(lw)
    lw['SEG'], lw['OV'] = seg, ov
    src = np.concatenate(lw['make_windows'](df, win))
    cover = np.bincount(src, minlength=len(df))
    assert (cover >= 1).all(), 'some samples are in no window'
    if seg in ('equal', 'cpd'):
        assert (cover == 1).all(), 'segments must partition the well'
    merged = np.bincount(src, weights=np.ones(len(src)), minlength=len(df)) / cover
    assert np.allclose(merged, 1.0)  # averaging overlapping predictions gives one value per sample


def test_no_overlap_between_test_and_validation_wells(lw):
    assert not set(lw['TEST']) & set(lw['VAL'])
    assert len(lw['TEST']) == 4 and len(lw['VAL']) == 4


def test_held_out_well_is_not_used_for_training(lw):
    lw['SEG'], lw['OV'] = 'fixed', 50
    wells = {name: make_well(lw, seed=i, n=2500 + 400 * i) for i, name in enumerate('ABC')}
    lw['wells'] = wells
    lw['_cache'].clear()
    seen = {}
    real_fit = lw['fit']

    def spy_fit(X, y):
        seen['rows'] = len(X)
        return real_fit(X, y)

    lw['fit'] = spy_fit
    out = []
    lw['run_split'](['A', 'B'], ['C'], 'full', 25, 'lowo',
                    lambda tag, fs, kind, win, well, y, p: out.append((well, len(y), np.isfinite(p).all(), len(p))))
    lw['fit'] = real_fit
    expected = sum(len(lw['window_block'](w, 25)[0]) for w in 'AB')
    assert seen['rows'] == expected, 'training rows must come only from the training wells'
    assert out == [('C', len(wells['C']), True, len(wells['C']))]  # one finite prediction per held-out sample
