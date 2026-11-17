"""WOFOST 7.2 in torch against PCSE 6.0.13, daily, every case.

The reference files (tests/data/crop_pcse/*.npz, or all of them from
$TORCHVIC_CROP_REF = <data root>/verification/crop_engine/ref) are written by
torchvic/crop/pcse_reference.py with PCSE itself: the daily weather it used,
the flat parameter dict and PCSE's daily output. Nothing here needs pcse.

Bar: DVS, LAI, TAGP, TWSO (and TWLV, TWST, TWRT, RD) within 1e-9
relative, all days. The measured difference is 1e-15 (rounding of the same
arithmetic); the only larger relative number is LAI after every leaf has died,
where PCSE keeps ~1e-19 of rounding residue from subtracting leaf classes one
by one and the torch version has exactly 0. The relative error is therefore
taken against max(|PCSE value|, 1e-6 x series maximum).
"""
import glob
import json
import os

import numpy as np
import pytest
import torch

from torchvic.crop import CropParams, Drivers, Wofost72

HERE = os.path.dirname(__file__)
REF = os.environ.get('TORCHVIC_CROP_REF', os.path.join(HERE, 'data', 'crop_pcse'))
FILES = sorted(glob.glob(os.path.join(REF, '*.npz')))
VARS = (('dvs', 'DVS'), ('lai', 'LAI'), ('tagp', 'TAGP'), ('twso', 'TWSO'),
        ('twlv', 'TWLV'), ('twst', 'TWST'), ('twrt', 'TWRT'), ('rd', 'RD'))
RTOL = 1e-9
MODE = {'pp': 'pp', 'wlp': 'cwb'}


def load(path):
    z = np.load(path)
    return z, json.loads(str(z['meta']))


def drv(z, i, dev='cpu'):
    f = lambda k: torch.full((1,), float(z['wx_' + k][i]), dtype=torch.float64, device=dev)
    c = lambda v: torch.full((1,), float(v), dtype=torch.float64, device=dev)
    return Drivers(tmin=f('TMIN'), tmax=f('TMAX'), irrad=f('IRRAD'), rain=f('RAIN'), e0=f('E0'),
                   es0=f('ES0'), et0=f('ET0'), lat=c(z['lat']), doy=c(z['doy'][i]))


def engine(meta, mode, dev='cpu'):
    hi = meta['harvest_idx']
    cp = CropParams.from_sets([meta['params']], n=1, device=dev)
    return Wofost72(cp, mode=mode, max_days=meta['ndays'] + 2,
                    harvest_idx=None if hi is None else [hi], max_duration=[meta['max_duration']])


def check(ref, mine, what):
    ref = np.asarray(ref, dtype=float)
    mine = np.asarray(mine, dtype=float)
    floor = max(1e-6 * np.nanmax(np.abs(ref)), 1e-300)
    err = np.abs(mine - ref) / np.maximum(np.abs(ref), floor)
    worst = float(err.max())
    assert worst <= RTOL, '%s: max rel err %.2e at day %d' % (what, worst, int(err.argmax()))
    return worst


def replay(path, ext=False, dev='cpu'):
    z, meta = load(path)
    D = meta['ndays']
    mode = 'external' if ext else MODE[meta['mode']]
    w = engine(meta, mode, dev)

    def e(i):
        if not ext:
            return {}
        r = z['o_RFTRA'][i]
        r = 1.0 if np.isnan(r) else r
        tra = z['o_TRA'][i]
        evs = z['o_EVS'][i]
        t = lambda v: torch.full((1,), 0.0 if np.isnan(v) else float(v), dtype=torch.float64, device=dev)
        return dict(rftra=t(r), tra=t(tra), evs=t(evs))
    outs = []
    w.start(drv(z, 0, dev), **e(0))
    outs.append({k: float(v[0]) for k, v in w.output().items()})
    for i in range(1, D):
        w.advance(drv(z, i, dev), **e(i))
        outs.append({k: float(v[0]) for k, v in w.output().items()})
    return z, meta, w, outs


pytestmark = pytest.mark.skipif(not FILES, reason='no PCSE reference files')


@pytest.mark.parametrize('path', FILES, ids=[os.path.basename(f)[:-4] for f in FILES])
def test_against_pcse(path):
    z, meta, w, outs = replay(path)
    for a, b in VARS:
        check(z['o_' + b], [o[a] for o in outs], a)
    if meta['mode'] == 'wlp':
        check(z['o_SM'], [o['sm'] for o in outs], 'sm')
        ok = ~np.isnan(z['o_TRA'])
        check(z['o_TRA'][ok], np.array([o['tra'] for o in outs])[ok], 'tra')
        check(z['o_RFTRA'][ok], np.array([o['rftra'] for o in outs])[ok], 'rftra')
    # the crop ends the same day PCSE ends it
    assert int(w.done_day[0]) == meta['ndays'] - 1
    s = meta['summary']
    assert abs(float(w.s['laimax'][0]) - s['LAIMAX']) <= RTOL * s['LAIMAX']
    assert abs(float(w.s['ctrat'][0]) - s['CTRAT']) <= RTOL * max(s['CTRAT'], 1e-9) + 1e-12
    assert abs(float(w.s['cevst'][0]) - s['CEVST']) <= RTOL * max(s['CEVST'], 1e-9) + 1e-12
    for key, col in (('DOE', 'doe'), ('DOA', 'doa'), ('DOM', 'dom')):
        got = int(w.s[col][0])
        want = s.get(key)
        if want in (None, 'None'):
            assert got < 0 or got >= meta['ndays'] - 1 or key == 'DOE'
        elif key != 'DOE' or got > 0:
            import datetime as dt
            d0 = dt.date.fromisoformat(meta['start'])
            assert (dt.date.fromisoformat(want) - d0).days == got, key


@pytest.mark.parametrize('path', [f for f in FILES if f.endswith('_wlp.npz')][:6],
                         ids=lambda p: os.path.basename(p)[:-4])
def test_external_stress_replays_pcse(path):
    """VIC-style coupling: the engine given PCSE's RFTRA / TRA / EVS series
    reproduces PCSE's water-limited crop without any soil water of its own."""
    z, meta, w, outs = replay(path, ext=True)
    for a, b in VARS:
        check(z['o_' + b], [o[a] for o in outs], a)
    s = meta['summary']
    assert abs(float(w.s['ctrat'][0]) - s['CTRAT']) <= RTOL * max(s['CTRAT'], 1e-9) + 1e-12


def test_batch_of_different_crops_equals_single_runs():
    """All cases in one tensor batch (different crops, parameters, weather,
    start types, lengths, end types): finished cells freeze, each cell equals its own PCSE run."""
    files = [f for f in FILES if f.endswith('_pp.npz')]
    zs = [load(f) for f in files]
    n, D = len(zs), max(m['ndays'] for _, m in zs)
    sets = [m['params'] for _, m in zs]
    cp = CropParams.from_sets(sets, n=n)
    wx = {k: torch.zeros(D, n, dtype=torch.float64) for k in
          ('TMIN', 'TMAX', 'IRRAD', 'RAIN', 'E0', 'ES0', 'ET0', 'doy')}
    for j, (z, m) in enumerate(zs):
        d = m['ndays']
        for k in wx:
            col = 'doy' if k == 'doy' else 'wx_' + k
            v = torch.tensor(z[col], dtype=torch.float64)
            wx[k][:d, j] = v
            wx[k][d:, j] = v[-1]
    lat = torch.tensor([float(z['lat']) for z, _ in zs], dtype=torch.float64)
    hidx = [BIG if m['harvest_idx'] is None else m['harvest_idx'] for _, m in zs]
    w = Wofost72(cp, mode='pp', max_days=D + 2, harvest_idx=hidx,
                 max_duration=[m['max_duration'] for _, m in zs])

    def dr(i):
        g = lambda k: wx[k][i]
        return Drivers(tmin=g('TMIN'), tmax=g('TMAX'), irrad=g('IRRAD'), rain=g('RAIN'),
                       e0=g('E0'), es0=g('ES0'), et0=g('ET0'), lat=lat, doy=g('doy'))
    series = {k: [] for k in ('dvs', 'lai', 'tagp', 'twso')}
    w.start(dr(0))
    for i in range(D):
        if i:
            w.advance(dr(i))
        o = w.output()
        for k in series:
            series[k].append(o[k].clone())
    for k in series:
        series[k] = torch.stack(series[k]).numpy()
    for j, (z, m) in enumerate(zs):
        d = m['ndays']
        for a, b in VARS[:4]:
            check(z['o_' + b], series[a][:d, j], '%s cell %d' % (a, j))
            # frozen after the crop is finished
            assert np.all(series[a][d - 1:, j] == series[a][d - 1, j])
        assert int(w.done_day[j]) == d - 1


BIG = 10 ** 9


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
@pytest.mark.parametrize('path', FILES[:: max(1, len(FILES) // 8)],
                         ids=lambda p: os.path.basename(p)[:-4])
def test_gpu_against_pcse(path):
    z, meta, w, outs = replay(path, dev='cuda')
    for a, b in VARS:
        check(z['o_' + b], [o[a] for o in outs], a)
    assert int(w.done_day[0]) == meta['ndays'] - 1


def test_afgen_follows_pcse_rule():
    import bisect
    from torchvic.crop.afgen import Afgen
    tab = [0., 0.1, 0.5, 0.4, 1.0, 0.9, 2.0, 0.2]
    a = Afgen.from_lists([tab, tab[:6]], 'cpu')
    xs = [-1., 0., 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for row, (xl, yl) in enumerate(((tab[0::2], tab[1::2]), (tab[0:6:2], tab[1:6:2]))):
        for x in xs:
            if x <= xl[0]:
                want = yl[0]
            elif x >= xl[-1]:
                want = yl[-1]
            else:
                i = bisect.bisect_left(xl, x) - 1
                want = yl[i] + ((yl[i + 1] - yl[i]) / (xl[i + 1] - xl[i])) * (x - xl[i])
            got = float(a.expand(torch.tensor([row]))(torch.tensor([x], dtype=torch.float64))[0])
            assert got == want
