# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""torchvic.crop.wofost_npk (WOFOST-NPK kernel).

Oracle: tests/oracle/ref_vicwur_wofost.c #includes the unchanged VIC-WUR 3.0
plugins/wofost/src and crops/src/{crop_run,wofost_run,
wofost_update_step_vars}.c, built twice (build_ref_vicwur_wofost.sh): as the
C model (float) and with `float` defined as `double`.

* fp64 kernel vs the double build: every dumped field of every unit on every
  day of 800-day multi-season runs (three option sets) to 1e-12 of the
  field's scale, flags and leaf-class counts exact, leaf lists to 1e-12.
* fp32 kernel (C float/double promotion emulated, leaf sums in the C order)
  vs the float build: bit for bit.
* fp32 fast mode (leaf sums as float64 reductions): the error bound.
* Function records: Afgen (also on a non-ascending table), Astro +
  DailyTotalAssimilation + Correct, Partioning, DyingLeaves (random leaf
  lists), NutritionINDX, in both precisions.
Cases: tests/wofost_npk_cases.py (R1 crop files: synthetic NPK values, so
this is code equivalence).  Replay of the R1 G9 run (WOFOST + CO2):
tests/test_vicwur_wofost_l2.py.
"""
import os
import subprocess

import numpy as np
import pytest
import torch

import wofost_npk_cases as C

OPTSETS = {
    'dist': dict(DIST_TSUM=True, DIST_FERT=True, CALC_MIN=True, PTEMP=True),
    'potential': dict(PIRR=True, PFERT=True, CONTINUE=True, FORCE_TSUM=True,
                      FORCE_FERT=True, DIST_TSUM=True, DIST_FERT=True),
    'plain': dict(),
}
NDAYS = 800

pytestmark = pytest.mark.skipif(not os.path.isdir(C.R1),
                                reason='R1 inputs not found')


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    try:
        return C.build_oracle(tmp_path_factory.mktemp('wofost_oracle'))
    except RuntimeError as e:
        pytest.skip('cannot build the WOFOST oracle: %s' % str(e)[-300:])


@pytest.fixture(scope='module')
def tmpl():
    return C.variant_templates()


def _season(oracle, tmpl, tmp_path, name, prec, seed):
    opts = OPTSETS[name]
    setup, dates, forc = C.make_case(seed, tmpl, ndays=NDAYS)
    inp = str(tmp_path / 'in.bin')
    out = str(tmp_path / ('out_' + prec))
    C.write_season(inp, tmpl, setup, opts, dates, forc, leafevery=50)
    names = C.run_oracle(oracle[prec], inp, out)
    N, B = setup['area_fract'].shape
    ref = C.read_season(out, names, len(dates), N, B, tmpl.nc)
    return opts, setup, dates, forc, names, ref, C.read_leaves(out)


INT_FIELDS = ('Sowing Emergence CultivateDay GrowthDay cultivating growing '
              'met_MeteoDay leaf_n').split()


@pytest.mark.parametrize('name', sorted(OPTSETS))
def test_l1_season_fp64(oracle, tmpl, tmp_path, name):
    opts, setup, dates, forc, names, ref, rl = _season(
        oracle, tmpl, tmp_path, name, 'f64', 11)
    got, lv, m, s = C.torch_season(tmpl, setup, opts, dates, forc,
                                   torch.float64, leafevery=50)
    assert int(s['err'].max()) == 0
    act = setup['active'][None, :, None, :]
    # the run covers sowing, emergence, growth and harvest on every unit
    assert np.nansum(np.diff(ref['growing'], axis=0) < 0) >= 20
    for k in INT_FIELDS:
        a = np.broadcast_to(act, ref[k].shape)
        assert np.array_equal(ref[k][a], got[k][a]), k
    res = C.compare(ref, got, setup['active'], names)
    worst = max(res.items(), key=lambda kv: kv[1][0])
    assert worst[1][0] <= 1e-12, worst
    # leaf lists (weight, age, area per class): each column to 1e-12 of
    # its scale in the list (the head class keeps the remainder of the
    # stress death, a difference of sums of the whole list)
    for key, a in rl.items():
        b = lv[key]
        assert a.shape == b.shape, key
        if a.size:
            scale = np.maximum(np.abs(a).max(0), 1.0)
            assert (np.abs(a - b).max(0) / scale).max() <= 1e-12, key


@pytest.mark.parametrize('name', sorted(OPTSETS))
def test_l1_season_fp32_bitwise(oracle, tmpl, tmp_path, name):
    """float32 with the C promotion rules and C-order leaf sums equals the
    float build bit for bit."""
    opts, setup, dates, forc, names, ref, rl = _season(
        oracle, tmpl, tmp_path, name, 'f32', 12)
    got, lv, m, s = C.torch_season(tmpl, setup, opts, dates, forc,
                                   torch.float32, leafevery=50,
                                   exact_sums=True)
    act = setup['active'][None, :, None, :]
    for k in names:
        a = np.broadcast_to(act, ref[k].shape)
        r, g = ref[k][a], got[k][a]
        same = (r == g) | (np.isnan(r) & np.isnan(g))
        if k in ('leaf_sumw', 'leaf_sumwa'):
            # diagnostic sums of the dump: double in C, same float terms
            assert np.allclose(r, g, rtol=1e-13, atol=0, equal_nan=True), k
            continue
        assert same.all(), (k, int((~same).sum()))
    for key, a in rl.items():
        assert np.array_equal(a, lv[key]), key


def test_l1_season_fp32_fast_bound(oracle, tmpl, tmp_path):
    """Default fp32 (leaf sums as float64 reductions): the error bound
    against the float C over two seasons, scaled by each field's range."""
    worst = 0.0
    for name in sorted(OPTSETS):
        opts, setup, dates, forc, names, ref, rl = _season(
            oracle, tmpl, tmp_path, name, 'f32', 13)
        got, lv, m, s = C.torch_season(tmpl, setup, opts, dates, forc,
                                       torch.float32)
        res = C.compare(ref, got, setup['active'], names)
        for k in INT_FIELDS:
            assert res[k][1] == 0, (name, k)
        worst = max(worst, max(v[0] for v in res.values()))
    assert worst <= 1e-4, worst


@pytest.mark.parametrize('prec', ['f64', 'f32'])
def test_l1_functions(oracle, tmp_path, prec):
    tm = C.function_template()
    recs = C.make_function_records(21, tm, 300)
    inp = str(tmp_path / 'fin.bin')
    out = str(tmp_path / 'fout')
    for ptemp in (False, True):
        C.write_functions(inp, tm, recs, ptemp=ptemp)
        subprocess.run([oracle[prec], inp, out], check=True)
        ref = C.read_functions(out)
        dt = torch.float64 if prec == 'f64' else torch.float32
        for exact in (False, True):
            got = C.torch_functions(tm, recs, dt, exact_sums=exact,
                                    ptemp=ptemp)
            for (kind, vals), r, g in zip(recs, ref, got):
                assert r.shape == g.shape
                if prec == 'f32' and (exact or kind != 3):
                    ok = (r == g) | (np.isnan(r) & np.isnan(g))
                    assert ok.all(), (kind, vals, r, g)
                else:
                    tol = 1e-12 if prec == 'f64' else 1e-6
                    assert np.allclose(g, r, rtol=tol, atol=1e-300,
                                       equal_nan=True), (kind, vals, r, g)


def test_error_flags(tmpl):
    """C exits (log_err) or crashes where the kernel sets error bits."""
    from torchvic.crop.wofost_npk import ERR_CC, ERR_CV
    setup, dates, forc = C.make_case(3, tmpl, ndays=400)
    setup = dict(setup)
    cc = setup['cc'].copy()
    cc[0, 0, :] = 0.0                    # cell 0, class 0: no cover
    cv = setup['cv'].copy()
    cv[2, 1] = 0.0                       # cell 2, class 1: Cv 0
    setup['cc'], setup['cv'] = cc, cv
    got, lv, m, s = C.torch_season(tmpl, setup, {}, dates, forc,
                                   torch.float64)
    err = s['err'].numpy()
    assert (err[0, :, 0] & ERR_CC).all()
    assert (err[2, :, 1] & ERR_CV).all()
    other = err.copy()
    other[0, :, 0] = 0
    other[2, :, 1] = 0
    assert not other.any()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_gpu_no_sync_and_cuda_graph(tmpl):
    """A day is one CUDA graph: eager GPU with sync debug 'error', then the
    captured graph replayed day by day gives the same state bit for bit;
    the GPU result equals the CPU one to float32 rounding of libm."""
    from torchvic.crop.wofost_npk import UnitSetup, Calendar, WofostNPK
    opts = OPTSETS['dist']
    setup, dates, forc = C.make_case(4, tmpl, ndays=420)
    dev = torch.device('cuda')
    su = UnitSetup(**setup)
    T = len(dates)
    f64 = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    ins = [tuple(f64(forc[k][t]) for k in ('air', 'sw', 'co2', 'pet',
                                           'ws')) for t in range(T)]
    for exact in (False, True):
        m = WofostNPK(tmpl, su, opts, dtype=torch.float32, device=dev,
                      exact_sums=exact)
        cal = Calendar(dates, tmpl, device=dev, dtype=torch.float32)
        s = m.init_state()
        eager = []
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode('error')
        try:
            for t in range(T):
                s = m.crop_run(s, cal, t, *ins[t])
                eager.append(s['st_storage'].clone())
        finally:
            torch.cuda.set_sync_debug_mode(0)
        # graph: static state, inputs and step index
        st = {k: v.clone() for k, v in m.init_state().items()}
        tix = torch.zeros((), dtype=torch.int64, device=dev)
        sin = [x.clone() for x in ins[0]]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                m.crop_run(st, cal, tix, *sin)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m.crop_run(st, cal, tix, *sin)
        for t in range(T):
            tix.fill_(t)
            for a, b in zip(sin, ins[t]):
                a.copy_(b)
            g.replay()
            for k in st:
                st[k].copy_(out[k])
            assert torch.equal(st['st_storage'], eager[t]), (exact, t)
        assert torch.equal(st['st_leaves'], s['st_leaves'])
        assert int(st['err'].max()) == 0
    # CPU vs GPU
    got, _, _, sc = C.torch_season(tmpl, setup, opts, dates, forc,
                                   torch.float32)
    ref = s['st_storage'].cpu().numpy()
    assert np.allclose(sc['st_storage'].numpy(), ref, rtol=1e-4, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('variant', ['ring', 'ring_eager', 'ring_ls',
                                     'ring_ls_eager'])
def test_gpu_lanes_ring_equal_units(tmpl, variant):
    """On CUDA with rows of 336 leaf classes (K > 128, as the ISC's): the
    lanes with the leaf classes on a ring (updated in place, the leaf sums
    as eager's trees on the live classes), with the units' state on the
    lanes or not, fused (no gradient) or op by op, equal the unit layout
    bit for bit: the running units' state, the live leaf classes and the
    OUT_CROP_* rates of every day."""
    from torchvic.crop.wofost_npk import UnitSetup, Calendar, WofostNPK
    from torchvic.crop.wofost_npk import leaf_kernels as LK
    opts = OPTSETS['dist']
    setup, dates, forc = C.make_case(5, tmpl, ndays=420)
    dev = torch.device('cuda')
    su = UnitSetup(**setup)
    T = len(dates)
    f64 = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    ins = [tuple(f64(forc[k][t]) for k in ('air', 'sw', 'co2', 'pet',
                                           'ws')) for t in range(T)]
    cal = Calendar(dates, tmpl, device=dev, dtype=torch.float32)

    def run(m, grad):
        s = m.init_state()
        rates, wmax = [], 0
        with (torch.enable_grad() if grad else torch.no_grad()):
            for t in range(T):
                s = m.crop_run(s, cal, t, *ins[t])
                rates.append(m.rate_outputs(s, cal, t))
                wmax = max(wmax, int((s['lt'] - s['lh']).max()))
        return s, rates, wmax
    mb = WofostNPK(tmpl, su, opts, dtype=torch.float32, device=dev,
                   lanes=False, nleaf=336)
    if not LK.supported(336, mb.U):
        pytest.skip('too few units for the ring kernels')
    sb, rb, wmax = run(mb, True)
    R = wmax + 2
    assert R < 336
    ma = WofostNPK(tmpl, su, opts, dtype=torch.float32, device=dev,
                   nleaf=336, ring=R, lane_state=variant.startswith('ring_ls'))
    sa, ra, _ = run(ma, variant.endswith('eager'))
    assert ma.lanes and (ma.is_lane_state(sa) == ('ls' in variant))
    assert int(sa['err'].max()) == 0 and int(sb['err'].max()) == 0
    for k in sb:
        if k in ('lw', 'la', 'lage') or k.startswith('con_'):
            continue
        x, y = ma.lane_values(sa, k), mb.lane_values(sb, k)
        assert torch.equal(x, y) or bool(
            ((x == y) | (torch.isnan(x) & torch.isnan(y))).all()), k
    h = mb.lane_values(sb, 'lh').long()
    tt = mb.lane_values(sb, 'lt').long()
    kk = torch.arange(336, device=dev)
    live = (kk >= h[:, None]) & (kk < tt[:, None])
    for k in ('lw', 'la', 'lage'):
        x = torch.where(live, sa[k][:, kk % R], torch.zeros(()).to(sa[k]))
        y = torch.where(live, sb[k], torch.zeros(()).to(sb[k]))
        assert torch.equal(x, y), k
    for x, y in zip(ra, rb):
        for k in y:
            assert torch.equal(x[k], y[k]) or bool(
                ((x[k] == y[k]) | (torch.isnan(x[k]) & torch.isnan(y[k])))
                .all()), k
