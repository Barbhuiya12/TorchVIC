"""VIC-WUR routing cascade: CascadeRouter against the VIC-WUR 3.0 C code.

Oracle: tests/oracle/ref_vicwur_route.c runs the unchanged VIC-WUR 3.0
rout_init.c, rout_run.c, rout_basin_run.c, convolute.c, cshift.c and wu_init.c
on random networks; the torch result must match to 1e-12 (it is bitwise in
practice).
Replay of the R1 G1a (n=1) and G1b (n=4) runs of the dump build: the
dumped per-step runoff drives the router, and discharge, inflow, stream and
the ring are compared with the dump; OUT_DISCHARGE with the run's output.
"""
import glob
import os
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import _paths                                               # noqa: E402
from torchvic.model import routing_cascade as RC            # noqa: E402

ORACLE = _paths.oracle('ref_vicwur_route')
VER = _paths.env_path(('TORCHVIC_VICWUR_VERIFY',), 'verification')
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _oracle():
    if not os.path.exists(ORACLE):
        subprocess.run(['bash', os.path.join(_paths.ORACLE_SRC,
                                             'build_ref_vicwur_route.sh')],
                       capture_output=True)
    if not os.path.exists(ORACLE):
        pytest.skip('ref_vicwur_route not built (VICWUR_SRC missing?)')
    return ORACLE


def random_network(rng, n, max_up=8, outlets=3):
    """Random forest: each non-outlet cell drains to a cell of lower rank;
    cell indices are shuffled so the order is not the index order."""
    rank_down = np.full(n, -1)
    nup = np.zeros(n, int)
    for r in range(outlets, n):
        while True:
            d = rng.integers(0, r)
            if nup[d] < max_up:
                break
        rank_down[r] = d
        nup[d] += 1
    p = rng.permutation(n)                    # rank -> cell
    down = np.arange(n)
    for r in range(n):
        if rank_down[r] >= 0:
            down[p[r]] = p[rank_down[r]]
    return down


def remote_edges(rng, down, k):
    """k sources, each with 1-3 receivers of a lower generation; receivers
    may be shared, so same-level sources need sub-levels. Some draws break
    the order rules; both codes must then refuse."""
    n = len(down)
    up, nup = RC.upstream_table(down)
    sch = RC.build_schedule(down)
    lv = sch.level
    rec = [[] for _ in range(n)]
    srcs = rng.choice(n, size=k, replace=False)
    for s in srcs:
        cand = [c for c in range(n) if lv[c] < lv[s] and c != s]
        if not cand:
            continue
        m = min(len(cand), int(rng.integers(1, 4)))
        rec[s] = sorted(rng.choice(cand, m, replace=False).tolist())
    return rec


def normalise(u):
    s = np.zeros(u.shape[1])
    for j in range(u.shape[0]):                     # rout_init.c:81-88
        s = s + u[j]
    return u / s


def run_c(tmp_path, down, uhi, uhr, area, dt, nsub, ring0, deficit0, rs, rb,
          fq, force=False, nonrenew_runoff=False, gwm=False, receiving=None,
          ids=None):
    n, L = len(down), uhi.shape[0]
    nt = rs.shape[0]
    R = L + nsub - 1
    ids = np.arange(1, n + 1) if ids is None else ids
    dfield = ids[down]
    dfield = np.where(down == np.arange(n), 0, dfield)   # 0: no match
    nrm = 0 if receiving is None else max(1, max(len(r) for r in receiving))
    parts = [np.array([n, L, nsub, nt, force, nonrenew_runoff, gwm,
                       receiving is not None, nrm], np.int64),
             np.array([dt], np.float64), area.astype(np.float64),
             ids.astype(np.int64), dfield.astype(np.int64),
             uhi.astype(np.float64), uhr.astype(np.float64)]
    if receiving is not None:
        # wu_init.c:75-122: receiving_id per cell; at receiver cell i, layer
        # k of 'receiving' holds its source's receiving_id; Nreceiving at
        # the source.
        rid = np.arange(1, n + 1, dtype=np.int64)
        lay = np.zeros((nrm, n), np.int64)
        cnt = np.zeros(n, np.int64)
        for s, rr in enumerate(receiving):
            for r in rr:
                k = 0
                while lay[k, r] != 0:
                    k += 1
                lay[k, r] = rid[s]
            cnt[s] = len(rr)
        parts += [rid, cnt, lay]
    parts += [ring0.astype(np.float64), deficit0.astype(np.float64)]
    for t in range(nt):
        parts += [rs[t], rb[t], fq[t]]
    fin = str(tmp_path / 'in.bin')
    fout = str(tmp_path / 'out.bin')
    with open(fin, 'wb') as f:
        for a in parts:
            f.write(np.ascontiguousarray(a).tobytes())
    r = subprocess.run([_oracle(), fin, fout], capture_output=True,
                       text=True)
    if r.returncode:
        return None, r.stderr
    b = np.fromfile(fout, np.uint8)
    o = 0

    def take(dtype, cnt):
        nonlocal o
        a = np.frombuffer(b[o:o + 8 * cnt].tobytes(), dtype)
        o += 8 * cnt
        return a
    out = {'order': take(np.int64, n), 'nup': take(np.int64, n),
           'up': take(np.int64, n * 8).reshape(n, 8),
           'uhi': take(np.float64, n * L).reshape(n, L),
           'uhr': take(np.float64, n * L).reshape(n, L)}
    steps = []
    for t in range(nt):
        st = {k: take(np.float64, n) for k in
              ('runoff', 'inflow', 'discharge', 'stream', 'deficit')}
        st['ring'] = take(np.float64, n * R).reshape(n, R)
        steps.append(st)
    out['steps'] = steps
    return out, r.stderr


def _close(a, b, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    err = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    err = np.where((a == b), 0.0, err)
    return float(err.max(initial=0.0))


CASES = [(1, 4), (4, 4), (1, 48), (4, 48)]


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('nsub,L', CASES)
@pytest.mark.parametrize('opts', [dict(), dict(force=True),
                                  dict(nonrenew_runoff=True),
                                  dict(gwm=True),
                                  dict(force=True, gwm=True, negative=True)])
def test_l1_random_network(tmp_path, device, nsub, L, opts):
    seed = 7 + nsub * 100 + L + 1000 * len(opts) + \
        (sum(map(ord, ''.join(opts))) if opts else 0)
    opts = dict(opts)
    # negative forced inflow (VIC-WUR 3.0 MODFLOW: a river losing water to
    # the aquifer): rout_basin_run.c:66-143 clips nothing, the ring goes
    # negative in C and here alike
    negative = opts.pop('negative', False)
    rng = np.random.default_rng(seed)
    n, nt = 60, 12
    down = random_network(rng, n)
    uhi = rng.random((L, n)) + 1e-3
    uhr = rng.random((L, n)) + 1e-3
    area = rng.uniform(1e6, 1e8, n)
    dt = 86400.0
    R = L + nsub - 1
    ring0 = rng.random((n, R)) * 10 * (rng.random((n, 1)) > 0.2)
    deficit0 = rng.random(n) * 3 * (rng.random(n) > 0.4)
    rs = rng.random((nt, n)) * 5 * (rng.random((nt, n)) > 0.3)
    rb = rng.random((nt, n)) * 2
    fq = rng.random((nt, n)) * 50 * (rng.random((nt, n)) > 0.5)
    if negative:
        fq = fq - 30.0 * rng.random((nt, n)) * (rng.random((nt, n)) > 0.4)
    c, err = run_c(tmp_path, down, uhi, uhr, area, dt, nsub, ring0, deficit0,
                   rs, rb, fq, **opts)
    assert c is not None, err
    r = RC.CascadeRouter(down, normalise(uhi), normalise(uhr), area, dt,
                         nsub, device=device, **opts)
    np.testing.assert_array_equal(r.schedule.order, c['order'])
    np.testing.assert_array_equal(r.schedule.nup, c['nup'])
    np.testing.assert_array_equal(r.schedule.up, c['up'])
    assert np.array_equal(normalise(uhi).T, c['uhi'])
    assert np.array_equal(normalise(uhr).T, c['uhr'])
    r.load_state(ring0, deficit0)
    worst = 0.0
    for t in range(nt):
        if opts.get('force'):
            r.set_force(fq[t])
        r.step(rs[t], rb[t])
        o = {k: v.cpu().numpy() for k, v in r.outputs().items()}
        s = c['steps'][t]
        for mine, theirs in (('OUT_STREAM_RUNOFF', 'runoff'),
                             ('OUT_STREAM_INFLOW', 'inflow'),
                             ('OUT_DISCHARGE', 'discharge'),
                             ('OUT_NONREN_DEFICIT', 'deficit')):
            worst = max(worst, _close(o[mine], s[theirs]))
        worst = max(worst, _close(r.state()['discharge_dt'].cpu().numpy(),
                                  s['ring']))
        st = (r.to_cells(r.stream)).cpu().numpy()
        worst = max(worst, _close(st, s['stream']))
    assert worst <= 1e-12, worst
    if device == 'cpu':
        assert worst == 0.0          # bitwise on the CPU path
    if negative:
        assert (c['steps'][-1]['ring'] < 0).any()      # reached


@pytest.mark.reference
def test_l1_remote_order(tmp_path):
    nsubl = 0
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        n = 80
        down = random_network(rng, n, outlets=4)
        rec = remote_edges(rng, down, 8)
        L, nsub = 4, 1
        z = np.zeros((1, n))
        c, err = run_c(tmp_path, down, np.ones((L, n)), np.ones((L, n)),
                       np.ones(n), 86400.0, nsub, np.zeros((n, L)),
                       np.zeros(n), z, z, z, receiving=rec)
        try:
            s = RC.build_schedule(down, rec)
        except RC.RoutingOrderError:
            assert c is None, 'torch refused an order C accepts'
            continue
        assert c is not None, err
        np.testing.assert_array_equal(s.order, c['order'])
        # sub-levels: two sources of one receiver never share a group
        grp = np.searchsorted(s.gptr, s.pos, side='right') - 1
        for r_ in range(n):
            srcs = [i for i in range(n) if r_ in rec[i]]
            assert len(set(grp[srcs])) == len(srcs)
        nsubl += int(s.sub.max() > 0)
    assert nsubl > 0                     # some draws needed sub-levels


@pytest.mark.reference
def test_l1_receiver_downstream_of_source_is_refused(tmp_path):
    down = np.array([1, 2, 2, 2])          # 0 -> 1 -> 2 (outlet), 3 -> 2
    rec = [[1], [], [], []]                # 0 supplies 1, its downstream
    with pytest.raises(RC.RoutingOrderError):
        RC.build_schedule(down, rec)
    z = np.zeros((1, 4))
    c, err = run_c(tmp_path, down, np.ones((2, 4)), np.ones((2, 4)),
                   np.ones(4), 86400.0, 1, np.zeros((4, 2)), np.zeros(4),
                   z, z, z, receiving=rec)
    assert c is None and 'ERROR' in err


def test_cycle_and_upstream_limit_raise():
    with pytest.raises(RC.RoutingOrderError):      # C loops for ever
        RC.build_schedule(np.array([1, 0, 2]))
    with pytest.raises(RC.RoutingOrderError):      # C overruns upstream[8]
        RC.build_schedule(np.array([0] * 10))


@pytest.mark.parametrize('device', DEVICES)
def test_slot_sweep_equals_fused_sweep(device):
    rng = np.random.default_rng(3)
    n, L, nsub = 300, 12, 4
    down = random_network(rng, n)
    args = (down, normalise(rng.random((L, n))),
            normalise(rng.random((L, n))), rng.uniform(1e6, 1e8, n),
            86400.0, nsub)
    a = RC.CascadeRouter(*args, device=device, tail=8)
    b = RC.CascadeRouter(*args, device=device)
    seen = []
    for t in range(5):
        rs, rb = rng.random(n), rng.random(n)
        a.step(rs, rb)
        b.step(rs, rb, slots={'wu_local': lambda r, c: seen.append(
            int(c.numel()))})
        assert torch.equal(a.ring_logical(), b.ring_logical())
        assert torch.equal(a.discharge, b.discharge)
    assert sum(seen) == 5 * n


def _ring_ops_c(q, S0, w, r, lo, L):
    """Scalar transcription of wu_run_local.c:470-525 for one cell: the
    bins are edited, and clipped at 0, only when the cell withdraws or
    returns (:472)."""
    q = list(q)
    if not (w > 0 or r > 0):
        return q
    S = 0.0
    for i in range(lo, len(q)):
        S += q[i]
    for i in range(lo, len(q)):
        d = q[i]
        if w > 0 and S > 0:
            q[i] -= w * (d / S)
        if r > 0:
            if S > 0:
                q[i] += r * (d / S)
            else:
                q[i] += r / (L - 1)
        if q[i] < 0:
            q[i] = 0.0
    return q


@pytest.mark.parametrize('device', DEVICES)
def test_ring_service_matches_c_loops(device):
    rng = np.random.default_rng(5)
    n, L, nsub = 40, 6, 4
    down = np.arange(n)
    r = RC.CascadeRouter(down, normalise(np.ones((L, n))),
                         normalise(np.ones((L, n))), np.ones(n), 86400.0,
                         nsub, device=device)
    ring = rng.random((n, r.R)) * (rng.random((n, 1)) > 0.3)
    # negative bins (a negative forced inflow) in some cells
    ring = ring - 0.6 * rng.random((n, r.R)) * (rng.random((n, 1)) > 0.6)
    r.load_state(ring)
    w = rng.random(n) * (rng.random(n) > 0.5) * 0.5
    ret = rng.random(n) * (rng.random(n) > 0.5)
    for lo in (0, nsub):
        r.load_state(ring)
        cells = torch.arange(n, device=r.device)
        S = ring[:, lo:].sum(1)
        ww = np.where(S > 0, w, 0.0)
        bad = r.scale_prop(cells, torch.as_tensor(ww, device=r.device),
                           torch.as_tensor(ret, device=r.device), lo)
        assert not bool(bad.any())
        got = r.state()['discharge_dt'].cpu().numpy()
        # a cell that neither withdraws nor returns keeps its negative bins
        idle = (ww == 0) & (ret == 0)
        assert (got[idle] < 0).any()
        for i in range(n):
            ref = _ring_ops_c(ring[i], None, ww[i], ret[i], lo, L)
            assert np.array_equal(got[i], np.array(ref)), (lo, i)


# ----------------------------------------------------------------- replay
G1 = [('G1a', 1), ('G1b', 4)]


@pytest.mark.reference
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('run,nsub', G1)
def test_l2_r1_replay(device, run, nsub):
    netCDF4 = pytest.importorskip('netCDF4')
    from torchvic.vicwur import params as P
    npz = '%s/vicwur_11/gruns/%s_dump/rout.npz' % (VER, run)
    out = sorted(glob.glob('%s/vicwur_03/gruns/R1/%s/W3/out/*.nc'
                           % (VER, run)))
    R1 = _paths.data('vicwur/R1')
    if not (os.path.exists(npz) and out and os.path.isdir(R1)):
        pytest.skip('G1 dump or outputs absent')
    z = np.load(npz)
    rout = z['rout']                                # [nt, nc, 5+R]
    grid = P.Grid(R1 + '/domain.nc', R1 + '/params.nc')
    rp = P.read_routing(R1 + '/rout_n%d.nc' % nsub, grid, 4 if nsub == 4
                        else 1)
    r = RC.CascadeRouter.from_params(rp, grid.area_active, 86400.0,
                                     nsub, device=device)
    assert r.R == rout.shape[2] - 5
    # OUT_RUNOFF / OUT_BASEFLOW of the run -> rout_run -> runoff (P1)
    qr, qb, qd = [], [], []
    for f in out:
        with netCDF4.Dataset(f) as d:
            d.set_auto_mask(False)
            qr.append(grid.take(np.array(d['OUT_RUNOFF'][:])))
            qb.append(grid.take(np.array(d['OUT_BASEFLOW'][:])))
            qd.append(grid.take(np.array(d['OUT_DISCHARGE'][:])))
    qr, qb, qd = (np.concatenate(a) for a in (qr, qb, qd))
    nt = rout.shape[0]
    assert qr.shape[0] == nt
    w = {'runoff': 0.0, 'from_out': 0.0, 'discharge': 0.0, 'inflow': 0.0,
         'stream': 0.0, 'ring': 0.0, 'out_discharge': 0.0}
    for t in range(nt):
        d = rout[t]
        r.rout_run(qr[t], qb[t])
        w['from_out'] = max(w['from_out'], _close(
            r.to_cells(r.runoff).cpu().numpy(), d[:, 0]))
        r.runoff.copy_(r.to_sweep(d[:, 0]))         # replay the C runoff
        r.begin()
        r.sweep()
        o = {k: v.cpu().numpy() for k, v in r.outputs().items()}
        w['inflow'] = max(w['inflow'], _close(o['OUT_STREAM_INFLOW'],
                                              d[:, 1]))
        w['stream'] = max(w['stream'], _close(
            r.to_cells(r.stream).cpu().numpy(), d[:, 2]))
        w['discharge'] = max(w['discharge'], _close(o['OUT_DISCHARGE'],
                                                    d[:, 4]))
        w['ring'] = max(w['ring'], _close(
            r.state()['discharge_dt'].cpu().numpy(), d[:, 5:]))
        # the history file is float32: our value rounded must be its value
        w['out_discharge'] = max(w['out_discharge'], float(np.abs(
            o['OUT_DISCHARGE'].astype(np.float32) - qd[t]).max()))
    print(run, device, w)
    for k in ('discharge', 'inflow', 'stream', 'ring'):
        assert w[k] <= 1e-12, (k, w[k])
    assert w['out_discharge'] == 0.0
    # rout_run from the float32 OUT_RUNOFF/OUT_BASEFLOW of the file
    assert w['from_out'] <= 1e-6, w['from_out']


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_cuda_graph_replay_matches_eager():
    rng = np.random.default_rng(11)
    n, L, nsub = 5000, 24, 4
    down = random_network(rng, n, outlets=20)
    args = (down, normalise(rng.random((L, n))),
            normalise(rng.random((L, n))), rng.uniform(1e6, 1e8, n),
            86400.0, nsub)
    a = RC.CascadeRouter(*args, device='cuda')
    b = RC.CascadeRouter(*args, device='cuda')
    c = RC.CascadeRouter(*args, device='cpu')
    b.capture()
    for t in range(4):
        rs, rb = rng.random(n), rng.random(n)
        a.step(rs, rb)
        c.step(rs, rb)
        b.rout_run(rs, rb)
        b.replay()
    assert torch.equal(a.ring_logical(), b.ring_logical())
    assert torch.equal(a.discharge, b.discharge)
    assert torch.equal(c.ring_logical(), a.ring_logical().cpu())
    assert torch.equal(c.discharge, a.discharge.cpu())
