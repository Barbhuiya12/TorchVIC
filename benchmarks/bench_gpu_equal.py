# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Dump every output series, carried state and ledger row of a connector
case, to compare two code versions with torch.equal (bit identity).

    python benchmarks/bench_gpu_equal.py --case river --device cuda --out X.pt
    python benchmarks/bench_gpu_equal.py --compare A.pt B.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'tests'))
import bench_gpu_audit as B                      # noqa: E402

DAY = 86400.0


def tensors(x, pre, out):
    if torch.is_tensor(x):
        out[pre] = x.detach().cpu().clone()
    elif isinstance(x, dict):
        for k, v in x.items():
            tensors(v, '%s/%s' % (pre, k), out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            tensors(v, '%s/%d' % (pre, i), out)
    elif isinstance(x, (int, float, bool, str)) or x is None:
        out[pre] = x


def dump_model(m, days):
    m.initialize()
    return dump_rest(m, days)


def dump_model_redo(m, days, at=4):
    """groundwater: after `at` days force the deferred-convergence path to
    launch fewer chunks than a step needs (on the branch: the step is
    redone synchronously; on main the attribute is unused)."""
    m.initialize()
    m.run(until=at * DAY)
    gw = m.connector('groundwater')
    gw.model._ahead = 1
    out = dump_rest(m, days)
    print('REDONE steps', getattr(gw, 'redone', 0))
    return out


def dump_rest(m, days):
    m.run(until=days * DAY)
    out = {}
    for name in m.runtime.connectors:
        try:
            tensors(m.outputs(name), 'out/' + name, out)
        except Exception as e:                   # noqa: BLE001
            out['out/%s/error' % name] = repr(e)[:40]
        tensors(m.connector(name).save_state(), 'state/' + name, out)
    for kind, rows in (('bal', m.ledger.balances), ('cpl', m.ledger.coupled)):
        for i, r in enumerate(rows):
            tensors({k: v for k, v in r.items()}, '%s/%d' % (kind, i), out)
    return out


def dump_flood(n, days, dev, up=False):
    from torchvic.coupling.connectors import flood as F
    from torchvic.coupling.connector import Context
    ny = nx = int(round(np.sqrt(n)))
    yy, xx = np.mgrid[0:ny, 0:nx]
    dem = 20.0 + 0.002 * ((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) ** 0.5
    if up:
        # three points share routing node 1's discharge (q dt / 3)
        fl = F.FloodConnector(dem=dem, dx=100.0, interval_s=DAY,
                              backend='engine', inflow_nodes=[1],
                              inflow_points=[(ny // 2, nx // 2),
                                             (ny // 3, nx // 3),
                                             (ny // 2, nx // 4)])
    else:
        fl = F.FloodConnector(dem=dem, dx=100.0, interval_s=DAY,
                              backend='engine',
                              inflow_points=[(ny // 2, nx // 2)])
    ctx = Context(t=0.0, device=str(dev), dtype=torch.float64)
    fl.initialize(ctx)
    g = torch.Generator().manual_seed(1)
    out = {}
    for k in range(days):
        ctx.t = k * DAY
        V = (torch.rand(ny * nx, generator=g, dtype=torch.float64) * 80
             ).to(dev)
        if up:
            q = (torch.rand(4, generator=g, dtype=torch.float64) * 0.7
                 ).to(dev)
            p = fl.compute(dict(runoff=V, upstream_discharge=q), ctx, DAY)
        else:
            Vp = torch.tensor([1.0e4 * (1 + k % 3)], dtype=torch.float64,
                              device=dev)
            p = fl.compute(dict(runoff=V, inflow=Vp), ctx, DAY)
        tensors(p.exports, 'exp/%d' % k, out)
        tensors({kk: v for kk, v in p.series.items()}, 'ser/%d' % k, out)
        tensors({kk: getattr(p.balance, kk) for kk in
                 ('I_ext', 'Q_out', 'dS', 'closure', 'scale')},
                'bal/%d' % k, out)
        fl.commit(p, ctx)
    tensors(fl.save_state(), 'state', out)
    return out


def compare(a, b):
    A, Bd = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
    bad = []
    for k in sorted(set(A) | set(Bd)):
        if k not in A or k not in Bd:
            bad.append((k, 'missing'))
            continue
        x, y = A[k], Bd[k]
        if torch.is_tensor(x):
            if not (torch.is_tensor(y) and x.shape == y.shape
                    and x.dtype == y.dtype and torch.equal(x, y)):
                bad.append((k, 'tensor differs'))
        elif x != y and not (x != x and y != y):
            bad.append((k, '%r vs %r' % (x, y)))
    print('EQUAL fields: %d, differing: %d' % (len(A) - len(bad), len(bad)))
    for k, w in bad[:20]:
        print('  DIFF', k, w)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case')
    ap.add_argument('--cells', type=int, default=3000)
    ap.add_argument('--days', type=int, default=12)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out')
    ap.add_argument('--compare', nargs=2)
    ap.add_argument('--config', help='a coupled YAML (run to the end)')
    a = ap.parse_args()
    if a.compare:
        raise SystemExit(0 if compare(*a.compare) else 1)
    dev = torch.device(a.device)
    c = a.case
    with torch.inference_mode():
        if c in ('river', 'routing'):
            m, _ = B.case_river(a.cells, a.days + 1, dev, quality=c == 'river',
                                reservoir=c == 'river')
            out = dump_model(m, a.days)
        elif c in ('irrigation', 'crop', 'both'):
            m, _ = B.case_agri(c, a.cells, a.days + 1, dev)
            out = dump_model(m, a.days)
        elif c == 'groundwater':
            m, _ = B.case_gw(a.cells, a.days, dev)
            out = dump_model(m, a.days)
        elif c == 'groundwater_redo':
            # one outer iteration per chunk: a step needs several chunks
            m, _ = B.case_gw(a.cells, a.days, dev,
                             extra=dict(solver=dict(outer_per_check=1)))
            out = dump_model_redo(m, a.days)
        elif c == 'flood':
            out = dump_flood(a.cells, a.days, dev)
        elif c == 'flood_up':
            out = dump_flood(a.cells, a.days, dev, up=True)
        elif c == 'config':
            from torchvic.coupling import TorchVIC
            m = TorchVIC.from_config(a.config, log=lambda *x: None)
            m.cfg['output']['dir'] = os.path.dirname(a.out) + '/cfg_out'
            out = dump_model(m, a.days)
    torch.save(out, a.out)
    print('DUMP', c, a.device, len(out), 'fields')


if __name__ == '__main__':
    main()
