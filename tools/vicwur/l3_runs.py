#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The R1 G-runs coupled, against the VIC-WUR 3.0 outputs, and their restarts.

  l3_runs.py run RUN MODE [--device cpu] [--out DIR] [--restart]
  l3_runs.py table [--out DIR]

`run` builds the G-run from its VIC-WUR 3.0 global file through
torchvic.coupling.connectors.vicwur_setup (VIC on the vicwur3 core in the
loop, every switched-on plugin as a connector of one phase group; G11 and
G11c with the VIC-WUR 3.0 MODFLOW coupling as the groundwater member) and
runs it over the 7 years through the coupling runtime. MODE is `fixed` (the
default: upstream bugs fixed) or `exact` (the C, quirks included). Against
the VIC-WUR 3.0 history files it writes per output KGE, percent bias, RMSE
and the gate number max |x - VIC-WUR 3.0| / max(1, |x|) (the history files
are float32) to <out>/<RUN>_<MODE>.json.

G12 is G10 split at 2007-01-01 (day 1280): the run stops there, saves a
checkpoint and a second run restarts from it (window_days 32 puts the
split on a window boundary); compared with VIC-WUR 3.0's own split run
(G12, which drops on restart every plugin state VIC-WUR 3.0 does not write:
dams, water-use deficit, WOFOST, land use's held Cv; `--w3-restart` emulates
that in exact mode) and with the continuous G10.

`--restart` also restarts the run at RESTART_DAYS (2005-08-14: inside a
crop season, a monthly land-use record and a MODFLOW month; 2006-06-20:
the sowing day of the kharif crops; 2006-11-22: between seasons) from
checkpoints saved by a run that stops there, and checks bit for bit that
the stopping run equals the continuous one, and that every restarted run
equals the continuous one: every module state at the end, every output
row after the restart and every ledger row.
"""
import argparse
import datetime
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from torchvic.coupling import TorchVIC                               # noqa
from torchvic.coupling.connectors import vicwur_setup as WS          # noqa
from torchvic.coupling.connectors import vicwur_common as VC         # noqa

ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
GR = ROOT + '/vicwur/gruns/R1'
GW4 = ROOT + '/verification/vicwur_4/gruns/R1'
OUT = ROOT + '/verification/vicwur_3/l3'
RUNS = ('G0a G0b G1a G1b G2 G3 G4 G4d G4e G4o G4p G5 G6a G6b G6c G7 G8 '
        'G9 G9b G10 G11 G11c G12').split()
START = datetime.date(2003, 7, 1)
NDAYS = 2557
SPLIT = 1280                     # 2007-01-01, G12
RESTART_DAYS = (775, 1085, 1240)
DAY = 86400

CROPV = ['OUT_CROP_' + s for s in (
    'CULTIVATE GROW EVAP EVAP_BARE EVAP_CANOP TRANSP_VEG DVS WLV WST WSO WRT '
    'WDLV WDST WDRT NLV NST NSO NRT LAI WSTRESS NSTRESS TMAXSTRESS '
    'TMINSTRESS NPKI NNI NSOIL NUPT NDEM PNI PSOIL PUPT PDEM KNI KSOIL KUPT '
    'KDEM ROOTDEPTH').split()]
VIC_VARS = (('OUT_EVAP', 'et'), ('OUT_BASEFLOW', 'baseflow'),
            ('OUT_TRANSP_VEG', 'transp'), ('OUT_EVAP_BARE', 'evap_bare'),
            ('OUT_EVAP_CANOP', 'evap_canop'), ('OUT_RUNOFF', 'runoff'),
            ('OUT_GWRECHARGE', 'gwrecharge'))
PLUGIN = (
    ('OUT_DISCHARGE', 'vicwur_routing', 'discharge'),
    ('OUT_STREAM_MOIST', 'vicwur_routing', 'stream_moist'),
    ('OUT_REQUIREMENT', 'vicwur_irrigation', 'requirement'),
    ('OUT_RECEIVED', 'vicwur_irrigation', 'received'),
    ('OUT_APPLIED', 'vicwur_irrigation', 'applied'),
    ('OUT_LEFTOVER', 'vicwur_irrigation', 'leftover'),
    ('OUT_SHORTAGE', 'vicwur_irrigation', 'shortage'),
    ('OUT_EFR_DISCHARGE', 'vicwur_efr', 'efr_discharge'),
    ('OUT_EFR_BASEFLOW', 'vicwur_efr', 'efr_baseflow'),
    ('OUT_EFR_MOIST', 'vicwur_efr', 'efr_moist'),
    ('OUT_NONREN_DEFICIT', 'vicwur_nonrenew', 'deficit'),
    ('OUT_CV', 'vicwur_landuse', 'cv')) + tuple(
        (k, 'vicwur_crop', k) for k in CROPV)


def global_path(run):
    if run == 'G11c':
        return GW4 + '/G11c/global_W3.txt'
    return os.path.join(GR, 'G10' if run == 'G12' else run, 'global_W3.txt')


def w3_files(run):
    if run == 'G11c':
        return sorted(glob.glob(GW4 + '/G11c/W3/out/fluxes.*.nc'))
    if run == 'G12':
        return (sorted(glob.glob(GR + '/G12/W3/out/fluxes.*.nc')) +
                sorted(glob.glob(GR + '/G12/W3/p2/out/fluxes.*.nc')))
    return sorted(glob.glob(os.path.join(GR, run, 'W3', 'out',
                                         'fluxes.*.nc')))


def w3(run):
    import xarray as xr
    return xr.concat([xr.open_dataset(f).load() for f in w3_files(run)],
                     dim='time')


def groundwater_cfg(run):
    if run not in ('G11', 'G11c'):
        return None
    import gw_runs as GWR
    # G11: the C run is VIC alone (POC, nothing back to VIC); G11c: the
    # VIC-WUR 3.0 coupled loop (capillary rise into the soil, baseflow to
    # routing)
    return dict(static=GWR.static_values(), foc=(run == 'G11c'))


def build(run, mode, scratch, days=None, device='cpu', group=None,
          restart=None, window_days=31, w3_restart=False):
    gp = global_path(run)
    days = NDAYS if days is None else days
    os.makedirs(scratch, exist_ok=True)
    c = WS.load_case(gp, scratch, steps=days, device=device)
    mods = WS.modules(gp, c, group=group or 'l3_%s_%s' % (run, mode),
                      device=device, window_days=window_days, mode=mode,
                      groundwater=groundwater_cfg(run))
    cfg = dict(run=dict(device=device, log=False), modules=mods,
               connections=[])
    if restart:
        cfg['checkpoint'] = dict(restart_from=restart)
        if w3_restart:
            cfg['checkpoint']['w3_restart'] = True
    VC._GROUPS.clear()
    return TorchVIC.from_dict(cfg), c


def _cells(y, case):
    """VIC's per-column series on the cells (AreaFract-weighted)."""
    ex = case.snowbands
    y = y.double().cpu().numpy()
    if ex is None:
        return y
    out = np.zeros((y.shape[0], int(ex['n_cell'])) + y.shape[2:])
    for j, (c, w) in enumerate(zip(np.asarray(ex['cell']),
                                   np.asarray(ex['weight']))):
        out[:, c] += y[:, j] * w
    return out


def collect(m, case):
    """{VIC-WUR 3.0 name: [T, cells, ...] float64} of the run's outputs."""
    o = m.outputs('vic')
    res = {}
    for vk, k in VIC_VARS:
        if k in o:
            res[vk] = _cells(o[k], case)
    if 'soil_liq' in o:
        res['OUT_SOIL_MOIST'] = _cells(o['soil_liq'] + o['soil_ice'], case)
    mods = m.runtime.connectors
    for vk, mod, k in PLUGIN:
        if mod in mods:
            ex = m.outputs(mod)
            if k in ex:
                res[vk] = ex[k].double().cpu().numpy()
    # OUT_WITHDRAWN: local + remote + non-renewable (wu_output.c:387-392)
    if 'vicwur_nonrenew' in mods:
        res['OUT_WITHDRAWN'] = m.outputs('vicwur_nonrenew')['withdrawn'] \
            .double().cpu().numpy()
    elif 'vicwur_wateruse_local' in mods:
        w = m.outputs('vicwur_wateruse_local')['withdrawn'].double().cpu() \
            .numpy()
        if 'vicwur_wateruse_remote' in mods:
            w = w + m.outputs('vicwur_wateruse_remote')[
                'withdrawn_tremote'].double().cpu().numpy()
        res['OUT_WITHDRAWN'] = w
    return res


def local_dam_cells(run):
    from torchvic.vicwur import params as P
    d = P.load_vicwur_inputs(global_path(run)).dams
    if d is None:
        return ()
    return tuple(int(c) for c, g in zip(d.dam_cell.tolist(),
                                        d.is_global.tolist()) if not g)


def stats(x, y):
    """KGE, percent bias, RMSE (pooled over days and cells) and the gate
    number max |x - y| / max(1, |x|); x TorchVIC, y VIC-WUR 3.0."""
    ok = np.isfinite(y) & np.isfinite(x)
    x, y = x[ok].astype(np.float64), y[ok].astype(np.float64)
    d = x - y
    gate = float((np.abs(d) / np.maximum(1.0, np.abs(x))).max()) \
        if d.size else 0.0
    rmse = float(np.sqrt((d * d).mean())) if d.size else 0.0
    sy, sx = y.std(), x.std()
    my, mx = y.mean(), x.mean()
    kge = pb = None
    if sy > 0 and sx > 0 and my != 0:
        r = float(np.corrcoef(x, y)[0, 1])
        kge = float(1 - np.sqrt((r - 1) ** 2 + (sx / sy - 1) ** 2 +
                                (mx / my - 1) ** 2))
    if y.sum() != 0:
        pb = float(100.0 * d.sum() / y.sum())
    return dict(kge=kge, pbias=pb, rmse=rmse, gate=gate, mean_w3=float(my),
                first_gate_day=None)


def compare(res, a, run):
    mask = np.isfinite(a['OUT_EVAP'].values[0])
    ldc = local_dam_cells(run)
    out = {}
    for vk, x in res.items():
        if vk not in a:
            continue
        y = a[vk].values
        y = np.moveaxis(y, 1, -1)[:, mask] if y.ndim == 4 else y[:, mask]
        n = min(len(x), len(y))
        x, y = x[:n], y[:n].reshape(x[:n].shape).astype(np.float64)
        if vk == 'OUT_RUNOFF' and ldc:
            # VIC-WUR 3.0 takes a local dam's inflow out of OUT_RUNOFF after
            # routing read it: compared where no local dam is
            keep = [c for c in range(x.shape[1]) if c not in ldc]
            x, y = x[:, keep], y[:, keep]
        s = stats(x, y)
        rel = np.abs(x - y) / np.maximum(1.0, np.abs(x))
        rel = np.where(np.isfinite(rel), rel, 0.0)
        bad = np.flatnonzero(rel.reshape(n, -1).max(1) >= 1e-6)
        s['first_gate_day'] = int(bad[0]) if bad.size else None
        s['days'] = int(n)
        out[vk] = s
    return out


def run_once(m):
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    return m


def states(m):
    return {k: c.save_state() for k, c in m.runtime.connectors.items()}


def _diff(a, b, path=''):
    """First path where two state trees differ (None if bitwise equal)."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return path + ' keys'
        for k in a:
            r = _diff(a[k], b[k], '%s/%s' % (path, k))
            if r:
                return r
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return path + ' len'
        for i, (u, v) in enumerate(zip(a, b)):
            r = _diff(u, v, '%s[%d]' % (path, i))
            if r:
                return r
        return None
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            return path + ' type'
        eq = (a == b) | (torch.isnan(a) & torch.isnan(b)) \
            if a.is_floating_point() else (a == b)
        return None if bool(eq.all()) else path
    if isinstance(a, np.ndarray):
        return None if np.array_equal(a, b, equal_nan=a.dtype.kind == 'f') \
            else path
    if isinstance(a, float) and isinstance(b, float):
        return None if (a == b or (a != a and b != b)) else path
    return None if a == b else path


def ledger_rows(m):
    return [{k: v for k, v in r.items() if k != 'ok'}
            for r in m.ledger.balances]


LEDGER_DIFF = []


def ledger_rel(a, b):
    """Largest |difference| / max(scale, |x|) over the numeric fields of
    equal ledger rows, nested dicts included, the residuals left out (0.0
    when bitwise; inf if the rows differ in number, keys or a non-numeric
    field)."""
    if len(a) != len(b):
        return float('inf')

    def walk(x, y, sc, key):
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                LEDGER_DIFF.append((key, 'keys'))
                return float('inf')
            # residuals are rounding noise: compared through ok, not here
            return max([walk(x[k], y[k], sc, k) for k in x
                        if 'residual' not in k] + [0.0])
        if isinstance(x, float) and isinstance(y, float):
            return 0.0 if x == y else abs(x - y) / max(sc, abs(x), 1e-30)
        if x != y:
            LEDGER_DIFF.append((key, repr(x)[:80], repr(y)[:80]))
            return float('inf')
        return 0.0
    w = 0.0
    for x, y in zip(a, b):
        sc = abs(float(x.get('scale', 0.0) or 0.0))
        w = max(w, walk(x, y, sc, ''))
    return w


def outputs_equal(full, part, k0):
    """Every output row of `part` (a run restarted at day k0) against the
    continuous run's same days, bit for bit."""
    bad = []
    for name in full.runtime.connectors:
        a, b = full.outputs(name), part.outputs(name)
        for k, x in a.items():
            if not torch.is_tensor(x) or x.dim() == 0 or k not in b:
                continue
            y = b[k]
            if x.shape[1:] != y.shape[1:]:
                continue                 # end-of-run fields
            if y.shape[0] == x.shape[0]:
                xs = x
            else:
                xs = x[x.shape[0] - y.shape[0]:]
            eq = (xs == y) | (torch.isnan(xs) & torch.isnan(y))
            if not bool(eq.all()):
                bad.append('%s.%s' % (name, k))
    return bad


def restart_check(run, mode, scratch, full, days_list=None,
                  window_days=31):
    """Stop-and-save run against the continuous one, then each restart."""
    rep = {}
    days_list = days_list or RESTART_DAYS
    m1, _ = build(run, mode, scratch, group='rs_%s_%s' % (run, mode),
                  window_days=window_days)
    cks = []
    with torch.inference_mode():
        m1.initialize()
        for d in days_list:
            m1.run(until=d * DAY)
            ck = os.path.join(scratch, 'ck_%s_%s_%d.pt' % (run, mode, d))
            m1.save_checkpoint(ck)
            cks.append((d, ck, m1.runtime.time.t))
        m1.run()
        m1.finalize(write=False)
    d0 = _diff(states(full), states(m1))
    rep['stopping_run_states'] = d0 or 'bitwise'
    rep['stopping_run_ledger'] = ('bitwise' if ledger_rows(full) ==
                                  ledger_rows(m1) else 'differs')
    rep['stopping_run_outputs'] = outputs_equal(full, m1, 0) or 'bitwise'
    for d, ck, t in cks:
        m2, _ = build(run, mode, scratch, group='rs_%s_%s' % (run, mode),
                      restart=ck, window_days=window_days)
        run_once(m2)
        dd = _diff(states(full), states(m2))
        lf = ledger_rows(full)
        l2 = ledger_rows(m2)
        rep['restart_%d' % d] = dict(
            date=str(START + datetime.timedelta(days=d)), t=int(t),
            states=dd or 'bitwise',
            ledger='bitwise' if lf == l2 else 'differs (%d vs %d rows)'
            % (len(lf), len(l2)),
            ledger_rel=ledger_rel(lf, l2),
            ledger_ok=all(r['ok'] for r in m2.ledger.balances),
            outputs=outputs_equal(full, m2, d) or 'bitwise')
        del m2
    return rep


def season_info(m, days_list):
    """At each restart day: crops growing, the Cv record and the MODFLOW
    month the checkpoint falls in."""
    info = {}
    mods = m.runtime.connectors
    for d in days_list:
        date = START + datetime.timedelta(days=d)
        e = dict(date=str(date), mid_month=date.day not in (1,))
        if 'vicwur_crop' in mods:
            g = m.outputs('vicwur_crop')['OUT_CROP_GROW']
            e['crops_growing'] = int((g[d - 1] > 0).sum()) if d - 1 < len(g) \
                else None
        info[d] = e
    return info


def do_run(run, mode, out, device='cpu', restart=False, w3_restart=False):
    os.makedirs(out, exist_ok=True)
    scratch = os.path.join(out, 'scratch', '%s_%s' % (run, mode))
    t0 = time.time()
    rep = dict(run=run, mode=mode, device=device)
    if run == 'G12':
        # G10 split at 2007-01-01; window_days 32 puts it on a boundary
        m1, case = build(run, mode, scratch, group='g12_%s' % mode,
                         window_days=32)
        ck = os.path.join(scratch, 'ck_g12_%s.pt' % mode)
        with torch.inference_mode():
            m1.initialize().run(until=SPLIT * DAY)
            m1.save_checkpoint(ck)
        m2, _ = build(run, mode, scratch, group='g12_%s' % mode,
                      restart=ck, window_days=32, w3_restart=w3_restart)
        run_once(m2)
        r1 = collect(m1, case)
        r2 = collect(m2, case)
        res = {k: np.concatenate([r1[k][:SPLIT], r2[k][-(NDAYS - SPLIT):]
                                  if len(r2[k]) >= NDAYS - SPLIT else r2[k]])
               for k in r1 if k in r2}
        full, _ = build(run, mode, scratch, group='g12f_%s' % mode,
                        window_days=32)
        run_once(full)
        rep['restart_vs_continuous_G10'] = dict(
            states=_diff(states(full), states(m2)) or 'bitwise',
            outputs=outputs_equal(full, m2, SPLIT) or 'bitwise')
        rep['w3_restart'] = bool(w3_restart)
        m = m2
        rep['vs_W3_G10'] = compare(res, w3('G10'), 'G10')
    else:
        m, case = build(run, mode, scratch)
        run_once(m)
        res = collect(m, case)
    rep['seconds'] = time.time() - t0
    rep['ledger_ok'] = all(r['ok'] for r in m.ledger.balances) and all(
        r['ok'] for r in m.ledger.coupled)
    rep['ledger_rows'] = len(m.ledger.balances)
    rep['stats'] = compare(res, w3(run), run)
    rep['gate_max'] = max([s['gate'] for s in rep['stats'].values()] + [0])
    if restart and run != 'G12':
        rep['season'] = {str(k): v for k, v in
                         season_info(m, RESTART_DAYS).items()}
        rep['restart'] = restart_check(run, mode, scratch, m)
    rep['seconds_total'] = time.time() - t0
    fn = os.path.join(out, '%s_%s%s.json' % (run, mode,
                                             '_w3rs' if w3_restart else ''))
    with open(fn, 'w') as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps(dict(run=run, mode=mode, gate=rep['gate_max'],
                          ledger=rep['ledger_ok'],
                          restart={k: v for k, v in rep.get('restart',
                                                            {}).items()}),
                     default=str))
    return rep


ORDER = ('OUT_EVAP', 'OUT_TRANSP_VEG', 'OUT_EVAP_BARE', 'OUT_EVAP_CANOP',
         'OUT_RUNOFF', 'OUT_BASEFLOW', 'OUT_SOIL_MOIST', 'OUT_GWRECHARGE',
         'OUT_DISCHARGE', 'OUT_STREAM_MOIST', 'OUT_WITHDRAWN',
         'OUT_NONREN_DEFICIT', 'OUT_REQUIREMENT', 'OUT_RECEIVED',
         'OUT_APPLIED', 'OUT_LEFTOVER', 'OUT_SHORTAGE', 'OUT_EFR_DISCHARGE',
         'OUT_EFR_BASEFLOW', 'OUT_EFR_MOIST', 'OUT_CV') + tuple(CROPV)


def table(out):
    """Markdown tables: per run and output, fixed-mode KGE / pbias / RMSE
    against VIC-WUR 3.0 and the exact-mode gate number."""
    rows = ['| Run | Output | KGE (fixed) | bias % (fixed) | RMSE (fixed) '
            '| exact: max rel |', '|---|---|---|---|---|---|']
    f = lambda v, p: '-' if v is None else p % v
    for run in RUNS:
        fx = os.path.join(out, '%s_fixed.json' % run)
        ex = os.path.join(out, '%s_exact.json' % run)
        if not os.path.exists(fx):
            continue
        A = json.load(open(fx))['stats']
        E = json.load(open(ex))['stats'] if os.path.exists(ex) else {}
        for vk in ORDER:
            if vk not in A:
                continue
            s = A[vk]
            e = E.get(vk, {}).get('gate')
            rows.append('| %s | %s | %s | %s | %s | %s |' % (
                run, vk[4:], f(s['kge'], '%.6f'), f(s['pbias'], '%.3g'),
                f(s['rmse'], '%.3g'), f(e, '%.1e')))
    return '\n'.join(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('run', 'table'))
    p.add_argument('run', nargs='?')
    p.add_argument('mode', nargs='?', default='fixed')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=OUT)
    p.add_argument('--restart', action='store_true')
    p.add_argument('--w3-restart', action='store_true')
    a = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    if a.cmd == 'table':
        print(table(a.out))
        return
    for r in a.run.split(','):
        do_run(r, a.mode, a.out, a.device, a.restart, a.w3_restart)


if __name__ == '__main__':
    main()
