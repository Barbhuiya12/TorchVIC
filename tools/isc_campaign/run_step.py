"""Run one step of the ISC build-up matrix and save the daily cell series.

  run_step.py STEP [--device cuda|cpu] [--days N] [--tag T] [--until DOY-DATE]
              [--ckpt-at YYYY-MM-DD] [--restart-from FILE] [--det]

Output (<data root>/verification/isc_campaign/out/<tag>/): series.npz (the 2005
window of every kept variable, cell level, float32, 'module.var'), ledger.json
(one row per module and interval: residual, scale, ok), meta.json.
VIC's per-column series are reduced to cells on the fly (the full set of
VIC outputs is 24 GB a year); the other modules' series are kept whole.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'tools', 'vicwur', 'isc'))
sys.path.insert(0, REPO)
import steps as S                                            # noqa: E402

#: VIC series kept (cell level): flux (sum when sub-daily) or state
VIC_KEEP = {'rainf': 'f', 'snowf': 'f', 'et': 'f', 'runoff': 'f',
            'baseflow': 'f', 'gwrecharge': 'f', 'recharge': 'f',
            'water_error': 'f', 'swe': 's', 'wdew': 's', 'air_temp': 's',
            'swdown': 's', 'snow_cover': 's'}
VIC_SOIL = ('soil_liq', 'soil_ice')
CS_KEEP = tuple(os.environ.get('TV_CS_KEEP', 'yield+harvest+sown').split('+'))
SKIP = {('vicwur_landuse', 'cv')}     # [T, N, 20]: 3 GB, yearly steps only


SPUN = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'vicwur/isc/init_spun/params_wur_spun.nc')


def install_reducer(ex, n_cells, spd):
    from torchvic.coupling import managers as M
    from torchvic.core import snowbands
    commit0 = M.OutputManager.commit
    nb = len(ex['cell']) if ex is not None else n_cells

    def red(v):
        if v.dim() >= 2 and v.shape[1] == nb and ex is not None and nb != n_cells:
            return snowbands.contract(v, ex, axis=1)
        return v

    def commit(self, name, proposal):
        ser = proposal.series
        if name == 'vic' and isinstance(ser, dict):
            out = {}
            for k, v in ser.items():
                if not torch.is_tensor(v):
                    continue
                if k in VIC_KEEP and v.dim() == 2:
                    out[k] = red(v).float()
            if all(k in ser for k in VIC_SOIL):
                out['soil_moist'] = (red(ser['soil_liq']).sum(-1)
                                     + red(ser['soil_ice']).sum(-1)).float()
            proposal.series = out
        elif name == 'vicwur_cropsyst' and isinstance(ser, dict):
            # per-tile series (up to 7e5 tiles): only the yield record stays
            proposal.series = {k: v for k, v in ser.items() if k in CS_KEEP and torch.is_tensor(v)}
        elif isinstance(ser, dict):
            # cell series only: per-tile / per-leaf-class tensors (WOFOST) are
            # tens of GB on the host
            proposal.series = {
                k: v for k, v in ser.items() if (name, k) not in SKIP
                and not (torch.is_tensor(v) and v.dim() > 2)}
        return commit0(self, name, proposal)
    M.OutputManager.commit = commit


def daily(a, k, spd):
    if spd == 1:
        return a
    d = a.reshape(a.shape[0] // spd, spd, *a.shape[1:])
    return d.sum(1) if k == 'f' else d.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--days', type=int, default=0)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--mode', default=None)
    ap.add_argument('--ckpt-at', default=None)
    ap.add_argument('--restart-from', default=None)
    ap.add_argument('--det', action='store_true')
    ap.add_argument('--start', default=None, help='YYYY-MM-DD')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD (default steps.END); needs forcing of every year up to it; the whole series is kept')
    ap.add_argument('--keep-all', action='store_true')
    ap.add_argument('--window', type=int, default=31)
    ap.add_argument('--out-root', default=None)
    ap.add_argument('--subset', action='store_true')
    ap.add_argument('--subset-dir', default=None,
                    help='sub-domain dir of subset.py (implies --subset; '
                         'default subset500/)')
    ap.add_argument('--cs-portions', type=int, default=0, help='CropSyst canopy portions per tile (default: the engine default)')
    ap.add_argument('--cs-canopy', action='store_true', help='CropSyst hooks canopy_evap and soil_evap')
    ap.add_argument('--no-spun', action='store_true', help='use the original init_moist, not init_spun/')
    ap.add_argument('--params', default=None, help='parameter file replacing the global file\'s PARAMETERS (spin_init.py init_moist)')
    ap.add_argument('--debug-nan', action='store_true')
    ap.add_argument('--cap', action='store_true', help="soil_add_mode 'cap' (take what fits) instead of refusing")
    ap.add_argument('--work', default=None, help='tag of the dir holding the global file and scratch (a restart needs the same config)')
    a = ap.parse_args()
    sub = a.subset_dir.rstrip('/') + '/' if a.subset_dir else \
        S.OUT + 'subset500/'
    a.subset = a.subset or bool(a.subset_dir)
    if a.det:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import isc_run as R
    tag = a.tag or a.step
    od = (a.out_root or S.OUT + 'out') + '/' + tag
    os.makedirs(od, exist_ok=True)
    wd = (a.out_root or S.OUT + 'out') + '/' + (a.work or tag)
    os.makedirs(wd, exist_ok=True)
    R.SCRATCH = wd + '/scratch'
    R.KEEP = 'host'
    R.FORCING = 'host'
    start = S.START if not a.start else tuple(int(x) for x in a.start.split('-'))
    gp = wd + '/global.txt'
    end = S.END if not a.end else tuple(int(x) for x in a.end.split('-'))
    mode = S.make_global(a.step, gp, start=start, end=end,
                         subset=sub if a.subset else None)
    mode = a.mode or mode
    if not a.params and not a.no_spun and not a.subset and os.path.exists(SPUN):
        a.params = SPUN
    if a.params:
        lines = [('PARAMETERS %s\n' % a.params if l.split()[:1] == ['PARAMETERS'] else l)
                 for l in open(gp).read().splitlines(True)]
        open(gp, 'w').write(''.join(lines))
    s0 = datetime.date(*start)
    n = (datetime.date(*end) - s0).days + 1
    if a.days:
        n = min(n, a.days)
    on = S.STEPS[a.step][0]
    static = dict(np.load(sub + 'gw_static_sub.npz' if a.subset
                          else S.SMOKE + 'gw_static.npz')) \
        if 'groundwater' in on else None
    t0 = time.time()
    log = lambda m: R.log('[%s] %s' % (tag, m))
    log('build %s mode %s days %d device %s' % (a.step, mode, n, a.device))
    cs = None
    if 'cropsyst' in on:
        cs = dict(n_portions=a.cs_portions or None,
                  hooks=dict(canopy_evap=a.cs_canopy, soil_evap=a.cs_canopy))
    m, c = R.build(gp, n, a.device, mode, a.window, static, cropsyst=cs)
    ex = c.snowbands
    ncell = int(ex['n_cell']) if ex is not None else None
    del c
    install_reducer(ex, ncell, 1)
    if a.cap:
        m.runtime.connectors['vic'].soil_add_mode = 'cap'
    if a.debug_nan:
        # check every step, and say where the first non-finite moisture is
        m.runtime.cfg['ledger']['speculate'] = 1
        from torchvic.coupling import vic as V
        v0 = V.VicConnector.validate

        def validate(self, proposal, context):
            st = proposal.state
            mo = st.get('moist') if isinstance(st, dict) else None
            if mo is not None and not torch.isfinite(mo).all():
                bad = (~torch.isfinite(mo)).nonzero()
                lanes = self._lanes(self.driver.mosaic(), st)
                lc = lanes.lane_cell[bad[:, 0]].cpu().numpy()
                lt = lanes.lane_tile[bad[:, 0]].cpu().numpy()
                log('NaN moisture: %d lanes in %d cells; shape %s'
                    % (len(lc), len(set(lc.tolist())), tuple(mo.shape)))
                np.save(od + '/nan_cells.npy', np.stack([lc, lt]))
                log('first cells %s tiles %s' % (lc[:10].tolist(), lt[:10].tolist()))
            return v0(self, proposal, context)
        V.VicConnector.validate = validate
    if a.restart_from:
        m.runtime.cfg['checkpoint']['restart_from'] = a.restart_from
    with torch.inference_mode():
        m.initialize()
        if a.ckpt_at:
            until = (datetime.date.fromisoformat(a.ckpt_at) - s0).days * 86400.0
            m.run(until=until)
            m.save_checkpoint(od + '/ckpt.pt')
            log('checkpoint at %s' % a.ckpt_at)
        m.run()
        m.finalize(write=False)
    log('run done %.0f s' % (time.time() - t0))
    ser = {}
    for name in m.runtime.connectors:
        o = m.outputs(name)
        for k, v in o.items():
            if k in ('state',) or not torch.is_tensor(v):
                continue
            if v.dim() < 2 or not v.is_floating_point():
                continue
            ser['%s.%s' % (name, k)] = v.cpu().numpy().astype(np.float32)
    # the last 365 rows of VIC (1 Jan - 31 Dec 2005) and 364 of the others
    # (the connectors' first and last day are dropped): the 2005 window
    if (not a.days or a.keep_all) and not a.subset and not a.end:
        ser = {k: (v[-365:] if k.startswith('vic.') else v[-364:])
               for k, v in ser.items()}
    np.savez(od + '/series.npz', **ser)
    if R.CS_TILES:
        np.savez(od + '/cs_tiles.npz', **R.CS_TILES)
    rows = []
    for r in m.ledger.balances:
        rows.append({k: (float(x) if isinstance(x, (int, float, np.floating))
                         and not isinstance(x, bool) else x)
                     for k, x in r.items()
                     if isinstance(x, (int, float, str, bool, np.floating,
                                       type(None)))})
    json.dump(dict(balances=rows, coupled=[
        {k: x for k, x in r.items()
         if isinstance(x, (int, float, str, bool, type(None)))}
        for r in m.ledger.coupled]), open(od + '/ledger.json', 'w'),
        default=float)
    json.dump(dict(step=a.step, mode=mode, days=n, start=str(s0),
                   device=a.device, wall_s=time.time() - t0,
                   gpu_gb=(torch.cuda.max_memory_allocated() / 2 ** 30
                           if a.device == 'cuda' else 0),
                   ledger=R.ledger_status(m)),
              open(od + '/meta.json', 'w'), default=float, indent=1)
    log('saved %d series; ledger all_ok=%s' % (
        len(ser), R.ledger_status(m)['all_ok']))
    if a.ckpt_at is None and not a.restart_from:
        pass


if __name__ == '__main__':
    main()
