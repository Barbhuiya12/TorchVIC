"""Coupled run of mode B (TorchVIC VIC + live CropSyst in the VIC-WUR phase group) against the
VIC-CropSyst reference build's own coupled outputs (<data root>/verification/cropsyst_i8i13/runs_off:
crop csv and fluxes per cell, 1979-1983).

`build(case, scratch, ...)` makes the coupled model of one CRB case from the reference run's
inputs: VIC 5 on the CRB files with 17 layers (tools/cropsyst_i12/crb_case.py), the core hooks
(Saxton hydraulics, Wpwp floor, CropSyst uptake as transpiration, CropSyst LAI), the member
`vicwur_cropsyst` and, for the irrigated cases, the irrigation scheme fed by the live crop.
`stats(case, ...)` returns KGE, bias and RMSE per variable against the reference.

usage: cropsyst_i12_l3.py [--device cpu|cuda] [--days N] [--out DIR] [--no-irrigation] [case ...]
"""
import csv
import glob
import json
import math
import os
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'cropsyst_i12'))
import _paths  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

RUNS = _paths.env_path(('TORCHVIC_CROPSYST_I8_RUNS',), 'verification/cropsyst_i8i13') + "/runs_off"
CASES = (330492, 308200, 328617, 310974, 259094, 315604, 318377)
ANNUAL = (330492, 308200, 328617, 310974, 259094)
#: the cases whose crops have an automatic irrigation operation (CRB management files)
IRRIGATED = {259094: 'CENTER_PIVOT', 310974: 'CENTER_PIVOT', 315604: 'DRIP', 318377: 'SPRINKLER'}
ORCHARD = (315604, 318377)
DAY = 86400.0


def run_dir(case):
    f = glob.glob(f"{RUNS}/Liu_Linux_{case}_*")
    if not f:
        raise FileNotFoundError(f"reference run of {case} not present in {RUNS}")
    return f[0]


def read_csv(case):
    """The reference crop csv: {tile code: {column: np.array}}, tile codes sorted."""
    f = glob.glob(f"{run_dir(case)}/VIC_output/vic_crop_daily*")[0]
    with open(f) as fh:
        rows = list(csv.reader(fh))
    H = {h: i for i, h in enumerate(rows[0])}
    by = defaultdict(list)
    for r in rows[1:]:
        by[r[H["CroppingSyst_code"]]].append(r)
    out = {}
    for k in sorted(by):
        out[k] = {h: np.array([r[i] for r in by[k]]) for h, i in H.items() if h}
    return out


def read_fluxes(case):
    f = glob.glob(f"{run_dir(case)}/VIC_output/fluxes_crop_*")[0]
    names, rows = None, []
    with open(f) as fh:
        for ln in fh:
            if ln.startswith('# YEAR'):
                names = ln[1:].split()
            elif not ln.startswith('#') and ln.strip():
                rows.append([float(x) for x in ln.split()])
    a = np.array(rows)
    return {n: a[:, i] for i, n in enumerate(names)}


def soil_inputs(case):
    soil = _control(case)['SOIL'][0]
    t = open(soil).read().split()
    f = lambda a, b: [float(x) for x in t[a:b]]
    return dict(thickness=f(78, 95), quartz=f(114, 131), clay=f(273, 290))


def _control(case):
    d = {}
    for ln in open(os.path.join(run_dir(case), 'vic_control.txt')):
        s = ln.split('#')[0].split()
        if len(s) >= 2:
            d.setdefault(s[0], s[1:])
    return d


def build(case, scratch, device='cpu', days=None, irrigation=None, hooks=None, group=None,
          window_days=31, restart=None, wpwp_floor=True, vcs_drainage=True, vcs_drain=True, runoff_steps=24, canopy_evap=False, soil_evap=False):
    """The coupled model of `case`. irrigation: None (the case's own: on for the irrigated
    cases), True or False. Returns (model, info)."""
    from torchvic import case as vcase
    from torchvic.coupling import TorchVIC
    from torchvic.coupling.vic import CaseDriver, VicConnector
    from torchvic.coupling.connectors import vicwur_common as VC
    from torchvic.crop.cropsyst import ptf
    from torchvic.crop.cropsyst import readers as rd
    from torchvic.vicwur import cs_core
    import crb_case

    dev = torch.device(device)
    ref = read_csv(case)
    tiles = [int(k) for k in ref]
    ci = crb_case.make_case(run_dir(case), os.path.join(scratch, f'c{case}'), tiles,
                            runoff_steps=runoff_steps)
    if device == 'cuda':
        torch.set_default_device('cuda')
    c = vcase.load(ci['global_file'], core='vicwur3', device=dev, steps=days or 0)
    soil = soil_inputs(case)
    D = torch.float64
    # FCAN_SRC FROM_DEFAULT has no cover table; the crop's LAI sets it (member hook)
    c.mosaic.fcanopy = torch.ones(c.mosaic.cv.shape + (12,), dtype=D, device=dev)
    # the reference's first-day layer moisture is the initial state (its init_moist convention
    # differs from the file's per-layer values by a factor of about 2: a spin-up choice)
    fx0 = read_fluxes(case)
    c.init_moist = torch.tensor([[fx0['OUT_SOIL_MOIST_%d' % i][0] for i in range(17)]], dtype=D, device=dev)
    t = lambda x: torch.tensor([x], dtype=D, device=dev)
    sx = ptf.saxton_ptf(t(soil['clay']), t(soil['quartz']), t(soil['thickness']))
    c.soil = cs_core.saxton_soil_params(c.soil, t(soil['clay']), t(soil['quartz']), t(soil['thickness']),
                                        vcs_drainage=vcs_drainage)
    group = group or f'crb{case}'
    T = len(tiles)
    irr_on = (case in IRRIGATED) if irrigation is None else bool(irrigation)
    nd = c.n_steps
    cs = dict(db_dir=DB,
              tiles=tiles, tile_cell=[0] * T, soil=soil, lon=[ci['lon']])
    mods = {}
    h = dict(transp=True, lai=True, wpwp_floor=wpwp_floor, vcs_drain=vcs_drain,
             canopy_evap=canopy_evap, soil_evap=soil_evap)
    h.update(hooks or {})
    mods['vicwur_cropsyst'] = dict(connector='vicwur_cropsyst', cropsyst=cs, tile=list(range(T)),
                                   lat=[ci['lat']], lon=[ci['lon']], n_cells=1, hooks=h, group=group,
                                   irrigated=[irr_on] * T)
    if irr_on:
        tech = IRRIGATED.get(case, 'CENTER_PIVOT')
        orch = case in ORCHARD
        observe = (dict(mode=0, const_depth_m=1.0) if orch
                   else dict(mode=1, root_fraction=0.6))
        mm = c.soil.max_moist.to(D)
        mods['vicwur_irrigation'] = dict(
            connector='vicwur_irrigation', group=group, potential=True, scheme='cropsyst',
            irr=dict(veg_class=list(range(1, T + 1)), paddy=[False] * T, active=[[True] * T],
                     gw_fraction=[[0.0] * T], efficiency=[[0.6] * T], offset=[[0] * T]),
            soil=dict(max_moist=mm, wcr=c.soil.w_crit_mm, wfc=sx.field_capacity_mm[0:1],
                      wpwp=c.soil.w_wilt_mm,
                      root=c.mosaic.root_frac[:, :T]),
            area=torch.as_tensor(np.asarray(c.dom.area, dtype=np.float64).reshape(-1)[:1], device=dev),
            cropsyst=dict(crop_fn='group', depth=t(soil['thickness']),
                          code=torch.full((1, T), rd.irrigation_type_code(tech)),
                          window=(3, 10), observe=observe))
    drv = CaseDriver(c)
    vic = VicConnector(driver=drv, window_days=window_days, soil_add_floor='liquid',
                       day_coupled=sorted(mods), phase_group=group)
    cfg = dict(run=dict(device=device, log=False), modules=dict(vic=dict(connector='vic', global_param=ci['global_file']), **mods),
               connections=[])
    if restart:
        cfg['checkpoint'] = dict(restart_from=restart)
    VC._GROUPS.pop(group, None)
    m = TorchVIC.from_dict(cfg, connectors=dict(vic=vic))
    return m, dict(case=case, tiles=tiles, nd=nd, dates=c.dates, ref=ref, c=c, info=ci)


DB = _paths.data('build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation/Database')


def run(case, scratch, device='cpu', days=None, **kw):
    m, info = build(case, scratch, device, days, **kw)
    with torch.inference_mode():
        m.initialize().run()
        m.finalize(write=False)
    return m, info


# ---------------------------------------------------------------------------------- statistics
def kge(sim, obs):
    sim, obs = np.asarray(sim, float), np.asarray(obs, float)
    if sim.size < 2 or obs.std() == 0 or obs.mean() == 0:
        return float('nan')
    r = np.corrcoef(sim, obs)[0, 1] if sim.std() > 0 else 0.0
    a = sim.std() / obs.std()
    b = sim.mean() / obs.mean()
    return float(1.0 - math.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2))


def score(sim, obs):
    sim, obs = np.asarray(sim, float), np.asarray(obs, float)
    ok = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[ok], obs[ok]
    if sim.size == 0:
        return dict(n=0, kge=float('nan'), bias=float('nan'), bias_pct=float('nan'), rmse=float('nan'),
                    ref_mean=float('nan'), sim_mean=float('nan'))
    d = sim - obs
    return dict(n=int(sim.size), kge=kge(sim, obs), bias=float(d.mean()),
                bias_pct=float(100 * d.mean() / obs.mean()) if obs.mean() != 0 else float('nan'),
                rmse=float(math.sqrt((d * d).mean())), ref_mean=float(obs.mean()), sim_mean=float(sim.mean()))


def series(m, name):
    """Per-day rows of a connector's committed series, [n, ...] (row j is day j)."""
    return {k: v.double().cpu().numpy() for k, v in m.outputs(name).items() if torch.is_tensor(v)}


def _irr_evap(m, n):
    """Evaporation of the irrigation system (scheme cropsyst, Ed), mm per day, zeros without it."""
    if 'vicwur_irrigation' in m.runtime.connectors:
        ir = series(m, 'vicwur_irrigation')
        if 'evap' in ir:
            return ir['evap'][:n, 0]
    return np.zeros(n)


def stats(m, info):
    """KGE, bias (sim - ref), RMSE of yield, biomass, LAI (GAI), ET and soil moisture."""
    ref, tiles = info['ref'], info['tiles']
    cs = series(m, 'vicwur_cropsyst')
    vic = series(m, 'vic')
    n = min(cs['lai'].shape[0], len(ref[str(tiles[0])]['Year']))
    out = {}
    cols = dict(biomass=('Biomass_kg_m2', 'biomass', 1.0), lai=('GAI', 'gai', 1.0))
    acc = {k: ([], []) for k in ('yield', 'biomass', 'lai')}
    sow = dict(ref=0, sim=0)
    harv = dict(ref=0, sim=0)
    for k, code in enumerate(tiles):
        r = ref[str(code)]
        present = r['Crop_name'][:n] != 'NOCROP'
        hv = (r['Yield_kg_m2'][:n].astype(float) > 0) & present
        for name, (col, key, fac) in cols.items():
            sel = present & ~hv if name != 'lai' else present & ~hv
            acc[name][0].extend(cs[key][:n, k][sel] * fac)
            acc[name][1].extend(r[col][:n].astype(float)[sel])
        ry = r['Yield_kg_m2'][:n].astype(float)[hv]
        sy = cs['yield'][:n, k][cs['harvest'][:n, k] > 0] * 1e-4
        harv['ref'] += int(hv.sum())
        harv['sim'] += int((cs['harvest'][:n, k] > 0).sum())
        sw = (np.diff(np.r_[0, present.astype(int)]) == 1).sum()
        sow['ref'] += int(sw)
        sow['sim'] += int((cs['sown'][:n, k] > 0).sum())
        m_ = min(len(ry), len(sy))
        acc['yield'][0].extend(sy[:m_])
        acc['yield'][1].extend(ry[:m_])
    for k, (s, o) in acc.items():
        out[k] = score(s, o)
    fx = info['flux']
    et_s = vic['et'][:n, 0] + _irr_evap(m, n)
    out['et'] = score(et_s, fx['OUT_EVAP'][:n])
    sm_ref = sum(fx['OUT_SOIL_MOIST_%d' % i][:n] for i in range(17))
    sm_s = (vic['soil_liq'][:n, 0] + vic['soil_ice'][:n, 0]).sum(-1)
    out['soil_moisture'] = score(sm_s, sm_ref)
    out['_events'] = dict(harvests=harv, sowings=sow, days=int(n))
    return out


def extra(m, info):
    """Diagnostics: annual totals of precipitation, ET, runoff, baseflow, irrigation."""
    fx = info['flux']
    vic = series(m, 'vic')
    n = min(vic['et'].shape[0], len(fx['OUT_EVAP']))
    d = dict(prec_ref=float(fx['OUT_PREC'][:n].sum()), et_ref=float(fx['OUT_EVAP'][:n].sum()),
             et_sim=float(vic['et'][:n, 0].sum() + _irr_evap(m, n).sum()),
             runoff_ref=float(fx['OUT_RUNOFF'][:n].sum()), runoff_sim=float(vic['runoff'][:n, 0].sum()),
             baseflow_ref=float(fx['OUT_BASEFLOW'][:n].sum()), baseflow_sim=float(vic['baseflow'][:n, 0].sum()),
             transp_sim=float(vic['transp'][:n, 0].sum()), evap_bare_sim=float(vic['evap_bare'][:n, 0].sum()),
             evap_canop_sim=float(vic['evap_canop'][:n, 0].sum()))
    if 'vicwur_irrigation' in m.runtime.connectors:
        ir = series(m, 'vicwur_irrigation')
        r = info['ref']
        d['irrig_applied_sim'] = float(ir['applied'].sum())
        d['irrig_ref_gross_mean_tile'] = float(np.mean([r[k]['irrig_total_mm'].astype(float).sum() for k in r]))
        d['irrig_ref_net_mean_tile'] = float(np.mean([r[k]['net_irrig_total_mm'].astype(float).sum() for k in r]))
    return d


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*', type=int)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--days', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-irrigation', action='store_true')
    ap.add_argument('--no-vcs-drain', action='store_true')
    ap.add_argument('--canopy-evap', action='store_true')
    ap.add_argument('--soil-evap', action='store_true')
    ap.add_argument('--runoff-steps', type=int, default=24)
    a = ap.parse_args(argv)
    res = {}
    for case in a.cases or CASES:
        scratch = tempfile.mkdtemp(prefix=f'i12_{case}_')
        try:
            m, info = run(case, scratch, a.device, a.days,
                          irrigation=False if a.no_irrigation else None,
                          vcs_drain=not a.no_vcs_drain, runoff_steps=a.runoff_steps,
                          canopy_evap=a.canopy_evap, soil_evap=a.soil_evap)
            info['flux'] = read_fluxes(case)
            res[case] = dict(stats=stats(m, info), extra=extra(m, info),
                             ledger_ok=all(r['ok'] for r in m.ledger.balances))
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                np.savez(os.path.join(a.out, f'series_{case}.npz'),
                         ir_evap=_irr_evap(m, len(info['flux']['OUT_EVAP'])),
                         **{('cs_' + k): v for k, v in series(m, 'vicwur_cropsyst').items()},
                         **{('vic_' + k): v for k, v in series(m, 'vic').items()
                            if v.ndim >= 2 and v.shape[0] > 100})
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            res[case] = dict(error=f'{type(e).__name__}: {e}')
        print(case, json.dumps(res[case], default=float), flush=True)
    if a.out:
        with open(os.path.join(a.out, 'l3_stats.json'), 'w') as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1:])
