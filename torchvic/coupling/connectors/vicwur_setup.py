# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""The VIC-WUR connectors' configuration from a VIC-WUR global file.

    from torchvic.coupling.connectors import vicwur_setup as WS
    case = WS.load_case(global_file, scratch_dir)      # the vicwur3 core
    mods = WS.modules(global_file, case)                # 'vic' included
    model = TorchVIC.from_dict(dict(run=..., modules=mods))

`modules` reads the plugin files of the global file (torchvic.vicwur.
params, the PLUGIN_FORCE_TYPE streams through torchvic.vicwur.forcing) and
returns one configuration block per switched-on plugin, all in one group,
plus the 'vic' block (the image global file load_case wrote, the vicwur3
core) whose phase group is that group. The
soil statics are formed the way the image driver forms them (vic_init.c:
636-795: max_moist = depth (1 - bulk/soil density) 1000, Wcr, Wpwp, Wfc =
Wpwp + (Wcr - Wpwp) / 0.7 as fractions times max_moist), the band
AreaFract and Tfactor as vic_init.c:1011-1072 does.
"""
import os
import datetime
import warnings

import numpy as np
import torch

from ...vicwur import params as P
from ...vicwur import forcing as F
from ...vicwur import wateruse as WU

SECTORS = ('MUN', 'ENE', 'MAN', 'LIV')      # wateruse.MUN .. LIV
LAPSE_RATE = -0.0065                         # initialize_parameters.c:39


def load_case(global_path, scratch, core='vicwur3', steps=0, device='cpu',
              dtype=torch.float64):
    """case.load of a VIC-WUR global file (its plugin keys dropped,
    vicwur.core.image_global) on the vicwur3 core."""
    import os
    from ... import case as _case
    from ...vicwur import core as VC
    gf = VC.image_global(global_path, os.path.join(scratch,
                                                   'image_global.txt'))
    c = _case.load(gf, device=torch.device(device), core=core, steps=steps,
                   dtype=dtype, band_tfactor=crop_tfactor(global_path))
    c.vicwur_image_global = (gf, core, steps)
    return c


def crop_tfactor(global_path):
    """{0: [N] crop nc Tfactor} when the run has WOFOST (VIC-WUR 3.0
    crop_init.c:157-163 sets soil_con Tfactor[0] from it for all of VIC),
    else None."""
    g = P.parse_global(global_path)
    if not _gb(g, 'WOFOST'):
        return None
    import netCDF4
    grid = P.Grid(_gv(g, 'DOMAIN'), _gv(g, 'PARAMETERS'))
    with netCDF4.Dataset(_gv(g, 'CROP_PARAMETERS')) as ds:
        tf = np.array(ds['Tfactor'][:], dtype=np.float64)
    return {0: grid.take(tf)}


def soil_statics(params_path, grid):
    """The soil_con values the plugins read, [N, L] (or [N]) numpy."""
    import netCDF4
    keys = ('depth', 'bulk_density', 'soil_density', 'Wcr_FRACT',
            'Wpwp_FRACT', 'resid_moist', 'Ksat', 'Dsmax', 'Ds', 'Ws', 'c',
            'rough')
    with netCDF4.Dataset(params_path) as ds:
        raw = {k: np.array(ds[k][:], dtype=np.float64) for k in keys}
        wfc_f = (np.array(ds['Wfc_FRACT'][:], dtype=np.float64)
                 if 'Wfc_FRACT' in ds.variables else None)
    f = lambda k: grid.take(raw[k])
    depth = f('depth').T
    por = 1 - f('bulk_density').T / f('soil_density').T
    mm = depth * por * 1000.
    wcr, wpwp = f('Wcr_FRACT').T, f('Wpwp_FRACT').T
    wfc = wpwp + (wcr - wpwp) / 0.7 if wfc_f is None else grid.take(wfc_f).T
    return dict(depth=depth, max_moist=mm, wcr=wcr * mm, wfc=wfc * mm,
                wpwp=wpwp * mm, resid=f('resid_moist').T, ksat=f('Ksat').T,
                dsmax=f('Dsmax'), ds=f('Ds'), ws=f('Ws'), c=f('c'),
                rough=f('rough'))


def thermal_statics(params_path, grid, nnodes=3):
    """The node statics lu_apply's energy step reads (QUICK_FLUX, the
    water-balance default: vic_init.c:866-876 dz_node = depth0, depth0,
    2 (dp - 1.5 depth0), Zsum_node = 0, depth0, dp; set_node_parameters for
    the node porosity, expt, bubble) and the layer statics of its heat
    capacities, as dict of [N, *] numpy (organic 0: ORGANIC_FRACT FALSE)."""
    import netCDF4
    from ...core import soiltemp
    keys = ('depth', 'bulk_density', 'soil_density', 'quartz', 'expt',
            'bubble', 'dp')
    with netCDF4.Dataset(params_path) as ds:
        raw = {k: np.array(ds[k][:], dtype=np.float64) for k in keys}
    f = lambda k: grid.take(raw[k])
    depth, dp = f('depth').T, f('dp')
    bd, sd = f('bulk_density').T, f('soil_density').T
    if nnodes != 3:
        raise ValueError('thermal_statics: QUICK_FLUX has 3 nodes')
    d0 = depth[:, 0]
    dz = np.stack([d0, d0, 2. * (dp - 1.5 * d0)], 1)
    zsum = np.stack([np.zeros_like(d0), d0, dp], 1)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a))
    por = 1 - bd / sd
    pn, en, bn = soiltemp.node_parameters(
        t(zsum), t(depth), t(por), t(f('expt').T), t(f('bubble').T), 3,
        depth.shape[1], False)[:3]
    return dict(dz_node=dz, zsum_node=zsum, max_moist_node=pn.numpy(),
                expt_node=en.numpy(), bubble_node=bn.numpy(), depth=depth,
                soil_dens_min=sd, bulk_dens_min=bd, soil_density=sd,
                bulk_density=bd, quartz=f('quartz').T,
                organic=np.zeros_like(bd),
                frost_fract=np.ones((depth.shape[0], 1)))


def bands(params_path, grid):
    """AreaFract and Tfactor [N, B] (vic_init.c:1011-1072)."""
    import netCDF4
    with netCDF4.Dataset(params_path) as ds:
        af = grid.take(np.array(ds['AreaFract'][:], np.float64)).T.copy()
        be = grid.take(np.array(ds['elevation'][:], np.float64)).T
        el = grid.take(np.array(ds['elev'][:], np.float64)).copy()
    n, nb = af.shape
    tf = np.zeros((n, nb))
    for i in range(n):
        s = 0.0
        for j in range(nb):
            s += af[i, j]
        if not abs(s - 1.0) <= 1e-5:
            af[i] = af[i] / s
        mean = 0.0
        for j in range(nb):
            mean += be[i, j] * af[i, j]
        if not abs(el[i] - mean) <= 1e-5:
            el[i] = mean
        for j in range(nb):
            tf[i, j] = (be[i, j] - el[i]) * LAPSE_RATE
    return af, tf


class _Stream(object):
    """A PLUGIN_FORCE_TYPE stream as a Series 'forcing' object, [N, ...]
    (cells last in the file become the first axis)."""

    def __init__(self, pf, transpose=True):
        self.pf, self.t = pf, transpose

    def __repr__(self):
        p = self.pf
        return 'PluginForcing(%s%s, %s, %s, %d steps, %s)' % (
            p.prefix, '<yyyy>.nc', p.varname, p.freq, p.nsteps,
            'cells first' if self.t else 'as read')

    def get(self, step):
        x = self.pf.get(step)
        return x.movedim(-1, 0) if self.t and x.dim() > 1 else x

    def reads(self, step):
        return self.pf.reads(step)


class _Sectors(object):
    """The water-use sector forcing [N, 5] (MUN ENE MAN LIV IRR) of one
    quantity; a sector without a stream is 0."""

    def __init__(self, streams, n):
        self.s, self.n = streams, n

    def __repr__(self):
        return 'Sectors(%s)' % ', '.join('%d: %r' % (j, s) for j, s in
                                         sorted(self.s.items()))

    def get(self, step):
        out = None
        for j, pf in self.s.items():
            x = pf.get(step).reshape(-1).to(torch.float64)
            if out is None:
                out = torch.zeros(self.n, WU.NSECT, dtype=torch.float64,
                                  device=x.device)
            out[:, j] = x
        if out is None:
            out = torch.zeros(self.n, WU.NSECT, dtype=torch.float64)
        return out

    def reads(self, step):
        return any(pf.reads(step) for pf in self.s.values())


def _gb(g, key):
    v = g.get(key)
    return bool(v) and v[-1][0].upper() == 'TRUE'


def _gv(g, key, default=None):
    v = g.get(key)
    return v[-1][0] if v else default


def nonrenew_limit(g):
    """NREN_LIM (mm) as VIC-WUR 3.0 reads it: a parameter of the CONSTANTS
    file only (get_parameters -> wu_get_parameters,
    wu_get_global_param.c:158-174); None (Inf, plugin_init_library.c:115)
    when the file does not set it. A NONRENEWABLE_LIMIT line in the global
    file is not a VIC-WUR 3.0 option: it warns "Unrecognized option in the
    global parameter file" and runs unlimited, so it is ignored here too,
    with the same warning."""
    if g.get('NONRENEWABLE_LIMIT'):
        warnings.warn('NONRENEWABLE_LIMIT in the global file is not read by '
                      'VIC-WUR (get_global_param.c of VIC-WUR 3.0: unrecognized option); '
                      'put it in the CONSTANTS file', stacklevel=2)
    lim = None
    cf = _gv(g, 'CONSTANTS')
    if cf and os.path.exists(cf):
        for ln in open(cf):
            w = ln.split('#')[0].split()
            if len(w) > 1 and w[0] == 'NONRENEWABLE_LIMIT':
                lim = float(w[1])
    return lim


def run_dates(g, nsteps=None):
    s = datetime.date(int(_gv(g, 'STARTYEAR')), int(_gv(g, 'STARTMONTH')),
                      int(_gv(g, 'STARTDAY')))
    e = datetime.date(int(_gv(g, 'ENDYEAR')), int(_gv(g, 'ENDMONTH')),
                      int(_gv(g, 'ENDDAY')))
    n = (e - s).days + 1 if nsteps is None else int(nsteps)
    return s, n


def modules(global_path, case, group='vicwur', nsteps=None, device='cpu',
            mode='fixed', window_days=31, prefetch=False, groundwater=None,
            cropsyst_irrigation=None, cropsyst_member=None):
    """{name: module block} for the switched-on plugins of `global_path`
    and for 'vic', the case `case` (load_case) was loaded from.

    `cropsyst_irrigation` (dict: crop_fn, code and the CsConfig options of
    torchvic.vicwur.cs_irrigation) switches the irrigation member to the
    CropSyst scheme; None keeps VIC-WUR's own.

    `cropsyst_member` (callable(case, start, nsteps) -> the option dict of the
    `vicwur_cropsyst` connector, without group) adds the CropSyst member as
    the group's crop (no WOFOST then: one crop role per group).

    GWM TRUE: routing, local water use and non-renewable run with `gwm`;
    `groundwater` (dict: `static` = the vicwur3 MODFLOW static values per
    cell, as tools/vicwur/gw_runs.py static_values; optional `foc`,
    `period`, `solver`, `well_layer`, `resolution`) adds the
    'groundwater' member (profile vicwur3, the VIC-WUR 3.0 MODFLOW
    coupling) and VIC's gwrecharge view. GWM without `groundwater` is
    refused: the GWM soil is asked for by the groundwater member."""
    g = P.parse_global(global_path)
    inp = P.load_vicwur_inputs(global_path, device='cpu')
    grid = inp.grid
    n = grid.n
    cells = np.flatnonzero(grid.active.reshape(-1))
    start, nsteps = run_dates(g, nsteps if nsteps is not None
                              else case.n_steps)
    st4 = (start.year, start.month, start.day, 0)
    met = [w for w in g.get('FORCE_TYPE', [])]
    met_start = F.file_info('%s%4d.nc' % (met[0][2], start.year),
                            met[0][1])['start']
    soil = soil_statics(_gv(g, 'PARAMETERS'), grid)
    af, tf = bands(_gv(g, 'PARAMETERS'), grid)
    lat = np.asarray(grid.lat, np.float64)
    lat_c = np.broadcast_to(lat[:, None], grid.active.shape)[grid.active] \
        if lat.ndim == 1 else lat[grid.active]

    def stream(key, transpose=True):
        for w in g.get('PLUGIN_FORCE_TYPE', []):
            if w[0].upper() == key:
                pf = F.PluginForcing(w[3], w[1], w[2], st4, nsteps, 1,
                                     device=device, cells=cells,
                                     met_start=met_start, prefetch=prefetch)
                return _Stream(pf, transpose)
        return None

    mods = {}
    G = dict(group=group)
    gwm = _gb(g, 'GWM')
    if gwm and groundwater is None:
        raise ValueError('vicwur_setup: GWM TRUE needs groundwater=dict('
                         'static=...) (the VIC-WUR 3.0 MODFLOW static values)')
    if inp.routing is not None:
        # under GWM the DISCHARGE forcing is the groundwater member's
        # baseflow (VIC-WUR 3.0's mf_run writes it for the next month)
        dis = None if gwm else stream('DISCHARGE', False)
        mods['vicwur_routing'] = dict(
            params=inp.routing, area=grid.area_active,
            rout_steps_per_day=int(_gv(g, 'ROUT_STEPS_PER_DAY')),
            force=None if dis is None else {'forcing': dis},
            nonrenew_runoff=_gb(g, 'NONRENEWABLE_RUNOFF'), gwm=gwm, **G)
    if inp.dams is not None:
        mods['vicwur_dams'] = dict(params=inp.dams,
                                   start_date=start.isoformat(),
                                   nsteps=nsteps, exact=(mode == 'exact'), **G)
    irr_on = inp.irrigation is not None
    if _gb(g, 'WATERUSE'):
        q = {}
        for nm in ('DEMAND', 'GROUNDWATER', 'CONSUMPTION'):
            q[nm] = _Sectors({j: stream('%s_%s' % (s, nm), False)
                              for j, s in enumerate(SECTORS)
                              if stream('%s_%s' % (s, nm), False)
                              is not None}, n)
        act = [bool(q['DEMAND'].s.get(j)) for j in range(4)] + [irr_on]
        if _gb(g, 'LOCAL_WITHDRAWAL'):
            mods['vicwur_wateruse_local'] = dict(
                demand={'forcing': q['DEMAND']},
                gw_frac={'forcing': q['GROUNDWATER']},
                cons_frac={'forcing': q['CONSUMPTION']}, active=act,
                soil=dict(resid_mm=soil['resid'][:, -1] * soil['depth'][:, -1]
                          * 1000.), gwm=gwm, **G)
        if _gb(g, 'REMOTE_WITHDRAWAL'):
            mods['vicwur_wateruse_remote'] = dict(params=inp.remote, **G)
        if _gb(g, 'NONRENEWABLE_WITHDRAWAL'):
            mods['vicwur_nonrenew'] = dict(limit=nonrenew_limit(g), gwm=gwm,
                                           **G)
    if _gb(g, 'EFR'):
        mods['vicwur_efr'] = dict(
            discharge={'forcing': stream('EFR_DISCHARGE', False)},
            baseflow={'forcing': stream('EFR_BASEFLOW', False)},
            soil=dict(resid=soil['resid'][:, -1], depth=soil['depth'][:, -1],
                      max_moist=soil['max_moist'][:, -1],
                      dsmax=soil['dsmax'], ds=soil['ds'], ws=soil['ws'],
                      c=soil['c']), exact=(mode == 'exact'), **G)
    # the case's tile tables on cells (first column of each cell)
    ex = case.snowbands
    first = (np.arange(case.mosaic.cv.shape[0]) if ex is None else
             np.array([int(np.flatnonzero(np.asarray(ex['cell']) == c)[0])
                       for c in range(n)]))
    root = case.mosaic.root_frac.detach().cpu().numpy()[first]   # [N, T, L]
    if irr_on:
        ir = inp.irrigation
        tix = ir.veg_class.numpy() - 1
        mods['vicwur_irrigation'] = dict(
            params=ir, soil=dict(max_moist=soil['max_moist'],
                                 wcr=soil['wcr'], wfc=soil['wfc'],
                                 wpwp=soil['wpwp'], root=root[:, tix]),
            efficient=_gb(g, 'EFFICIENT_IRRIGATION'),
            potential=_gb(g, 'POTENTIAL_IRRIGATION'),
            flag_hold=(mode != 'exact'),
            share_layer_moist=_gb(g, 'SHARE_LAYER_MOIST'), **G)
        if cropsyst_irrigation is not None:
            mods['vicwur_irrigation'].update(
                scheme='cropsyst',
                cropsyst=dict(depth=soil['depth'], **cropsyst_irrigation))
    if stream('CO2') is not None:
        mods['vicwur_co2'] = dict(co2={'forcing': stream('CO2', False)},
                                  b_co2=inp.b_co2, n_cells=n, **G)
    if _gb(g, 'LANDUSE_FORCE'):
        pb = _gv(g, 'PBARE')
        mods['vicwur_landuse'] = dict(
            cv={'forcing': stream('CV', True)},
            soil=dict(max_moist=soil['max_moist'], lat=lat_c,
                      thermal=thermal_statics(_gv(g, 'PARAMETERS'), grid)),
            pbare=None if pb is None else int(pb), mode=mode, **G)
    if inp.crop is not None:
        mods['vicwur_crop'] = crop_block(g, inp, grid, af, tf, lat_c, start,
                                         nsteps, soil=soil, **G)
        # exact: the leaf-class sums in the C order (the float C's own
        # rounding, docs/connectors/vicwur.md)
        mods['vicwur_crop']['exact_sums'] = (mode == 'exact')
    if cropsyst_member is not None:
        if 'vicwur_crop' in mods:
            raise ValueError('vicwur_setup: cropsyst_member and WOFOST are '
                             'both crop roles of the group; switch one off')
        mods['vicwur_cropsyst'] = dict(cropsyst_member(case, start, nsteps),
                                       **G)
    if gwm:
        la, lo = np.meshgrid(np.asarray(grid.lat, np.float64),
                             np.asarray(grid.lon, np.float64), indexing='ij')
        gw = dict(groundwater)
        mods['groundwater'] = dict(
            profile='vicwur3', domain=dict(lat=grid.take(la),
                                           lon=grid.take(lo),
                                           area=grid.area_active),
            start_date=start.isoformat(), mode=mode, **gw, **G)
    # the VIC-WUR 3.0 throughfall leak (exact)
    for k_ in mods:
        if k_ not in ('vicwur_crop', 'vicwur_cropsyst'):
            mods[k_]['throughfall'] = 'exact' if mode == 'exact' else 'fixed'
            break
    gf, core, steps = case.vicwur_image_global
    out = dict(vic=dict(connector='vic', global_param=gf, core=core,
                        steps=steps, window_days=window_days,
                        day_coupled=sorted(mods), phase_group=group,
                        soil_add_floor='zero' if mode == 'exact' else 'liquid',
                        soil_add_bands=ex is not None))
    if gwm:
        # capillary rise can leave a tile above max_moist, as in VIC-WUR 3.0
        out['vic']['soil_add_mode'] = 'over'
    if not mods:                  # plugins off: VIC alone, no group
        del out['vic']['phase_group']
        del out['vic']['day_coupled']
    if gwm:
        out['vic']['gwrecharge_view'] = True
    btf = crop_tfactor(global_path)
    if btf is not None:
        out['vic']['band_tfactor'] = btf
    for k, v in mods.items():
        out[k] = dict(connector=k, **v)
    return out


def crop_block(g, inp, grid, af, tf, lat, start, nsteps, soil=None,
               group='vicwur', feedback=True):
    """The vicwur_crop block (crop_init / wofost_init as setup_from_vicwur
    reads them; the crop nc Tfactor on band 0, crop_init.c:157-163); with
    `soil` (soil_statics) and `feedback` the soil_con values
    crop_update_step_vars reads."""
    from ...crop.wofost_npk.params import CropTemplates, setup_from_vicwur
    opts = {k: _gb(g, 'WOFOST_' + v) for k, v in (
        ('PIRR', 'POTENTIAL_IRRIGATION'), ('PFERT', 'POTENTIAL_FERTILIZER'),
        ('PTEMP', 'POTENTIAL_TEMPERATURE'), ('DIST_TSUM', 'DIST_TSUM'),
        ('DIST_FERT', 'DIST_FERTILIZER'), ('CALC_MIN', 'CALC_MINERALIZATION'),
        ('CONTINUE', 'CONTINUE'), ('FORCE_TSUM', 'FORCE_TSUM'),
        ('FORCE_FERT', 'FORCE_FERTILIZER'))}
    crop = inp.crop
    tf = tf.copy()
    tf[:, 0] = crop.Tfactor.numpy()
    tmpl = CropTemplates.from_wofost_text(inp.wofost)
    cv = P.read_cv(_gv(g, 'PARAMETERS'), grid)                  # [nveg, N]
    vcls = crop.crop_veg_class.numpy()                           # [NC, N]
    tile = np.where(vcls >= 0, vcls, 0).T                        # [N, NC]
    cvc = np.take_along_axis(cv.T, tile, 1) * (vcls.T >= 0)
    setup = setup_from_vicwur(
        tmpl, crop, lat, af, tf, cvc,
        dist_season=_gb(g, 'WOFOST_DIST_SEASON'),
        dist_tsum=opts['DIST_TSUM'], dist_fert=opts['DIST_FERT'],
        dist_min=_gb(g, 'WOFOST_DIST_MINERALIZATION'),
        calc_min=opts['CALC_MIN'], continue_=opts['CONTINUE'])
    dates = []
    for k in range(nsteps):
        d = start + datetime.timedelta(days=k)
        dates.append((d.year, d.month, d.day, d.timetuple().tm_yday))
    present = np.zeros((tile.shape[0], cv.shape[0] + 1), bool)
    present[:, :cv.shape[0]] = cv.T > 0                         # vidx
    blk = dict(tmpl=tmpl, setup=setup, wofost_options=opts, dates=dates,
               tile=tile, veg_map=vcls.T.copy(), tile_present=present,
               dist_season=_gb(g, 'WOFOST_DIST_SEASON'),
               calendar=_gv(g, 'CALENDAR', 'standard'), group=group)
    if soil is not None and feedback:
        blk['soil'] = dict(depth0=soil['depth'][:, 0], wpwp=soil['wpwp'],
                           wfc=soil['wfc'], wcr=soil['wcr'],
                           rough=soil['rough'])
    return blk
