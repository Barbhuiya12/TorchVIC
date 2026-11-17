#!/usr/bin/env python
"""MODFLOW 6 reference of the VIC-WUR 3.0 groundwater coupling.

Runs VIC-WUR 3.0's OWN Python driver code,
`plugins/modflow/config_module.py` (the package set) and `mf_run.py`
(`mfrun.run_modflow`, `PostProcessMF`), with the MODFLOW 6 6.7.0 executable,
on the grids of a TorchVIC test domain. Only the file plumbing around it is
replaced:

  * the static grids come from an .npz (written by tools/vicwur/gw_runs.py)
    instead of the global 5-arcmin netCDF files; `osgeo`, `xarray`,
    `netCDF4` are stubs (the validation venv has flopy but not those, and
    the code paths used here only read arrays);
  * the head carried from one period to the next is the previous period's
    MF6 head in float64 (VIC-WUR 3.0 goes through a float32 netCDF file,
    mf_run.py:224-247; plumbing, not ported);
  * the input files are written with 17 significant digits (flopy's default
    is 9);
  * IMS closes tightly (outer/inner dvclose 1e-10, rclose 1e-8; VIC-WUR 3.0
    uses 5 m and 10 m3/d, mf_run.py:28-32, a convergence setting, not the
    exchange).

Mode `fixed` (docs/connectors/vicwur.md; torchvic.vicwur.groundwater)
applies two textual edits to VIC-WUR 3.0's own functions: get_rch_param
without the clip of negative recharge (config_module.py:298) and
get_riv_param with the Manning depth parenthesised as its comment says
(:333); and adds a WEL package (wells of the period, m3/d, in `well_layer`).

  python mf_w3.py run IN.npz OUT.npz      (validation venv: flopy + mf6)

IN.npz: the static grids (VIC-WUR 3.0 names, (R, C) or (2, R, C)), `periods`
[P, 4] = (year, month, day, length in days), `gwrecharge_mm` [P, R, C] (the
period's OUT_GWRECHARGE sum), `discharge` [P, R, C] (the period's mean
OUT_DISCHARGE, m3 s-1), optional `wells` [P, R, C] (m3/d, < 0 = out),
`head0` [2, R, C] (initial head; else the static initial head), `mode`,
`foc`, `well_layer`, `xres`, `yres` (degrees), `ws` (work dir).
OUT.npz per period: `head` [P, 2, R, C]; flows per package and node
`q_RIV`, `q_DRN`, `q_EVT`, `q_CHD`, `q_RCH`, `q_WEL`, `q_STO-SS`
[P, 2, R, C] (m3/d, + into the aquifer); `w3_baseflow` and `w3_cpr`
[P, R, C], VIC-WUR 3.0's own PostProcessMF results (with its 1/days factor
and int truncation, to show them).
"""
import calendar
import datetime
import inspect
import os
import sys
import textwrap
import types
from unittest import mock

import numpy as np

W3MF = (os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'build/vicwur3/src_tree/vic/plugins/modflow'))
MF6 = (os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'verification/design_gwcrop/mf6bin/mf6'))


class _Var(np.ndarray):
    """A netCDF variable stand-in: an array that takes attributes."""


class _Dataset(object):
    """netCDF4.Dataset stand-in for the files mf_run writes (the gwl head
    file, mf_run.py:228-254): keeps nothing."""

    def __init__(self, *a, **k):
        self.dims = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def createDimension(self, name, n):
        self.dims[name] = n

    def createVariable(self, name, dtype, dims, **k):
        return np.zeros([self.dims[d] or 1 for d in dims]).view(_Var)


def _stubs():
    """osgeo, xarray, netCDF4 as stubs: VIC-WUR 3.0's modules import them at
    the top and config_module builds four configs from the global files at
    import time (config_module.py:447-461)."""
    dummy = types.SimpleNamespace(values=np.zeros((2, 2)))
    gt = types.SimpleNamespace(GetGeoTransform=lambda: (0, 1, 0, 0, 0, -1))
    osgeo = types.ModuleType('osgeo')
    osgeo.gdal = types.SimpleNamespace(Open=lambda *a, **k: gt)
    xr = types.ModuleType('xarray')
    xr.DataArray = object
    xr.open_dataarray = lambda *a, **k: dummy
    xr.open_dataset = lambda *a, **k: mock.MagicMock()
    ncm = types.ModuleType('netCDF4')
    ncm.Dataset = _Dataset
    ncm.date2num = lambda *a, **k: 0.0
    for name, m in (('osgeo', osgeo), ('xarray', xr), ('netCDF4', ncm)):
        sys.modules[name] = m
    # outside the validation venv (the TorchVIC tests import config_module
    # and vic_runner only): flopy and matplotlib as empty modules
    for name in ('flopy', 'matplotlib'):
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules['matplotlib'], 'pyplot'):
        try:
            import matplotlib.pyplot                     # noqa: F401
        except ImportError:
            sys.modules['matplotlib'].pyplot = types.ModuleType('pyplot')
            sys.modules['matplotlib.pyplot'] = sys.modules[
                'matplotlib'].pyplot
    sys.path.insert(0, W3MF)
    import config_module as cm
    import mf_run as mr
    return cm, mr


def _fix(cm, mode):
    """Mode fixed: VIC-WUR 3.0's own functions with the two documented edits."""
    if mode != 'fixed':
        return
    for fn, old, new in (
            ('get_rch_param',
             'recharge_inp = np.where(recharge_inp <0, 0, recharge_inp)',
             'pass'),
            ('get_riv_param',
             'riv_depth_avg1 = (riv_manning * self.ts_discharge) / '
             '(riv_width * riv_slope_used ** 0.5) ** (3.0 / 5.0)',
             'riv_depth_avg1 = ((riv_manning * self.ts_discharge) / '
             '(riv_width * riv_slope_used ** 0.5)) ** (3.0 / 5.0)')):
        src = textwrap.dedent(inspect.getsource(getattr(cm.config, fn)))
        assert old in src, (fn, old)
        ns = dict(np=cm.np)
        exec(src.replace(old, new), ns)
        setattr(cm.config, fn, ns[fn])


def _config(cm, z, ws):
    """A VIC-WUR 3.0 `config` (config_module.py:121-150) over the arrays of z."""
    R, C = z['landmask'].shape
    xres, yres = float(z['xres']), float(z['yres'])
    paths = types.SimpleNamespace(
        dem_ini=z['dem'], aqdepth_ini=z['aqdepth'], ksat_log=z['ksat_log'],
        cellarea=z['cellarea'], qbank=z['qbank'], riv_slope1=z['riv_slope'],
        Z0_floodplain=z['z0'], min_dem=z['min_dem'], KQ3=z['kq3'],
        conflayers=None, ksat_l1_conf_log=z['ksat_l1_conf_log'],
        ksat_l2_conf_log=z['ksat_l2_conf_log'], spe_yi_inp=z['sy'],
        landmask=z['landmask'], initialhead=z['head0'],
        ibound=z['ibound'], ldd=z['ldd'],
        capillary={k: types.SimpleNamespace(values=z[k]) for k in
                   ('cpsurface', 'cprate', 'cpdistinct')},
        clonemap=types.SimpleNamespace(
            GetGeoTransform=lambda: (0.0, xres, 0.0, 0.0, 0.0, -yres)),
        mf6exe=MF6, modflow_result_dir=os.path.join(ws, 'result'),
        get_modflow_workspace_dir=lambda name: os.path.join(ws, name),
        get_gwl_dir=lambda *a: os.path.join(ws, 'gwl'))
    c = cm.config.__new__(cm.config)
    c.paths = paths
    c.startstamp = datetime.datetime(1979, 1, 1)
    c.ts_gwrecharge = np.zeros((R, C))
    c.ts_discharge = np.zeros((R, C))
    c.humanimpact = False
    c.foc = None
    c.vic_out_suffix = 'tv'
    c.dem = np.where(np.isnan(paths.landmask), 0, paths.dem_ini)
    c.missingvalue = paths.aqdepth_ini[0][0]
    c.idomain = paths.ibound
    c.ibound = paths.ibound
    c.Nlay = 2
    c.Nrow, c.Ncol = paths.landmask.shape
    c.delrow = paths.clonemap.GetGeoTransform()[1] * 111 * 1000
    c.delcol = abs(paths.clonemap.GetGeoTransform()[5] * 111 * 1000)
    c.stress_period = -1
    c.set_foc(bool(z['foc']))
    return c


def _shim_flopy(mr, wells, layer):
    """flopy as mf_run sees it, plus a WEL package created with the DRN
    package when `wells` (a dict holding the period's rates) is set."""
    import flopy
    real = flopy.mf6.ModflowGwfdrn

    def drn(gwf, **kw):
        p = real(gwf, **kw)
        q = wells.get('q')
        if q is not None:
            R, C = q.shape
            spd = [[layer, i, j, float(q[i, j])] for i in range(R)
                   for j in range(C) if q[i, j] != 0]
            if spd:
                flopy.mf6.ModflowGwfwel(gwf, stress_period_data=spd,
                                        save_flows=True)
        return p
    m6 = types.SimpleNamespace(**{k: getattr(flopy.mf6, k) for k in dir(
        flopy.mf6) if not k.startswith('__')})
    m6.ModflowGwfdrn = drn
    real_sim = flopy.mf6.MFSimulation

    def sim(*a, **k):
        # full double precision in the written input files (flopy's
        # default is 9 significant digits: 1e-9 relative input noise)
        s = real_sim(*a, **k)
        s.simulation_data.float_precision = 17
        s.simulation_data.float_characters = 26
        return s
    m6.MFSimulation = sim
    m6.mfgwfoc = flopy.mf6.mfgwfoc
    mr.flopy = types.SimpleNamespace(mf6=m6, utils=flopy.utils)


def run(inp, out):
    import flopy
    z = dict(np.load(inp, allow_pickle=True))
    mode = str(z.get('mode', 'exact'))
    ws = str(z['ws'])
    os.makedirs(ws, exist_ok=True)
    cm, mr = _stubs()
    _fix(cm, mode)
    cfg = _config(cm, z, ws)
    wells = {}
    _shim_flopy(mr, wells, int(z.get('well_layer', 1)))
    R, C = z['landmask'].shape
    per = z['periods'].astype(int)
    head = (np.asarray(z['head0'], float) if 'head0_run' not in z
            else np.asarray(z['head0_run'], float))
    head = [np.where(np.isnan(head[0]), 0, head[0]),
            np.where(np.isnan(head[1]), 0, head[1])]
    keys = ('RIV', 'DRN', 'EVT', 'CHD', 'RCH', 'WEL', 'STO-SS')
    res = {'head': [], 'w3_baseflow': [], 'w3_cpr': []}
    res.update({'q_' + k: [] for k in keys})
    for p, (y, mo, d, ln) in enumerate(per.tolist()):
        date = datetime.datetime(y, mo, d)
        # vic_runner.PostProcessVIC: the month's sum / days / 1000 (m/d)
        gw = types.SimpleNamespace(
            values=(z['gwrecharge_mm'][p] / ln / 1000)[None])
        q = types.SimpleNamespace(values=z['discharge'][p][None])
        wells['q'] = z['wells'][p] if 'wells' in z else None
        with mock.patch.object(mr.mfrun, 'get_startinghead',
                               lambda self, h=head: list(h)), \
                mock.patch.object(
                    mr.mfrun, 'online_cp_stress_period_data',
                    lambda self, ln=ln: [int(ln), 1, 1]):
            m = mr.mfrun(cfg, date, gw, q, 0)
            m.rclose, m.maxinner, m.maxouter = 1e-8, 2000, 500
            m.outer_dvclose = m.inner_dvclose = 1e-10
            m.run_modflow()
        end = m.end_date                         # mf_run.py:24, the names
        wsd = os.path.join(ws, cfg.mfname)
        hf = flopy.utils.HeadFile(os.path.join(
            wsd, '%s_%s.hds' % (cfg.mfname, end.strftime('%Y%m%d'))),
            precision='double')
        h = np.asarray(hf.get_data(), float)
        head = [h[0].copy(), h[1].copy()]
        res['head'].append(h)
        cb = flopy.utils.CellBudgetFile(os.path.join(
            wsd, '%s_%s.cbb' % (cfg.mfname, end.strftime('%Y%m%d'))),
            precision='double')
        names = [x.decode().strip() for x in cb.get_unique_record_names()]
        for k in keys:
            g = np.zeros(2 * R * C)
            if k in names:
                a = cb.get_data(text=k)[0]
                if a.dtype.names:
                    np.add.at(g, a['node'] - 1, a['q'])
                else:
                    g = np.ravel(a).astype(float)
            res['q_' + k].append(g.reshape(2, R, C))
        pp = mr.PostProcessMF(cfg, date)
        res['w3_baseflow'].append(pp.get_baseflow_array())
        res['w3_cpr'].append(pp.get_cpr_array() if cfg.foc
                             else np.zeros((R, C)))
    np.savez(out, **{k: np.asarray(v) for k, v in res.items()})


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != 'run':
        sys.exit(__doc__)
    run(sys.argv[2], sys.argv[3])
