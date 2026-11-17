"""VIC-WUR plugin parameter readers.

Every plugin input file that the R1 reference domain uses is read once, at
setup, into device tensors over the active cells (row-major over the grid, as
VIC's io_idx).  The readers follow the C code of VIC-WUR 3.0, not its
documentation:

  routing      rout_start.c:47-62, rout_init.c:36-110 and 148-181
  dams         dam_start.c, dam_alloc_free.c, dam_init.c:40-290
  remote       wu_alloc_free.c:70-96, wu_init.c:60-122
  irrigation   irr_alloc_free.c:36-70, irr_start.c:36-60, irr_init.c:60-178
  crop nc      crop_init.c:25-163, wofost_init.c:130-498 (flags select variables)
  wofost text  wofost_init.c:25-90, GetSimInput.c, Cropdata.c, Management.c,
               wofost_const_library.c:8-135 (77 scalars, 22 tables, read as
               float32 as in C)
  b_co2        the veg_class x lat x lon field of the parameter file

All netCDF data are read raw (no mask and scale), as nc_get_vara_* does.  A
variable that is in a file but not in the reader's schema is an error
(UnknownVariableError); so is a missing variable, a grid that differs from the
domain, or a value the C code stops on.  Conditions on which C only warns
(unit hydrograph sums, a cell without a downstream cell, receiving or service
counts that do not match, dams referenced by services that do not exist) are
reported as warnings and recorded in `.warnings`.

Forcing files (yearly demand, EFR, Cv, CO2 files) are read by
torchvic.vicwur.forcing.
"""
import re
import warnings

import numpy as np

import torch

#: forcing files and wofost 'crop nc' use these dimensions
MAX_UPSTREAM = 8
NODATA = -1


class UnknownVariableError(ValueError):
    """A plugin file holds a variable the reader does not know."""


class VicWurInputError(ValueError):
    """A plugin input does what VIC-WUR stops on."""


# ----------------------------------------------------------------- helpers
def _nc():
    import netCDF4
    return netCDF4


def _open(path):
    ds = _nc().Dataset(path)
    ds.set_auto_maskandscale(False)
    return ds


def _warn(store, msg):
    store.append(msg)
    warnings.warn(msg, stacklevel=3)


class Grid(object):
    """Active cells of the domain: `run_cell` of the parameter file bounded
    by the domain mask, row-major (torchvic.io.domain.active_cells)."""

    def __init__(self, domain_path, params_path):
        from ..io.domain import active_cells
        with _open(domain_path) as d:
            self.lat = np.array(d['lat'][:])
            self.lon = np.array(d['lon'][:])
            mask = np.array(d['mask'][:])
            self.area = np.array(d['area'][:], dtype=np.float64)
        with _open(params_path) as p:
            run = np.array(p['run_cell'][:])
        self.ny, self.nx = mask.shape
        self.active = active_cells(run, mask)
        self.n = int(self.active.sum())
        self.area_active = self.area[self.active]

    def take(self, a):
        """[..., ny, nx] -> [..., n]."""
        a = np.asarray(a)
        if a.shape[-2:] != (self.ny, self.nx):
            raise VicWurInputError('field has grid %s, domain has %s'
                                   % (a.shape[-2:], (self.ny, self.nx)))
        return a[..., self.active]

    def check_file(self, ds, path):
        """compare_ncdomain_with_global_domain: same lat and lon."""
        for nm, ref in (('lat', self.lat), ('lon', self.lon)):
            if nm not in ds.variables:
                raise VicWurInputError('%s: no %s' % (path, nm))
            v = np.array(ds[nm][:])
            if v.shape != ref.shape or not np.allclose(v, ref, 0, 1e-6):
                raise VicWurInputError('%s: %s differs from the domain'
                                       % (path, nm))


def _check_vars(ds, allowed, path, required=()):
    names = set(ds.variables)
    unknown = sorted(names - set(allowed))
    if unknown:
        raise UnknownVariableError('%s: unknown variable(s) %s'
                                   % (path, ', '.join(unknown)))
    missing = sorted(set(required) - names)
    if missing:
        raise VicWurInputError('%s: missing variable(s) %s'
                               % (path, ', '.join(missing)))


def _t(a, device, dtype=None):
    t = torch.from_numpy(np.array(a))     # a writable copy
    if dtype is not None and t.is_floating_point():
        t = t.to(dtype)
    return t.to(device)


class _Base(object):
    def tensors(self):
        return {k: v for k, v in vars(self).items()
                if isinstance(v, torch.Tensor)}

    def __repr__(self):
        """Class and a digest of the contents (a coupling configuration
        that holds these is checkpointed under a digest of its repr)."""
        import hashlib
        h = hashlib.sha256()
        for k in sorted(vars(self)):
            v = vars(self)[k]
            h.update(k.encode())
            if isinstance(v, torch.Tensor):
                h.update(str((v.dtype, tuple(v.shape))).encode())
                h.update(v.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(v, _Base):
                h.update(repr(v).encode())
            else:
                h.update(repr(v).encode())
        return '<%s %s>' % (type(self).__name__, h.hexdigest()[:16])


# ----------------------------------------------------------------- routing
ROUT_VARS = ('lat', 'lon', 'time', 'downstream_id', 'downstream',
             'uh_inflow', 'uh_runoff')


class RoutingParams(_Base):
    """downstream[N] (own index for an outlet), outlet[N], uh_inflow and
    uh_runoff [UH_LENGTH, N] renormalised by a sequential sum as in C,
    upstream[N, MAX_UPSTREAM] (-1 padded) and n_upstream[N]."""

    def __init__(self):
        self.warnings = []


def read_routing(path, grid, rout_steps_per_day, device='cpu',
                 dtype=torch.float64):
    out = RoutingParams()
    with _open(path) as ds:
        _check_vars(ds, ROUT_VARS, path, ROUT_VARS)
        grid.check_file(ds, path)
        sec = np.array(ds['time'][:]).astype(np.int64)
        if 86400 % rout_steps_per_day:
            raise VicWurInputError('ROUT_STEPS_PER_DAY must divide 86400')
        dt = 86400 // rout_steps_per_day
        if not np.array_equal(sec, np.arange(len(sec)) * dt):
            raise VicWurInputError('ROUT_STEPS_PER_DAY and time in routing '
                                   'parameter file do not match')
        out.uh_length = len(sec)
        out.rout_dt = dt
        cid = grid.take(np.array(ds['downstream_id'][:])).astype(np.int64)
        down = grid.take(np.array(ds['downstream'][:])).astype(np.int64)
        uhs = {}
        for nm in ('uh_inflow', 'uh_runoff'):
            u = grid.take(np.array(ds[nm][:], dtype=np.float64))
            s = np.zeros(grid.n)
            for j in range(u.shape[0]):          # sequential sum, as in C
                s = s + u[j]
            if not np.allclose(s, 1.0, 0, 1e-8):
                _warn(out.warnings, 'sum of %s != 1 in %d cell(s); '
                      'adjusting fractions' % (nm, int((~np.isclose(
                          s, 1.0, 0, 1e-8)).sum())))
            uhs[nm] = u / s
    n = grid.n
    ds_idx = np.arange(n)
    found = np.zeros(n, bool)
    for i in range(n):
        j = np.nonzero(cid == down[i])[0]
        if j.size:                               # no break: the last wins
            ds_idx[i] = j[-1]
            found[i] = True
    if (~found).any():
        _warn(out.warnings, 'no downstream cell found for %d cell(s); '
              'setting them as outflow points' % int((~found).sum()))
    ups = [[] for _ in range(n)]
    for i in range(n):
        if ds_idx[i] != i:
            ups[ds_idx[i]].append(i)
    up = -np.ones((n, MAX_UPSTREAM), np.int64)
    for j, u in enumerate(ups):
        if len(u) > MAX_UPSTREAM:
            raise VicWurInputError('cell %d has %d upstream cells (max %d)'
                                   % (j, len(u), MAX_UPSTREAM))
        up[j, :len(u)] = u
    out.downstream = _t(ds_idx, device)
    out.outlet = _t(ds_idx == np.arange(n), device)
    out.upstream = _t(up, device)
    out.n_upstream = _t(np.array([len(u) for u in ups], np.int64), device)
    out.uh_inflow = _t(uhs['uh_inflow'], device, dtype)
    out.uh_runoff = _t(uhs['uh_runoff'], device, dtype)
    return out


# -------------------------------------------------------------------- dams
DAM_VARS = ('lat', 'lon', 'dam_class', 'id_map', 'id', 'type', 'year',
            'capacity', 'inflow_fraction', 'Nservice', 'service',
            'service_fraction')


class DamParams(_Base):
    """One entry per active dam, ordered by cell then class: dam_cell,
    dam_class, is_global (type 1), year, capacity, inflow_fraction,
    nservice, service[ND, maxsvc] (cell index, -1 padded) and
    service_frac.  Several cells may share one dam id (C matches
    id == id_map for every cell)."""

    def __init__(self):
        self.warnings = []


def read_dams(path, grid, device='cpu', dtype=torch.float64):
    out = DamParams()
    with _open(path) as ds:
        _check_vars(ds, DAM_VARS, path, DAM_VARS)
        grid.check_file(ds, path)
        nd = ds.dimensions['dam_class'].size
        idm = grid.take(np.array(ds['id_map'][:])).astype(np.int64)
        did = np.array(ds['id'][:]).astype(np.int64)
        typ = np.array(ds['type'][:]).astype(np.int64)
        year = np.array(ds['year'][:]).astype(np.int64)
        cap = np.array(ds['capacity'][:], dtype=np.float64)
        infr = np.array(ds['inflow_fraction'][:], dtype=np.float64)
        nserv = np.array(ds['Nservice'][:]).astype(np.int64)
        svc = grid.take(np.array(ds['service'][:])).astype(np.int64)
        sfr = grid.take(np.array(ds['service_fraction'][:],
                                 dtype=np.float64))
    if not np.isin(typ, (0, 1)).all():
        raise VicWurInputError('%s: unknown dam type (0 local, 1 global)'
                               % path)
    n = grid.n
    didx = -np.ones((n, nd), np.int64)
    nact = np.zeros(n, np.int64)
    for j in range(nd):
        for i in range(n):
            if did[j] == idm[i]:
                didx[i, j] = nact[i]
                nact[i] += 1
    # service mapping: dam_init.c:70-170; one counter per class
    nsv_tmp = np.zeros(nd, np.int64)
    lists = {}
    errors = 0
    for j in range(nd):
        for i in range(n):
            if svc[j, i] > 0:
                for k in range(n):
                    if svc[j, i] != idm[k]:
                        continue
                    if didx[k, j] == NODATA:
                        _warn(out.warnings, 'servicing cell %d references a '
                              'dam (%d) that does not exist' % (i, k))
                        continue
                    if nsv_tmp[j] >= nserv[j]:
                        raise VicWurInputError(
                            'number of servicing cells (%d) is larger than '
                            'specified in the parameter file (%d)'
                            % (nsv_tmp[j], nserv[j]))
                    lists.setdefault((k, j), []).append((i, sfr[j, i]))
                    nsv_tmp[j] += 1
    for j in range(nd):
        for i in range(n):
            if didx[i, j] != NODATA and nsv_tmp[j] != nserv[j]:
                errors += 1
    if errors:
        _warn(out.warnings, 'several servicing cells were not found for %d '
              'dam(s); removing servicing cells' % errors)
    dams = sorted((i, j) for i in range(n) for j in range(nd)
                  if didx[i, j] != NODATA)
    maxs = max([len(lists.get(d, [])) for d in dams] + [1])
    svm = -np.ones((len(dams), maxs), np.int64)
    sfm = np.full((len(dams), maxs), -1.0)
    for a, d in enumerate(dams):
        for b, (ci, fr) in enumerate(lists.get(d, [])):
            svm[a, b] = ci
            sfm[a, b] = fr
    cls = np.array([d[1] for d in dams], np.int64)
    out.n_dams = len(dams)
    out.dam_cell = _t(np.array([d[0] for d in dams], np.int64), device)
    out.dam_class = _t(cls, device)
    out.dam_id = _t(did[cls], device)
    out.is_global = _t(typ[cls] == 1, device)
    out.year = _t(year[cls], device)
    out.capacity = _t(cap[cls], device, dtype)
    out.inflow_fraction = _t(infr[cls], device, dtype)
    out.nservice = _t(np.array([len(lists.get(d, [])) for d in dams],
                               np.int64), device)
    out.service = _t(svm, device)
    out.service_frac = _t(sfm, device, dtype)
    return out


# ------------------------------------------------------------------ remote
WU_VARS = ('lat', 'lon', 'wu_receiving', 'receiving_id', 'Nreceiving',
           'receiving')


class RemoteParams(_Base):
    """nreceiving[N] and receiving[N, maxrec] (cell index of each receiving
    cell, -1 padded).  The value at a receiving cell is its SOURCE's
    receiving_id; Nreceiving is read at the source cells."""

    def __init__(self):
        self.warnings = []


def read_remote(path, grid, device='cpu'):
    out = RemoteParams()
    with _open(path) as ds:
        _check_vars(ds, WU_VARS, path, WU_VARS)
        grid.check_file(ds, path)
        nrk = ds.dimensions['wu_receiving'].size
        rid = grid.take(np.array(ds['receiving_id'][:])).astype(np.int64)
        nrec = grid.take(np.array(ds['Nreceiving'][:])).astype(np.int64)
        rec = grid.take(np.array(ds['receiving'][:])).astype(np.int64)
    n = grid.n
    cnt = np.zeros(n, np.int64)
    lists = [[] for _ in range(n)]
    for k in range(nrk):
        for i in range(n):
            if rec[k, i] > 0:
                for j in range(n):
                    if rec[k, i] != rid[j]:
                        continue
                    if cnt[j] >= nrec[j]:
                        raise VicWurInputError(
                            'number of receiving cells (%d) is larger than '
                            'specified in the parameter file (%d)'
                            % (cnt[j], nrec[j]))
                    lists[j].append(i)
                    cnt[j] += 1
                    break
    bad = int((cnt != nrec).sum())
    if bad:
        _warn(out.warnings, 'no receiving cells were found for %d cell(s); '
              'removing receiving cells' % bad)
    mr = max([len(x) for x in lists] + [1])
    r = -np.ones((n, mr), np.int64)
    for j, x in enumerate(lists):
        r[j, :len(x)] = x
    out.nreceiving = _t(cnt, device)
    out.receiving = _t(r, device)
    return out


# -------------------------------------------------------------- irrigation
IRR_VARS = ('lat', 'lon', 'irr_class', 'veg_class', 'paddy',
            'groundwater_fraction', 'irrigation_efficiency',
            'irrigation_offset')
PADDY_BARE_UNSET = 99999


class IrrigationParams(_Base):
    """Per cell and irrigation class (NI classes): active[N, NI] (the
    vegetation class has cover in the cell), veg_index[N, NI] (index among
    the cell's active vegetation classes, -1), gw_fraction (paddy: 0),
    efficiency (one per cell, copied to every class) and offset (days;
    zero unless OFFSET_IRRIGATION).  paddy[NI], veg_class[NI] (1-based) and
    pbare (None when PBARE is not set)."""

    def __init__(self):
        self.warnings = []


def read_irrigation(path, grid, cv, offset_irrigation=False, pbare=None,
                    nbare=1, device='cpu', dtype=torch.float64):
    """`cv` [nveg, N]: land cover of the parameter file (the vegetation tile
    exists where Cv > 0)."""
    out = IrrigationParams()
    nveg = cv.shape[0]
    if pbare is not None:
        if pbare == PADDY_BARE_UNSET:
            pbare = nveg
        if pbare > nveg or pbare <= nveg - nbare:
            raise VicWurInputError(
                'paddy bare type (%d) is not within bare types (%d - %d)'
                % (pbare, nveg - nbare + 1, nveg))
    with _open(path) as ds:
        _check_vars(ds, IRR_VARS, path, IRR_VARS[:-1])
        grid.check_file(ds, path)
        vc = np.array(ds['veg_class'][:]).astype(np.int64)
        pad = np.array(ds['paddy'][:]).astype(np.int64)
        gwf = grid.take(np.array(ds['groundwater_fraction'][:],
                                 dtype=np.float64))
        eff = grid.take(np.array(ds['irrigation_efficiency'][:],
                                 dtype=np.float64))
        if offset_irrigation:
            if 'irrigation_offset' not in ds.variables:
                raise VicWurInputError('%s: OFFSET_IRRIGATION needs '
                                       'irrigation_offset' % path)
            off = grid.take(np.array(ds['irrigation_offset'][:])
                            ).astype(np.int64)
        else:
            off = np.zeros(grid.n, np.int64)
    if ((vc < 1) | (vc > nveg)).any():
        raise VicWurInputError('%s: veg_class outside 1..%d' % (path, nveg))
    ni, n = len(vc), grid.n
    present = np.asarray(cv) > 0                         # [nveg, N]
    vidx = np.cumsum(present, axis=0) - 1                # index among active
    act = present[vc - 1, :].T                           # [N, NI]
    vix = np.where(act, vidx[vc - 1, :].T, NODATA)
    paddy = pad == 1
    g = np.where(paddy[None, :], 0.0, gwf[:, None])
    out.n_classes = ni
    out.pbare = pbare
    out.veg_class = _t(vc, device)
    out.paddy = _t(paddy, device)
    out.active = _t(act, device)
    out.veg_index = _t(vix.astype(np.int64), device)
    out.gw_fraction = _t(np.broadcast_to(g, (n, ni)), device, dtype)
    out.efficiency = _t(np.broadcast_to(eff[:, None], (n, ni)), device,
                        dtype)
    out.offset = _t(np.broadcast_to(off[:, None], (n, ni)), device)
    return out


# --------------------------------------------------------------- crop (nc)
CROP_VARS = ('lat', 'lon', 'crop_class', 'month', 'fertilizer_times',
             'Ncrop', 'Cc', 'crop_veg_class', 'plant_day', 'harvest_day',
             'TSUM1', 'TSUM2', 'Tfactor', 'DVS_point', 'N_amount',
             'P_amount', 'K_amount', 'N_mins', 'P_mins', 'K_mins',
             'N_recovery', 'P_recovery', 'K_recovery',
             'mineralization_period', 'carbon', 'pH')
CROP_ALWAYS = ('Ncrop', 'Cc', 'crop_veg_class', 'Tfactor')
CROP_SEASON = ('plant_day', 'harvest_day')
CROP_TSUM = ('TSUM1', 'TSUM2')
CROP_FERT = ('DVS_point', 'N_amount', 'P_amount', 'K_amount')
CROP_MIN_CALC = ('carbon', 'pH', 'mineralization_period', 'N_mins',
                 'N_recovery', 'P_mins', 'P_recovery', 'K_mins',
                 'K_recovery')


class CropParams(_Base):
    """Crop nc: Cc[12, NC, N], crop_veg_class[NC, N] (0-based veg class,
    -1 none), ncrop[N], cidx[N, NC] (crop tile index, -1 for a crop with
    zero cover), Tfactor[N] and the optional fields named in CROP_* that the
    WOFOST_DIST_* flags ask for, kept in `.fields` by name ([NC, N] or
    [NF, NC, N] or [N])."""

    def __init__(self):
        self.warnings = []
        self.fields = {}


def read_crop(path, grid, nveg, dist_season=False, dist_tsum=False,
              dist_fert=False, dist_min=False, calc_min=False,
              device='cpu', dtype=torch.float64):
    out = CropParams()
    want = list(CROP_ALWAYS)
    if dist_season:
        want += CROP_SEASON
    if dist_tsum:
        want += CROP_TSUM
    if dist_fert:
        want += CROP_FERT
    if dist_min and calc_min:
        want += CROP_MIN_CALC
    elif dist_min:
        want += [v for v in CROP_MIN_CALC if v not in ('carbon', 'pH')]
    with _open(path) as ds:
        _check_vars(ds, CROP_VARS, path, want)
        grid.check_file(ds, path)
        nc_ = ds.dimensions['crop_class'].size
        cvc = grid.take(np.array(ds['crop_veg_class'][:])).astype(np.int64)
        cc = grid.take(np.array(ds['Cc'][:], dtype=np.float64))
        ncrop = grid.take(np.array(ds['Ncrop'][:])).astype(np.int64)
        tf = grid.take(np.array(ds['Tfactor'][:], dtype=np.float64))
        for nm in want:
            if nm in CROP_ALWAYS:
                continue
            a = np.array(ds[nm][:])
            a = grid.take(a)
            out.fields[nm] = _t(a.astype(np.float64), device, dtype)
        out.n_fert_times = (ds.dimensions['fertilizer_times'].size
                            if 'fertilizer_times' in ds.dimensions else 0)
    if (cvc > nveg).any():
        raise VicWurInputError('%s: crop veg class larger than the number '
                               'of vegetation classes' % path)
    vegc = np.where(cvc > 0, cvc - 1, NODATA)
    n = grid.n
    csum = cc.sum(axis=0)                                # [NC, N]
    cidx = -np.ones((n, nc_), np.int64)
    for i in range(n):
        k = 0
        for j in range(nc_):
            if csum[j, i] > 0:
                if vegc[j, i] == NODATA:
                    raise VicWurInputError(
                        'crop %d has cover in cell %d but no vegetation '
                        'class' % (j, i))
                cidx[i, j] = k
                k += 1
        if k != ncrop[i]:
            raise VicWurInputError(
                'number of crop tiles with nonzero area (%d) != ncrop (%d) '
                'in cell %d' % (k, ncrop[i], i))
    out.n_crop_types = nc_
    out.Cc = _t(cc, device, dtype)
    out.crop_veg_class = _t(vegc, device)
    out.ncrop = _t(ncrop, device)
    out.cidx = _t(cidx, device)
    out.Tfactor = _t(tf, device, dtype)
    return out


# ------------------------------------------------------------- wofost text
CROP_SCALARS = (
    'TBASEM TEFFMX TSUMEM IDSL DLO DLC VERNSAT VERNBASE TSUM1 TSUM2 DVSI '
    'DVSEND TDWI RGRLAI SPA SPAN TBASE CVL CVO CVR CVS Q10 RML RMO RMR RMS '
    'PERDL CFET DEPNR IAIRDU RDI RRI RDMCR RDRLV_NPK DVS_NPK_STOP '
    'DVS_NPK_TRANSL NPK_TRANSLRT_FR NCRIT_FR PCRIT_FR KCRIT_FR NMAXRT_FR '
    'NMAXST_FR PMAXRT_FR PMAXST_FR KMAXRT_FR KMAXST_FR NLAI_NPK NLUE_NPK '
    'NMAXSO PMAXSO KMAXSO NPART NSLA_NPK NRESIDLV NRESIDST NRESIDRT '
    'PRESIDLV PRESIDST PRESIDRT KRESIDLV KRESIDST KRESIDRT TCNT TCPT TCKT '
    'NFIX_FR MaxHeight RGL RadAtten WindAtten TrunkRatio Albedo '
    'MinStomResist MaxArchResist Fcanopy FRTRL CRITLAI').split()
CROP_TABLES = (
    'VERNRTB DTSMTB SLATB SSATB KDIFTB EFFTB AMAXTB TMPFTB TMNFTB CO2AMAXTB '
    'CO2EFFTB CO2TRATB RFSETB FRTB FLTB FSTB FOTB RDRRTB RDRSTB NMAXLV_TB '
    'PMAXLV_TB KMAXLV_TB').split()
MNG_SCALARS = ('NRFTAB PRFTAB KRFTAB NMINS RTNMINS PMINS RTPMINS KMINS '
               'RTKMINS').split()
MNG_TABLES = ['FERNTAB', 'FERPTAB', 'FERKTAB', 'IRRTAB']
assert len(CROP_SCALARS) == 77 and len(CROP_TABLES) == 22
#: read only when IDSL >= 2 (Fillvar.c:19-30, Cropdata.c:114)
VERN_ONLY = ('VERNSAT', 'VERNBASE')


def _f32(s):
    return float(np.float32(float(s)))


def _scalar(text, name, path):
    """fscanf('%s') token search, skip to '=', read a float (as float32)."""
    m = re.search(r'(?<!\S)%s(?!\S)[^=]*=\s*([^\s]+)' % re.escape(name),
                  text)
    if m is None:
        return None
    try:
        return _f32(m.group(1))
    except ValueError:
        raise VicWurInputError('%s: bad value for %s: %r'
                               % (path, name, m.group(1)))


def parse_crop_file(path):
    """-> (scalars dict name->float32 value, tables dict name->(x, y))."""
    text = open(path).read()
    lines = text.splitlines()
    sc = {}
    for nm in CROP_SCALARS:
        v = _scalar(text, nm, path)
        if v is not None:
            sc[nm] = v
    idsl = int(sc.get('IDSL', 0))
    for nm in CROP_SCALARS:
        if nm not in sc:
            if nm in VERN_ONLY and idsl < 2:
                sc[nm] = -99.0
            else:
                raise VicWurInputError('%s: missing crop variable %s'
                                       % (path, nm))
    if idsl < 2:
        sc['VERNSAT'] = sc['VERNBASE'] = -99.0
    tb = {}
    for nm in CROP_TABLES:
        for a, ln in enumerate(lines):
            if not ln or ln[0] in '* \r\n':
                continue
            w = ln.split()
            if not w or w[0] != nm:
                continue
            if len(w) < 5:
                raise VicWurInputError('%s: bad first row of table %s'
                                       % (path, nm))
            xs, ys = [_f32(w[2])], [_f32(w[4])]
            for ln2 in lines[a + 1:]:
                w2 = ln2.split()
                if len(w2) < 3:
                    break
                try:
                    x, y = _f32(w2[0]), _f32(w2[2])
                except ValueError:
                    break
                xs.append(x)
                ys.append(y)
            tb[nm] = (xs, ys)
            break
    for nm in CROP_TABLES:
        if nm not in tb and not (nm == 'VERNRTB' and idsl < 2):
            raise VicWurInputError('%s: missing crop table %s' % (path, nm))
    return sc, tb


def parse_management_file(path):
    text = open(path).read()
    lines = text.splitlines()
    sc = {}
    for nm in MNG_SCALARS:
        v = _scalar(text, nm, path)
        if v is None:
            raise VicWurInputError('%s: missing management variable %s'
                                   % (path, nm))
        sc[nm] = v
    tb = {}
    for nm in MNG_TABLES:
        for a, ln in enumerate(lines):
            if not ln or ln[0] in '* \n':
                continue
            w = ln.split()
            if not w or w[0] != nm:
                continue
            if len(w) < 4:
                raise VicWurInputError('%s: bad first row of table %s'
                                       % (path, nm))
            rows = [w[2:4]]
            for ln2 in lines[a + 1:]:
                w2 = ln2.split()
                if len(w2) < 2:
                    break
                rows.append(w2[:2])
            mon, day, amt = [], [], []
            for d, v in rows:
                if len(d) >= 6:
                    raise VicWurInputError('%s: date %r too long' % (path, d))
                m_, d_ = d.split('-')
                mon.append(int(m_))
                day.append(int(d_))
                amt.append(_f32(v))
            tb[nm] = (mon, day, amt)
            break
        if nm not in tb:
            raise VicWurInputError('%s: missing management table %s'
                                   % (path, nm))
    return sc, tb


class WofostText(_Base):
    """One row per crop class of the wofost text file: scalars
    [NC, 77] (CROP_SCALARS order, float32 values held in float64), tables
    (name -> x[NC, P], y[NC, P], n[NC]), management scalars [NC, 9],
    management tables (name -> month, day, amount [NC, P], n[NC]),
    start_month/start_day[NC] and emergence[NC]."""

    def __init__(self):
        self.tables = {}
        self.mng_tables = {}
        self.crop_files = []


def read_wofost_text(path, n_crop_types, device='cpu', dtype=torch.float64):
    out = WofostText()
    rows = []
    with open(path) as f:
        for ln in f:
            if not ln or ln[0] in '* \n\r':
                continue
            rows.append(ln)
    if len(rows) > n_crop_types:
        raise VicWurInputError('crops in the wofost text file (%d) are more '
                               'than crops in the wofost netcdf file (%d)'
                               % (len(rows), n_crop_types))
    if len(rows) + 1 < n_crop_types:
        raise VicWurInputError('crops in the wofost text file are less than '
                               'crops in the wofost netcdf file')
    sc, mn = [], []
    ctab = {k: [] for k in CROP_TABLES}
    mtab = {k: [] for k in MNG_TABLES}
    sm, sd, em = [], [], []
    for ln in rows:
        w = ln.split()
        if len(w) < 5:
            raise VicWurInputError('%s: line needs path crop management '
                                   'start emergence: %r' % (path, ln))
        base, cf, mf, start, emer = w[:5]
        cs, ct = parse_crop_file(base + cf)
        ms, mt = parse_management_file(base + mf)
        out.crop_files.append((base + cf, base + mf))
        sc.append([cs[k] for k in CROP_SCALARS])
        mn.append([ms[k] for k in MNG_SCALARS])
        for k in CROP_TABLES:
            ctab[k].append(ct.get(k, ([], [])))
        for k in MNG_TABLES:
            mtab[k].append(mt[k])
        try:
            m_, d_ = start.split('-')
            sm.append(int(m_))
            sd.append(int(d_))
        except ValueError:
            raise VicWurInputError('%s: start %r is not %%m-%%d'
                                   % (path, start))
        em.append(int(emer))
    out.n_crops = len(rows)
    out.scalars = _t(np.array(sc, np.float64), device, dtype)
    out.scalar_names = CROP_SCALARS
    out.mng_scalars = _t(np.array(mn, np.float64), device, dtype)
    out.mng_scalar_names = MNG_SCALARS
    for k, v in ctab.items():
        p = max([len(x) for x, _ in v] + [1])
        X = np.full((len(v), p), np.nan)
        Y = np.full((len(v), p), np.nan)
        for a, (x, y) in enumerate(v):
            X[a, :len(x)] = x
            Y[a, :len(y)] = y
        out.tables[k] = (_t(X, device, dtype), _t(Y, device, dtype),
                         _t(np.array([len(x) for x, _ in v]), device))
    for k, v in mtab.items():
        p = max([len(r[0]) for r in v] + [1])
        M = np.zeros((len(v), p), np.int64)
        D = np.zeros((len(v), p), np.int64)
        A = np.full((len(v), p), np.nan)
        for a, (mm, dd, am) in enumerate(v):
            M[a, :len(mm)] = mm
            D[a, :len(dd)] = dd
            A[a, :len(am)] = am
        out.mng_tables[k] = (_t(M, device), _t(D, device),
                             _t(A, device, dtype),
                             _t(np.array([len(r[0]) for r in v]), device))
    out.start_month = _t(np.array(sm), device)
    out.start_day = _t(np.array(sd), device)
    out.emergence = _t(np.array(em), device)
    return out


# ------------------------------------------------------------------ b_co2
def read_bco2(params_path, grid, device='cpu', dtype=torch.float64):
    """b_co2 [veg_class, lat, lon] of the parameter file -> [nveg, N]."""
    with _open(params_path) as ds:
        if 'b_co2' not in ds.variables:
            raise VicWurInputError('%s: no b_co2 (BCO2_SRC FROM_VEGPARAM)'
                                   % params_path)
        a = grid.take(np.array(ds['b_co2'][:], dtype=np.float64))
    return _t(a, device, dtype)


def read_cv(params_path, grid):
    """Cv [veg_class, lat, lon] -> numpy [nveg, N]."""
    with _open(params_path) as ds:
        return grid.take(np.array(ds['Cv'][:], dtype=np.float64))


# ------------------------------------------------------------ global file
_BOOL = {'TRUE': True, 'FALSE': False}


def parse_global(path):
    """Key -> list of token lists (every occurrence), for the keys the
    plugin readers need; the rest are kept too and ignored."""
    out = {}
    with open(path) as f:
        for ln in f:
            ln = ln.split('#')[0].strip()
            if not ln:
                continue
            w = ln.split()
            out.setdefault(w[0].upper(), []).append(w[1:])
    return out


def _g(g, key, default=None):
    v = g.get(key)
    return v[-1][0] if v else default


def _gb(g, key):
    v = _g(g, key)
    if v is None:
        return False
    if v.upper() not in _BOOL:
        raise VicWurInputError('%s must be TRUE or FALSE, got %s' % (key, v))
    return _BOOL[v.upper()]


class VicWurInputs(_Base):
    """Everything read from the plugin files of one global file; an
    attribute is None where its plugin is switched off."""

    def __init__(self):
        self.grid = None
        self.routing = self.dams = self.remote = self.irrigation = None
        self.crop = self.wofost = self.b_co2 = None
        self.warnings = []


def load_vicwur_inputs(global_path, device='cpu', dtype=torch.float64):
    """Read the plugin inputs named in a VIC-WUR global file."""
    g = parse_global(global_path)
    out = VicWurInputs()
    params = _g(g, 'PARAMETERS')
    out.grid = grid = Grid(_g(g, 'DOMAIN'), params)
    cv = read_cv(params, grid)
    nveg = cv.shape[0]
    if _gb(g, 'ROUTING'):
        rspd = _g(g, 'ROUT_STEPS_PER_DAY')
        if rspd is None or int(rspd) == 0:
            # rout_get_global_param.c:94-96 (rout_validate_global_param)
            raise VicWurInputError(
                'Routing time steps per day has not been defined.  Make sure '
                'that the global file defines ROUT_STEPS_PER_DAY.')
        out.routing = read_routing(
            _g(g, 'ROUTING_PARAMETERS'), grid, int(rspd), device, dtype)
    if _gb(g, 'DAMS'):
        if not _gb(g, 'ROUTING'):
            raise VicWurInputError('DAMS needs ROUTING')
        out.dams = read_dams(_g(g, 'DAMS_PARAMETERS'), grid, device, dtype)
    if _gb(g, 'WATERUSE') and _gb(g, 'REMOTE_WITHDRAWAL'):
        out.remote = read_remote(_g(g, 'WATERUSE_PARAMETERS'), grid, device)
    if _gb(g, 'IRRIGATION'):
        pb = None
        if 'PBARE' in g:
            pb = int(_g(g, 'PBARE', PADDY_BARE_UNSET))
        out.irrigation = read_irrigation(
            _g(g, 'IRRIGATION_PARAMETERS'), grid, cv,
            _gb(g, 'OFFSET_IRRIGATION'), pb, int(_g(g, 'NBARE', 1)),
            device, dtype)
    if _gb(g, 'WOFOST'):
        out.crop = read_crop(
            _g(g, 'CROP_PARAMETERS'), grid, nveg,
            _gb(g, 'WOFOST_DIST_SEASON'), _gb(g, 'WOFOST_DIST_TSUM'),
            _gb(g, 'WOFOST_DIST_FERTILIZER'),
            _gb(g, 'WOFOST_DIST_MINERALIZATION'),
            _gb(g, 'WOFOST_CALC_MINERALIZATION'), device, dtype)
        out.wofost = read_wofost_text(_g(g, 'WOFOST_TEXT_PARAMETERS'),
                                      out.crop.n_crop_types, device, dtype)
    if (_g(g, 'BCO2_SRC') or '').upper() == 'FROM_VEGPARAM':
        out.b_co2 = read_bco2(params, grid, device, dtype)
    for a in ('routing', 'dams', 'remote', 'irrigation', 'crop'):
        c = getattr(out, a)
        if c is not None:
            out.warnings += c.warnings
    return out
