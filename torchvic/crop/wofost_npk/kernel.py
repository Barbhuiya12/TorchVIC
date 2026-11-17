# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR WOFOST-NPK crop kernel (C12): batched, GPU-native.

A port of the WOFOST-NPK C code that VIC-WUR 3.0 couples to VIC
(plugins/wofost/src and plugins/crops/src/{crop_run,wofost_run,
wofost_update_step_vars}.c, <data root>/build/vicwur3/src_tree/vic).  Every
SimUnit of the C grid (cell, band, crop) is one lane of a [N, B, NC] tensor;
the C control flow (sowing, emergence, growth, harvest) becomes masks, the
linked list of leaf classes a padded array, and there is no loop over cells,
no host sync and no data-dependent Python branch, so a day can be captured
in a CUDA graph.

Daily only, like the C code (crop_get_global_param.c:91-99 forces
WOFOST_STEPS_PER_DAY 1): one `crop_run` is the crop part of one
VIC-WUR day: crop_reset_meteo/soil, crop_register_meteo/soil, and the
WOFOST day of every unit whose band has AreaFract > 0 (crop_run.c:181-332).

Precision.  The C state is float.  `dtype=float64` evaluates the C code in
double (the reference of the 1e-12 gate is the same C compiled with
`float` defined as `double`, tests/oracle/ref_vicwur_wofost.c).
`dtype=float32` with `c_promote=True` emulates the float C: values are
stored in float32, and every sub-expression the C evaluates in double (a
double literal or a libm call in it) is evaluated in float64 and rounded on
assignment, so most fields match the float C bit for bit.  `c_promote=False`
is plain float32 (faster, a few ulp more).  The only sums done in a
different order than C are the leaf-class sums (LAI, death by stress or
age) and the fertiliser List_cumsum; they run in float64 before rounding.

Where VIC-WUR's WOFOST differs from WOFOST 7.2 (torchvic/crop/wofost72.py,
the PCSE port):
  NPK      the whole nutrient module (Nutrients.c, NutrientPartitioning.c,
           NutrientTranslocation.c, CropNutrientRates.c, NutritionIndex.c,
           SoilNutrientRates.c, IntegrationNutrients.c) and its effects:
           GASS x min(NutrientStress, WaterStress) (RateCalculationCrop.c:
           27-30), leaf partitioning exp(-NPART(1-NNI)) (DM-Partioning.c:
           27-40), SLA x exp(-NSLA(1-NPKI)) (LeaveGrowth.c:24-27), early LAI
           growth x WS x exp(-NLAI(1-NPKI)) (:32-35), extra leaf death
           RDRLV_NPK(1-NPKI) (Leaves.c:30-31); soil N/P/K pools with
           mineralisation, fertiliser tables and VIC-WUR's distributed
           fertiliser (crop_run.c:242-294).
  CO2      EFF x CO2EFFTB(CO2) and AMAX x CO2AMAXTB(CO2) (Assim.c:103-120);
           CO2 from the VIC-WUR CO2 forcing.  WOFOST 7.2 has no CO2.
  VIC      water stress is the daily mean of VIC's cell.water_stress of the
           crop's tile (crop_run.c:108-148) instead of TRA/TRAMX of a crop
           water balance; Tmin = Tmax = Temp = DayTemp = air_temp + Tfactor
           in daily runs (crop_run.c:73-80); no soil limit on rooting depth
           (Growth.c:51-59); the crop state feeds VIC back.
  other    partitioning under water stress FR x max(1, 1/(WS+0.5)) <= 0.6
           (WOFOST 8 style, DM-Partioning.c:15-26); critical LAI =
           CRITLAI/KDIF (Leaves.c:19-26); stem death translocated to storage
           after anthesis (Growth.c:32-37); Tmin averaged over min(GrowthDay,
           7) days (Assim.c:150-173); DVS capped at 1 until anthesis
           (IntegrationCrop.c:26-35); vernalisation only while DVS < 0.3
           (Develop.c:32-45); the youngest leaf class does not age
           (IntegrationCrop.c:54-60); LAI recomputed before integration
           (wofost_run.c:42); CultivateDay incremented twice on the emergence
           day (wofost_run.c:25,49); RDRRTB/RDRSTB swapped (Cropdata.c:
           143-144); P and K demand states accumulate the N demand
           (IntegrationNutrients.c:23-25).
The same equations as WOFOST 7.2 (astronomy, three-point Gaussian canopy
assimilation, maintenance respiration, conversion) are evaluated here in
the C operation order and C float/double promotion, which PCSE does not
follow, so wofost72.py's kernels are not called; its AFGEN padding idea is
reused, but the C Afgen differs (first segment with x_i <= X < x_i+1,
Afgen.c:9-26).

Error bits in state['err'] (the C code exits or crashes there):
  1  growing or sowing with crop cover Cc <= 0 (crop_run.c:299-315)
  2  growing or sowing with tile Cv <= 0 (crop_run.c:316-323)
  4  every leaf class died: NULL dereference in Leaves.c:41 / LeaveGrowth.c:69
  8  more leaf classes than the padded ring holds (not a C condition)
  16 |latitude| > 90 (Astro.c:26-28)
"""
import math

import numpy as np
import torch

from .params import TABLES, DAYS_PER_YEAR

XGAUSS = (0.1127017, 0.5000000, 0.8872983)   # wofost_const_library.c:4-6
WGAUSS = (0.2777778, 0.4444444, 0.2777778)   # :7-9
SCATCOEF = 0.2                               # :3
RAD_PER_DEG = 0.0174532925                   # vic_physical_constants.h:49
CONST_PI = 3.14159265358979323846            # :63
SEC_PER_DAY = 86400
MM_PER_CM = 10
DAYS_PER_WEEK = 7

ERR_CC, ERR_CV, ERR_NOLEAF, ERR_RING, ERR_LAT = 1, 2, 4, 8, 16

NUT = 'NPK'
#: nutrient_states fields kept (wofost.h:233-261, those the code writes)
NST = ('roots stems leaves storage Demand Max_lv Max_st Max_ro Max_so '
       'Optimum_lv Optimum_st Indx Uptake death_lv death_st death_ro Avail '
       'Avail_lv Avail_st Avail_ro').split()
#: nutrient_rates fields (wofost.h:209-231)
NRT = ('roots stems leaves storage Demand Demand_lv Demand_st Demand_ro '
       'Demand_so Supply Transloc Transloc_lv Transloc_st Transloc_ro Uptake '
       'Uptake_lv Uptake_st Uptake_ro death_lv death_st death_ro').split()
GROWTH_RT = 'roots stems leaves LAIExp storage Development RootDepth ' \
            'vernalization'.split()
GROWTH_ST = ('roots stems leaves LAI LAIExp storage Development '
             'Development_prev RootDepth RootDepth_prev vernalization').split()
MET = ('Temp DayTemp Radiation CO2 PotEvaptrans AtmosphTransm '
       'AngotRadiation Daylength PARDaylength SinLD CosLD DiffRadPP DSinBE '
       'TempStress TminStress').split()
LEAF = ('lw', 'la', 'lage')
INT_STATES = ('Sowing Emergence CultivateDay GrowthDay cultivating growing '
              'met_MeteoDay lh lt err').split()


def _where(m, a, b):
    """torch.where with the mask broadcast over trailing dims of a/b."""
    if torch.is_tensor(a) and a.dim() > m.dim():
        m = m.reshape(m.shape + (1,) * (a.dim() - m.dim()))
    elif torch.is_tensor(b) and b.dim() > m.dim():
        m = m.reshape(m.shape + (1,) * (b.dim() - m.dim()))
    return torch.where(m, a, b)


def _at(a, t):
    """a[t] along dim 0.  A 0-d device tensor t is read with index_select:
    a[t] would turn it into a Python int (a host sync, and not allowed
    while a CUDA graph is captured)."""
    if torch.is_tensor(t):
        return a.index_select(0, t.reshape(1).to(torch.int64))[0]
    return a[t]


class _Lanes(object):
    """Class values [NC, ...] read on the lanes: src[k] gathered by the
    lanes' class at each read (nothing kept; in a fused region the gather
    is an indexed load of the consumer)."""

    def __init__(self, src, cls, shape):
        self._src, self._cls, self._shape = src, cls, shape

    def _one(self, x):
        g = x.index_select(0, self._cls)
        return g.view(self._shape + tuple(x.shape[1:]))

    def __getitem__(self, k):
        x = self._src[k]
        if isinstance(x, tuple):                # an AFGEN table
            return tuple(self._one(y) for y in x)
        return self._one(x)

    def __contains__(self, k):
        return k in self._src

    def keys(self):
        return self._src.keys()


#: lanes of the dry run that finds the fields a day writes
_DRY = 64


def _lane_subview(v, n):
    """Lane view v on its first n lanes (the same code path: the leaf
    kernels' row settings are v's)."""
    import copy
    n = max(1, min(int(n), v.U))
    w = copy.copy(v)
    U = v.U
    for k, x in list(vars(v).items()):
        if torch.is_tensor(x) and x.dim() >= 1 and x.shape[0] == U:
            setattr(w, k, x[:n])
    w._lidx = tuple(x[:n] for x in v._lidx)
    w.U = n
    w.shape = (n, 1, 1)
    w.p = _Lanes(v.p._src, w._lcls, w.shape)
    w.m = _Lanes(v.m._src, w._lcls, w.shape)
    w.tab = _Lanes(v.tab._src, w._lcls, w.shape)
    w._aftab = _Lanes(v._aftab._src, w._lcls, w.shape)
    w._keys = w._ls_keys = None
    return w


def _own_rings(s):
    """s with its leaf rings copied (a day writes them in place)."""
    s = dict(s)
    for k in LEAF:
        if k in s:
            s[k] = s[k].clone()
    return s


def _lane_body(v, s, cal, t, cv, keys):
    """The day of lane view v from the unit state s: its units gathered,
    the WOFOST day, the fields in `keys` (None: every field the day
    wrote, found here)."""
    L0 = v._lgather(s)
    cvl = None if cv is None else cv.reshape(-1).index_select(
        0, v._lidx[1]).view(v.U, 1)
    L = v._unit_day(dict(L0), cal, t, v.active, cvl)
    if keys is None:
        return {k: y for k, y in L.items() if y is not L0.get(k)}
    return {k: L[k] for k in keys}


_COMPILED = [None]



def _fusable(v):
    """The lane day as one fused region (core.fusion: inductor with
    eager's arithmetic, reductions eager's kernels): CUDA, fusion on, no
    gradient, float64 or float32 with the C promotion (the C's libm calls
    in float64; float32 exp/pow are not fused), not the C-order sums."""
    from ...core import fusion, xmath
    return (v.device.type == 'cuda' and fusion.FUSION and not xmath.EXACT
            and not torch.is_grad_enabled() and not v.exact_sums
            and v.wd == torch.float64)


def _lane_day(v, s, cal, t, cv):
    """_lane_body, compiled where _fusable. The first call of a lane view
    runs eagerly (it finds the fields the day writes) and compiles; a call
    inside a CUDA graph capture never compiles (an uncompiled variant runs
    eagerly, the same numbers)."""
    keys = getattr(v, '_keys', None)
    if keys is None:
        # the fields the day writes, from a dry run on a few lanes
        w = _lane_subview(v, _DRY)
        s1 = {k: (x[:w.U].clone() if k in LEAF and x.dim() == 2 else x)
              for k, x in s.items()}
        keys = v._keys = tuple(_lane_body(w, s1, cal, t, cv, None))
    if not _fusable(v):
        return _lane_body(v, s, cal, t, cv, keys)
    return _compiled()(v, s, cal, t, cv, keys)


def _ls_body(v, s, cal, t, x, cv, keys):
    """The day of a lane state on lane view v: the inputs gathered to the
    lanes, crop_run's body on them; the fields in `keys` (None: every
    field the day wrote, found here)."""
    U = v.U
    lane, cnc = v._lidx
    cell = v._lcell
    at, sw, co2, pe, ws = x
    xs = [at.index_select(0, cell), sw.index_select(0, cell),
          co2.index_select(0, cell),
          pe.reshape(-1).index_select(0, lane).view(U, 1, 1),
          ws.reshape(-1).index_select(0, lane).view(U, 1, 1)]
    cvl = None if cv is None else cv.reshape(-1).index_select(
        0, cnc).view(U, 1)
    s2 = dict(s)
    for k, y in s.items():
        if k.startswith('con_'):            # [N, NC(, NF)] -> the lanes'
            r = tuple(y.shape[2:])
            s2[k] = y.reshape((-1,) + r).index_select(0, cnc).view(
                (U, 1) + r)
    out = v.crop_run(s2, cal, t, *xs, cv=cvl)
    if keys is None:
        return {k: y for k, y in out.items() if y is not s2.get(k)}
    return {k: out[k] for k in keys}


_COMPILED_LS = [None]


def _ls_day(v, s, cal, t, x, cv):
    """_ls_body, compiled where _fusable (as _lane_day)."""
    keys = getattr(v, '_ls_keys', None)
    if keys is None:
        # the fields the day writes, from a dry run on a few lanes
        w = _lane_subview(v, _DRY)
        s1 = {k: (y if k.startswith('con_') else y[:w.U].clone())
              for k, y in s.items()}
        keys = v._ls_keys = tuple(_ls_body(w, s1, cal, t, x, cv, None))
    if not _fusable(v):
        return _ls_body(v, s, cal, t, x, cv, keys)
    return _compiled_ls()(v, s, cal, t, x, cv, keys)


def _compiled_ls():
    if _COMPILED_LS[0] is None:
        from .fuse import compile_region
        _COMPILED_LS[0] = compile_region(_ls_body)
    return _COMPILED_LS[0]


_COMPILED_RATES = [None]


def _rates_body(m, s, cal, t, v=None):
    return m._rate_outputs(s, cal, t, v)


def _compiled_rates():
    if _COMPILED_RATES[0] is None:
        from .fuse import compile_region
        _COMPILED_RATES[0] = compile_region(_rates_body)
    return _COMPILED_RATES[0]


def _compiled():
    if _COMPILED[0] is None:
        from .fuse import compile_region
        _COMPILED[0] = compile_region(_lane_body)
    return _COMPILED[0]


@torch.library.custom_op("torchvic::wofost_eager_sum", mutates_args=())
def _eager_sum_op(x: torch.Tensor) -> torch.Tensor:
    return x.sum(-1)


@_eager_sum_op.register_fake
def _(x):
    return x.new_empty(x.shape[:-1])


@torch.library.custom_op("torchvic::wofost_eager_cumsum", mutates_args=())
def _eager_cumsum_op(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, -1)


@_eager_cumsum_op.register_fake
def _(x):
    return torch.empty_like(x)


def _eager_sum(x):
    """x.sum(-1), eager's own kernel inside a fused region too (the leaf
    sums of the rows of K classes: K may be beyond core.xsum's checked
    trees)."""
    if torch.compiler.is_compiling():
        return _eager_sum_op(x.contiguous())
    return x.sum(-1)


def _eager_cumsum(x):
    """torch.cumsum(x, -1), eager's own kernel inside a fused region."""
    if torch.compiler.is_compiling():
        return _eager_cumsum_op(x.contiguous())
    return torch.cumsum(x, -1)


def _cmin(a, b):
    """vic_def.h:91 min(a, b) = (a < b) ? a : b."""
    return torch.where(a < b, a, b)


def _cmax(a, b):
    """vic_def.h:92 max(a, b) = (a > b) ? a : b."""
    return torch.where(a > b, a, b)


class WofostNPK(object):
    """Batched WOFOST-NPK over (cell, band, crop class) units.

    tmpl   CropTemplates; setup UnitSetup; options dict of the WOFOST_*
    switches (PIRR PFERT PTEMP DIST_TSUM DIST_FERT CALC_MIN CONTINUE
    FORCE_TSUM FORCE_FERT, default False).  nleaf: slots of the leaf-class
    ring (default: the longest CycleLength + 2, which can never overflow).
    exact_sums: the leaf-class sums of Lai.c and Leaves.c run slot by slot
    in the C order (K small kernels per sum; for validation of the float
    build); by default they are float64 masked reductions.
    """

    #: crop_run writes the units' fields into the state's tensors (set on
    #: the chunk views of crop_run_inplace / day_graphed)
    _inplace = False

    OPTIONS = ('PIRR PFERT PTEMP DIST_TSUM DIST_FERT CALC_MIN CONTINUE '
               'FORCE_TSUM FORCE_FERT').split()

    def __init__(self, tmpl, setup, options=None, dtype=torch.float32,
                 c_promote=True, device='cpu', nleaf=None, exact_sums=False,
                 compact=True, lanes=None, ring=None, lane_state=False):
        opts = dict.fromkeys(self.OPTIONS, False)
        opts.update(options or {})
        bad = set(opts) - set(self.OPTIONS)
        if bad:
            raise ValueError('unknown WOFOST options %s' % sorted(bad))
        self.opt = opts
        self.dt = dtype
        #: sum the leaf classes one by one in the C order (a loop over the
        #: ring slots); otherwise in float64 with masked reductions
        self.exact_sums = bool(exact_sums)
        self.device = torch.device(device)
        promote = c_promote and dtype != torch.float64
        #: dtype of the C `double` variables and sub-expressions
        self.wd = torch.float64 if (c_promote or dtype == torch.float64) \
            else dtype
        self.D = (lambda x: x.to(torch.float64)) if promote else \
            (lambda x: x)
        self.S = lambda x: x.to(dtype)
        npf = np.float32 if dtype == torch.float32 else np.float64
        rnd = lambda v: float(npf(v))
        self.xg = [rnd(v) for v in XGAUSS]
        self.wg = [rnd(v) for v in WGAUSS]
        self.scat = rnd(SCATCOEF)
        # (1 - ScatCoef) in float (Assim.c:60: int 1 - float)
        self.one_m_scat_f = float(npf(1) - npf(self.scat))
        # 0.23 (Astro.c:91) and 0.03 (Leaves.c:22) as the model's floats
        self.c023, self.c003 = rnd(0.23), rnd(0.03)
        self.sq_scat = math.sqrt(1. - self.scat)      # double (Assim.c:33)
        self.tmpl = tmpl
        self.su = setup
        N, B, NC = setup.N, setup.B, setup.NC
        self.shape = (N, B, NC)
        if tmpl.nc != NC:
            raise ValueError('templates have %d crop classes, setup %d'
                             % (tmpl.nc, NC))
        dev, dt = self.device, dtype
        T = lambda a, d=dt: torch.as_tensor(np.asarray(a), dtype=d,
                                            device=dev)
        # class-level parameters [NC]
        self.p = {k: T(np.array(v, np.float64)) for k, v in
                  tmpl.scalars.items()}
        self.idsl = torch.as_tensor(
            np.array(tmpl.scalars['IdentifyAnthesis']).astype(np.int64),
            device=dev)
        self.mopt = T(tmpl.max_optimum_temp)
        self.m = {k: T(np.array(v, np.float64)) for k, v in tmpl.mng.items()}
        #: the class values of m as [NC] (also on a lane view)
        self.m_full = self.m
        self.emerg_flag = torch.as_tensor(np.array(tmpl.emergence) != 0,
                                          device=dev)
        self.tab = tmpl.table_tensors(dev, dt)
        #: AFGEN tables for the lanes: (lower bounds of the segments, +inf
        #: past the table's length (no match there); X; Y; last value)
        self._aftab = {}
        for k_, (X_, Y_, n_) in self.tab.items():
            nn_ = n_.clamp(min=1)
            seg_ = torch.arange(X_.shape[1] - 1, device=dev) < \
                (nn_ - 1)[:, None]
            lo_ = torch.where(seg_, X_[:, :-1],
                              torch.full_like(X_[:, :-1], math.inf))
            yl_ = Y_.gather(1, (nn_ - 1)[:, None])[:, 0]
            self._aftab[k_] = (lo_, X_, Y_, yl_)
        # unit set-up
        self.active = torch.as_tensor(setup.active, device=dev)[:, None, :]
        self.active = self.active.expand(N, B, NC)
        # AreaFract is double in C (soil_con_struct)
        self.af = torch.as_tensor(setup.area_fract, dtype=torch.float64,
                                  device=dev)[:, :, None]
        self.tfactor = torch.as_tensor(setup.tfactor, dtype=self.wd,
                                       device=dev)[:, :, None]
        self.lat = T(setup.lat)[:, None, None]
        self.ph = torch.as_tensor(setup.ph, dtype=self.wd,
                                  device=dev)[:, None, None]
        self.carbon = torch.as_tensor(setup.carbon, dtype=self.wd,
                                      device=dev)[:, None, None]
        self.cv = torch.as_tensor(setup.cv, dtype=torch.float64,
                                  device=dev)[:, None, :]
        self.cc = torch.as_tensor(setup.cc, dtype=torch.float64,
                                  device=dev)          # [N,NC,12]
        self.smon = torch.as_tensor(setup.start_month, device=dev)[:, None]
        self.sday = torch.as_tensor(setup.start_day, device=dev)[:, None]
        self.cycle = torch.as_tensor(setup.cycle, device=dev)[:, None, :]
        self.cycle = self.cycle.to(torch.int32)
        self.NF = setup.NF
        K = int(setup.cycle[setup.active].max()) + 2 if setup.active.any() \
            else 2
        self.K = int(nleaf) if nleaf else max(K, 2)
        self.kidx = torch.arange(self.K, device=dev)
        #: slots of the leaf-class ring of a new state (class k in slot
        #: k mod R; R = K is the plain row): R < K on the CUDA lanes only,
        #: where the leaf sums run on the live classes (leaf_kernels)
        self.ring = self.K if ring is None else int(ring)
        if self.ring > self.K:
            self.ring = self.K
        #: a new state holds the units' fields on the lanes only ([U, 1,
        #: 1]: the running units; the others never change but for the
        #: day's meteo registration, which nothing reads where AreaFract
        #: is 0); see full_state. Not with FORCE_TSUM / FORCE_FERT (their
        #: sowing test runs on every band)
        self.lane_state = bool(lane_state)
        # Leaf classes are kept for the units that can grow only: a unit
        # runs where its crop is active and its band has AreaFract > 0
        # (crop_run.c:181-332), and every write of lw/la/lage is merged
        # under a mask inside that set, so the others stay 0 for ever.
        # compact: lw/la/lage are [U, K] over those U units (row-major
        # (cell, band, crop) order) instead of [N, B, NC, K].
        self.compact = bool(compact)
        run = (self.active & (self.af > 0)).reshape(-1)
        self.lane = torch.nonzero(run).reshape(-1)
        self.U = int(self.lane.numel())
        self.lane_cls = self.lane % NC
        # first lane of every cell (host): the lanes of a cell range
        lc = np.bincount((self.lane // (B * NC)).cpu().numpy(),
                         minlength=N)
        self._lptr = np.concatenate([[0], np.cumsum(lc)]).tolist()
        # the lanes of each band (positions) and their (cell, crop) index
        lb = (self.lane // NC) % B
        lcnc = (self.lane // (B * NC)) * NC + self.lane_cls
        self._lband = []
        for b in range(B):
            pos = torch.nonzero(lb == b).reshape(-1)
            self._lband.append((pos, lcnc.index_select(0, pos)))
        #: lanes: the day of the U running units is computed on [U] lanes
        #: (gathered from and scattered back into the [N, B, NC] state)
        #: instead of on every (cell, band, crop) unit: elementwise the
        #: same operations on the same values. On CUDA every element of an
        #: op runs the same code, so the same bits; on the CPU a libm call
        #: (exp, pow, ...) differs between the vector body and the tail of
        #: an op, so a layout change can move the last bit. Default: on
        #: CUDA. Needs the compact leaf classes.
        if lanes is None:
            lanes = self.device.type == 'cuda'
        self.lanes = bool(lanes) and self.compact
        self._lanemode = False
        self._lvkey, self._lvc = (0, N), {}

    # ------------------------------------------------------------ state
    def init_state(self):
        """Initial unit state: wofost_init_library.c (all zero), then
        Cropdata.c:160-285 and GetSimInput.c:35-39."""
        dev, dt, wd = self.device, self.dt, self.wd
        mins = torch.as_tensor(self.su.mins, dtype=dt, device=dev)
        if self._ls_on():
            v = self._lane_view()
            N, B, NC = v.shape
            p = v.p
            ml = mins.reshape(-1, mins.shape[-1]).index_select(
                0, v._lidx[1]).view(N, B, NC, mins.shape[-1])
            minsf = lambda j: ml[..., j].clone()
        else:
            N, B, NC = self.shape
            p = self.p
            minsf = lambda j: mins[:, None, :, j].expand(N, B, NC).clone()
        z = lambda: torch.zeros(N, B, NC, dtype=dt, device=dev)
        zi = lambda: torch.zeros(N, B, NC, dtype=torch.int32, device=dev)
        s = {}
        for k in INT_STATES:
            s[k] = zi()
        for k in ('NPK_Indx NutrientStress TSumEmergence fac_ro fac_lv '
                  'fac_st fac_so soil_WaterStress').split():
            s[k] = z()
        s['NPK_Indx'] = s['NPK_Indx'] + 1
        s['NutrientStress'] = s['NutrientStress'] + 1
        for k in GROWTH_RT:
            s['rt_' + k] = z()
        for k in GROWTH_ST:
            s['st_' + k] = z()
        for k in ('roots', 'stems', 'leaves'):
            s['drt_' + k] = z()
            s['dst_' + k] = z()
        for X in NUT:
            for k in NST:
                s['%s_st_%s' % (X, k)] = z()
            s['%s_st_Indx' % X] = s['%s_st_Indx' % X] + 1
            for k in NRT:
                s['%s_rt_%s' % (X, k)] = z()
            for k in ('st_%s_tot', 'st_%s_mins', 'rt_%s_tot', 'rt_%s_mins'):
                s['ste_' + k % X] = z()
            s['mng_%s_external' % X] = z()
        for j, k in enumerate(('N_Mins NRecoveryFrac P_Mins PRecoveryFrac '
                               'K_Mins KRecoveryFrac').split()):
            s['mng_' + k] = minsf(j)
        for k in MET:
            s['met_' + k] = z()
        s['met_Tmin'] = torch.zeros(N, B, NC, DAYS_PER_WEEK, dtype=wd,
                                    device=dev)
        s['met_Tmax'] = torch.zeros(N, B, NC, dtype=wd, device=dev)
        s['prm_TempSum1'] = (p['TempSum1'] + z())
        s['prm_TempSum2'] = (p['TempSum2'] + z())
        s['prm_MaxRootingDepth'] = (p['MaxRootingDepth'] + z())
        s['lw'] = (torch.zeros(self.U, self.ring, dtype=dt, device=dev)
                   if self.compact else
                   torch.zeros(N, B, NC, self.K, dtype=dt, device=dev))
        s['la'] = torch.zeros_like(s['lw'])
        s['lage'] = torch.zeros_like(s['lw'])
        su = self.su
        s['con_tsum1'] = torch.as_tensor(su.tsum1, dtype=wd, device=dev)
        s['con_tsum2'] = torch.as_tensor(su.tsum2, dtype=wd, device=dev)
        fert = torch.as_tensor(su.fert, dtype=wd, device=dev)
        for j, k in enumerate(('dvs', 'N', 'P', 'K')):
            s['con_' + k] = fert[..., j].clone()        # [N,NC,NF]
        return s

    # ------------------------------------------------------- leaf lanes
    @staticmethod
    def _lanes(w):
        """Whether the leaf classes of state `w` are compact [U, K]."""
        return w['lw'].dim() == 2

    def _ringip(self):
        """The leaf ring is updated in place, unit by unit (the lanes with
        the leaf sums on the live classes): the day's ring is its own
        (crop_run clones it unless it writes the state in place)."""
        return self._lanemode and self._xr

    def _ls_on(self):
        """New states on the lanes (lane_state, where it applies)."""
        return (self.lane_state and self.lanes and self.compact
                and self.U > 0 and not self._lanemode
                and not (self.opt['FORCE_TSUM'] or self.opt['FORCE_FERT']))

    def is_lane_state(self, s):
        """Whether state s holds the units' fields on the lanes."""
        x = s['st_LAI']
        return (not self._lanemode and self.U > 0
                and tuple(x.shape) == (self.U, 1, 1)
                and tuple(self.shape) != (self.U, 1, 1))

    def lane_values(self, s, k):
        """Field k of state s on the lanes [U, ...]."""
        x = s[k]
        if self.is_lane_state(s):
            return x.reshape((self.U,) + tuple(x.shape[3:]))
        return x.reshape((-1,) + tuple(x.shape[3:])).index_select(
            0, self.lane)

    #: a unit's fields that start at 1 (init_state); every other field a
    #: lane state leaves out starts at 0
    _ONES = ('NPK_Indx', 'NutrientStress', 'N_st_Indx', 'P_st_Indx',
             'K_st_Indx')

    def full_state(self, s, keys=None):
        """The fields `keys` (default all) of state s on [N, B, NC] units:
        a lane state's values on the lanes, the initial value on the units
        that do not run (where AreaFract is 0: every reader multiplies
        their values by it). A unit-layout state as it is."""
        if not self.is_lane_state(s):
            return s if keys is None else {k: s[k] for k in keys}
        N, B, NC = self.shape
        out = {}
        for k in (s if keys is None else keys):
            x = s[k]
            if k in LEAF or k.startswith('con_'):
                out[k] = x
                continue
            r = tuple(x.shape[3:])
            base = (torch.ones if k in self._ONES else torch.zeros)(
                (N * B * NC,) + r, dtype=x.dtype, device=x.device)
            out[k] = base.index_copy(0, self.lane, x.reshape(
                (self.U,) + r)).view((N, B, NC) + r)
        return out

    def _cl(self, x, c):
        """A unit field [N, B, NC] (or broadcastable to it) in the layout
        of the leaf classes: [U] when compact (c), else as it is."""
        if not c:
            return x
        if self._lanemode:
            return x.expand(self.shape).reshape(-1)
        return x.expand(self.shape).reshape(-1).index_select(0, self.lane)

    def _dn(self, xu, c, base):
        """A leaf-layout field back on the units: `base` [N, B, NC] with
        the U units' values replaced (as it is when not compact)."""
        if not c:
            return xu
        if self._lanemode:
            return xu.to(base.dtype).reshape(self.shape)
        b = base.expand(self.shape).reshape(-1)
        return b.index_copy(0, self.lane, xu.to(b.dtype)).reshape(self.shape)

    def _pk(self, name, c):
        """Class parameter `name` against the leaf classes [..., K]."""
        v = self.p[name]
        if self._lanemode:
            return v.reshape(-1, 1)
        return v.index_select(0, self.lane_cls)[:, None] if c else \
            v[..., None]

    #: per-cell set-up, sliced by `cells`
    _PER_CELL = ('active', 'af', 'tfactor', 'lat', 'ph', 'carbon', 'cv',
                 'cc', 'smon', 'sday', 'cycle')

    def cells(self, c0, c1):
        """The model on cells c0 .. c1-1: the per-cell set-up sliced (views,
        nothing copied), the class parameters shared. Every unit's day
        depends on its own cell only, so a run on the chunks is the run on
        all cells, bit for bit."""
        import copy
        v = copy.copy(self)
        N, B, NC = self.shape
        v.shape = (c1 - c0, B, NC)
        for k in self._PER_CELL:
            setattr(v, k, getattr(self, k)[c0:c1])
        u0, u1 = self._lptr[c0], self._lptr[c1]
        v.lane = self.lane[u0:u1] - c0 * B * NC
        v.lane_cls = self.lane_cls[u0:u1]
        v.U = u1 - u0
        v._lptr = [x - u0 for x in self._lptr[c0:c1 + 1]]
        v._lvkey = (c0, c1)
        v._scan_cells = None
        return v

    # ------------------------------------------------------- lane view
    def _lane_view(self):
        """This model on its U running units as [U, 1, 1] lanes: every
        per-unit, per-cell and per-class value gathered to its lane (class
        parameters and tables on first use). Cached per chunk."""
        scan = getattr(self, '_scan_cells', None)
        key = self._lvkey + (scan,)
        v = self._lvc.get(key)
        if v is not None:
            return v
        import copy
        v = copy.copy(self)
        N, B, NC = self.shape
        U = self.U
        lane = self.lane
        cls = self.lane_cls
        cell = lane // (B * NC)
        cb = cell * B + (lane // NC) % B
        cnc = cell * NC + cls
        sh = (U, 1, 1)
        g = lambda x, idx: x.reshape(-1).index_select(0, idx).view(sh)
        v._lanemode = True
        v._full = (N, B, NC)
        v._lcell = cell
        v._lidx = (lane, cnc)
        v._lcls = cls
        v.shape = sh
        v.active = torch.ones(sh, dtype=torch.bool, device=self.device)
        v.af = g(self.af.expand(N, B, 1), cb)
        v.tfactor = g(self.tfactor.expand(N, B, 1), cb)
        for k in ('lat', 'ph', 'carbon'):
            setattr(v, k, g(getattr(self, k), cell))
        for k in ('cv', 'smon', 'sday', 'cycle'):
            setattr(v, k, g(getattr(self, k).expand(N, 1, NC), cnc))
        v.cc = self.cc.reshape(N * NC, 12).index_select(0, cnc).view(U, 1, 12)
        for k in ('idsl', 'mopt', 'emerg_flag'):
            setattr(v, k, g(getattr(self, k), cls))
        v.p = _Lanes(self.p, cls, sh)
        v.m = _Lanes(self.m, cls, sh)
        v.tab = _Lanes(self.tab, cls, sh)
        v._aftab = _Lanes(self._aftab, cls, sh)
        v.lane = None
        from . import leaf_kernels as LK
        from ...core import xscan
        # the rows of each lane's [rows, K] leaf tensor in eager's chunked
        # day (chunks of `scan` cells; one chunk without), and the scan
        # chunk eager's cumsum used for that row count
        lp = self._lptr
        rows = [(lp[c0], lp[c1]) for c0, c1 in self.chunks(scan)]
        # filled on the device (no host copy: no sync in the step)
        v._ch = torch.zeros(U, dtype=torch.int32, device=self.device)
        chmax = 0
        for u0, u1 in rows:
            if u1 > u0:
                c_ = xscan._chunk(u1 - u0, self.K)
                v._ch[u0:u1].fill_(c_)
                chmax = max(chmax, c_)
        #: the leaf sums as eager's trees on the live classes (a ring)
        v._xr = (self.device.type == 'cuda' and not self.exact_sums
                 and all(LK.supported(self.K, u1 - u0)
                         for u0, u1 in rows if u1 > u0)
                 and chmax <= 1024)
        self._lvc[key] = v
        return v

    def _cls(self, x):
        """A class vector [NC] (of the calendar) in the unit layout."""
        if self._lanemode:
            return x.index_select(0, self._lcls).view(self.shape)
        return x

    def _lgather(self, s):
        """The state's units on the lanes (lane view)."""
        N, B, NC = self._full
        lane, cnc = self._lidx
        U = self.U
        out = {}
        for k, x in s.items():
            if k in LEAF and x.dim() == 2:
                out[k] = x
            elif k.startswith('con_'):
                r = tuple(x.shape[2:])
                out[k] = x.reshape((N * NC,) + r).index_select(
                    0, cnc).view((U, 1) + r)
            else:
                if tuple(x.shape[:3]) != (N, B, NC):
                    raise ValueError('WOFOST lanes: field %s of shape %s'
                                     % (k, tuple(x.shape)))
                r = tuple(x.shape[3:])
                out[k] = x.reshape((N * B * NC,) + r).index_select(
                    0, lane).view((U, 1, 1) + r)
        return out

    def _lscatter(self, s, L, inplace):
        """The lanes' new values L (the fields the day wrote) back into the
        units of `s` (in place into s's tensors when `inplace`)."""
        lane = self._lidx[0]
        U = self.U
        for k, y in L.items():
            if k in LEAF and y.dim() == 2:
                s[k] = y
                continue
            if k not in s or k.startswith('con_'):
                raise ValueError('WOFOST lanes: field %s written' % k)
            x = s[k]
            r = tuple(x.shape[3:])
            y = y.reshape((U,) + r)
            if inplace and x.dtype == y.dtype:
                x.view((-1,) + r).index_copy_(0, lane, y)
            else:
                x = x.to(y.dtype)
                s[k] = x.reshape((-1,) + r).index_copy(0, lane, y).view(
                    x.shape)
        return s

    def _slice(self, s, c0, c1):
        if self.is_lane_state(s):
            if (c0, c1) != (0, self.shape[0]):
                raise ValueError('WOFOST: a lane state runs in one chunk')
            return dict(s)
        u0, u1 = self._lptr[c0], self._lptr[c1]
        return {k: (x[u0:u1] if k in LEAF and x.dim() == 2 else x[c0:c1])
                for k, x in s.items()}

    def chunks(self, cells):
        N = self.shape[0]
        if not cells or cells >= N:
            return [(0, N)]
        return [(c, min(c + cells, N)) for c in range(0, N, cells)]

    @staticmethod
    def own(s):
        """Make every tensor of state `s` contiguous and the only owner of
        its memory (no two fields share it, no broadcast views), so that
        `crop_run_inplace` can write into it."""
        seen = set()
        for k in list(s):
            x = s[k]
            if not torch.is_tensor(x):
                continue
            ptr = x.untyped_storage().data_ptr()
            if (not x.is_contiguous() or ptr in seen
                    or x.untyped_storage().nbytes()
                    != x.numel() * x.element_size()):
                x = x.clone(memory_format=torch.contiguous_format)
                s[k] = x
                ptr = x.untyped_storage().data_ptr()
            seen.add(ptr)
        return s

    def crop_run_inplace(self, s, cal, t, air_temp, shortwave, co2,
                         pot_evap, water_stress, cv=None, cells=None):
        """`crop_run` with the state written in place, `cells` cells at a
        time (the day's temporaries are a chunk's, and no second copy of
        the state is built). `s` must be owned (`own`). Returns s."""
        sl = lambda x, a, b: None if x is None else x[a:b]
        chunks, scan = self._run_chunks(s, cells)
        for c0, c1 in chunks:
            v = self.cells(c0, c1)
            v._inplace = True
            v._scan_cells = scan
            sub = self._slice(s, c0, c1)
            out = v.crop_run(sub, cal, t, sl(air_temp, c0, c1),
                             sl(shortwave, c0, c1), sl(co2, c0, c1),
                             sl(pot_evap, c0, c1), sl(water_stress, c0, c1),
                             cv=sl(cv, c0, c1))
            mine = {x.untyped_storage().data_ptr() for x in sub.values()
                    if torch.is_tensor(x)}
            new = {}
            for k, x in out.items():
                y = sub.get(k)
                if x is y:
                    continue
                if y is None or y.shape != x.shape or y.dtype != x.dtype:
                    raise ValueError('crop_run_inplace: field %s changed '
                                     'shape or type' % k)
                # a result that is (a view of) an input field is read
                # before any field is written
                if x.untyped_storage().data_ptr() in mine:
                    x = x.clone()
                new[k] = x
            for k, x in new.items():
                sub[k].copy_(x)
        return s

    def _run_chunks(self, s, cells):
        """(chunks, scan): the chunks a day runs in, and the chunking of
        `cells` the leaf sums follow. On the lanes the day is one chunk
        (its temporaries are the running units'), its leaf sums each
        lane's chunk of `cells` (eager's scan of the [rows, K] tensor
        depends on the row count): where leaf_kernels reproduce eager's
        sums of each chunk, or the sums are the C-order loops (exact_sums,
        no reduction); else the chunks of `cells`."""
        if self.is_lane_state(s):
            return [(0, self.shape[0])], cells
        if self.lanes and self.U > 0 and self._lanes(s):
            from . import leaf_kernels as LK
            lp = self._lptr
            if self.exact_sums or all(
                    LK.supported(self.K, lp[c1] - lp[c0])
                    for c0, c1 in self.chunks(cells) if lp[c1] > lp[c0]):
                return [(0, self.shape[0])], cells
        return self.chunks(cells), None

    def _chunk_day(self, s, cal, t, x, cv, c0, c1, scan=None):
        """crop_run of cells c0..c1-1 written in place into s, then its
        rate_outputs; x = (air_temp, shortwave, co2, pot_evap,
        water_stress)."""
        v = self.cells(c0, c1)
        v._scan_cells = scan
        sub = self._slice(s, c0, c1)
        xs = [y[c0:c1] for y in x]
        v._inplace = True
        out = v.crop_run(sub, cal, t, *xs,
                         cv=None if cv is None else cv[c0:c1])
        mine = {y.untyped_storage().data_ptr() for y in sub.values()
                if torch.is_tensor(y)}
        new = {}
        for k, y in out.items():
            z = sub.get(k)
            if y is z:
                continue
            if z is None or z.shape != y.shape or z.dtype != y.dtype:
                raise ValueError('crop_run_inplace: field %s changed shape '
                                 'or type' % k)
            if y.untyped_storage().data_ptr() in mine:
                y = y.clone()
            new[k] = y
        for k, y in new.items():
            sub[k].copy_(y)
        return v.rate_outputs(sub, cal, t)

    def day_graphed(self, s, cal, t, air_temp, shortwave, co2, pot_evap,
                    water_stress, cv=None, cells=None):
        """`crop_run_inplace` and `rate_outputs_cells` of day t with every
        chunk replayed from a CUDA graph (the kernel has no host sync and
        no data-dependent branch): the same kernels on the same data, so
        the same bits, without the launch cost of some 10^4 small ops per
        chunk. The first call of a state layout runs eagerly and captures;
        a new layout (another state's tensors, inputs of other shapes)
        captures again. Returns the rate outputs ([N, NC] float64, fresh
        tensors)."""
        x = (air_temp, shortwave, co2, pot_evap, water_stress)
        chunks, scan = self._run_chunks(s, cells)
        key = (tuple((k, v.data_ptr(), v.dtype, tuple(v.shape))
                     for k, v in s.items()),
               tuple((y.dtype, tuple(y.shape)) for y in x),
               None if cv is None else (cv.dtype, tuple(cv.shape)),
               id(cal), tuple(chunks))
        G = getattr(self, '_graphs', None)
        if G is not None and G['key'] == key:
            for b, y in zip(G['x'], x):
                b.copy_(y)
            if cv is not None:
                G['cv'].copy_(cv)
            G['t'].fill_(int(t))
            for g in G['g']:
                g.replay()
            outs = G['out']
            if len(outs) == 1:
                return {k: v.clone() for k, v in outs[0].items()}
            return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
        # eager (this is the day's run), then capture for the next days
        parts = [self._chunk_day(s, cal, t, x, cv, c0, c1, scan)
                 for c0, c1 in chunks]
        self._graphs = None
        torch.cuda.synchronize(self.device)
        xb = [y.clone() for y in x]
        cvb = None if cv is None else cv.clone()
        tb = torch.zeros((), dtype=torch.long, device=self.device)
        pool = torch.cuda.graph_pool_handle()
        graphs, outs = [], []
        for c0, c1 in chunks:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                outs.append(self._chunk_day(s, cal, tb, xb, cvb, c0, c1,
                                            scan))
            graphs.append(g)
        self._graphs = dict(key=key, x=xb, cv=cvb, t=tb, g=graphs, out=outs)
        if len(parts) == 1:
            return parts[0]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def rate_outputs_cells(self, s, cal, t, cells=None):
        """`rate_outputs` a chunk of cells at a time."""
        parts = []
        chunks, scan = self._run_chunks(s, cells)
        for c0, c1 in chunks:
            v = self.cells(c0, c1)
            v._scan_cells = scan
            parts.append(v.rate_outputs(self._slice(s, c0, c1), cal, t))
        if len(parts) == 1:
            return parts[0]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def leaf_arrays(self, s):
        """lw, la, lage of state `s` as [N, B, NC, K] (for checks)."""
        out = {}
        R = s['lw'].shape[-1]
        if s['lw'].dim() == 2 and R != self.K:
            # the ring unrolled: class k from slot k mod R, 0 off [lh, lt)
            idx = torch.arange(self.K, device=s['lw'].device)
            h = self.lane_values(s, 'lh').long()
            t = self.lane_values(s, 'lt').long()
            live = (idx >= h[:, None]) & (idx < t[:, None])
            s = dict(s)
            for k in ('lw', 'la', 'lage'):
                x = s[k][:, idx % R]
                s[k] = torch.where(live, x, torch.zeros_like(x))
        for k in ('lw', 'la', 'lage'):
            x = s[k]
            if x.dim() == 2:
                d = torch.zeros(self.shape + (self.K,), dtype=x.dtype,
                                device=x.device)
                d.view(-1, self.K).index_copy_(0, self.lane, x)
                x = d
            out[k] = x
        return out

    # ------------------------------------------------------------ helpers
    def afgen(self, name, x):
        """Afgen.c:9-26 on the class table `name`, x [N,B,NC] (dtype)."""
        if self._lanemode or torch.compiler.is_compiling():
            # element-wise (the same values): fuses with its neighbours
            return self._afgen_lanes(name, x)
        X, Y, n = self.tab[name]
        N, B, NC = x.shape
        P = X.shape[-1]
        Xe = X.expand(N, B, NC, P)
        Ye = Y.expand(N, B, NC, P)
        nn = n.clamp(min=1)
        xx = x[..., None]
        if P > 1:
            seg = torch.arange(P - 1, device=x.device) < (nn - 1)[:, None]
            xi, xn = Xe[..., :-1], Xe[..., 1:]
            match = seg & (xx >= xi) & (xx < xn)
            anym = match.any(-1)
            first = match.to(torch.int32).argmax(-1, keepdim=True)
            g = lambda A: A.gather(-1, first)[..., 0]
            x0, x1 = g(Xe[..., :-1]), g(Xe[..., 1:])
            y0, y1 = g(Ye[..., :-1]), g(Ye[..., 1:])
            val = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        else:
            anym = torch.zeros_like(x, dtype=torch.bool)
            val = x
        ylast = Ye.gather(-1, (nn - 1).expand(N, B, NC)[..., None])[..., 0]
        out = torch.where(anym, val, ylast)
        return torch.where(x <= Xe[..., 0], Ye[..., 0], out)

    def _afgen_lanes(self, name, x):
        """afgen on the lanes, element-wise: every segment's interpolation
        (the same operations in the same order), the first matching
        segment kept by a chain of selections from the last segment to the
        first; no match: the last value; x <= the first point: its value.
        The same values as the masked argmax of `afgen`."""
        # [U,1,1,P-1], [U,1,1,P], ... on the lanes; [NC, P-1], ... else
        lo, X, Y, ylast = self._aftab[name]
        P = X.shape[-1]
        out = ylast
        for i in range(P - 2, -1, -1):
            x0, x1 = X[..., i], X[..., i + 1]
            y0, y1 = Y[..., i], Y[..., i + 1]
            match = (x >= lo[..., i]) & (x < x1)
            val = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
            out = torch.where(match, val, out)
        if P <= 1:
            out = out.expand_as(x)
        return torch.where(x <= X[..., 0], Y[..., 0], out)

    def limit(self, a, b, c):
        """Functions.c:4-18, float arguments."""
        c = self.S(c)
        return torch.where(c < a, torch.full_like(c, a) if not
                           torch.is_tensor(a) else a.to(c.dtype).expand_as(c),
                           torch.where(c <= b, c, torch.full_like(c, b) if not
                                       torch.is_tensor(b) else
                                       b.to(c.dtype).expand_as(c)))

    @staticmethod
    def insw(x1, x2, x3):
        """Functions.c:31-37."""
        return torch.where(x1 < 0, x2, x3)

    # ------------------------------------------------------------ U phase
    def update_step_vars(self, s, cal, t, forced=None):
        """wofost_update_step_vars.c:4-71 (U phase, FORCE_TSUM/FERT only).

        forced: dict with tsum1, tsum2 [N,NC] and dvs, N, P, K [N,NC,NF] of
        the crop forcing record in use.  IfSowing runs on every band, also
        AreaFract-0 ones, and crop_con of a cell changes when any band of it
        has Sowing == 1.
        """
        if not (self.opt['FORCE_TSUM'] or self.opt['FORCE_FERT']):
            return s
        s = dict(s)
        sow = self._sowing_today(cal, t) & self.active
        s['Sowing'] = torch.where(sow, torch.ones_like(s['Sowing']),
                                  s['Sowing'])
        upd = ((s['Sowing'] == 1) & self.active).any(1)       # [N,NC]
        wd = self.wd
        if self.opt['FORCE_TSUM']:
            for k in ('tsum1', 'tsum2'):
                s['con_' + k] = torch.where(upd, forced[k].to(wd),
                                            s['con_' + k])
        if self.opt['FORCE_FERT']:
            for k in ('dvs', 'N', 'P', 'K'):
                s['con_' + k] = torch.where(upd[..., None],
                                            forced[k].to(wd), s['con_' + k])
        return s

    def _sowing_today(self, cal, t):
        """IfSowing.c:11-22: [N,1,NC] bool."""
        return (_at(cal.month, t) == self.smon) & \
            (_at(cal.day, t) == self.sday)

    # ------------------------------------------------------------ P3 phase
    def crop_run(self, s, cal, t, air_temp, shortwave, co2, pot_evap,
                 water_stress, cv=None):
        """One VIC-WUR day of crop_run (crop_run.c:181-332).

        cal Calendar, t step index (python int or 0-d device long tensor).
        air_temp, shortwave, co2 [N] (air_temp[NR] in C, shortwave[NR] in
        W m-2, CO2 in ppm); pot_evap [N,B,NC] (mm, of the crop's tile),
        water_stress [N,B,NC] (cell.water_stress of the tile); cv [N,NC]
        Cv of the crop's tile when land use changes it (default: the
        set-up's).  Returns the new state dict (on a view of
        crop_run_inplace, the units' fields are written into s's tensors
        where their type allows).
        """
        if self.is_lane_state(s):
            # the whole day on the lanes (inputs gathered)
            v = self._lane_view()
            if v._xr and not self._inplace:
                s = _own_rings(s)
            if not torch.is_tensor(t):
                t = torch.full((), int(t), dtype=torch.long,
                               device=self.device)
            new = _ls_day(v, s, cal, t, (air_temp, shortwave, co2, pot_evap,
                                         water_stress), cv)
            out = dict(s)
            for k, y in new.items():
                x = s[k]
                if (self._inplace and x.shape == y.shape
                        and x.dtype == y.dtype):
                    x.copy_(y)
                else:
                    out[k] = y
            return out
        D, S, wd, dt = self.D, self.S, self.wd, self.dt
        s = dict(s)
        N, B, NC = self.shape
        # crop_reset_meteo / crop_reset_soil (crop_run.c:4-31,87-106):
        # every step of a daily run is the first of its day
        tm = s['met_Tmin']
        tm = torch.cat([torch.full_like(tm[..., :1], 9999.),
                        tm[..., :DAYS_PER_WEEK - 1]], -1)
        tmax = torch.full_like(s['met_Tmax'], -9999.)
        # crop_register_meteo (crop_run.c:33-85); the step ratios are 1
        at = air_temp.to(wd)[:, None, None]
        x = at + self.tfactor
        t0 = _cmin(tm[..., 0], x)
        tm = torch.cat([t0[..., None], tm[..., 1:]], -1)
        tmax = _cmax(tmax, x)
        s['met_Tmin'] = tm
        s['met_Tmax'] = tmax
        s['met_CO2'] = S((co2.to(wd) * 1.0)[:, None, None].expand(N, B, NC))
        s['met_Radiation'] = S((shortwave.to(wd) * SEC_PER_DAY *
                                1.0)[:, None, None].expand(N, B, NC))
        s['met_PotEvaptrans'] = S(pot_evap.to(wd) / MM_PER_CM * 1.0)
        temp = S(0.5 * (tm[..., 0] + tmax))
        s['met_Temp'] = temp
        s['met_DayTemp'] = S(0.5 * (temp.to(wd) + tmax))
        # crop_register_soil (crop_run.c:108-148)
        if self.opt['PIRR']:
            s['soil_WaterStress'] = S(torch.full((N, B, NC), 1.0, dtype=wd,
                                                 device=self.device))
        else:
            s['soil_WaterStress'] = S(water_stress.to(wd) / 1.0 / 1.0)

        run = self.active & (self.af > 0)
        s['met_MeteoDay'] = torch.where(
            run, _at(cal.doy, t).to(torch.int32).expand(N, B, NC),
            s['met_MeteoDay'])
        if s['lw'].shape[-1] != self.K and not self._lanemode and not (
                self.lanes and self.U > 0 and self._lanes(s)
                and self._lane_view()._xr):
            raise ValueError('WOFOST: a leaf-class ring of %d < K = %d slots '
                             'needs the CUDA lanes (and leaf_kernels.'
                             'supported)' % (s['lw'].shape[-1], self.K))
        if (self.lanes and not self._lanemode and self.U > 0
                and self._lanes(s)):
            # the rest of the day touches the running units only
            v = self._lane_view()
            if v._xr and not self._inplace:
                s = _own_rings(s)
            if not torch.is_tensor(t):
                t = torch.full((), int(t), dtype=torch.long,
                               device=self.device)
            new = _lane_day(v, s, cal, t, cv)
            return v._lscatter(s, new, self._inplace)
        return self._unit_day(s, cal, t, run, cv)

    def _unit_day(self, s, cal, t, run, cv):
        """crop_run.c:181-332 after the meteo and soil registration: the
        WOFOST day of the units in `run` and the coverage check."""
        D, S = self.D, self.S
        N, B, NC = self.shape
        sow = self._sowing_today(cal, t)
        one = torch.ones_like(s['Sowing'])
        s['Sowing'] = torch.where(run & sow, one, s['Sowing'])
        if self.opt['DIST_TSUM']:
            for k, nm in (('prm_TempSum1', 'con_tsum1'),
                          ('prm_TempSum2', 'con_tsum2')):
                s[k] = torch.where(run, S(s[nm])[:, None, :].expand(N, B, NC),
                                   s[k])
        if self.opt['DIST_FERT']:
            dvsi = D(self.p['InitialDVS'])
            for f in range(self.NF):
                pt = s['con_dvs'][:, None, :, f]
                c1 = (pt <= dvsi) & (s['Sowing'] == 1)
                c2 = (~c1) & (pt > dvsi) & (D(s['st_Development']) >= pt) & \
                    (D(s['st_Development_prev']) <= pt) & \
                    (s['rt_Development'] != 0.)
                add = run & (c1 | c2)
                for X in NUT:
                    k = 'mng_%s_external' % X
                    s[k] = torch.where(
                        add, S(D(s[k]) + s['con_' + X][:, None, :, f]), s[k])
        if self.opt['CALC_MIN']:
            ph = self.ph
            fN = 0.25 * (ph - 3)
            fN = torch.where(ph > 7.0, torch.ones_like(fN),
                             torch.where(ph < 4.7, torch.full_like(fN, 0.4),
                                         fN))
            temp = s['met_Temp']
            aN = 2 * torch.pow(torch.full_like(D(temp), 2.0),
                               (D(temp) - 9.0) / 9.0)
            nm = S(fN * aN * self.carbon)
            s['mng_N_Mins'] = torch.where(run, nm, s['mng_N_Mins'])
            s['ste_st_N_mins'] = torch.where(run, nm, s['ste_st_N_mins'])
        s = self._wofost_run(s, cal, t, run)
        # crop coverage check (crop_run.c:298-324)
        chk = run & (((s['growing'] == 1) & (not self.opt['CONTINUE'])) |
                     (s['Sowing'] == 1))
        ccm = self.cc[:, None, :, :].expand(N, B, NC, 12)
        mon = (_at(cal.month, t) - 1).expand(N, B, NC)[..., None]
        ccv = ccm.gather(-1, mon)[..., 0]
        err = s['err']
        err = err | torch.where(chk & (ccv <= 0), ERR_CC, 0).to(torch.int32)
        cvv = self.cv if cv is None else cv.to(torch.float64)[:, None, :]
        err = err | torch.where(chk & (cvv <= 0), ERR_CV, 0).to(torch.int32)
        s['err'] = err
        return s

    # ------------------------------------------------------------ wofost_run
    def _wofost_run(self, s, cal, t, run):
        """wofost_run.c:4-69 for the lanes in `run`."""
        S = self.S
        sow = self._sowing_today(cal, t)
        s['Sowing'] = torch.where(run & sow, torch.ones_like(s['Sowing']),
                                  s['Sowing'])
        # sowing / emergence block (:15-26)
        A = run & (s['Sowing'] >= 1) & (s['Emergence'] == 0)
        s['cultivating'] = torch.where(A, torch.ones_like(s['cultivating']),
                                       s['cultivating'])
        flag = self.emerg_flag.expand_as(A)
        # EmergenceCrop (InitializeCrop.c:9-36)
        was1 = s['Sowing'] == 1
        two = torch.full_like(s['Sowing'], 2)
        acc = A & (~flag) & (~was1)
        dts = self.limit(0., self.p['TempEffMax'] -
                         self.p['TempBaseEmergence'],
                         s['met_Temp'] - self.p['TempBaseEmergence'])
        tsum = torch.where(acc, s['TSumEmergence'] + dts, s['TSumEmergence'])
        s['TSumEmergence'] = tsum
        emerged = A & (flag | (acc & (tsum >= self.p['TSumEmergence'])))
        s['Sowing'] = torch.where(A & (flag | was1), two, s['Sowing'])
        w = dict(s)
        if self._ringip():
            w['__lmask'] = emerged
        w = self._initialize(w, cal, t)
        w.pop('__lmask', None)
        s = self._merge(s, w, emerged)
        s['growing'] = torch.where(emerged, torch.ones_like(s['growing']),
                                   s['growing'])
        s['CultivateDay'] = torch.where(A, s['CultivateDay'] + 1,
                                        s['CultivateDay'])
        # growth / harvest block (:28-68)
        G = run & (s['Sowing'] >= 1) & (s['Emergence'] == 1)
        grow = G & (s['st_Development'] <= self.p['DevelopStageHarvest']) & \
            (s['CultivateDay'] < self.cycle)
        harv = G & ~grow
        w = dict(s)
        if self._ringip():
            w['__lmask'] = grow
        w = self._grow(w, cal, t)
        w.pop('__lmask', None)
        s = self._merge(s, w, grow)
        zi = torch.zeros_like(s['Sowing'])
        for k in ('cultivating', 'growing', 'CultivateDay', 'Emergence',
                  'Sowing', 'lh', 'lt'):
            s[k] = torch.where(harv, zi, s[k])
        for k in ('TSumEmergence', 'st_LAI'):
            s[k] = torch.where(harv, torch.zeros_like(s[k]), s[k])
        return s

    def _merge(self, s, w, mask):
        out = dict(s)
        mu = None
        for k, v in w.items():
            if k.startswith('__'):
                continue
            if v is not s.get(k):
                if k in LEAF and v.dim() == 2:
                    if mu is None:
                        mu = self._cl(mask, True)
                    out[k] = _where(mu, v, s[k])
                else:
                    out[k] = _where(mask, v, s[k])
        return out

    # ------------------------------------------------------------ init
    def _initialize(self, w, cal, t):
        """InitializeCrop.c:43-104 and InitializeNutrients.c:9-124."""
        p, S, D = self.p, self.S, self.D
        dvs = p['InitialDVS'] + torch.zeros_like(w['st_Development'])
        w['st_Development'] = dvs
        fr = self.afgen('Roots', dvs)
        fs = 1 - fr
        isw = p['InitialDryWeight'] * fs
        w['st_roots'] = p['InitialDryWeight'] * fr
        w['st_RootDepth'] = p['InitRootingDepth'] + torch.zeros_like(dvs)
        w['st_stems'] = isw * self.afgen('Stems', dvs)
        w['st_leaves'] = isw * self.afgen('Leaves', dvs)
        w['st_storage'] = isw * self.afgen('Storage', dvs)
        w['prm_MaxRootingDepth'] = _cmax(
            p['InitRootingDepth'].expand_as(dvs), w['prm_MaxRootingDepth'])
        sla = self.afgen('SpecificLeaveArea', dvs)
        laiem = w['st_leaves'] * sla
        w['st_LAIExp'] = laiem
        w['st_LAI'] = laiem + w['st_stems'] * \
            self.afgen('SpecificStemArea', dvs) + \
            w['st_storage'] * p['SpecificPodArea']
        first = (self._slots(w) == 0)
        c = self._lanes(w)
        if '__lmask' in w:
            # in place, the emerging units only (_ringip)
            from . import leaf_kernels as LK
            LK.ring_put_((w['lw'], w['la'], w['lage']),
                         torch.zeros_like(self._cl(w['lh'], c)),
                         self._cl(w['__lmask'], c),
                         (self._cl(w['st_leaves'], c), self._cl(sla, c), 0.))
        else:
            w['lw'] = torch.where(first, self._cl(w['st_leaves'],
                                                  c)[..., None], w['lw'])
            w['la'] = torch.where(first, self._cl(sla, c)[..., None],
                                  w['la'])
            w['lage'] = torch.where(first, torch.zeros_like(w['lage']),
                                    w['lage'])
        w['lh'] = torch.zeros_like(w['lh'])
        w['lt'] = torch.ones_like(w['lt'])
        w['Emergence'] = torch.ones_like(w['Emergence'])
        w['GrowthDay'] = torch.ones_like(w['GrowthDay'])
        for k in ('roots', 'stems', 'leaves'):
            w['dst_' + k] = torch.zeros_like(dvs)
        w['st_vernalization'] = torch.zeros_like(dvs)
        # InitializeNutrients
        z = torch.zeros_like(dvs)
        cd = w['CultivateDay'].to(torch.int64)
        for j, X in enumerate(NUT):
            mlv = self.afgen('%s_MaxLeaves' % X, dvs)
            w['%s_st_Max_lv' % X] = mlv
            w['%s_st_Max_st' % X] = p['%s_MaxStems' % X] * mlv
            w['%s_st_Max_ro' % X] = p['%s_MaxRoots' % X] * mlv
            w['%s_st_Max_so' % X] = z
            w['%s_st_leaves' % X] = mlv * w['st_leaves']
            w['%s_st_stems' % X] = w['%s_st_Max_st' % X] * w['st_stems']
            w['%s_st_roots' % X] = w['%s_st_Max_ro' % X] * w['st_roots']
            w['%s_st_storage' % X] = z
            for k in ('death_lv', 'death_st', 'death_ro', 'Demand',
                      'Uptake'):
                w['%s_st_%s' % (X, k)] = z
            # List_cumsum(table, CultivateDay) * Uptake_frac
            lc = self._list_cumsum(cal, t, j, cd)
            w['ste_st_%s_tot' % X] = lc * self.m['%s_Uptake_frac' % X]
            w['ste_st_%s_mins' % X] = w['mng_%s_Mins' % X]
            w['%s_st_Indx' % X] = z + 1
        w['NPK_Indx'] = z + 1
        w['N_st_Optimum_lv'] = z
        w['N_st_Optimum_st'] = z
        return w

    def _list_cumsum(self, cal, t, j, n):
        """List.c:26-50: sum of table j over steps t-n+1 .. t -> [N,B,NC]."""
        pre = cal.fert_prefix[:, :, j]                    # [T+1, NC]
        hi = _at(pre, t + 1)                              # [NC]
        lo_idx = (t + 1 - n).clamp(min=0)                 # [N,B,NC]
        if self._lanemode:
            fp = cal.fert_prefix
            J = fp.shape[2]
            lo = fp.take((lo_idx * fp.shape[1] + self._lcls.view(
                self.shape)) * J + j)
            return self.S(self._cls(hi) - lo)
        lo = pre.t().expand(*n.shape[:2], *pre.t().shape).gather(
            -1, lo_idx[..., None])[..., 0]
        return self.S(hi - lo)

    # ------------------------------------------------------------ growth
    def _grow(self, w, cal, t):
        """One growth day (wofost_run.c:28-56) on all lanes."""
        self._astro(w)
        self._rates_to_zero(w)
        self._partioning(w)
        self._rate_nutrients(w, cal, t)
        self._rate_crop(w)
        w['st_LAI'] = self._lai(w)
        self._integration_crop(w)
        self._integration_nutrients(w)
        w['CultivateDay'] = w['CultivateDay'] + 1
        w['GrowthDay'] = w['GrowthDay'] + 1
        for X in NUT:
            w['mng_%s_external' % X] = torch.zeros_like(
                w['mng_%s_external' % X])
        return w

    def _astro(self, w):
        """Astro.c:13-103."""
        D, S = self.D, self.S
        wd = self.wd
        day = S(w['met_MeteoDay'].to(self.dt))           # float day_fl
        dd = D(day)
        lat = self.lat
        bad = (lat.abs() > 90.).expand_as(day)
        w['err'] = w['err'] | torch.where(bad, ERR_LAT, 0).to(torch.int32)
        decl = S(-torch.asin(math.sin(23.45 * RAD_PER_DEG) *
                             torch.cos(2. * CONST_PI * (dd + 10.) / 365.)))
        sc = S(1370. * (1. + 0.033 * torch.cos(2. * CONST_PI * (dd) / 365.)))
        latr = RAD_PER_DEG * D(lat)
        sinld = S(torch.sin(latr) * torch.sin(D(decl)))
        cosld = S(torch.cos(latr) * torch.cos(D(decl)))
        aob = sinld / cosld
        tmp = torch.where(1 < aob, torch.ones_like(aob), aob)
        aob = torch.where(-1 > tmp, -torch.ones_like(tmp), tmp)
        e = 12.0 * (1. + 2. * torch.asin(D(aob)) / CONST_PI)
        tmp = S(torch.where(24. < e, torch.full_like(e, 24.), e))
        dl = torch.where(0 > tmp, torch.zeros_like(tmp), tmp)
        aob2 = S((-math.sin(-4.0 * RAD_PER_DEG) + D(sinld)) / D(cosld))
        tmp = torch.where(1 < aob2, torch.ones_like(aob2), aob2)
        aob2 = torch.where(-1 > tmp, -torch.ones_like(tmp), tmp)
        e = 12.0 * (1. + 2. * torch.asin(D(aob2)) / CONST_PI)
        tmp = S(torch.where(24. < e, torch.full_like(e, 24.), e))
        dlp = torch.where(0 > tmp, torch.zeros_like(tmp), tmp)
        sq = torch.sqrt(1. - D(aob * aob))
        inner = D(aob) <= 1.0
        dsinb = S(torch.where(
            inner, 3600. * (D(dl * sinld) + (24. / CONST_PI) * D(cosld) * sq),
            3600. * D(dl * sinld)))
        t1 = D(dl) * (D(sinld) + 0.4 * (D(sinld * sinld) +
                                        D(cosld * cosld) * 0.5))
        t2 = 12. * D(cosld) * (2. + 3. * 0.4 * D(sinld)) * sq / CONST_PI
        dsinbe = S(torch.where(inner, 3600. * (t1 + t2), 3600. * t1))
        angot = sc * dsinb
        atm = w['met_Radiation'] / angot
        a = D(atm)
        fdr = torch.ones_like(a)
        fdr = torch.where(a > 0.75, torch.full_like(a, self.c023), fdr)
        fdr = torch.where((a <= 0.75) & (a > 0.35), 1.33 - 1.46 * a, fdr)
        fdr = torch.where((a <= 0.35) & (a > 0.07),
                          1. - 2.3 * ((a - 0.07) * (a - 0.07)), fdr)
        fdr = S(fdr)
        w['met_DiffRadPP'] = S(0.5 * D(fdr) * a * D(sc))
        w['met_SinLD'] = sinld
        w['met_CosLD'] = cosld
        w['met_Daylength'] = dl
        w['met_PARDaylength'] = dlp
        w['met_DSinBE'] = dsinbe
        w['met_AngotRadiation'] = angot
        w['met_AtmosphTransm'] = atm

    def _rates_to_zero(self, w):
        """RatesToZero.c:9-122 (the crop and nutrient part)."""
        z = torch.zeros_like(w['st_roots'])
        for k in ('roots', 'leaves', 'stems'):
            w['drt_' + k] = z
        for k in GROWTH_RT:
            w['rt_' + k] = z
        for X in NUT:
            for k in ('death_lv death_st death_ro leaves stems storage roots '
                      'Demand Demand_lv Demand_st Demand_ro Demand_so '
                      'Transloc Uptake Uptake_lv Uptake_st Uptake_ro'
                      ).split():
                w['%s_rt_%s' % (X, k)] = z
            w['ste_rt_%s_mins' % X] = z
            w['ste_rt_%s_tot' % X] = z

    def _partioning(self, w):
        """DM-Partioning.c:9-41."""
        D, S, p = self.D, self.S, self.p
        dvs = w['st_Development']
        ws = w['soil_WaterStress']
        nindx = w['N_st_Indx']
        c = ws < nindx
        f = 1. / (D(ws) + 0.5)
        factor = S(torch.where(1. > f, torch.ones_like(f), f))
        fr = self.afgen('Roots', dvs)
        flv = self.afgen('Leaves', dvs)
        fst = self.afgen('Stems', dvs)
        fso = self.afgen('Storage', dvs)
        x = D(fr * factor)
        fro1 = S(torch.where(0.6 < x, torch.full_like(x, 0.6), x))
        fac2 = S(torch.exp(D(-p['N_lv_partitioning']) * (1. - D(nindx))))
        flv2 = flv * fac2
        w['fac_ro'] = torch.where(c, fro1, fr)
        w['fac_lv'] = torch.where(c, flv, flv2)
        w['fac_st'] = torch.where(c, fst, fst + flv - flv2)
        w['fac_so'] = fso

    # ------------------------------------------------------------ nutrients
    def _rate_nutrients(self, w, cal, t):
        """Nutrients.c:116-140 RateCalcultionNutrients."""
        self._nutrient_max(w)
        self._nutrient_demand(w)
        self._nutrient_optimum(w)
        self._nutrient_loss(w)
        self._soil_nutrient_rates(w, cal, t)
        self._nutrient_partioning(w)
        self._nutrient_translocation(w)
        self._crop_nutrient_rates(w)
        self._soil_nutrient_rates(w, cal, t)
        self._nutrition_indx(w)

    def _nutrient_max(self, w):
        """Nutrients.c:9-36."""
        p = self.p
        dvs = w['st_Development']
        for X in NUT:
            mlv = self.afgen('%s_MaxLeaves' % X, dvs)
            w['%s_st_Max_lv' % X] = mlv
            w['%s_st_Max_st' % X] = p['%s_MaxStems' % X] * mlv
            w['%s_st_Max_ro' % X] = p['%s_MaxRoots' % X] * mlv
            w['%s_st_Max_so' % X] = p['Max_%s_storage' % X] + \
                torch.zeros_like(mlv)

    def _nutrient_demand(self, w):
        """Nutrients.c:68-114."""
        tc = {'N': 'TCNT', 'P': 'TCPT', 'K': 'TCKT'}
        zero = torch.zeros_like(w['st_roots'])
        for X in NUT:
            for org, st in (('lv', 'leaves'), ('st', 'stems'),
                            ('ro', 'roots')):
                x = w['%s_st_Max_%s' % (X, org)] * w['st_' + st] - \
                    w['%s_st_%s' % (X, st)]
                w['%s_rt_Demand_%s' % (X, org)] = _cmax(x, zero)
            x = w['%s_st_Max_so' % X] * w['st_storage'] - \
                w['%s_st_storage' % X]
            w['%s_rt_Demand_so' % X] = _cmax(x, zero) / self.p[tc[X]]

    def _nutrient_optimum(self, w):
        """Nutrients.c:43-62."""
        p = self.p
        for X in NUT:
            of = p['Opt_%s_Frac' % X]
            w['%s_st_Optimum_lv' % X] = of * w['%s_st_Max_lv' % X] * \
                w['st_leaves']
            w['%s_st_Optimum_st' % X] = of * w['%s_st_Max_st' % X] * \
                w['st_stems']

    def _nutrient_loss(self, w):
        """NutrientLoss.c:10-33."""
        p = self.p
        for X in NUT:
            for org, st in (('lv', 'leaves'), ('st', 'stems'),
                            ('ro', 'roots')):
                w['%s_rt_death_%s' % (X, org)] = \
                    p['%s_ResidualFrac_%s' % (X, org)] * w['drt_' + st]

    def _soil_nutrient_rates(self, w, cal, t):
        """SoilNutrientRates.c:9-50 (wofost_steps_per_day 1)."""
        D = self.D
        dvs = w['st_Development']
        c = (D(dvs) > 0.) & (dvs <= self.p['DevelopmentStageNLimit'])
        rec = {'N': 'NRecoveryFrac', 'P': 'PRecoveryFrac',
               'K': 'KRecoveryFrac'}
        zero = torch.zeros_like(dvs)
        for j, X in enumerate(NUT):
            mr = _cmin(w['mng_%s_Mins' % X] * w['mng_' + rec[X]],
                       w['ste_st_%s_mins' % X])
            w['ste_rt_%s_mins' % X] = torch.where(c, mr, zero)
            fert = self._cls(_at(cal.fert_list, t)[:, j]) * \
                self.m['%s_Uptake_frac' % X]
            w['ste_rt_%s_tot' % X] = fert / 1.0 - w['%s_rt_Uptake' % X] + \
                w['ste_rt_%s_mins' % X] + w['mng_%s_external' % X]

    def _nutrient_partioning(self, w):
        """NutrientPartitioning.c:9-130 (wofost_steps_per_day 1)."""
        D, S, p = self.D, self.S, self.p
        dvs = w['st_Development']
        zero = torch.zeros_like(dvs)
        for X in NUT:
            w['%s_rt_Demand' % X] = w['%s_rt_Demand_lv' % X] + \
                w['%s_rt_Demand_st' % X] + w['%s_rt_Demand_ro' % X]
        lim = (dvs < p['DevelopmentStageNLimit']) & \
            (D(w['soil_WaterStress']) > 0.01)
        nlim = torch.where(lim, zero + 1, zero)
        x = p['N_fixation'] * w['N_rt_Demand']
        nfix = _cmax(x, zero) * nlim
        for X in NUT:
            dem = w['%s_rt_Demand' % X]
            if X == 'N':
                dem = dem - nfix
            if self.opt['PFERT']:
                tmin = dem
            else:
                tmin = _cmin(dem, w['ste_st_%s_tot' % X] +
                             w['ste_rt_%s_tot' % X])
            w['%s_rt_Uptake' % X] = _cmax(tmin, zero) * nlim / 1.0
        for X in NUT:
            dem = w['%s_rt_Demand' % X]
            big = dem > 0.001
            up = w['%s_rt_Uptake' % X]
            if X == 'N':
                up = up + nfix
            for org in ('lv', 'st', 'ro'):
                v = (w['%s_rt_Demand_%s' % (X, org)] / dem) * up
                w['%s_rt_Uptake_%s' % (X, org)] = torch.where(big, v, zero)

    def _nutrient_translocation(self, w):
        """NutrientTranslocation.c:9-79."""
        p = self.p
        zero = torch.zeros_like(w['st_roots'])
        tc = {'N': 'TCNT', 'P': 'TCPT', 'K': 'TCKT'}
        dvs = w['st_Development']
        nt = dvs > p['DevelopmentStageNT']
        for X in NUT:
            alv = _cmax(zero, w['%s_st_leaves' % X] - w['st_leaves'] *
                        p['%s_ResidualFrac_lv' % X])
            ast = _cmax(zero, w['%s_st_stems' % X] - w['st_stems'] *
                        p['%s_ResidualFrac_st' % X])
            aro = _cmax((alv + ast) * p['FracTranslocRoots'],
                        w['%s_st_roots' % X] - w['st_roots'] *
                        p['%s_ResidualFrac_ro' % X])
            w['%s_st_Avail_lv' % X] = alv
            w['%s_st_Avail_st' % X] = ast
            w['%s_st_Avail_ro' % X] = aro
            w['%s_st_Avail' % X] = alv + ast + aro
            w['%s_rt_Supply' % X] = torch.where(
                nt, (alv + ast + aro) / p[tc[X]], zero)

    def _crop_nutrient_rates(self, w):
        """CropNutrientRates.c:10-111."""
        D = self.D
        zero = torch.zeros_like(w['st_roots'])
        nt = w['st_Development'] > self.p['DevelopmentStageNT']
        for X in NUT:
            for org, st in (('lv', 'leaves'), ('st', 'stems'),
                            ('ro', 'roots')):
                w['%s_rt_%s' % (X, st)] = w['%s_rt_Uptake_%s' % (X, org)] - \
                    w['%s_rt_Transloc_%s' % (X, org)] - \
                    w['%s_rt_death_%s' % (X, org)]
            w['%s_rt_storage' % X] = torch.where(
                nt, _cmin(w['%s_rt_Demand_so' % X], w['%s_rt_Supply' % X]),
                w['%s_rt_storage' % X])
        for X in NUT:
            av = w['%s_st_Avail_lv' % X] + w['%s_st_Avail_st' % X] + \
                w['%s_st_Avail_ro' % X]
            big = D(av) > 0.001
            for org in ('lv', 'st', 'ro'):
                v = w['%s_rt_storage' % X] * w['%s_st_Avail_%s' % (X, org)] \
                    / av
                w['%s_rt_Transloc_%s' % (X, org)] = torch.where(big, v, zero)

    def _nutrition_indx(self, w):
        """NutritionIndex.c:8-110."""
        D, S, p = self.D, self.S, self.p
        lv, st = w['st_leaves'], w['st_stems']
        vm = lv + st
        pos = D(vm) > 0.
        zero = torch.zeros_like(vm)
        tiny = 0.001
        idx = {}
        for X in NUT:
            veg = (w['%s_st_leaves' % X] + w['%s_st_stems' % X] +
                   w['%s_rt_Uptake_lv' % X] + w['%s_rt_Uptake_st' % X]) / vm
            res = (lv * p['%s_ResidualFrac_lv' % X] +
                   st * p['%s_ResidualFrac_st' % X]) / vm
            opt = (w['%s_st_Optimum_lv' % X] + w['%s_st_Optimum_st' % X]) \
                / vm
            veg = torch.where(pos, veg, zero)
            res = torch.where(pos, res, zero)
            opt = torch.where(pos, opt, zero)
            c = (opt - res) > tiny
            v = self.limit(tiny, 1.0, (veg - res) / (opt - res))
            idx[X] = torch.where(c, v, torch.full_like(v, tiny))
            w['%s_st_Indx' % X] = idx[X]
        npk = torch.where(idx['N'] < idx['P'], idx['N'], idx['P'])
        npk = torch.where(npk < idx['K'], npk, idx['K'])
        w['NPK_Indx'] = npk
        q = 1.0001 - D(npk)
        w['NutrientStress'] = self.limit(0., 1.0, 1. - D(p['NLUE']) *
                                         (q * q))

    # ------------------------------------------------------------ crop rates
    def _rate_crop(self, w):
        """RateCalculationCrop.c:11-44."""
        self._development_rate(w)
        gross = self._daily_total_assimilation(w)
        stress = _cmin(w['NutrientStress'], w['soil_WaterStress'])
        total = stress * self._correct(w, gross)
        maint = self._respiration(w, total)
        gg = self._conversion(w, total - maint)
        self._growth(w, gg)
        self._nutrient_loss(w)
        self._crop_nutrient_rates(w)

    def _development_rate(self, w):
        """Develop.c:10-52."""
        D, S, p = self.D, self.S, self.p
        dvs = w['st_Development']
        temp = w['met_Temp']
        dtsm = self.afgen('DeltaTempSum', temp)
        r1 = dtsm / w['prm_TempSum1']
        dl = self.limit(0., 1., (w['met_PARDaylength'] -
                                 p['CriticalDaylength']) /
                        (p['OptimumDaylength'] - p['CriticalDaylength']))
        idsl = self.idsl
        r1 = torch.where((idsl == 1) | (idsl == 2), r1 * dl, r1)
        x1 = S(D(dvs) - 0.3)
        vr = self.insw(x1, self.afgen('VernalizationRate', temp),
                       torch.zeros_like(dvs))
        vf = self.limit(0., 1., (w['st_vernalization'] -
                                 p['BaseVernRequirement']) /
                        (p['SatVernRequirement'] - p['BaseVernRequirement']))
        r1v = r1 * self.insw(x1, vf, torch.ones_like(vf))
        r1 = torch.where(idsl == 2, r1v, r1)
        lt1 = D(dvs) < 1.
        w['rt_Development'] = torch.where(lt1, r1,
                                          dtsm / w['prm_TempSum2'])
        w['rt_vernalization'] = torch.where(lt1 & (idsl == 2), vr,
                                            w['rt_vernalization'])

    def _daily_total_assimilation(self, w):
        """Assim.c:90-144."""
        D, S, p = self.D, self.S, self.p
        dvs, lai = w['st_Development'], w['st_LAI']
        dayt, co2 = w['met_DayTemp'], w['met_CO2']
        kdif = self.afgen('KDiffuseTb', dvs)
        eff = self.afgen('EFFTb', dayt) * self.afgen('CO2EFFTB', co2)
        ts = self.afgen('FactorAssimRateTemp', dayt)
        if self.opt['PTEMP']:
            ts = torch.where(dayt > self.mopt, torch.ones_like(ts), ts)
        w['met_TempStress'] = ts
        am = ts * self.afgen('MaxAssimRate', dvs) * \
            self.afgen('CO2AMAXTB', co2)
        dl = w['met_Daylength']
        pos = (D(am) > 0.) & (D(lai) > 0.) & (dl > 0)
        dta = torch.zeros_like(dvs)
        for i in range(3):
            hour = S(12.0 + 0.5 * D(dl) * self.xg[i])
            e = D(w['met_SinLD']) + D(w['met_CosLD']) * \
                torch.cos(2. * CONST_PI * (D(hour) + 12.) / 24.)
            sinb = S(torch.where(0. > e, torch.zeros_like(e), e))
            par = S(0.5 * D(w['met_Radiation']) * D(sinb) *
                    (1. + 0.4 * D(sinb)) / D(w['met_DSinBE']))
            pardif = _cmin(par, sinb * w['met_DiffRadPP'])
            pardir = par - pardif
            inst = self._instant_assimilation(lai, kdif, eff, am, sinb,
                                              pardif, pardir)
            dta = torch.where(pos, dta + inst * self.wg[i], dta)
        return dta * dl

    def _instant_assimilation(self, lai, kdif, eff, am, sinb, pardif,
                              pardir):
        """Assim.c:11-83."""
        D, S = self.D, self.S
        sq = self.sq_scat
        refl = S((1. - sq) / (1. + sq) * (2 / (1 + 1.6 * D(sinb))))
        kbl = S((0.5 / D(sinb)) * D(kdif) / (0.8 * sq))
        ktl = S(D(kbl) * sq)
        gross = torch.zeros_like(lai)
        tmax = S(torch.where(2.0 > D(am), torch.full_like(D(am), 2.0),
                             D(am)))
        adl = self.one_m_scat_f * pardir / sinb
        for i in range(3):
            laic = lai * self.xg[i]
            ardif = S((1. - D(refl)) * D(pardif) * D(kdif) *
                      torch.exp(D(-kdif * laic)))
            artot = S((1. - D(refl)) * D(pardir) * D(ktl) *
                      torch.exp(D(-ktl * laic)))
            ardir = S((1. - self.scat) * D(pardir) * D(kbl) *
                      torch.exp(D(-kbl * laic)))
            asl = ardif + artot - ardir
            ash = S(D(am) * (1. - torch.exp(D(-asl * eff / tmax))))
            sun = S(D(am) * (1. - D(am - ash) *
                             (1 - torch.exp(D(-adl * eff / tmax))) /
                             D(eff * adl)))
            sun = torch.where(adl <= 0, ash, sun)
            fsl = S(torch.exp(D(-kbl * laic)))
            atot = S(D(fsl * sun) + (1. - D(fsl)) * D(ash))
            gross = gross + atot * self.wg[i]
        return gross * lai

    def _correct(self, w, assim):
        """Assim.c:150-173."""
        D, S = self.D, self.S
        gd = w['GrowthDay']
        number = torch.where(gd < DAYS_PER_WEEK - 1, gd,
                             torch.full_like(gd, DAYS_PER_WEEK))
        v = torch.zeros_like(assim)
        tmin = w['met_Tmin']
        for c in range(DAYS_PER_WEEK):
            v = torch.where(c < number, S(D(v) + tmin[..., c]), v)
        avg = v / number.to(self.dt)
        ts = self.afgen('FactorGrossAssimTemp', avg)
        w['met_TminStress'] = ts
        return S(D(assim * ts) * 30. / 44.)

    def _respiration(self, w, total):
        """Respiration.c:10-29."""
        D, S, p = self.D, self.S, self.p
        r = p['RelRespiLeaves'] * w['st_leaves']
        r = r + p['RelRespiStorage'] * w['st_storage']
        r = r + p['RelRespiRoots'] * w['st_roots']
        r = r + p['RelRespiStems'] * w['st_stems']
        r = r * self.afgen('FactorSenescence', w['st_Development'])
        r = S(D(r) * torch.pow(D(p['Q10']) + torch.zeros_like(D(r)),
                               0.1 * D(w['met_Temp'] - 25.)))
        return _cmin(r, total)

    def _conversion(self, w, net):
        """Conversion.c:9-22."""
        p = self.p
        root = w['fac_ro'] / p['ConversionRoots']
        shoots = w['fac_st'] / p['ConversionStems']
        shoots = shoots + w['fac_lv'] / p['ConversionLeaves']
        shoots = shoots + w['fac_so'] / p['ConversionStorage']
        return net / (shoots * (1 - w['fac_ro']) + root)

    def _growth(self, w, newpm):
        """Growth.c:10-60 (wofost_steps_per_day 1)."""
        D, p = self.D, self.p
        dvs = w['st_Development']
        w['drt_roots'] = w['st_roots'] * self.afgen('DeathRateRoots', dvs)
        w['rt_roots'] = newpm * w['fac_ro'] - w['drt_roots']
        shoots = newpm * (1 - w['fac_ro'])
        w['drt_stems'] = w['st_stems'] * self.afgen('DeathRateStems', dvs)
        w['rt_stems'] = shoots * w['fac_st'] - w['drt_stems']
        trl = torch.where(D(dvs) >= 1., w['drt_stems'] *
                          p['TranslocationFrac'], torch.zeros_like(dvs))
        w['rt_storage'] = shoots * w['fac_so'] + trl
        w['drt_leaves'] = self._dying_leaves(w)
        w['rt_leaves'] = shoots * w['fac_lv']
        self._leave_growth(w)
        w['rt_leaves'] = w['rt_leaves'] - w['drt_leaves']
        rd = _cmin(w['prm_MaxRootingDepth'] - w['st_RootDepth'],
                   (p['MaxIncreaseRoot'] * 1.0).expand_as(dvs))
        w['rt_RootDepth'] = torch.where(w['fac_ro'] <= 0.0,
                                        torch.zeros_like(rd), rd)

    # ------------------------------------------------------------ leaves
    @staticmethod
    def _slots(w):
        """Slot indices of the state's leaf classes [R]."""
        return torch.arange(w['lw'].shape[-1], device=w['lw'].device)

    def _rlive(self, w, h, t):
        """Slots holding a class of [h, t) ([U] each; ring of R slots,
        t - h <= R): (live [U, R], logical class of each slot [U, R])."""
        j = self._slots(w)
        R = j.shape[0]
        k = h[:, None] + (j - h[:, None]) % R
        return k < t[:, None], k

    def _live(self, h, t):
        k = self.kidx
        return (k >= h[..., None].long()) & (k < t[..., None].long())

    def _dying_leaves(self, w):
        """Leaves.c:10-66.  Leaf classes are lw/la/lage[..., k] for
        lh <= k < lt, oldest first (the C list head)."""
        D, S, p = self.D, self.S, self.p
        dvs, lv = w['st_Development'], w['st_leaves']
        crit = p['CritLAIFactor'] / self.afgen('KDiffuseTb', dvs)
        d1 = S(D(lv) * (1. - D(w['soil_WaterStress'])) *
               D(p['MaxRelDeathRate']))
        c = 0.03 * D(w['st_LAI'] - crit) / D(crit)
        d2 = lv * self.limit(0., self.c003, c)
        death = _cmax(d1, d2)
        death = S(D(death) + D(lv * p['DyingLeaves_NPK_Stress']) *
                  (1. - D(w['NPK_Indx'])))
        death = torch.where(death < 0.001, torch.zeros_like(death), death)
        dstress = death
        h, t = w['lh'].long(), w['lt'].long()
        c = self._lanes(w)
        hu, tu, du = self._cl(h, c), self._cl(t, c), self._cl(death, c)
        if self._lanemode and self._xr:
            return dstress + self._dying_ring(w, du, h, hu, tu, c)
        live = self._live(hu, tu)
        lw = w['lw']
        if self.exact_sums:
            return dstress + self._dying_exact(w, du, hu, tu, live, h, c)
        lw64 = torch.where(live, lw.to(torch.float64), torch.zeros_like(
            lw, dtype=torch.float64))
        cs = _eager_cumsum(lw64)
        del lw64                       # [.., K] float64: freed at once
        d64 = du.to(torch.float64)
        m = (live & (d64[..., None] > cs)).sum(-1)
        n = tu - hu
        gone = self._dn((n > 0) & (m >= n), c, torch.zeros_like(
            h, dtype=torch.bool))
        w['err'] = w['err'] | torch.where(gone, ERR_NOLEAF, 0).to(torch.int32)
        h1 = hu + m
        csm = cs.gather(-1, (h1 - 1).clamp(0, self.K - 1)[..., None])[..., 0]
        del cs
        dr = torch.where(m > 0, S(d64 - csm), du)
        at_h1 = (self.kidx == h1[..., None]) & live
        lw = torch.where(at_h1, lw - dr[..., None], lw)
        live1 = self._live(h1, tu)
        notold = live1 & ~(w['lage'] > self._pk('LifeSpan', c))
        # the first slot that is live and not too old (K if none): the
        # first maximum of the mask (argmax returns the first), not a
        # [.., K] int64 table
        fb = torch.where(notold.any(-1),
                         notold.to(torch.uint8).argmax(-1),
                         torch.full_like(tu, self.K))
        del notold
        fb = torch.where(fb >= self.K, tu, fb)
        aged = live1 & (self.kidx < fb[..., None])
        dage = S(_eager_sum(torch.where(aged, lw.to(torch.float64),
                                        torch.zeros_like(
                                            lw, dtype=torch.float64))))
        w['lw'] = lw
        w['lh'] = self._dn(fb, c, h).to(w['lh'].dtype)
        return dstress + self._dn(dage, c, torch.zeros_like(dstress))

    def _dying_ring(self, w, du, h, hu, tu, c):
        """_dying_leaves' classes on the ring of the lanes: the same
        values, the sums as eager's trees on the live classes."""
        from . import leaf_kernels as LK
        S = self.S
        lw = w['lw']
        R = lw.shape[-1]
        d64 = du.to(torch.float64)
        m, csm = LK.ring_scan(lw, hu, tu, d64, self._ch)
        n = tu - hu
        gone = self._dn((n > 0) & (m >= n), c, torch.zeros_like(
            h, dtype=torch.bool))
        w['err'] = w['err'] | torch.where(gone, ERR_NOLEAF, 0).to(torch.int32)
        h1 = hu + m
        dr = torch.where(m > 0, S(d64 - csm), du)
        if '__lmask' in w:
            # in place, the growing units only
            LK.ring_put_((lw,), h1 % R, self._cl(w['__lmask'], c) & (h1 < tu),
                         (('sub', dr),))
            fb = LK.ring_young(w['lage'], h1, tu,
                               self._pk('LifeSpan', c).reshape(-1))
        else:
            j = self._slots(w)
            at_h1 = (j == (h1 % R)[:, None]) & (h1 < tu)[:, None]
            lw = torch.where(at_h1, lw - dr[:, None], lw)
            live1, k = self._rlive(w, h1, tu)
            notold = live1 & ~(w['lage'] > self._pk('LifeSpan', c))
            fb = torch.where(notold, k, tu[:, None]).amin(-1)
        dage = S(LK.ring_sum(lw, h1, fb, self.K))
        w['lw'] = lw
        w['lh'] = self._dn(fb, c, h).to(w['lh'].dtype)
        return self._dn(dage, c, torch.zeros_like(w['st_leaves']))

    def _dying_exact(self, w, death, h, t, live, h_units, c):
        """Leaves.c:39-62 slot by slot: the stress loop subtracts whole
        classes from the head while Death > weight, then the head class
        loses the rest; then the classes older than SPAN are dropped from
        the head and summed (DeathAge)."""
        lw = w['lw']
        d = death
        h1 = h
        stop = torch.zeros_like(live[..., 0])
        for k in range(self.K):
            lk = live[..., k] & ~stop
            wk = lw[..., k]
            take = lk & (d > wk)
            d = torch.where(take, d - wk, d)
            stop = stop | (lk & ~take)
            h1 = torch.where(take, torch.full_like(h1, k + 1), h1)
        gone = self._dn((t > h) & ~stop, c, torch.zeros_like(
            h_units, dtype=torch.bool))
        w['err'] = w['err'] | torch.where(gone, ERR_NOLEAF, 0).to(torch.int32)
        at_h1 = (self.kidx == h1[..., None]) & live
        lw = torch.where(at_h1, lw - d[..., None], lw)
        live1 = self._live(h1, t)
        span = self._pk('LifeSpan', c)[..., 0]
        dage = torch.zeros_like(d)
        stop = torch.zeros_like(stop)
        h2 = h1
        for k in range(self.K):
            lk = live1[..., k] & ~stop
            old = lk & (w['lage'][..., k] > span)
            dage = torch.where(old, dage + lw[..., k], dage)
            stop = stop | (lk & ~old)
            h2 = torch.where(old, torch.full_like(h2, k + 1), h2)
        w['lw'] = lw
        w['lh'] = self._dn(h2, c, h_units).to(w['lh'].dtype)
        return self._dn(dage, c, torch.zeros_like(w['st_leaves']))

    def _leave_growth(self, w):
        """LeaveGrowth.c:10-80."""
        D, S, p = self.D, self.S, self.p
        dvs, npk = w['st_Development'], w['NPK_Indx']
        sla = S(D(self.afgen('SpecificLeaveArea', dvs)) *
                torch.exp(D(-p['NutrientStessSLA']) * (1. - D(npk))))
        rtl = w['rt_leaves']
        c = (w['st_LAIExp'] < 6) & (rtl > 0.)
        early = (D(dvs) < 0.2) & (D(w['st_LAI']) < 0.75)
        st1 = S(D(w['soil_WaterStress']) *
                torch.exp(D(-p['NitrogenStressLAI']) * (1. - D(npk))))
        stress = torch.where(early, st1, torch.ones_like(st1))
        x = w['met_Temp'] - p['TempBaseLeaves']
        dteff = torch.where(0. > x, torch.zeros_like(x), x)
        gel = w['st_LAIExp'] * p['RelIncreaseLAI'] * dteff * stress
        gsl = rtl * sla
        sla2 = _cmin(gel, gsl) / rtl
        sla = torch.where(c, sla2, sla)
        gel = torch.where(c, gel, torch.zeros_like(gel))
        h, t = w['lh'].long(), w['lt'].long()
        empty = t <= h
        R = w['lw'].shape[-1]
        # a class beyond the row (or the ring: R = K is the row)
        over = (t >= self.K) | (t - h >= R)
        w['err'] = w['err'] | torch.where(empty, ERR_NOLEAF, 0).to(
            torch.int32) | torch.where(over, ERR_RING, 0).to(torch.int32)
        c = self._lanes(w)
        if '__lmask' in w:
            from . import leaf_kernels as LK
            LK.ring_put_((w['lw'], w['la'], w['lage']), self._cl(t % R, c),
                         self._cl(w['__lmask'] & ~over, c),
                         (self._cl(rtl, c), self._cl(sla, c), 0.))
        else:
            at = (self._slots(w) == self._cl(t % R, c)[..., None]) & \
                ~self._cl(over, c)[..., None]
            w['lw'] = torch.where(at, self._cl(rtl, c)[..., None], w['lw'])
            w['la'] = torch.where(at, self._cl(sla, c)[..., None], w['la'])
            w['lage'] = torch.where(at, torch.zeros_like(w['lage']),
                                    w['lage'])
        w['lt'] = torch.where(over, w['lt'], w['lt'] + 1)
        w['rt_LAIExp'] = gel

    def _lai(self, w):
        """Lai.c:8-32."""
        c = self._lanes(w)
        if self._lanemode and self._xr:
            from . import leaf_kernels as LK
            s = self.S(LK.ring_sum(w['lw'], self._cl(w['lh'].long(), c),
                                   self._cl(w['lt'].long(), c), self.K,
                                   b=w['la']))
            s = self._dn(s, c, torch.zeros_like(w['st_LAI']))
            dvs = w['st_Development']
            return s + w['st_stems'] * self.afgen('SpecificStemArea',
                                                  dvs) + \
                w['st_storage'] * self.p['SpecificPodArea']
        live = self._live(self._cl(w['lh'].long(), c),
                          self._cl(w['lt'].long(), c))
        prod = w['lw'] * w['la']
        if self.exact_sums:
            s = torch.zeros_like(self._cl(w['st_LAI'], c))
            for k in range(self.K):
                s = torch.where(live[..., k], s + prod[..., k], s)
        else:
            s = self.S(_eager_sum(torch.where(
                live, prod.to(torch.float64),
                torch.zeros_like(prod, dtype=torch.float64))))
        s = self._dn(s, c, torch.zeros_like(w['st_LAI']))
        dvs = w['st_Development']
        return s + w['st_stems'] * self.afgen('SpecificStemArea', dvs) + \
            w['st_storage'] * self.p['SpecificPodArea']

    # ------------------------------------------------------------ integrate
    def _integration_crop(self, w):
        """IntegrationCrop.c:10-64."""
        D, S, p = self.D, self.S, self.p
        for k in ('roots', 'stems', 'leaves', 'storage', 'LAIExp'):
            w['st_' + k] = w['st_' + k] + w['rt_' + k]
        w['st_RootDepth_prev'] = w['st_RootDepth']
        w['st_RootDepth'] = w['st_RootDepth'] + w['rt_RootDepth']
        dvs = w['st_Development']
        w['st_Development_prev'] = dvs
        lt1 = D(dvs) < 1.
        nd = dvs + w['rt_Development']
        nd1 = torch.where(D(nd) > 1., torch.ones_like(nd), nd)
        w['st_Development'] = torch.where(lt1, nd1, nd)
        for k in ('roots', 'stems', 'leaves'):
            w['dst_' + k] = w['dst_' + k] + w['drt_' + k]
        w['st_vernalization'] = torch.where(
            self.idsl == 2, w['st_vernalization'] + w['rt_vernalization'],
            w['st_vernalization'])
        x = D(w['met_Temp'] - p['TempBaseLeaves']) / \
            (35. - D(p['TempBaseLeaves']))
        pa = S(torch.where(0. > x, torch.zeros_like(x), x))
        h, t = w['lh'].long(), w['lt'].long()
        c = self._lanes(w)
        if '__lmask' in w:
            from . import leaf_kernels as LK
            LK.ring_age_(w['lage'], self._cl(h, c), self._cl(t, c),
                         self._cl(pa, c), self._cl(w['__lmask'], c))
            return
        if self._lanemode and self._xr:
            old = self._rlive(w, self._cl(h, c), self._cl(t, c) - 1)[0]
        else:
            old = self._live(self._cl(h, c), self._cl(t, c) - 1)
        w['lage'] = torch.where(old, w['lage'] + self._cl(pa, c)[..., None],
                                w['lage'])

    def _integration_nutrients(self, w):
        """IntegrationNutrients.c:10-60."""
        zero = torch.zeros_like(w['st_roots'])
        for X in NUT:
            w['ste_st_%s_tot' % X] = _cmax(
                w['ste_st_%s_tot' % X] + w['ste_rt_%s_tot' % X], zero)
            w['ste_st_%s_mins' % X] = _cmax(
                w['ste_st_%s_mins' % X] - w['ste_rt_%s_mins' % X], zero)
        nd = w['N_rt_Demand']
        for X in NUT:
            # P and K accumulate the N demand (IntegrationNutrients.c:23-25)
            w['%s_st_Demand' % X] = w['%s_st_Demand' % X] + nd
            w['%s_st_Uptake' % X] = w['%s_st_Uptake' % X] + \
                w['%s_rt_Uptake' % X]
            for k in ('leaves', 'stems', 'roots', 'storage'):
                w['%s_st_%s' % (X, k)] = w['%s_st_%s' % (X, k)] + \
                    w['%s_rt_%s' % (X, k)]
            for k in ('death_lv', 'death_st', 'death_ro'):
                w['%s_st_%s' % (X, k)] = w['%s_st_%s' % (X, k)] + \
                    w['%s_rt_%s' % (X, k)]

    # ------------------------------------------------------------ outputs
    def rate_outputs(self, s, cal, t):
        """`_rate_outputs`, one fused region where the lane day is
        (_fusable)."""
        v = self._lane_view() if self.is_lane_state(s) else None
        if _fusable(self) and not torch.compiler.is_compiling():
            if not torch.is_tensor(t):
                t = torch.full((), int(t), dtype=torch.long,
                               device=self.device)
            return _compiled_rates()(self, s, cal, t, v)
        return self._rate_outputs(s, cal, t, v)

    #: the state fields _rate_outputs reads
    RATE_KEYS = tuple(('cultivating growing GrowthDay st_Development '
                       'rt_Development rt_leaves rt_stems rt_storage '
                       'rt_roots drt_leaves drt_stems drt_roots N_rt_leaves '
                       'N_rt_stems N_rt_storage N_rt_roots ste_rt_N_tot '
                       'ste_rt_P_tot ste_rt_K_tot N_rt_Uptake P_rt_Uptake '
                       'K_rt_Uptake N_rt_Demand P_rt_Demand K_rt_Demand '
                       'rt_RootDepth st_LAI N_st_Indx P_st_Indx K_st_Indx '
                       'NPK_Indx soil_WaterStress NutrientStress '
                       'met_TempStress met_TminStress').split())

    def _rate_outputs(self, s, cal, t, v=None):
        """OUT_CROP_* of crop_put_rate_data (crop_output.c:1034-1281), the
        WOFOST part: [N, NC] float64 per variable (sum over bands of
        value x AreaFract x growing).  OUT_CROP_EVAP* (VIC fluxes, :970-1033)
        belong to the crop feedback."""
        if self.is_lane_state(s):
            return self._rate_outputs_lanes(
                s, cal, t, self._lane_view() if v is None else v)
        N, B, NC = self.shape
        af = self.af
        gr = s['growing'].to(torch.float64)
        ex = self.active
        one = lambda v: v.to(torch.float64)
        g2 = (s['GrowthDay'] == 2)
        rows = self._rate_rows(s, cal, t)
        out = {}
        for nm, (v, add, by_gr) in rows.items():
            acc = torch.zeros(N, NC, dtype=torch.float64, device=self.device)
            for b in range(B):
                x = one(v[:, b]) * af[:, b]
                if by_gr:
                    x = x * gr[:, b]
                acc = acc + torch.where(ex[:, b], x, torch.zeros_like(x))
                if add is not None:
                    # GrowthDay 2: add the initial states (:1163-1234)
                    a2 = one(add[:, b]) * af[:, b] * gr[:, b]
                    acc = torch.where(g2[:, b] & ex[:, b], acc + a2, acc)
            out['OUT_CROP_' + nm] = acc
        return out

    def _rate_outputs_lanes(self, s, cal, t, v):
        """_rate_outputs of a lane state: each row's band terms on the
        lanes, added into [cell, crop] band by band (a unit off the lanes
        adds 0.0 there, as its AreaFract 0 makes its term)."""
        N, B, NC = self.shape
        one = lambda x: x.to(torch.float64)
        af = v.af.reshape(-1)
        gr = one(s['growing']).reshape(-1)
        g2 = (s['GrowthDay'] == 2).reshape(-1)
        rows = v._rate_rows(s, cal, t)
        out = {}
        for nm, (x0, add, by_gr) in rows.items():
            x = one(x0).reshape(-1) * af
            if by_gr:
                x = x * gr
            a2 = None if add is None else one(add).reshape(-1) * af * gr
            acc = torch.zeros(N * NC, dtype=torch.float64,
                              device=self.device)
            for pos, tgt in self._lband:
                cur = acc.index_select(0, tgt) + x.index_select(0, pos)
                if a2 is not None:
                    cur = torch.where(g2.index_select(0, pos),
                                      cur + a2.index_select(0, pos), cur)
                acc = acc.index_copy(0, tgt, cur)
            out['OUT_CROP_' + nm] = acc.view(N, NC)
        return out

    def _rate_rows(self, s, cal, t):
        """The rows of _rate_outputs: name -> (value, value added on the
        second growth day or None, times growing)."""
        N, B, NC = self.shape
        p = self.p
        tdev = s['st_Development'] - s['rt_Development']
        fr = self.afgen('Roots', tdev)
        tdw_root = p['InitialDryWeight'] * fr
        tdw_leaf = p['InitialDryWeight'] * (1 - fr) * self.afgen('Leaves',
                                                                 tdev)
        tdw_stem = p['InitialDryWeight'] * (1 - fr) * self.afgen('Stems',
                                                                 tdev)
        tdw_so = p['InitialDryWeight'] * (1 - fr) * self.afgen('Storage',
                                                               tdev)
        nmx = self.afgen('N_MaxLeaves', tdev)
        nst = p['N_MaxStems'] * nmx
        nrt = p['N_MaxRoots'] * nmx
        dvsi = p['InitialDVS'] + torch.zeros_like(tdev)
        fl = _at(cal.fert_list, t)
        lst = {X: self._cls(fl[:, j] * self.m_full['%s_Uptake_frac' % X]
                            ).expand(N, B, NC) for j, X in enumerate(NUT)}
        rows = {
            'CULTIVATE': (s['cultivating'], None, False),
            'GROW': (s['growing'], None, False),
            'DVS': (s['rt_Development'], dvsi, True),
            'WLV': (s['rt_leaves'], tdw_leaf, True),
            'WST': (s['rt_stems'], tdw_stem, True),
            'WSO': (s['rt_storage'], tdw_so, True),
            'WRT': (s['rt_roots'], tdw_root, True),
            'WDLV': (s['drt_leaves'], None, True),
            'WDST': (s['drt_stems'], None, True),
            'WDRT': (s['drt_roots'], None, True),
            'NLV': (s['N_rt_leaves'], nmx * tdw_leaf, True),
            'NST': (s['N_rt_stems'], nst * tdw_stem, True),
            'NSO': (s['N_rt_storage'], None, True),
            'NRT': (s['N_rt_roots'], nrt * tdw_root, True),
            'NSOIL': (s['ste_rt_N_tot'], lst['N'], True),
            'PSOIL': (s['ste_rt_P_tot'], lst['P'], True),
            'KSOIL': (s['ste_rt_K_tot'], lst['K'], True),
            'NUPT': (s['N_rt_Uptake'], None, True),
            'PUPT': (s['P_rt_Uptake'], None, True),
            'KUPT': (s['K_rt_Uptake'], None, True),
            'NDEM': (s['N_rt_Demand'], None, True),
            'PDEM': (s['P_rt_Demand'], None, True),
            'KDEM': (s['K_rt_Demand'], None, True),
            'ROOTDEPTH': (s['rt_RootDepth'], None, True),
            'LAI': (s['st_LAI'], None, True),
            'NNI': (s['N_st_Indx'], None, True),
            'PNI': (s['P_st_Indx'], None, True),
            'KNI': (s['K_st_Indx'], None, True),
            'NPKI': (s['NPK_Indx'], None, True),
            'WSTRESS': (s['soil_WaterStress'], None, True),
            'NSTRESS': (s['NutrientStress'], None, True),
            'TMAXSTRESS': (s['met_TempStress'], None, True),
            'TMINSTRESS': (s['met_TminStress'], None, True),
        }
        return rows
