# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Inputs of the WOFOST-NPK kernel: crop templates, unit set-up, calendar.

C references are VIC-WUR 3.0 (<data root>/build/vicwur3/src_tree/vic/plugins):
  wofost/src/Fillvar.c:9-111,186-207   scalar order of the crop and
                                       management files
  wofost/src/Cropdata.c:112-159        table assignment, MaxOptimumTemp
  crops/src/wofost_init.c:4-93         one template per crop class, copied
                                       to every (cell, band) unit
  crops/src/wofost_init.c:95-532       DIST_SEASON / TSUM / FERT / MIN
                                       values and CycleLength
  plugins/general/src/plugin_time_utils.c:127-162   dmy_doy
  wofost/src/List.c:9-50               fertiliser tables by date

Shapes: N cells, B snow bands, NC crop classes.  A unit is (cell, band,
crop class); a unit exists where the class has cover in the cell
(crop_con_map.cidx != NODATA, crop_init.c:93-113).  Values the C code holds
in float are kept as their float32 values (the readers of
torchvic/vicwur/params.py already round them) and cast to the kernel dtype.
"""
import numpy as np
import torch

#: plugins/wofost/src/Fillvar.c:12-104, in CropParam order
#: (crops/src/wofost_const_library.c:24-103)
SCALARS = (
    'TempBaseEmergence TempEffMax TSumEmergence IdentifyAnthesis '
    'OptimumDaylength CriticalDaylength SatVernRequirement '
    'BaseVernRequirement TempSum1 TempSum2 InitialDVS DevelopStageHarvest '
    'InitialDryWeight RelIncreaseLAI SpecificPodArea LifeSpan TempBaseLeaves '
    'ConversionLeaves ConversionStorage ConversionRoots ConversionStems Q10 '
    'RelRespiLeaves RelRespiStorage RelRespiRoots RelRespiStems '
    'MaxRelDeathRate CorrectionTransp CropGroupNumber Airducts '
    'InitRootingDepth MaxIncreaseRoot MaxRootingDepth DyingLeaves_NPK_Stress '
    'DevelopmentStageNLimit DevelopmentStageNT FracTranslocRoots Opt_N_Frac '
    'Opt_P_Frac Opt_K_Frac N_MaxRoots N_MaxStems P_MaxRoots P_MaxStems '
    'K_MaxRoots K_MaxStems NitrogenStressLAI NLUE Max_N_storage '
    'Max_P_storage Max_K_storage N_lv_partitioning NutrientStessSLA '
    'N_ResidualFrac_lv N_ResidualFrac_st N_ResidualFrac_ro P_ResidualFrac_lv '
    'P_ResidualFrac_st P_ResidualFrac_ro K_ResidualFrac_lv K_ResidualFrac_st '
    'K_ResidualFrac_ro TCNT TCPT TCKT N_fixation MaxHeight RGL '
    'RadiationAttenuation WindAttenuation TrunkRatio Albedo '
    'MinStomatalResistance MaxArchitecturalResistance Fcanopy '
    'TranslocationFrac CritLAIFactor').split()
assert len(SCALARS) == 77

#: Cropdata.c:124-147, in CropParam2 order.  Note the C swap kept here:
#: RDRRTB (the 18th table) becomes DeathRateStems and RDRSTB DeathRateRoots
#: (Cropdata.c:143-144).
TABLES = (
    'VernalizationRate DeltaTempSum SpecificLeaveArea SpecificStemArea '
    'KDiffuseTb EFFTb MaxAssimRate FactorAssimRateTemp FactorGrossAssimTemp '
    'CO2AMAXTB CO2EFFTB CO2TRATB FactorSenescence Roots Leaves Stems Storage '
    'DeathRateStems DeathRateRoots N_MaxLeaves P_MaxLeaves K_MaxLeaves'
).split()
assert len(TABLES) == 22

#: Fillvar.c:194-202 (ManageParam order NRFTAB ... RTKMINS)
MNG_SCALARS = ('N_Uptake_frac P_Uptake_frac K_Uptake_frac N_Mins '
               'NRecoveryFrac P_Mins PRecoveryFrac K_Mins KRecoveryFrac'
               ).split()
#: Management.c:108-111 (FERNTAB FERPTAB FERKTAB IRRTAB)
MNG_TABLES = ['N_Fert_table', 'P_Fert_table', 'K_Fert_table', 'Irrigation']

DAYS_PER_YEAR = 365
NODATA = -1


def _f32(v):
    return float(np.float32(v))


def _digest(obj):
    """Class and a content digest (a coupling configuration holding the
    object is checkpointed under a digest of its repr)."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(vars(obj)):
        v = vars(obj)[k]
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(repr(v).encode())
    return '<%s %s>' % (type(obj).__name__, h.hexdigest()[:16])


class CropTemplates(object):
    """One WOFOST-NPK template per crop class (the parsed crop and
    management files, wofost_init.c:52-62).

    scalars   dict name -> list[NC] of float (float32 values)
    tables    dict name -> list[NC] of (xs, ys) python lists
    mng       dict name -> list[NC]
    mng_tables dict name -> list[NC] of (months, days, amounts)
    emergence list[NC] of int (start at emergence 1 / sowing 0)
    start     list[NC] of (month, day) of the text file
    """

    def __init__(self, scalars, tables, mng, mng_tables, emergence, start):
        self.scalars = scalars
        self.tables = tables
        self.mng = mng
        self.mng_tables = mng_tables
        self.emergence = list(emergence)
        self.start = list(start)
        self.nc = len(self.emergence)
        idsl = [int(v) for v in scalars['IdentifyAnthesis']]
        # Fillvar.c:20-27: no vernalisation parameters below IDSL 2
        for k in range(self.nc):
            if idsl[k] < 2:
                scalars['SatVernRequirement'][k] = -99.0
                scalars['BaseVernRequirement'][k] = -99.0
                # Cropdata.c:112-122: VernalizationRate NULL
                tables['VernalizationRate'][k] = ([], [])
        # Cropdata.c:149-159: x of the last maximum of TMPFTB
        self.max_optimum_temp = []
        for k in range(self.nc):
            xv = yv = np.float32(0)
            for x, y in zip(*tables['FactorAssimRateTemp'][k]):
                if np.float32(y) >= yv:
                    xv, yv = np.float32(x), np.float32(y)
            self.max_optimum_temp.append(float(xv))

    __repr__ = _digest

    @classmethod
    def from_wofost_text(cls, wt):
        """From torchvic.vicwur.params.read_wofost_text."""
        from torchvic.vicwur.params import (CROP_TABLES, MNG_TABLES as MT,
                                            CROP_SCALARS, MNG_SCALARS as MS)
        sc = wt.scalars.detach().cpu().double().numpy()
        nc = sc.shape[0]
        scalars = {SCALARS[i]: [float(sc[k, i]) for k in range(nc)]
                   for i in range(77)}
        assert len(CROP_SCALARS) == 77
        tables = {}
        for i, nm in enumerate(CROP_TABLES):
            X, Y, n = wt.tables[nm]
            X = X.cpu().double().numpy()
            Y = Y.cpu().double().numpy()
            n = n.cpu().numpy()
            tables[TABLES[i]] = [(list(X[k, :n[k]]), list(Y[k, :n[k]]))
                                 for k in range(nc)]
        ms = wt.mng_scalars.cpu().double().numpy()
        mng = {MNG_SCALARS[i]: [float(ms[k, i]) for k in range(nc)]
               for i in range(len(MS))}
        mtab = {}
        for i, nm in enumerate(MT):
            M, D, A, n = wt.mng_tables[nm]
            M, D = M.cpu().numpy(), D.cpu().numpy()
            A, n = A.cpu().double().numpy(), n.cpu().numpy()
            mtab[MNG_TABLES[i]] = [(list(M[k, :n[k]]), list(D[k, :n[k]]),
                                    list(A[k, :n[k]])) for k in range(nc)]
        start = list(zip(wt.start_month.cpu().tolist(),
                         wt.start_day.cpu().tolist()))
        return cls(scalars, tables, mng, mtab, wt.emergence.cpu().tolist(),
                   start)

    # --------------------------------------------------------------- device
    def table_tensors(self, device, dtype):
        """name -> (X [NC,P], Y [NC,P], n [NC]); padding is NaN (masked)."""
        out = {}
        for nm in TABLES:
            rows = self.tables[nm]
            p = max([len(x) for x, _ in rows] + [1])
            X = np.full((self.nc, p), np.nan)
            Y = np.full((self.nc, p), np.nan)
            n = np.zeros(self.nc, np.int64)
            for k, (x, y) in enumerate(rows):
                X[k, :len(x)] = x
                Y[k, :len(y)] = y
                n[k] = len(x)
            out[nm] = (torch.tensor(X, dtype=dtype, device=device),
                       torch.tensor(Y, dtype=dtype, device=device),
                       torch.tensor(n, device=device))
        return out


class UnitSetup(object):
    """Per (cell, class) and per cell values of one run (host arrays).

    Arrays (numpy):
      lat [N]; ph [N]; carbon [N]
      area_fract [N,B]; tfactor [N,B]
      active [N,NC] bool (cidx != NODATA)
      cv [N,NC]: Cv of the crop's vegetation tile
      cc [N,NC,12]: monthly crop cover (crop_con_map.Cc)
      start_month/start_day/start_doy/end_doy [N,NC] (int)
      tsum1/tsum2 [N,NC] (crop_con, double)
      fert [N,NC,NF,4]: DVS_point, N, P, K amounts (crop_con, double)
      mins [N,NC,6]: N_Mins NRecoveryFrac P_Mins PRecoveryFrac K_Mins
                     KRecoveryFrac of the units (float32 values)
      cycle [N,NC]: CycleLength (wofost_init.c:500-529)
    """

    FIELDS = ('lat ph carbon area_fract tfactor active cv cc start_month '
              'start_day start_doy end_doy tsum1 tsum2 fert mins').split()

    __repr__ = _digest

    def __init__(self, continue_=False, **kw):
        for k in self.FIELDS:
            setattr(self, k, np.asarray(kw[k]))
        self.N, self.B = self.area_fract.shape
        self.NC = self.active.shape[1]
        self.NF = self.fert.shape[2]
        s, e = self.start_doy.astype(np.int64), self.end_doy.astype(np.int64)
        if continue_:
            self.cycle = np.full_like(s, DAYS_PER_YEAR - 10)
        else:
            self.cycle = np.where(s < e, e - s, DAYS_PER_YEAR - s + e)


def dmy_doy(doy, year=1900, leap=False):
    """plugin_time_utils.c:127-162 for a non-leap year: -> (month, day,
    day_in_year)."""
    days = [31, 28 + int(leap), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if doy < 1 or doy >= 365 + int(leap) + 1:
        raise ValueError('doy %r out of range' % doy)
    d = float(doy)
    for i, nd in enumerate(days):
        if d < nd + 1:
            return i + 1, int(d), int(doy)
        d -= nd
    raise ValueError(doy)


def setup_from_vicwur(tmpl, crop, lat, area_fract, tfactor, cv, ph=None,
                      carbon=None, dist_season=False, dist_tsum=False,
                      dist_fert=False, dist_min=False, calc_min=False,
                      continue_=False):
    """UnitSetup the way crop_init / wofost_init build it.

    tmpl CropTemplates; crop torchvic.vicwur.params.CropParams (read_crop);
    lat [N]; area_fract, tfactor [N,B] (Tfactor of band 0 already the crop
    nc value, crop_init.c:157-163); cv [N,NC] Cv of each crop's vegetation
    tile.  ph/carbon [N] are needed with calc_min (crop nc 'pH', 'carbon').
    """
    N, B = np.asarray(area_fract).shape
    NC = tmpl.nc
    cidx = crop.cidx.cpu().numpy()
    active = cidx >= 0
    cc = np.transpose(crop.Cc.cpu().double().numpy(), (2, 1, 0))  # N,NC,12
    f = {k: v.cpu().double().numpy() for k, v in crop.fields.items()}
    NF = max(1, int(getattr(crop, 'n_fert_times', 0) or 0))
    sm = np.zeros((N, NC), np.int64)
    sd = np.zeros((N, NC), np.int64)
    sdoy = np.zeros((N, NC), np.int64)
    edoy = np.zeros((N, NC), np.int64)
    for k in range(NC):
        sm[:, k], sd[:, k] = tmpl.start[k]
    if dist_season:
        # wofost_init.c:147-195: dmy_doy(plant_day / harvest_day, 1900)
        for i in range(N):
            for k in range(NC):
                if not active[i, k]:
                    continue
                m, d, y = dmy_doy(f['plant_day'][k, i])
                sm[i, k], sd[i, k], sdoy[i, k] = m, d, y
                edoy[i, k] = dmy_doy(f['harvest_day'][k, i])[2]
    ts1 = np.array([tmpl.scalars['TempSum1']] * N, np.float64)
    ts2 = np.array([tmpl.scalars['TempSum2']] * N, np.float64)
    if dist_tsum:
        ts1 = f['TSUM1'].T.copy()
        ts2 = f['TSUM2'].T.copy()
    fert = np.zeros((N, NC, NF, 4))
    if dist_fert:
        for j, nm in enumerate(('DVS_point', 'N_amount', 'P_amount',
                                'K_amount')):
            fert[:, :, :, j] = np.transpose(f[nm], (2, 1, 0))
    mins = np.zeros((N, NC, 6))
    for j, nm in enumerate(('N_Mins', 'NRecoveryFrac', 'P_Mins',
                            'PRecoveryFrac', 'K_Mins', 'KRecoveryFrac')):
        mins[:, :, j] = np.array(tmpl.mng[nm])[None, :]
    if dist_min:
        if calc_min:
            # wofost_init.c:340-360: NRecoveryFrac = 1. / period (float)
            per = f['mineralization_period'].T
            mins[:, :, 1] = (1.0 / per).astype(np.float32)
        else:
            for j, nm in enumerate(('N_mins', 'N_recovery', 'P_mins',
                                    'P_recovery', 'K_mins', 'K_recovery')):
                mins[:, :, j] = f[nm].T.astype(np.float32)
    if calc_min:
        ph = f['pH'] if ph is None else ph
        carbon = f['carbon'] if carbon is None else carbon
    ph = np.zeros(N) if ph is None else np.asarray(ph, np.float64)
    carbon = np.zeros(N) if carbon is None else np.asarray(carbon,
                                                           np.float64)
    return UnitSetup(continue_=continue_, lat=np.asarray(lat, np.float64),
                     ph=ph, carbon=carbon,
                     area_fract=np.asarray(area_fract, np.float64),
                     tfactor=np.asarray(tfactor, np.float64), active=active,
                     cv=np.asarray(cv, np.float64), cc=cc, start_month=sm,
                     start_day=sd, start_doy=sdoy, end_doy=edoy, tsum1=ts1,
                     tsum2=ts2, fert=fert, mins=mins)


class Calendar(object):
    """Host-precomputed per-step tensors of a daily run (no host sync in
    the step).

    dates: list of (year, month, day, day_in_year).  For every crop class
    the fertiliser tables give, per step, List() (first matching date,
    List.c:9-24) and the prefix sum over steps of all matching dates, so
    that List_cumsum(table, n) (List.c:26-50) is prefix[t+1] - prefix[t+1-n].
    """

    def __init__(self, dates, tmpl, device='cpu', dtype=torch.float64):
        T = len(dates)
        self.T = T
        a = np.array(dates, np.int64).reshape(T, 4)
        self.year = torch.tensor(a[:, 0], device=device)
        self.month = torch.tensor(a[:, 1], device=device)
        self.day = torch.tensor(a[:, 2], device=device)
        self.doy = torch.tensor(a[:, 3], device=device)
        lst = np.zeros((T, tmpl.nc, 3))
        allm = np.zeros((T, tmpl.nc, 3))
        for j, nm in enumerate(MNG_TABLES[:3]):
            for k in range(tmpl.nc):
                ms, ds, am = tmpl.mng_tables[nm][k]
                for t in range(T):
                    first = None
                    tot = np.float32(0)
                    for m, d, v in zip(ms, ds, am):
                        if a[t, 1] == m and a[t, 2] == d:
                            if first is None:
                                first = v
                            tot = np.float32(tot + np.float32(v))
                    lst[t, k, j] = 0.0 if first is None else first
                    allm[t, k, j] = tot
        self.fert_list = torch.tensor(lst, dtype=dtype, device=device)
        pre = np.zeros((T + 1, tmpl.nc, 3))
        pre[1:] = np.cumsum(allm, axis=0)
        self.fert_prefix = torch.tensor(pre, dtype=torch.float64,
                                        device=device)
