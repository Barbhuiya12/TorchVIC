"""Physical sanity checker for a TorchVIC run on a gridded domain.

  python sanity.py RUN_DIR [--out DIR]       # RUN_DIR has series.npz, ledger.json

`check(series, ledger, area_m2, dates)` -> list of findings (dicts) ranked by
severity, module, number of cells. A series is {'module.var': [T, N] array}
(units: VIC variables mm/day or mm; groundwater baseflow m3/s; see UNITS).

Checks, per variable and cell
  nonfinite      NaN / Inf.
  negative       a quantity that cannot be negative (discharge, storages,
                 runoff, ET, withdrawals) below -tol.
  et_budget      annual ET > P + irrigation + capillary rise - dS (+ margin):
                 ET cannot exceed the water that entered the cell.
  et_potential   annual ET > 1.3 x Priestley-Taylor potential (alpha 1.26,
                 Priestley and Taylor 1972; net radiation from SWdown, albedo
                 0.23, net longwave 50 W/m2: Allen et al. 1998, FAO-56).
  runoff_ratio   annual (runoff + baseflow) > P + irrigation + capillary rise
                 + storage loss (+ margin): runoff ratio above 1 where
                 nothing adds water.
  baseflow_high  annual baseflow above BOUND_MM (3000 mm/yr; the
                 wettest ISC cells, Cherrapunji, get 11000 mm of rain and a
                 baseflow share well below 50 %) or above the cell's P + I
                 + capillary rise (water that left the aquifer through EVT
                 and entered the soil is water that entered the cell, as in
                 et_budget and runoff_ratio). Recharge / gwrecharge (gross
                 drainage of the bottom layer) has no absolute bound: only
                 P + I + capillary rise + soil storage lost.
  baseflow_budget  (added, no bound loosened) MODFLOW net exchange with the
                 rivers summed over a catchment of at least BASIN_KM2 above
                 the same sum of P + I + capillary rise.
  swe_persist    snow water equivalent above 5 mm for > 45 days with < 20 mm
                 of snowfall in the year, or left on the last day with
                 a last-30-day mean T > 10 C.
  oscillation    day-to-day sign flips of the increments (fraction of days
                 > 0.35 with an amplitude > 10 % of the mean) or > 30 % of the
                 variance at periods <= 3 d, on discharge, storages and exchanges.
  drift          a storage whose last-month mean differs from its first-month
                 mean by more than its own seasonal range and a floor (spin-up).
Ledger
  ledger         a module's or the coupled domain's row that does not close
                 (ok False), and a relative residual above 1e-6 of the scale.
  vic_water_error the domain total of VIC's own water error above 0.1 % of P.
"""
import argparse
import json
import os

import numpy as np

SEV = {'high': 3, 'medium': 2, 'low': 1}
BOUND_MM = 3000.0
#: smallest catchment (km2) the groundwater exchange budget is checked on
BASIN_KM2 = 20000.0

#: nonnegative variables and the tolerance (units of the variable)
NONNEG = {
    'vic.runoff': 1e-6, 'vic.baseflow': 1e-6, 'vic.et': 1e-6,
    'vic.rainf': 1e-9, 'vic.snowf': 1e-9, 'vic.swe': 1e-6,
    'vic.soil_moist': 1e-6, 'vic.wdew': 1e-6, 'vic.gwrecharge': 1e-6,
    'vic.recharge': 1e-6,
    'vicwur_routing.discharge': 1e-3, 'vicwur_routing.stream_moist': 1e-3,
    'vicwur_dams.storage': 1e-3, 'vicwur_wateruse_local.withdrawn': 1e-9,
    'vicwur_nonrenew.deficit': 1e-9, 'vicwur_nonrenew.withdrawn': 1e-9,
    'vicwur_efr.efr_discharge': 1e-9, 'vicwur_efr.efr_baseflow': 1e-9,
    'vicwur_efr.efr_moist': 1e-9,
    'vicwur_irrigation.shortage': 1e-9, 'vicwur_irrigation.received': 1e-9,
    'vicwur_irrigation.applied': 1e-9, 'vicwur_irrigation.leftover': 1e-9,
    'groundwater.capillary_rise': 1e-9,
}
#: irrigation 'requirement' is not listed above: VIC-WUR's irr_run.c never
#: clamps it (negative where the soil is wetter than field capacity).
#: 'applied' (not 'received') is the water that enters the soil: it includes
#: the re-application of the previous step's leftover, and closes the budget.
#: exchange series that follow demand steps: oscillation is at most 'medium'
EXCH = {'vicwur_wateruse_local.withdrawn', 'vicwur_nonrenew.withdrawn',
        'vicwur_irrigation.received', 'vicwur_irrigation.applied'}
#: oscillation targets: discharge, storages, exchanges
OSC = ['vicwur_routing.discharge', 'vicwur_routing.stream_moist',
       'vicwur_dams.storage', 'vicwur_efr.efr_moist', 'groundwater.head',
       'groundwater.head2', 'groundwater.baseflow',
       'groundwater.capillary_rise', 'vic.soil_moist', 'vic.swe',
       'vicwur_wateruse_local.withdrawn', 'vicwur_nonrenew.withdrawn',
       'vicwur_irrigation.received', 'vicwur_irrigation.applied',
       'vicwur_routing.stream_moist']
#: storages for the drift check: (floor in the variable's units)
DRIFT = {'vic.soil_moist': 30.0, 'vic.swe': 50.0,
         'vicwur_routing.stream_moist': None, 'vicwur_dams.storage': None,
         'groundwater.head': 5.0, 'groundwater.head2': 5.0,
         'vicwur_efr.efr_moist': 20.0}


def _first(mask2d):
    """(n cells, first day index) of a [T, N] boolean."""
    anyc = mask2d.any(0)
    if not anyc.any():
        return 0, -1
    return int(anyc.sum()), int(mask2d[:, anyc].argmax(0).min())


def _f(module, var, check, ncell, sev, first, detail, dates=None, n=None):
    day = ''
    if first is not None and first >= 0 and dates is not None:
        day = str(dates[min(first, len(dates) - 1)])
    return dict(module=module, variable=var, check=check, cells=int(ncell),
                frac=(ncell / n if n else None), severity=sev,
                first_day=day, detail=detail)


def _sev_cells(nc, n, hi=0.01, me=0.0005):
    fr = nc / max(n, 1)
    return 'high' if fr > hi else ('medium' if fr > me else 'low')


def _mod(key):
    return key.split('.', 1)[0]


def _spectral_hf(x):
    """Fraction of the variance at periods <= 3 d, [N]."""
    x = x - x.mean(0, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(x.shape[0])
    tot = p[1:].sum(0)
    return p[f >= 1 / 3.0].sum(0) / np.maximum(tot, 1e-30), tot


def _accumulate(down, x):
    """Sum of x over each cell and everything upstream of it; down[i] is
    the cell i drains to (itself at an outlet)."""
    n = len(down)
    acc = np.array(x, dtype=np.float64)
    indeg = np.bincount(down[down != np.arange(n)], minlength=n)
    st = list(np.nonzero(indeg == 0)[0])
    while st:
        i = st.pop()
        j = down[i]
        if j == i:
            continue
        acc[j] += acc[i]
        indeg[j] -= 1
        if indeg[j] == 0:
            st.append(j)
    return acc


def pet_pt(tair, swdown):
    """Priestley-Taylor potential ET, mm/day [T, N]."""
    t = tair.astype(np.float64)
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    delta = 4098.0 * es / (t + 237.3) ** 2
    gamma = 0.066
    rn = np.maximum(0.77 * swdown - 50.0, 0.0)          # W/m2
    return 1.26 * delta / (delta + gamma) * rn * 0.0864 / 2.45


def check(S, ledger=None, area_m2=None, dates=None, window=None, down=None):
    """Findings (list of dicts, ranked)."""
    F = []
    N = next(iter(S.values())).shape[1]
    # a window shorter than a year: the annual bounds scale with its length
    T = max(v.shape[0] for v in S.values())
    fac = min(T / 365.0, 1.0) if T < 300 else 1.0
    BOUND = BOUND_MM * fac
    keys = sorted(S)
    A = lambda k: S[k].astype(np.float64)
    # ---- nonfinite, negative
    for k in keys:
        a = S[k]
        bad = ~np.isfinite(a)
        if bad.any():
            nc, fd = _first(bad)
            F.append(_f(_mod(k), k.split('.', 1)[1], 'nonfinite', nc, 'high',
                        fd, '%d values NaN/Inf' % bad.sum(), dates, N))
    for k, tol in NONNEG.items():
        if k not in S:
            continue
        a = S[k]
        sc = max(float(np.nanmax(np.abs(a))) * 1e-7, tol) if a.size else tol
        neg = a < -sc
        if k == 'vic.et':
            # vapour deposition on a snow pack (and dew) is a legitimate
            # negative ET in VIC: only what is not on snow, beyond 0.05 mm/d
            on_snow = (S['vic.swe'] > 0) if 'vic.swe' in S else False
            if 'vic.swe' in S:                  # incl. the day the pack melts out
                on_snow = on_snow.copy()
                on_snow[1:] |= S['vic.swe'][:-1] > 0
            neg = (a < -0.05) & ~on_snow
        if neg.any():
            nc, fd = _first(neg)
            mn = float(np.nanmin(a))
            sev = 'high' if (nc / N > 0.001 or abs(mn) > 100 * sc) else 'medium'
            if k == 'vic.et' and nc / N <= 0.001 and mn > -3.0:
                sev = 'medium'          # a few cells, down to -3 mm/d
            F.append(_f(_mod(k), k.split('.', 1)[1], 'negative', nc, sev, fd,
                        'min %.4g (tol %.2g); %.3g%% of cell-days'
                        % (mn, sc, 100.0 * neg.mean()), dates, N))
    # ---- annual budgets (VIC water in mm over the window)
    g = lambda k: S.get(k)
    if g('vic.rainf') is not None and g('vic.et') is not None:
        P = A('vic.rainf').sum(0) + (A('vic.snowf').sum(0)
                                     if g('vic.snowf') is not None else 0)
        ET = A('vic.et').sum(0)
        I = A('vicwur_irrigation.applied').sum(0) \
            if g('vicwur_irrigation.applied') is not None else 0.0
        cap = 0.0
        if g('groundwater.capillary_rise') is not None and area_m2 is not None:
            cap = A('groundwater.capillary_rise').sum(0) * 86400.0 \
                / area_m2 * 1000.0
        dS = np.zeros(N)
        for k in ('vic.soil_moist', 'vic.swe', 'vic.wdew'):
            if g(k) is not None:
                dS += A(k)[-1] - A(k)[0]
        marg = np.maximum(50.0 * fac, 0.05 * P)
        ex = ET - (P + I + cap - dS)
        bad = ex > marg
        if bad.any():
            F.append(_f('vic', 'et', 'et_budget', int(bad.sum()),
                        _sev_cells(bad.sum(), N), -1,
                        'ET exceeds P+irrigation+capillary rise-dS by up to '
                        '%.0f mm/yr (median %.0f)' % (ex[bad].max(),
                                                      np.median(ex[bad])),
                        dates, N))
        if g('vic.air_temp') is not None and g('vic.swdown') is not None:
            pet = pet_pt(A('vic.air_temp'), A('vic.swdown')).sum(0)
            bad = (ET > 1.3 * pet + 100.0 * fac) & (ET > 100 * fac)
            if bad.any():
                F.append(_f('vic', 'et', 'et_potential', int(bad.sum()),
                            _sev_cells(bad.sum(), N), -1,
                            'ET above 1.3 x PT potential: max ratio %.2f '
                            '(mean PET %.0f mm/yr)' % (
                                (ET[bad] / np.maximum(pet[bad], 1)).max(),
                                pet.mean()), dates, N))
        if g('vic.runoff') is not None:
            Q = A('vic.runoff').sum(0)
            if g('vic.baseflow') is not None:
                Q = Q + A('vic.baseflow').sum(0)
            sl = np.maximum(-dS, 0.0)
            bad = Q > P + I + cap + sl + np.maximum(50.0 * fac, 0.05 * P)
            if bad.any():
                F.append(_f('vic', 'runoff', 'runoff_ratio', int(bad.sum()),
                            _sev_cells(bad.sum(), N), -1,
                            'runoff+baseflow above all inflow: max %.0f mm/yr'
                            ' over (P=%.0f)' % ((Q - P - I - cap - sl)[bad].max(),
                                                P[bad].mean()), dates, N))
        lim = P + I + cap + 50.0 * fac
        sl = np.maximum(-dS, 0.0)
        for k in ('vic.baseflow', 'vic.recharge', 'vic.gwrecharge'):
            if g(k) is None:
                continue
            ann = A(k).sum(0)
            var = k.split('.')[1]
            if var == 'baseflow':
                bad = (ann > BOUND) | (ann > lim)
            else:
                # recharge / gwrecharge = gross drainage of the bottom layer
                # (runoff.c Q12): bounded by the water that entered the
                # column (P + I + capillary rise) plus the soil storage lost
                # over the window, not by an absolute depth: the wettest
                # cells (P up to 11000 mm) drain more than BOUND_MM.
                bad = ann > lim + sl
            if bad.any():
                F.append(_f('vic', var, 'baseflow_high', int(bad.sum()),
                            _sev_cells(bad.sum(), N), -1,
                            'window max %.0f mm (bound %.0f)' % (
                                ann.max(), BOUND), dates, N))
        if g('groundwater.baseflow') is not None and area_m2 is not None:
            bf = A('groundwater.baseflow').sum(0) * 86400.0 / area_m2 * 1000.0
            bad = (bf > BOUND) | (bf > lim)
            nc = int(bad.sum())
            if nc:
                F.append(_f('groundwater', 'baseflow', 'baseflow_high', nc,
                            _sev_cells(nc, N, 0.005, 0.0002), -1,
                            'MODFLOW baseflow up to %.0f mm in the window '
                            '(bound %.0f); 99th pct %.0f' % (
                                bf.max(), BOUND, np.percentile(bf, 99)),
                            dates, N))
            if down is not None:
                ua = _accumulate(down, area_m2) / 1e6
                avail = lim * area_m2 / 1000.0
                va = _accumulate(down, avail)
                vb = _accumulate(down, bf * area_m2 / 1000.0)
                big = ua >= BASIN_KM2
                bad = big & (vb > 1.05 * va)
                if bad.any():
                    F.append(_f('groundwater', 'baseflow', 'baseflow_budget',
                                int(bad.sum()), _sev_cells(bad.sum(), N, 0.0,
                                                           0.0), -1,
                                'catchments of >= %.0f km2: net exchange up '
                                'to %.2f of the water entering them' % (
                                    BASIN_KM2, (vb / np.maximum(va, 1))[
                                        big].max()), dates, N))
        if g('vic.swe') is not None and g('vic.snowf') is not None:
            sn = A('vic.snowf').sum(0)
            sw = A('vic.swe')
            tl = A('vic.air_temp')[-30:].mean(0) if g('vic.air_temp') is not None \
                else np.zeros(N)
            # snow that stays > 45 days (scaled to the window) with next to
            # no snowfall to feed it, or on the last day while the last 30 days average > 10 C
            bad = (((sn < 20.0 * fac) & ((sw > 5.0).sum(0) > 45 * fac))
                   | ((tl > 10.0) & (sw[-1] > 5.0)))
            if bad.any():
                nc, fd = _first(sw[:, bad.nonzero()[0]] > 5.0) \
                    if False else (int(bad.sum()), -1)
                F.append(_f('vic', 'swe', 'swe_persist', nc,
                            _sev_cells(nc, N), 0,
                            'SWE up to %.0f mm where it does not snow'
                            % sw[:, bad].max(), dates, N))
    # ---- oscillation
    for k in sorted(set(OSC)):
        if k not in S:
            continue
        x = A(k)
        if x.shape[0] < 20:
            continue
        mean = np.abs(x).mean(0)
        d = np.diff(x, axis=0)
        sig = np.abs(d) > 0.10 * np.maximum(mean, 1e-12)
        flip = (d[1:] * d[:-1] < 0) & sig[1:] & sig[:-1]
        fr = flip.mean(0)
        hf, tot = _spectral_hf(x)
        amp = np.ptp(x, axis=0) / np.maximum(mean, 1e-12)
        floor = np.percentile(mean, 5) * 0 + 1e-9
        osc = ((fr > 0.35) | ((hf > 0.30) & (amp > 0.10))) & (mean > floor)
        if osc.any():
            idx = osc.nonzero()[0]
            sev = _sev_cells(len(idx), N, 0.01, 0.0005)
            if k in EXCH and sev == 'high':
                sev = 'medium'
            fd = int(flip[:, idx].argmax(0).min()) + 1
            F.append(_f(_mod(k), k.split('.', 1)[1], 'oscillation', len(idx),
                        sev, fd, 'sign-flip fraction max %.2f, HF variance '
                        'share max %.2f' % (fr[idx].max(), hf[idx].max()),
                        dates, N))
    # ---- drift
    for k, fl in DRIFT.items():
        if k not in S or S[k].shape[0] < 90:
            continue
        x = A(k)
        a0, a1 = x[:30].mean(0), x[-30:].mean(0)
        rng = np.ptp(x, axis=0)
        if fl is None:
            fl = 0.05 * (np.abs(x).mean(0) + 1e-12)
        bad = (np.abs(a1 - a0) > rng) & (np.abs(a1 - a0) > fl)
        if bad.any():
            nc = int(bad.sum())
            F.append(_f(_mod(k), k.split('.', 1)[1], 'drift', nc,
                        _sev_cells(nc, N, 0.2, 0.02), -1,
                        'first-30-d to last-30-d mean change: max %.3g, '
                        'median %.3g' % (np.abs(a1 - a0)[bad].max(),
                                         np.median(np.abs(a1 - a0)[bad])),
                        dates, N))
    # ---- ledger
    if ledger is not None:
        for r in ledger.get('balances', []):
            pass
        byrow = {}
        for r in ledger.get('balances', []):
            m = r.get('module', '?')
            b = byrow.setdefault(m, dict(n=0, bad=0, rel=0.0, first=None))
            b['n'] += 1
            sc = abs(r.get('scale') or 0.0) or 1.0
            rel = abs(r.get('residual') or 0.0) / sc
            b['rel'] = max(b['rel'], rel)
            if not r.get('ok', True):
                b['bad'] += 1
                if b['first'] is None:
                    b['first'] = r.get('t0')
        for m, b in byrow.items():
            if b['bad']:
                F.append(_f(m, 'water balance', 'ledger', b['bad'], 'high',
                            None, '%d of %d rows do not close; worst relative '
                            'residual %.2g' % (b['bad'], b['n'], b['rel']),
                            dates, None))
            elif b['rel'] > 1e-6:
                F.append(_f(m, 'water balance', 'ledger', 0, 'low', None,
                            'closes; worst relative residual %.2g' % b['rel'],
                            dates, None))
        for r in ledger.get('coupled', []):
            if not r.get('ok', True):
                F.append(_f('domain', 'coupled water balance', 'ledger', 1,
                            'high', None, str(r)[:200], dates, None))
                break
    if g('vic.water_error') is not None and g('vic.rainf') is not None:
        we = np.abs(A('vic.water_error')).sum()
        pt = A('vic.rainf').sum() + (A('vic.snowf').sum()
                                     if g('vic.snowf') is not None else 0)
        if we > 1e-3 * pt:
            F.append(_f('vic', 'water_error', 'vic_water_error', 0,
                        'high' if we > 1e-2 * pt else 'medium', None,
                        'domain sum |water_error| %.3g mm = %.2g of P'
                        % (we, we / pt), dates, None))
    F.sort(key=lambda f: (-SEV[f['severity']], f['module'], -f['cells']))
    for i, f in enumerate(F):
        f['rank'] = i + 1
    return F


def load_run(run_dir):
    z = np.load(os.path.join(run_dir, 'series.npz'))
    S = {k: z[k] for k in z.files}
    lp = os.path.join(run_dir, 'ledger.json')
    led = json.load(open(lp)) if os.path.exists(lp) else None
    return S, led


def domain_area():
    import netCDF4 as nc
    d = nc.Dataset(os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'vic_params_isc/domain.nc'))
    m = np.ma.filled(d['mask'][:], 0) > 0
    return np.ma.filled(d['area'][:], 0)[m].astype(np.float64)


def domain_downstream():
    """Cell each ISC cell drains to (routing network of the smoke runs)."""
    import netCDF4 as nc
    R = os.environ.get('TORCHVIC_DATA_ROOT', 'data').rstrip('/') + '/'
    m = np.ma.filled(nc.Dataset(R + 'vic_params_isc/domain.nc')['mask'][:],
                     0) > 0
    d = nc.Dataset(R + 'vicwur/isc/smoke/rout_n4.nc')
    cid = np.ma.filled(d['downstream_id'][:], -1)[m]
    dwn = np.ma.filled(d['downstream'][:], -1)[m]
    pos = {int(c): i for i, c in enumerate(cid)}
    return np.array([pos.get(int(x), i) for i, x in enumerate(dwn)])


def write(F, out_dir, name='findings'):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(F, open(os.path.join(out_dir, name + '.json'), 'w'), indent=1)
    cols = ['rank', 'module', 'variable', 'check', 'cells', 'severity',
            'first_day', 'detail']
    with open(os.path.join(out_dir, name + '.md'), 'w') as fh:
        fh.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for f in F:
            fh.write('| ' + ' | '.join(str(f[c]) for c in cols) + ' |\n')


def run_dir_check(run_dir, out_dir=None):
    import datetime
    S, led = load_run(run_dir)
    T = max(v.shape[0] for v in S.values())
    t = min(v.shape[0] for v in S.values())
    # windows differ by one row between VIC and the connectors: align right
    dates = [datetime.date(2005, 1, 1) + datetime.timedelta(i)
             for i in range(365)]
    S = {k: v[-min(v.shape[0], 364):] for k, v in S.items()} \
        if all(v.shape[0] >= 364 for v in S.values()) else S
    dates = dates[-next(iter(S.values())).shape[0]:]
    N = next(iter(S.values())).shape[1]
    area = domain_area() if N == 112825 else None
    F = check(S, led, area, dates,
              down=domain_downstream() if N == 112825 else None)
    write(F, out_dir or run_dir)
    return F


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    for f in run_dir_check(a.run_dir, a.out):
        print('%-4d %-8s %-16s %-14s %7d  %-6s %s  %s' % (
            f['rank'], f['severity'], f['module'], f['check'], f['cells'],
            f['variable'], f['first_day'], f['detail']))
