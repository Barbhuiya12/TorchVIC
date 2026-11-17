# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR global and local dams (C2, C3): batched tensor kernels.

Every kernel is a pure function over a [D] batch of dams (D can be
thousands); the only Python loops run over the 12 months, the 24-month
op_month scan, the 5 history years, the routing sub-steps of a model step,
the service cells and the sectors, never over dams.  Calendar flags (new
month, new year, op_month test, steps per month) are host precomputed
tensors indexed by step (`make_calendar`), all branching is `torch.where`,
so a step has no host sync and can be captured in a CUDA graph.  fp64
throughout, in the operation order of the C code.

C references are VIC-WUR 3.0 plugins/dams/src
(<data root>/build/vicwur3/src_tree/vic/plugins/dams/src; the Hanasaki
release maths is shared with the TorchVIC reservoir engine through
torchvic.reservoir.system.hanasaki_*):
  dam_operation.c:33-87    dam_calc_opt_release     monthly release before
                           the correction (hanasaki_release, ratio_first)
  dam_operation.c:93-119   dam_corr_opt_release     capacity correction
                           (hanasaki_capacity)
  dam_operation.c:124-135  dam_calc_k_factor        k = S / (alpha C), unclamped
  dam_operation.c:140-160  dam_calc_c_factor        c = C / sum_i I[i] steps[i]
  dam_operation.c:165-186  dam_calc_opt_storage     diagnostic only
  dam_operation.c:230-265  dam_corr_release         per-step correction
  dam_register.c:33-99     dam_register_operation_start  op_month (24-month scan)
  dam_register.c:104-148   dam_register_operation   plan at op_month
  dam_register.c:153-170   dam_register_history     monthly history shift
  dam_register.c:175-229   demand / EFR / inflow registration
  dam_register.c:263-297   local dam activation and inflow
  dam_register.c:299-340   global dam: history, op_month, plan, activation
  dam_run.c:33-71          dam_operate              fill, release, empty, full
  dam_run.c:78-91          local_dam_operate        write-back to OUT_RUNOFF
  dam_run.c:97-127         global_dam_operate       write-back to the ring
  dam_run.c:133-187        local_dam_run, global_dam_run
  dam_populate.c:33-51     cold start S = alpha C
  dam_init_library.c:33-60 initial dam_var (op_month = -1)
  plugin_init_library.c:112-114   alpha 0.85, beta 0.6, gamma 5 d

`exact=True` reproduces the C code including its two quirks: the dam demand
uses the area of the DAM cell for every service cell
(dam_register.c:200-228), and the steps-per-month factor of the c factor and
of op_storage is calendar-indexed while the climatology is indexed from the
current month (dam_register.c:129-133).  `exact=False` uses the service
cell's area and a month-aligned step count.  The C mass check of
dam_operate (dam_run.c:64-71) takes abs() of a double through the integer
abs, so it can only fire for a residual of 1 hm3 or more; it is not
reproduced (the identity is tested instead).  The C `max(a, b)` macro is
(a > b) ? a : b (vic_def.h:92), which differs from torch.maximum on NaN.

Two dams of one cell run in class order and compound inflow_frac through
the ring, so a `GlobalDams` / `LocalDams` set must hold at most one dam of
each cell; `rank_in_cell` splits a list into such sets.  No dam state is
saved by the C code (plugin_simulation.c:228-230); `state_dict` saves all of
it.
"""
import collections

import torch

from torchvic.reservoir.system import (hanasaki_capacity, hanasaki_release)

MONTHS = 12
HIST_YEARS = 5                  # dams.h:33 DAM_HIST_YEARS
HIST = MONTHS * HIST_YEARS
NODATA_DAM = -1                 # dams.h:32
M3_PER_HM3 = 1e6
MM_PER_M = 1000.0

STATE = ('active', 'storage', 'inflow', 'demand', 'efr', 'release',
         'history_inflow', 'history_demand', 'history_efr', 'op_release',
         'op_storage', 'total_inflow', 'total_demand', 'total_efr',
         'register_steps', 'op_month', 'months_running')


# ------------------------------------------------------------------ calendar
CALENDARS = {'standard': 0, 'gregorian': 1, 'proleptic_gregorian': 2,
             'noleap': 3, '365_day': 4, '360_day': 5, 'julian': 6,
             'all_leap': 7, '366_day': 8}


def leap_year(year, calendar):
    """drivers/shared_all/src/vic_time.c:652-673 (host)."""
    cal = CALENDARS.get(calendar, calendar)
    leap = False
    if cal in (6, 0, 1, 2) and year % 4 == 0:
        leap = True
        if cal == 2 and year % 100 == 0 and year % 400 != 0:
            leap = False
        elif cal in (0, 1) and year % 100 == 0 and year % 400 != 0 \
                and year > 1583:
            leap = False
    return leap


def days_per_month(month, year, calendar):
    """plugins/general/src/plugin_time_utils.c:109-121 (host).  Note that it
    only knows the leap day: a 360_day calendar still has 28/31-day months."""
    d = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d[1] += int(leap_year(year, calendar))
    return d[month - 1]


class DamCalendar(collections.namedtuple(
        'DamCalendar', 'year month new_month new_year steps new_month_host')):
    """Per-step calendar tensors of the dams, built once on the host.

    year, month [T] long (month 1-12); new_month, new_year [T] bool
    (`current > 0` and the month / year differs from the previous step's, as
    dam_register.c:304-307); steps [T,12] float, days_per_month(i + 1, year)
    * model_steps_per_day for the year of that step (dam_register.c:129-133);
    new_month_host a list of Python bools (the host knows the calendar, so a
    caller can skip the month-change block on the other steps without a
    device read)."""
    __slots__ = ()

    def at(self, t):
        """The flags of step t (0-d tensors; `steps` is [12]).  t is a
        Python int or a long tensor (no host sync either way)."""
        if torch.is_tensor(t):
            ix = t.reshape(1)
            g = lambda a: torch.index_select(a, 0, ix)[0]
        else:
            g = lambda a: a[t]
        return DamStep(g(self.year), g(self.month), g(self.new_month),
                       g(self.new_year), g(self.steps),
                       None if torch.is_tensor(t) else self.new_month_host[t])


DamStep = collections.namedtuple(
    'DamStep', 'year month new_month new_year steps new_month_host')


def make_calendar(years, months, calendar='standard', steps_per_day=1,
                  device='cpu', dtype=torch.float64):
    """DamCalendar from the (year, month) of every model step."""
    ys = [int(v) for v in years]
    ms = [int(v) for v in months]
    nm = [t > 0 and ms[t] != ms[t - 1] for t in range(len(ys))]
    ny = [t > 0 and ys[t] != ys[t - 1] for t in range(len(ys))]
    st = [[days_per_month(i + 1, y, calendar) * steps_per_day
           for i in range(MONTHS)] for y in ys]
    return DamCalendar(
        torch.tensor(ys, dtype=torch.long, device=device),
        torch.tensor(ms, dtype=torch.long, device=device),
        torch.tensor(nm, dtype=torch.bool, device=device),
        torch.tensor(ny, dtype=torch.bool, device=device),
        torch.tensor(st, dtype=dtype, device=device), nm)


def rank_in_cell(cell):
    """Position of each dam among the dams of its cell in list order (the
    class order of C), a numpy int array.  Run the sets of rank 0, 1, ...
    one after the other."""
    import numpy as np
    cell = np.asarray(cell)
    rank = np.zeros(len(cell), dtype=np.int64)
    seen = {}
    for i, c in enumerate(cell.tolist()):
        rank[i] = seen.get(c, 0)
        seen[c] = rank[i] + 1
    return rank


# ------------------------------------------------------------------- kernels
def array_average(hist, years_running):
    """array_average(history, years_running, 1, i, 11) for every offset i
    (plugins/general/src/array_average.c:33-56): the mean over the first
    `years_running` years of month slot i + 12 j, accumulated year by year in
    C order; 0 when years_running = 0.  hist [D,60], years_running [D] long
    -> [D,12]."""
    h = hist.reshape(hist.shape[0], HIST_YEARS, MONTHS)
    acc = torch.zeros_like(h[:, 0])
    for j in range(HIST_YEARS):
        acc = acc + torch.where((years_running > j)[:, None], h[:, j],
                                torch.zeros_like(h[:, j]))
    reps = years_running.to(hist.dtype)[:, None]
    return torch.where(reps > 0, acc / reps.clamp(min=1), torch.zeros_like(acc))


def years_running(months_running):
    """dam_register.c:57-60 and :113-116."""
    return torch.clamp(months_running // MONTHS, max=HIST_YEARS)


def climatology(hist, months_running):
    """The flipped array_average of dam_register_operation: [:,0] is the
    current month, [:,m] the month m later (dam_register.c:136-147;
    sort.c:155-165)."""
    return array_average(hist, years_running(months_running)).flip(1)


def seq_sum(x):
    """Sum over the last axis in index order, as a C loop."""
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        acc = acc + x[..., i]
    return acc


def opt_release(inflow, demand, efr):
    """dam_calc_opt_release (dam_operation.c:33-87) for [D,12] arrays: the
    monthly Hanasaki release including the environmental requirement."""
    ia = seq_sum(inflow - efr) / MONTHS
    da = seq_sum(demand) / MONTHS
    ib = torch.where(ia < 0, torch.zeros_like(ia), ia)
    return hanasaki_release(ib[:, None], da[:, None], demand, efr,
                            ratio_first=True)


def corr_opt_release(inflow, release, k, c):
    """dam_corr_opt_release (dam_operation.c:93-119); k, c [D]."""
    return hanasaki_capacity(release, inflow, k, c)


def k_factor(capacity, storage, alpha):
    """dam_calc_k_factor (dam_operation.c:124-135), not clamped."""
    return storage / (capacity * alpha)


def c_factor(inflow, capacity, steps):
    """dam_calc_c_factor (dam_operation.c:140-160); inflow [D,12] hm3 per
    step, steps [12] or [D,12] steps per month.  0 total gives inf."""
    return capacity / seq_sum(inflow * steps)


def opt_storage(inflow, release, steps, cur_storage):
    """dam_calc_opt_storage (dam_operation.c:165-186), [D,12]."""
    out = [cur_storage]
    for i in range(1, MONTHS):
        out.append(out[-1] + (inflow[:, i - 1] - release[:, i - 1])
                   * steps[..., i - 1])
    return torch.stack(out, 1)


def corr_release(release, storage, capacity, alpha, beta, gamma, spd=1):
    """dam_corr_release (dam_operation.c:230-265): the per-step release
    correction towards the band [(1 - alpha) C, alpha C]."""
    hi = capacity * alpha
    lo = capacity - capacity * alpha
    zero = torch.zeros_like(storage)
    diff = torch.where(storage > hi, storage - hi,
                       torch.where(storage < lo, storage - lo, zero))
    # the C max() macro, a > b ? a : b (NaN-propagating the same way)
    a, b = 1 - storage / capacity, storage / capacity
    f = torch.where(a > b, a, b)
    f = torch.pow((f - alpha) / (1 - alpha), beta)
    f = torch.where(diff != 0, f, torch.ones_like(f))
    corrected = release + diff * f / (gamma * spd)
    return torch.where(0 > corrected, zero, corrected)


def operate(storage, inflow, op0, capacity, alpha, beta, gamma, spd=1):
    """dam_operate (dam_run.c:33-62): fill, correct release, empty, full.
    Returns (storage, release)."""
    s = storage + inflow
    r = corr_release(op0, s, capacity, alpha, beta, gamma, spd)
    s = s - r
    neg = s < 0
    r = torch.where(neg, r + s, r)
    s = torch.where(neg, torch.zeros_like(s), s)
    over = s > capacity
    r = torch.where(over, r + (s - capacity), r)
    s = torch.where(over, capacity, s)
    return s, r


def operation_start(history_inflow, months_running, month):
    """dam_register_operation_start (dam_register.c:33-99): the first month
    of the operational year (1-12) from the mean climatology.  The scan
    covers 24 months from `month` and follows the longest cumulative run
    above the annual mean; with no history it returns 1.  month is a 0-d
    long tensor."""
    infl = climatology(history_inflow, months_running)
    avg = seq_sum(infl) / MONTHS
    cum = torch.zeros_like(avg)
    cmax = torch.zeros_like(avg)
    mmax = torch.zeros(avg.shape, dtype=torch.long, device=avg.device)
    for i in range(2 * MONTHS):
        mi = (month - 1 + i) % MONTHS
        x = torch.index_select(infl, 1, mi.reshape(1))[:, 0]
        above = x > avg
        cum = torch.where(above, cum + x, torch.zeros_like(cum))
        upd = above & (cum > cmax)
        cmax = torch.where(upd, cum, cmax)
        mmax = torch.where(upd, mi + 1, mmax)
    mmax = mmax + 1
    return torch.where(mmax > MONTHS, mmax - MONTHS, mmax)


def service_demand(demand_surf, sector_active, svc, svc_frac, svc_valid, area):
    """Dam demand in hm3 per step (dam_register.c:200-228).

    demand_surf [C,S] mm per step of the surface demand of every cell and
    sector; sector_active [C,S] bool (the C loop runs over the cell's active
    sectors); svc [D,M] long cell index of each service cell (any valid index
    where svc_valid is False), svc_frac, svc_valid [D,M], area [D] or [D,M]
    m2: the dam cell's area in exact mode (the C quirk), the service cell's
    otherwise.  Terms in C order: ds * area * frac / 1000 / 1e6, summed over
    service cells then sectors."""
    nd, nm_ = svc.shape
    area = area if area.dim() == 2 else area[:, None].expand(nd, nm_)
    dem = torch.zeros(nd, dtype=demand_surf.dtype, device=demand_surf.device)
    for m in range(nm_):
        cell = svc[:, m]
        for s in range(demand_surf.shape[1]):
            t = demand_surf[cell, s] * area[:, m] * svc_frac[:, m] \
                / MM_PER_M / M3_PER_HM3
            ok = svc_valid[:, m] & sector_active[cell, s]
            dem = dem + torch.where(ok, t, torch.zeros_like(t))
    return dem


# ------------------------------------------------------------------ dam sets
class _Dams:
    """State and constants of a set of dams, one per cell.

    capacity [D] hm3, inflow_frac [D], build_year [D] int.  alpha, beta,
    gamma (days) and the model steps per day are scalars (plugin_param,
    global_param).  Cold start S = alpha C (dam_populate.c:33-51)."""

    def __init__(self, capacity, inflow_frac, build_year, alpha=0.85,
                 beta=0.6, gamma=5.0, steps_per_day=1, exact=False,
                 device=None, dtype=torch.float64):
        dev = torch.device(device) if device is not None else None
        f = lambda a: torch.as_tensor(a, dtype=dtype, device=dev)
        self.capacity = f(capacity)
        self.inflow_frac = f(inflow_frac)
        self.build_year = torch.as_tensor(build_year, dtype=torch.long,
                                          device=self.capacity.device)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.spd = int(steps_per_day)
        self.exact = bool(exact)
        self.D = self.capacity.shape[0]
        self.dev, self.dtype = self.capacity.device, dtype
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=self.dev)
        zl = lambda *s: torch.zeros(*s, dtype=torch.long, device=self.dev)
        D = self.D
        self.active = torch.zeros(D, dtype=torch.bool, device=self.dev)
        self.storage = self.capacity * self.alpha
        self.inflow, self.demand, self.efr, self.release = z(D), z(D), z(D), z(D)
        self.history_inflow, self.history_demand = z(D, HIST), z(D, HIST)
        self.history_efr = z(D, HIST)
        self.op_release, self.op_storage = z(D, MONTHS), z(D, MONTHS)
        self.total_inflow, self.total_demand, self.total_efr = z(D), z(D), z(D)
        self.register_steps = zl(D)
        self.op_month = torch.full((D,), NODATA_DAM, dtype=torch.long,
                                   device=self.dev)
        self.months_running = zl(D)

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in STATE}

    def load_state_dict(self, sd):
        for k in STATE:
            getattr(self, k).copy_(sd[k].to(self.dev))

    def _operate(self):
        """dam_run.c:33-62 for the active dams; inactive dams zero their
        inflow, demand and efr and keep their release (dam_run.c:171-186)."""
        s, r = operate(self.storage, self.inflow, self.op_release[:, 0],
                       self.capacity, self.alpha, self.beta, self.gamma,
                       self.spd)
        a = self.active
        self.storage = torch.where(a, s, self.storage)
        self.release = torch.where(a, r, self.release)
        z = torch.zeros_like(self.inflow)
        self.inflow = torch.where(a, self.inflow, z)
        self.demand = torch.where(a, self.demand, z)
        self.efr = torch.where(a, self.efr, z)


class GlobalDams(_Dams):
    """Global (river) dams, C2: inflow is the cell's river discharge, the
    release is written back to the routing ring."""

    def _steps(self, st):
        """Steps per month for c and op_storage: calendar indexed (C) or
        aligned with the climatology that starts at the current month."""
        if self.exact:
            return st.steps
        ix = (st.month - 1 + torch.arange(MONTHS, device=self.dev)) % MONTHS
        return torch.index_select(st.steps, 0, ix)

    def _register_history(self, nm):
        """dam_register_history (dam_register.c:153-170) where nm."""
        n = self.register_steps
        shift = lambda h, m: torch.cat([m[:, None], h[:, :-1]], 1)
        roll = lambda a: torch.cat([a[:, 1:], a[:, :1]], 1)
        self.history_inflow = torch.where(
            nm, shift(self.history_inflow, self.total_inflow / n),
            self.history_inflow)
        self.history_demand = torch.where(
            nm, shift(self.history_demand, self.total_demand / n),
            self.history_demand)
        self.history_efr = torch.where(
            nm, shift(self.history_efr, self.total_efr / n), self.history_efr)
        self.op_release = torch.where(nm, roll(self.op_release), self.op_release)
        self.op_storage = torch.where(nm, roll(self.op_storage), self.op_storage)
        z = torch.zeros_like(self.total_inflow)
        self.total_inflow = torch.where(nm, z, self.total_inflow)
        self.total_demand = torch.where(nm, z, self.total_demand)
        self.total_efr = torch.where(nm, z, self.total_efr)
        self.register_steps = torch.where(
            nm, torch.zeros_like(n), n)
        self.months_running = self.months_running + nm.to(torch.long)

    def _plan(self, mask, st):
        """dam_register_operation (dam_register.c:104-148) where mask [D]."""
        mr = self.months_running
        ci = climatology(self.history_inflow, mr)
        cd = climatology(self.history_demand, mr)
        ce = climatology(self.history_efr, mr)
        steps = self._steps(st)
        rel = opt_release(ci, cd, ce)
        k = k_factor(self.capacity, self.storage, self.alpha)
        c = c_factor(ci, self.capacity, steps)
        rel = corr_opt_release(ci, rel, k, c)
        sto = opt_storage(ci, rel, steps, self.storage)
        self.op_release = torch.where(mask[:, None], rel, self.op_release)
        self.op_storage = torch.where(mask[:, None], sto, self.op_storage)

    def register(self, st, discharge, dt, demand=None, efr_q=None):
        """global_dam_register (dam_register.c:299-340).

        st: `DamCalendar.at(t)`; discharge [D] m3/s of the dam cell as the
        ring holds it now (after this step's local water use, before the
        dam); dt s; demand [D] hm3 (`service_demand`, WATERUSE) and efr_q
        [D] m3/s (EFR forcing), None when the plugin is off (the C value
        then stays 0)."""
        if st.new_month_host is None or st.new_month_host:
            nm = st.new_month
            self._register_history(nm)
            ny = nm & st.new_year
            self.op_month = torch.where(
                ny, operation_start(self.history_inflow, self.months_running,
                                    st.month), self.op_month)
            at_op = nm & (self.op_month == st.month)
            self.active = self.active | (at_op & (st.year >= self.build_year))
            self._plan(at_op, st)
        self.inflow = discharge * dt * self.inflow_frac / M3_PER_HM3
        if demand is not None:
            self.demand = demand
        if efr_q is not None:
            self.efr = efr_q * dt * self.inflow_frac / M3_PER_HM3   # efr.dam_efr
        self.total_inflow = self.total_inflow + self.inflow
        self.total_demand = self.total_demand + self.demand
        self.total_efr = self.total_efr + self.efr
        self.register_steps = self.register_steps + 1

    def write_back(self, ring, discharge, dt):
        """global_dam_operate write-back (dam_run.c:97-127): the first n bins
        of the ring [D,n] become q (1 - frac) + R 1e6 / dt / n, clipped at 0,
        and the cell discharge is their sum; inactive dams keep their ring.
        discharge [D] is the value to keep for the inactive ones."""
        n = ring.shape[1]
        add = self.release * M3_PER_HM3 / dt / n
        q = ring * (1 - self.inflow_frac)[:, None] + add[:, None]
        q = torch.where(q < 0, torch.zeros_like(q), q)
        d = seq_sum(q)
        a = self.active
        return (torch.where(a[:, None], q, ring), torch.where(a, d, discharge))

    def run(self, st, ring, discharge, dt, demand=None, efr_q=None):
        """global_dam_run for the set (dam_run.c:163-187): register, operate
        the active dams, write back.  ring [D,n] (n = rout_steps_per_day /
        model_steps_per_day sub-steps), discharge [D].  Returns the new
        (ring, discharge)."""
        self.register(st, discharge, dt, demand, efr_q)
        self._operate()
        return self.write_back(ring, discharge, dt)


class LocalDams(_Dams):
    """Local dams, C3: inflow is the cell's runoff, the release replaces it
    in OUT_RUNOFF.  They never plan (op_release stays 0), so the release is
    the per-step correction plus spill only."""

    def register(self, st, runoff, area):
        """local_dam_register (dam_register.c:263-297); runoff [D] mm per
        step, area [D] m2 of the dam cell."""
        nm = st.new_month if (st.new_month_host is None
                              or st.new_month_host) else None
        if nm is not None:
            self.active = self.active | (nm & (st.year >= self.build_year))
        self.inflow = runoff * area * self.inflow_frac / M3_PER_HM3 / MM_PER_M

    def write_back(self, runoff, area):
        """local_dam_operate write-back (dam_run.c:78-91):
        runoff (1 - frac) + R 1e6 1000 / area for the active dams."""
        new = runoff * (1 - self.inflow_frac) \
            + self.release * M3_PER_HM3 * MM_PER_M / area
        return torch.where(self.active, new, runoff)

    def run(self, st, runoff, area):
        """local_dam_run for the set (dam_run.c:133-157); returns the new
        runoff [D] mm per step."""
        self.register(st, runoff, area)
        self._operate()
        return self.write_back(runoff, area)
