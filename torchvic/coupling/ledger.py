"""The water ledger: every transfer between modules and every module's
balance, per exchange interval, and the checks that reject a step whose
water does not add up.

For one module over an interval [t0, t1) (Balance, volumes in m3):

    R = P + I_ext - ET - Q_out - E_other - dS - closure

P is precipitation from outside the coupled system, I_ext water another
module or the boundary added, ET evaporation and transpiration, Q_out what
left (to another module or out of the domain), E_other other losses, dS the
module's storage change and `closure` the model's own reported balance error
(VIC's OUT_WATER_ERROR: VIC does not close its balance exactly where it
withholds evaporation from a layer at residual, runoff.c:108-133, and the
ledger must not call VIC's known residual a coupling error).

For the coupled domain the transfers between modules cancel: what a module
sends (counted in its Q_out) is either delivered (in the receiver's I_ext) or
still in transit in the runtime's exchange buffers, so

    R_coupled = sum_m R_m + internal_out - internal_in - dS_buffers

where internal_in is measured at the receiver after spatial mapping and
dS_buffers from the buffers' own contents: a mapping that loses or creates
water shows up here.

Tolerances come from the floating-point precision of the values summed:
|R| <= factor * eps * n_terms * sum(|terms|), eps the dtype's machine
epsilon (2.2e-16 for float64) and n_terms the number of values the terms
were summed from (rounding of a sum of n values grows at most like n eps).
"""
import csv
import math

import torch

#: default safety factor on the precision bound (4: one rounding per
#: operation of the sum, the conversion and the difference)
DEFAULT_FACTOR = 4.0


class BalanceViolation(Exception):
    """A module's or the coupled domain's balance did not close."""


def tolerance(balance, factor=DEFAULT_FACTOR):
    return (factor * balance.eps * max(int(balance.n_terms), 1)
            * balance.magnitude())


def cell_vector(per_cell, factor=DEFAULT_FACTOR):
    """Device side of the per-element check of a Balance.per_cell
    ({'residual': [n] mm, 'scale': [n] mm, 'n_terms': int, 'eps': float}):
    the 4-vector [worst excess, residual there, tolerance there, index]
    (no host sync; a non-finite residual is the worst element)."""
    res = per_cell['residual']
    tol = factor * per_cell['eps'] * max(int(per_cell['n_terms']), 1) \
        * per_cell['scale']
    excess = torch.where(torch.isfinite(res), res.abs() - tol,
                         torch.full_like(res, float('inf')))
    i = torch.argmax(excess).reshape(1)
    # index_select, not excess[i]: indexing by a 0-d tensor reads it on
    # the host
    return torch.cat([excess.index_select(0, i), res.index_select(0, i),
                      tol.index_select(0, i), i.to(res.dtype)])


def cell_result(vec):
    """(ok, worst index, worst residual, its tolerance) from the host
    values of `cell_vector`."""
    e, r, t, ii = vec
    return e <= 0.0, int(ii), r, t


def check_cells(per_cell, factor=DEFAULT_FACTOR):
    """The per-element check of a Balance.per_cell: (ok, worst index, worst
    residual, its tolerance). Works on any device; one host sync."""
    from .hostfetch import host
    return cell_result(host(cell_vector(per_cell, factor)))


_KEYS = ('P', 'I_ext', 'ET', 'Q_out', 'E_other', 'dS', 'closure')


def gather(balance, F, factor=DEFAULT_FACTOR):
    """Queue every device value of `balance` on the Fetch `F` (no sync);
    `resolve` turns the handles into the host balance after F.run()."""
    import dataclasses
    vals = {k: getattr(balance, k) for k in _KEYS}
    anyt = any(torch.is_tensor(v) for v in list(vals.values())
               + [balance.scale])
    h = {}
    if anyt:
        for k, v in vals.items():
            h[k] = F.add(0.0 if v is None else v)
        h['scale'] = None if balance.scale is None else F.add(balance.scale)
    cell = None
    if balance.per_cell is not None:
        cell = F.add(cell_vector(balance.per_cell, factor))
    det = {k: F.add(v) for k, v in balance.detail.items()
           if torch.is_tensor(v)}
    return (balance, h if anyt else None, cell, det)


def resolve(g):
    import dataclasses
    balance, h, cell, det = g
    if det:
        balance = dataclasses.replace(balance, detail=dict(
            balance.detail, **{k: x.value() for k, x in det.items()}))
    if h is not None:
        kw = {k: h[k].value() for k in _KEYS}
        balance = dataclasses.replace(
            balance, **kw,
            scale=None if h['scale'] is None else h['scale'].value())
    return balance, (None if cell is None else cell_result(cell.value()))


def materialise(balance):
    """`balance` with every term a Python float (one device-to-host copy
    for all of them when they are device tensors)."""
    from .hostfetch import Fetch
    F = Fetch()
    g = gather(balance, F)
    F.run()
    return resolve(g)[0]


class Ledger(object):
    """Rows of transfers and balances, kept as plain floats (cheap to keep
    for a long run, picklable for a checkpoint)."""

    def __init__(self, factor=DEFAULT_FACTOR, on_violation='reject'):
        if on_violation not in ('reject', 'warn'):
            raise ValueError("ledger on_violation must be 'reject' or "
                             "'warn', not %r" % (on_violation,))
        self.factor = float(factor)
        self.on_violation = on_violation
        self.transfers = []
        self.balances = []
        self.coupled = []
        self.warnings = []

    # -- rows -----------------------------------------------------------------
    @staticmethod
    def transfer(source, destination, quantity, amount, units, location,
                 t0, t1, internal=True):
        return dict(source=source, destination=destination,
                    quantity=quantity, amount=float(amount), units=units,
                    location=location, t0=float(t0), t1=float(t1),
                    internal=bool(internal))

    def check_module(self, name, t0, t1, balance, resolved=None):
        """(row, [problems]) for one module's Balance. Terms held as device
        tensors are brought to the host here, once (the only sync), or
        come already on the host in `resolved` = (balance, cell result)
        from `gather`/`resolve` (the runtime batches a whole step or
        window in one copy)."""
        if resolved is None:
            from .hostfetch import Fetch
            F = Fetch()
            g = gather(balance, F, self.factor)
            F.run()
            resolved = resolve(g)
        balance, cres = resolved
        r = balance.residual()
        tol = tolerance(balance, self.factor)
        problems = []
        if not math.isfinite(r) or abs(r) > tol:
            problems.append('%s water balance over [%g, %g) s does not '
                            'close: R = %.6e m3, tolerance %.3e m3 (P %.6e, '
                            'I_ext %.6e, ET %.6e, Q_out %.6e, E_other %.6e, '
                            'dS %.6e, closure %.6e)'
                            % (name, t0, t1, r, tol, balance.P,
                               balance.I_ext, balance.ET, balance.Q_out,
                               balance.E_other, balance.dS,
                               balance.closure))
        cell = None
        if balance.per_cell is not None:
            ok, i, ri, ti = cres
            cell = dict(worst=i, residual_mm=ri, tolerance_mm=ti)
            if not ok:
                problems.append('%s water balance does not close at element '
                                '%d: R = %.6e mm, tolerance %.3e mm'
                                % (name, i, ri, ti))
        row = dict(module=name, t0=float(t0), t1=float(t1), P=balance.P,
                   I_ext=balance.I_ext, ET=balance.ET, Q_out=balance.Q_out,
                   E_other=balance.E_other, dS=balance.dS,
                   closure=balance.closure, residual=r, tolerance=tol,
                   ok=not problems, cell=cell, **{
                       'detail_' + k: v for k, v in balance.detail.items()})
        return row, problems

    def check_coupled(self, t0, t1, rows, transfers, dS_buffers,
                      buffer_magnitude=0.0):
        """(row, [problems]) for the coupled domain over [t0, t1), from the
        module rows of the interval and its transfers (see module doc)."""
        R = sum(r['residual'] for r in rows)
        out_ = sum(x['amount'] for x in transfers
                   if x['internal'] and x.get('leg') == 'sent')
        in_ = sum(x['amount'] for x in transfers
                  if x['internal'] and x.get('leg') == 'delivered')
        Rc = R + out_ - in_ - dS_buffers
        tol = sum(r['tolerance'] for r in rows) + self.factor * \
            2.220446049250313e-16 * (abs(out_) + abs(in_)
                                     + abs(buffer_magnitude)) * 4
        row = dict(t0=float(t0), t1=float(t1), residual=Rc, tolerance=tol,
                   internal_sent=out_, internal_delivered=in_,
                   dS_buffers=dS_buffers,
                   P=sum(r['P'] for r in rows),
                   ET=sum(r['ET'] for r in rows),
                   dS=sum(r['dS'] for r in rows) + dS_buffers,
                   Q_boundary=sum(r['Q_out'] for r in rows) - out_,
                   I_boundary=sum(r['I_ext'] for r in rows) - in_)
        problems = []
        if not math.isfinite(Rc) or abs(Rc) > tol:
            problems.append('coupled water balance over [%g, %g) s does not '
                            'close: R = %.6e m3, tolerance %.3e m3 (sent '
                            '%.6e, delivered %.6e, in transit change %.6e)'
                            % (t0, t1, Rc, tol, out_, in_, dS_buffers))
        row['ok'] = not problems
        return row, problems

    def accept(self, module_rows, coupled_row, transfers):
        self.balances.extend(module_rows)
        if coupled_row is not None:
            self.coupled.append(coupled_row)
        self.transfers.extend(transfers)

    # -- summaries -------------------------------------------------------------
    def totals(self, module=None):
        rows = [r for r in self.balances
                if module is None or r['module'] == module]
        keys = ('P', 'I_ext', 'ET', 'Q_out', 'E_other', 'dS', 'closure',
                'residual')
        return {k: sum(r[k] for r in rows) for k in keys}

    def state(self):
        return dict(factor=self.factor, on_violation=self.on_violation,
                    transfers=list(self.transfers),
                    balances=list(self.balances), coupled=list(self.coupled),
                    warnings=list(self.warnings))

    def load(self, blob):
        self.factor = blob['factor']
        self.on_violation = blob['on_violation']
        self.transfers = list(blob['transfers'])
        self.balances = list(blob['balances'])
        self.coupled = list(blob['coupled'])
        self.warnings = list(blob.get('warnings', []))

    def write_csv(self, path):
        """The module balance rows as CSV (one row per module and
        interval)."""
        keys = ['module', 't0', 't1', 'P', 'I_ext', 'ET', 'Q_out', 'E_other',
                'dS', 'closure', 'residual', 'tolerance', 'ok']
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for r in self.balances:
                w.writerow([r[k] for k in keys])
        return path
