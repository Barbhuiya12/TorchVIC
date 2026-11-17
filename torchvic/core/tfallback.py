"""TFALLBACK: what happens when a temperature solve finds no root.

VIC 5.1.0 guards every temperature solve with `root_brent`'s -998 and
`options.TFALLBACK` (initialize_options.c:51 sets it TRUE):

    TRUE  the solve keeps its previous temperature and counts a fallback
          (Tsurf_fbcount, Tcanopy_fbcount, Tfoliage_fbcount,
          surf_temp_fbcount, Tfbcount[j])
    FALSE the solve calls its error_* routine, which dumps the solve's
          variables, and returns ERROR: vic_run abandons that cell's step.
          What the image driver does next (vic_image_run.c:49 ignores the
          return value) was measured on pristine vic_image.exe 5.1.0: after
          a failed surface solve the error dump itself dies (SIGSEGV, exit
          -11) at that step; after a failed canopy foliage or snow pack solve
          the run carries on with that cell's state half advanced. Neither
          is a result to reproduce, so TorchVIC stops in both cases, at the
          first step and cell VIC fails at (the log's first root_brent
          "failed to bracket the root", Gridcell io_idx and timestep).

The solves, and the C that guards each (reference/VIC-5.1.0/vic/vic_run/src):

    'canopy foliage'   snow_intercept.c:342   (Tfoliage, canopy snow)
    'snow pack'        snow_melt.c:267        (pack surface temperature)
    'surface'          calc_surf_energy_bal.c:311 and :434 (bare/soil surface)
    'canopy air'       calc_atmos_energy_bal.c:87 (CLOSE_ENERGY)
    'ice pack'         ice_melt.c:346         (lakes: not run, LAKES refused)
    'soil node'        frozen_soil.c:408, :465, :515 (the finite-difference
                       profile: not run, QUICK_FLUX FALSE is refused)

The solvers here all work on whole tensors, so each reports a boolean mask
of the elements whose bracket held no root (`note`), and the model calls
`end_step` once the step is done. With TFALLBACK TRUE `end_step` only adds
the masks to the run's counts (a device tensor sum, no host wait); with
FALSE it raises `TemperatureSolveError` for the first failing cell (lowest
cell index, as VIC's loop over cells meets it first), naming the step, the
cell, the tile and each solve that failed there.

The CUDA-graph path cannot ask the device a question, so it records nothing
(the model does not take it when TFALLBACK is FALSE).
"""
import torch

#: VIC's order of calls within a cell and step (solve_snow.c, then
#: surface_fluxes.c), used only to list the solves that failed together.
ORDER = ('canopy foliage', 'snow pack', 'surface', 'canopy air')


class TemperatureSolveError(RuntimeError):
    """A temperature solve found no root and TFALLBACK is FALSE."""

    def __init__(self, step, cell, tile, kinds, detail=''):
        self.step, self.cell, self.tile, self.kinds = step, cell, tile, kinds
        self.detail = detail
        RuntimeError.__init__(
            self, 'TFALLBACK FALSE: temperature solve failed (%s) at step %d, '
            'cell %d%s%s -- VIC 5.1.0 stops here (the solve\'s bracket holds '
            'no root; with TFALLBACK TRUE it keeps the previous temperature)'
            % (', '.join(kinds), step, cell,
               '' if tile is None else ', tile %d' % tile, detail))


class _State(object):
    fallback = True        # TFALLBACK
    pending = []           # [(kind, bool tensor with leading cell axis)]
    counts = {}            # kind -> device tensor, fallbacks so far
    step = 0
    suspended = 0          # >0: record nothing (graph capture)


S = _State()


def start(fallback=True):
    """Begin a run: TFALLBACK's value, counts at zero."""
    S.fallback = bool(fallback)
    S.pending = []
    S.counts = {}
    S.step = 0
    S.suspended = 0


def set_step(t):
    S.step = int(t)


def note(kind, failed):
    """Record the elements of `failed` ([cell, ...] bool) whose `kind` solve
    found no root. The caller has already restricted it to the elements that
    run that solve in VIC."""
    if S.suspended or failed is None:
        return
    from . import graphsafe
    if graphsafe.ACTIVE:
        return
    S.pending.append((kind, failed))


def mark():
    return len(S.pending)


def restrict(since, keep):
    """Keep only the entries recorded after `since` where `keep` is true
    (a [cell] or [cell, tile] mask): for a stage the model runs for every
    element and then selects from."""
    for i in range(since, len(S.pending)):
        kind, f = S.pending[i]
        k = keep
        while k.dim() > f.dim():
            k = k.any(-1)
        while k.dim() < f.dim():
            k = k.unsqueeze(-1)
        S.pending[i] = (kind, f & k)


def end_step(t=None):
    """Close a step: count the failures, or stop at the first one."""
    pend, S.pending = S.pending, []
    if not pend:
        return
    if S.fallback:
        for kind, f in pend:
            n = f.sum()
            S.counts[kind] = S.counts[kind] + n if kind in S.counts else n
        return
    rows = []
    for kind, f in pend:
        f = f.reshape(f.shape[0], -1)
        if bool(f.any()):
            rows.append((kind, f))
    if not rows:
        return
    cell = min(int(torch.nonzero(f.any(1))[0, 0]) for _, f in rows)
    kinds, tile = [], None
    for kind, f in sorted(rows, key=lambda r: ORDER.index(r[0])
                          if r[0] in ORDER else len(ORDER)):
        if bool(f[cell].any()) and kind not in kinds:
            kinds.append(kind)
            if tile is None and f.shape[1] > 1:
                tile = int(torch.nonzero(f[cell])[0, 0])
    raise TemperatureSolveError(S.step if t is None else int(t), cell, tile,
                                kinds)


def counts():
    """{solve: fallbacks taken} for the run so far (one host read)."""
    return {k: int(v) for k, v in S.counts.items()}
