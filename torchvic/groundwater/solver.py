"""Preconditioned conjugate gradient on the structured 7-point stencil.

The symmetric positive-definite system is stored as diagonals only:

    (A x)_n = diag_n x_n - sum_faces c_nm x_m

with six full-shape (L, R, C) neighbour coefficients ``c = (cE, cW, cS, cN,
cD, cU)`` (east = column + 1, south = row + 1, down = layer + 1; zero at the
grid edge and between a free and a fixed cell; padded once per assembly,
which fuses better than padding face arrays inside every mat-vec). A
mat-vec is shifted slices; no sparse matrix object and no atomics.

Functions update the buffers they are given in place and never read a
value back to the host, so they can be fused by ``torch.compile`` and
captured in a CUDA graph. Iteration control lives on the device: ``done``
freezes the iterate once the MODFLOW 6 IMS inner closure holds (update
``max|alpha p| <= inner_dvclose`` and residual ``max|r| <= inner_rclose``).

Preconditioners: ``jacobi`` (diagonal) and ``rbgs`` (red-black symmetric
Gauss-Seidel, M = (D-L) D^-1 (D-U) in red-black order: symmetric, fully
parallel).
"""
import torch
import torch.nn.functional as F


def faces_to_full(cx, cy, cz):
    """Face conductances (L,R,C-1), (L,R-1,C), (L-1,R,C) -> six full-shape
    neighbour coefficients (cE, cW, cS, cN, cD, cU)."""
    return (F.pad(cx, (0, 1)), F.pad(cx, (1, 0)),
            F.pad(cy, (0, 0, 0, 1)), F.pad(cy, (0, 0, 1, 0)),
            F.pad(cz, (0, 0, 0, 0, 0, 1)), F.pad(cz, (0, 0, 0, 0, 1, 0)))


def nbsum(v, c):
    """sum over the six neighbours of c * v_neighbour."""
    cE, cW, cS, cN, cD, cU = c
    out = (cE * F.pad(v[..., 1:], (0, 1)) + cW * F.pad(v[..., :-1], (1, 0))
           + cS * F.pad(v[..., 1:, :], (0, 0, 0, 1))
           + cN * F.pad(v[..., :-1, :], (0, 0, 1, 0)))
    if v.shape[0] > 1:
        out = out + (cD * F.pad(v[1:], (0, 0, 0, 0, 0, 1))
                     + cU * F.pad(v[:-1], (0, 0, 0, 0, 1, 0)))
    return out


def matvec(v, diag, c):
    return diag * v - nbsum(v, c)


def precondition(r, minv, c, red, kind):
    """z = M^-1 r; ``minv`` = 1/diag (a multiply: FP64 division is slow on
    consumer GPUs)."""
    if kind == 'jacobi':
        return r * minv
    zero = torch.zeros_like(r)
    y = torch.where(red, r * minv, zero)
    y = torch.where(red, y, (r + nbsum(y, c)) * minv)
    yb = torch.where(red, zero, y)
    return torch.where(red, y + nbsum(yb, c) * minv, y)


def pcg_start(x, b, diag, c, minv, red, r, p, rz, done, it, frozen, kind):
    """Initial residual and direction; ``frozen`` (0-d bool) starts the
    solve already done (masked no-op solve)."""
    minv.copy_(1.0 / diag)
    r0 = b - matvec(x, diag, c)
    z = precondition(r0, minv, c, red, kind)
    r.copy_(r0)
    p.copy_(z)
    rz.copy_(torch.sum(r0 * z))
    done.copy_(frozen)
    it.zero_()


def pcg_iter(x, diag, c, minv, red, r, p, rz, done, it, dvclose, rclose,
             kind, check=True):
    """One PCG iteration, frozen (no-op) once ``done``. The closure test
    (two max-abs reductions) runs only when ``check``: extra iterations
    before the test only refine x."""
    Ap = matvec(p, diag, c)
    pap = torch.sum(p * Ap)
    act = ~done
    alpha = torch.where(act & (pap > 0), rz / pap, torch.zeros_like(pap))
    step = alpha * p
    rn = r - alpha * Ap
    x.copy_(x + step)
    z = precondition(rn, minv, c, red, kind)
    rzn = torch.sum(rn * z)
    beta = torch.where(rz > 0, rzn / rz, torch.zeros_like(rzn))
    r.copy_(rn)
    p.copy_(z + beta * p)
    rz.copy_(rzn)
    it.copy_(it + act.to(it.dtype))
    if check:
        conv = (step.abs().amax() <= dvclose) & (rn.abs().amax() <= rclose)
        done.copy_(done | (act & conv))
