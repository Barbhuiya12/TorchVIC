"""Cumulative sums over the last dimension in eager CUDA's own addition order, fast.

torch.cumsum on a CUDA tensor's last dimension (ATen/native/cuda/ScanUtils.cuh,
tensor_kernel_scan_innermost_dim) scans each row in chunks of 2 * num_threads_x elements:
the chunk's first element gets the running total of the chunks before it (x0 + total; for the
first chunk + 0), then a Sklansky scan runs in shared memory: at step m (m = 0, 1, ...) every
element i whose bit m is set adds the last element of the lower half of its aligned 2^(m+1)
block (row_buf[ti] = row_buf[ti] + row_buf[si]). num_threads_x comes from the row count and
length (get_log_num_threads_x_inner_scan, in uint32 arithmetic: with more rows than columns it
is 512 and a row of up to 1024 is one chunk). So the sum of element i is a fixed tree that is
not the CPU's sequential one, and eager on the GPU computes it with one thread block per row:
a [338 475, 3] cumsum takes 2.4 ms on an H200, a [338 475, 400] one 2.7 ms.

Here the same trees, bit for bit:
- rows of up to SMALL elements: written out as torch element-wise operations (a fused region
  turns them into part of one kernel);
- longer rows: a Triton kernel, one program per row, the chunks and the Sklansky steps as
  above (contraction off; only additions);
- cumsum_last(x) = cumsum(x)[..., -1] for a row that is one chunk: the last element only, from
  the pairwise block totals (the last element of an aligned 2^k block after k steps is
  upper total + lower total) folded over the set bits of R - 1 -- one read of the row.

On the CPU (sequential there) and for other dtypes these are torch.cumsum itself.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:          # pragma: no cover
    triton = None

#: rows up to this length are emulated with element-wise torch operations
SMALL = 8
#: rows a program of the one-row-per-lane kernels
LANES = 64


def log_threads_x(num_rows, row_size):
    """get_log_num_threads_x_inner_scan (ScanUtils.cuh), uint32 arithmetic included."""
    M = 0xFFFFFFFF
    lx = 0
    while (1 << lx) < row_size:
        lx += 1
    ly = 0
    while (1 << ly) < num_rows:
        ly += 1
    diff = (lx - ly) & M
    lx = ((9 + diff) & M) // 2
    return min(max(4, lx), 9)


def _chunk(num_rows, row_size):
    return 2 << log_threads_x(num_rows, row_size)


def _p2(n):
    return 1 << max(int(n) - 1, 0).bit_length()


def _emulate(x):
    """The one-chunk scan of a short row with element-wise operations: list of the R results."""
    R = x.shape[-1]
    v = [x[..., 0] + 0.0] + [x[..., i] for i in range(1, R)]
    m = 0
    while (1 << m) < R:
        s = 1 << m
        new = list(v)
        for i in range(R):
            if i & s:
                si = ((i >> (m + 1)) << (m + 1)) + s - 1
                new[i] = v[i] + v[si]
        v = new
        m += 1
    return v


def _use_torch(x):
    return (not x.is_cuda) or x.dtype != torch.float64 or x.dim() == 0 or x.numel() == 0 or triton is None


_KERNELS = {}

_HEAD = """import triton
import triton.language as tl


@triton.jit
def _pick(v, sel, axis: tl.constexpr):
    b = tl.where(sel, v.to(tl.int64, bitcast=True), 0)
    return tl.sum(b, axis=axis).to(tl.float64, bitcast=True)
"""


def _rows_per_program(width):
    return max(1, 2048 // width)


def _scan_source(R, CH, NCH, LAST):
    """ROWS rows a program: the chunks, their carry and the Sklansky steps, written out."""
    RW = _rows_per_program(CH)
    L = ["", "", "@triton.jit", "def kern(x_ptr, y_ptr, N):",
         "    r = tl.program_id(0).to(tl.int64) * %d + tl.arange(0, %d)" % (RW, RW),
         "    okr = (r < N)[:, None]",
         "    j = tl.arange(0, %d)[None, :]" % CH]
    for c in range(NCH):
        L.append("    col = %d + j" % (c * CH))
        L.append("    v = tl.load(x_ptr + r[:, None] * %d + col, mask=okr & (col < %d), other=0.0)" % (R, R))
        L.append("    v = tl.where(j == 0, v + %s, v)" % ("0.0" if c == 0 else "carry[:, None]"))
        s = 1
        while s < CH:
            L.append("    v2 = tl.reshape(v, (%d, %d, %d))" % (RW, CH // (2 * s), 2 * s))
            L.append("    jj = tl.arange(0, %d)[None, None, :]" % (2 * s))
            L.append("    last = _pick(v2, jj == %d, 2)" % (s - 1))
            L.append("    v = tl.reshape(tl.where(jj >= %d, v2 + last[:, :, None], v2), (%d, %d))" % (s, RW, CH))
            s *= 2
        if LAST:
            if c == NCH - 1:
                L.append("    tl.store(y_ptr + r[:, None] + j * 0, v, mask=okr & (col == %d))" % (R - 1))
        else:
            L.append("    tl.store(y_ptr + r[:, None] * %d + col, v, mask=okr & (col < %d))" % (R, R))
        if c < NCH - 1:
            L.append("    carry = _pick(v, j == %d, 1)" % (CH - 1))
    return "\n".join(L) + "\n"


def _last_source(R, P):
    """cumsum(x)[R - 1] of a one-chunk row. In the Sklansky scan, element i adds at each step m
    whose bit is set in i the last element of the lower half of its 2^(m+1) block, which then
    holds that 2^m block's pairwise total (upper half total + lower half total). These blocks
    tile [0, R - 1): each is read once and reduced pairwise, and the totals are added to x[R - 1]
    in ascending m (acc = acc + total, as row_buf[ti] + row_buf[si])."""
    RW = _rows_per_program(P)
    i = R - 1
    L = ["", "", "@triton.jit", "def kern(x_ptr, y_ptr, N):",
         "    r = tl.program_id(0).to(tl.int64) * %d + tl.arange(0, %d)" % (RW, RW),
         "    okr = r < N",
         "    acc = tl.load(x_ptr + r * %d + %d, mask=okr, other=0.0)" % (R, i)]
    if i == 0:
        L.append("    acc = acc + 0.0")
    m = 0
    while (1 << m) <= i:
        if (i >> m) & 1:
            start = (i >> (m + 1)) << (m + 1)
            n = 1 << m
            L.append("    c = %d + tl.arange(0, %d)[None, :]" % (start, n))
            L.append("    b = tl.load(x_ptr + r[:, None] * %d + c, mask=okr[:, None], other=0.0)" % R)
            if start == 0:
                L.append("    b = tl.where(c == 0, b + 0.0, b)")
            k = n
            while k > 1:
                L.append("    lo, hi = tl.split(tl.reshape(b, (%d, %d, 2)))" % (RW, k // 2))
                L.append("    b = hi + lo")
                k //= 2
            L.append("    acc = acc + tl.reshape(b, (%d,))" % RW)
        m += 1
    L.append("    tl.store(y_ptr + r, acc, mask=okr)")
    return "\n".join(L) + "\n"


U = 16          # elements a lane loads at once (static low bits of the element index)
TILED = True    # load them as one [BLOCK, U] tile and split it (coalesced)


def _lane_source(R, LAST, TILE=None):
    """One row per lane, for a row that is one chunk. Before element i, the elements [0, i) are
    the aligned blocks of the set bits of i (a binary counter), and after Sklansky step m the last
    element of the 2^m block holds its pairwise total (upper total + lower total). So element i
    is x_i plus those totals, smallest block first (row_buf[ti] + row_buf[si]); then x_i is
    pushed: it merges with the blocks of the trailing set bits of i (newer + older) into the
    block of the first clear bit. s<m> holds the total of the 2^m block while bit m is set.
    The row is read U elements at a time (the loads overlap); the low bits of i are then static,
    the higher ones the same in every lane (uniform branches)."""
    K = max(1, (R - 1).bit_length())
    u = U.bit_length() - 1
    tiled = TILE if TILE is not None else TILED
    L = ["", "", "@triton.jit", "def kern(x_ptr, y_ptr, N, BLOCK: tl.constexpr):",
         "    r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
         "    ok = r < N",
         "    xb = x_ptr + r * %d" % R,
         "    yb = y_ptr + r * %d" % R,
         "    z = tl.zeros([BLOCK], dtype=tl.float64)"]
    L += ["    s%d = z" % m for m in range(K)]

    def element(ind, c, i0, store, push):
        """Lines for element i = i0 + c: i0 a multiple of U (a name, dynamic) or None (i = c static)."""
        out = []
        x = "x%d" % c
        bit = lambda m: (((c >> m) & 1) == 1) if (i0 is None or m < u) else None
        if store:
            out.append(ind + "acc = %s" % x)
            for m in range(K):
                b = bit(m)
                if b is None:
                    out.append(ind + "if ((%s >> %d) & 1) != 0:" % (i0, m))
                    out.append(ind + "    acc = acc + s%d" % m)
                elif b:
                    out.append(ind + "acc = acc + s%d" % m)
            if LAST:
                out.append(ind + "tl.store(y_ptr + r, acc, mask=ok)")
            elif i0 is not None and tiled:
                out.append(ind + "y%d = acc" % c)
            else:
                out.append(ind + "tl.store(yb + %s, acc, mask=ok)" % (str(c) if i0 is None else "%s + %d" % (i0, c)))
        if push:
            out.append(ind + "carry = %s" % x)
            dyn = False
            for m in range(K):
                b = bit(m)
                if b is None:
                    if not dyn:
                        out.append(ind + "go = %s >= 0" % i0)
                        dyn = True
                    out.append(ind + "if go:")
                    out.append(ind + "    if ((%s >> %d) & 1) != 0:" % (i0, m))
                    out.append(ind + "        carry = carry + s%d" % m)
                    out.append(ind + "    else:")
                    out.append(ind + "        s%d = carry" % m)
                    out.append(ind + "        go = %s < 0" % i0)
                elif b:
                    out.append(ind + "carry = carry + s%d" % m)
                else:
                    out.append(ind + "s%d = carry" % m)
                    break
        return out

    n = R - 1 if LAST else R             # elements that are pushed (and stored when not LAST)
    full = (n // U) * U
    if full:
        L.append("    for i0 in range(0, %d, %d):" % (full, U))
        if tiled:
            # one coalesced [BLOCK, U] load, its columns taken apart by halves (even, odd)
            L.append("        cu = tl.arange(0, %d)[None, :]" % U)
            L.append("        t_ = tl.load(xb[:, None] + i0 + cu, mask=ok[:, None], other=0.0)")
            names = {(): "t_"}
            level = [()]
            w = U
            while w > 1:
                nxt = []
                for path in level:
                    a, b = names[path] + "e", names[path] + "o"
                    L.append("        %s, %s = tl.split(tl.reshape(%s, (BLOCK, %d, 2)))" % (a, b, names[path], w // 2)
                             if w > 2 else "        %s, %s = tl.split(%s)" % (a, b, names[path]))
                    names[path + (0,)], names[path + (1,)] = a, b
                    nxt += [path + (0,), path + (1,)]
                level = nxt
                w //= 2
            for path in level:
                col = sum(bit << k for k, bit in enumerate(path))
                L.append("        x%d = %s" % (col, names[path]))
        else:
            for c in range(U):
                L.append("        x%d = tl.load(xb + i0 + %d, mask=ok, other=0.0)" % (c, c))
        L.append("        x0 = tl.where(i0 == 0, x0 + 0.0, x0)")
        for c in range(U):
            L += element("        ", c, "i0", not LAST, True)
        if tiled and not LAST:
            # the U results joined back into one [BLOCK, U] tile (inverse of the split), one store
            level = {}
            for c in range(U):
                path = tuple((c >> k) & 1 for k in range(u))
                level[path] = "y%d" % c
            w, depth = 1, u
            while depth > 0:
                nxt = {}
                for path in sorted(set(p[:depth - 1] for p in level)):
                    a, b = level[path + (0,)], level[path + (1,)]
                    nm = "j%d_%s" % (depth, "".join(map(str, path)))
                    L.append("        %s = tl.reshape(tl.join(%s, %s), (BLOCK, %d))" % (nm, a, b, 2 * w))
                    nxt[path] = nm
                level, w, depth = nxt, 2 * w, depth - 1
            L.append("        tl.store(yb[:, None] + i0 + cu, %s, mask=ok[:, None])" % level[()])
    for i in range(full, R):              # the tail: i static
        L.append("    x%d = tl.load(xb + %d, mask=ok, other=0.0)%s" % (i, i, " + 0.0" if i == 0 else ""))
    for i in range(full, R):
        L += element("    ", i, None, (not LAST) or i == R - 1, i < n)
    return "\n".join(L) + "\n"


def _kernel(kind, *key):
    k = _KERNELS.get((kind,) + key)
    if k is not None:
        return k
    import hashlib
    import importlib.util
    import os
    import sys
    if kind == "segsum":
        src = _segsum_source(*key)
    else:
        src = _HEAD + {"scan": _scan_source, "last": _last_source, "lane": _lane_source}[kind](*key)
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    d = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    d = os.path.join(d, "torchvic_xscan")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "xscan_%s_%s.py" % (kind, h))
    if not os.path.exists(path):
        tmp = "%s.%d" % (path, os.getpid())
        with open(tmp, "w") as fh:
            fh.write(src)
        os.replace(tmp, path)
    name = "torchvic_xscan_%s_%s" % (kind, h)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    k = _KERNELS[(kind,) + key] = mod.kern
    return k


def _rows(x):
    R = int(x.shape[-1])
    return R, x.numel() // R


@torch.library.custom_op("torchvic::xscan_cumsum", mutates_args=())
def _cumsum_op(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    R, N = _rows(x)
    y = torch.empty_like(x)
    CH = _chunk(N, R)
    if R <= CH:                         # one chunk
        _kernel("lane", R, False)[((N + LANES - 1) // LANES,)](x, y, N, BLOCK=LANES, num_warps=LANES // 32,
                                                                 enable_fp_fusion=False)
        return y
    nch = (R + CH - 1) // CH
    grid = ((N + _rows_per_program(CH) - 1) // _rows_per_program(CH),)
    _kernel("scan", R, CH, nch, False)[grid](x, y, N, num_warps=4, enable_fp_fusion=False)
    return y


@_cumsum_op.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("torchvic::xscan_cumsum_last", mutates_args=())
def _last_op(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    R, N = _rows(x)
    y = torch.empty(x.shape[:-1], dtype=x.dtype, device=x.device)
    CH = _chunk(N, R)
    if R <= CH:
        _kernel("lane", R, True)[((N + LANES - 1) // LANES,)](x, y, N, BLOCK=LANES, num_warps=LANES // 32,
                                                                enable_fp_fusion=False)
    else:
        nch = (R + CH - 1) // CH
        grid = ((N + _rows_per_program(CH) - 1) // _rows_per_program(CH),)
        _kernel("scan", R, CH, nch, True)[grid](x, y, N, num_warps=4, enable_fp_fusion=False)
    return y


@_last_op.register_fake
def _(x):
    return x.new_empty(x.shape[:-1])


def cumsum(x):
    """torch.cumsum(x, -1), the same bits on every device, fast on CUDA."""
    if _use_torch(x):
        return torch.cumsum(x, -1)
    if x.shape[-1] <= SMALL:
        return torch.stack(_emulate(x), -1)
    return _cumsum_op(x)


def cumsum_last(x):
    """torch.cumsum(x, -1)[..., -1], the same bits on every device, fast on CUDA."""
    if _use_torch(x):
        return torch.cumsum(x, -1)[..., -1]
    if x.shape[-1] <= SMALL:
        return _emulate(x)[-1]
    return _last_op(x)


# ----------------------------------------------------------------------------- fixed-order segment sums

def _segsum_source(N, L):
    """One row per lane: out[l] = ((0 + v_j1) + v_j2) + ... over the slots j with idx_j == l, in
    slot order (the CPU's scatter_add order). Adding +0.0 for the other slots changes nothing:
    the running sums start at +0.0 and can never become -0.0."""
    src = ["import triton", "import triton.language as tl", "", "", "@triton.jit",
           "def kern(v_ptr, i_ptr, o_ptr, N, BLOCK: tl.constexpr):",
           "    r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
           "    ok = r < N"]
    src += ["    a%d = tl.zeros([BLOCK], dtype=tl.float64)" % l for l in range(L)]
    src += ["    for j in range(0, %d):" % N,
            "        v = tl.load(v_ptr + r * %d + j, mask=ok, other=0.0)" % N,
            "        i = tl.load(i_ptr + r * %d + j, mask=ok, other=0)" % N]
    src += ["        a%d = a%d + tl.where(i == %d, v, 0.0)" % (l, l, l) for l in range(L)]
    src += ["    tl.store(o_ptr + r * %d + %d, a%d, mask=ok)" % (L, l, l) for l in range(L)]
    return "\n".join(src) + "\n"


@torch.library.custom_op("torchvic::xscan_segment_sum", mutates_args=())
def _segsum_op(v: torch.Tensor, idx: torch.Tensor, L: int) -> torch.Tensor:
    N = int(v.shape[-1])
    rows = v.numel() // max(N, 1)
    out = torch.empty(v.shape[:-1] + (L,), dtype=v.dtype, device=v.device)
    if rows:
        k = _kernel("segsum", N, L)
        k[((rows + 127) // 128,)](v.contiguous(), idx.contiguous(), out, rows, BLOCK=128, num_warps=4,
                                  enable_fp_fusion=False)
    return out


@_segsum_op.register_fake
def _(v, idx, L):
    return v.new_empty(v.shape[:-1] + (L,))


def segment_sum(v, idx, L):
    """zeros(..., L).scatter_add(-1, idx, v) with the additions in slot order on every device
    (torch's CUDA scatter_add adds in an order that changes from run to run); idx in [0, L)."""
    if _use_torch(v) or v.shape[-1] == 0:
        return torch.zeros(v.shape[:-1] + (L,), dtype=v.dtype, device=v.device).scatter_add(-1, idx, v)
    return _segsum_op(v, idx, int(L))
