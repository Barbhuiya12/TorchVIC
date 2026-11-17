"""The residue pool lists of the nitrogen day as per-tile Triton kernels (CUDA): the
organic-matter start of day (organic_matter.start_day) and the tillage residue response
(organic_matter.tillage), bit for bit the torch functions they replace.

Why: the torch forms loop over the S (96) surface and B (192) subsurface slots of every tile with
whole-array operations, every day and for every tile whether a slot is in use or a tile is tilled
(1.4 s and 2 x 0.65 s a day at 338,475 tiles on an H200). Here one program handles one tile: it
reads the tile's lists once, walks only its slots in use, and a tile without the operation does
no work.

- start_day(r) -> Residues: new tensors (the state is never changed in place). The stable
  compaction is computed as each slot's destination, so the lists, the free slots included, come
  out exactly as compact() orders them; detrition walks the active surface slots in order and
  matches each against the subsurface list by destination order.
- tillage_(r, ...) -> r: changes r's tensors IN PLACE. Only for lists the caller owns (the day's
  lists after start_day); organic_matter.tillage stays the functional form.

Exactness: only + - * / and comparisons run in the kernels (float64 IEEE; contraction off,
enable_fp_fusion=False); the exp of the detrition and the layer coefficients and fractions of the
tillage (pow, exp) are computed by torch outside (organic_matter.tillage_layers). A value is taken
out of a slot vector through its bits (an integer sum of one element), so signed zeros and NaN
payloads pass unchanged. The order of every sequential step (surface slots, the 8 sifting chains
and their layers, the burial slots and layers, the appends) is the torch functions'.

Both are torch custom ops (torchvic::cs_om_start_day, torchvic::cs_om_tillage_), so a
torch.compile'd day calls them as single operations and a CUDA graph captures them (no host
syncs). TORCHVIC_CROPSYST_KERNELS=0 runs the torch forms instead.
"""
import os
from typing import List

import torch

from . import organic_matter as om

try:
    import triton
    import triton.language as tl
except ImportError:          # pragma: no cover - CPU-only installs
    triton = None

KERNELS = os.environ.get("TORCHVIC_CROPSYST_KERNELS", "1") != "0"


def enabled(device) -> bool:
    return KERNELS and triton is not None and torch.device(device).type == "cuda"


def _p2(n):
    return 1 << max(int(n) - 1, 0).bit_length()


if triton is not None:
    _BIG = tl.constexpr(1 << 29)

    @triton.jit
    def _abs_le(d, tol):
        d = tl.where(d < 0.0, -d, d)
        return d <= tol

    @triton.jit
    def _add_mass(m, x):
        """Biomass::add_mass (organic_matter.add_mass)."""
        nz = m != 0.0
        return tl.where(nz, m * ((m + x) / tl.where(nz, m, 1.0)), m)

    @triton.jit
    def _reset_mass(m, new):
        """organic_matter.reset_mass."""
        e = _abs_le(0.0 - m, 0.000001)
        safe = tl.where(e, 1.0, m)
        return tl.where(e, _add_mass(m, new), m * (1.0 + ((new - m) / safe)))

    @triton.jit
    def _add_different(mass, cfrac, cn, k, a_mass, a_cfrac, a_cn, a_k):
        """organic_matter.add_different_mass (above ground False: the area per mass is kept)
        with merge_intensive for k. Returns (mass, cfrac, cn, k)."""
        go = a_mass > 0.0000001
        cur_c = mass * cfrac
        add_c = a_mass * a_cfrac
        cur_n = cur_c / cn
        add_n = add_c / a_cn
        m2 = mass + a_mass
        c_tot = cur_c + add_c
        n_tot = cur_n + add_n
        cn2 = c_tot / n_tot
        cf2 = tl.where(_abs_le(0.0 - m2, 0.000001), 0.0, c_tot / m2)
        # merge_intensive(k, a_k, a_mass, m2)
        s = a_mass + m2
        sz = _abs_le(0.0 - s, 0.000001)
        safe = tl.where(sz, 1.0, s)
        w = a_k * tl.where(sz, 0.0, a_mass / safe) + k * tl.where(sz, 0.0, m2 / safe)
        w = tl.where(_abs_le(0.0 - m2, 0.000001), a_k, w)
        w = tl.where(_abs_le(0.0 - a_mass, 0.000001), k, w)
        k2 = tl.where(a_k != k, w, k)
        return (tl.where(go, m2, mass), tl.where(go, cf2, cfrac), tl.where(go, cn2, cn),
                tl.where(go, k2, k))

    @triton.jit
    def _pf(v, sel):
        """The float64 of the one selected lane, through its bits."""
        b = tl.where(sel, v.to(tl.int64, bitcast=True), 0)
        return tl.sum(b, axis=0).to(tl.float64, bitcast=True)

    @triton.jit
    def _pi(v, sel):
        return tl.sum(tl.where(sel, v, 0), axis=0)

    @triton.jit
    def _compact_copy(src, dst, base, o, d, m):
        tl.store(dst + base + d, tl.load(src + base + o, mask=m), mask=m)

    @triton.jit
    def _start_day_kernel(
            sa, st, sp, sl, sc, se, sm, scf, scn, sk, skd, sap, keep,
            ba, bt, bp, bl, bc, be, bm, bcf, bcn, bk, bkd, bap, over_in,
            sa_o, st_o, sp_o, sl_o, sc_o, se_o, sm_o, scf_o, scn_o, sk_o, skd_o, sap_o, sn_o,
            ba_o, bt_o, bp_o, bl_o, bc_o, be_o, bm_o, bcf_o, bcn_o, bk_o, bkd_o, bap_o, bn_o, over_o,
            dm_o, det_o,
            S: tl.constexpr, B: tl.constexpr, SP: tl.constexpr, BP: tl.constexpr):
        """Part 1 of start_day: both lists compacted (stable; each slot to its destination), the
        surface pools' detrition loss (mass * keep, the detritus to dm_o), the negligible surface
        pools deleted; a tile without a detritus (det_o False) has its negligible subsurface pools
        deleted here too, the others in part 2."""
        pid = tl.program_id(0).to(tl.int64)
        i_s = tl.arange(0, SP)
        ms = i_s < S
        o_s = pid * S + i_s
        act = (tl.load(sa + o_s, mask=ms, other=0) != 0) & ms
        a32 = act.to(tl.int32)
        na = tl.sum(a32, axis=0)
        ina = (ms & ~act).to(tl.int32)
        dest = tl.where(act, tl.cumsum(a32, 0) - 1, na + tl.cumsum(ina, 0) - 1)
        d_s = pid * S + dest
        pos = tl.load(sp + o_s, mask=ms, other=0)
        kd = tl.load(skd + o_s, mask=ms, other=0.0)
        m = tl.load(sm + o_s, mask=ms, other=0.0)
        kp = tl.load(keep + o_s, mask=ms, other=1.0)
        on = act & (pos != 0) & (kd > 0.000000001)
        m_new = tl.where(on, m * kp, m)
        tl.store(sa_o + d_s, act & ~(m_new < 0.0000001), mask=ms)
        tl.store(sm_o + d_s, m_new, mask=ms)
        tl.store(dm_o + d_s, m * (1.0 - kp), mask=on)
        tl.store(sp_o + d_s, pos, mask=ms)
        tl.store(skd_o + d_s, kd, mask=ms)
        _compact_copy(st, st_o, pid * S, i_s, dest, ms)
        _compact_copy(sl, sl_o, pid * S, i_s, dest, ms)
        _compact_copy(sc, sc_o, pid * S, i_s, dest, ms)
        _compact_copy(se, se_o, pid * S, i_s, dest, ms)
        _compact_copy(scf, scf_o, pid * S, i_s, dest, ms)
        _compact_copy(scn, scn_o, pid * S, i_s, dest, ms)
        _compact_copy(sk, sk_o, pid * S, i_s, dest, ms)
        _compact_copy(sap, sap_o, pid * S, i_s, dest, ms)
        tl.store(sn_o + pid, na.to(tl.int64))
        det = tl.sum(on.to(tl.int32), axis=0) > 0
        tl.store(det_o + pid, det)
        i_b = tl.arange(0, BP)
        mb = i_b < B
        o_b = pid * B + i_b
        b_act = (tl.load(ba + o_b, mask=mb, other=0) != 0) & mb
        b32 = b_act.to(tl.int32)
        nb = tl.sum(b32, axis=0)
        inb = (mb & ~b_act).to(tl.int32)
        b_dest = tl.where(b_act, tl.cumsum(b32, 0) - 1, nb + tl.cumsum(inb, 0) - 1)
        b_m = tl.load(bm + o_b, mask=mb, other=0.0)
        tl.store(ba_o + pid * B + b_dest, tl.where(det, b_act, b_act & ~(b_m < 0.0000001)), mask=mb)
        tl.store(bm_o + pid * B + b_dest, b_m, mask=mb)
        _compact_copy(bt, bt_o, pid * B, i_b, b_dest, mb)
        _compact_copy(bp, bp_o, pid * B, i_b, b_dest, mb)
        _compact_copy(bl, bl_o, pid * B, i_b, b_dest, mb)
        _compact_copy(bc, bc_o, pid * B, i_b, b_dest, mb)
        _compact_copy(be, be_o, pid * B, i_b, b_dest, mb)
        _compact_copy(bcf, bcf_o, pid * B, i_b, b_dest, mb)
        _compact_copy(bcn, bcn_o, pid * B, i_b, b_dest, mb)
        _compact_copy(bk, bk_o, pid * B, i_b, b_dest, mb)
        _compact_copy(bkd, bkd_o, pid * B, i_b, b_dest, mb)
        _compact_copy(bap, bap_o, pid * B, i_b, b_dest, mb)
        tl.store(bn_o + pid, nb.to(tl.int64))
        tl.store(over_o + pid, tl.load(over_in + pid) != 0)

    @triton.jit
    def _detritus_kernel(sa, st, sp, sc, scf, scn, sk, skd, sap, sn, dm,
                         ba, bt, bp, bl, bc, be, bm, bcf, bcn, bk, bkd, bap, bn, over, det,
                         S: tl.constexpr, B: tl.constexpr, BP: tl.constexpr):
        """Part 2 of start_day, in place on the compacted lists of a tile with detritus: each
        surface pool's detritus, in list order, to the first comparable subsurface pool of layer
        2 or a new one; then the negligible subsurface pools are deleted."""
        pid = tl.program_id(0).to(tl.int64)
        if tl.load(det + pid) != 0:
            i_b = tl.arange(0, BP)
            mb = i_b < B
            o_b = pid * B + i_b
            b_act = (tl.load(ba + o_b, mask=mb, other=0) != 0) & mb
            b_t = tl.load(bt + o_b, mask=mb, other=0)
            b_p = tl.load(bp + o_b, mask=mb, other=0)
            b_l = tl.load(bl + o_b, mask=mb, other=0)
            b_c = tl.load(bc + o_b, mask=mb, other=0)
            b_e = tl.load(be + o_b, mask=mb, other=0)
            b_m = tl.load(bm + o_b, mask=mb, other=0.0)
            b_cf = tl.load(bcf + o_b, mask=mb, other=0.0)
            b_cn = tl.load(bcn + o_b, mask=mb, other=0.0)
            b_k = tl.load(bk + o_b, mask=mb, other=0.0)
            b_kd = tl.load(bkd + o_b, mask=mb, other=0.0)
            b_ap = tl.load(bap + o_b, mask=mb, other=0.0)
            ov = tl.load(over + pid) != 0
            nbc = tl.load(bn + pid)
            tl.debug_barrier()
            ns = tl.load(sn + pid)
            for i in range(0, ns):
                o = pid * S + i
                on_i = (tl.load(sp + o) != 0) & (tl.load(skd + o) > 0.000000001)
                dmi = tl.load(dm + o, mask=on_i, other=0.0)
                t_i = tl.load(st + o)
                c_i = tl.load(sc + o)
                cf_i = tl.load(scf + o)
                cn_i = tl.load(scn + o)
                k_i = tl.load(sk + o)
                ap_i = tl.load(sap + o)
                cmp = b_act & (b_t == t_i) & (b_p == 0) & (b_l == 2) & (b_c == c_i) & \
                    _abs_le(cf_i - b_cf, 0.000001) & _abs_le(cn_i - b_cn, 0.000001) & \
                    _abs_le(k_i - b_k, 0.000001) & _abs_le(0.0 - b_kd, 0.000001)
                jm = tl.min(tl.where(cmp, i_b, _BIG), axis=0)
                found = jm < _BIG
                sel = (i_b == jm) & (on_i & found)
                b_m = tl.where(sel, _add_mass(b_m, dmi), b_m)
                app = on_i & ~found
                at = mb & (i_b == nbc) & app
                b_act = b_act | at
                b_t = tl.where(at, t_i, b_t)
                b_p = tl.where(at, 0, b_p)
                b_l = tl.where(at, 2, b_l)
                b_c = tl.where(at, c_i, b_c)
                b_e = tl.where(at, 2, b_e)
                b_m = tl.where(at, dmi, b_m)
                b_cf = tl.where(at, cf_i, b_cf)
                b_cn = tl.where(at, cn_i, b_cn)
                b_k = tl.where(at, k_i, b_k)
                b_kd = tl.where(at, 0.0, b_kd)
                b_ap = tl.where(at, ap_i, b_ap)
                ov = ov | (app & (nbc >= B))
                nbc = nbc + app.to(tl.int64)
            b_act = b_act & ~(b_m < 0.0000001)
            tl.debug_barrier()
            tl.store(ba + o_b, b_act, mask=mb)
            tl.store(bt + o_b, b_t, mask=mb)
            tl.store(bp + o_b, b_p, mask=mb)
            tl.store(bl + o_b, b_l, mask=mb)
            tl.store(bc + o_b, b_c, mask=mb)
            tl.store(be + o_b, b_e, mask=mb)
            tl.store(bm + o_b, b_m, mask=mb)
            tl.store(bcf + o_b, b_cf, mask=mb)
            tl.store(bcn + o_b, b_cn, mask=mb)
            tl.store(bk + o_b, b_k, mask=mb)
            tl.store(bkd + o_b, b_kd, mask=mb)
            tl.store(bap + o_b, b_ap, mask=mb)
            tl.store(bn + pid, nbc)
            tl.store(over + pid, ov)

    @triton.jit
    def _tillage_kernel(
            sa, st, sp, sl, sc, se, sm, scf, scn, sk, skd, sap, sn,
            ba, bt, bp, bl, bc, be, bm, bcf, bcn, bk, bkd, bap, bn,
            over, mask, flat_r, bur_r, mix_, last_, coef_, frac_,
            S: tl.constexpr, B: tl.constexpr, SP: tl.constexpr, BP: tl.constexpr, L: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        if tl.load(mask + pid) != 0:
            # ---- 1. standing stubble partly flattened, the flat clones appended in list order
            i_s = tl.arange(0, SP)
            ms = i_s < S
            o_s = pid * S + i_s
            act = (tl.load(sa + o_s, mask=ms, other=0) != 0) & ms
            pos = tl.load(sp + o_s, mask=ms, other=0)
            m = tl.load(sm + o_s, mask=ms, other=0.0)
            v_t = tl.load(st + o_s, mask=ms, other=0)
            v_c = tl.load(sc + o_s, mask=ms, other=0)
            v_e = tl.load(se + o_s, mask=ms, other=0)
            v_cf = tl.load(scf + o_s, mask=ms, other=0.0)
            v_cn = tl.load(scn + o_s, mask=ms, other=0.0)
            v_k = tl.load(sk + o_s, mask=ms, other=0.0)
            v_kd = tl.load(skd + o_s, mask=ms, other=0.0)
            v_ap = tl.load(sap + o_s, mask=ms, other=0.0)
            ns = tl.load(sn + pid)
            fr = tl.load(flat_r + pid)
            tl.debug_barrier()
            on_s = act & (pos == 2)
            tl.store(sm + o_s, m * (1.0 - fr), mask=on_s)
            on32 = on_s.to(tl.int32)
            rank = tl.cumsum(on32, 0).to(tl.int64) - 1 + ns
            cnt = tl.sum(on32, axis=0)
            ok = on_s & (rank < S)
            ov = tl.sum((on_s & (rank >= S)).to(tl.int32), axis=0) > 0
            r_s = pid * S + rank
            tl.store(sa + r_s, on_s, mask=ok)
            tl.store(st + r_s, v_t, mask=ok)
            tl.store(sp + r_s, tl.full([SP], 1, tl.int64), mask=ok)
            tl.store(sl + r_s, tl.full([SP], 0, tl.int64), mask=ok)
            tl.store(sc + r_s, v_c, mask=ok)
            tl.store(se + r_s, v_e, mask=ok)
            tl.store(sm + r_s, m * fr, mask=ok)
            tl.store(scf + r_s, v_cf, mask=ok)
            tl.store(scn + r_s, v_cn, mask=ok)
            tl.store(sk + r_s, v_k, mask=ok)
            tl.store(skd + r_s, v_kd, mask=ok)
            tl.store(sap + r_s, v_ap * 2.0, mask=ok)
            ns2 = ns + cnt
            tl.store(sn + pid, ns2)
            tl.debug_barrier()
            if tl.load(mix_ + pid) != 0:
                last = tl.load(last_ + pid)
                i_b = tl.arange(0, BP)
                mb = i_b < B
                o_b = pid * B + i_b
                b_act = (tl.load(ba + o_b, mask=mb, other=0) != 0) & mb
                b_t = tl.load(bt + o_b, mask=mb, other=0)
                b_p = tl.load(bp + o_b, mask=mb, other=0)
                b_l = tl.load(bl + o_b, mask=mb, other=0)
                b_c = tl.load(bc + o_b, mask=mb, other=0)
                b_e = tl.load(be + o_b, mask=mb, other=0)
                b_m = tl.load(bm + o_b, mask=mb, other=0.0)
                b_cf = tl.load(bcf + o_b, mask=mb, other=0.0)
                b_cn = tl.load(bcn + o_b, mask=mb, other=0.0)
                b_k = tl.load(bk + o_b, mask=mb, other=0.0)
                b_kd = tl.load(bkd + o_b, mask=mb, other=0.0)
                b_ap = tl.load(bap + o_b, mask=mb, other=0.0)
                nbc = tl.load(bn + pid)
                # ---- 2. sifting: per (type, cycling) a chain down the tilled layers; a clone
                # that finds no pool in the next layer is appended (in chain order; it can never
                # match a later search of the sifting, so appending at once is the same)
                for t in tl.static_range(2):
                    for c in tl.static_range(4):
                        for l in tl.static_range(L):
                            inl = (l + 1) <= last
                            cond = b_act & (b_t == t) & (b_p == 0) & (b_c == c) & (b_l == l + 1)
                            j = tl.min(tl.where(cond, i_b, _BIG), axis=0)
                            found = j < _BIG
                            j = tl.where(found, j, 0)
                            sel = i_b == j
                            hit = inl & found
                            mass = _pf(b_m, sel)
                            cfrac = _pf(b_cf, sel)
                            cn = _pf(b_cn, sel)
                            k = _pf(b_k, sel)
                            apm = _pf(b_ap, sel)
                            out = mass * tl.load(coef_ + pid * L + l)
                            if l > 0:
                                mrg = hit & r_on
                                m2, cf2, cn2, k2 = _add_different(mass, cfrac, cn, k, r_m, r_cf, r_cn, r_k)
                                mass = tl.where(mrg, m2, mass)
                                cfrac = tl.where(mrg, cf2, cfrac)
                                cn = tl.where(mrg, cn2, cn)
                                k = tl.where(mrg, k2, k)
                                # no pool here: the incoming clone joins the list unchanged
                                mv = inl & ~found & r_on
                                at = mb & (i_b == nbc) & mv
                                b_act = b_act | at
                                b_t = tl.where(at, r_t, b_t)
                                b_p = tl.where(at, 0, b_p)
                                b_l = tl.where(at, r_l, b_l)
                                b_c = tl.where(at, r_c, b_c)
                                b_e = tl.where(at, r_e, b_e)
                                b_m = tl.where(at, r_m, b_m)
                                b_cf = tl.where(at, r_cf, b_cf)
                                b_cn = tl.where(at, r_cn, b_cn)
                                b_k = tl.where(at, r_k, b_k)
                                b_kd = tl.where(at, r_kd, b_kd)
                                b_ap = tl.where(at, r_ap, b_ap)
                                ov = ov | (mv & (nbc >= B))
                                nbc = nbc + mv.to(tl.int64)
                            mass = tl.where(hit & (out > 0.0), _add_mass(mass, -out), mass)
                            upd = sel & hit
                            b_m = tl.where(upd, mass, b_m)
                            b_cf = tl.where(upd, cfrac, b_cf)
                            b_cn = tl.where(upd, cn, b_cn)
                            b_k = tl.where(upd, k, b_k)
                            new_on = hit & ~_abs_le(0.0 - out, 0.000001)
                            c_m = _reset_mass(mass, out)
                            c_t = _pi(b_t, sel)
                            c_c = _pi(b_c, sel)
                            c_l = _pi(b_l, sel)
                            c_e = _pi(b_e, sel)
                            c_kd = _pf(b_kd, sel)
                            if l == 0:
                                r_on = new_on
                                r_m, r_cf, r_cn, r_k, r_ap = c_m, cfrac, cn, k, apm
                                r_t, r_c, r_l, r_e, r_kd = c_t, c_c, c_l, c_e, c_kd
                            else:
                                keep_old = ~(inl & (found | r_on))
                                r_m = tl.where(new_on, c_m, r_m)
                                r_cf = tl.where(new_on, cfrac, r_cf)
                                r_cn = tl.where(new_on, cn, r_cn)
                                r_k = tl.where(new_on, k, r_k)
                                r_ap = tl.where(new_on, apm, r_ap)
                                r_t = tl.where(new_on, c_t, r_t)
                                r_c = tl.where(new_on, c_c, r_c)
                                r_l = tl.where(new_on, c_l, r_l)
                                r_e = tl.where(new_on, c_e, r_e)
                                r_kd = tl.where(new_on, c_kd, r_kd)
                                r_on = tl.where(keep_old, r_on, new_on)
                # ---- 3. burial of flat residue, surface slots in order; the target of each
                # (type, cycling, layer) is the first such pool before the burial's own clones
                nb3 = nbc
                br = tl.load(bur_r + pid)
                hi = tl.minimum(ns2, S)
                for jj in range(0, hi):
                    o = pid * S + jj
                    on = (tl.load(sa + o) != 0) & (tl.load(sp + o) == 1)
                    m0 = tl.load(sm + o)
                    t_j = tl.load(st + o)
                    c_j = tl.load(sc + o)
                    e_j = tl.load(se + o)
                    cf_j = tl.load(scf + o)
                    cn_j = tl.load(scn + o)
                    k_j = tl.load(sk + o)
                    ap_j = tl.load(sap + o)
                    tc_ok = (t_j >= 0) & (t_j <= 1) & (c_j >= 0) & (c_j <= 3)
                    for l in tl.static_range(L):
                        f_l = tl.load(frac_ + pid * L + l)
                        go = on & ((l + 1) <= last) & (f_l > 0.0)
                        cm = _reset_mass(m0, m0 * br * f_l)
                        cond = b_act & (i_b < nb3) & (b_p == 0) & (b_t == t_j) & (b_c == c_j) & (b_l == l + 1)
                        j = tl.min(tl.where(cond, i_b, _BIG), axis=0)
                        found = (j < _BIG) & tc_ok
                        j = tl.where(found, j, 0)
                        sel = i_b == j
                        hit = go & found
                        m2, cf2, cn2, k2 = _add_different(_pf(b_m, sel), _pf(b_cf, sel), _pf(b_cn, sel),
                                                          _pf(b_k, sel), cm, cf_j, cn_j, k_j)
                        upd = sel & hit
                        b_m = tl.where(upd, m2, b_m)
                        b_cf = tl.where(upd, cf2, b_cf)
                        b_cn = tl.where(upd, cn2, b_cn)
                        b_k = tl.where(upd, k2, b_k)
                        app = go & ~found
                        at = mb & (i_b == nbc) & app
                        b_act = b_act | at
                        b_t = tl.where(at, t_j, b_t)
                        b_p = tl.where(at, 0, b_p)
                        b_l = tl.where(at, l + 1, b_l)
                        b_c = tl.where(at, c_j, b_c)
                        b_e = tl.where(at, e_j, b_e)
                        b_m = tl.where(at, cm, b_m)
                        b_cf = tl.where(at, cf_j, b_cf)
                        b_cn = tl.where(at, cn_j, b_cn)
                        b_k = tl.where(at, k_j, b_k)
                        b_kd = tl.where(at, 0.0, b_kd)
                        b_ap = tl.where(at, ap_j, b_ap)
                        ov = ov | (app & (nbc >= B))
                        nbc = nbc + app.to(tl.int64)
                    tl.store(sm + o, m0 * (1.0 - br), mask=on)
                tl.debug_barrier()
                tl.store(ba + o_b, b_act, mask=mb)
                tl.store(bt + o_b, b_t, mask=mb)
                tl.store(bp + o_b, b_p, mask=mb)
                tl.store(bl + o_b, b_l, mask=mb)
                tl.store(bc + o_b, b_c, mask=mb)
                tl.store(be + o_b, b_e, mask=mb)
                tl.store(bm + o_b, b_m, mask=mb)
                tl.store(bcf + o_b, b_cf, mask=mb)
                tl.store(bcn + o_b, b_cn, mask=mb)
                tl.store(bk + o_b, b_k, mask=mb)
                tl.store(bkd + o_b, b_kd, mask=mb)
                tl.store(bap + o_b, b_ap, mask=mb)
                tl.store(bn + pid, nbc)
            tl.store(over + pid, (tl.load(over + pid) != 0) | ov)

    @triton.jit
    def _harvest_kernel(
            sa, st, sp, sl, sc, se, sm, scf, scn, sk, skd, sap, sn,
            ba, bt, bp, bl, bc, be, bm, bcf, bcn, bk, bkd, bap, bn,
            over, go_, s_on, s_comp, s_cn, s_apm, kc_, kdet_, cfrac_, r_on, r_comp, r_cn,
            S: tl.constexpr, B: tl.constexpr, BP: tl.constexpr, L: tl.constexpr):
        """The harvest's residue: add_surface (flat, then stubble; one pool per cycling, never
        merged) and add_subsurface (dead roots of each layer by cycling, merged into the first
        comparable pool or appended in layer order)."""
        pid = tl.program_id(0).to(tl.int64)
        if tl.load(go_ + pid) != 0:
            ov = tl.load(over + pid) != 0
            cf = tl.load(cfrac_ + pid)
            kd = tl.load(kdet_ + pid)
            ns = tl.load(sn + pid)
            for q in tl.static_range(6):               # flat c = 1..3, then stubble c = 1..3
                on = tl.load(s_on + pid * 6 + q) != 0
                o = pid * S + ns
                ok = on & (ns < S)
                tl.store(sa + o, on, mask=ok)
                tl.store(st + o, 0, mask=ok)
                tl.store(sp + o, 1 + q // 3, mask=ok)
                tl.store(sl + o, 0, mask=ok)
                tl.store(sc + o, 1 + q % 3, mask=ok)
                tl.store(se + o, 1, mask=ok)
                tl.store(sm + o, tl.load(s_comp + pid * 6 + q), mask=ok)
                tl.store(scf + o, cf, mask=ok)
                tl.store(scn + o, tl.load(s_cn + pid * 6 + q), mask=ok)
                tl.store(sk + o, tl.load(kc_ + pid * 3 + q % 3), mask=ok)
                tl.store(skd + o, kd, mask=ok)
                tl.store(sap + o, tl.load(s_apm + pid * 2 + q // 3), mask=ok)
                ov = ov | (on & (ns >= S))
                ns = ns + on.to(tl.int64)
            tl.store(sn + pid, ns)
            i_b = tl.arange(0, BP)
            mb = i_b < B
            o_b = pid * B + i_b
            b_act = (tl.load(ba + o_b, mask=mb, other=0) != 0) & mb
            b_t = tl.load(bt + o_b, mask=mb, other=0)
            b_p = tl.load(bp + o_b, mask=mb, other=0)
            b_l = tl.load(bl + o_b, mask=mb, other=0)
            b_c = tl.load(bc + o_b, mask=mb, other=0)
            b_e = tl.load(be + o_b, mask=mb, other=0)
            b_m = tl.load(bm + o_b, mask=mb, other=0.0)
            b_cf = tl.load(bcf + o_b, mask=mb, other=0.0)
            b_cn = tl.load(bcn + o_b, mask=mb, other=0.0)
            b_k = tl.load(bk + o_b, mask=mb, other=0.0)
            b_kd = tl.load(bkd + o_b, mask=mb, other=0.0)
            b_ap = tl.load(bap + o_b, mask=mb, other=0.0)
            nbc = tl.load(bn + pid)
            for c in tl.static_range(3):
                k = tl.load(kc_ + pid * 3 + c)
                for l in tl.static_range(L):
                    on = tl.load(r_on + pid * L + l) != 0
                    comp = tl.load(r_comp + (pid * 3 + c) * L + l)
                    cn = tl.load(r_cn + (pid * 3 + c) * L + l)
                    # the pools of one cycling go to distinct layers: appending at once is the same
                    cond = b_act & (b_t == 0) & (b_p == 0) & (b_c == c + 1) & _abs_le(cf - b_cf, 0.000001) & \
                        _abs_le(k - b_k, 0.000001) & _abs_le(0.0 - b_kd, 0.000001) & (b_l == l + 1) & \
                        _abs_le(cn - b_cn, 0.000001)
                    j = tl.min(tl.where(cond, i_b, _BIG), axis=0)
                    found = j < _BIG
                    hit = (i_b == j) & on & found
                    b_m = tl.where(hit, _add_mass(b_m, comp), b_m)
                    app = on & ~found
                    at = mb & (i_b == nbc) & app
                    b_act = b_act | at
                    b_t = tl.where(at, 0, b_t)
                    b_p = tl.where(at, 0, b_p)
                    b_l = tl.where(at, l + 1, b_l)
                    b_c = tl.where(at, c + 1, b_c)
                    b_e = tl.where(at, l + 1, b_e)
                    b_m = tl.where(at, comp, b_m)
                    b_cf = tl.where(at, cf, b_cf)
                    b_cn = tl.where(at, cn, b_cn)
                    b_k = tl.where(at, k, b_k)
                    b_kd = tl.where(at, 0.0, b_kd)
                    b_ap = tl.where(at, 0.0, b_ap)
                    ov = ov | (app & (nbc >= B))
                    nbc = nbc + app.to(tl.int64)
            tl.debug_barrier()
            tl.store(ba + o_b, b_act, mask=mb)
            tl.store(bt + o_b, b_t, mask=mb)
            tl.store(bp + o_b, b_p, mask=mb)
            tl.store(bl + o_b, b_l, mask=mb)
            tl.store(bc + o_b, b_c, mask=mb)
            tl.store(be + o_b, b_e, mask=mb)
            tl.store(bm + o_b, b_m, mask=mb)
            tl.store(bcf + o_b, b_cf, mask=mb)
            tl.store(bcn + o_b, b_cn, mask=mb)
            tl.store(bk + o_b, b_k, mask=mb)
            tl.store(bkd + o_b, b_kd, mask=mb)
            tl.store(bap + o_b, b_ap, mask=mb)
            tl.store(bn + pid, nbc)
            tl.store(over + pid, ov)


_FIELDS = 12          # Pools fields before n


def _warps(n):
    return 4 if n > 128 else 2


@torch.library.custom_op("torchvic::cs_om_start_day", mutates_args=())
def _start_day_op(sur: List[torch.Tensor], sub: List[torch.Tensor], keep: torch.Tensor,
                  over: torch.Tensor) -> List[torch.Tensor]:
    T, S = sur[0].shape
    B = sub[0].shape[1]
    so = [torch.empty_like(x) for x in sur[:_FIELDS]] + [torch.empty((T,), dtype=torch.long, device=keep.device)]
    bo = [torch.empty_like(x) for x in sub[:_FIELDS]] + [torch.empty((T,), dtype=torch.long, device=keep.device)]
    oo = torch.empty_like(over)
    if T:
        dm = torch.empty_like(keep)
        det = torch.empty_like(over)
        _start_day_kernel[(T,)](*sur[:_FIELDS], keep, *sub[:_FIELDS], over, *so, *bo, oo, dm, det,
                                S=S, B=B, SP=_p2(S), BP=_p2(B), num_warps=_warps(B), enable_fp_fusion=False)
        # surface: active type position cycling cfrac cn k kdet apm n
        _detritus_kernel[(T,)](so[0], so[1], so[2], so[4], so[7], so[8], so[9], so[10], so[11], so[12], dm,
                               bo[0], bo[1], bo[2], bo[3], bo[4], bo[5], bo[6], bo[7], bo[8], bo[9], bo[10], bo[11],
                               bo[12], oo, det, S=S, B=B, BP=_p2(B), num_warps=_warps(B), enable_fp_fusion=False)
    return so + bo + [oo]


@_start_day_op.register_fake
def _(sur, sub, keep, over):
    T = sur[0].shape[0]
    return ([torch.empty_like(x) for x in sur[:_FIELDS]] + [sur[0].new_empty((T,), dtype=torch.long)]
            + [torch.empty_like(x) for x in sub[:_FIELDS]] + [sub[0].new_empty((T,), dtype=torch.long)]
            + [torch.empty_like(over)])


def _contig(p: om.Pools):
    return [x.contiguous() for x in p]


def start_day(r: om.Residues) -> om.Residues:
    """organic_matter.start_day, one program per tile (new tensors)."""
    sur, sub = _contig(r.sur), _contig(r.sub)
    keep = torch.exp(-sur[10])            # the detrition's exp(-kdet), at every slot as torch computes it
    out = _start_day_op(sur, sub, keep, r.overflow.contiguous())
    n = _FIELDS + 1
    return om.Residues(om.Pools(*out[:n]), om.Pools(*out[n:2 * n]), r.flat_contact, out[2 * n])


@torch.library.custom_op("torchvic::cs_om_tillage_",
                         mutates_args=("sur", "sub", "over"))
def _tillage_op(sur: List[torch.Tensor], sub: List[torch.Tensor], over: torch.Tensor, mask: torch.Tensor,
                flat_r: torch.Tensor, bur_r: torch.Tensor, mix: torch.Tensor, last: torch.Tensor,
                coef: torch.Tensor, frac: torch.Tensor) -> None:
    T, S = sur[0].shape
    B = sub[0].shape[1]
    L = coef.shape[1]
    if T:
        _tillage_kernel[(T,)](*sur, *sub, over, mask, flat_r, bur_r, mix, last, coef, frac,
                              S=S, B=B, SP=_p2(S), BP=_p2(B), L=L, num_warps=_warps(B), enable_fp_fusion=False)


@_tillage_op.register_fake
def _(sur, sub, over, mask, flat_r, bur_r, mix, last, coef, frac):
    return None


def tillage_(r: om.Residues, mask, thickness, depth, effect, flatten_ratio, burial_ratio) -> om.Residues:
    """organic_matter.tillage IN PLACE on r (lists the caller owns); returns r."""
    T = mask.shape[0]
    dt = r.sur.mass.dtype
    effect = torch.broadcast_to(effect, mask.shape)
    depth = torch.broadcast_to(depth, mask.shape)
    mix, last, coef, frac = om.tillage_layers(mask, torch.broadcast_to(thickness, (T,) + thickness.shape[-1:]),
                                              depth, effect)
    bc = lambda v: torch.broadcast_to(torch.as_tensor(v, dtype=dt, device=mask.device), mask.shape).contiguous()
    for x in list(r.sur) + list(r.sub) + [r.overflow]:
        assert x.is_contiguous()
    _tillage_op(list(r.sur), list(r.sub), r.overflow, mask.contiguous(), bc(flatten_ratio), bc(burial_ratio),
                mix.contiguous(), last.contiguous(), coef.contiguous(), frac.contiguous())
    return r


@torch.library.custom_op("torchvic::cs_om_harvest_", mutates_args=("sur", "sub", "over"))
def _harvest_op(sur: List[torch.Tensor], sub: List[torch.Tensor], over: torch.Tensor, go: torch.Tensor,
                s_on: torch.Tensor, s_comp: torch.Tensor, s_cn: torch.Tensor, s_apm: torch.Tensor, kc: torch.Tensor,
                kdet: torch.Tensor, cfrac: torch.Tensor, r_on: torch.Tensor, r_comp: torch.Tensor,
                r_cn: torch.Tensor) -> None:
    T, S = sur[0].shape
    B = sub[0].shape[1]
    L = r_on.shape[1]
    if T:
        _harvest_kernel[(T,)](*sur, *sub, over, go, s_on, s_comp, s_cn, s_apm, kc, kdet, cfrac, r_on, r_comp, r_cn,
                              S=S, B=B, BP=_p2(B), L=L, num_warps=_warps(B), enable_fp_fusion=False)


@_harvest_op.register_fake
def _(sur, sub, over, go, s_on, s_comp, s_cn, s_apm, kc, kdet, cfrac, r_on, r_comp, r_cn):
    return None


def harvest_residue_(r: om.Residues, flat_mask, chaff, stubble_mask, stubble, root_mask, root_biomass, n_conc,
                     root_n_conc, P: om.ResidueParams) -> om.Residues:
    """om.add_surface(FLAT), om.add_surface(STUBBLE), om.add_subsurface in that order, IN PLACE
    on r's lists (lists the caller owns); returns the residues with the new flat contact."""
    C = (om.FAST, om.SLOW, om.LIGNIFIED)
    kdet = om.halflife_to_constant(P.detrition_halflife)
    kc = torch.stack([om.halflife_to_constant(P.halflife[..., c]) for c in C], -1)
    on, comp, cn = [], [], []
    for mask, biomass in ((flat_mask, chaff), (stubble_mask, stubble)):
        for c in C:
            x = biomass * P.top[..., c]
            on.append(mask & (x > 0.0))
            comp.append(x)
            cn.append(om.calc_cn(x, n_conc, P.carbon_fraction))
    fc = torch.where(on[3] | on[4] | on[5], torch.full_like(r.flat_contact, om.STUBBLE_CONTACT_AFTER_CLEAR),
                     r.flat_contact)
    r_on = root_mask[..., None] & (root_biomass > 0.00001)
    r_comp = torch.stack([root_biomass * P.root[..., c][..., None] for c in C], 1)
    r_cn = om.calc_cn(r_comp, root_n_conc[..., None, None], P.carbon_fraction[..., None, None])
    s_on = torch.stack(on, -1)
    go = s_on.any(-1) | r_on.any(-1)
    _harvest_op(list(r.sur), list(r.sub), r.overflow, go, s_on, torch.stack(comp, -1).contiguous(),
                torch.stack(cn, -1).contiguous(), torch.stack([P.flat_apm, P.stubble_apm], -1).contiguous(),
                kc.contiguous(), kdet.contiguous(), P.carbon_fraction.contiguous(), r_on.contiguous(),
                r_comp.contiguous(), r_cn.contiguous())
    return r._replace(flat_contact=fc)


# ----------------------------------------------------------------------------- process_day

#: TORCHVIC_CROPSYST_PROCESS_KERNEL=0 keeps organic_matter.process (its per-layer sums are
#: torch's CUDA scatter_add, whose order of additions changes from run to run)
PROCESS_KERNEL = os.environ.get("TORCHVIC_CROPSYST_PROCESS_KERNEL", "1") != "0"

if triton is not None:
    from triton.language.extra import libdevice as _ld

    @triton.jit
    def _sel_l(v0, v1, v2, v3, v4, v5, v6, v7, i, L: tl.constexpr):
        """Element i (0 <= i < L <= 8) of a per-lane list of L values."""
        out = v0
        if L > 1:
            out = tl.where(i == 1, v1, out)
        if L > 2:
            out = tl.where(i == 2, v2, out)
        if L > 3:
            out = tl.where(i == 3, v3, out)
        if L > 4:
            out = tl.where(i == 4, v4, out)
        if L > 5:
            out = tl.where(i == 5, v5, out)
        if L > 6:
            out = tl.where(i == 6, v6, out)
        if L > 7:
            out = tl.where(i == 7, v7, out)
        return out


def _process_source(L, PHASE):
    """One tile per lane, the pools of each list in order (sums in list order, as the CPU's
    scatter_add). PHASE 1: N mineralization and immobilization demand by layer; PHASE 2: actual
    decomposition with the layer reduction factor, CO2 by layer, the pools' carbon loss and
    exhaust_all (in place)."""
    vs = lambda p: ", ".join(["%s%d" % (p, l) if l < L else "%s0" % p for l in range(8)])
    I = "        "
    src = ["import triton", "import triton.language as tl",
           "from triton.language.extra import libdevice as ld",
           "from torchvic.crop.cropsyst.om_kernels import _sel_l, _add_mass, _abs_le", "", "",
           "@triton.jit",
           "def kern(sa, sc, se, sm, scf, scn, sk, sp, sn, ba, bc, be, bl, bm, bcf, bcn, bk, bp, bn,",
           "         fc_ptr, tf_ptr, mf_ptr, rf_ptr, par_ptr, o1_ptr, o2_ptr, N, S, B, BLOCK: tl.constexpr):",
           "    t = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
           "    ok = t < N",
           "    micro = tl.load(par_ptr + 0)",
           "    meta = tl.load(par_ptr + 1)",
           "    hum = tl.load(par_ptr + 2)",
           "    fc = tl.load(fc_ptr + t, mask=ok, other=0.0)"]
    for l in range(L):
        src.append("    tf%d = tl.load(tf_ptr + t * %d + %d, mask=ok, other=0.0)" % (l, L, l))
        src.append("    mf%d = tl.load(mf_ptr + t * %d + %d, mask=ok, other=0.0)" % (l, L, l))
        if PHASE == 2:
            src.append("    rf%d = tl.load(rf_ptr + t * %d + %d, mask=ok, other=0.0)" % (l, L, l))
    for lst in ("s", "b"):
        for l in range(L):
            src += ["    %s_x%d = tl.zeros([BLOCK], dtype=tl.float64)" % (lst, l)]
            if PHASE == 1:
                src += ["    %s_y%d = tl.zeros([BLOCK], dtype=tl.float64)" % (lst, l)]
    for lst, (pa, pc, pe, pm, pcf, pcn, pk, pp, pn, NS) in (("s", ("sa", "sc", "se", "sm", "scf", "scn", "sk", "sp", "sn", "S")),
                                                          ("b", ("ba", "bc", "be", "bm", "bcf", "bcn", "bk", "bp", "bn", "B"))):
        src += ["    n_ = tl.minimum(tl.load(%s + t, mask=ok, other=0), %s)" % (pn, NS),
                "    for j in range(0, tl.max(n_, axis=0)):",
                I + "on = ok & (j < n_)",
                I + "o = t * %s + j" % NS,
                I + "act = (tl.load(%s + o, mask=on, other=0) != 0) & on" % pa,
                I + "mass = tl.load(%s + o, mask=act, other=0.0)" % pm,
                I + "cfrac = tl.load(%s + o, mask=act, other=0.0)" % pcf,
                I + "cn = tl.load(%s + o, mask=act, other=0.0)" % pcn,
                I + "k = tl.load(%s + o, mask=act, other=0.0)" % pk,
                I + "cyc = tl.load(%s + o, mask=act, other=0)" % pc,
                I + "e = tl.minimum(tl.maximum(tl.load(%s + o, mask=act, other=1) - 1, 0), %d)" % (pe, L - 1),
                I + "tfe = _sel_l(%s, e, %d)" % (vs("tf"), L),
                I + "mfe = _sel_l(%s, e, %d)" % (vs("mf"), L),
                I + "carbon = mass * cfrac",
                I + "ex = act & (mass > 0.0) & (carbon > 0.0)",
                I + "pot = carbon * %s * (1.0 - ld.exp(-k * tfe)) * mfe" % ("fc" if lst == "s" else "1.0"),
                I + "pot = tl.where(ex, pot, 0.0)"]
        if lst == "s":
            src.append(I + "ly = e * 0")
        else:
            src.append(I + "ly = tl.minimum(tl.maximum(tl.load(bl + o, mask=act, other=1) - 1, 0), %d)" % (L - 1))
        if PHASE == 1:
            src += [I + "ncd = tl.where(cn == 0.0, 0.0, 1.0 / tl.where(cn == 0.0, 1.0, cn))",
                    I + "dec_n = pot * ncd",
                    I + "rcn = tl.where(cyc == 3, meta, micro)",
                    I + "ncr = tl.where(rcn == 0.0, 0.0, 1.0 / tl.where(rcn == 0.0, 1.0, rcn))",
                    I + "net = dec_n - (hum * pot) * ncr",
                    I + "cn_ok = ~(cn < 1.0)",
                    I + "dem = tl.where(ex & cn_ok & (net < 0.0), -net, 0.0)",
                    I + "nmin = tl.where(ex & cn_ok & ~(net < 0.0), net, 0.0)"]
            for l in range(L if lst == "b" else 1):
                src += [I + "%s_x%d = tl.where(on & (ly == %d), %s_x%d + nmin, %s_x%d)" % (lst, l, l, lst, l, lst, l),
                        I + "%s_y%d = tl.where(on & (ly == %d), %s_y%d + dem, %s_y%d)" % (lst, l, l, lst, l, lst, l)]
        else:
            src += [I + "c = mass * cfrac",
                    I + "a = pot * _sel_l(%s, ly, %d)" % (vs("rf"), L),
                    I + "a = tl.where(c < a, c, a)",
                    I + "a = tl.where(ex, a, 0.0)",
                    I + "co2 = a * 0.55",
                    I + "pend = co2 + a * hum",
                    I + "go = ex & (pend > 0.0)",
                    I + "removed = pend / tl.where(go, cfrac, 1.0)",
                    I + "m2 = tl.where(go, _add_mass(mass, -removed), mass)",
                    I + "tl.store(%s + o, m2, mask=go)" % pm,
                    I + "pos = tl.load(%s + o, mask=act, other=0)" % pp,
                    I + "gone = act & _abs_le(0.0 - m2, 0.0000001) & (pos != 3)",
                    I + "tl.store(%s + o, act & ~gone, mask=gone)" % pa]
            for l in range(L if lst == "b" else 1):
                src += [I + "%s_x%d = tl.where(on & (ly == %d), %s_x%d + co2, %s_x%d)" % (lst, l, l, lst, l, lst, l)]
    for l in range(L):
        src.append("    tl.store(o1_ptr + t * %d + %d, s_x%d + b_x%d, mask=ok)" % (L, l, l, l))
        if PHASE == 1:
            src.append("    tl.store(o2_ptr + t * %d + %d, s_y%d + b_y%d, mask=ok)" % (L, l, l, l))
    return "\n".join(src) + "\n"


_PKERNELS = {}


def _process_kernel(L, PHASE):
    k = _PKERNELS.get((L, PHASE))
    if k is None:
        import hashlib
        import importlib.util
        import sys
        src = _process_source(L, PHASE)
        h = hashlib.sha256(src.encode()).hexdigest()[:16]
        d = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
        d = os.path.join(d, "torchvic_omproc")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "omproc_%s.py" % h)
        if not os.path.exists(path):
            tmp = "%s.%d" % (path, os.getpid())
            with open(tmp, "w") as fh:
                fh.write(src)
            os.replace(tmp, path)
        name = "torchvic_omproc_%s" % h
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        k = _PKERNELS[(L, PHASE)] = mod.kern
    return k


_PBLOCK = 64


def _pargs(sur, sub):
    # active, cycling, env, mass, cfrac, cn, k, position, n (surface); + layer (subsurface)
    a = [sur[0], sur[4], sur[5], sur[6], sur[7], sur[8], sur[9], sur[2], sur[12]]
    b = [sub[0], sub[4], sub[5], sub[3], sub[6], sub[7], sub[8], sub[9], sub[2], sub[12]]
    return a, b


@torch.library.custom_op("torchvic::cs_om_process_n", mutates_args=())
def _proc1_op(sur: List[torch.Tensor], sub: List[torch.Tensor], fc: torch.Tensor, tf: torch.Tensor,
              mf: torch.Tensor, par: torch.Tensor) -> List[torch.Tensor]:
    T, L = tf.shape
    mineral, demand = torch.empty_like(tf), torch.empty_like(tf)
    if T:
        a, b = _pargs(sur, sub)
        _process_kernel(L, 1)[((T + _PBLOCK - 1) // _PBLOCK,)](
            *a, *b, fc, tf, mf, tf, par, mineral, demand, T, sur[0].shape[1], sub[0].shape[1],
            BLOCK=_PBLOCK, num_warps=2, enable_fp_fusion=False)
    return [mineral, demand]


@_proc1_op.register_fake
def _(sur, sub, fc, tf, mf, par):
    return [torch.empty_like(tf), torch.empty_like(tf)]


@torch.library.custom_op("torchvic::cs_om_process_c_", mutates_args=("sur", "sub"))
def _proc2_op(sur: List[torch.Tensor], sub: List[torch.Tensor], fc: torch.Tensor, tf: torch.Tensor,
              mf: torch.Tensor, rf: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
    T, L = tf.shape
    co2 = torch.empty_like(tf)
    if T:
        a, b = _pargs(sur, sub)
        _process_kernel(L, 2)[((T + _PBLOCK - 1) // _PBLOCK,)](
            *a, *b, fc, tf, mf, rf, par, co2, co2, T, sur[0].shape[1], sub[0].shape[1],
            BLOCK=_PBLOCK, num_warps=2, enable_fp_fusion=False)
    return co2


@_proc2_op.register_fake
def _(sur, sub, fc, tf, mf, rf, par):
    return torch.empty_like(tf)


def process_(r: om.Residues, env, s, microbial_cn=om.MICROBIAL_CN_DEFAULT, metastable_cn=om.METASTABLE_CN_DEFAULT,
             humification=om.HUMIFICATION_FRACTION):
    """organic_matter.process IN PLACE on r's lists (the day's own), the per-layer sums in list
    order (the CPU's scatter_add order; torch's CUDA scatter_add adds in a different order on every
    run). Returns (residues, soil N, Decomposition without the per-pool arrays)."""
    from . import soil_n as sn
    tf = env.tf_daily_mean.contiguous()
    mf = env.moisture_daily.contiguous()
    fl = lambda v: torch.full((), v, dtype=tf.dtype, device=tf.device)
    par = torch.stack([fl(microbial_cn), fl(metastable_cn), fl(humification)])
    for x in list(r.sur) + list(r.sub):
        assert x.is_contiguous()
    fc = r.flat_contact.contiguous()
    mineral, demand = _proc1_op(list(r.sur), list(r.sub), fc, tf, mf, par)
    rf, im4, im3 = om.reduction_factor(demand, sn.no3_n(s), sn.nh4_n(s))
    co2 = _proc2_op(list(r.sur), list(r.sub), fc, tf, mf, rf.contiguous(), par)
    no3 = s.no3 - om._cmin(im3 * sn.N_TO_NO3, s.no3)
    nh4 = s.nh4 - om._cmin(im4 * sn.N_TO_NH4, s.nh4)
    nh4 = nh4 + mineral * sn.N_TO_NH4
    return r, sn.SoilN(no3, nh4), om.Decomposition(co2, mineral, demand, None, None)
