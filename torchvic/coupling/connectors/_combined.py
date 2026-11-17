"""Reservoirs and water quality in one router sweep.

`WaterQuality._sweep` (withdraw / inject with constituents and heat) and
`Reservoirs.apply` (intercept, operate, release) each walk the router's
levels, so neither can host the other. `ReservoirQuality` is a
`WaterQuality` whose sweep does both, level by level, upstream to
downstream, so a dam sees the water and the mass its upstream neighbours
released in the same step.

Water: exactly `Reservoirs.apply`'s arithmetic (the take is
min(cap, passing), the release and spill are injected at the same site),
so discharge and storage equal a reservoir-only router bit for bit.

Constituents and heat: the mass the take removes from the channel rings
(in proportion to the water, as for any withdrawal) enters the dam's store.
Each dam is a well-mixed store:

    M_decay = M exp(-k dt)                      first-order decay (k = 0 for heat)
    V_mix   = S + I - E                         volume after inflow and evaporation
    C       = (M_decay + M_in) / V_mix
    released   C (env + release + spill)        injected at the site through the
                                                channel kernels (as an injection)
    supplied   C W                              leaves with the water-use supply
    M_new   = M_decay + M_in - C (outflow)

so mass (and heat, m3 K) is conserved: the store identity
    M_store = removed - released - supplied - decayed
holds to rounding and is exposed (`store_residual`). Evaporation removes
water only (the mass stays, concentrating the store); there is no heat
exchange with the atmosphere in the dam, and the initial storage is clean
(zero mass, zero heat). A dam not yet built passes mass through like water.
"""
import torch

from torchvic.quality import WaterQuality


class ReservoirQuality(WaterQuality):
    def __init__(self, router, area, constituents, temperature, site_cells,
                 reservoirs):
        super().__init__(router, area, constituents or None,
                         temperature=temperature, site_cells=site_cells)
        self.res = reservoirs
        dev, dt = self.dev, self.dtype
        R = reservoirs.R
        self.cn = len(self.names)
        self.ns = self.cn + (1 if self.temp is not None else 0)
        z = lambda *s: torch.zeros(*s, dtype=dt, device=dev)
        self.store = z(self.ns, R)
        self.cum = z(4, self.ns, R)           # removed, released, supplied, decayed
        kk = [float(constituents[n]) for n in self.names]
        if self.temp is not None:
            kk.append(0.0)
        self.decay = torch.exp(-torch.tensor(kk, dtype=dt, device=dev)
                               * self.dt).unsqueeze(1)          # [ns, 1]
        self._ext = None
        self.outlets = None            # set by the connector

    def state_tensors(self):
        return [self.store, self.cum]

    def store_residual(self):
        """[species, R]: store - (removed - released - supplied - decayed);
        species are the constituents, then heat."""
        c = self.cum
        return self.store - (c[0] - c[1] - c[2] - c[3])

    # -- the combined sweep ------------------------------------------------
    def _species(self, x):
        """[channels, n] -> [species, n]: constituents, then heat (the sum
        of the two temperature channels)."""
        if self.temp is None:
            return x
        return torch.cat([x[:self.cn], x[self.cn:].sum(0, keepdim=True)], 0)

    def _sweep(self, got, withdraw, inject, point, inject_conc, inject_temp):
        r, res, dt, L = self.r, self.res, self.dt, self.r.L
        ext = self._ext
        res.nstep += 1
        slots = r._slots()
        zs = torch.zeros(r.n_site, dtype=self.dtype, device=self.dev)
        for li, lv in enumerate(r.levels):
            sid, ids = lv['sites'], lv['tree'].ids
            n = len(sid)
            cap = r._cap(lv)
            node = r.site_node[sid]
            avail = r.head_discharge()[node].clamp(min=0) * dt
            g = res.groups.get(li)
            ours = res._is_res[sid]
            take, put = zs[:n], zs[:n]
            if withdraw is not None:
                take = torch.where(ours, 0.0, withdraw[sid].clamp(min=0))
                take = torch.minimum(take, cap)
                got.index_copy_(0, sid, take)
            if inject is not None:
                put = torch.where(ours, 0.0, inject[sid].clamp(min=0))
            if g is not None:
                pos, ix = g['pos'], g['ix']
                I = torch.minimum(cap[pos], avail[pos])
                take = take.index_copy(0, pos, I)
                S_old = res.storage.index_select(0, ix)
                down = res._operate(g, I, ext)
                put = put.index_copy(0, pos, down)
            take = torch.minimum(take.clamp(min=0), cap)
            put = put.clamp(min=0)
            # a dam not yet built passes the water only: its mass stays in
            # the river untouched
            take_m = take
            if g is not None:
                on = torch.ones(n, dtype=self.dtype, device=self.dev)
                on = on.index_copy(0, pos, (res.ctl.index_select(0, ix)[:, 3]
                                            > 0.5).to(self.dtype))
                take_m = take * on
            # mass and heat removed with the water (proportionally, as for
            # any withdrawal), from the water ring before this level moves
            frac = torch.where(avail > 0, take_m / avail.clamp(min=1e-300),
                               torch.zeros_like(take))
            passing = self.ring_all.index_select(1, r.head.view(1)) \
                .squeeze(1)[:, node].clamp(min=0)               # [C, n]
            removed = frac * passing * dt
            self.wd_all.index_add_(1, sid, removed)
            t = lv['tree'].apply(-take_m[lv['pair_site']] / dt)  # [ids, L + 1]
            si, ni = slots[:, None], ids[None, :]
            qw = r.ring[si, ni]
            a_ = t[:, :L].t()
            fr = torch.where(qw > 0, a_ / qw.clamp(min=1e-300),
                             torch.zeros_like(qw)).clamp(-1.0, 0.0)
            cur = self.ring_all[:, si, ni]
            d = cur * fr
            self.ring_all[:, si, ni] = cur + d
            self.hooks_all.index_add_(1, ids, d.sum(1) * dt)
            # the water: take and put together, as Reservoirs.apply
            r._apply_level(lv, put - take)
            r.withdrawn.index_add_(0, sid, take)
            r.injected.index_add_(0, sid, put)
            m_put = torch.zeros(self.ns, n, dtype=self.dtype, device=self.dev)
            if g is not None:
                # what leaves the outlets' rings with this level's take is
                # the mass that is gone from the river; the dams get their
                # share of it (the head-slot estimate `removed` differs
                # from it by the concentration gradient along the ring)
                # the level's outlet columns, found once (a boolean mask
                # here would sync the host on every sweep)
                oc = self.__dict__.setdefault('_out_cols', {})
                if li not in oc:
                    oc[li] = torch.isin(ids, self.outlets).nonzero()[:, 0]
                gone = -(d.sum(1).index_select(1, oc[li]).sum(1)) * dt  # [C]
                est = removed.sum(1)
                ratio = torch.where(est > 0, gone / est.clamp(min=1e-300),
                                    torch.ones_like(est))
                m_put = self._store_step(g, pos, ix, S_old, I,
                                         removed * ratio.unsqueeze(1), m_put)
            # other injections and sources, as WaterQuality._sweep
            inj_w = None if inject is None else torch.where(
                ours, 0.0, inject[sid].clamp(min=0))
            for i, nm in enumerate(self.names):
                m = m_put[i]
                if point and nm in point:
                    m = m + point[nm][sid].clamp(min=0)
                if inject_conc and nm in inject_conc and inj_w is not None:
                    m = m + inj_w * inject_conc[nm][sid].clamp(min=0)
                self._put(i, li, sid, m)
            if self.temp is not None:
                m = m_put[self.cn]
                if inject_temp is not None and inj_w is not None:
                    m = m + inj_w * inject_temp[sid].clamp(min=0)
                self._put(self.cn, li, sid, m)

    def _store_step(self, g, pos, ix, S_old, I, removed, m_put):
        """Well-mixed store of the dams of one level; returns m_put with the
        released mass [species, n_level_sites] filled."""
        res = self.res
        L = res.last
        act = res.ctl.index_select(0, ix)[:, 3] > 0.5
        raw = L['raw'].index_select(0, ix)
        infl = L['inflow'].index_select(0, ix)
        ev = L['evap'].index_select(0, ix)
        W = L['withdrawn'].index_select(0, ix)
        outd = (L['env'] + L['release'] + L['spill']).index_select(0, ix)
        m_in = self._species(removed)[:, pos]                    # [ns, nr]
        M = self.store[:, ix]
        Md = M * self.decay
        dec = M - Md
        tot = Md + m_in
        vmix = S_old + infl - ev
        conc = torch.where(vmix > 0, tot / vmix.clamp(min=1e-300),
                           torch.zeros_like(tot))
        rel_a = conc * outd
        sup = conc * W
        Mn = tot - rel_a - sup
        zero = torch.zeros_like(Mn)
        # a dam not yet built neither takes nor releases mass
        sup = torch.where(act, sup, zero)
        dec = torch.where(act, dec, zero)
        rel = torch.where(act, rel_a, zero)
        Mn = torch.where(act, Mn, M)
        self.store[:, ix] = Mn
        inn = torch.where(act, m_in, zero)
        self.cum[0, :, ix] += inn
        self.cum[1, :, ix] += rel
        self.cum[2, :, ix] += sup
        self.cum[3, :, ix] += dec
        m_put[:, pos] = rel
        return m_put
