"""Irrigation and human water-use engine, after VIC-WUR (Droppers et al. 2020).

Equations only; no VIC-WUR source is copied.  Reference lines are in
vicwur/vic/plugins (<data root>/static/upstream/vicwur, bb12fb3d).

All tensors live on one device, every operation is vectorised over cells, and
a step has no host synchronisation and no data-dependent shapes (CUDA-graph
capturable).  Volumes are m3 per step; soil quantities are mm over the class
area (VIC convention).

Sectors are columns of a [cell, S] demand array in PRIORITY ORDER (VIC-WUR
enum order: municipal, energy, manufacturing, livestock, irrigation;
plugin_def.h:218-226).  The last column is the dynamic irrigation sector when
`irrigation=True`; its demand is computed from the soil, not read.

Per cell and sector, a withdrawal W is split
    conveyance loss   cl*W
    delivered         (1-cl)*W
        consumed      c*delivered   (irrigation: goes into the soil)
        field loss    (1-c)*delivered
Losses and non-irrigation return flows go to the river (`inject`).  For
irrigation they go to the river or to the soil (`loss_return`).

Ledger identities (cumulative, per cell, m3), checked by `balance()`:
    demand            = withdrawn + unmet
    withdrawn         = consumed_other + (conveyance + field loss of others)
                        + irr_soil_in + irr_loss_river + irr_loss_soil
    irr_soil_in + irr_loss_soil = soil_applied + leftover_change
"""
from dataclasses import dataclass
import torch


@dataclass
class WaterUseConfig:
    # Shortfall sharing between sectors on one source.  VIC-WUR code is a
    # sequential priority (wu_run_local.c:194-250); its paper/docs say
    # proportional.  Default follows the code.
    allocation: str = "priority"          # priority | proportional
    # Irrigation losses (conveyance + field): river = code (wu_run_local.c:
    # 288-335, 469-540), soil = paper (Sect. 2.3.1).
    loss_return: str = "river"            # river | soil
    # Unmet demand of the groundwater (surface) portion tries the other source.
    fallback: bool = True
    # Surface order is an engine choice, not the code: wu_run_local.c has no reservoir-first step;
    # local dam and river availability form one pool shared pro rata (wu_run_local.c:194-250).
    reservoir_first: bool = True
    # Irrigated top layer may be filled only to Wfc[0] (EFFICIENT_IRRIGATION,
    # irr_use.c:238-244) instead of max_moist[0].
    efficient: bool = False
    eff_min: float = 0.05                 # clamp of application efficiency
    dtype: torch.dtype = torch.float64


class StubSource:
    """Groundwater / reservoir engine interface.

    available() -> [cell] m3 ; withdraw(request[cell]) -> delivered[cell].
    The stub has a fixed (or unlimited) store and keeps the cumulative draw.
    A real engine implements the same two methods."""

    def __init__(self, store):
        self.store = store.clone()
        self.drawn = torch.zeros_like(store)

    def available(self):
        return self.store

    def withdraw(self, request):
        d = torch.minimum(request.clamp(min=0), self.store)
        self.store = self.store - d
        self.drawn = self.drawn + d
        return d


def irrigation_requirement(moist, ice, frost_fract, wfc, wcr, max_moist0, root,
                           leftover, paddy, season_open):
    """Requirement and demand flag per class.

    moist, ice [N,K,Lyr] mm; frost_fract [N,K,Nf] or None (ice then [N,K,Lyr]);
    wfc, wcr [N,K,Lyr]; max_moist0 [N,K]; root [N,K,Lyr] (rooted if > 0);
    leftover [N,K]; paddy bool [N,K]; season_open bool [N,K].
    Returns (req [N,K] mm >= 0 where flagged else 0, flag [N,K]).
    irr_run.c:129-177: liquid = sum (moist-ice)*frost_fract; conventional
    req = sum_root Wfc - (sum_root liquid + leftover) (155-157); paddy
    req = max_moist[0] - (liquid[0] + leftover) (148-150); flag when the
    season offset is passed, req > 0 and the critical point Wcr is undershot
    (conventional, 172-177) or layer 0 is below max_moist[0] (paddy, 165-171).
    """
    liquid = moist - ice
    rooted = root > 0
    tot = _ssum(liquid * rooted)
    tfc = _ssum(wfc * rooted)
    tcr = _ssum(wcr * rooted)
    req_c = tfc - (tot + leftover)
    req_p = max_moist0 - (liquid[..., 0] + leftover)
    req = torch.where(paddy, req_p, req_c)
    trig = torch.where(paddy, max_moist0 - (liquid[..., 0] + leftover),
                       tcr - (tot + leftover))
    flag = season_open & (trig > 0) & (req > 0)
    return torch.where(flag, req, torch.zeros_like(req)), flag


def _ssum(x, keepdim=False):
    """Sum over the last (short, static) axis as unrolled adds: far faster on a
    GPU than a reduce/scan kernel over a 3-5 wide innermost dim."""
    c = x.unbind(-1)
    t = c[0]
    for u in c[1:]:
        t = t + u
    return t.unsqueeze(-1) if keepdim else t


def _scum(x):
    c, t, out = x.unbind(-1), None, []
    for u in c:
        t = u if t is None else t + u
        out.append(t)
    return torch.stack(out, -1)


def _alloc(d, a, proportional):
    """Share availability a[N] over sector demands d[N,S] (columns = priority)."""
    if proportional:
        tot = _ssum(d, True)
        f = torch.where(tot > 0, (a.unsqueeze(-1) / tot.clamp(min=1e-300)),
                        torch.zeros_like(tot)).clamp(max=1.0)
        return d * f
    before = _scum(d) - d
    return (a.unsqueeze(-1) - before).clamp(min=0).minimum(d)


class WaterUseEngine:
    """cells N, irrigation classes K (tile x band), sectors S.

    area [N] m2.  site_of_cell [N] long: router site index of the cell's
    withdrawal/injection point, -1 for none (no river water, returns go to
    the `return_unrouted` ledger).  cons_frac [N,S], conv_loss [N,S],
    gw_frac [N,S] are sector parameters (irrigation column: gw_frac is the
    demand-weighted value of the classes; consumption = application
    efficiency).
    """

    def __init__(self, area, site_of_cell=None, n_other=4, irrigation=True,
                 cons_frac=None, conv_loss=None, gw_frac=None,
                 irr_eff=None, irr_gw_frac=None, irr_conv_loss=None,
                 cfg=None):
        self.cfg = cfg or WaterUseConfig()
        dt, dev = self.cfg.dtype, area.device
        self.dt_, self.dev = dt, dev
        self.area = area.to(dt)
        self.N = area.shape[0]
        self.n_other, self.irrigation = n_other, irrigation
        self.S = n_other + int(irrigation)
        z = lambda *s: torch.zeros(*s, dtype=dt, device=dev)
        self.cons = z(self.N, n_other) if cons_frac is None else cons_frac.to(dt)
        self.cl = z(self.N, n_other) if conv_loss is None else conv_loss.to(dt)
        self.gwf = z(self.N, n_other) if gw_frac is None else gw_frac.to(dt)
        self.irr_eff = (torch.ones(self.N, dtype=dt, device=dev)
                        if irr_eff is None else irr_eff.to(dt))
        self.irr_gwf = z(self.N) if irr_gw_frac is None else irr_gw_frac.to(dt)
        self.irr_cl = z(self.N) if irr_conv_loss is None else irr_conv_loss.to(dt)
        self.site = (torch.full((self.N,), -1, dtype=torch.long, device=dev)
                     if site_of_cell is None else site_of_cell.to(dev))
        self.has_site = self.site >= 0
        self.site_c = self.site.clamp(min=0)
        self.n_site = int(self.site.max().item()) + 1 if self.has_site.any() else 0
        self.reset()

    # -- state and ledger --------------------------------------------------
    def reset(self):
        z = lambda *s: torch.zeros(*s, dtype=self.dt_, device=self.dev)
        self.leftover = None          # [N,K] m3 (VIC-WUR keeps mm), first step
        self.led = {k: z(self.N) for k in (
            "demand", "wd_gw", "wd_res", "wd_riv", "unmet",
            "consumed_other", "ret_river_other", "irr_soil_in",
            "irr_loss_river", "irr_loss_soil", "soil_applied",
            "return_unrouted")}
        self.led_S = {k: z(self.N, self.S) for k in ("demand", "withdrawn")}
        self.leftover_m3 = z(self.N)  # current stored leftover, m3

    def state_dict(self):
        return {"leftover": None if self.leftover is None else self.leftover.clone(),
                "leftover_m3": self.leftover_m3.clone(),
                "led": {k: v.clone() for k, v in self.led.items()},
                "led_S": {k: v.clone() for k, v in self.led_S.items()}}

    def load_state_dict(self, sd):
        self.leftover = None if sd["leftover"] is None else sd["leftover"].clone()
        self.leftover_m3 = sd["leftover_m3"].clone()
        for k in self.led:
            self.led[k] = sd["led"][k].clone()
        for k in self.led_S:
            self.led_S[k] = sd["led_S"][k].clone()

    def balance(self):
        """[N] m3 residuals: (demand closure, withdrawal closure, soil closure)."""
        L = self.led
        r1 = L["demand"] - L["wd_gw"] - L["wd_res"] - L["wd_riv"] - L["unmet"]
        wd = L["wd_gw"] + L["wd_res"] + L["wd_riv"]
        r2 = wd - (L["consumed_other"] + L["ret_river_other"]
                   + L["irr_soil_in"] + L["irr_loss_river"] + L["irr_loss_soil"])
        r3 = L["irr_soil_in"] + L["irr_loss_soil"] - L["soil_applied"] - self.leftover_m3
        return torch.stack([r1, r2, r3])

    # -- one step -----------------------------------------------------------
    def step(self, demand_other=None, soil=None, router=None, gw=None, res=None,
             river_avail=None):
        """One daily step; returns a dict of per-cell volumes and the soil add.

        demand_other [N,n_other] m3 withdrawal demand (priority order).
        soil: dict with moist, ice, wfc, wcr, root [N,K,Lyr], max_moist0 [N,K],
              frac [N,K] (class area fraction of the cell), paddy,
              season_open [N,K] bool, moist0 [N,K] (layer-0 moisture).
        router: StepRouter (between begin() and end()); sites = site_of_cell.
        gw, res: objects with available()/withdraw() (StubSource interface).
        river_avail: [N] m3 override when no router is given.
        """
        c, dt, dev, N = self.cfg, self.dt_, self.dev, self.N
        prop = c.allocation == "proportional"
        zero = torch.zeros(N, dtype=dt, device=dev)
        # -- demands ---------------------------------------------------------
        cons, cl, gwf = self.cons, self.cl, self.gwf
        d = torch.zeros(N, 0, dtype=dt, device=dev) if demand_other is None \
            else demand_other.to(dt).clamp(min=0)
        if self.irrigation:
            if self.leftover is None:
                self.leftover = torch.zeros(soil["frac"].shape, dtype=dt, device=dev)
            frac = soil["frac"].to(dt)
            ca = (frac * self.area.unsqueeze(-1) / 1000.0).clamp(min=1e-300)
            req, flag = irrigation_requirement(
                soil["moist"], soil["ice"], None, soil["wfc"], soil["wcr"],
                soil["max_moist0"], soil["root"], self.leftover / ca,
                soil["paddy"], soil["season_open"])
            req_cell_m3 = _ssum(req * frac) * self.area / 1000.0   # m3 to soil
            e = self.irr_eff.clamp(min=c.eff_min, max=1.0)
            d_irr = req_cell_m3 / (e * (1.0 - self.irr_cl))           # irr_use.c:97
            d = torch.cat([d, d_irr.unsqueeze(-1)], 1)
            cons = torch.cat([cons, e.unsqueeze(-1)], 1)
            cl = torch.cat([cl, self.irr_cl.unsqueeze(-1)], 1)
            gwf = torch.cat([gwf, self.irr_gwf.unsqueeze(-1)], 1)
        # -- availability -----------------------------------------------------
        a_gw = zero if gw is None else gw.available().to(dt).clamp(min=0)
        a_res = zero if res is None else res.available().to(dt).clamp(min=0)
        if router is not None and self.n_site:
            av_site = router.available().to(dt)
            a_riv = torch.where(self.has_site, av_site[self.site_c], zero)
        elif river_avail is not None:
            a_riv = river_avail.to(dt).clamp(min=0)
        else:
            a_riv = zero
        # -- allocation by source in order ------------------------------------
        p_gw, p_sf = d * gwf, d * (1.0 - gwf)
        sur = [("res", a_res), ("riv", a_riv)] if c.reservoir_first \
            else [("riv", a_riv), ("res", a_res)]
        got = {"gw": torch.zeros_like(d), "res": torch.zeros_like(d),
               "riv": torch.zeros_like(d)}
        rem = {"gw": a_gw, "res": a_res, "riv": a_riv}
        plan = [("gw", p_gw, "gw")] + [(n, p_sf, "sf") for n, _ in sur]
        if c.fallback:
            plan += [("gw", p_sf, "sf")] + [(n, p_gw, "gw") for n, _ in sur]
        r_gw, r_sf = p_gw, p_sf
        for src, _, portion in plan:
            r = r_gw if portion == "gw" else r_sf
            a = _alloc(r, rem[src], prop)
            got[src] = got[src] + a
            rem[src] = rem[src] - _ssum(a)
            if portion == "gw":
                r_gw = r_gw - a
            else:
                r_sf = r_sf - a
        # -- external engines may deliver less ---------------------------------
        def ratio(deliv, req):
            return torch.where(req > 0, deliv / req.clamp(min=1e-300),
                               torch.ones_like(req)).clamp(max=1.0)
        if gw is not None:
            rq = _ssum(got["gw"])
            got["gw"] = got["gw"] * ratio(gw.withdraw(rq).to(dt), rq).unsqueeze(-1)
        if res is not None:
            rq = _ssum(got["res"])
            got["res"] = got["res"] * ratio(res.withdraw(rq).to(dt), rq).unsqueeze(-1)
        if router is not None and self.n_site:
            rq_c = _ssum(got["riv"])
            rq_s = torch.zeros(self.n_site, dtype=dt, device=dev).index_add_(
                0, self.site_c, torch.where(self.has_site, rq_c, zero))
            dl_s = router.withdraw(rq_s).to(dt)
            rs = ratio(dl_s, rq_s)
            got["riv"] = got["riv"] * rs[self.site_c].unsqueeze(-1)
        W = got["gw"] + got["res"] + got["riv"]                  # [N,S] m3
        unmet = d - W
        # -- splits --------------------------------------------------------------
        conv = W * cl
        deliv = W - conv
        consumed = deliv * cons
        field = deliv - consumed
        loss = conv + field
        if self.irrigation:
            irr_cons, irr_loss = consumed[:, -1], loss[:, -1]
            consumed_o = _ssum(consumed[:, :-1]) if self.n_other else zero
            ret_o_river = _ssum(loss[:, :-1]) if self.n_other else zero
        else:
            irr_cons = irr_loss = zero
            consumed_o = _ssum(consumed)
            ret_o_river = _ssum(loss)
        to_soil_loss = irr_loss if c.loss_return == "soil" else zero
        irr_river = irr_loss if c.loss_return == "river" else zero
        # conveyance and field losses are counted in the return items (a partition of W)
        river_ret = ret_o_river + irr_river                       # m3 injected
        inj_c = torch.where(self.has_site, river_ret, zero)
        if router is not None and self.n_site:
            inj_s = torch.zeros(self.n_site, dtype=dt, device=dev).index_add_(
                0, self.site_c, inj_c)
            router.inject(inj_s)
        # -- soil ----------------------------------------------------------------
        out = {}
        if self.irrigation:
            soil_in = irr_cons + to_soil_loss                     # m3 per cell
            d_irr_ = d[:, -1]
            f = ratio(irr_cons, req_cell_m3)                      # fraction of req met
            # share of cell soil_in per class: proportional to req*frac
            w = (req * frac)
            ws = _ssum(w, True)
            share = torch.where(ws > 0, w / ws.clamp(min=1e-300), torch.zeros_like(w))
            # mm over the class area (class area = frac*area); leftover state in m3
            recv = soil_in.unsqueeze(-1) * share / ca + self.leftover / ca
            cap_to = soil["wfc"][..., 0] if c.efficient else soil["max_moist0"]
            cap_to = torch.where(soil["paddy"], soil["max_moist0"], cap_to) \
                if c.efficient else soil["max_moist0"]
            head = (cap_to - soil["moist0"]).clamp(min=0)
            applied = torch.minimum(recv, head)                   # irr_use.c:238-253
            left = recv - applied
            self.leftover = left * ca
            applied_m3 = _ssum(applied * ca)
            left_m3 = _ssum(self.leftover)
            self.leftover_m3 = left_m3
            out.update(soil_add=applied, overflow=left, req=req, flag=flag,
                       soil_in=soil_in, applied_m3=applied_m3, leftover_m3=left_m3)
            # ledger: leftover carried in is part of soil closure through leftover_m3 only
            # (soil_in this step + previous leftover = applied + new leftover)
        else:
            soil_in = applied_m3 = zero
        L = self.led
        # unrouted returns leave the ledger as an explicit item
        unrouted = torch.where(self.has_site, zero, river_ret)
        L["demand"] += _ssum(d)
        L["wd_gw"] += _ssum(got["gw"])
        L["wd_res"] += _ssum(got["res"])
        L["wd_riv"] += _ssum(got["riv"])
        L["unmet"] += _ssum(unmet)
        L["consumed_other"] += consumed_o
        L["ret_river_other"] += ret_o_river
        L["irr_soil_in"] += irr_cons
        L["irr_loss_river"] += irr_river
        L["irr_loss_soil"] += to_soil_loss
        L["soil_applied"] += applied_m3
        L["return_unrouted"] += unrouted
        self.led_S["demand"] += d
        self.led_S["withdrawn"] += W
        out.update(demand=d, withdrawn=W, unmet=unmet, gw=got["gw"],
                   res=got["res"], river=got["riv"], consumed=consumed,
                   river_return=river_ret, conv_loss=conv, field_loss=field)
        return out
