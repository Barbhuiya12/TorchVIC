"""VIC output variables formed from rows `basin.simulate` already returns.

VIC 5.1.0 writes 88 variables in a plain water-balance run; before this
module TorchVIC wrote 53. Most of the 35 it left out are not new physics:
they are functions of a row the step loop holds (put_data.c's "derived
variables", the forcing echo, a store VIC zeroes where an option is off).
`derive` forms those, in the units and with the arithmetic put_data.c uses,
from the `out` dict at the end of the call, so no step of the loop changes
and every row that existed is left exactly as it was.

What is formed here (put_data.c 5.1.0 line numbers; 5.0.1's are the same
statements a few lines earlier):

* OUT_QAIR (:106-107): CONST_EPS * vp / pressure, both in Pa.
* OUT_SMLIQFRAC, OUT_SMFROZFRAC, OUT_SOIL_LIQ_FRAC, OUT_SOIL_ICE_FRAC
  (:703-717): from the layer liquid and ice the cell reports, the depth in
  mm (`depth * MM_PER_M`, the product first). OUT_SMLIQFRAC is liq / (liq +
  ice) and is NaN, as VIC's is, in a layer holding no water at all.
* OUT_SURFSTOR / OUT_DELSURFSTOR (:305-308, :730): 0 without LAKES (the lake
  step fills them otherwise).
* OUT_CATM, OUT_FDIR, OUT_PAR (:134-145): the carbon forcing echo with
  CARBON TRUE, VIC's MISSING (-99999) without it.
* OUT_ADV_SENS: the advected sensible heat of SPATIAL_SNOW, 0 without it
  (SnowPackEnergyBalance.c:212-215).
* OUT_SUB_BLOWING: 0 without BLOWING (snow->blowing_flux stays at the 0
  initialize_snow.c gives it).
* OUT_SOIL_TNODE_WL: the wetland's node temperatures, 0 without LAKES.
* OUT_ENERGY_ERROR, `energy_error='vic'`: put_data.c:538-552 forms it from
  the cell's reported rows (calc_energy_balance_error), not from the root
  solver's residual; with FULL_ENERGY FALSE it is MISSING.

Every variable named above is left out of `out` when the configuration does
not define it (a water-balance run has no energy rows to form OUT_ENERGY_ERROR
from, and VIC writes its MISSING).
"""
import torch

from ..core import snow as _snow

#: vic_def.h:34
MISSING = -99999.0
#: vic_physical_constants.h:37
PPM_TO_MIXRATIO = 1.0e-6
MM_PER_M = 1000.0
MAX_FRONTS = 3
#: vic_def.h / parameters: param.HUGE_RESIST, and float.h DBL_EPSILON
HUGE_RESIST = 1.0e20
DBL_EPSILON = 2.220446049250313e-16

#: put_data.c:538-552's terms, as `out` keys
ENERGY_ERROR_ROWS = ('eb_net_short', 'eb_net_long', 'eb_latent', 'eb_sensible',
                     'eb_grnd_flux', 'eb_delta_h', 'eb_fusion',
                     'eb_advection', 'eb_deltacc', 'eb_snow_flux',
                     'eb_rfrz_energy')


def energy_error(out):
    """calc_energy_balance_error() on the reported rows, put_data.c:539-551.

    VIC passes (SWNET + LWNET, LATENT + LATENT_SUB, SENSIBLE + ADV_SENS,
    GRND_FLUX + DELTAH + FUSION, ADVECTION - DELTACC + SNOW_FLUX +
    RFRZ_ENERGY) and returns net_rad - latent - sensible - grnd_flux +
    snow_fluxes; the grouping and the order are VIC's, so the result is the
    C's to the bit when the rows are."""
    z = torch.zeros_like(out['eb_net_short'])
    g = lambda k: out.get(k, z)
    net_rad = g('eb_net_short') + g('eb_net_long')
    latent = g('eb_latent') + g('eb_latent_sub')
    sensible = g('eb_sensible') + g('eb_advected_sensible')
    grnd = g('eb_grnd_flux') + g('eb_delta_h') + g('eb_fusion')
    snow_fluxes = (g('eb_advection') - g('eb_deltacc') + g('eb_snow_flux')
                   + g('eb_rfrz_energy'))
    return net_rad - latent - sensible - grnd + snow_fluxes


def find_0_degree_fronts(T, Z):
    """find_0_degree_fronts(), soil_conduction.c 5.1.0:722-766, on whole
    tensors. `T` and `Z` are [..., node] (node temperature in C, node depth
    in m, `Z` broadcastable to `T`). Returns (fdepth, tdepth), each
    [..., MAX_FRONTS = 3] in m, MISSING where there is no front: the nodes
    are scanned from the second to last upwards, a thawing front (T[n] > 0,
    T[n+1] <= 0) or a freezing front (T[n] < 0, T[n+1] >= 0) is placed by
    linear_interp(0, T[n], T[n+1], Z[n], Z[n+1]) = (0 - T[n]) / (T[n+1] -
    T[n]) * (Z[n+1] - Z[n]) + Z[n], and at most three of each are kept."""
    Z = Z.expand_as(T) if Z.dim() == T.dim() else Z
    n = T.shape[-1]
    shape = T.shape[:-1]
    miss = torch.full(shape + (MAX_FRONTS,), MISSING, dtype=T.dtype,
                      device=T.device)
    fd, td = miss.clone(), miss.clone()
    nthaw = torch.zeros(shape, dtype=torch.long, device=T.device)
    nfrost = torch.zeros_like(nthaw)
    slots = torch.arange(MAX_FRONTS, device=T.device)
    for k in range(n - 2, -1, -1):
        t0, t1 = T[..., k], T[..., k + 1]
        zero = torch.zeros_like(t0)
        thaw = (t0 > 0.0) & (t1 <= 0.0) & (nthaw < MAX_FRONTS)
        frost = ~((t0 > 0.0) & (t1 <= 0.0)) & (t0 < 0.0) & (t1 >= 0.0) \
            & (nfrost < MAX_FRONTS)
        den = torch.where(thaw | frost, t1 - t0, torch.ones_like(t0))
        z = ((zero - t0) / den) * (Z[..., k + 1] - Z[..., k]) + Z[..., k]
        put_t = thaw.unsqueeze(-1) & (slots == nthaw.unsqueeze(-1))
        put_f = frost.unsqueeze(-1) & (slots == nfrost.unsqueeze(-1))
        td = torch.where(put_t, z.unsqueeze(-1), td)
        fd = torch.where(put_f, z.unsqueeze(-1), fd)
        nthaw = nthaw + thaw.long()
        nfrost = nfrost + frost.long()
    return fd, td


def front_rows(fd, area):
    """put_data.c:788-798: OUT_FDEPTH (or OUT_TDEPTH) of a cell from its
    tiles' fronts `fd` [cell, tile, 3] in m: `fd * AreaFactor * CM_PER_M`
    where the front exists, per tile (the caller sums over tiles). `area` is [cell, tile]."""
    w = (fd * area.unsqueeze(-1)) * 100.0
    return torch.where(fd != MISSING, w, torch.zeros_like(w))


#: The OUT_*_BAND variables that are a column's own row: put_data.c:858-924
#: forms each as `value * Cv * lakefactor` per tile, which is the cell-level
#: variable's expression without AreaFract, and a pseudo-cell of SNOW_BAND
#: TRUE carries exactly that sum (the AreaFract weight is applied by the
#: contraction). {band key: the row it is}.
BAND_ALIASES = {
    'band_swe': 'swe', 'band_snow_depth': 'snow_depth',
    'band_snow_canopy': 'snow_canopy', 'band_snow_melt': 'snow_melt',
    'band_snow_cover': 'snow_cover', 'band_deltacc': 'eb_deltacc',
    'band_snow_flux': 'eb_snow_flux', 'band_melt_energy': 'eb_melt_energy',
    'band_swnet': 'eb_net_short', 'band_lwnet': 'eb_net_long',
    'band_albedo': 'albedo', 'band_grnd_flux': 'eb_grnd_flux',
}


#: vic_physical_constants.h
CONST_MWC = 12.01
MOLE_PER_KMOLE = 1000
CONST_CDAY = 86400
SEC_PER_DAY = 86400

#: OUT_* carbon variable -> (the out key, how put_data.c 5.1.0:719-728 scales
#: it): GPP, RAUT and NPP are `* CONST_MWC / MOLE_PER_KMOLE * CONST_CDAY`;
#: the pools, litterfall and the leaf PAR are as they are.
CARBON_ROWS = {
    'gpp': ('carbon_GPP', 'flux'), 'raut': ('carbon_Raut', 'flux'),
    'npp': ('carbon_NPP', 'flux'), 'apar': ('carbon_aPAR', None),
    'litterfall': ('carbon_Litterfall', None),
    'clitter': ('carbon_CLitter', None), 'cinter': ('carbon_CInter', None),
    'cslow': ('carbon_CSlow', None),
}


def carbon_rows(out, dt_seconds):
    """OUT_GPP, RAUT, NPP, APAR, LITTERFALL, CLITTER, CINTER, CSLOW, RHET and
    NEE from the `carbon_*` rows of a CARBON run. OUT_RHET is RhTot scaled
    to g m-2 d-1 by dt / SEC_PER_DAY (put_data.c:507) and OUT_NEE = NPP - RHET
    (:508), both formed after the aggregation sums, as VIC forms them."""
    for name, (key, kind) in CARBON_ROWS.items():
        if key in out:
            x = out[key]
            out[name] = (x * CONST_MWC / MOLE_PER_KMOLE * CONST_CDAY
                         if kind == 'flux' else x)
    if 'carbon_RhTot' in out:
        out['rhet'] = out['carbon_RhTot'] * (dt_seconds / SEC_PER_DAY)
        if 'npp' in out:
            out['nee'] = out['npp'] - out['rhet']


#: rows a LAKES TRUE run does not form (see the end of derive)
LAKE_UNSUPPORTED = ('aero_cond1', 'aero_cond2', 'aero_resist1', 'aero_resist2',
                    'eb_tnode', 'soil_tnode_wl', 'vic_zwt', 'vic_zwt_lumped',
                    'band_rfz', 'band_surft', 'band_packt', 'band_adv', 'band_adv_sens',
                    'band_lat', 'band_lsub', 'band_sen', 'surf_frost_frac',
                    'fdepth', 'tdepth')


class Wanted(object):
    """Which OUT_* variables a run is asked for.

    Every row out-vars added costs a [step, cell] array on the device for the
    whole window, and the step loop's extra rows cost kernels, so none of it
    is formed unless a name asks for it: `Wanted(None)` wants nothing (the
    cost and memory of a run that never had these rows), `Wanted('all')`
    everything, `Wanted({'OUT_QAIR', ...})` those names (a global file's
    OUTVAR lines, a caller's list). `want(*names)` is true when any of the
    names is wanted.
    """

    def __init__(self, names=None):
        if isinstance(names, Wanted):
            names = names.names
        self.all = isinstance(names, str) and names == 'all'
        if isinstance(names, str) and not self.all:
            raise ValueError("out_vars must be None, 'all' or a set of OUT_* "
                             "names; got %r" % (names,))
        self.names = frozenset() if (names is None or self.all) \
            else frozenset(names)

    def __call__(self, *names):
        return self.all or any(n in self.names for n in names)

    def __bool__(self):
        return self.all or bool(self.names)

    def __repr__(self):
        return 'Wanted(all)' if self.all else 'Wanted(%d names)' % len(self.names)


class NullList(list):
    """A row list nothing asked for: it takes appends and keeps none, so the
    step loop's code is unchanged and no memory is held."""

    def append(self, x):
        pass

    def extend(self, xs):
        pass


#: the OUT_* names that need each water-balance energy row (wbenergy)
WB_ENERGY_ROWS = {
    'net_short': ('OUT_SWNET', 'OUT_SWNET_BAND', 'OUT_R_NET'),
    'net_long': ('OUT_LWNET', 'OUT_LWNET_BAND', 'OUT_R_NET'),
    'net_rad': ('OUT_R_NET',), 'latent': ('OUT_LATENT',),
    'sensible': ('OUT_SENSIBLE',), 't_surf': ('OUT_SURF_TEMP',),
    'aero_resist': ('OUT_AERO_RESIST',), 'layer_temp': ('OUT_SOIL_TEMP',),
    'aero_cond': ('OUT_AERO_COND', 'OUT_AERO_RESIST'),
    'rad_temp': ('OUT_RAD_TEMP',), 'vegt': ('OUT_VEGT',),
    'snow_flux': ('OUT_SNOW_FLUX', 'OUT_SNOW_FLUX_BAND'),
    'aero_cond1': ('OUT_AERO_COND1', 'OUT_AERO_COND2', 'OUT_AERO_RESIST1',
                   'OUT_AERO_RESIST2'),
    'aero_cond2': ('OUT_AERO_COND1', 'OUT_AERO_COND2', 'OUT_AERO_RESIST1',
                   'OUT_AERO_RESIST2'),
    'tnode': ('OUT_SOIL_TNODE', 'OUT_SOIL_TNODE_WL'),
    'pet': ('OUT_PET',),
}
#: every name a water-balance run needs the energy block for
WB_ENERGY_OUT = tuple(sorted(
    set(n for v in WB_ENERGY_ROWS.values() for n in v)
    | {'OUT_LATENT_BAND', 'OUT_SENSIBLE_BAND'}))


#: the OUT_* name each BAND_ALIASES row is
BAND_OUT = {
    'band_swe': 'OUT_SWE_BAND', 'band_snow_depth': 'OUT_SNOW_DEPTH_BAND',
    'band_snow_canopy': 'OUT_SNOW_CANOPY_BAND',
    'band_snow_melt': 'OUT_SNOW_MELT_BAND',
    'band_snow_cover': 'OUT_SNOW_COVER_BAND',
    'band_deltacc': 'OUT_DELTACC_BAND', 'band_snow_flux': 'OUT_SNOW_FLUX_BAND',
    'band_melt_energy': 'OUT_MELT_ENERGY_BAND',
    'band_swnet': 'OUT_SWNET_BAND', 'band_lwnet': 'OUT_LWNET_BAND',
    'band_albedo': 'OUT_ALBEDO_BAND', 'band_grnd_flux': 'OUT_GRND_FLUX_BAND',
}
FBFLAGS = ('OUT_SURFT_FBFLAG', 'OUT_SNOWT_FBFLAG', 'OUT_TFOL_FBFLAG',
           'OUT_TCAN_FBFLAG', 'OUT_SOILT_FBFLAG')
CARBON_OUT = ('OUT_GPP', 'OUT_RAUT', 'OUT_NPP', 'OUT_APAR', 'OUT_LITTERFALL',
              'OUT_CLITTER', 'OUT_CINTER', 'OUT_CSLOW', 'OUT_RHET', 'OUT_NEE')
AERO_OUT = ('OUT_AERO_COND1', 'OUT_AERO_COND2', 'OUT_AERO_RESIST1',
            'OUT_AERO_RESIST2')


def derive(out, atm, depth=None, carbon=False, lakes=False, blowing=False,
           energy_balance=False, vic_energy_error=False, cv_overstory=None,
           dt_seconds=None, fallback_free=False, nnode=3,
           spatial_snow=False, any_overstory=False, want=None):
    """Add to `out` the VIC variables formed from its rows, those `want`
    (a Wanted; None wants all) names. `atm` is the forcing dict of the call
    ([step, cell] tensors, Pa for pressure and vapour pressure), `depth` the
    layer thickness [cell, layer] in m (or None). Returns `out`."""
    want = Wanted('all') if want is None else want
    ref = out['swe']
    zero = torch.zeros_like(ref) if want else None
    if want('OUT_QAIR') and 'eact_air' in atm and 'press' in atm:
        out['qair'] = (_snow.CONST_EPS * atm['eact_air']) / atm['press']
    if ('soil_liq' in out and 'soil_ice' in out
            and want('OUT_SMLIQFRAC', 'OUT_SMFROZFRAC', 'OUT_SOIL_LIQ_FRAC',
                     'OUT_SOIL_ICE_FRAC')):
        liq, ice = out['soil_liq'], out['soil_ice']
        # put_data.c:703-717: SOIL_MOIST = LIQ + ICE; the fractions are
        # formed from the layer totals
        if want('OUT_SMLIQFRAC', 'OUT_SMFROZFRAC'):
            moist = liq + ice
            out['smliqfrac'] = liq / moist
            out['smfrozfrac'] = 1.0 - out['smliqfrac']
        if depth is not None:
            d_mm = (depth * MM_PER_M).unsqueeze(0)
            if want('OUT_SOIL_LIQ_FRAC'):
                out['soil_liq_frac'] = liq / d_mm
            if want('OUT_SOIL_ICE_FRAC'):
                out['soil_ice_frac'] = ice / d_mm
    # OUT_DISCHARGE is set only by the RVIC routing extension
    # (extensions/rout_rvic/src/rout_run.c:72), which a plain image-driver
    # run does not have: VIC writes the zero zero_output_list leaves
    if want('OUT_DISCHARGE'):
        out['discharge_rvic'] = zero
    if fallback_free and want(*FBFLAGS):
        # OUT_*_FBFLAG (put_data.c:870-880) are 1 on a step whose temperature
        # solve found no root and kept the previous temperature
        # (TFALLBACK TRUE). A run that took no fallback reports them as the
        # zeros VIC does; a run that did is left without them rather than
        # with a series that hides it (see torchvic.core.tfallback)
        for k in ('surft_fbflag', 'snowt_fbflag', 'tfol_fbflag',
                  'tcan_fbflag'):
            out[k] = zero
        out['soilt_fbflag'] = zero.unsqueeze(-1).expand(
            zero.shape + (nnode,)).clone()
    if not lakes:
        if want('OUT_SURFSTOR'):
            out['surfstor'] = zero
        if want('OUT_DELSURFSTOR'):
            out['delsurfstor'] = zero
    if want(*CARBON_OUT):
        if carbon and dt_seconds is not None:
            carbon_rows(out, dt_seconds)
        if not carbon:
            # CARBON FALSE: the carbon outputs are the zeros zero_output_list
            # leaves (put_data.c:96-100, :510-514 only touch them with the
            # option on), OUT_NEE included
            for k in CARBON_ROWS:
                out[k] = zero
            out['rhet'] = zero
            out['nee'] = zero
    if want('OUT_CATM', 'OUT_FDIR', 'OUT_PAR'):
        if carbon and 'Catm' in atm and 'fdir' in atm and 'par' in atm:
            out['catm'] = atm['Catm'] / PPM_TO_MIXRATIO
            out['fdir'] = atm['fdir']
            out['par'] = atm['par']
        elif not carbon:
            full = torch.full_like(ref, MISSING)
            out['catm'] = full
            out['fdir'] = full
            out['par'] = full
    if want('OUT_SUB_BLOWING', 'OUT_SUB_SURFACE'):
        if blowing and 'sub_blowing' in out and 'sub_snow' in out:
            # surface_flux = vapor_flux - blowing_flux inside VIC, and only
            # vapor_flux is flipped for output (snow_melt.c:479): the surface
            # share of the sublimation is -(OUT_SUB_SNOW) - OUT_SUB_BLOWING
            out['sub_surface'] = -out['sub_snow'] - out['sub_blowing']
        if not blowing:
            out['sub_blowing'] = zero
            if 'sub_snow' in out:
                # snow.surface_flux keeps the sign the pack's own balance
                # gives it (negative: mass leaving), while snow_melt.c:479 and
                # calc_surf_energy_bal.c:718 flip vapor_flux on the way out;
                # with no blowing the two are the same mass flux
                # (latent_heat_from_snow.c:41), so OUT_SUB_SURFACE
                # (put_data.c:606) is the negative of OUT_SUB_SNOW, to the bit
                out['sub_surface'] = -out['sub_snow']
    if 'eb_aero_cond1' in out and want(*AERO_OUT):
        # put_data.c:471-492: the conductances are summed over tiles, the
        # overstory one normalised by the overstory cover, and each
        # resistance is the inverse of its conductance above DBL_EPSILON,
        # HUGE_RESIST (param.HUGE_RESIST = 1e20) at or below it
        huge = torch.full_like(ref, HUGE_RESIST)
        c1 = out['eb_aero_cond1']
        c2 = out['eb_aero_cond2']
        if cv_overstory is not None:
            cvo = cv_overstory.unsqueeze(0)
            c2 = torch.where(cvo > 0.0, c2 / torch.where(
                cvo > 0.0, cvo, torch.ones_like(cvo)), c2)
        out['aero_cond1'], out['aero_cond2'] = c1, c2
        out['aero_resist1'] = torch.where(
            c1 > DBL_EPSILON, 1.0 / torch.where(c1 > DBL_EPSILON, c1,
                                                torch.ones_like(c1)), huge)
        out['aero_resist2'] = torch.where(
            c2 > DBL_EPSILON, 1.0 / torch.where(c2 > DBL_EPSILON, c2,
                                                torch.ones_like(c2)), huge)
    # OUT_ZWT / OUT_ZWT_LUMPED: the rows as they are (a lake or wetland tile
    # takes another branch; see LAKE_UNSUPPORTED)
    for k, nm in (('zwt', 'OUT_ZWT'), ('zwt_lumped', 'OUT_ZWT_LUMPED')):
        if k in out and want(nm):
            out['vic_' + k] = out[k]
    for bk, src in BAND_ALIASES.items():
        if src in out and want(BAND_OUT[bk]):
            out[bk] = out[src]
    # OUT_ADV_SENS_BAND (put_data.c:900-902): -advected_sensible * Cv, which is
    # the cell row without the `!overstory` gate: equal where no tile has an
    # overstory, left out where one does (SPATIAL_SNOW only; 0 otherwise)
    if want('OUT_ADV_SENS_BAND'):
        if not spatial_snow:
            out['band_adv_sens'] = zero
        elif not any_overstory and 'eb_advected_sensible' in out:
            out['band_adv_sens'] = out['eb_advected_sensible']
    # put_data.c never sets OUT_IN_LONG_BAND (the zero it starts at is VIC's)
    if want('OUT_IN_LONG_BAND'):
        out['band_in_long'] = zero
    if 'eb_tnode' in out and not lakes and want('OUT_SOIL_TNODE_WL'):
        out['soil_tnode_wl'] = torch.zeros_like(out['eb_tnode'])
    if energy_balance:
        if want('OUT_ADV_SENS', 'OUT_ADV_SENS_BAND') or vic_energy_error:
            out.setdefault('eb_advected_sensible', zero if zero is not None
                           else torch.zeros_like(ref))
        if vic_energy_error and all(k in out for k in ENERGY_ERROR_ROWS):
            out['eb_energy_error'] = energy_error(out)
    elif want('OUT_ENERGY_ERROR'):
        # put_data.c:554: FULL_ENERGY FALSE
        out['energy_error_missing'] = torch.full_like(ref, MISSING)
    if lakes:
        # the lake and wetland tiles of put_data.c (:213-400) take their own
        # branches for these; they are not formed for them, so a LAKES TRUE
        # run leaves them out rather than write a land-only value
        for k in LAKE_UNSUPPORTED:
            out.pop(k, None)
    return out
