"""The verification matrix: TorchVIC against pristine VIC 5.0.1 and 5.1.0.

Every row is one configuration that VIC itself accepts. A row names a case
(domain, parameters, forcing, period), a core (the VIC release TorchVIC
runs as, and the pristine vic_image.exe it is compared with), the model
step and the global-file options that differ from the case's base. The
global file is written from scratch for both models (`global_text`), with
every output variable at every model step in double precision
(AGGFREQ NSTEPS 1, OUT_TYPE_DOUBLE), so the comparison is step by step.

`expect` records what TorchVIC's README/option inventory claims for the
row: 'supported' (it must agree), or 'gap' (a documented or suspected
departure, run so that its size is on the scoreboard).

The options TorchVIC refuses, or ought to, are probed separately
(PROBES, verification/refusals.py): each is a one-option change to a base
case, run by both models for a couple of steps, and the outcome recorded.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get('TORCHVIC_VERIFY_WORK',
                      os.path.join(REPO, 'verification', 'work'))
WINDOWS = os.path.join(WORK, 'windows')
SAMPLE = os.path.join(REPO, 'data', 'VIC_sample_data', 'image')
STEHE = os.path.join(REPO, 'data', 'samples', 'image', 'Stehekin')
VIC = {'5.1.0': os.path.join(REPO, 'reference', 'VIC-5.1.0', 'vic', 'drivers',
                             'image', 'vic_image.exe'),
       '5.0.1': os.path.join(REPO, 'reference', 'VIC-VIC.5.0.1', 'vic',
                             'drivers', 'image', 'vic_image.exe')}

_ISC_FORCE = (('AIR_TEMP', 'air_temp'), ('PREC', 'prec'),
              ('PRESSURE', 'pressure'), ('SWDOWN', 'swdown'),
              ('LWDOWN', 'lwdown'), ('VP', 'vp'), ('WIND', 'wind'))
_LIVNEH_FORCE = (('AIR_TEMP', 'temp'), ('PREC', 'prec'),
                 ('PRESSURE', 'air_pressure'), ('SWDOWN', 'shortwave'),
                 ('LWDOWN', 'longwave'), ('VP', 'vapor_pressure'),
                 ('WIND', 'wind'))


def _isc(name, blurb):
    d = os.path.join(WINDOWS, name)
    return dict(domain=os.path.join(d, 'domain.nc'),
                params=os.path.join(d, 'params.nc'),
                forcing=os.path.join(d, 'isc_3h.'), force_type=_ISC_FORCE,
                forcing_spd=8, start=(2004, 1, 1), end=(2004, 12, 31),
                wind_h=10.0, blurb=blurb,
                base={'LAI_SRC': 'FROM_VEGLIB', 'FCAN_SRC': 'FROM_VEGLIB',
                      'ALB_SRC': 'FROM_VEGLIB', 'SNOW_BAND': 'FALSE'})


#: The cases. `forcing_spd` is the forcing's records per day, which VIC
#: requires to equal SNOW_STEPS_PER_DAY (vic_force.c 5.1.0:565-572).
CASES = {
    'igp': _isc('igp', 'ISC Indo-Gangetic plain 26.0N 82.0E, cropland, '
                'monsoon (36 cells, 2004)'),
    'thar': _isc('thar', 'ISC Thar desert 27.0N 71.0E, shrub/barren, '
                 '187 mm/yr (36 cells, 2004)'),
    'himal': _isc('himal', 'ISC Gangotri 30.9N 79.0E, 3450-5600 m, '
                  'seasonal snow, 4-5 bands (36 cells, 2004)'),
    'forest': _isc('forest', 'ISC Anamalai, Western Ghats 10.5N 76.9E, '
                   '~50% forest with overstory (36 cells, 2004)'),
    'tntip': _isc('tntip', 'ISC Tamil Nadu/Kerala tip 8.3N 77.4E, mixed '
                  'cover, 1-5 bands (35 cells, 2004)'),
    'wg': dict(
        domain=os.path.join(SAMPLE, 'WalnutGulch', 'parameters',
                            'domain.WalnutGulch.0.0625_deg.nc'),
        params=os.path.join(SAMPLE, 'WalnutGulch', 'parameters',
                            'params.WalnutGulch.L2015.0.0625_deg.nc'),
        forcing=os.path.join(SAMPLE, 'WalnutGulch', 'forcings',
                             'livneh_NAmerExt_15Oct2014.WalnutGulch.'),
        force_type=_LIVNEH_FORCE, forcing_spd=8, start=(2008, 1, 1),
        end=(2008, 12, 31), wind_h=10.0,
        blurb='UW-Hydro Walnut Gulch, semi-arid (15 cells, 2008, 3-hourly)',
        base={'LAI_SRC': 'FROM_VEGPARAM', 'FCAN_SRC': 'FROM_DEFAULT',
              'ALB_SRC': 'FROM_VEGPARAM', 'SNOW_BAND': 'FALSE'}),
    'stehe': dict(
        domain=os.path.join(STEHE, 'parameters', 'domain.stehekin.20151028.nc'),
        params=os.path.join(STEHE, 'parameters',
                            'Stehekin_test_params_20160327.nc'),
        forcing=os.path.join(STEHE, 'forcings',
                             'Stehekin_image_test.forcings_10days.'),
        force_type=(('AIR_TEMP', 'tas'), ('PREC', 'prcp'),
                    ('PRESSURE', 'pres'), ('SWDOWN', 'dswrf'),
                    ('LWDOWN', 'dlwrf'), ('VP', 'vp'), ('WIND', 'wind')),
        forcing_spd=24, start=(1949, 1, 1), end=(1949, 1, 10), wind_h=10.0,
        blurb='VIC sample Stehekin, snow (16 cells, 10 days Jan 1949, '
              'hourly, 3-5 bands)',
        base={'LAI_SRC': 'FROM_VEGPARAM', 'SNOW_BAND': 'FALSE'}),
    'findley': dict(
        domain=os.path.join(SAMPLE, 'FindleyLake', 'parameters',
                            'domain.FindleyLake.nc'),
        params=os.path.join(SAMPLE, 'FindleyLake', 'parameters',
                            'params.FindleyLake.nc'),
        forcing=os.path.join(SAMPLE, 'FindleyLake', 'forcings',
                             'livneh_NAmerExt_15Oct2014.FindleyLake.'),
        force_type=_LIVNEH_FORCE, forcing_spd=24, start=(1970, 1, 1),
        end=(1970, 12, 31), wind_h=10.0,
        blurb='UW-Hydro Findley Lake, Cascades, wet snow (1 cell, 1970, '
              'hourly; LAKES FALSE)',
        base={'LAI_SRC': 'FROM_VEGPARAM', 'FCAN_SRC': 'FROM_VEGPARAM',
              'ALB_SRC': 'FROM_VEGPARAM', 'SNOW_BAND': 'FALSE'}),
}

#: Walnut Gulch with the carbon library and forcings (make_variants.py).
_WGC = os.path.join(WORK, 'variants', 'wg_carbon')
CASES['wgc'] = dict(
    CASES['wg'], params=os.path.join(_WGC, 'params.nc'),
    forcing=os.path.join(_WGC, 'forc.'),
    force_type=_LIVNEH_FORCE + (('CATM', 'CATM'), ('PAR', 'PAR'),
                                ('FDIR', 'FDIR')),
    blurb='Walnut Gulch + photosynthesis library and CATM/PAR/FDIR forcing '
          '(15 cells, 2008, 3-hourly)')

#: Physics modes (global-file lines, in order: VIC's FROZEN_SOIL resets
#: QUICK_FLUX, get_global_param.c 5.1.0:101-116, so QUICK_FLUX follows).
WB = (('FULL_ENERGY', 'FALSE'), ('FROZEN_SOIL', 'FALSE'))
EB = (('FULL_ENERGY', 'TRUE'), ('FROZEN_SOIL', 'FALSE'))
# Frozen soil on VIC's QUICK_FLUX path, the one TorchVIC runs. VIC's
# defaults IMPLICIT and EXP_TRANS TRUE are refused with QUICK_FLUX
# (get_global_param.c 5.1.0:933-936), so both are set FALSE.
FS = (('FULL_ENERGY', 'TRUE'), ('FROZEN_SOIL', 'TRUE'), ('QUICK_FLUX', 'TRUE'),
      ('IMPLICIT', 'FALSE'), ('EXP_TRANS', 'FALSE'))
FSWB = (('FULL_ENERGY', 'FALSE'), ('FROZEN_SOIL', 'TRUE'),
        ('QUICK_FLUX', 'TRUE'), ('IMPLICIT', 'FALSE'), ('EXP_TRANS', 'FALSE'))
MODES = {'WB': WB, 'EB': EB, 'FS': FS, 'FSWB': FSWB}


def row(case, mode, core='5.1.0', daily=False, expect='supported', note='',
        tag='', params=None, extra=(), **opts):
    """One matrix row; `opts` are extra global-file options."""
    parts = [case, mode, core.replace('.', ''), 'D' if daily else
             '%dh' % (24 // CASES[case]['forcing_spd'])]
    for k, v in sorted(opts.items()):
        parts.append('%s=%s' % (k, v))
    if tag:
        parts.append(tag)
    return dict(id='_'.join(parts).replace('/', ''), case=case, mode=mode,
                core=core, daily=daily, opts=dict(opts), expect=expect,
                note=note, params=params, extra=list(extra))


ISC = ('igp', 'thar', 'himal', 'forest', 'tntip')
FS_STEHE = os.path.join(STEHE, 'parameters',
                        'Stehekin_test_params_20160327.FROZEN_SOIL.nc')

MATRIX = []
# 1. The two physics modes on every case, under both cores.
for c in ISC + ('wg', 'stehe', 'findley'):
    for core in ('5.1.0', '5.0.1'):
        MATRIX.append(row(c, 'WB', core))
        MATRIX.append(row(c, 'EB', core))
# 2. Snow bands (cases with more than one band per cell).
for c in ('himal', 'forest', 'tntip', 'stehe'):
    MATRIX.append(row(c, 'WB', SNOW_BAND='TRUE',
                      note='one column per band with area '
                           '(core/snowbands.py)'))
MATRIX.append(row('stehe', 'EB', SNOW_BAND='TRUE',
                  note='as above, energy balance'))
MATRIX.append(row('himal', 'EB', SNOW_BAND='TRUE',
                  note='as above, energy balance'))
# 3. Frozen soil (QUICK_FLUX path), energy and water-balance modes.
MATRIX.append(row('himal', 'FS'))
MATRIX.append(row('himal', 'FS', core='5.0.1'))
MATRIX.append(row('himal', 'FSWB'))
MATRIX.append(row('stehe', 'FS', params=FS_STEHE, tag='fsparams'))
MATRIX.append(row('stehe', 'FS', core='5.0.1', params=FS_STEHE,
                  tag='fsparams'))
# 4. Blowing snow (default sub-options, then each sub-option off); the
#    per-tile blowing fields come from verification/make_variants.py.
_BH = os.path.join(WORK, 'variants', 'himal_blowing', 'params.nc')
_BS = os.path.join(WORK, 'variants', 'stehe_blowing', 'params.nc')
MATRIX.append(row('himal', 'EB', BLOWING='TRUE', params=_BH))
MATRIX.append(row('stehe', 'EB', BLOWING='TRUE', params=_BS))
MATRIX.append(row('himal', 'WB', BLOWING='TRUE', params=_BH))
for sub in ('BLOWING_VAR_THRESHOLD', 'BLOWING_CALC_PROB', 'BLOWING_FETCH',
            'BLOWING_SPATIAL_WIND'):
    MATRIX.append(row('himal', 'EB', BLOWING='TRUE', params=_BH,
                      **{sub: 'FALSE'}))
MATRIX.append(row('himal', 'EB', BLOWING='TRUE', BLOWING_SIMPLE='TRUE',
                  params=_BH))
# 5. SHARE_LAYER_MOIST FALSE (VIC default TRUE).
for c in ('thar', 'igp', 'wg'):
    MATRIX.append(row(c, 'WB', SHARE_LAYER_MOIST='FALSE'))
MATRIX.append(row('forest', 'EB', SHARE_LAYER_MOIST='FALSE'))
# 6. Baseflow formulation.
MATRIX.append(row('igp', 'WB', BASEFLOW='NIJSSEN2001',
                  note='d1..d4 reach the column as ARNO Ds/Dsmax/Ws/c'))
MATRIX.append(row('wg', 'WB', BASEFLOW='NIJSSEN2001',
                  note='d1..d4 reach the column as ARNO Ds/Dsmax/Ws/c'))
# 7. Daily model step (SNOW_STEPS_PER_DAY = the forcing's records/day).
for c in ('igp', 'thar', 'himal', 'wg', 'findley'):
    MATRIX.append(row(c, 'WB', daily=True))
for c in ('igp', 'himal', 'findley'):
    MATRIX.append(row(c, 'EB', daily=True))
MATRIX.append(row('igp', 'WB', core='5.0.1', daily=True))
# The daily step as a VIC user writes it: MODEL_STEPS_PER_DAY 1 in the
# global file and nothing else (no --daily on the command line).
MATRIX.append(row('wg', 'WB', daily=True, tag='noflag', expect='gap',
                  note='torchvic run without --daily'))
# 8. Vegetation sources.
MATRIX.append(row('forest', 'WB', FCAN_SRC='FROM_DEFAULT'))
MATRIX.append(row('forest', 'EB', FCAN_SRC='FROM_DEFAULT'))
MATRIX.append(row('findley', 'EB', FCAN_SRC='FROM_DEFAULT'))
MATRIX.append(row('wg', 'EB', FCAN_SRC='FROM_VEGPARAM', expect='gap',
                  note='params has no fcanopy: VIC refuses or defaults'))
MATRIX.append(row('igp', 'EB', core='5.0.1', FCAN_SRC='FROM_DEFAULT'))
MATRIX.append(row('forest', 'EB', ALB_SRC='FROM_VEGPARAM',
                  LAI_SRC='FROM_VEGPARAM'))
# 9. Other options VIC exposes that change physics and TorchVIC does not
#    refuse (the inventory calls them unsupported): size of the departure.
MATRIX.append(row('himal', 'EB', SNOW_DENSITY='DENS_SNTHRM', expect='gap',
                  note='inventory: only DENS_BRAS'))
MATRIX.append(row('forest', 'EB', AERO_RESIST_CANSNOW='AR_406',
                  expect='gap', note='inventory: one formulation'))
MATRIX.append(row('himal', 'EB', QUICK_FLUX='FALSE',
                  note='finite-difference thermal profile, 3 nodes, '
                       'explicit (core/soiltemp.py)'))
MATRIX.append(row('igp', 'WB', RUNOFF_STEPS_PER_DAY='24'))
MATRIX.append(row('forest', 'EB', AERO_RESIST_CANSNOW='AR_410', expect='gap',
                  note='inventory: one formulation'))
# AERO_RESIST_CANSNOW where it acts (overstory under snow): VIC's output
# moves with it on Stehekin (verification/noeffect.py), not on the forest
# case; TorchVIC passes the value to basin.simulate.
MATRIX.append(row('stehe', 'EB', AERO_RESIST_CANSNOW='AR_406'))
MATRIX.append(row('stehe', 'EB', AERO_RESIST_CANSNOW='AR_410'))
MATRIX.append(row('stehe', 'EB', AERO_RESIST_CANSNOW='AR_406_LS'))
# 10. Options needing input variants (verification/make_variants.py).
VARIANTS = os.path.join(WORK, 'variants')
MATRIX.append(row('himal', 'EB', SPATIAL_SNOW='TRUE',
                  params=os.path.join(VARIANTS, 'himal_spatial_snow',
                                      'params.nc'),
                  note='image SPATIAL_SNOW: max_snow_distrib_slope 0.5 m'))
MATRIX.append(row('himal', 'WB', SPATIAL_SNOW='TRUE',
                  params=os.path.join(VARIANTS, 'himal_spatial_snow',
                                      'params.nc'),
                  note='image SPATIAL_SNOW, water balance: the thin pack '
                       'folded into the surface with its cover '
                       '(func_surf_energy_bal.c:617-628, '
                       'calc_surf_energy_bal.c:680-689)'))
# The same on a daily model step (MODEL_STEPS_PER_DAY 1, SNOW_STEPS_PER_DAY 8):
# the snow sub-steps of surface_fluxes.c:258-269 with calc_snow_coverage.
for _m in ('EB', 'WB'):
    MATRIX.append(row('himal', _m, daily=True, SPATIAL_SNOW='TRUE',
                      params=os.path.join(VARIANTS, 'himal_spatial_snow',
                                          'params.nc'),
                      note='SPATIAL_SNOW on the daily step, 8 snow sub-steps'))
# FORCING2 vegetation history (vic_force.c 5.1.0:264-359): LAI, fcanopy and
# albedo per forcing record from igp_veghist (a perturbed copy of the
# parameter file's monthly values, make_variants.veghist). The FORCE_TYPE
# line after FORCING2 is what makes VIC open the file
# (forcing_utils.c:108-148). A daily model step averages the day's 8 records.
_VH = ['FORCING2 %s' % os.path.join(VARIANTS, 'igp_veghist', 'veghist.'),
       'FORCE_TYPE SKIP veg']
for _mode, _daily, _src in (
        ('EB', False, ('LAI_SRC', 'FCAN_SRC', 'ALB_SRC')),
        ('EB', True, ('LAI_SRC', 'FCAN_SRC', 'ALB_SRC')),
        ('WB', False, ('LAI_SRC', 'FCAN_SRC'))):
    MATRIX.append(row('igp', _mode, daily=_daily, tag='veghist', extra=_VH,
                      note='%s from FORCING2 (igp_veghist, perturbed)'
                           % '/'.join(k.split('_')[0] for k in _src),
                      **{k: 'FROM_VEGHIST' for k in _src}))
# The same with LAI 0 on two days a month: the leafless rule
# (vic_force.c:419-439). VIC drops the canopy water of those steps
# (OUT_WATER_ERROR 0.18 mm in VIC and TorchVIC alike), so the closure grade
# is VIC's own warn.
MATRIX.append(row('igp', 'EB', tag='veghist0', LAI_SRC='FROM_VEGHIST',
                  FCAN_SRC='FROM_VEGHIST', ALB_SRC='FROM_VEGHIST',
                  extra=[x.replace('igp_veghist', 'igp_veghist0')
                         for x in _VH],
                  note='LAI 0 two days a month: the leafless rule; VIC\'s '
                       'own water error 0.18 mm'))
# 11. Ground flux formulation (Stehekin's 10 January days move with it on
#     46 variables, verification/noeffect.py): the VIC 4.0.6 single-term
#     flux beside the default two-term one (func_surf_energy_bal.c:385).
MATRIX.append(row('stehe', 'EB', GRND_FLUX_TYPE='GF_406',
                  note='kappa1/D1 (T1 - Tsurf) alone, no deep term, no /2 '
                       '(energy.grnd_flux_type)'))
# 12. Sub-grid frost distribution: SPATIAL_FROST TRUE n spreads each
#     layer's temperature over Nfrost areas of width frost_slope
#     (get_global_param.c 5.1.0:221-229, :833-843; vic_init.c:703-710),
#     the frost_slope field from verification/make_variants.py. The
#     two-token value goes in `extra` (as FORCING2 above) so the row id
#     stays a clean tag.
MATRIX.append(row('himal', 'FSWB', params=os.path.join(VARIANTS, 'himal_frost',
                                                       'params.nc'),
                  tag='frost3', extra=['SPATIAL_FROST TRUE 3'],
                  note='Nfrost 3 over frost_slope 4 C'))
MATRIX.append(row('himal', 'FS', params=os.path.join(VARIANTS, 'himal_frost',
                                                     'params.nc'),
                  tag='frost3', extra=['SPATIAL_FROST TRUE 3'],
                  note='Nfrost 3 over frost_slope 4 C'))
MATRIX.append(row('stehe', 'FS', params=os.path.join(VARIANTS, 'stehe_frost',
                                                     'params.nc'),
                  tag='frost3', extra=['SPATIAL_FROST TRUE 3'],
                  note='Nfrost 3 over frost_slope 4 C'))
# 14. QUICK_FLUX FALSE: the finite-difference soil temperature profile
#     (func_surf_energy_bal.c:404-445) under each solver, grid, bottom
#     boundary and frozen-soil switch. The thermal lines go in `extra`,
#     after the row's own, because VIC's parser is order-dependent:
#     FROZEN_SOIL TRUE turns QUICK_FLUX off, FROZEN_SOIL FALSE turns
#     IMPLICIT and EXP_TRANS off (get_global_param.c 5.1.0:101-112).
for _tag, _lines in (
        ('n12-implicit', ['NODES 12', 'IMPLICIT TRUE']),
        ('n12-exptrans', ['NODES 12', 'EXP_TRANS TRUE']),
        ('n12-noflux', ['NODES 12', 'NOFLUX TRUE']),
        ('n12-quicksolve', ['NODES 12', 'QUICK_SOLVE TRUE']),
        ('n12-frozen', ['NODES 12', 'FROZEN_SOIL TRUE', 'IMPLICIT FALSE']),
        ('n12-implicit-exptrans', ['NODES 12', 'IMPLICIT TRUE',
                                   'EXP_TRANS TRUE']),
        ('n12-frozen-implicit', ['NODES 12', 'FROZEN_SOIL TRUE',
                                 'IMPLICIT TRUE']),
        ('n12-frozen-exptrans', ['NODES 12', 'FROZEN_SOIL TRUE',
                                 'IMPLICIT FALSE', 'EXP_TRANS TRUE'])):
    MATRIX.append(row('himal', 'EB', QUICK_FLUX='FALSE', tag=_tag,
                      extra=_lines,
                      note='finite-difference profile: ' + ', '.join(_lines)))
# 13. Closing the energy balance (Stehekin's 10 January days move with it
#     on 45 variables, verification/noeffect.py): the canopy-air
#     temperature is solved, not Tair (calc_atmos_energy_bal.c:73-102),
#     and the ground/canopy balances iterate to closure
#     (surface_fluxes.c:443-721, vic_init.c:1556).
MATRIX.append(row('stehe', 'EB', CLOSE_ENERGY='TRUE',
                  note='solved Tcanopy, iterated ground/canopy balances '
                       '(basin.close_energy)'))
# 14. The carbon cycle, and the canopy resistance from photosynthesis
#     (RC_MODE RC_PHOTO, canopy_evap.c:296-307): the carbon library and the
#     CATM/PAR/FDIR forcing from verification/make_variants.py.
for _mode in ('WB', 'EB'):
    MATRIX.append(row('wgc', _mode, CARBON='TRUE', VEGLIB_PHOTO='TRUE',
                      note='RC_JARVIS: the carbon step reads the resistance'))
    MATRIX.append(row('wgc', _mode, CARBON='TRUE', VEGLIB_PHOTO='TRUE',
                      RC_MODE='RC_PHOTO',
                      note='canopy resistance from photosynthetic demand'))

# 14. Inputs the image driver reads under an option (verification/
#     make_variants.py): the organic soil fractions (vic_init.c:572-605,
#     soil_conduction.c:17-99), the combined bulk density (:607-617), the
#     per-class maximum snow albedo (:186-196, surface_fluxes.c:524) and
#     the gauge undercatch correction (correct_precip.c, vic_run.c:106).
_SO = os.path.join(VARIANTS, 'stehe_organic', 'params.nc')
_WO = os.path.join(VARIANTS, 'wg_organic', 'params.nc')
MATRIX.append(row('stehe', 'EB', ORGANIC_FRACT='TRUE', params=_SO,
                  note='organic 0.30/0.20/0.10, bulk_density_org 300, '
                       'soil_density_org 1300'))
MATRIX.append(row('wg', 'EB', ORGANIC_FRACT='TRUE', params=_WO,
                  note='as above, Walnut Gulch'))
MATRIX.append(row('stehe', 'EB', ORGANIC_FRACT='TRUE',
                  BULK_DENSITY_COMB='TRUE', params=_SO,
                  note='bulk density read as bulk_density_comb'))
MATRIX.append(row('stehe', 'EB', BULK_DENSITY_COMB='TRUE', params=_SO,
                  note='no organic fraction; bulk_density_comb alone'))
MATRIX.append(row('stehe', 'EB', MAX_SNOW_ALBEDO='TRUE',
                  params=os.path.join(VARIANTS, 'stehe_maxalb', 'params.nc'),
                  note='max_snow_albedo 0.72..0.90 by class'))
MATRIX.append(row('stehe', 'EB', CORRPREC='TRUE',
                  note='gauge undercatch (rain and snow)'))
MATRIX.append(row('stehe', 'WB', CORRPREC='TRUE',
                  note='gauge undercatch, water balance'))
MATRIX.append(row('wg', 'WB', CORRPREC='TRUE',
                  note='gauge undercatch, rain only (semi-arid)'))

# 15. Lakes (vic_run.c 5.1.0:132-466, put_data.c 5.1.0:148-400): UW-Hydro's
#     Findley Lake, a 10-node basin profile on the cell's first tile (8.85
#     per cent), rpercent 1. VIC refuses LAKES without FULL_ENERGY
#     (get_global_param.c 5.1.0:980), and LAKE_PROFILE FALSE on this file
#     (vic_init.c:1463-1489: the 3-D basin_depth read with a 2-D start).
MATRIX.append(row('findley', 'EB', LAKES='TRUE', LAKE_PROFILE='TRUE',
                  note='lake and wetland tile (model/lakecell.py)'))


# ---------------------------------------------------------------------------
# Output variables
# ---------------------------------------------------------------------------
#: Every variable either model is asked for, at every step, as double.
#: The TorchVIC-computed set is torchvic.io.history.OUTVAR_SOURCES; the
#: extras are VIC terms of its own water/energy balance (put_data.c 5.1.0
#: :515-552) that TorchVIC may not write.
#: The variables torchvic.io.history gained from out-vars (OUTVAR_SOURCES_OUTVARS)
#: -- the default list leaves them out, as before.
def _new():
    from torchvic.io.history import OUTVAR_SOURCES_OUTVARS
    return tuple(k for k in OUTVAR_SOURCES_OUTVARS if k != 'OUT_ENERGY_ERROR')



def outvars(core='5.1.0', full=False):
    """The OUTVAR list of the global files. By default the variables
    TorchVIC wrote before out-vars (OUTVAR_SOURCES, plus a few it had no row
    for); with `full` the whole OUT_* inventory of VIC 5.1.0 except the two
    run-time timers, which is what the matrix rows ask both models for now (a
    variable TorchVIC leaves out is named by the history writer and listed
    in compare's not_written_by_torchvic). VIC 5.0.1's enum has no
    OUT_DISCHARGE, and it stops on the name."""
    from torchvic.io.history import OUTVAR_SOURCES, outvar_metadata
    if full:
        names = set(OUTVAR_SOURCES) | set(outvar_metadata())
        names -= {'OUT_TIME_VICRUN_WALL', 'OUT_TIME_VICRUN_CPU'}
        if core == '5.0.1':
            names.discard('OUT_DISCHARGE')
        return sorted(names)
    names = (set(OUTVAR_SOURCES) - set(_new())) | {
        'OUT_SNOW_CANOPY', 'OUT_ADV_SENS', 'OUT_SURFSTOR', 'OUT_SUB_BLOWING',
        'OUT_SUB_SURFACE', 'OUT_SOIL_TNODE', 'OUT_FDEPTH', 'OUT_TDEPTH'}
    return sorted(names)


def global_text(r, result_dir, log_dir, outvar_list, params=None,
                extra_lines=()):
    """The global parameter file of matrix row `r` (both models)."""
    c = CASES[r['case']]
    spd_f = c['forcing_spd']
    model = 1 if r['daily'] else spd_f
    opts = dict(c['base'])
    opts.update(r['opts'])
    runoff = opts.pop('RUNOFF_STEPS_PER_DAY', None) or spd_f
    L = ['MODEL_STEPS_PER_DAY %d' % model, 'SNOW_STEPS_PER_DAY %d' % spd_f,
         'RUNOFF_STEPS_PER_DAY %s' % runoff,
         'STARTYEAR %d' % c['start'][0], 'STARTMONTH %d' % c['start'][1],
         'STARTDAY %d' % c['start'][2], 'ENDYEAR %d' % c['end'][0],
         'ENDMONTH %d' % c['end'][1], 'ENDDAY %d' % c['end'][2],
         'CALENDAR PROLEPTIC_GREGORIAN']
    L += ['%s %s' % kv for kv in MODES[r['mode']]]
    L += ['NODES 3', 'DOMAIN %s' % c['domain']]
    L += ['DOMAIN_TYPE %s %s' % kv for kv in (
        ('LAT', 'lat'), ('LON', 'lon'), ('MASK', 'mask'), ('AREA', 'area'),
        ('FRAC', 'frac'), ('YDIM', 'lat'), ('XDIM', 'lon'))]
    L += ['FORCING1 %s' % c['forcing']]
    L += ['FORCE_TYPE %s %s' % kv for kv in c['force_type']]
    L += ['WIND_H %s' % c['wind_h'],
          'PARAMETERS %s' % (params or r.get('params') or c['params'])]
    L += ['%s %s' % kv for kv in sorted(opts.items())]
    L += list(r.get('extra') or ()) + list(extra_lines)
    L += ['RESULT_DIR %s/' % result_dir, 'LOG_DIR %s/' % log_dir,
          'OUTFILE fluxes', 'AGGFREQ NSTEPS 1', 'HISTFREQ NYEARS 1',
          'COMPRESS FALSE', 'OUT_FORMAT NETCDF4']
    L += ['OUTVAR %s * OUT_TYPE_DOUBLE * *' % v for v in outvar_list]
    return '\n'.join(L) + '\n'
