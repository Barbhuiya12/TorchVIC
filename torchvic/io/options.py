"""What TorchVIC does with every option of VIC 5's image-driver global file.

One table, `IMAGE_OPTIONS`, decides it, and three things are built from it:
the check `torchvic.case.load` runs on every global file (`check`), the
announcements it logs, and the options table in docs/OPTIONS.md
(`markdown`, kept in step by tests/test_verification_matrix.py).

Every (option, value) pair has one of three dispositions:

* **implemented** -- TorchVIC runs it as VIC 5 does. Some carry a condition
  on the input files (SNOW_BAND, FCAN_SRC, BLOWING, the model step), which
  `torchvic.case.load` checks against the parameter and forcing files and
  refuses by name when it does not hold.
* **no effect** -- VIC accepts it and it cannot change a result in any
  configuration TorchVIC runs; the note says why. It is announced, once,
  so a user sees that it was read. (The thermal switches -- IMPLICIT,
  EXP_TRANS, NODES, NOFLUX, QUICK_SOLVE -- are without effect on the
  QUICK_FLUX path and select the finite-difference profile's solver,
  grid and boundaries off it; `check` announces them only where they
  change nothing.)
* **refused** -- the run stops with a message naming the option and the
  value. Either VIC would run a physics TorchVIC does not have, or VIC
  itself stops on the file in every driver (COMPUTE_TREELINE without
  JULY_TAVG_SUPPLIED, a lake with a treeline). A key only the image
  driver stops on, and the classic driver or VIC 4.2.d runs, is run as
  they run it (ATMOS_STEPS_PER_DAY, COMPUTE_TREELINE, OUTPUT_FORCE,
  ARNO_PARAMS, NIJSSEN2001_BASEFLOW).

A value the table does not list is refused: VIC's own parser stops on an
unknown value of nearly every key (log_err "Unrecognized value of ..."),
and a value TorchVIC has not classified is one it cannot claim to run.

An option VIC's image driver does not dispatch on at all (a classic-driver
key, or a typo) is what VIC itself warns about and skips
(get_global_param.c 5.1.0:538); it is announced, not refused.

Line numbers are VIC 5.1.0's (reference/VIC-5.1.0/vic).

The base row's warn. A snow-option row on Gangotri (himal) is accepted at
a warn only when it is no worse than its base row, himal_EB_510_3h (warn
7.1e-04 TRANSP_VEG / 3.9e-03 ADVECTION / 9.9e-03 PET), and the base row's
warn has this mechanism:
LongUnderOut carried from the previous ground energy balance differs by
5 ulp, which moves the canopy's root Tfoliage by 2.9e-14 at step 777
(TV -3.1855039544898327, VIC -3.185503954489862 C); IntSnow then differs
by 1 ulp after snow_intercept.c:528, and at step 779 MassRelease's min()
(massrelease.c:56,67) leaves VIC 1.3e-19 of intercepted snow where
TorchVIC leaves 0; at step 780, cell 6, the `> 0` gate of
solve_snow.c:146 takes different branches and the snow packs fork.
Replaying VIC's step-779 inputs through TorchVIC's melt_and_release
reproduces VIC's 1.3e-19 bit for bit, so the code is VIC's; the exact-math
mode (ATEN_CPU_CAPABILITY=default) does not change it. A pass there needs
bit-identical rounding of the ground balance, not different physics.
"""
import re

IMPLEMENTED, NO_EFFECT, REFUSED = 'implemented', 'no effect', 'refused'

#: Any value (a path, a number, a list of tokens).
ANY = '*'

_B = ('TRUE', 'FALSE')

_HIST = 'history output (torchvic.io.history writes the OUTFILE streams)'
_WINDOW = ('the simulation window: the forcing records from STARTYEAR/'
           'STARTMONTH/STARTDAY/STARTSEC to the end of ENDYEAR/ENDMONTH/'
           'ENDDAY (or NRECS model steps) are run, as vic_start.c sets it; '
           'a forcing that does not cover the window is refused')
#: verification/noeffect.py: VIC 5.1.0 run with and without the option.
_SAME = ('; VIC 5.1.0\'s own output is bit-identical with and without it '
         '(verification/noeffect.py: Walnut Gulch 31 days, Stehekin 10 days, '
         'energy balance, every output variable)')
_STATE = ('a VIC image-driver state file is written at the end of the step '
          'that reaches STATEYEAR/STATEMONTH/STATEDAY/STATESEC '
          '(check_save_state_flag.c; vic_store.c) in every configuration: '
          'the STATE_* variables of FULL_ENERGY TRUE and FALSE, '
          'FROZEN_SOIL/SPATIAL_FROST, SPATIAL_SNOW, SNOW_BAND, CARBON, '
          'LAKES, CLOSE_ENERGY, QUICK_FLUX FALSE and the daily step, '
          'dz_node, node_depth and the coordinates, within the row\'s own '
          'TorchVIC-vs-VIC agreement of VIC\'s file (verification/'
          'state_matrix.py). TorchVIC adds TV_* variables (routing tail, '
          'SPATIAL_SNOW stores, lake state, snow-free stand-in) that VIC '
          'ignores and TorchVIC reads, so its own restart is bit for bit')
_VIC_STOPS = 'VIC itself stops on this (get_global_param.c)'
_OLD_SPELLING = ('VIC 4.2.d\'s spelling, which every VIC release stops on '
                 'while naming its replacement (4.2.d get_global_param.c:'
                 '548-565; 5.1.0 image :369-392, classic :576-603); read as '
                 '"%s" = BASEFLOW %s, the run is the BASEFLOW run')


def _o(*rows):
    """{value: (status, note)} from (values, status, note) rows."""
    out = {}
    for values, status, note in rows:
        for v in (values if isinstance(values, tuple) else (values,)):
            out[v] = (status, note)
    return out


#: The image-driver keys VIC 5.1.0 dispatches on (82, tools/
#: extract_vic_options.py), each value's disposition in TorchVIC.
IMAGE_OPTIONS = {
    # -- time --------------------------------------------------------------
    'MODEL_STEPS_PER_DAY': _o(
        (ANY, IMPLEMENTED,
         'the model step. Equal to the forcing\'s records a day, the model '
         'steps on the forcing; 1 with a sub-daily forcing, the day\'s '
         'records are averaged into one step as vic_force.c:404-412 does '
         '(`--daily` forces this from the command line). Any other step is '
         'refused')),
    'SNOW_STEPS_PER_DAY': _o(
        (ANY, IMPLEMENTED,
         'must equal the forcing\'s records a day (VIC stops otherwise) and, '
         'for a sub-daily model step, the model step. With a daily model '
         'step a day with a pack, canopy snow, or snowflag is sub-stepped '
         'at this rate (surface_fluxes.c:258-269): the surface once per '
         'snow step, the soil column once on the summed throughfall. A '
         'snow-free day stays the daily mean')),
    'RUNOFF_STEPS_PER_DAY': _o(
        (ANY, IMPLEMENTED, 'sub-steps the runoff solution '
                           '(column.SoilParams)')),
    'STARTYEAR': _o((ANY, IMPLEMENTED, _WINDOW)),
    'STARTMONTH': _o((ANY, IMPLEMENTED, _WINDOW)),
    'STARTDAY': _o((ANY, IMPLEMENTED, _WINDOW)),
    'STARTSEC': _o((ANY, IMPLEMENTED, _WINDOW)),
    'ENDYEAR': _o((ANY, IMPLEMENTED, _WINDOW)),
    'ENDMONTH': _o((ANY, IMPLEMENTED, _WINDOW)),
    'ENDDAY': _o((ANY, IMPLEMENTED, _WINDOW)),
    'NRECS': _o((ANY, IMPLEMENTED, _WINDOW)),
    'CALENDAR': _o(
        (('STANDARD', 'GREGORIAN', 'PROLEPTIC_GREGORIAN', 'NOLEAP',
          '365_DAY', '360_DAY', 'JULIAN', 'ALL_LEAP', '366_DAY'),
         IMPLEMENTED,
         'the dates are the forcing file\'s; a CALENDAR that disagrees with '
         'the forcing\'s calendar is refused')),
    # -- files --------------------------------------------------------------
    'DOMAIN': _o((ANY, IMPLEMENTED, 'the domain file (io.domain)')),
    'DOMAIN_TYPE': _o((ANY, IMPLEMENTED,
                       'every domain variable name is taken from the file')),
    'PARAMETERS': _o((ANY, IMPLEMENTED, 'the parameter file (io.params)')),
    'FORCING1': _o((ANY, IMPLEMENTED,
                    'the forcing prefix; every year is read (io.forcing)')),
    'FORCE_TYPE': _o((ANY, IMPLEMENTED,
                      'the forcing variable names are taken from the file')),
    'FORCING2': _o(
        (ANY, IMPLEMENTED,
         'the vegetation history file <prefix><year>.nc (io.veghist): its '
         'LAI, fcanopy and albedo (time, veg_class, y, x) for the *_SRC '
         'set to FROM_VEGHIST; it needs a FORCE_TYPE line after it, or VIC '
         'never opens the file. VIC reads no other variable from it '
         '(vic_force.c:62-205 reads every met field from FORCING1). igp '
         '(36 cells, 2004, perturbed history of all three): EB 3-hourly, '
         'EB daily and WB 3-hourly pass, TorchVIC-vs-VIC water 8.8e-13 / '
         '2.5e-09 / 9.5e-13, energy 4.3e-14 / 8.7e-10 / 5.6e-15; with LAI 0 '
         'on two days a month VIC itself loses 0.18 mm of canopy water on '
         'those steps (its own closure warn, reproduced by TorchVIC to '
         '1e-13). VIC 5.1.0 core only')),
    'CONSTANTS': _o(
        (ANY, IMPLEMENTED,
         'VEG_LAI_WATER_FACTOR is applied; any other constant in the file '
         'is refused by name (GlobalParam.constants)')),
    'INIT_STATE': _o(
        ('FALSE', NO_EFFECT, 'VIC\'s default: no state is restored'),
        (ANY, IMPLEMENTED,
         'the run starts from the VIC image-driver state file '
         '(vic_restore.c; torchvic.io.state_file) laid over the cold start, '
         'as vic_populate_model_state.c does, in every configuration; '
         'dimensions and coordinates are checked as check_init_state_file '
         'does. VIC\'s own restart is reproduced where it departs from '
         'its continuous run: with LAKES TRUE the lake\'s soil moisture is '
         'read from STATE_LAKE_SOIL_NODE_TEMP (vic_restore.c 5.1.0:666-675), '
         'and with SPATIAL_SNOW TRUE the stores the file lacks start at '
         'initialize_snow\'s zeros; TorchVIC\'s own file also carries '
         'those in TorchVIC-only TV_* variables, so its round trip is '
         'exact; verification/state_matrix.py')),
    'STATENAME': _o((ANY, IMPLEMENTED, _STATE)),
    'STATEYEAR': _o((ANY, IMPLEMENTED, _STATE)),
    'STATEMONTH': _o((ANY, IMPLEMENTED, _STATE)),
    'STATEDAY': _o((ANY, IMPLEMENTED, _STATE)),
    'STATESEC': _o((ANY, IMPLEMENTED, _STATE)),
    'STATE_FORMAT': _o((ANY, NO_EFFECT,
                        'the state file is written as NETCDF4 whatever '
                        'STATE_FORMAT says; the variables are the same')),
    'ROUT_PARAM': _o(
        (ANY, NO_EFFECT,
         'read only by VIC\'s RVIC routing extension; the stock image build '
         'links the routing stub, which ignores it. TorchVIC\'s own '
         'unit-hydrograph discharge is an extra output' + _SAME)),
    'LOG_DIR': _o((ANY, NO_EFFECT, 'where VIC writes its log')),
    # -- history output -----------------------------------------------------
    'OUTFILE': _o((ANY, IMPLEMENTED, _HIST)),
    'OUTVAR': _o((ANY, IMPLEMENTED, _HIST)),
    'AGGFREQ': _o((ANY, IMPLEMENTED, _HIST)),
    'HISTFREQ': _o((ANY, IMPLEMENTED, _HIST)),
    'COMPRESS': _o((ANY, IMPLEMENTED, _HIST)),
    'OUT_FORMAT': _o((ANY, IMPLEMENTED, _HIST)),
    'OUT_TIME_UNITS': _o((ANY, IMPLEMENTED, _HIST)),
    'RESULT_DIR': _o((ANY, IMPLEMENTED, _HIST)),
    # -- energy balance and soil temperature --------------------------------
    'FULL_ENERGY': _o((_B, IMPLEMENTED, 'the surface energy balance')),
    'FROZEN_SOIL': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        ('TRUE', IMPLEMENTED,
         'with FULL_ENERGY TRUE or FALSE on the QUICK_FLUX path, and with '
         'FULL_ENERGY TRUE on the finite-difference profile with freezing '
         '(QUICK_FLUX FALSE). FROZEN_SOIL TRUE turns QUICK_FLUX off '
         '(get_global_param.c:104-107), so the QUICK_FLUX path needs a '
         '"QUICK_FLUX TRUE" line after it; with FULL_ENERGY FALSE VIC runs '
         'frozen soil in water-balance mode (Tsurf = Tair, the fusion term '
         'in NetBareRad, the ice from the quick-flux layer temperatures), '
         'which TorchVIC runs as VIC does (basin.simulate frozen_soil '
         'without full_energy) on the QUICK_FLUX path only; its '
         'finite-difference profile is refused (to implement). The himal '
         'row warns at its base row\'s grade (TRANSP_VEG/ADVECTION '
         '1.4e-04/2.5e-04)')),
    'QUICK_FLUX': _o(
        ('TRUE', IMPLEMENTED, 'the two-node ground heat flux'),
        ('FALSE', IMPLEMENTED,
         'the finite-difference soil temperature profile '
         '(core/soiltemp.py), solved at every trial surface temperature '
         '(func_surf_energy_bal.c:404-445); needs FULL_ENERGY or '
         'FROZEN_SOIL, and NODES 3-50 (see NODES and IMPLICIT). himal '
         'full year: warn at the base row\'s grade (water 1.2e-03 '
         'TRANSP_VEG, energy 5.2e-03 MELT_ENERGY; base 7.1e-04/3.9e-03): '
         'through step 779 every variable agrees to 1.3e-12 relative, and '
         'both rows first leave VIC at the same step 780 (2004-04-07T12, '
         'cell 6), the base row\'s inherited April canopy/melt event; '
         'closure is VIC\'s own 0.0209 mm. 14 days: water 9.8e-12, energy '
         '1.4e-15, closure VIC\'s own 0.0011 mm; NODES 12 with EXP_TRANS, '
         'NOFLUX, QUICK_SOLVE, FROZEN_SOIL and FROZEN_SOIL + EXP_TRANS: '
         'water <= 3.0e-11, energy <= 1.5e-14 over 14 days; with IMPLICIT, '
         'IMPLICIT + EXP_TRANS and FROZEN_SOIL + IMPLICIT: pass, <= '
         '1.4e-09 / 4.0e-09 over 3 days (the Newton solve is slow on CPU)')),
    'IMPLICIT': _o(
        ('FALSE', IMPLEMENTED,
         'VIC\'s default with FROZEN_SOIL FALSE; the Gauss-Seidel sweep '
         'where QUICK_FLUX is FALSE (frozen_soil.c:129), read and without '
         'effect where it is TRUE'),
        ('TRUE', IMPLEMENTED,
         'the Newton solve of the finite-difference profile with '
         'QUICK_FLUX FALSE (frozen_soil.c:235), falling back to the sweep '
         'where it fails; with QUICK_FLUX TRUE VIC stops '
         '(get_global_param.c:933-936), and so does TorchVIC. With NOFLUX '
         'FALSE a 3-node column (QUICK_SOLVE can shrink any column to 3) '
         'is refused: VIC reads past its single unknown there '
         '(frozen_soil.c:793-796), uninitialized memory rather than '
         'physics, so use NOFLUX TRUE, NODES >= 4, or IMPLICIT FALSE')),
    'EXP_TRANS': _o(
        ('FALSE', IMPLEMENTED,
         'VIC\'s default with FROZEN_SOIL FALSE; the uniform node grid '
         'where QUICK_FLUX is FALSE (vic_init.c:846), read and without '
         'effect where it is TRUE'),
        ('TRUE', IMPLEMENTED,
         'the exponential node grid with QUICK_FLUX FALSE '
         '(vic_init.c:887); with QUICK_FLUX TRUE VIC stops '
         '(get_global_param.c:933-936), and so does TorchVIC')),
    'QUICK_SOLVE': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        ('TRUE', IMPLEMENTED,
         'bracketing on the reduced column with QUICK_FLUX FALSE '
         '(calc_surf_energy_bal.c:257), re-solved on the full one where '
         'the surface crosses freezing (:397); needs NOFLUX FALSE and '
         'EXP_TRANS FALSE, as VIC requires (get_global_param.c:943-949). '
         'With QUICK_FLUX TRUE VIC only re-solves the surface temperature '
         'on the same nodes when it changes sign (:397), which finds the '
         'same root' + _SAME)),
    'NOFLUX': _o(
        (('TRUE', 'FALSE'), IMPLEMENTED,
         'the bottom boundary of the finite-difference profile with '
         'QUICK_FLUX FALSE (soil_conduction.c:170-176); with QUICK_FLUX '
         'TRUE it is not read. Only a TRUE line sets it '
         '(get_global_param.c:125-130)')),
    'NO_FLUX': _o(
        (('TRUE', 'FALSE'), IMPLEMENTED, 'alias of NOFLUX (see NOFLUX)')),
    'GRND_FLUX_TYPE': _o(
        ('GF_410', IMPLEMENTED, 'VIC\'s default ground flux'),
        ('GF_406', IMPLEMENTED,
         'the VIC 4.0.6 ground flux: the surface term kappa1/D1 (T1 - '
         'Tsurf) alone, no deep term, no /2 '
         '(func_surf_energy_bal.c:385-391), passed to basin.simulate. '
         'Stehekin (10 days, energy balance): ET +6.87% in VIC and in '
         'TorchVIC, SWE -0.70% in both; TorchVIC-vs-VIC water 7.8e-14, '
         'energy 1.3e-15')),
    'NODES': _o(
        (ANY, IMPLEMENTED,
         'the thermal node count with QUICK_FLUX FALSE (vic_init.c:833; '
         '3-50, and 4 or more with IMPLICIT and NOFLUX FALSE, see '
         'IMPLICIT); with QUICK_FLUX TRUE VIC warns and resets it to 3 '
         '(get_global_param.c:925-931)')),
    'CLOSE_ENERGY': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default: one canopy pass'),
        ('TRUE', IMPLEMENTED,
         'the canopy air temperature is solved, not Tair '
         '(calc_atmos_energy_bal.c:73-102), and the ground/canopy '
         'balances iterate to closure (surface_fluxes.c:442-720, '
         'vic_init.c:1556; basin._ce_step), passed to basin.simulate. '
         'Needs FULL_ENERGY TRUE, as VIC does. Stehekin row: ET -24.62% '
         'in VIC and in TorchVIC, SWE +0.32% in both; TorchVIC-vs-VIC '
         'water 7.9e-13, energy 1.4e-15')),
    'TFALLBACK': _o(
        ('TRUE', IMPLEMENTED,
         'VIC\'s default: a temperature solve whose bracket holds no root '
         'keeps the previous temperature (calc_surf_energy_bal.c:311, '
         'snow_melt.c:267, snow_intercept.c:342, calc_atmos_energy_bal.c:87), '
         'as TorchVIC\'s solvers do; `torchvic run` reports the fallbacks '
         'taken per solve (surface, snow pack, canopy foliage, canopy air)'),
        ('FALSE', IMPLEMENTED,
         'the first step on which any of those solves finds no root stops '
         'the run (core/tfallback.TemperatureSolveError names the step, the '
         'lowest cell, the tile and the solves), at the step and cell where '
         'pristine VIC 5.1.0 fails: its error_* routine dumps the solve and '
         'the image driver then dies in the surface dump (SIGSEGV) or, for '
         'the canopy foliage and snow pack, carries on with that cell\'s '
         'state half advanced, which TorchVIC does not reproduce. Not reachable, so not guarded: the lake ice pack '
         '(ice_melt.c:346; LAKES is refused) and the soil temperature '
         'profile (frozen_soil.c:408,465,515; QUICK_FLUX FALSE is refused). '
         'VIC\'s cold-nose hack (frozen_soil.c:301,495) runs only with '
         'TFALLBACK TRUE and is that profile\'s. Not available on the CUDA-graph '
         'path (GRAPHS), which the run leaves for the eager one. Where '
         'nothing fails VIC\'s output is the same either way '
         '(verification/noeffect.py)')),
    # -- snow ----------------------------------------------------------------
    'SNOW_BAND': _o(
        ('FALSE', IMPLEMENTED, 'one band per cell, at the file\'s elev'),
        ('TRUE', IMPLEMENTED,
         'every band with area runs as its own column, at its lapsed air '
         'temperature and scaled precipitation (surface_fluxes.c:358-359), '
         'and every output is area-weighted back (put_data.c; '
         'core/snowbands.py). One band with area is the cell itself '
         '(vic_init.c:1020-1034). Stehekin, forest and tntip banded rows '
         'pass at 1e-12..1e-15; the himal rows warn at their base rows\' '
         'grade (TRANSP_VEG/ADVECTION 4-7e-04)')),
    'SNOW_DENSITY': _o(
        ('DENS_BRAS', IMPLEMENTED, 'VIC\'s default'),
        ('DENS_SNTHRM', IMPLEMENTED,
         'the SNTHERM89 pack densification (snow_utility.c:61-130), '
         'passed to basin.simulate; bit-exact on 9216 oracle rows. '
         'Himal EB 3h vs VIC 5.1.0: warn 5.8e-04 TRANSP_VEG / 4.0e-03 '
         'ADVECTION / 9.9e-03 PET, the base row\'s warn (7.1e-04 / '
         '3.9e-03 / 9.9e-03): no graded var diverges before the base '
         'row\'s first, 2004-04-07T12:00 cell 6 MELT_ENERGY')),
    'SPATIAL_SNOW': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        ('TRUE', IMPLEMENTED,
         'calc_snow_coverage.c (core/snow.py snow_coverage, basin.simulate '
         'spatial_snow); TRUE reads max_snow_distrib_slope from the '
         'parameter file as vic_init.c:691-700 does, and refuses a file '
         'without it or with a negative value (vic_init.c:799-802). '
         'Himal EB 3h vs VIC 5.1.0: warn 2.4e-03 TRANSP_VEG / 7.0e-03 '
         'ADVECTION, closure 2.1e-02 inherited; the warn is at the base '
         'row\'s grade (himal_EB_510_3h 4.2e-04 / 3.7e-03): 1-ulp canopy '
         'straddles strand dust in VIC itself and fork the snow pack, so '
         'a pass would need bit-identical rounding')),
    'MAX_SNOW_ALBEDO': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        ('TRUE', IMPLEMENTED,
         'reads the per-class max_snow_albedo (vic_init.c:186-196) as the '
         'fresh-snow albedo of every vegetation tile; the bare-soil tile '
         'keeps SNOW_NEW_SNOW_ALB (surface_fluxes.c:524-535); a file '
         'without the variable is refused. Matrix row Stehekin EB 1 h '
         'PASS, worst graded 9.8e-14 vs vic_image.exe 5.1.0 (VIC moves '
         '14% in ET and 28% in runoff)')),
    'BLOWING': _o(
        (_B, IMPLEMENTED,
         'CalcBlowingSnow.c (core/blowing.py); TRUE reads sigma_slope, '
         'lag_one and fetch from the parameter file as vic_init.c:1281-1330 '
         'does, and refuses values VIC refuses (sigma_slope, lag_one <= 0, '
         'fetch <= 1)')),
    'BLOWING_VAR_THRESHOLD': _o((_B, IMPLEMENTED, 'with BLOWING')),
    'BLOWING_CALC_PROB': _o((_B, IMPLEMENTED, 'with BLOWING')),
    'BLOWING_SIMPLE': _o((_B, IMPLEMENTED, 'with BLOWING')),
    'BLOWING_FETCH': _o((_B, IMPLEMENTED, 'with BLOWING')),
    'BLOWING_SPATIAL_WIND': _o((_B, IMPLEMENTED, 'with BLOWING')),
    'COMPUTE_TREELINE': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        (ANY, IMPLEMENTED,
         'VIC\'s image driver stops on it (get_global_param.c:968-970); the '
         'classic driver runs it and so does TorchVIC (core/treeline.py): '
         'a band whose July_Tavg + Tfactor is <= 10 C is above the '
         'treeline (compute_treeline.c:76-85), its overstory tiles leave '
         'the cell\'s outputs and the others are re-weighted '
         '(put_data.c:76-103, :180-190); with more than one snow band a '
         'cell of overstory only gives 0.001 to bare soil or to the tile '
         'class the value names (read_vegparam.c:371-433). Needs '
         'JULY_TAVG_SUPPLIED TRUE and July_Tavg in the parameter file '
         '(VIC stops without, in both drivers: '
         'get_global_param.c:971-973 image, :1299-1303 classic). '
         'Verified against vic_classic.exe (verification/classic_rows.py)')),
    'JULY_TAVG_SUPPLIED': _o(
        (_B, IMPLEMENTED,
         'the cell\'s July air temperature, read only by COMPUTE_TREELINE '
         '(compute_treeline.c:44-47; the soil file\'s last column, '
         'read_soilparam.c:574-586); without the treeline VIC 5.1.0\'s own '
         'output is bit-identical with and without it (verification/'
         'noeffect.py)')),
    # -- soil and runoff -----------------------------------------------------
    'BASEFLOW': _o(
        ('ARNO', IMPLEMENTED, 'VIC\'s default'),
        ('NIJSSEN2001', IMPLEMENTED,
         'the file\'s d1..d4 reach the column as ARNO\'s Ds, Dsmax, Ws, c '
         '(vic_init.c:815-829; read_soilparam.c:693-707). Walnut Gulch '
         'passes at water 2.5e-11; igp fails only on VIC\'s own 3.3e-12 '
         'Q12-cancellation bottom-layer noise (TV holds the true ~0)')),
    # global_param.GlobalParam reads these as the BASEFLOW line they stand
    # for, so a run on either spelling is the BASEFLOW run (tests/
    # test_refused_zero.py: bit-identical). The rows document the reading.
    'ARNO_PARAMS': _o(
        ('TRUE', IMPLEMENTED, _OLD_SPELLING % ('ARNO_PARAMS TRUE',
                                               'NIJSSEN2001')),
        (ANY, IMPLEMENTED, _OLD_SPELLING % ('ARNO_PARAMS FALSE', 'ARNO'))),
    'NIJSSEN2001_BASEFLOW': _o(
        ('TRUE', IMPLEMENTED, _OLD_SPELLING % ('NIJSSEN2001_BASEFLOW TRUE',
                                               'NIJSSEN2001')),
        (ANY, IMPLEMENTED, _OLD_SPELLING % ('NIJSSEN2001_BASEFLOW FALSE',
                                            'ARNO'))),
    'SHARE_LAYER_MOIST': _o((_B, IMPLEMENTED, 'canopy_evap.c:287-362')),
    'SPATIAL_FROST': _o(
        ('TRUE', IMPLEMENTED,
         'the sub-grid frost distribution over Nfrost areas '
         '(SPATIAL_FROST TRUE n, n = 1..10): the frost_slope temperature '
         'spread per cell (vic_init.c:703-710), per-area ice '
         '(soil_conduction.c:636-688), the evap split (runoff.c:86-123) '
         'and area-weighted columns (:361-368). Needs frost_slope in the '
         'parameter file; without FROZEN_SOIL every area is the same '
         'unfrozen column and the option runs as one (announced). Both '
         'himal frost rows warn at base grade (FSWB 7.0e-04/1.2e-03, '
         'FS 1.6e-03/6.8e-03)'),
        (ANY, IMPLEMENTED, 'off (VIC\'s default): any value but TRUE '
                           '(get_global_param.c:221-229)')),
    'ORGANIC_FRACT': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        ('TRUE', IMPLEMENTED,
         'reads organic, bulk_density_org and soil_density_org '
         '(vic_init.c:572-605); the soil\'s bulk and particle densities '
         'are the organic-weighted mix (:621-625, :735-737), the thermal '
         'conductivity and heat capacity mix the mineral and organic '
         'parts (soil_conduction.c:17-99). Matrix rows PASS vs '
         'vic_image.exe 5.1.0: Stehekin EB 1 h worst graded 6.0e-13, '
         'Walnut Gulch EB 3 h 1.1e-13')),
    'BULK_DENSITY_COMB': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        ('TRUE', IMPLEMENTED,
         'reads bulk_density_comb as the soil\'s bulk density '
         '(vic_init.c:607-617); bulk_density stays the mineral density '
         'of the thermal conductivity, and bulk_density_org is not read '
         '(:583). Matrix rows PASS vs vic_image.exe 5.1.0, Stehekin EB '
         '1 h, worst graded 2.5e-13 (alone) and 1.5e-13 (with '
         'ORGANIC_FRACT)')),
    'LAKES': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        ('TRUE', IMPLEMENTED, 'the lake and wetland tile (core/lake.py '
                              'coupled by model/lakecell.py, vic_run.c '
                              '5.1.0:132-466, put_data.c 5.1.0:148-400); '
                              'needs FULL_ENERGY TRUE (VIC\'s own rule, '
                              'get_global_param.c:980), one snow band and '
                              'one lake node count per run. Findley Lake 1970 '
                              'hourly against VIC 5.1.0: every graded '
                              'variable and all 33 OUT_LAKE_* <= 7e-8 '
                              '(worst OUT_ASAT 6.7e-08). OUT_EVAP_BARE is '
                              'VIC\'s uninitialised lake esoil (all_vars is '
                              'malloc\'d, vic_alloc.c:61, and '
                              'initialize_lake.c never sets it): 0 in one '
                              'VIC run, 8.8e177 in a rerun; TorchVIC reports '
                              '0')),
    'LAKE_PROFILE': _o(
        ('TRUE', IMPLEMENTED, 'the basin_depth/basin_area profile, read only '
                              'with LAKES TRUE (vic_init.c 5.1.0:1444-1462)'),
        ('FALSE', IMPLEMENTED, 'the generated basin (compute_lake_params.c '
                               '5.1.0:31-46); a file whose basin_depth has '
                               'the lake_node dimension is refused, as VIC '
                               'stops on it (vic_init.c:1463-1489); no '
                               'matrix row yet (needs a 2-D basin file)')),
    'LAKE_NODES': _o((ANY, NO_EFFECT, 'read only with LAKES TRUE' + _SAME)),
    # -- vegetation and canopy -----------------------------------------------
    'LAI_SRC': _o(
        (('FROM_VEGLIB', 'FROM_VEGPARAM'), IMPLEMENTED,
         'both read the parameter file\'s LAI (vic_init.c:198-213)'),
        ('FROM_VEGHIST', IMPLEMENTED,
         'the per-step LAI of FORCING2 (vic_force.c:303-319), averaged over '
         'the model step, replaces the parameter file\'s monthly value; '
         'needs FORCING2 and a FORCE_TYPE line (io.veghist)')),
    'ALB_SRC': _o(
        (('FROM_VEGLIB', 'FROM_VEGPARAM'), IMPLEMENTED,
         'both read the parameter file\'s albedo (vic_init.c:215-230)'),
        ('FROM_VEGHIST', IMPLEMENTED,
         'the per-step albedo of FORCING2 (vic_force.c:342-359), averaged over '
         'the model step, replaces the parameter file\'s monthly value; '
         'needs FORCING2 and a FORCE_TYPE line (io.veghist)')),
    'FCAN_SRC': _o(
        ('FROM_DEFAULT', IMPLEMENTED, 'fcanopy 1 (vic_init.c:257-268)'),
        (('FROM_VEGLIB', 'FROM_VEGPARAM'), IMPLEMENTED,
         'the parameter file\'s fcanopy (vic_init.c:270-281); a parameter '
         'file without it is refused, as VIC stops on it'),
        ('FROM_VEGHIST', IMPLEMENTED,
         'the per-step fcanopy of FORCING2 (vic_force.c:322-339), averaged over '
         'the model step, replaces the parameter file\'s monthly value; '
         'needs FORCING2 and a FORCE_TYPE line (io.veghist)')),
    'AERO_RESIST_CANSNOW': _o(
        (('AR_406_FULL', 'AR_406'), IMPLEMENTED,
         'the snow-covered canopy\'s aerodynamic resistance '
         '(func_canopy_energy_bal.c:166-227), passed to basin.simulate. '
         'AR_406 against VIC\'s default on Stehekin (10 days, energy '
         'balance): ET +93% in VIC, +82% in TorchVIC; SWE -4.3% and -3.6% '
         '(verification/results/sensitivity.csv)'),
        ('AR_410', IMPLEMENTED,
         'the stability-corrected resistance to intercepted snow '
         '(func_canopy_energy_bal.c:178-188: Ra[1] divided by the stability '
         'correction at the trial foliage temperature). Stehekin (10 days, '
         'energy balance): ET +37.49% in VIC and in TorchVIC, SWE -1.66% '
         'in both; TorchVIC-vs-VIC water 3.5e-13, energy 1.4e-15'),
        ('AR_406_LS', IMPLEMENTED,
         'ten times the canopy resistance with intercepted snow, Ra[0] '
         'without (func_canopy_energy_bal.c:166-227, the AR_406 branch '
         'without the divide-back). Stehekin (10 days, energy balance): '
         'ET +1.43% in VIC and in TorchVIC; TorchVIC-vs-VIC water '
         '2.4e-13, energy 1.7e-15')),
    'RC_MODE': _o(
        ('RC_JARVIS', IMPLEMENTED, 'VIC\'s default'),
        ('RC_PHOTO', IMPLEMENTED,
         'canopy resistance from photosynthetic demand, per soil layer '
         'and shared case (canopy_evap.c:296-307, :400-411; penman.c '
         'calc_rc_ps). Needs CARBON and the library/forcing it reads; '
         'without CARBON VIC resets it to RC_JARVIS. Walnut Gulch 2008 '
         '(carbon library, 3-hourly): transpiration x0.29 and ET x0.39 '
         'of RC_JARVIS in VIC and in TorchVIC alike; TorchVIC-vs-VIC '
         'water 1.2e-13, energy 6.3e-15 (water balance 1.2e-13 / 9.8e-16)')),
    'CARBON': _o((_B, IMPLEMENTED, 'the carbon cycle (core/carbon.py)')),
    'CANOPY_LAYERS': _o((ANY, IMPLEMENTED, 'the carbon step\'s canopy '
                                           'layers')),
    'VEGLIB_PHOTO': _o((_B, IMPLEMENTED,
                        'the photosynthesis fields are read from the '
                        'parameter file when present')),
    'CORRPREC': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        ('TRUE', IMPLEMENTED,
         'the gauge undercatch correction of correct_precip.c, applied to '
         'the rain and the snow of every step from the step\'s wind and '
         'the soil file\'s rough and snow_rough (vic_run.c:106-114, '
         'solve_snow.c:117-120); not with a daily step over a sub-daily '
         'forcing. Matrix rows PASS vs vic_image.exe 5.1.0, worst '
         'graded 7.0e-13 (Stehekin EB 1 h), 7.8e-14 (Stehekin WB 1 h), '
         '1.0e-13 (Walnut Gulch WB 3 h)')),
    'WIND_H': _o((ANY, IMPLEMENTED, 'the wind measurement height '
                                    '(case.resolve_wind_h)')),
    # -- keys VIC's image driver stops on, that another driver or release runs
    # -- (nothing stays refused for that) --------------------------------------
    'ATMOS_STEPS_PER_DAY': _o(
        (ANY, NO_EFFECT,
         'a classic-driver key (get_global_param.c:68-70) that the image '
         'driver stops on (:525-528); the classic driver validates it '
         '(:811-856) and derives atmos_dt from it, which only the log '
         'prints (display_current_settings.c:77, vic_classic_timing.c:83) '
         '-- the forcing is read and disaggregated at SNOW_STEPS_PER_DAY, '
         'so no result moves. Validated here as VIC validates it')),
    'OUTPUT_FORCE': _o(
        ('FALSE', IMPLEMENTED, 'VIC\'s default'),
        (ANY, IMPLEMENTED,
         'VIC 5 removed it (ReleaseNotes: the classic driver; image '
         'get_global_param.c:529-532 stops); VIC 4.2.d has it and so does '
         'TorchVIC: the model is not run, the forcing as initialize_atmos.c '
         'leaves it is written by write_forcing_file.c as 4.2.d\'s per-cell '
         'files (full_data_<lat>_<lon>, the 8 default variables or the '
         'global file\'s N_OUTFILES/OUTVAR; ASCII or BINARY_OUTPUT, '
         'ALMA_OUTPUT, PRT_HEADER). Byte-identical to vicNl on sub-daily '
         'forcing (tests/test_refused_zero.py)')),
}

#: VIC's defaults where the file is silent (initialize_options.c 5.1.0).
DEFAULTS = {'QUICK_FLUX': 'TRUE', 'IMPLICIT': 'TRUE', 'EXP_TRANS': 'TRUE',
            'FROZEN_SOIL': 'FALSE', 'FULL_ENERGY': 'FALSE',
            'NOFLUX': 'FALSE', 'QUICK_SOLVE': 'FALSE', 'NODES': 3}

_TRUE = ('TRUE',)


def _value(rest):
    return str(rest[0]).upper() if rest else ''


def disposition(key, value):
    """(status, note) for one KEY value, or None for a key VIC's image
    driver does not dispatch on."""
    table = IMAGE_OPTIONS.get(key.upper())
    if table is None:
        return None
    v = str(value).upper()
    if v in table:
        return table[v]
    if ANY in table:
        return table[ANY]
    return (REFUSED, 'not a value TorchVIC knows for %s (it runs %s)'
            % (key, ', '.join(sorted(k for k, (s, _) in table.items()
                                     if s != REFUSED))))


def thermal(gp):
    """The effective QUICK_FLUX, IMPLICIT, EXP_TRANS, FROZEN_SOIL,
    FULL_ENERGY, NOFLUX, QUICK_SOLVE and NODES, as VIC's parser leaves
    them: defaults, then the file's lines in order, with FROZEN_SOIL
    TRUE turning QUICK_FLUX off and FROZEN_SOIL FALSE turning IMPLICIT
    and EXP_TRANS off (get_global_param.c 5.1.0:101-112). NODES is an
    integer; the rest are booleans. NO_FLUX is NOFLUX's alias, and only
    a TRUE line of either sets it (get_global_param.c:125-130)."""
    eff = {k: (v if isinstance(v, int) else v == 'TRUE')
           for k, v in DEFAULTS.items()}
    for key, rest in gp.lines:
        if key in eff and rest:
            if key == 'NODES':
                # atoi, as VIC reads it.
                _m = re.match(r'\s*[+-]?\d+', str(rest[0]))
                eff[key] = int(_m.group(0)) if _m else 0
            elif key == 'NOFLUX':
                if _value(rest) == 'TRUE':
                    eff['NOFLUX'] = True
            else:
                on = _value(rest) == 'TRUE'
                eff[key] = on
                if key == 'FROZEN_SOIL':
                    if on:
                        eff['QUICK_FLUX'] = False
                    else:
                        eff['IMPLICIT'] = eff['EXP_TRANS'] = False
        elif key == 'NO_FLUX' and rest:
            if _value(rest) == 'TRUE':
                eff['NOFLUX'] = True
    return eff


#: vic_def.h:45, the most frost sub-areas VIC allows.
MAX_FROST_AREAS = 10


def spatial_frost_n(gp):
    """The effective Nfrost: the last SPATIAL_FROST line's count when it
    says TRUE, else 1 (get_global_param.c:221-229). 0 when that line has
    no usable count, which check() refuses."""
    n = 1
    for rest in gp.raw.get('SPATIAL_FROST', []):
        if rest and str(rest[0]).upper() == 'TRUE':
            try:
                n = int(rest[1]) if len(rest) > 1 else 0
            except (TypeError, ValueError):
                n = 0
        else:
            n = 1
    return n


#: vic_def.h:56-57, the bounds VIC puts on a sub-daily step count.
MIN_SUBDAILY_STEPS_PER_DAY = 4
MAX_SUBDAILY_STEPS_PER_DAY = 1440
HOURS_PER_DAY = 24


def atmos_steps_errors(gp):
    """The classic driver's checks on ATMOS_STEPS_PER_DAY, as refusals
    (get_global_param.c 5.1.0:811-856). A line that does not read as an
    unsigned integer is 0, which the classic driver defaults to hourly
    (sscanf %zu leaves it 0; :812-815). The model and snow step counts are
    the global file's; a check against one it does not give is skipped."""
    if 'ATMOS_STEPS_PER_DAY' not in gp.raw:
        return []
    try:
        n = int(gp.get('ATMOS_STEPS_PER_DAY'))
        n = n if n >= 0 else 0
    except (TypeError, ValueError):
        n = 0
    if n == 0:
        n = HOURS_PER_DAY
    name = 'ATMOS_STEPS_PER_DAY %s' % gp.get('ATMOS_STEPS_PER_DAY')
    if n < MIN_SUBDAILY_STEPS_PER_DAY:
        return ['%s: below the minimum of %d steps a day; VIC stops '
                '(get_global_param.c:816-822)' % (name,
                                                  MIN_SUBDAILY_STEPS_PER_DAY)]
    if n > HOURS_PER_DAY and n % HOURS_PER_DAY:
        return ['%s: above 24 and not a multiple of 24; VIC stops '
                '(get_global_param.c:824-828)' % name]
    if n > MAX_SUBDAILY_STEPS_PER_DAY:
        return ['%s: above the maximum of %d steps a day; VIC stops '
                '(get_global_param.c:830-836)' % (name,
                                                  MAX_SUBDAILY_STEPS_PER_DAY)]
    out = []
    for key in ('MODEL_STEPS_PER_DAY', 'SNOW_STEPS_PER_DAY'):
        m = gp.get_int(key)
        if m and n % m:
            out.append('%s: not a multiple of %s %d; VIC stops '
                       '(get_global_param.c:838-851)' % (name, key, m))
    return out


def treeline(gp):
    """(on, veg_class) for COMPUTE_TREELINE as the classic driver reads it
    (get_global_param.c:189-197): FALSE is off; any other token is on and
    atoi of it is the default vegetation class above the treeline (so TRUE
    reads as class 0), -1 (initialize_options.c) being bare soil."""
    on, veg = False, -1
    for rest in getattr(gp, 'raw', {}).get('COMPUTE_TREELINE', []):
        tok = str(rest[0]) if rest else ''
        if tok.upper() == 'FALSE':
            on = False
        else:
            on = True
            m = re.match(r'\s*[+-]?\d+', tok)
            veg = int(m.group(0)) if m else 0
    return on, veg


def check(gp, no_energy=False):
    """(refused, announced) for a parsed global file, from the file alone.

    `refused` is a list of lines, each naming an option and value and why;
    `announced` the options that are read and change nothing here, and the
    keys VIC's image driver does not know. The checks that need the
    parameter or forcing file are torchvic.case's."""
    refused, announced = [], []
    seen, quiet = set(), {}           # quiet: note -> ['KEY value', ...]
    eff = thermal(gp)
    for key, rest in gp.lines:
        value = _value(rest)
        if key in ('QUICK_FLUX', 'IMPLICIT', 'EXP_TRANS', 'FROZEN_SOIL',
                   'NOFLUX', 'NO_FLUX', 'NODES', 'QUICK_SOLVE'):
            continue                    # by their effective values, below
        d = disposition(key, value if rest else '')
        if d is None:
            if key not in seen:
                announced.append(
                    '%s: not an option of VIC 5\'s image driver; VIC warns '
                    'and ignores it (get_global_param.c:538), and so does '
                    'TorchVIC' % key)
            seen.add(key)
            continue
        status, note = d
        line = ('%s %s' % (key, ' '.join(rest))).strip()
        if status == REFUSED and '%s: %s' % (line, note) not in refused:
            refused.append('%s: %s' % (line, note))
        elif status == NO_EFFECT and key not in seen:
            quiet.setdefault(note, []).append(line)
        seen.add(key)
    qf = eff['QUICK_FLUX']
    if not qf:
        if not eff['FULL_ENERGY'] and not eff['FROZEN_SOIL']:
            refused.append(
                'QUICK_FLUX FALSE with FULL_ENERGY FALSE and FROZEN_SOIL '
                'FALSE: VIC stops on this (get_global_param.c:939-945)')
        elif not eff['FULL_ENERGY'] or no_energy:
            refused.append(
                'QUICK_FLUX FALSE in the water balance (%s): VIC runs '
                'the finite-difference profile there with FROZEN_SOIL, '
                'which TorchVIC does not (to implement); add "QUICK_FLUX '
                'TRUE" after FROZEN_SOIL for the water-balance frozen soil '
                'TorchVIC runs'
                % ('--no-energy' if no_energy and eff['FULL_ENERGY']
                   else 'FULL_ENERGY FALSE'))
        if eff['FROZEN_SOIL'] and spatial_frost_n(gp) > 1:
            refused.append(
                'QUICK_FLUX FALSE with SPATIAL_FROST TRUE: the profile\'s '
                'per-frost-area layer ice (soil_conduction.c:469-615) is '
                'not implemented; add "QUICK_FLUX TRUE" after FROZEN_SOIL')
        _tfb = [_value(rest) for key, rest in gp.lines
                if key == 'TFALLBACK' and rest]
        if _tfb and _tfb[-1] == 'FALSE':
            refused.append(
                'TFALLBACK FALSE with QUICK_FLUX FALSE: VIC stops at a '
                'failed node solve of the profile too (frozen_soil.c:301, '
                ':408, :465, :495, :515); TorchVIC\'s profile always falls '
                'back and does not report which node failed. Use '
                'TFALLBACK TRUE (VIC\'s default)')
        if eff['NODES'] > 50:
            refused.append(
                'NODES %d: VIC stops above MAX_NODES 50 '
                '(get_global_param.c:956)' % eff['NODES'])
        elif eff['NODES'] < 3:
            refused.append(
                'QUICK_FLUX FALSE with NODES %d: VIC reads Tnew_node[1] '
                'and Tnew_node[2] after the profile solve '
                '(func_surf_energy_bal.c:449-450), so fewer than 3 nodes '
                'is not a runnable grid' % eff['NODES'])
        if eff['QUICK_SOLVE'] and (eff['NOFLUX'] or eff['EXP_TRANS']):
            refused.append(
                'QUICK_SOLVE TRUE with QUICK_FLUX FALSE needs NOFLUX '
                'FALSE and EXP_TRANS FALSE: VIC stops on this '
                '(get_global_param.c:943-949)')
        if (eff['IMPLICIT'] and not eff['NOFLUX']
                and (eff['NODES'] == 3 or eff['QUICK_SOLVE'])):
            refused.append(
                'IMPLICIT TRUE with NOFLUX FALSE on a 3-node column '
                '(QUICK_SOLVE can shrink any column to 3): VIC reads '
                'T_2[1] past its single unknown there '
                '(frozen_soil.c:793-796), uninitialized memory rather '
                'than physics. Use NOFLUX TRUE, NODES >= 4, or IMPLICIT '
                'FALSE')
    else:
        for key in ('IMPLICIT', 'EXP_TRANS'):
            if eff[key]:
                refused.append(
                    '%s TRUE with QUICK_FLUX TRUE: VIC stops on this '
                    '(get_global_param.c:933-936). %s is TRUE by default '
                    'until a "FROZEN_SOIL FALSE" line resets it; set '
                    '"%s FALSE" after FROZEN_SOIL' % (key, key, key))
        # On the QUICK_FLUX path the thermal switches are read and change
        # nothing: IMPLICIT/EXP_TRANS FALSE are what VIC requires there,
        # NODES is reset to 3, the rest are not read at all.
        for key in ('IMPLICIT', 'EXP_TRANS'):
            if not eff[key] and key in gp.raw:
                quiet.setdefault('VIC requires them FALSE with QUICK_FLUX '
                                 'TRUE (get_global_param.c:933-936)',
                                 []).append('%s FALSE' % key)
        if 'NODES' in gp.raw:
            quiet.setdefault(
                'QUICK_FLUX runs 3 nodes whatever this says (VIC warns '
                'and resets it, get_global_param.c:925-931)'
                + _SAME, []).append('NODES %d' % eff['NODES'])
        for key in ('NOFLUX', 'NO_FLUX'):
            if key in gp.raw:
                for rest in gp.raw[key]:
                    line = ('%s %s' % (key, ' '.join(rest))).strip()
                    quiet.setdefault(
                        'the bottom boundary of the finite-difference '
                        'profile, which only QUICK_FLUX FALSE solves'
                        + _SAME, []).append(line)
        if 'QUICK_SOLVE' in gp.raw:
            for rest in gp.raw['QUICK_SOLVE']:
                line = 'QUICK_SOLVE %s' % ' '.join(rest)
                quiet.setdefault(
                    'with QUICK_FLUX VIC only re-solves the surface '
                    'temperature on the same nodes when it changes sign '
                    '(calc_surf_energy_bal.c:397), which finds the same '
                    'root' + _SAME, []).append(line.strip())
    for note, lines in quiet.items():
        announced.append('%s: read, no effect here: %s'
                         % (', '.join(lines), note))
    for rest in gp.raw.get('SPATIAL_FROST', []):
        if rest and str(rest[0]).upper() == 'TRUE':
            try:
                n = int(rest[1]) if len(rest) > 1 else 0
            except (TypeError, ValueError):
                n = 0
            if not 1 <= n <= MAX_FROST_AREAS:
                refused.append(
                    'SPATIAL_FROST TRUE %s: VIC stops unless the frost '
                    'area count is 1..%d (get_global_param.c:833-843)'
                    % (' '.join(rest[1:]) or '(no count)',
                       MAX_FROST_AREAS))
    hist_src = [k for k in ('LAI_SRC', 'FCAN_SRC', 'ALB_SRC')
                if str(gp.get(k) or '').upper() == 'FROM_VEGHIST']
    if hist_src:
        if gp.forcing2_prefix is None:
            refused.append(
                '%s FROM_VEGHIST without FORCING2: VIC opens '
                '<FORCING2 prefix><year>.nc (get_global_param.c:784-791) '
                'and stops when there is none' % hist_src[0])
        elif not gp.force2_type:
            refused.append(
                '%s FROM_VEGHIST with FORCING2 but no FORCE_TYPE line after '
                'it: VIC counts those lines (forcing_utils.c:108-148) and '
                'opens the file only when there is one, so it stops reading '
                'LAI/fcanopy/albedo from a file it never opened'
                % hist_src[0])
    elif gp.forcing2_prefix is not None:
        announced.append(
            'FORCING2: read, no effect here: without a *_SRC FROM_VEGHIST '
            'VIC opens the file and reads nothing from it (vic_force.c reads '
            'every met variable from FORCING1)')
    if spatial_frost_n(gp) > 1 and not eff['FROZEN_SOIL']:
        announced.append(
            'SPATIAL_FROST TRUE: read, no effect here: without FROZEN_SOIL '
            'every frost area holds the same unfrozen column, so the run '
            'is one area (VIC redistributes the same demand to ~1ulp)')
    for line in gp.translated:
        announced.append('%s: VIC 4.2.d\'s spelling, read as the BASEFLOW '
                         'line it stands for' % line)
    refused.extend(atmos_steps_errors(gp))
    if treeline(gp)[0]:
        if not any(r and str(r[0]).upper() == 'TRUE'
                   for r in gp.raw.get('JULY_TAVG_SUPPLIED', [])[-1:]):
            refused.append(
                'COMPUTE_TREELINE with JULY_TAVG_SUPPLIED FALSE: VIC stops '
                'in both drivers (get_global_param.c:971-973 image, '
                ':1299-1303 classic); set JULY_TAVG_SUPPLIED TRUE and give '
                'July_Tavg in the parameter file')
    ce = None
    for key, rest in gp.lines:
        if key == 'CLOSE_ENERGY' and rest:
            ce = _value(rest) == 'TRUE'
    if ce and (not eff['FULL_ENERGY'] or no_energy):
        refused.append(
            'CLOSE_ENERGY TRUE with FULL_ENERGY FALSE%s: VIC stops on this '
            '(get_global_param.c:961-964); set FULL_ENERGY TRUE'
            % (' (--no-energy)' if no_energy and eff['FULL_ENERGY'] else ''))
    return refused, announced


def _label(v, table):
    return v if v != ANY else ('any' if len(table) == 1 else 'any other')


def markdown():
    """The options table of docs/OPTIONS.md, from IMAGE_OPTIONS."""
    L = ['# VIC 5 image-driver options in TorchVIC', '',
         'Generated from `torchvic/io/options.py` (`python -m '
         'torchvic.io.options > docs/OPTIONS.md`, and `python -m '
         'torchvic.io.options README.md` for the README\'s summary); '
         '`tests/test_verification_matrix.py` fails when this file and the '
         'code disagree.', '',
         'Every key VIC 5.1.0\'s image driver reads from the global '
         'parameter file, and every value of it: **implemented** (TorchVIC '
         'runs it as VIC does, under the condition stated), **no effect** '
         '(VIC accepts it and it cannot change a result in a configuration '
         'TorchVIC runs; announced in the log) or **refused** (the run '
         'stops, naming the option). A value not listed is refused. A key '
         'that is not an image-driver option is announced and ignored, as '
         'VIC warns and ignores it.', '',
         'Defaults that matter: VIC\'s IMPLICIT and EXP_TRANS are TRUE '
         'until a `FROZEN_SOIL FALSE` line resets them, and VIC refuses '
         'them TRUE with QUICK_FLUX TRUE, so a file must carry '
         '`FROZEN_SOIL FALSE` (or set both FALSE), as VIC itself requires.',
         '', '| option | value | TorchVIC | note |', '|---|---|---|---|']
    for key in sorted(IMAGE_OPTIONS):
        table = IMAGE_OPTIONS[key]
        groups = {}
        for v, (status, note) in table.items():
            groups.setdefault((status, note), []).append(v)
        for (status, note), values in groups.items():
            vals = ', '.join(_label(v, table) for v in values)
            L.append('| %s | %s | %s | %s |'
                     % (key, vals, '**%s**' % status if status == REFUSED
                        else status, note.replace('|', '/')))
    counts = {}
    for table in IMAGE_OPTIONS.values():
        for status in set(s for s, _ in table.values()):
            counts[status] = counts.get(status, 0) + 1
    L += ['', '%d options. With at least one value: %s.'
          % (len(IMAGE_OPTIONS), ', '.join(
              '%s %d' % (s, counts.get(s, 0))
              for s in (IMPLEMENTED, NO_EFFECT, REFUSED))), '']
    return '\n'.join(L)


def summary():
    """The README's short form of the table: the options with a value
    refused, the options read to no effect, the rest implemented."""
    refused, quiet, done = [], [], []
    for key, table in sorted(IMAGE_OPTIONS.items()):
        st = set(s for s, _ in table.values())
        bad = sorted(_label(v, table) for v, (s, _) in
                     table.items() if s == REFUSED)
        if bad:
            refused.append('%s %s' % (key, '/'.join(bad)))
        if st == {NO_EFFECT} or (NO_EFFECT in st and not bad
                                 and IMPLEMENTED not in st):
            quiet.append(key)
        elif IMPLEMENTED in st:
            done.append(key)
    return '\n'.join([
        '| TorchVIC | options (image driver; values in docs/OPTIONS.md) |',
        '|---|---|',
        '| implemented (some values or conditions refused, see below) | %s |'
        % ', '.join(done),
        '| no effect (read, announced) | %s |' % ', '.join(quiet),
        '| **refused** by name | %s |' % ', '.join(refused)])


def refused_values():
    """Every (key, value) the table refuses on the global file alone, a
    concrete value standing for ANY (for the refusal tests)."""
    out = []
    for key, table in sorted(IMAGE_OPTIONS.items()):
        for v, (status, _) in table.items():
            if status == REFUSED:
                out.append((key, '/some/file.nc' if v == ANY and key in (
                    'FORCING2', 'INIT_STATE') else ('TRUE' if v == ANY
                                                    else v)))
    return out


README_BEGIN, README_END = '<!-- options:begin -->', '<!-- options:end -->'


def readme_block(text):
    """`text` (README.md) with its options block replaced by summary()."""
    a = text.index(README_BEGIN) + len(README_BEGIN)
    b = text.index(README_END)
    return text[:a] + '\n' + summary() + '\n' + text[b:]


if __name__ == '__main__':                                  # pragma: no cover
    # python -m torchvic.io.options              -> docs/OPTIONS.md on stdout
    # python -m torchvic.io.options README.md    -> its options block, in place
    import sys
    if len(sys.argv) > 1:
        t = open(sys.argv[1]).read()
        with open(sys.argv[1], 'w') as f:
            f.write(readme_block(t))
    else:
        sys.stdout.write(markdown())
