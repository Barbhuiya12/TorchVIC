"""A TorchVIC case (VIC 5 classic driver files) from a VIC-CropSyst CRB reference run.

`make_case(run_dir, out_dir)` reads the reference run's vic_control.txt and writes, in out_dir, the
classic global file, the soil file and the veg files of the reference with each CropSyst rotation
tile as its own vegetation class (code*100 + cycle index, Cv = Cv of the rotation / number of
tiles, the library row of the rotation code), as VIC-CropSyst's land units are
(VCS/full_energy.c:255-293, read_vegparam.c). Returns dict(global_file, codes, tiles, cell, lat, lon).
"""
import os
import re


def _read_control(path):
    d = {}
    for ln in open(path):
        s = ln.split('#')[0].split()
        if len(s) >= 2:
            d.setdefault(s[0], s[1:])
    return d


def make_case(run_dir, out_dir, tiles, start=None, end=None, one_band=True, runoff_steps=24, lai_water_factor=0.2):
    """`tiles`: the runtime codes of the crop csv (e.g. [300100, 300101, 300102] or [2002]).
    `runoff_steps`: RUNOFF_STEPS_PER_DAY; VIC 4.1.2 (the reference) runs 24 hourly sub-steps of
    Ksat / 24 in every daily step (runoff.c:394, :528). `lai_water_factor`: the canopy store per
    unit LAI, 0.2 mm in the reference (user_def.h:118, read_vegparam.c:340, full_energy.c:1181)
    where VIC 5's constant is 0.1."""
    os.makedirs(out_dir, exist_ok=True)
    c = _read_control(os.path.join(run_dir, 'vic_control.txt'))
    soil, veglib, vegparam = c['SOIL'][0], c['VEGLIB'][0], c['VEGPARAM'][0]
    lines = [l for l in open(soil) if l.strip()]
    tok = lines[0].split()
    cell, lat, lon = int(tok[1]), float(tok[2]), float(tok[3])
    # veglib: one row per tile code, the library row of the rotation (its code / 100 when the
    # rotation has several tiles, the code itself when the tile code is the library class)
    lib = {}
    head = []
    for ln in open(veglib):
        if ln.startswith('#') or not ln.strip():
            head.append(ln)
            continue
        lib[int(ln.split()[0])] = ln
    rows = []
    for t in tiles:
        base = t // 100 if (t // 100) in lib and t not in lib else t
        row = lib[base].split('\t') if '\t' in lib[base] else lib[base].split(None, 1)
        rest = lib[base].split(None, 1)[1]
        rows.append('%d\t%s' % (t, rest))
    vl = os.path.join(out_dir, 'veglib.txt')
    open(vl, 'w').write(''.join(head) + ''.join(rows))
    vp_lines = [l.split() for l in open(vegparam) if l.strip()]
    ncell = vp_lines[0]
    rz = vp_lines[1]
    n = len(tiles)
    cv0 = float(rz[1])
    with open(os.path.join(out_dir, 'vegparam.txt'), 'w') as f:
        f.write('%s %d\n' % (ncell[0], n))
        for t in tiles:
            f.write('%d %.10g %s\n' % (t, cv0 / n, ' '.join(rz[2:])))
    if start is None:
        start = (int(c['STARTYEAR'][0]), int(c['STARTMONTH'][0]), int(c['STARTDAY'][0]))
    if end is None:
        end = (int(c['ENDYEAR'][0]), int(c['ENDMONTH'][0]), int(c['ENDDAY'][0]))
    g = os.path.join(out_dir, 'global.txt')
    with open(g, 'w') as f:
        w = lambda *a: f.write(' '.join(str(x) for x in a) + '\n')
        cf = os.path.join(out_dir, 'constants.txt')
        open(cf, 'w').write('VEG_LAI_WATER_FACTOR %g\n' % lai_water_factor)
        w('CONSTANTS', cf)
        w('MODEL_STEPS_PER_DAY 1'); w('SNOW_STEPS_PER_DAY 1'); w('RUNOFF_STEPS_PER_DAY %d' % runoff_steps)
        w('STARTYEAR', start[0]); w('STARTMONTH', start[1]); w('STARTDAY', start[2])
        w('ENDYEAR', end[0]); w('ENDMONTH', end[1]); w('ENDDAY', end[2])
        w('CALENDAR PROLEPTIC_GREGORIAN')
        w('NLAYER', c['NLAYER'][0]); w('NODES 3')
        w('FULL_ENERGY FALSE'); w('FROZEN_SOIL FALSE')
        w('FORCING1', c['FORCING1'][0]); w('FORCE_FORMAT ASCII')
        w('FORCE_TYPE PREC'); w('FORCE_TYPE TMAX'); w('FORCE_TYPE TMIN'); w('FORCE_TYPE WIND')
        w('FORCE_STEPS_PER_DAY 1')
        w('FORCEYEAR', c['FORCEYEAR'][0]); w('FORCEMONTH', c['FORCEMONTH'][0])
        w('FORCEDAY', c['FORCEDAY'][0]); w('GRID_DECIMAL 5')
        w('WIND_H', c['WIND_H'][0])
        w('SOIL', soil); w('BASEFLOW ARNO'); w('JULY_TAVG_SUPPLIED TRUE'); w('ORGANIC_FRACT TRUE')
        w('VEGLIB', vl); w('VEGPARAM', os.path.join(out_dir, 'vegparam.txt'))
        w('ROOT_ZONES 3'); w('VEGPARAM_LAI FALSE'); w('LAI_SRC FROM_VEGLIB')
        w('SNOW_BAND', c['SNOW_BAND'][0], c['SNOW_BAND'][1])
        w('RESOLUTION 0.0625')
        w('RESULT_DIR', out_dir); w('LOG_DIR', out_dir)
        w('N_OUTFILES 1'); w('OUTFILE fluxes'); w('AGGFREQ NDAYS 1')
        for v in ('PREC', 'EVAP', 'RUNOFF', 'BASEFLOW', 'SOIL_MOIST', 'SWE', 'TRANSP_VEG',
                  'EVAP_BARE', 'EVAP_CANOP'):
            w('OUTVAR OUT_' + v)
    return dict(global_file=g, tiles=list(tiles), cell=cell, lat=lat, lon=lon, control=c)
