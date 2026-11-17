#!/usr/bin/env python
"""CropSyst India parameter set (.crp/.mgt/.rot + calendars) for the cropsyst backend.

Output (data, not git): <data root>/cropsyst_india/Database/{Crop,Management,Rotation}/Name,
Rotation/Code/<code>.CS_rotation, cropcodelib.txt, calendars_india_states.csv,
provenance.csv, README.md.   Run with the project venv, python -I.

Method (every override is a row of provenance.csv):
  1. base = a PUBLISHED CropSyst crop file (the CropSyst Database files shipped in
     <data root>/static/upstream/vic-cropsyst-v3-crb), reduced to the keys of the V5 schema
     the torchvic reader accepts (torchvic/crop/cropsyst/schema.py).  Rice, cotton,
     sugarcane, groundnut have only 12-line stubs in that database (RUE_max, TUE only):
     those two numbers are kept, the rest comes from a SURROGATE published file (flagged).
  2. thermal time (emergence, flowering, filling, peak LAI, maturity, base temperature) from the
     WOFOST crop parameter library (van Heemst 1988 tropical sets, IR72 rice; TSUM1/TSUM2/TBASE),
     mapped to CropSyst thermal time: DERIVED, flagged.
  3. Kc, canopy height, rooting depth: FAO-56 (Allen et al. 1998) tables 12 and 22, mid of range.
  4. harvest index: typical literature values (flagged).
  5. calendars: MIRCA-OS 2000 unit calendars for India (dominant planting / maturity month by area).
"""
import os
import re
import sys
import glob
import yaml
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from torchvic.crop.cropsyst.schema import CRP_SCHEMA            # noqa

A = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
UP = A + '/static/upstream'
DB = UP + '/vic-cropsyst-v3-crb/Simulation/Database/Crop/name/'
REACCH = UP + '/vic-cropsyst-v3-crb/REACCH/Database/Crop/'
WOF = A + '/raw/vicwur/wofost_crop_parameters_wofost81/repo/'
MIRCA = glob.glob(A + '/vicwur/isc_work/mirca/bag/*/data/contents/Crop Calendar')[0]
OUT = os.environ.get('CS_INDIA_OUT', A + '/cropsyst_india')
VAR = os.environ.get('CS_INDIA_VARIANT', 'final')
PHYLO = DB + 'Phylogenetic/Plantae/Angiosperms/'

# name: dict(base, stub, wofost (file, variety) or None, mirca names, seasons)
C = {}
def add(name, base, **kw):
    C[name] = dict(base=base, **kw)

WS = DB + 'default/wheat - Spring (Triticum).crp'
SOY = DB + 'default/Soybean (Glycine max).crp'
SORG = DB + 'default/sorghum.crp'
add('rice_kharif', WS, surrogate='wheat - Spring (Triticum)', stub='Rice (Oryza)', wof=('rice', 'Rice_IR72_WS'),
    mirca=('Rice1', 'Rice2', 'Rice3'), season='kharif', fao=(1.20, 1.0, 0.75), hi=0.42, desc='Rice, kharif/wet season (IR72 WS)', c3=True)
add('rice_boro', WS, surrogate='wheat - Spring (Triticum)', stub='Rice (Oryza)', wof=('rice', 'Rice_IR72_DS'),
    mirca=('Rice1', 'Rice2', 'Rice3'), season='rabi', fao=(1.20, 1.0, 0.75), hi=0.42, desc='Rice, rabi/boro/dry season (IR72 DS)', c3=True)
add('wheat', WS, wof=None, lai0=('wheat', 'Arminda'), extra=[('leaf', 'stem_leaf_partition', 2.5, 'published CropSyst REACCH wheat_spring-high.crp and pea_spring-high.crp stem_leaf_partition 2.5 (calibrated spring crops); default file value was 3.0 (wheat) / 9.0 (canola)', 'FLAG: chosen to lift cool-season canopy growth; uncertain')], mirca=('Wheat1', 'Wheat2'), season='rabi', window=(10, 11, 12, 1, 2),
    fao=(1.15, 1.0, 1.65), hi=None, desc='Spring wheat (India rabi)')
add('maize', DB + 'default/Maize (Zea mays).crp', wof=('maize', 'Maize_VanHeemst_1988'), mirca=('Maize',), season='any',
    fao=(1.20, 2.0, 1.35), hi=None, desc='Maize')
add('sorghum', SORG, wof=('sorghum', 'Sorghum_VanHeemst_1988'), mirca=('Sorghum',), season='any',
    fao=(1.05, 1.5, 1.5), hi=0.35, desc='Sorghum')
add('pearl_millet', PHYLO + 'Monocots/Commelinids/Poales/Poaceae/Panicoideae/Pennisetum/glaucum/millet - pearl (Pennisetum glaucum).crp',
    wof=('millet', 'Millet_VanHeemst_1988'), mirca=('Millet',), season='any', fao=(1.00, 1.5, 1.5), hi=0.28,
    desc='Pearl millet')
add('chickpea_pulses', DB + 'default/Lentil (Lens culinaris).crp', surrogate='Lentil (Lens culinaris)',
    wof=('chickpea', 'Chickpea_VanHeemst_1988'), mirca=('Pulses',), season='rabi', fao=(1.15, 0.4, 0.8), hi=0.35,
    desc='Chickpea / pulses (rabi)')
add('sugarcane', SORG, surrogate='sorghum', stub='Sugarcane (Saccharum)', wof=('sugarcane', 'Sugarcane_VanHeemst_1988'),
    mirca=('Sugar cane',), season='any', fao=(1.25, 3.0, 1.6), hi=None, canopy=True, extra=[('growth', 'TUE_at_1pKa_VPD', 3.85, 'CALIBRATED: published stub TUE 7.4 (maize-like) gave 102 t/ha dry cane biomass in 12 months (about 240 t/ha fresh); halved so potential cane is about 100-130 t/ha fresh, the range assumed for Indian potential (no table fetched). Muchow et al. 1994 (Field Crops Res 36:193) and Inman-Bamber 1994 (Field Crops Res 36:41) give sugarcane RUE 1.7-1.8 g/MJ and biomass 40-60 t/ha dm/yr, from memory', 'FLAG: calibration, uncertain')], desc='Sugarcane (plant crop)')
add('cotton', SOY, surrogate='Soybean (Glycine max)', stub='Cotton (Gossypium)', wof=('cotton', 'Cotton_VanHeemst_1988'),
    mirca=('Cotton',), season='any', fao=(1.15, 1.35, 1.35), hi=0.30, desc='Cotton (seed cotton)')
add('groundnut', SOY, surrogate='Soybean (Glycine max)', stub='Peanut (Arachis hypogaea)',
    wof=('groundnut', 'Groundnut_VanHeemst_1988'), mirca=('Groundnuts',), season='any', fao=(1.15, 0.4, 0.75), hi=0.40,
    desc='Groundnut (pods)')
add('mustard_rapeseed', REACCH + 'canola_spring-high.crp', surrogate='REACCH canola_spring-high', stub='Rapeseed (Brassica napus)',
    wof=None, lai0=('rapeseed', 'Oilseed_rape_1004'), extra=[('leaf', 'stem_leaf_partition', 2.5, 'published CropSyst REACCH wheat_spring-high.crp and pea_spring-high.crp stem_leaf_partition 2.5 (calibrated spring crops); default file value was 3.0 (wheat) / 9.0 (canola)', 'FLAG: chosen to lift cool-season canopy growth; uncertain')], mirca=('Rapeseed',), season='rabi', window=(9, 10, 11, 12, 1), fao=(1.15, 0.6, 1.25), hi=0.28,
    desc='Mustard / rapeseed (rabi)')
add('potato', DB + 'default/Potato (Solanum tuberosum).crp', wof=None, lai0=('potato', 'Potato_704'), mirca=('Potatoes',), season='rabi',
    fao=(1.15, 0.6, 0.5), hi=None, desc='Potato (rabi)')
add('soybean', SOY, wof=('soybean', 'Soybean_VanHeemst_1988'), mirca=('Soybeans',), season='kharif',
    fao=(1.15, 0.75, 0.95), hi=0.35, desc='Soybean (kharif)')
STUB_FILES = {'Rice (Oryza)', 'Sugarcane (Saccharum)', 'Cotton (Gossypium)', 'Peanut (Arachis hypogaea)', 'Rapeseed (Brassica napus)'}
FERT_N = {'rice_kharif': 100, 'rice_boro': 100, 'wheat': 120, 'maize': 120, 'sorghum': 60, 'pearl_millet': 40,
          'chickpea_pulses': 20, 'sugarcane': 150, 'cotton': 80, 'groundnut': 20, 'mustard_rapeseed': 80,
          'potato': 120, 'soybean': 20}


# ------------------------------------------------------------------ file model
def parse(path):
    secs, cur = [], None
    for ln in open(path, encoding='latin-1').read().replace('\r', '').split('\n'):
        m = re.match(r'^\[([^\]]+)\]\s*$', ln)
        if m:
            cur = [m.group(1), []]
            secs.append(cur)
        elif cur is not None and '=' in ln:
            k, v = ln.split('=', 1)
            cur[1].append([k, v])
    return secs


def reduce_schema(secs):
    out, seen = [], set()
    for name, kv in secs:
        base = name.split(':')[0]
        if base not in CRP_SCHEMA or base in ('GUI', 'options', 'version') or name in seen:
            continue
        seen.add(name)
        keep, ks = [], set()
        for k, v in kv:
            if k in CRP_SCHEMA[base] and k not in ks:
                ks.add(k)
                keep.append([k, v])
        out.append([name, keep])
    return out


def get(secs, sec, key):
    for n, kv in secs:
        if n == sec:
            for k, v in kv:
                if k == key:
                    return float(re.match(r'\s*([-+0-9.eE]+)', v).group(1))
    return None


def put(secs, sec, key, val, prov, crop, src, flag):
    s = '%.8g' % val if not isinstance(val, str) else val
    for n, kv in secs:
        if n == sec:
            for p in kv:
                if p[0] == key:
                    prov.append((crop, sec, key, p[1].split()[0] if p[1].strip() else '', s, src, flag))
                    p[1] = s
                    return
            kv.append([key, s])
            prov.append((crop, sec, key, '(absent)', s, src, flag))
            return
    secs.append([sec, [[key, s]]])
    prov.append((crop, sec, key, '(absent)', s, src, flag))


def write_crp(path, secs, extra_head):
    L = ['[version]', 'major=4', 'release=18', 'minor=9']
    # [crop] first as in the CRB files
    ordered = sorted(secs, key=lambda s: s[0] != 'crop')
    L = []
    for n, kv in ordered:
        L.append('[%s]' % n)
        L += ['%s=%s' % (k, v) for k, v in kv]
        if n == 'crop':
            L += ['[version]', 'major=4', 'release=18', 'minor=9']
    open(path, 'w').write('\n'.join(L) + '\n')


# ------------------------------------------------------------------ crops
def wofost(w):
    d = yaml.safe_load(open(WOF + w[0] + '.yaml'))['CropParameters']['Varieties'][w[1]]
    return {k: float(d[k][0]) for k in ('TSUMEM', 'TSUM1', 'TSUM2', 'TBASE')}


def build_crops(prov):
    os.makedirs(OUT + '/Database/Crop/Name', exist_ok=True)
    info = {}
    for name, c in C.items():
        secs = reduce_schema(parse(c['base']))
        src_base = 'base file ' + os.path.relpath(c['base'], UP)
        if 'surrogate' in c:
            prov.append((name, '-', '-', '-', '-', 'SURROGATE structure: %s (%s)' % (c['surrogate'], src_base), 'FLAG: not the crop itself'))
        if 'stub' in c:
            stub = parse(DB + 'default/%s.crp' % c['stub'])
            for k in ('RUE_max', 'TUE_at_1pKa_VPD'):
                put(secs, 'growth', k, get(stub, 'growth', k), prov, name,
                    'published CropSyst stub default/%s.crp [growth]' % c['stub'], 'published (stub)')
        put(secs, 'crop', 'description', c['desc'] + ' - India set (TorchVIC)', prov, name, 'this set', '')
        for sec in ('vernalization', 'photoperiod'):
            put(secs, sec, 'enabled', 'false', prov, name, 'no vernalization/photoperiod for India set; CropSyst parameter not tuned', 'FLAG') \
                if name != 'wheat' else None
        if c.get('canopy'):
            put(secs, 'crop', 'harvested_part', 'canopy', prov, name,
                'sugarcane stalk yield approximated by canopy biomass (yield model KIND_CANOPY)', 'FLAG: dry biomass, not fresh cane')
        # thermal time
        e_b, fl_b, pk_b = get(secs, 'phenology', 'emergence'), get(secs, 'phenology', 'flowering'), get(secs, 'phenology', 'peak_LAI')
        fi_b, mat_b = get(secs, 'phenology', 'filling'), get(secs, 'phenology', 'maturity')
        if c['wof']:
            w = wofost(c['wof'])
            E = w['TSUMEM'] if w['TSUMEM'] > 0 else e_b
            fl, mat = E + w['TSUM1'], E + w['TSUM1'] + w['TSUM2']
            ff = (fi_b - fl_b) / (mat_b - fl_b) if mat_b > fl_b else 0.1
            ff = min(max(ff, 0.05), 0.6)
            r = min(max((pk_b - e_b) / (fl_b - e_b), 0.4), 1.0)
            pk = E + r * (fl - E)
            fi = fl + ff * w['TSUM2']
            src = 'WOFOST library wofost81 %s/%s TSUMEM %g TSUM1 %g TSUM2 %g TBASE %g (van Heemst 1988; Bouman et al. 2001 for IR72); mapped E=TSUMEM, flowering=E+TSUM1, maturity=flowering+TSUM2; filling/peak keep base-file proportions' % (
                c['wof'][0], c['wof'][1], w['TSUMEM'], w['TSUM1'], w['TSUM2'], w['TBASE'])
            fg = 'DERIVED (WOFOST thermal sums are not CropSyst GDD; uncertain)'
            for k, v in (('emergence', E), ('flowering', fl), ('peak_LAI', pk), ('filling', fi), ('maturity', mat),
                         ('max_root_depth', fl), ('base_temp', w['TBASE'])):
                put(secs, 'phenology', k, v, prov, name, src, fg)
            dur = get(secs, 'leaf', 'duration')
            put(secs, 'leaf', 'duration', dur * mat / mat_b, prov, name, 'base leaf duration x maturity ratio', fg)
        else:
            prov.append((name, 'phenology', '*', '-', 'base', 'published CropSyst values kept: ' + src_base +
                         ' (no India-type WOFOST set)', 'FLAG: temperate calibration'))
        # seedling GAI from the WOFOST initial leaf area: TDWI x FLTB(0) x SLATB(0) (ha/kg -> m2/m2)
        lv = c.get('lai0') or c['wof']
        if lv:
            y = yaml.safe_load(open(WOF + lv[0] + '.yaml'))['CropParameters']['Varieties'][lv[1]]
            gai0 = float(y['TDWI'][0]) * float(y['FLTB'][0][1]) * float(y['SLATB'][0][1])
            put(secs, 'leaf', 'initial_GAI', gai0, prov, name,
                'WOFOST library %s/%s: TDWI %g kg/ha x FLTB(0) %g x SLATB(0) %g ha/kg (initial leaf area index)' % (
                    lv[0], lv[1], float(y['TDWI'][0]), float(y['FLTB'][0][1]), float(y['SLATB'][0][1])),
                'DERIVED: published CropSyst initial_GAI (0.011-0.03) starts too slowly in the Indian seasons; uncertain')
        for sec_, key_, val_, src_, fl_ in c.get('extra', []):
            put(secs, sec_, key_, val_, prov, name, src_, fl_)
        kc, h, rd = c['fao']
        S = 'FAO-56 (Allen et al. 1998) Table %s, mid of range, from memory'
        put(secs, 'transpiration', 'ET_crop_coef', kc, prov, name, S % '12 Kc mid', 'FLAG: verify vs table')
        put(secs, 'morphology', 'max_canopy_height', h, prov, name, S % '12 height', 'FLAG: verify vs table')
        put(secs, 'root', 'max_root_depth', rd, prov, name, S % '22 rooting depth', 'FLAG: verify vs table')
        if c['hi'] is not None:
            put(secs, 'harvest_index', 'unstressed', c['hi'], prov, name,
                'typical harvest index (Hay 1995 Ann. Appl. Biol. 126:197; Sinclair 1998 Crop Sci. 38:638; Indian field data), from memory',
                'FLAG: uncertain +-20 %')
        write_crp('%s/Database/Crop/Name/%s.crp' % (OUT, name), secs, '')
        info[name] = dict(E=get(secs, 'phenology', 'emergence'), mat=get(secs, 'phenology', 'maturity'),
                          base=get(secs, 'phenology', 'base_temp'))
    return info


# ------------------------------------------------------------------ calendars
def doy15(m):
    return int(sum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][:m - 1]) + 15)


def calendars():
    cal = {s: pd.read_csv('%s/MIRCA-OS_2000_%s.csv' % (MIRCA, s), encoding='latin1') for s in ('ir', 'rf')}
    for s in cal:
        cal[s] = cal[s][cal[s].unit_code // 10000 == 356]
    return cal


def pick(sub, names, season, window):
    s = sub[sub.Crop.isin(names)]
    if window is not None and s.Planting_Month.isin(window).any():
        s = s[s.Planting_Month.isin(window)]
    k = s.Planting_Month.between(4, 9)
    s2 = s[k] if season == 'kharif' else s[~k] if season == 'rabi' else s
    if len(s2) == 0:
        s2 = s
    if len(s2) == 0:
        return None
    g = s2.groupby(['Planting_Month', 'Maturity_Month']).Growing_area.sum()
    (pm, mm), w = g.idxmax(), g.max()
    return int(pm), int(mm), float(w)


def write_mgt(path, name, irrigated, template):
    t = open(template, encoding='latin-1').read().replace('\r', '')
    blocks = re.split(r'(?m)^(?=\[)', t)

    def blk(prefix):
        return [b for b in blocks if b.startswith('[' + prefix)]
    head = [b for b in blocks if not re.match(r'\[(tillage|harvest|fertilization|automatic_irrigation)', b)]
    hv = blk('harvest:1')[0]
    hv = re.sub(r'(?m)^event_offset=.*$', 'event_offset=7', hv)
    fz = blk('fertilization:1')[0]
    n = FERT_N[name]
    fz = re.sub(r'(?m)^NH4_N=.*$', 'NH4_N=%.8f' % n, fz)
    fz = re.sub(r'(?m)^NH4_amount=.*$', 'NH4_amount=%.8f' % n, fz)
    fz = re.sub(r'(?m)^description=.*$', 'description=%g kg N/ha before planting (FLAG: indicative Indian rate)' % n, fz)
    parts = []
    for b in head:
        if b.startswith('[management]'):
            b = re.sub(r'(?m)^description=.*$', 'description=%s %s, India set' % (name, 'irrigated' if irrigated else 'rainfed'), b)
            b = re.sub(r'(?m)^tillage:count=.*$', 'tillage:count=0', b)
            b = re.sub(r'(?m)^automatic_irrigation:count=.*$', 'automatic_irrigation:count=%d' % int(irrigated), b)
        parts.append(b)
    parts += [hv, fz]
    if irrigated:
        ai = blk('automatic_irrigation:1')[0]
        ai = re.sub(r'(?m)^max_allowable_depletion=.*$', 'max_allowable_depletion=0.50000000 FLAG: typical', ai)
        parts.append(ai)
    open(path, 'w').write(''.join(parts))


def main():
    prov = []
    for d in ('Crop/Name', 'Management/Name', 'Rotation/Name', 'Rotation/Code'):
        os.makedirs('%s/Database/%s' % (OUT, d), exist_ok=True)
    info = build_crops(prov)
    cal = calendars()
    tmpl_ir = DB + '../../Management/Name/CT_corn_grain-irrigated.mgt'
    tmpl_rf = DB + '../../Management/Name/CT_wheat_spring-high.mgt'
    rows, codes = [], []
    code = 7001
    for name, c in C.items():
        for sysm in ('ir', 'rf'):
            res = pick(cal[sysm], c['mirca'], c['season'], c.get('window'))
            if res is None:
                res = pick(cal['ir' if sysm == 'rf' else 'rf'], c['mirca'], c['season'], c.get('window'))
            pm, mm, area = res
            tag = '%s_%s' % (name, sysm)
            write_mgt('%s/Database/Management/Name/%s.mgt' % (OUT, tag), name, sysm == 'ir',
                      tmpl_ir if sysm == 'ir' else tmpl_rf)
            sow = doy15(pm)
            open('%s/Database/Rotation/Name/%s.rot' % (OUT, tag), 'w').write(
                '[rotation]\ndescription=%s %s, India (MIRCA-OS 2000 plant month %d, maturity month %d)\nyears=1\nsowing:count=1\n'
                '[version]\nmajor=4\n\n[sowing:1]\nID=%s\nevent_synchronization=relative_date\n'
                'event_date=0%03d (0/0/%d (relative date))\nenabled=true\ndescription=%s\n'
                'crop=../../Crop/Name/%s.crp\nmanagement=../../Management/Name/%s.mgt\n'
                % (c['desc'], sysm, pm, mm, tag, sow, sow, tag, name, tag))
            link = '%s/Database/Rotation/Code/%d.CS_rotation' % (OUT, code)
            if os.path.lexists(link):
                os.remove(link)
            os.symlink('../Name/%s.rot' % tag, link)
            codes.append((code, tag))
            rows.append(dict(code=code, crop=name, system=sysm, plant_month=pm, maturity_month=mm, sow_doy=sow,
                             mirca_area_ha=area, sim_maturity_gdd=info[name]['mat']))
            code += 1
    open(OUT + '/cropcodelib.txt', 'w').write('\n'.join('%d\t%s' % x for x in codes) + '\n')
    pd.DataFrame(rows).to_csv(OUT + '/calendars_india.csv', index=False)
    pd.DataFrame(prov, columns=['crop', 'section', 'key', 'base_value', 'value', 'source', 'flag']).to_csv(
        OUT + '/provenance.csv', index=False)
    print('wrote', len(C), 'crops,', len(codes), 'rotations')


if __name__ == '__main__':
    main()
