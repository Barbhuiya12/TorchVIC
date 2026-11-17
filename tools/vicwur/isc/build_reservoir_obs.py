"""Build <data root>/vicwur/isc/reservoir_obs.nc from open observed/satellite reservoir storage sources.

Sources (all under <data root>/raw/vicwur/reservoir_storage/):
  cwc_bulletin  parsed CSV from parse_cwc_bulletins.py  (weekly, ~2015-present, in-situ, no coordinates)
  grs_zenodo7855477 (zip)  Global Reservoir Storage (GRanD id, monthly 1999-2018, satellite, CC-BY-4.0)
  glolakes      GloLakes v1.0 Landsat+Sentinel2 absolute storage (HydroLAKES id, CC-BY-4.0)
Coordinates for CWC reservoirs and GRS come from GDW v1.0 (match by GRAND_ID or by name+capacity).
Layout: ragged (reservoir, obs). Also writes the GDW match table CSV.
Usage: build_reservoir_obs.py ROOT   (ROOT = the <data root> directory)
Nothing downloaded is executed; files are only read as data.
"""
import csv, difflib, re, struct, sys
from pathlib import Path
import numpy as np
import netCDF4 as nc


def read_dbf(path, keep):
    with open(path, 'rb') as f:
        h = f.read(32)
        n, hl, rl = struct.unpack('<xxxxIHH20x', h)
        fields, off = [], 1
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            nm = d[:11].split(b'\0')[0].decode()
            fields.append((nm, d[16], off))
            off += d[16]
        f.seek(hl)
        for _ in range(n):
            r = f.read(rl)
            if r[:1] == b'*':
                continue
            yield {nm: r[o:o + ln].decode('latin1').strip() for nm, ln, o in fields if nm in keep}


def gdw_india(root):
    p = root / 'vicwur/isc_work/gdw/GDW_v1_0_shp/GDW_barriers_v1_0.dbf'
    keep = {'GDW_ID', 'RES_NAME', 'DAM_NAME', 'ALT_NAME', 'COUNTRY', 'ADMIN_UNIT', 'CAP_MCM', 'LAT_DAM', 'LONG_DAM', 'LAT_RIV', 'LONG_RIV', 'GRAND_ID', 'HYLAK_ID'}
    out = []
    for r in read_dbf(p, keep):
        if r['COUNTRY'] == 'India':
            for k in ('CAP_MCM', 'LAT_DAM', 'LONG_DAM', 'LAT_RIV', 'LONG_RIV'):
                r[k] = float(r[k]) if r[k] not in ('', '-99') else np.nan
            # GRanD dams carry the point in *_RIV; LAT_DAM/LONG_DAM are blank for them
            if not (np.isfinite(r['LAT_DAM']) and np.isfinite(r['LONG_DAM']) and abs(r['LAT_DAM']) > 0):
                r['LAT_DAM'], r['LONG_DAM'] = r['LAT_RIV'], r['LONG_RIV']
            r['GRAND_ID'] = int(float(r['GRAND_ID'])) if r['GRAND_ID'] not in ('', '0') else 0
            r['HYLAK_ID'] = int(float(r['HYLAK_ID'])) if r['HYLAK_ID'] not in ('', '0') else 0
            out.append(r)
    return out


STOP = {'DAM', 'RESERVOIR', 'PROJECT', 'BARRAGE', 'LAKE', 'TANK', 'RES', 'WEIR', 'THE'}


def norm(s):
    s = re.sub(r'[^A-Z0-9 ]', ' ', s.upper())
    return ' '.join(t for t in s.split() if t not in STOP)


def name_variants(s):
    s = s.upper()
    v = {norm(re.sub(r'\(.*?\)', ' ', s))}
    for g in re.findall(r'\((.*?)\)', s):
        v.add(norm(g))
    v.discard('')
    return v


def match_cwc(name, cap_bcm, gdw_index):
    cv = name_variants(name)
    best = (0.0, None)
    for g, gv in gdw_index:
        s = max((difflib.SequenceMatcher(None, a, b).ratio() for a in cv for b in gv), default=0)
        gc = g['CAP_MCM']
        capok = np.isfinite(gc) and gc > 0 and 0.3 <= gc / (cap_bcm * 1000) <= 3.5
        s += 0.15 if capok else -0.25
        if s > best[0]:
            best = (s, g)
    return best


def main(root):
    root = Path(root)
    raw = root / 'raw/vicwur/reservoir_storage'
    gdw = gdw_india(root)
    gdw_index = [(g, {x for x in (name_variants(g['DAM_NAME']) | name_variants(g['RES_NAME']) | name_variants(g['ALT_NAME'])) if x}) for g in gdw]
    by_grand = {g['GRAND_ID']: g for g in gdw if g['GRAND_ID']}
    by_hylak = {g['HYLAK_ID']: g for g in gdw if g['HYLAK_ID']}
    res, obs = [], []  # res: dict; obs: (res_idx, days since 1900, storage_bcm)
    epoch = np.datetime64('1900-01-01')
    match_rows = []

    def add(r, series):
        res.append(r)
        for d, v in series:
            obs.append((len(res) - 1, int((np.datetime64(d) - epoch) / np.timedelta64(1, 'D')), v))
        match_rows.append([r['source'], r['source_id'], r['name'], r['lat'], r['lon'], r['capacity_bcm'], r['gdw_id'], r['grand_id'], r['match_method'], r['match_score']])

    # CWC bulletins
    cw = raw / 'cwc_bulletin/cwc_storage_tidy.csv'
    if cw.exists():
        ser = {}
        for row in csv.DictReader(open(cw)):
            ser.setdefault(row['reservoir'], []).append(row)
        for name, rows in sorted(ser.items()):
            cap = float(np.median([float(x['capacity_bcm']) for x in rows]))
            sc, g = match_cwc(name, cap, gdw_index)
            ok = g is not None and sc >= 0.85
            r = dict(source='CWC weekly bulletin', source_id=name, name=name, capacity_bcm=cap,
                     lat=g['LAT_DAM'] if ok else np.nan, lon=g['LONG_DAM'] if ok else np.nan,
                     gdw_id=int(g['GDW_ID']) if ok else 0, grand_id=g['GRAND_ID'] if ok else 0,
                     match_method='name+capacity' if ok else 'unmatched', match_score=round(sc, 3))
            dd = {x['date']: float(x['live_storage_bcm']) for x in rows}
            add(r, sorted(dd.items()))
    # GRS (GRanD-id keyed)
    gz = raw / 'grs_zenodo7855477/Global_Reservoir_Storage_dataset.zip'
    if gz.exists():
        import io, zipfile
        z = zipfile.ZipFile(gz)
        for gid, g in sorted(by_grand.items()):
            fn = f'Global_Reservoir_Storage_dataset/{gid}.csv'
            if fn not in z.namelist():
                continue
            s, cap = [], np.nan
            for row in csv.DictReader(io.TextIOWrapper(z.open(fn))):
                if not row['Storage(km3)'] or not row['Normalized Storage (%)']:
                    continue
                v, nrm = float(row['Storage(km3)']), float(row['Normalized Storage (%)'])
                s.append((row['Month'], v))
                if nrm > 0:
                    cap = v / (nrm / 100)
            r = dict(source='GRS (Zenodo 7855477)', source_id=str(gid), name=g['DAM_NAME'] or g['RES_NAME'], capacity_bcm=cap,
                     lat=g['LAT_DAM'], lon=g['LONG_DAM'], gdw_id=int(g['GDW_ID']), grand_id=gid, match_method='GRAND_ID', match_score=1.0)
            add(r, s)
    # GloLakes Landsat+Sentinel2 (HydroLAKES id)
    gl = raw / 'glolakes/v1.0_abs_LandsatPlusSentinel2_1984-present.nc'
    if gl.exists():
        d = nc.Dataset(gl)
        t0 = np.datetime64('1984-01-01')
        days = np.array(d['time'][:], dtype=int)
        dates = [str(t0 + np.timedelta64(int(x), 'D')) for x in days]
        cn = [str(x) for x in d['country_name'][:]]
        st = d['lake_storage'][:]
        for i, c in enumerate(cn):
            if 'India' not in c:
                continue
            col = np.ma.filled(st[:, i].astype(float), np.nan) / 1000.0  # file unit is MCM (storage_unit attr) -> BCM
            ok = np.isfinite(col)
            if ok.sum() < 12:
                continue
            hid = int(d['ID'][i])
            g = by_hylak.get(hid)
            r = dict(source='GloLakes v1.0 Landsat+Sentinel2', source_id=str(hid), name=str(d['lake_name'][i]) or (g['DAM_NAME'] if g else ''),
                     capacity_bcm=float(np.nanmax(col)), lat=float(d['latitude'][i]), lon=float(d['longitude'][i]),
                     gdw_id=int(g['GDW_ID']) if g else 0, grand_id=g['GRAND_ID'] if g else 0,
                     match_method='HYLAK_ID' if g else 'unmatched', match_score=1.0 if g else 0.0)
            add(r, [(dates[k], float(col[k])) for k in np.where(ok)[0]])
    # write
    out = root / 'vicwur/isc/reservoir_obs.nc'
    ds = nc.Dataset(out, 'w')
    ds.title = 'Open observed/satellite reservoir storage for Indian reservoirs (ragged layout)'
    ds.history = 'built by tools/vicwur/isc/build_reservoir_obs.py'
    ds.licence = 'CWC bulletins: public; GRS and GloLakes: CC-BY-4.0. Storage is live (CWC) or total (GRS, GloLakes) storage, BCM.'
    ds.createDimension('reservoir', len(res))
    ds.createDimension('obs', len(obs))
    ds.createDimension('strlen', 64)
    def sv(name, key, ln=64):
        v = ds.createVariable(name, 'S1', ('reservoir', 'strlen'))
        v[:] = np.array([str(r[key]).encode('ascii', 'replace')[:ln] for r in res], dtype='S64').view('S1').reshape(len(res), 64)
    ds.createVariable('reservoir_id', 'i4', ('reservoir',))[:] = np.arange(len(res))
    for k, nm in (('lat', 'lat'), ('lon', 'lon'), ('capacity_bcm', 'capacity')):
        v = ds.createVariable(nm, 'f8', ('reservoir',), fill_value=np.nan)
        v[:] = [r[k] for r in res]
    ds['lat'].units = 'degrees_north'; ds['lon'].units = 'degrees_east'; ds['capacity'].units = 'BCM'
    for k in ('gdw_id', 'grand_id'):
        ds.createVariable(k, 'i4', ('reservoir',))[:] = [r[k] for r in res]
    sv('name', 'name'); sv('source', 'source'); sv('source_id', 'source_id'); sv('match_method', 'match_method')
    ds.createVariable('obs_reservoir', 'i4', ('obs',))[:] = [o[0] for o in obs]
    v = ds.createVariable('time', 'i4', ('obs',)); v[:] = [o[1] for o in obs]; v.units = 'days since 1900-01-01'; v.calendar = 'standard'
    v = ds.createVariable('storage', 'f4', ('obs',)); v[:] = [o[2] for o in obs]; v.units = 'BCM'
    ds.close()
    with open(root / 'vicwur/isc/reservoir_obs_gdw_match.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'source_id', 'name', 'lat', 'lon', 'capacity_bcm', 'gdw_id', 'grand_id', 'match_method', 'match_score'])
        w.writerows(match_rows)
    from collections import Counter
    print(len(res), 'reservoirs', len(obs), 'obs', dict(Counter(r['source'] for r in res)),
          'unmatched CWC', sum(1 for r in res if r['match_method'] == 'unmatched' and r['source'].startswith('CWC')))


if __name__ == '__main__':
    main(sys.argv[1])
