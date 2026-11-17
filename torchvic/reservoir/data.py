"""Dam data: Global Dam Watch (GDW v1, includes GRanD) reader and placement.

Data: Lehner, B. et al. (2024) Global Dam Watch database v1, licensed CC BY 4.0
(http://creativecommons.org/licenses/by/4.0/); GRanD: Lehner et al. (2011),
Front. Ecol. Environ. 9, 494-502.  The files are read as shipped
(`GDW_barriers_v1_delta.dbf`); the data are not redistributed by this
repository.  Attribution must be given in any output that uses them.

The .dbf is parsed directly (no GIS library is needed; coordinates are the
LAT_RIV/LONG_RIV fields, LAT_DAM/LONG_DAM where those are 0).
"""
import struct

import numpy as np


def _read_dbf(path, fields):
    with open(path, 'rb') as f:
        buf = f.read()
    n, hl, rl = struct.unpack('<IHH', buf[4:12])
    cols, i, off = {}, 32, 1
    while buf[i] != 13:
        name = buf[i:i + 11].split(b'\0')[0].decode()
        cols[name] = (off, buf[i + 16], chr(buf[i + 11]))
        off += buf[i + 16]
        i += 32
    out = {}
    recs = np.frombuffer(buf, dtype=np.uint8, count=n * rl, offset=hl).reshape(n, rl)
    for k in fields:
        o, w, kind = cols[k]
        raw = [bytes(r).decode('latin-1').strip() for r in recs[:, o:o + w]]
        out[k] = (np.array([float(x) if x else np.nan for x in raw])
                  if kind == 'N' else np.array(raw))
    return out


def read_gdw_dams(path, country='India', bbox=None, min_capacity_mcm=0.0):
    """Dams of the GDW barriers file as arrays (missing numeric = nan).

    Returns dict: lat, lon, cap_mcm, year, area_skm, catch_km2, irrigation
    (bool, USE_IRRI set), mean_q_m3s (long-term mean discharge), main_use, name, gdw_id.  Filtered by country and/or
    bbox=(lat0, lat1, lon0, lon1) and a minimum capacity."""
    f = ['GDW_ID', 'RES_NAME', 'DAM_NAME', 'COUNTRY', 'YEAR_DAM', 'CAP_MCM',
         'AREA_SKM', 'CATCH_SKM', 'CATCH_REP', 'DIS_AVG_LS', 'USE_IRRI', 'MAIN_USE', 'LAT_DAM', 'LONG_DAM', 'LAT_RIV', 'LONG_RIV']
    d = _read_dbf(path, f)
    for k in ('YEAR_DAM', 'CAP_MCM', 'AREA_SKM', 'CATCH_SKM', 'CATCH_REP',
              'DIS_AVG_LS'):
        d[k] = np.where(d[k] < 0, np.nan, d[k])          # -99 = missing
    # river-snapped location first; the wall location only where it is missing
    lat = np.where(d['LAT_RIV'] != 0, d['LAT_RIV'], d['LAT_DAM'])
    lon = np.where(d['LONG_RIV'] != 0, d['LONG_RIV'], d['LONG_DAM'])
    d['LAT_DAM'], d['LONG_DAM'] = lat, lon
    m = np.ones(len(lat), bool)
    if country:
        m &= d['COUNTRY'] == country
    if bbox:
        la0, la1, lo0, lo1 = bbox
        m &= ((d['LAT_DAM'] >= la0) & (d['LAT_DAM'] <= la1)
              & (d['LONG_DAM'] >= lo0) & (d['LONG_DAM'] <= lo1))
    m &= np.nan_to_num(d['CAP_MCM'], nan=0.0) >= max(min_capacity_mcm, 1e-9)
    s = lambda k: d[k][m]
    return dict(lat=s('LAT_DAM'), lon=s('LONG_DAM'), cap_mcm=s('CAP_MCM'),
                year=s('YEAR_DAM'), area_skm=s('AREA_SKM'),
                catch_km2=np.where(np.isfinite(s('CATCH_SKM')), s('CATCH_SKM'),
                                   s('CATCH_REP')),
                mean_q_m3s=s('DIS_AVG_LS') / 1000.0, irrigation=(s('USE_IRRI') != '')
                & (s('USE_IRRI') != 'No'), main_use=s('MAIN_USE'),
                name=np.where(s('DAM_NAME') != '', s('DAM_NAME'), s('RES_NAME')),
                gdw_id=s('GDW_ID'))


def place_dams(dams, lat, lon, upstream_area_km2, mask=None, radius=3,
               tol=0.5, dist_penalty=0.05):
    """Snap each dam to a river cell of our flow map by upstream-area matching.

    Candidates are cells within `radius` cells of the dam; the score is
    |ln(A_cell / A_dam)| + dist_penalty * (cell distance), the smallest wins,
    and the dam is placed only if |ln ratio| <= tol (default a factor 1.65).
    Dams without a catchment area use the nearest cell with the largest
    upstream area in the window.  Returns (iy, ix, ok): grid indices, and a
    bool mask of the placed dams (iy, ix are -1 where not placed)."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    A = np.asarray(upstream_area_km2, dtype=float)
    ny, nx = A.shape
    n = len(dams['lat'])
    iy = -np.ones(n, np.int64)
    ix = -np.ones(n, np.int64)
    dy, dx = np.meshgrid(np.arange(-radius, radius + 1),
                         np.arange(-radius, radius + 1), indexing='ij')
    dist = np.hypot(dy, dx).ravel()
    for k in range(n):
        r0 = int(np.argmin(np.abs(lat - dams['lat'][k])))
        c0 = int(np.argmin(np.abs(lon - dams['lon'][k])))
        if (abs(lat[r0] - dams['lat'][k]) > 1.0 or abs(lon[c0] - dams['lon'][k]) > 1.0):
            continue
        rr, cc = r0 + dy.ravel(), c0 + dx.ravel()
        ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        rr, cc, ds = rr[ok], cc[ok], dist[ok]
        a = A[rr, cc]
        good = np.isfinite(a) & (a > 0)
        if mask is not None:
            good &= mask[rr, cc]
        if not good.any():
            continue
        rr, cc, ds, a = rr[good], cc[good], ds[good], a[good]
        ca = dams['catch_km2'][k]
        if np.isfinite(ca) and ca > 0:
            err = np.abs(np.log(a / ca))
            j = int(np.argmin(err + dist_penalty * ds))
            if err[j] > tol:
                continue
        else:
            j = int(np.argmax(a - 1e-6 * ds))
        iy[k], ix[k] = rr[j], cc[j]
    return iy, ix, iy >= 0


def merge_by_cell(dams, iy, ix, ok, year_default=0):
    """Merge dams that landed in one cell (capacity and area add; the earliest
    build year; irrigation if any is).  Returns arrays sorted by cell:
    iy, ix, capacity_m3, build_year, irrigation, area_m2, n_merged."""
    sel = np.nonzero(ok)[0]
    key = iy[sel] * 10**6 + ix[sel]
    u, inv = np.unique(key, return_inverse=True)
    cap = np.zeros(len(u)); area = np.zeros(len(u)); n = np.zeros(len(u), int)
    yr = np.full(len(u), np.inf); irr = np.zeros(len(u), bool)
    for q, k in enumerate(sel):
        g = inv[q]
        cap[g] += dams['cap_mcm'][k] * 1e6
        area[g] += np.nan_to_num(dams['area_skm'][k]) * 1e6
        y = dams['year'][k]
        yr[g] = min(yr[g], y if np.isfinite(y) else year_default)
        irr[g] |= bool(dams['irrigation'][k])
        n[g] += 1
    yr[~np.isfinite(yr)] = year_default
    return u // 10**6, u % 10**6, cap, yr, irr, area, n
