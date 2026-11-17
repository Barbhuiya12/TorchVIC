#!/usr/bin/env python3
"""Build ISC (474x507, 1/16 deg) D8 flow-direction maps from fine-resolution sources.

  fdir_sources.py --src hs_eam|hs_ihu|hs_trace|merit_trace --out X.nc

Sources (downloaded data is untrusted: only read as raw arrays / by rasterio, never executed):
  hs_*         HydroSHEDS v1 15 arcsec (<data root>/static/hydrosheds_v1/as_dir_15s.bil, raw BIL, 8 bit).
               15 arcsec x 15 = 1/16 deg, exactly aligned with the ISC grid.
               hs_eam / hs_ihu: pyflwdir.upscale(15, method='eam'|'ihu') (Eilander et al. 2021).
               hs_trace: outlet-cell tracing (below), same fine map.
  merit_trace  MERIT-Hydro IHU 30 arcsec (<data root>/static/merit_ihu/30sec_flwdir.tif).
               1/16 deg = 7.5 cells of 30 arcsec, so pyflwdir.upscale (integer factors only) cannot
               reach the ISC grid; the outlet-cell tracing is used instead.
Outlet-cell tracing: per ISC cell take the fine cell with the largest upstream area and follow
its flow path until it enters another ISC cell; that neighbour is the D8 target.
Common finish: loops broken at the weakest cell, unassigned mask cells filled by priority flood
(tools/build_isc_fdir_grit.py), then upstream area.  Directions use the project convention of
build_isc_fdir_grit.ESRI (row index increases with latitude).

Python needs numpy, scipy, netCDF4, pyflwdir, rasterio (scratch venv; see tools/requirements_fdir.txt).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_isc_fdir_grit as bg  # noqa: E402

ROOT = os.environ.get('TORCHVIC_DATA_ROOT', 'data')
LAT0, LAT1, LON0, LON1 = 5.0, 40.0, 60.0, 100.0           # download/processing window
# pyflwdir D8 (north-up, dy positive south) -> project code (dr positive north)
PYF = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
LUT = np.full(256, 255, np.uint8)          # 255 = no direction (ocean / missing)
for _p, (_dy, _dx) in PYF.items():
    LUT[_p] = bg.CODE[(-_dy, _dx)]
LUT[0] = 0                                  # pit / sink


def read_hs15():
    """HydroSHEDS Asia 15 s window (north-up) as D8 codes; ocean (255) -> 247 (missing)."""
    d = ROOT + '/static/hydrosheds_v1'
    hdr = dict(l.split() for l in open(d + '/as_dir_15s.hdr') if l.strip())
    nr, nc = int(hdr['NROWS']), int(hdr['NCOLS'])
    ulx, uly, dx = float(hdr['ULXMAP']), float(hdr['ULYMAP']), float(hdr['XDIM'])
    x0, y0 = ulx - dx / 2, uly + dx / 2                      # upper-left corner of the raster
    mm = np.memmap(d + '/as_dir_15s.bil', np.uint8, 'r', shape=(nr, nc))
    r0, r1 = int(round((y0 - LAT1) / dx)), int(round((y0 - LAT0) / dx))
    c0, c1 = int(round((LON0 - x0) / dx)), int(round((LON1 - x0) / dx))
    a = np.array(mm[r0:r1, c0:c1])
    a[a == 255] = 247
    return a, dx


def read_merit30():
    import rasterio
    with rasterio.open(ROOT + '/static/merit_ihu/30sec_flwdir.tif') as ds:
        t = ds.transform
        dx = t.a
        r0, r1 = int(round((t.f - LAT1) / dx)), int(round((t.f - LAT0) / dx))
        c0, c1 = int(round((LON0 - t.c) / dx)), int(round((LON1 - t.c) / dx))
        a = ds.read(1, window=((r0, r1), (c0, c1)))
    a = np.where(np.isin(a, (0, 1, 2, 4, 8, 16, 32, 64, 128)), a, 247).astype(np.uint8)
    return a, dx


def flw_raster(a, dx):
    import pyflwdir
    from affine import Affine
    tr = Affine(dx, 0, LON0, 0, -dx, LAT1)
    return pyflwdir.from_array(a, ftype='d8', transform=tr, latlon=True, check_ftype=False)


def to_isc(fd_n, lat, lon):
    """North-up coarse array on LAT0..LAT1/LON0..LON1 (project codes) -> ISC window."""
    step = 0.0625
    r_top = int(round((LAT1 - (lat[-1] + step / 2)) / step))
    c_left = int(round((lon[0] - step / 2 - LON0) / step))
    sub = fd_n[r_top:r_top + len(lat), c_left:c_left + len(lon)]
    return sub[::-1].copy()


def upscale_pyflwdir(a, dx, method, lat, lon):
    flw = flw_raster(a, dx)
    ratio = 0.0625 / dx
    assert abs(ratio - round(ratio)) < 1e-6, 'source does not align with 1/16 deg'
    up = flw.upstream_area(unit='km2') if method == 'eam' else None
    fo, _ = flw.upscale(int(round(ratio)), method=method, uparea=up) if up is not None \
        else flw.upscale(int(round(ratio)), method=method)
    arr = fo.to_array('d8')
    return to_isc(LUT[arr], lat, lon)


def upscale_trace(a, dx, lat, lon, maxstep=20000):
    """Outlet-cell tracing onto the ISC grid (works for non-integer ratios)."""
    flw = flw_raster(a, dx)
    ny, nx = a.shape
    up = flw.upstream_area(unit='km2').ravel()
    ds = flw.idxs_ds.astype(np.int64)                      # downstream cell, self for pits
    step = 0.0625
    rr, cc = np.divmod(np.arange(a.size), nx)
    ycen = LAT1 - (rr + 0.5) * dx
    xcen = LON0 + (cc + 0.5) * dx
    ci = np.floor((ycen - (lat[0] - step / 2)) / step).astype(np.int64)      # ISC row (lat ascending)
    cj = np.floor((xcen - (lon[0] - step / 2)) / step).astype(np.int64)
    inside = (ci >= 0) & (ci < len(lat)) & (cj >= 0) & (cj < len(lon))
    cid = np.where(inside, ci * len(lon) + cj, -1)
    ok = np.nonzero(inside & (a.ravel() != 247))[0]
    order = ok[np.lexsort((up[ok], cid[ok]))]                # sorted by cell id, then area
    last = np.r_[cid[order][1:] != cid[order][:-1], True]
    outlet = order[last]                                     # fine outlet per coarse cell
    oc = cid[outlet]
    cur = outlet.copy()
    res = np.full(len(outlet), -3, np.int64)                  # -3 unresolved, -1 left domain/pit
    active = np.arange(len(outlet))
    for _ in range(maxstep):
        nxt = ds[cur[active]]
        stuck = nxt == cur[active]                            # pit / mouth
        c2 = cid[nxt]
        moved = (c2 != oc[active]) & ~stuck
        res[active[stuck]] = -1
        res[active[moved]] = c2[moved]
        left = moved & (c2 < 0)
        res[active[left]] = -1
        keep = ~(stuck | moved)
        cur[active[keep]] = nxt[keep]
        active = active[keep]
        if not len(active):
            break
    out = np.full(len(lat) * len(lon), 255, np.uint8)
    for k in range(len(outlet)):
        t = res[k]
        if t >= 0:
            r, c = divmod(oc[k], len(lon))
            r2, c2 = divmod(t, len(lon))
            out[oc[k]] = bg.CODE.get((r2 - r, c2 - c), 255)
        elif t == -1:
            out[oc[k]] = 0
    return out.reshape(len(lat), len(lon))


def finish(fd, mask, elev, carea_km2, wt=None):
    """Make a raw direction map a valid ISC network: only mask cells, no loops,
    no unassigned cells. Returns fdir (255 outside mask), source flag, upstream area, info."""
    fd = np.where(mask, fd.astype(np.int16), -1)
    fd[fd == 255] = -1
    raw = fd >= 0
    # raw targets leaving the grid are mouths; directions into non-mask cells are mouths too
    if wt is None:
        wt = carea_km2
    nb_tot = 0
    for _ in range(100):
        nb = bg.break_cycles(fd, mask, wt)
        nb_tot += nb
        good = bg.drains_to_terminal(fd, mask)
        before = int((fd[mask] == -1).sum())
        fd = bg.priority_flood_fill(fd, mask, elev, good)
        if nb == 0 and (before == 0 or int((fd[mask] == -1).sum()) == before):
            break
    left = mask & (fd == -1)
    fd[left] = bg.SINK
    up = bg.accumulate(np.where(mask, fd, bg.SINK), mask, carea_km2)
    info = dict(raw_cells=int((raw & mask).sum()), mask_cells=int(mask.sum()), loops_broken=nb_tot,
                filled=int((mask & ~raw).sum()), sinks=int(((fd == bg.SINK) & mask).sum()))
    return np.where(mask, fd, 255).astype(np.uint8), np.where(raw, 1, 2).astype(np.uint8), up, info


def write_nc(path, title, source, lat, lon, mask, fd, flag, up, flag_meanings='outside source filled'):
    import netCDF4
    o = netCDF4.Dataset(path, 'w')
    o.title, o.source = title, source
    o.comment = 'fdir: project D8 (build_isc_fdir_grit.ESRI; row index increases with latitude); 0=sink; 255=outside mask'
    o.createDimension('lat', len(lat)); o.createDimension('lon', len(lon))
    for nme, v in (('lat', lat), ('lon', lon)):
        x = o.createVariable(nme, 'f8', (nme,)); x[:] = v
        x.units = 'degrees_north' if nme == 'lat' else 'degrees_east'
    v = o.createVariable('fdir', 'u1', ('lat', 'lon'), zlib=True); v[:] = fd
    v = o.createVariable('upstream_area', 'f8', ('lat', 'lon'), zlib=True); v[:] = np.where(mask, up, np.nan)
    v.units = 'km2'
    v = o.createVariable('source', 'u1', ('lat', 'lon'), zlib=True); v[:] = np.where(mask, flag, 0)
    v.flag_meanings = flag_meanings
    o.close()


def main():
    import netCDF4
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, choices=['hs_eam', 'hs_ihu', 'hs_trace', 'merit_trace'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--domain', default=ROOT + '/vic_params_isc/domain.nc')
    a = ap.parse_args()
    d = netCDF4.Dataset(a.domain)
    lat, lon = d['lat'][:].data, d['lon'][:].data
    mask = np.asarray(d['mask'][:]) > 0
    elev = np.asarray(d['elev'][:], dtype=np.float64)
    carea = np.asarray(d['area'][:], dtype=np.float64) / 1e6
    if a.src.startswith('hs_'):
        arr, dx = read_hs15()
        raw = upscale_trace(arr, dx, lat, lon) if a.src == 'hs_trace' \
            else upscale_pyflwdir(arr, dx, a.src[3:], lat, lon)
        src = 'HydroSHEDS v1 15s dir (Lehner et al. 2008)'
    else:
        arr, dx = read_merit30()
        raw = upscale_trace(arr, dx, lat, lon)
        src = 'MERIT-Hydro IHU 30s flwdir (Eilander et al. 2021, doi:10.5281/zenodo.7936280)'
    fd, flag, up, info = finish(raw, mask, elev, carea)
    print(a.src, info, flush=True)
    write_nc(a.out, 'ISC 1/16 deg D8 flow directions from ' + src, src + '; ' + a.src + '; tools/fdir_sources.py',
             lat, lon, mask, fd, flag, up)


if __name__ == '__main__':
    main()
