"""Build a hydraulic test basin for the flood connector from Copernicus GLO-30.

GLO-30 tiles (1 arc-second, COG GeoTIFF, public S3 bucket copernicus-dem-30m)
are cut to a lon/lat box of whole VIC 1/16-degree cells and resampled to a
local metric grid (equirectangular at the box's mid latitude; the metres per
degree are stored) with spacing dx. Writes <out>.npz (dem, georeference,
suggested inflow cell) and <out>_vic_pairs.csv (the conservative mapping from
the box's VIC cells, south->west order, to hydraulic cells).

    python make_basin.py --tile Copernicus_DSM_COG_10_N26_00_E091_00_DEM.tif \
        --box 91 92 26 27 --dx 100 --out brahmaputra_guwahati
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torchvic.coupling.connectors.flood import vic_pairs, write_pairs_csv  # noqa

DVIC = 0.0625


def metres_per_degree(lat):
    p = np.radians(lat)
    m_lat = 111132.92 - 559.82 * np.cos(2 * p) + 1.175 * np.cos(4 * p)
    m_lon = 111412.84 * np.cos(p) - 93.5 * np.cos(3 * p)
    return m_lon, m_lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tile', required=True)
    ap.add_argument('--tile-lon0', type=float, required=True)
    ap.add_argument('--tile-lat1', type=float, required=True, help='north edge')
    ap.add_argument('--box', nargs=4, type=float, required=True,
                    metavar=('LON0', 'LON1', 'LAT0', 'LAT1'))
    ap.add_argument('--dx', type=float, default=100.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lon0, lon1, lat0, lat1 = a.box
    for v in a.box:
        assert abs(v / DVIC - round(v / DVIC)) < 1e-9, 'box must follow the VIC lattice'
    z = np.array(Image.open(a.tile), dtype=np.float64)
    npx = z.shape[0]
    pix = 1.0 / npx                       # degrees per pixel
    z = uniform_filter(z, size=3, mode='nearest')
    m_lon, m_lat = metres_per_degree(0.5 * (lat0 + lat1))
    nx = int(np.floor((lon1 - lon0) * m_lon / a.dx))
    ny = int(np.floor((lat1 - lat0) * m_lat / a.dx))
    j = np.arange(nx) + 0.5
    i = np.arange(ny) + 0.5
    lon = lon0 + j * a.dx / m_lon
    lat = lat0 + (ny - i) * a.dx / m_lat
    col = (lon - a.tile_lon0) / pix - 0.5
    row = (a.tile_lat1 - lat) / pix - 0.5
    cc, rr = np.meshgrid(col, row)
    dem = map_coordinates(z, [rr, cc], order=1, mode='nearest')
    meta = dict(dem=dem.astype(np.float64), dx=a.dx, nx=nx, ny=ny, lon0=lon0,
                lat0=lat0, lon1=lon1, lat1=lat1, m_lon=m_lon, m_lat=m_lat)
    # VIC cells of the box, ordered south->north then west->east
    nlo = int(round((lon1 - lon0) / DVIC))
    nla = int(round((lat1 - lat0) / DVIC))
    vlon = np.tile(lon0 + (np.arange(nlo) + 0.5) * DVIC, nla)
    vlat = np.repeat(lat0 + (np.arange(nla) + 0.5) * DVIC, nlo)
    pairs = vic_pairs(meta, vlon, vlat, DVIC)
    # inflow candidate: lowest smoothed cell of the column 3 cells from the east edge
    colm = dem[:, nx - 4]
    r_in = int(np.argmin(uniform_filter(colm, size=9, mode='nearest')))
    meta['inflow_rc'] = np.array([[r_in, nx - 4]])
    meta['vic_lon'], meta['vic_lat'] = vlon, vlat
    np.savez_compressed(a.out + '.npz', **meta)
    write_pairs_csv(a.out + '_vic_pairs.csv', pairs)
    print('grid %d x %d at %g m (%.2f M cells); dem %.1f..%.1f m; %d VIC cells; inflow cell row %d col %d z=%.1f'
          % (ny, nx, a.dx, ny * nx / 1e6, dem.min(), dem.max(), len(vlon), r_in, nx - 4, dem[r_in, nx - 4]))


if __name__ == '__main__':
    main()
