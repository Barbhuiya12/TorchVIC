#!/usr/bin/env python3
"""VIC 5 classic parameter files for the full Indian subcontinent (ISC).

Cells: every cell of the calibrated soil file (the ISC domain, 1/16 deg,
VICGlobal grid and cell numbering) plus the coastal fringe it leaves out:
VICGlobal land cells south of COAST_LAT_MAX, within COAST_CELLS cells of the
sea and connected to the calibrated domain (this takes in the Tamil Nadu /
Kerala tip and the strip of coastal basins along every coast; inland basin
boundaries and Sri Lanka, which is not connected, stay outside).

  soil       calibrated rows as given; added cells take VICGlobal's soil row
  vegetation VICGlobal (global_vegetation_1_16_IGBP.txt)
  snow bands VICGlobal (snowbands_MERIT.txt)
  veg library VICGlobal veglib_nh.txt

A cell with no VICGlobal vegetation or snow-band row (coastal cells that
VICGlobal treats as sea) takes the rows of the nearest cell that has them;
those cells are listed in filled_cells.csv.

    python build_isc_params.py --soil soil_discharge_final_e010.txt \\
        --vicglobal parameters_classic.zip --out DIR
"""
import argparse
import io
import os
import sys
import zipfile

import numpy as np

COAST_LAT_MAX = 24.5      # degrees N: the peninsula, Gujarat and the Bengal delta
COAST_CELLS = 16          # 1/16-degree cells (~100 km) from the sea
GRID = (8.03125, 37.59375, 66.15625, 97.78125)   # the ISC forcing grid (cell centres)
EAST_LON_MAX = 92.3       # Bangladesh-Myanmar border on the coast
SRI_LANKA = (10.0, 79.5)  # south of this latitude and east of this longitude
RES = 0.0625
BBOX = (7.53125, 37.96875, 65.53125, 98.46875)   # cell centres: lat_min, lat_max, lon_min, lon_max


def read_rows(z, member, keep, key_of):
    """{key: text} for rows of a classic file whose key is in keep."""
    out = {}
    with z.open(member) as f:
        for line in io.TextIOWrapper(f):
            k = key_of(line)
            if k in keep:
                out[k] = line if line.endswith('\n') else line + '\n'
    return out


def read_veg(z, member, keep):
    """{gridcell: block of text} from a classic vegetation file."""
    out = {}
    with z.open(member) as f:
        it = io.TextIOWrapper(f)
        for head in it:
            gc, nveg = head.split()[:2]
            gc, nveg = int(gc), int(nveg)
            rows = [next(it) for _ in range(nveg)]
            if gc in keep:
                out[gc] = head + ''.join(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--soil', required=True)
    ap.add_argument('--vicglobal', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z = zipfile.ZipFile(args.vicglobal)

    cal = open(args.soil).read().splitlines(keepends=True)
    cal_gc = [int(l.split()[1]) for l in cal]
    have = set(cal_gc)

    # VICGlobal land in the box, keyed by grid index (lat/lon at cell centres)
    from scipy import ndimage
    rows = {}
    with z.open('Classic/soils_3L_MERIT.txt') as f:
        for line in io.TextIOWrapper(f):
            t = line.split(None, 4)
            la, lo = float(t[2]), float(t[3])
            if BBOX[0] <= la <= BBOX[1] and BBOX[2] <= lo <= BBOX[3]:
                rows[(int(round((la - BBOX[0]) / RES)),
                      int(round((lo - BBOX[2]) / RES)))] = (int(t[1]), la, lo, line)
    ny = int(round((BBOX[1] - BBOX[0]) / RES)) + 1
    nx = int(round((BBOX[3] - BBOX[2]) / RES)) + 1
    land = np.zeros((ny, nx), bool)
    for (i, j) in rows:
        land[i, j] = True
    gc_ij = {v[0]: k for k, v in rows.items()}
    dom = np.zeros_like(land)
    for g in cal_gc:
        if g in gc_ij:
            dom[gc_ij[g]] = True
    lat_g = BBOX[0] + RES * np.arange(ny)
    to_sea = ndimage.distance_transform_edt(land)      # cells to the nearest sea cell
    lon_g = BBOX[2] + RES * np.arange(nx)
    LA, LO = lat_g[:, None], lon_g[None, :]
    cand = (land & ~dom & (to_sea <= COAST_CELLS) & (LA < COAST_LAT_MAX)
            & (LA >= GRID[0] - 1e-6) & (LA <= GRID[1] + 1e-6)
            & (LO >= GRID[2] - 1e-6) & (LO <= min(GRID[3], EAST_LON_MAX))
            & ~((LA < SRI_LANKA[0]) & (LO > SRI_LANKA[1])))
    # 4-connected: a diagonal touch across a strait does not join land
    lab, _ = ndimage.label(dom | cand)
    keep = np.isin(lab, np.unique(lab[dom])) & cand
    add = sorted((rows[(i, j)] for i, j in zip(*np.nonzero(keep))),
                 key=lambda r: r[0])
    tip = {r[0]: r[3] for r in add}
    soil = cal + [r[3] if r[3].endswith('\n') else r[3] + '\n' for r in add]
    cells = cal_gc + [r[0] for r in add]
    lat = np.array([float(l.split()[2]) for l in soil])
    lon = np.array([float(l.split()[3]) for l in soil])
    print('cells: %d calibrated + %d coastal (VICGlobal soil) = %d'
          % (len(cal), len(add), len(cells)))

    want = set(cells)
    veg = read_veg(z, 'Classic/global_vegetation_1_16_IGBP.txt', want)
    snow = read_rows(z, 'Classic/snowbands_MERIT.txt', want,
                     lambda l: int(l.split(None, 1)[0]))
    print('VICGlobal rows found: vegetation %d, snow bands %d' % (len(veg), len(snow)))

    # nearest donor for cells VICGlobal has no row for
    donors = np.array([i for i, c in enumerate(cells) if c in veg and c in snow])
    filled = []
    vtxt, stxt = [], []
    for i, c in enumerate(cells):
        if c in veg and c in snow:
            vb, sb = veg[c], snow[c]
        else:
            d2 = (lat[donors] - lat[i]) ** 2 + (lon[donors] - lon[i]) ** 2
            j = cells[donors[int(np.argmin(d2))]]
            vb = '%d %s' % (c, veg[j].split(None, 1)[1])
            sb = '%d\t%s' % (c, snow[j].split(None, 1)[1])
            filled.append((c, lat[i], lon[i], j))
        vtxt.append(vb)
        stxt.append(sb)

    with open(os.path.join(args.out, 'soil.txt'), 'w') as f:
        f.writelines(soil)
    with open(os.path.join(args.out, 'vegetation.txt'), 'w') as f:
        f.writelines(vtxt)
    with open(os.path.join(args.out, 'snowbands.txt'), 'w') as f:
        f.writelines(stxt)
    with open(os.path.join(args.out, 'veglib.txt'), 'wb') as f:
        f.write(z.read('Classic/veglib_nh.txt'))
    with open(os.path.join(args.out, 'filled_cells.csv'), 'w') as f:
        f.write('gridcell,lat,lon,donor_gridcell\n')
        f.writelines('%d,%.5f,%.5f,%d\n' % r for r in filled)
    with open(os.path.join(args.out, 'added_cells.csv'), 'w') as f:
        f.write('gridcell,lat,lon\n')
        f.writelines('%d,%s,%s\n' % (k, tip[k].split()[2], tip[k].split()[3])
                     for k in sorted(tip))
    print('filled from nearest cell: %d; written to %s' % (len(filled), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
