#!/usr/bin/env python
"""Irrigated area-months by state from a monthly Cv forcing directory.

usage: report_calendar.py LABEL CV_DIR [YEAR ...]
Area growing in month m = (Cv class 18 + class 19) x cell area (Mha); area-months =
sum over the 12 months.  States: Natural Earth admin-1 (isc_work/ne_admin1, read
without GIS libraries).  One log line per state and year.
"""
import os
import struct
import sys
import numpy as np
import netCDF4 as nc
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa

STATES = [('Punjab', ['Punjab']), ('Haryana', ['Haryana']),
          ('UP', ['Uttar Pradesh']), ('Bihar', ['Bihar']),
          ('WB', ['West Bengal']), ('AP+TG', ['Andhra Pradesh', 'Telangana']),
          ('TN', ['Tamil Nadu'])]


def read_shp(base):
    """(name, adm0_a3, [rings]) of every polygon of a shapefile."""
    dbf = open(base + '.dbf', 'rb').read()
    nrec, hlen, rlen = struct.unpack('<xxxxIHH', dbf[:12])
    fields, off = [], 32
    while dbf[off] != 0x0D:
        fields.append((dbf[off:off + 11].split(b'\0')[0].decode(), dbf[off + 16]))
        off += 32
    recs = []
    for i in range(nrec):
        r = dbf[hlen + i * rlen: hlen + (i + 1) * rlen]
        p, row = 1, {}
        for nm, ln in fields:
            row[nm] = r[p:p + ln].decode('utf-8', 'replace').replace('\0', '').strip()
            p += ln
        recs.append(row)
    shp = open(base + '.shp', 'rb').read()
    pos, out = 100, []
    for i in range(nrec):
        ln = struct.unpack('>i', shp[pos + 4:pos + 8])[0] * 2
        c = shp[pos + 8:pos + 8 + ln]
        pos += 8 + ln
        if struct.unpack('<i', c[:4])[0] not in (5, 15, 25):
            out.append([])
            continue
        npart, npts = struct.unpack('<ii', c[36:44])
        parts = list(struct.unpack('<%di' % npart, c[44:44 + 4 * npart])) + [npts]
        xy = np.frombuffer(c, '<f8', 2 * npts, 44 + 4 * npart).reshape(-1, 2)
        out.append([xy[parts[k]:parts[k + 1]] for k in range(npart)])
    return [(r['name'], r['adm0_a3'], rings) for r, rings in zip(recs, out)]


def state_masks(g):
    LAT, LON = np.meshgrid(g.lat, g.lon, indexing='ij')
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    ne = read_shp(WORK + '/ne_admin1/ne_10m_admin_1_states_provinces')
    res = {}
    for lab, names in STATES:
        m = np.zeros(len(pts), bool)
        for nm, a3, rings in ne:
            if a3 == 'IND' and nm in names:
                inside = np.zeros(len(pts), bool)
                for ring in rings:       # even-odd: holes (Delhi in Haryana) drop out
                    inside ^= Path(ring).contains_points(pts)
                m |= inside
        res[lab] = m.reshape(g.mask.shape) & g.active
    cm = np.load(WORK + '/country_masks.npz')
    res['India'] = np.array(cm['IND']).astype(bool) & g.active
    return res


def main():
    label, cvdir = sys.argv[1], sys.argv[2]
    years = [int(y) for y in sys.argv[3:]] or [2005]
    g = ISC()
    masks = state_masks(g)
    for y in years:
        with nc.Dataset('%s/cv.%d.nc' % (cvdir, y)) as d:
            cv = np.ma.filled(d['cv'][:], 0.0)
        a = (cv[:, 17] + cv[:, 18]) * g.area / 1e10        # Mha
        pa = cv[:, 18] * g.area / 1e10
        for lab, m in masks.items():
            mo, pm = a[:, m].sum(1), pa[:, m].sum(1)
            log('%s %-7s %d irr+pad Mha by month %s | area-months %.1f Mha-mo; paddy area-months %.1f; Jan %.2f Aug %.2f'
                % (label, lab, y, ' '.join('%.2f' % v for v in mo), mo.sum(), pm.sum(), mo[0], mo[7]))


if __name__ == '__main__':
    main()
