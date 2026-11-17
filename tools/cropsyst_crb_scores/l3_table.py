"""KGE, bias (sim - ref) and RMSE of the 17-layer soil moisture, baseflow and ET of the seven
CRB cases in the coupled run, from the series saved by tests/cropsyst_i12_l3.py --out DIR, against
the reference build's fluxes (daily, 1979-1983). usage: l3_table.py DIR [DIR ...]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tests'))
import numpy as np  # noqa: E402
import cropsyst_i12_l3 as L  # noqa: E402

CASES = L.CASES


def table(d):
    rows = []
    for case in CASES:
        z = np.load(os.path.join(d, 'series_%d.npz' % case))
        fx = L.read_fluxes(case)
        n = min(z['vic_et'].shape[0], len(fx['OUT_EVAP']))
        ire = np.zeros(n)
        if 'ir_evap' in z.files:
            ire[:min(n, len(z['ir_evap']))] = z['ir_evap'][:n]
        sm_ref = sum(fx['OUT_SOIL_MOIST_%d' % i][:n] for i in range(17))
        sm = (z['vic_soil_liq'][:n, 0] + z['vic_soil_ice'][:n, 0]).sum(-1)
        for name, s, r in (('soil moisture (mm)', sm, sm_ref),
                           ('baseflow (mm d-1)', z['vic_baseflow'][:n, 0], fx['OUT_BASEFLOW'][:n]),
                           ('ET (mm d-1)', z['vic_et'][:n, 0] + ire, fx['OUT_EVAP'][:n])):
            sc = L.score(s, r)
            rows.append((case, name, sc['kge'], sc['bias'], sc['bias_pct'], sc['rmse'], sc['ref_mean'],
                         float(s.sum() / 5) if 'baseflow' in name else None,
                         float(r.sum() / 5) if 'baseflow' in name else None))
    return rows


if __name__ == '__main__':
    for d in sys.argv[1:]:
        print('###', d)
        print('| case | variable | KGE | bias (sim - ref) | RMSE | reference mean |')
        print('|---|---|---|---|---|---|')
        for c, n, k, b, bp, r, m, *_ in table(d):
            print('| %d | %s | %.2f | %+.3g (%+.0f%%) | %.3g | %.3g |' % (c, n, k, b, bp, r, m))
