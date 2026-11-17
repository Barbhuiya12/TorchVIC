"""Conservative area-weighted regridding between regular lat/lon grids.

Separable: overlap matrices in sin(lat) (spherical area) and in lon, applied
as Wlat @ X @ Wlon.T with scipy.sparse.  NaN in the source is excluded and
reported as coverage (fraction of the destination cell covered by valid
source).  `mass_check` compares the destination total with an independent
source-side integral (each source cell clipped to the destination box).
"""
import numpy as np
import scipy.sparse as sp
from scipy import ndimage

R_EARTH = 6371007.2  # m


def edges_from_centers(c):
    c = np.asarray(c, float)
    d = np.diff(c)
    e = np.empty(len(c) + 1)
    e[1:-1] = 0.5 * (c[1:] + c[:-1])
    e[0] = c[0] - d[0] / 2
    e[-1] = c[-1] + d[-1] / 2
    return e


def overlap_1d(src_e, dst_e, coord=lambda x: x):
    """Sparse [ndst, nsrc] overlap lengths of ascending edge arrays."""
    s, d = coord(np.asarray(src_e, float)), coord(np.asarray(dst_e, float))
    nd, ns = len(d) - 1, len(s) - 1
    i0 = np.clip(np.searchsorted(s, d[:-1], 'right') - 1, 0, ns - 1)
    i1 = np.clip(np.searchsorted(s, d[1:], 'left') - 1, 0, ns - 1)
    cnt = i1 - i0 + 1
    rows = np.repeat(np.arange(nd), cnt)
    cols = np.concatenate([np.arange(a, b + 1) for a, b in zip(i0, i1)])
    lo = np.maximum(s[cols], d[rows])
    hi = np.minimum(s[cols + 1], d[rows + 1])
    w = np.maximum(hi - lo, 0.0)
    return sp.csr_matrix((w, (rows, cols)), shape=(nd, ns))


def _sin(x):
    return np.sin(np.radians(x))


class Regridder:
    """src/dst given as ascending centre arrays (lat, lon)."""

    def __init__(self, src_lat, src_lon, dst_lat, dst_lon):
        self.se_lat = edges_from_centers(src_lat)
        self.se_lon = edges_from_centers(src_lon)
        self.de_lat = edges_from_centers(dst_lat)
        self.de_lon = edges_from_centers(dst_lon)
        self.Wlat = overlap_1d(self.se_lat, self.de_lat, _sin)
        self.Wlon = overlap_1d(self.se_lon, self.de_lon)
        self.full = (np.asarray(self.Wlat.sum(1)).ravel()[:, None]
                     * np.asarray(self.Wlon.sum(1)).ravel()[None, :])
        # destination area of the part covered by the source grid, m2
        self.cov_area = self.full * np.radians(1.0) * R_EARTH ** 2

    def _apply(self, a):
        return (self.Wlat @ (self.Wlon @ a.T).T)

    def mean(self, x):
        """Area-weighted mean of valid source cells -> (mean, coverage)."""
        x = np.asarray(x, float)
        m = np.isfinite(x)
        num = self._apply(np.where(m, x, 0.0))
        den = self._apply(m.astype(float))
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(den > 0, num / den, np.nan)
            cov = den / np.where(self.full > 0, self.full, np.nan)
        return out, cov

    def src_area(self):
        """Source cell areas m2."""
        dl = np.radians(np.diff(self.se_lon))
        ds = np.diff(_sin(self.se_lat))
        return R_EARTH ** 2 * ds[:, None] * dl[None, :]

    def src_inbox_fraction(self):
        """Fraction of each source cell inside the destination box."""
        lat = np.clip(self.se_lat, self.de_lat[0], self.de_lat[-1])
        lon = np.clip(self.se_lon, self.de_lon[0], self.de_lon[-1])
        fl = np.diff(_sin(lat)) / np.diff(_sin(self.se_lat))
        fo = np.diff(lon) / np.diff(self.se_lon)
        return fl[:, None] * fo[None, :]

    def mass_check(self, x, dst_mean):
        """x: source intensive field, dst_mean: result of mean(x).
        Returns (src_total, dst_total, pct_error), totals = integral x dA
        over the part of the destination box that has valid source."""
        x = np.asarray(x, float)
        m = np.isfinite(x)
        src_tot = np.nansum(x * self.src_area() * self.src_inbox_fraction())
        # source cells clipped to the box; valid only
        dm = np.where(np.isfinite(dst_mean), dst_mean, 0.0)
        _, cov = self.mean(x)
        dst_tot = np.nansum(dm * np.nan_to_num(cov) * self.cov_area)
        pe = 100.0 * (dst_tot - src_tot) / src_tot if src_tot != 0 else 0.0
        return src_tot, dst_tot, pe


def fill_nearest(a, valid=None):
    """Fill non-finite (or ~valid) cells with the nearest valid value."""
    a = np.asarray(a, float)
    bad = ~np.isfinite(a) if valid is None else ~valid
    if not bad.any() or bad.all():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                         return_indices=True)
    return a[tuple(idx)]
