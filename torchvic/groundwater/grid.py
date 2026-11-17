"""Structured groundwater grid with general control-volume geometry.

Topology is a structured block of ``nlay x nrow x ncol`` cells (7-point
stencil), node number ``n = k*nrow*ncol + i*ncol + j`` as MODFLOW 6 DIS
(0-based). Geometry is stored per cell and per face, so the same solver runs
on a MODFLOW 6 DIS grid (rectangular cells, ``delr``/``delc``) and on a
regular latitude-longitude grid where one VIC cell is one model cell (cell
widths shrink with cos(latitude); MODFLOW 6 needs DISU for this geometry and
the validation feeds it exactly these arrays).

Per-face arrays follow MODFLOW 6 connection data (``CL12`` and ``HWVA`` of
DISU; DIS derives them from ``delr``/``delc``):

* x faces, between (i, j) and (i, j+1): ``clx1``/``clx2`` distance from each
  cell centre to the face, ``wx`` face width; shape (nrow, ncol-1).
* y faces, between (i, j) and (i+1, j): ``cly1``/``cly2``, ``wy``; shape
  (nrow-1, ncol).
* vertical faces use the cell area (HWVA of a vertical connection).
"""
from dataclasses import dataclass

import numpy as np

#: Earth radius used by VIC 5 (vic_physical_constants.h, CONST_REARTH), m.
R_EARTH = 6.37122e6


@dataclass
class Grid:
    nlay: int
    nrow: int
    ncol: int
    top: np.ndarray        # (nrow, ncol) top of layer 1
    botm: np.ndarray       # (nlay, nrow, ncol) cell bottoms
    area: np.ndarray       # (nrow, ncol) plan area
    clx1: np.ndarray
    clx2: np.ndarray
    wx: np.ndarray
    cly1: np.ndarray
    cly2: np.ndarray
    wy: np.ndarray
    idomain: np.ndarray    # (nlay, nrow, ncol) >0 active

    @property
    def shape(self):
        return (self.nlay, self.nrow, self.ncol)

    @property
    def nodes(self):
        return self.nlay * self.nrow * self.ncol

    def cell_tops(self):
        """Top of every cell: layer 1 = ``top``, layer k = bottom of k-1."""
        t = np.empty(self.shape)
        t[0] = self.top
        t[1:] = self.botm[:-1]
        return t

    def node(self, k, i, j):
        """Flat 0-based node number(s) of (layer, row, column)."""
        return (np.asarray(k) * self.nrow + np.asarray(i)) * self.ncol \
            + np.asarray(j)

    # ------------------------------------------------------------ builders
    @classmethod
    def dis(cls, nlay, nrow, ncol, delr, delc, top, botm, idomain=None):
        """MODFLOW 6 DIS geometry (``delr`` along columns, ``delc`` along
        rows)."""
        delr = np.broadcast_to(np.asarray(delr, float), (ncol,)).copy()
        delc = np.broadcast_to(np.asarray(delc, float), (nrow,)).copy()
        top = np.broadcast_to(np.asarray(top, float), (nrow, ncol)).copy()
        botm = np.asarray(botm, float)
        if botm.ndim == 1:
            botm = botm[:, None, None]
        botm = np.broadcast_to(botm, (nlay, nrow, ncol)).copy()
        area = delc[:, None] * delr[None, :]
        clx1 = np.broadcast_to(0.5 * delr[None, :-1], (nrow, ncol - 1)).copy()
        clx2 = np.broadcast_to(0.5 * delr[None, 1:], (nrow, ncol - 1)).copy()
        wx = np.broadcast_to(delc[:, None], (nrow, ncol - 1)).copy()
        cly1 = np.broadcast_to(0.5 * delc[:-1, None], (nrow - 1, ncol)).copy()
        cly2 = np.broadcast_to(0.5 * delc[1:, None], (nrow - 1, ncol)).copy()
        wy = np.broadcast_to(delr[None, :], (nrow - 1, ncol)).copy()
        if idomain is None:
            idomain = np.ones((nlay, nrow, ncol), np.int32)
        idomain = np.broadcast_to(np.asarray(idomain, np.int32),
                                  (nlay, nrow, ncol)).copy()
        return cls(nlay, nrow, ncol, top, botm, area, clx1, clx2, wx,
                   cly1, cly2, wy, idomain)

    @classmethod
    def latlon(cls, lat, lon, top, botm, mask=None, area=None,
               radius=R_EARTH):
        """One VIC cell = one model cell on a regular lat/lon grid.

        ``lat`` (nrow,), ``lon`` (ncol,) are cell-centre degrees with uniform
        spacing. Distances are arcs on a sphere of ``radius``: north-south
        spacing ``radius*dlat``, east-west spacing at a row
        ``radius*cos(lat)*dlon``; a y-face width uses the face latitude.
        ``area`` overrides the spherical cell area (pass the VIC domain area
        so recharge volumes equal VIC's runoff volumes).
        ``botm`` is (nrow, ncol) for one layer or (nlay, nrow, ncol).
        """
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        nrow, ncol = lat.size, lon.size
        dlat = np.diff(lat)
        dlon = np.diff(lon)
        if not np.allclose(dlat, dlat[0], rtol=1e-9, atol=0):
            raise ValueError('latlon grid needs uniform latitude spacing')
        if not np.allclose(dlon, dlon[0], rtol=1e-9, atol=0):
            raise ValueError('latlon grid needs uniform longitude spacing')
        dla = np.deg2rad(abs(dlat[0]))
        dlo = np.deg2rad(abs(dlon[0]))
        phi = np.deg2rad(lat)
        botm = np.asarray(botm, float)
        if botm.ndim == 2:
            botm = botm[None]
        nlay = botm.shape[0]
        botm = np.broadcast_to(botm, (nlay, nrow, ncol)).copy()
        top = np.broadcast_to(np.asarray(top, float), (nrow, ncol)).copy()
        dx = radius * np.cos(phi) * dlo                      # (nrow,)
        dy = radius * dla
        if area is None:
            area = (radius ** 2 * dlo
                    * np.abs(np.sin(phi + dla / 2) - np.sin(phi - dla / 2)))
            area = np.broadcast_to(area[:, None], (nrow, ncol)).copy()
        else:
            area = np.asarray(area, float).copy()
        clx1 = np.broadcast_to(0.5 * dx[:, None], (nrow, ncol - 1)).copy()
        clx2 = clx1.copy()
        wx = np.full((nrow, ncol - 1), dy)
        cly1 = np.full((nrow - 1, ncol), 0.5 * dy)
        cly2 = cly1.copy()
        phif = 0.5 * (phi[:-1] + phi[1:])
        wy = np.broadcast_to((radius * np.cos(phif) * dlo)[:, None],
                             (nrow - 1, ncol)).copy()
        if mask is None:
            mask = np.ones((nrow, ncol), bool)
        idomain = np.broadcast_to((np.asarray(mask) > 0).astype(np.int32),
                                  (nlay, nrow, ncol)).copy()
        return cls(nlay, nrow, ncol, top, botm, area, clx1, clx2, wx,
                   cly1, cly2, wy, idomain)

    # ------------------------------------------------------- DISU export
    def disu_arrays(self):
        """Active-cell connection arrays in MODFLOW 6 DISU form.

        Returns ``nodes`` (active flat node numbers in DISU order), ``top``,
        ``bot``, ``area``, ``iac``, ``ja`` (0-based), ``ihc``, ``cl12``,
        ``hwva``: the identical model for MODFLOW 6 (validation only).
        Vectorised over cells; connections per row are sorted by node.
        """
        L, R, C = self.shape
        N = self.nodes
        act = (self.idomain > 0).ravel()
        nodes = np.flatnonzero(act)
        rmap = -np.ones(N, np.int64)
        rmap[nodes] = np.arange(nodes.size)
        tops = self.cell_tops().ravel()
        bots = self.botm.ravel()
        area = np.broadcast_to(self.area, self.shape).ravel()
        kk, ii, jj = np.unravel_index(np.arange(N), self.shape)
        half = 0.5 * (tops - bots)
        z = np.zeros(N)
        # candidate neighbours in increasing node order: up, north, west,
        # self, east, south, down
        cand = []
        ok = kk > 0
        cand.append((np.where(ok, np.arange(N) - R * C, -1), 0, half, area))
        ok = ii > 0
        im = np.clip(ii - 1, 0, max(R - 2, 0))
        cand.append((np.where(ok, np.arange(N) - C, -1), 1,
                     self.cly2[im, jj] if R > 1 else z,
                     self.wy[im, jj] if R > 1 else z))
        ok = jj > 0
        jm = np.clip(jj - 1, 0, max(C - 2, 0))
        cand.append((np.where(ok, np.arange(N) - 1, -1), 1,
                     self.clx2[ii, jm] if C > 1 else z,
                     self.wx[ii, jm] if C > 1 else z))
        cand.append((np.arange(N), -9, z, z))
        ok = jj < C - 1
        jp = np.clip(jj, 0, max(C - 2, 0))
        cand.append((np.where(ok, np.arange(N) + 1, -1), 1,
                     self.clx1[ii, jp] if C > 1 else z,
                     self.wx[ii, jp] if C > 1 else z))
        ok = ii < R - 1
        ip = np.clip(ii, 0, max(R - 2, 0))
        cand.append((np.where(ok, np.arange(N) + C, -1), 1,
                     self.cly1[ip, jj] if R > 1 else z,
                     self.wy[ip, jj] if R > 1 else z))
        ok = kk < L - 1
        cand.append((np.where(ok, np.arange(N) + R * C, -1), 0, half, area))
        m = np.stack([c[0] for c in cand], 1)               # (N, 7)
        valid = (m >= 0) & act[np.clip(m, 0, N - 1)] & act[:, None]
        hcol = np.array([c[1] for c in cand])
        cl = np.stack([c[2] for c in cand], 1)
        hw = np.stack([c[3] for c in cand], 1)
        valid = valid[nodes]
        m, cl, hw = m[nodes], cl[nodes], hw[nodes]
        # DISU wants the diagonal first in each row
        order = [3, 0, 1, 2, 4, 5, 6]
        m, cl, hw, valid = (a[:, order] for a in (m, cl, hw, valid))
        hcol = hcol[order]
        iac = valid.sum(1)
        ja = rmap[m[valid]]
        ihc = np.broadcast_to(hcol, valid.shape)[valid].copy()
        ihc[ihc == -9] = 1
        cl12 = cl[valid]
        hwva = hw[valid]
        return dict(nodes=nodes, top=tops[nodes], bot=bots[nodes],
                    area=area[nodes], iac=iac, ja=ja, ihc=ihc, cl12=cl12,
                    hwva=hwva)
