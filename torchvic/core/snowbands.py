"""VIC's sub-grid elevation bands, as an expansion of the cell list.

**Why this is not a detail on the Indian 6 km domain.** 56,073 of its
104,841 cells -- 53.5 per cent -- carry more than one band with non-zero
area, and the elevation spread across a cell's bands reaches 7,841 m.
Running one column at the area-weighted mean elevation collapses all of
that, and snow is the process least willing to be averaged: melt is a
threshold in temperature, so a cell that is half above freezing and half
below does not behave like a cell at the mean.

**What VIC actually does per band**, and it is less than one might fear.
`surface_fluxes.c:385-386` is the whole of it:

    Tair      = force->air_temp[hidx] + soil_con->Tfactor[band];
    step_prec = force->prec[hidx]     * soil_con->Pfactor[band];

Two forcings shifted; everything else is the cell's. The column, the
canopy, the snow pack and the energy balance all run unchanged, and
`vic_run.c:349-422` weights every output by `Cv * AreaFract[band]`.

So bands need no new physics. They need the cell list expanded: one
pseudo-cell per (cell, band) with non-zero area, its air temperature
shifted and its precipitation scaled, run through exactly the machinery a
single-band cell runs through, and the outputs area-weighted back. That is
what this module does, and it is deliberately a module of its own -- it
adds nothing to `basin.simulate` and cannot change a single-band answer.

`vic_init.c:1011-1059` defines the two factors, and both have a wrinkle
worth naming:

  * `Tfactor[j] = (BandElev[j] - elevation) * LAPSE_RATE` with
    LAPSE_RATE = -0.0065. `elevation` there is the **area-weighted mean
    band elevation**, not the file's `elev`, because :1005-1010 overwrites
    the cell's elevation with the band mean and warns while doing it. In
    VIC 5.0.1 that happens through an out-of-range comparison (an upstream
    defect); VIC 5.1.0 does it on purpose (GH#701). Both give the band mean,
    which this project reproduces (`io.build.band_mean_elevation`), and it
    feeds straight into Tfactor, so it has to be reproduced here too or the
    bands are lapsed from the wrong datum.

  * `Pfactor` is read as a precipitation *fraction* -- the share of the
    cell's precipitation falling on that band. :1042-1050 normalises the
    fractions to sum to one, and :1052-1059 then divides each by its own
    `AreaFract`, turning a share of the cell's total into a multiplier on
    the cell's rate. A band holding 10 per cent of the area and 20 per
    cent of the precipitation gets Pfactor 2.0. Skipping that division
    silently halves or doubles the precipitation on every band.
"""
import numpy as np

#: initialize_parameters.c:39. Negative: temperature falls with height.
LAPSE_RATE = -0.0065

#: Below this a band is not run at all, matching `AreaFract[band] > 0` at
#: vic_run.c:187 and the Pfactor guard at vic_init.c:1052-1058.
AREA_EPS = 0.0


def band_factors(area_fract, band_elev, pfactor, cell_elev=None):
    """Tfactor, Pfactor and area weights, from the raw parameter arrays.

    Each argument is [band, cell]; each result is too. The arithmetic is
    vic_init.c:1011-1059 and the order matters -- normalise the fractions,
    then divide by area.

    NaN is read as "no band", which is what the Indian file writes for the
    bands a cell does not have, the same convention its vegetation cover
    uses (see `image_run.build_mosaic`).
    """
    area = np.nan_to_num(np.asarray(area_fract, dtype=np.float64))
    elev = np.asarray(band_elev, dtype=np.float64)
    pf = np.nan_to_num(np.asarray(pfactor, dtype=np.float64))
    if area.shape != pf.shape or area.shape != elev.shape:
        raise ValueError('AreaFract %s, elevation %s and Pfactor %s must '
                         'share [band, cell]'
                         % (area.shape, elev.shape, pf.shape))

    if (area < 0).any():
        raise ValueError('negative snow band area fraction; vic_init.c:980 '
                         'makes this a fatal error, not a warning')
    if (pf < 0).any():
        raise ValueError('negative snow band precipitation fraction; '
                         'vic_init.c:1049 makes this a fatal error, not a '
                         'warning')

    # vic_init.c:988-996 -- when the fractions do not sum to one, VIC
    # divides each by the sum and warns. The Indian 6 km file needs it:
    # 165 of its 104,841 cells sum to less than one, the worst to 0.506.
    # Without this the area-weighted mean is not a mean -- a uniform field
    # contracts to 0.506 instead of 1 -- which is how the omission was
    # found, by asserting exactly that on the real domain.
    asum = area.sum(axis=0)
    if (~(asum > 0)).any():
        raise ValueError('a cell with no snow band area at all; '
                         'vic_init.c:1016 makes a zero area sum a fatal '
                         'error, not a warning')
    area = np.where(asum > 0, area / np.where(asum > 0, asum, 1.0), area)

    live = area > AREA_EPS
    if ((pf > 0) & ~live).any():
        raise ValueError('a snow band with no area carries precipitation; '
                         'vic_init.c:1058-1059 makes this a fatal error, not '
                         'a warning')
    # vic_init.c:1005-1010 -- the cell's elevation IS the area-weighted band
    # mean, replacing the file's own `elev`. Reproduced, not corrected.
    wsum = area.sum(axis=0)
    mean_elev = np.where(wsum > 0,
                         (np.where(live, elev, 0.0) * area).sum(axis=0)
                         / np.where(wsum > 0, wsum, 1.0),
                         np.nan_to_num(elev[0]))
    if cell_elev is not None:
        # The classic driver tests the band mean against the cell's
        # elevation and replaces it only when they differ by more than 1 m
        # (read_snowband.c:79-85); the image driver always replaces: VIC
        # 5.0.1's test reads one band past the end (vic_init.c:1003-1007),
        # 5.1.0's compares with the band mean itself (tolerance 1e-20).
        ce = np.asarray(cell_elev, dtype=np.float64)
        mean_elev = np.where(np.abs(mean_elev - ce) > 1.0, mean_elev, ce)

    tfactor = np.where(live, (np.nan_to_num(elev) - mean_elev) * LAPSE_RATE,
                       0.0)

    # :1042-1050 normalise the fractions; :1052-1059 divide by area.
    psum = np.where(live, pf, 0.0).sum(axis=0)
    frac = np.where(psum > 0, pf / np.where(psum > 0, psum, 1.0), 0.0)
    pmult = np.where(live & (area > 0), frac / np.where(area > 0, area, 1.0),
                     0.0)
    return tfactor, pmult, np.where(live, area, 0.0), mean_elev


def expand(area_fract, band_elev, pfactor, cell_elev=None):
    """The pseudo-cell list: one entry per (cell, band) with area.

    Returns a dict with, for `m` pseudo-cells drawn from `n` cells:

        cell      [m] which cell each pseudo-cell belongs to
        band      [m] which band
        tfactor   [m] degrees C to add to that band's air temperature
        pfactor   [m] multiplier on that band's precipitation
        weight    [m] AreaFract, what the output is weighted by
        n_cell    n
        mean_elev [n] the elevation VIC substitutes for the cell's

    A single-band domain expands to exactly itself with tfactor 0 and
    pfactor 1, which is the property `tests/test_snowbands.py`
    pins so that turning bands on cannot move a single-band answer.
    """
    tf, pm, area, mean_elev = band_factors(area_fract, band_elev, pfactor,
                                           cell_elev=cell_elev)
    # Cell-major: all of cell 0's bands, then all of cell 1's. np.nonzero
    # on a [band, cell] array gives band-major, which works but makes the
    # pseudo-cell list interleave cells and is harder to reason about when
    # reading a slice of the output.
    band, cell = np.nonzero(area > AREA_EPS)
    order = np.lexsort((band, cell))
    band, cell = band[order], cell[order]
    return dict(cell=cell, band=band,
                tfactor=tf[band, cell], pfactor=pm[band, cell],
                weight=area[band, cell], n_cell=area.shape[1],
                n_band=area.shape[0],
                mean_elev=mean_elev)


def one_band_for_crop(area_fract, band_elev, pfactor, crop_cells,
                      cell_elev):
    """VIC-CropSyst's ONE_SNOWBAND_FOR_CROP (vicNl.c:447-455, "assume one
    snowband for cells that contain crop"): in a cell with a crop,
    AreaFract[0] = 1, Pfactor[0] = 1, Tfactor[0] = 0 and BandElev[0] = the
    cell's elevation, options.SNOW_BAND = 1.

    Arguments are [band, cell] (`cell_elev`, `crop_cells` [cell]); returns
    (area_fract, band_elev, pfactor) of the same shape, the crop cells
    reduced to band 0 (the other bands have no area, and no precipitation
    share, which band_factors requires of a dead band), every other cell as
    given. With no crop cell the values are the inputs'. Band 0 then carries
    the whole cell: area 1, Tfactor 0 (its elevation is the cell's), Pfactor 1.
    """
    a = np.array(area_fract, dtype=np.float64)
    e = np.array(band_elev, dtype=np.float64)
    pf = np.array(pfactor, dtype=np.float64)
    crop = np.asarray(crop_cells, dtype=bool)
    if crop.shape != a.shape[1:]:
        raise ValueError('crop_cells %s must be [cell] %s'
                         % (crop.shape, a.shape[1:]))
    ce = np.asarray(cell_elev, dtype=np.float64)
    a[:, crop] = 0.0
    pf[:, crop] = 0.0
    a[0, crop] = 1.0
    pf[0, crop] = 1.0
    e[0, crop] = ce[crop]
    return a, e, pf


def contract(values, ex, axis=-1):
    """Area-weight pseudo-cell results back onto cells.

    `values` has the pseudo-cell axis at `axis`; the result has the cell
    axis there instead. This is vic_run.c:349-422's `* AreaFract[band]`,
    with the per-tile `Cv` already applied upstream by the mosaic.

    Works on numpy arrays and torch tensors alike -- the model returns
    torch, and the tests check the arithmetic in numpy.
    """
    w = ex['weight']
    n = ex['n_cell']
    cell = ex['cell']
    import torch
    # numpy 2 arrays have a `device` attribute too, so ask for the type.
    if isinstance(values, torch.Tensor):
        v = values.movedim(axis, -1)
        wt = torch.as_tensor(w, dtype=v.dtype, device=v.device)
        idx = torch.as_tensor(cell, dtype=torch.long, device=v.device)
        out = torch.zeros(v.shape[:-1] + (n,), dtype=v.dtype,
                          device=v.device)
        out.index_add_(-1, idx, v * wt)
        return out.movedim(-1, axis)
    v = np.moveaxis(np.asarray(values), axis, -1)
    out = np.zeros(v.shape[:-1] + (n,), dtype=v.dtype)
    np.add.at(out, (Ellipsis, cell), v * w)
    return np.moveaxis(out, -1, axis)


def to_bands(values, ex):
    """[step, pseudo-cell] -> [step, cell, band]: each pseudo-cell's value
    placed on its own (cell, band), zero on the bands a cell does not run
    (VIC's loop skips a band with AreaFract 0 and the output stays at the
    zero it was initialised to). The band axis has the file's band count
    (`n_band`, else the highest live band + 1)."""
    import torch
    nb = int(ex['n_band']) if 'n_band' in ex else int(max(ex['band'])) + 1
    out = torch.zeros(values.shape[:1] + (ex['n_cell'], nb),
                      dtype=values.dtype, device=values.device)
    cell = torch.as_tensor(ex['cell'], dtype=torch.long, device=values.device)
    band = torch.as_tensor(ex['band'], dtype=torch.long, device=values.device)
    out[:, cell, band] = values
    return out


def summary(ex):
    """One line saying what the expansion did, for a run's log."""
    n, m = ex['n_cell'], len(ex['cell'])
    per = np.bincount(ex['cell'], minlength=n)
    multi = int((per > 1).sum())
    return ('  snow bands: %d cell(s) -> %d column(s); %d cell(s) have more '
            'than one band (max %d), air temperature shifted %.2f .. %.2f C'
            % (n, m, multi, int(per.max()) if m else 0,
               float(ex['tfactor'].min()) if m else 0.0,
               float(ex['tfactor'].max()) if m else 0.0))
