"""COMPUTE_TREELINE, as VIC 5.1.0's classic driver runs it.

VIC's image driver stops on the option (get_global_param.c:968-970); the
classic driver runs it, and so does this module, because TorchVIC's rule
is that nothing stays refused. The reference is the classic driver.

**What the treeline is in VIC, and what it is not.** It is not physics.
`vic_run` never reads `AboveTreeLine`: every tile of every band is
simulated as it would be without the option. The option changes two things
around the physics.

1. *Which bands are above the treeline.* compute_treeline.c:76-85: a band
   is above it when the cell's average July air temperature (the soil
   file's last column, read_soilparam.c:574-586), lapsed to the band
   (`+ Tfactor[band]`), is at most 10 C. A cell whose July temperature is
   -999 is skipped (classic vic_force.c:292-296) and has no band above.
2. *How the outputs are weighted* (put_data.c:68-100, :180-190). In a band
   above the treeline every overstory tile is dropped from the cell's
   output, and each tile that stays (the others, and the bare soil) is
   weighted by 1 / (1 - Cv of the overstory tiles).

Because every output is a sum of tile terms weighted by `Cv * AreaFract *
TreeAdjust`, both are one operation on the cover of the (cell, band)
columns the snow-band expansion already runs: the overstory tiles of an
above-treeline column get cover 0 and the others are scaled. `adjust_cover`
does it; the physics is not touched, which is exactly what VIC does.

3. *The tile VIC adds* (read_vegparam.c:371-433). With more than one snow
   band, a cell whose vegetation is all overstory and sums to 1 has no tile
   that could stand above the treeline, so the vegetation parameter reader
   makes one: 0.001 of the cell goes to bare soil (COMPUTE_TREELINE
   FALSE-less default, AboveTreelineVeg < 0) or to a tile of the
   non-overstory class COMPUTE_TREELINE names (`add_above_treeline_tile`).
   This changes the cover the physics sees by 0.001, as VIC's does.
"""
import numpy as np

#: compute_treeline.c:79: the lapsed July temperature at the treeline, C.
TREELINE_TEMP = 10.0
#: classic vic_force.c:292: a July_Tavg of this turns the treeline off for the cell.
JULY_OFF = -999.0
#: vic_def.h, the floor under a canopy fraction (read_veglib.c:203, bare soil).
MIN_FCANOPY = 0.0001
#: read_vegparam.c:380-398: the cover the above-treeline tile takes.
TILE_COVER = 0.001


def above_treeline(july_tavg, tfactor):
    """Which (cell, band) columns are above the treeline.

    `july_tavg` and `tfactor` are per column (numpy). compute_treeline.c:
    76-85, with the cells VIC skips (July_Tavg == -999) never above.
    """
    july = np.asarray(july_tavg, dtype=np.float64)
    tf = np.asarray(tfactor, dtype=np.float64)
    return (july != JULY_OFF) & ((july + tf) <= TREELINE_TEMP)


def adjust_cover(cv, overstory, above):
    """The effective cover of the columns, put_data.c:76-103 and :180-190.

    `cv` [column, tile] (numpy or torch), `overstory` [column, tile] 0/1,
    `above` [column] bool. A column not above the treeline keeps its cover.
    One above the treeline loses its overstory tiles and scales the rest by
    1 / (1 - the overstory cover); with no cover left (the overstory is all
    of it, put_data.c:95 would divide by zero, but nothing is then summed)
    every tile is 0.
    """
    import torch
    if torch.is_tensor(cv):
        ab = torch.as_tensor(np.asarray(above, dtype=bool), device=cv.device)
        ov = torch.as_tensor(np.asarray(overstory) > 0.5,
                             device=cv.device) & ab[:, None]
        cover = (cv * ov.to(cv.dtype)).sum(1)
        rest = 1.0 - cover
        scale = torch.where(ab & (rest > 0.0),
                            1.0 / torch.where(rest > 0.0, rest,
                                              torch.ones_like(rest)),
                            torch.ones_like(rest))
        return torch.where(ov, torch.zeros_like(cv), cv * scale[:, None])
    cv = np.asarray(cv, dtype=np.float64)
    ab = np.asarray(above, dtype=bool)
    ov = (np.asarray(overstory) > 0.5) & ab[:, None]
    rest = 1.0 - (cv * ov).sum(1)
    scale = np.where(ab & (rest > 0.0), 1.0 / np.where(rest > 0.0, rest, 1.0),
                     1.0)
    return np.where(ov, 0.0, cv * scale[:, None])


def add_above_treeline_tile(par, veg_class=-1, log=None):
    """read_vegparam.c:371-433 on the parameter arrays, in place.

    For each cell whose tiles are all overstory and sum to 1: `veg_class`
    < 0 moves 0.001 of the cover to bare soil (:378-383); otherwise the
    named class, which must exist in the library and have no overstory
    (:411-432), takes 0.001 as a tile with the root zones of the cell's
    last tile (:398-409). Every existing tile gives up 0.001 / n
    (:380-381, :391-394). Returns the number of cells changed.

    Bare soil is the last class of the cover array, as VIC's library
    ends with it (read_veglib.c) and as the classic reader lays it out.
    """
    cv = par.veg['Cv']                                    # [class, cell]
    ov = np.asarray(par.veg['overstory'])                 # [class, cell]
    n_class, n_cell = cv.shape
    bare = n_class - 1
    ids = list(getattr(par, 'veg_class_ids', range(n_class)))
    cls = None
    if veg_class >= 0:
        if veg_class not in ids:
            raise ValueError(
                'the vegetation class id %d defined for above-treeline is '
                'not defined in the vegetation library file '
                '(read_vegparam.c:421-425)' % veg_class)
        cls = ids.index(veg_class)
        if float(np.asarray(ov[cls]).max()) > 0.5:
            raise ValueError(
                'vegetation class %d is defined to have overstory, so it '
                'cannot be the default vegetation type for above canopy '
                'snow bands (read_vegparam.c:428-432)' % veg_class)
    changed = 0
    if cls is not None and 'fcanopy' not in par.veg:
        # FCAN_SRC FROM_DEFAULT: the library's values (1, bare soil
        # MIN_FCANOPY), which the classic tile arrays hold for every tile
        # but this one (below). Kept as an explicit array so that tile can
        # differ; the caller then reads it as FROM_VEGLIB.
        dflt = getattr(par, 'fcanopy_default', None)
        if dflt is None:
            dflt = np.ones((n_class, 12, n_cell))
            dflt[bare] = MIN_FCANOPY
        par.veg['fcanopy'] = np.array(dflt, dtype=np.float64)
        par.fcanopy_from_default = True
    for i in range(n_cell):
        tiles = [j for j in range(bare) if cv[j, i] > 0.0]
        if not tiles or any(ov[j, i] < 0.5 for j in tiles):
            continue                                # NoOverstory (:345-348)
        if abs(float(cv[:, i].sum()) - 1.0) > 1e-9 or cv[bare, i] > 0.0:
            continue                                # Cv_sum == 1 (:372)
        n = len(tiles)
        for j in tiles:
            cv[j, i] -= TILE_COVER / n
        if cls is None:
            cv[bare, i] += TILE_COVER
        else:
            last = tiles[-1]
            cells = getattr(par, 'cells', None)
            if cells is not None:
                # the file's last tile line, not the highest class
                order = cells[int(par.gridcell[i])]['tiles']
                last = order[-1]['class_index']
            cv[cls, i] = TILE_COVER
            # read_vegparam.c:392-409 sets the class, the cover and the
            # root zones and nothing else: the tile's monthly albedo, LAI,
            # fcanopy, roughness and displacement stay calloc'd zeros, and
            # vic_force.c:199-212 copies those zeros into the forcing of the
            # tile. Reproduced, not corrected.
            for k in ('LAI', 'albedo', 'veg_rough', 'displacement',
                      'fcanopy'):
                if k in par.veg:
                    par.veg[k][cls, :, i] = 0.0
            for k in ('sigma_slope', 'lag_one', 'fetch'):
                if k in par.veg:
                    par.veg[k][cls, i] = 0.0
            for k in ('root_depth', 'root_fract'):
                if k in par.veg:
                    par.veg[k][cls, :, i] = par.veg[k][last, :, i]
        changed += 1
    if changed and log:
        log('COMPUTE_TREELINE: %d cell(s) have only overstory vegetation; '
            '%s takes %g of each (read_vegparam.c:371-433)'
            % (changed, 'bare soil' if cls is None
               else 'class %d' % veg_class, TILE_COVER))
    return changed
