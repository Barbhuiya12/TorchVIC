"""Read UW-Hydro's official Walnut Gulch sample data unchanged.

The point of these tests is that a VIC user's own image-driver set-up loads
with no conversion step. Walnut Gulch is the hardest available case for that
claim, because the officially published files contain two defects -- a wind
variable labelled `mm s-1` and a PARAMETERS line naming a file that is not
there -- and a reader that either crashed on them or silently accepted them
would be wrong in opposite directions.

Data: https://github.com/UW-Hydro/VIC_sample_data (image/WalnutGulch)
"""
import os
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('xarray')

from torchvic.io.global_param import GlobalParam
from torchvic.io.domain import ImageDomain
from torchvic.io.forcing import ImageForcing
from torchvic.io.params import ImageParams

import _paths                                           # noqa: E402
HERE = _paths.REPO
SD = _paths.SAMPLE_DATA
WG = os.path.join(SD, 'image', 'WalnutGulch', 'parameters')
GLOBAL = os.path.join(WG, 'global_param.WalnutGulch.L2015.txt')

pytestmark = pytest.mark.skipif(
    not os.path.exists(GLOBAL),
    reason='VIC sample data not present (docs/downloads/data.md)')


@pytest.fixture(scope='module')
def gp():
    return GlobalParam(GLOBAL, sample_data=SD)


@pytest.fixture(scope='module')
def domain(gp):
    return ImageDomain(gp.domain_file, names=gp.domain_vars)


def test_global_file_declares_three_hourly_steps(gp):
    assert gp.get_int('MODEL_STEPS_PER_DAY') == 8
    assert gp.get_int('SNOW_STEPS_PER_DAY') == 8
    assert gp.get_int('RUNOFF_STEPS_PER_DAY') == 24


def test_forcing_variable_names_come_from_the_file(gp):
    assert gp.force_type == {'AIR_TEMP': 'temp', 'PREC': 'prec',
                             'PRESSURE': 'air_pressure', 'SWDOWN': 'shortwave',
                             'LWDOWN': 'longwave', 'VP': 'vapor_pressure',
                             'WIND': 'wind'}


def test_unsupported_options_are_listed_not_ignored(gp):
    # Silence about an option VIC honours and this model does not is the
    # failure mode; the file must be readable *and* say what was skipped.
    # QUICK_FLUX left the list when QUICK_FLUX FALSE was implemented (the
    # finite-difference profile); AERO_RESIST_CANSNOW is still announced
    assert 'QUICK_FLUX TRUE' not in gp.unsupported
    assert 'AERO_RESIST_CANSNOW' in gp.unsupported
    # LAI_SRC left the list with FORCING2 (FROM_VEGHIST): every value of it
    # is now read, FROM_VEGPARAM from the parameter file (vic_init.c:198-213)
    assert 'LAI_SRC' not in gp.unsupported


def test_sample_data_root_may_be_relative(tmp_path):
    # ${VIC_SAMPLE_DATA} given relative used to be joined onto the global
    # file's own directory, producing a path that exists nowhere.
    rel = os.path.relpath(SD, os.getcwd())
    g = GlobalParam(GLOBAL, sample_data=rel)
    assert os.path.exists(g.domain_file)


def test_domain_is_fifteen_active_cells_at_one_sixteenth_degree(domain):
    import numpy as np
    assert domain.n_active == 15
    assert (domain.ny, domain.nx) == (3, 5)
    # domain.lat is one entry per active cell, so the axis is its unique set.
    lat = np.unique(domain.lat)
    lon = np.unique(domain.lon)
    assert lat.size == 3 and lon.size == 5
    assert abs(float(np.diff(lat)[0]) - 0.0625) < 1e-9
    assert abs(float(np.diff(lon)[0]) - 0.0625) < 1e-9
    assert abs(float(domain.area.sum()) / 1e6 - 616.26) < 0.01


def test_both_parameter_sets_load_and_cover_every_cell(domain):
    import numpy as np
    for tag, nveg in (('L2015', 12), ('MOD_IGBP', 17)):
        p = ImageParams(os.path.join(
            WG, 'params.WalnutGulch.%s.0.0625_deg.nc' % tag), domain)
        assert p.n_layers == 3
        assert p.n_veg == nveg
        assert p.soil['infilt'].shape == (15,)
        assert p.soil['depth'].shape == (3, 15)
        cv = np.asarray(p.veg['Cv'])
        assert cv.shape == (nveg, 15)
        # Every cell must be fully tiled or the water balance loses area.
        assert np.allclose(cv.sum(axis=0), 1.0, atol=1e-6)


def test_missing_parameter_file_names_the_candidates(gp, domain):
    # The shipped global file says `parameters.` where the file is `params.`.
    with pytest.raises(IOError) as e:
        ImageParams(gp.params_file, domain)
    assert 'params.WalnutGulch.L2015' in str(e.value)


@pytest.mark.parametrize('declared,expected', [
    ('W m-2', 'w/m2'), ('mm timestep-1', 'mm/step'), ('m s-1', 'm/s'),
    ('kg kg-1', 'kg/kg'), ('kPa', 'kpa'), ('W/m2', 'w/m2'), ('%', '%'),
])
def test_cf_unit_spellings_fold_onto_one_key(declared, expected):
    assert ImageForcing.normalise_units(declared) == expected


def test_kelvin_is_not_a_multiplier():
    # K -> C is an offset; a 'k': 1.0 entry would pass 273 K through as 273 C.
    for f in ('tair', 'tmax', 'tmin'):
        assert (f, 'k') not in ImageForcing.CONVERT


def test_forcing_reads_three_hourly_with_units_converted(gp, domain):
    fc = ImageForcing(gp.forcing_prefix, domain, force_type=gp.force_type,
                      years=[2008])
    seen = []
    f, t = fc.read(dtype=torch.float64, log=seen.append)
    assert t.size == 2928                      # 366 days x 8 steps, leap year
    for v in f.values():
        assert tuple(v.shape) == (2928, 15)
    # kPa -> Pa, or every energy balance is out by a thousand.
    assert 8.0e4 < float(f['press'].mean()) < 9.0e4
    assert 5.0e2 < float(f['vp'].mean()) < 1.5e3
    # The wind override must be announced, never silent.
    assert any('mm s-1' in m for m in seen)
    assert 0.2 < float(f['wind'].min()) and float(f['wind'].max()) < 20.0
    assert abs(float(f['prcp'].sum(0).mean()) - 414.1) < 1.0


# ---------------------------------------------------------------------------
# Defects that Walnut Gulch itself cannot show.
#
# The sample data is well behaved in exactly the places the reader was weakest:
# every forcing variable carries a units attribute, run_cell equals the mask in
# all fifteen cells, the grid is 3x5 so a transposition cannot go unnoticed for
# the wrong reason, and there is one forcing file. A reader can therefore be
# wrong in all four ways and pass every test above. The files below are built
# in tmp_path to be wrong on purpose, one defect at a time, so that each fix is
# held up by a measurement rather than by the absence of a complaint.
# ---------------------------------------------------------------------------
def _domain_file(tmp_path, mask, name='domain.nc'):
    import numpy as np
    import xarray as xr
    ny, nx = mask.shape
    lat = 31.0 + 0.0625 * np.arange(ny)
    lon = -110.0 + 0.0625 * np.arange(nx)
    ds = xr.Dataset(
        {'mask': (('lat', 'lon'), np.asarray(mask, dtype='int32')),
         'area': (('lat', 'lon'), np.full((ny, nx), 1.0e6)),
         'frac': (('lat', 'lon'), np.ones((ny, nx)))},
        coords={'lat': lat, 'lon': lon})
    p = str(tmp_path / name)
    ds.to_netcdf(p)
    return p


def _params_file(tmp_path, ny, nx, run_cell=None, off_gmt=-7.0, elev=None,
                 transpose=(), name='params.nc', nlayer=3):
    """A minimal VIC parameter file: run_cell, off_gmt and an elevation.

    `transpose` names variables to be written (lon, lat) instead of
    (lat, lon) -- a real hazard on a square grid, where every shape still
    matches and only the numbers move.
    """
    import numpy as np
    import xarray as xr
    lat = 31.0 + 0.0625 * np.arange(ny)
    lon = -110.0 + 0.0625 * np.arange(nx)
    if run_cell is None:
        run_cell = np.ones((ny, nx))
    if elev is None:
        elev = np.arange(ny * nx, dtype='float64').reshape(ny, nx) * 100.0
    v = {'run_cell': np.asarray(run_cell, dtype='float64'),
         'off_gmt': np.full((ny, nx), float(off_gmt)),
         'elev': np.asarray(elev, dtype='float64')}
    data = {}
    for k, arr in v.items():
        if k in transpose:
            data[k] = (('lon', 'lat'), arr.T.copy())
        else:
            data[k] = (('lat', 'lon'), arr)
    depth = np.tile(np.arange(1, nlayer + 1, dtype='float64')[:, None, None],
                    (1, ny, nx))
    data['depth'] = ((('lon', 'lat', 'nlayer'), depth.T.copy())
                     if 'depth' in transpose else
                     (('nlayer', 'lat', 'lon'), depth))
    ds = xr.Dataset(data, coords={'lat': lat, 'lon': lon})
    p = str(tmp_path / name)
    ds.to_netcdf(p)
    return p


def _forcing_file(tmp_path, values, units, times, ny=2, nx=2,
                  name='forc.2008.nc', no_units=()):
    import numpy as np
    import xarray as xr
    nt = len(times)
    data = {}
    for k, val in values.items():
        arr = np.full((nt, ny, nx), np.nan)
        arr[:] = np.asarray(val).reshape((nt, 1, 1))
        data[k] = (('time', 'lat', 'lon'), arr)
    ds = xr.Dataset(data, coords={
        'time': times, 'lat': 31.0 + 0.0625 * np.arange(ny),
        'lon': -110.0 + 0.0625 * np.arange(nx)})
    for k in values:
        if k not in no_units:
            ds[k].attrs['units'] = units[k]
    p = str(tmp_path / name)
    ds.to_netcdf(p)
    return p


def _three_hourly(ndays=2, start='2008-01-01'):
    import numpy as np
    return np.arange(np.datetime64(start + 'T00'),
                     np.datetime64(start + 'T00') +
                     np.timedelta64(24 * ndays, 'h'),
                     np.timedelta64(3, 'h')).astype('datetime64[ns]')


def _simple_domain(tmp_path, ny=2, nx=2):
    import numpy as np
    return ImageDomain(_domain_file(tmp_path, np.ones((ny, nx), dtype=int)))


# -- the unit check ----------------------------------------------------------
def test_a_missing_units_attribute_is_refused_not_waved_through(tmp_path):
    # `if units and canon in self.UNITS:` skipped the whole block when the
    # attribute was absent, so a pressure file stripped of `units` read 85.0
    # and entered the energy balance as 85 Pa instead of 85 kPa -- the exact
    # factor of a thousand the check exists to catch, through its own door.
    t = _three_hourly(1)
    f = _forcing_file(tmp_path, {'air_pressure': [85.0] * len(t)},
                      {'air_pressure': 'kPa'}, t, no_units=('air_pressure',))
    dom = _simple_domain(tmp_path)
    fc = ImageForcing(str(tmp_path / 'forc.'), dom,
                      force_type={'PRESSURE': 'air_pressure'})
    with pytest.raises(ValueError) as e:
        fc.read()
    assert 'units' in str(e.value) and 'press' in str(e.value)
    assert os.path.basename(f) in str(e.value)


def test_a_field_with_no_canonical_unit_is_announced_not_skipped(tmp_path):
    # CHANNEL_IN has no entry in UNITS, so it cannot be checked. Passing it
    # through is defensible; passing it through in silence is not.
    t = _three_hourly(1)
    _forcing_file(tmp_path, {'runoff_in': [1.0] * len(t)},
                  {'runoff_in': 'mm'}, t)
    dom = _simple_domain(tmp_path)
    fc = ImageForcing(str(tmp_path / 'forc.'), dom,
                      force_type={'CHANNEL_IN': 'runoff_in'})
    seen = []
    fields, _ = fc.read(log=seen.append)
    assert 'channel_in' in fields
    assert any('no canonical unit' in m for m in seen), seen


def test_force_type_naming_a_variable_no_file_has_is_refused(tmp_path):
    # `if var not in ds: continue` lost the field without a word; the run then
    # completes with the forcing simply absent.
    t = _three_hourly(1)
    _forcing_file(tmp_path, {'prec': [0.1] * len(t)}, {'prec': 'mm'}, t)
    dom = _simple_domain(tmp_path)
    fc = ImageForcing(str(tmp_path / 'forc.'), dom,
                      force_type={'PREC': 'prec', 'AIR_TEMP': 'temp'})
    with pytest.raises(ValueError) as e:
        fc.read()
    assert 'temp' in str(e.value) and 'FORCE_TYPE' in str(e.value)


@pytest.mark.parametrize('declared', ['mm/day', 'mm d-1', 'kg m-2 s-1'])
def test_precipitation_stated_as_a_rate_is_not_summed_as_a_depth(tmp_path,
                                                                 declared):
    # ('prcp', 'mm/day') sat in CONVERT with a multiplier of 1.0, so eight
    # three-hourly records of 0.5 mm/day summed to 4.0 mm -- eight times the
    # rain the file describes. `mm d-1` was absent from the table and raised
    # instead, so the same file was readable or not depending on spelling.
    from torchvic.io.forcing import aggregate_to_daily
    t = _three_hourly(1)
    rate = 0.5 if declared != 'kg m-2 s-1' else 0.5 / 86400.0
    _forcing_file(tmp_path, {'prec': [rate] * len(t)}, {'prec': declared}, t)
    dom = _simple_domain(tmp_path)
    fc = ImageForcing(str(tmp_path / 'forc.'), dom,
                      force_type={'PREC': 'prec'})
    seen = []
    fields, times = fc.read(log=seen.append)
    assert any('rate' in m for m in seen), seen
    daily, dates = aggregate_to_daily(fields, times, utc_offset=0.0)
    # 0.5 mm over the day, not 0.5 mm eight times over.
    assert abs(float(daily['prcp'][0, 0]) - 0.5) < 1e-9


# -- run_cell, not the mask -------------------------------------------------
def test_the_active_set_comes_from_run_cell_not_from_the_mask(tmp_path):
    # get_global_domain.c:53-78: run_cell is read from the parameter file and
    # counted into ncells_active; the mask only bounds it. Where they differ,
    # selecting on the mask gives a different basin and closes the water
    # balance perfectly while doing it.
    import numpy as np
    mask = np.ones((3, 3), dtype=int)
    run = np.ones((3, 3))
    run[2, :] = 0                       # VIC runs 6 cells, the mask says 9
    d = _domain_file(tmp_path, mask)
    p = _params_file(tmp_path, 3, 3, run_cell=run)
    dom = ImageDomain(d, params=p)
    assert dom.n_active == 6
    assert dom.selected_on == 'run_cell'
    assert set(dom.rows.tolist()) == {0, 1}


def test_a_mask_selected_domain_meeting_a_disagreeing_run_cell_is_refused(
        tmp_path):
    # The older call `ImageDomain(domain_file, names)` cannot see run_cell, so
    # the parameter file is the last place the substitution can be caught.
    import numpy as np
    run = np.ones((3, 3))
    run[2, :] = 0
    d = _domain_file(tmp_path, np.ones((3, 3), dtype=int))
    p = _params_file(tmp_path, 3, 3, run_cell=run)
    dom = ImageDomain(d)
    assert dom.n_active == 9            # the mask's answer
    with pytest.raises(ValueError) as e:
        ImageParams(p, dom)
    assert 'run_cell' in str(e.value) and '6' in str(e.value)


def test_run_cell_outside_the_mask_is_vics_own_error(tmp_path):
    # log_err("Run_cell = 1 should only appear within the mask of the domain
    # file."), get_global_domain.c:64-69.
    import numpy as np
    mask = np.ones((3, 3), dtype=int)
    mask[0, 0] = 0
    run = np.ones((3, 3))               # runs a cell the domain masks out
    d = _domain_file(tmp_path, mask)
    p = _params_file(tmp_path, 3, 3, run_cell=run)
    with pytest.raises(ValueError) as e:
        ImageDomain(d, params=p)
    assert 'mask' in str(e.value) and 'run_cell' in str(e.value)


# -- the trailing axes really are (y, x) ------------------------------------
def test_a_transposed_square_grid_is_refused_not_quietly_misread(tmp_path):
    # take() indexed v[..., rows, cols] and never looked at y_dim/x_dim. On a
    # square grid a (lon, lat) variable then returns the transpose of what was
    # meant: elevations of [0, 600, 1200] down the first column came back as
    # [0, 900, 300], plausible and wrong.
    import numpy as np
    mask = np.zeros((3, 3), dtype=int)
    mask[:, 0] = 1                      # one column, so the transpose shows
    d = _domain_file(tmp_path, mask)
    p = _params_file(tmp_path, 3, 3, run_cell=mask, transpose=('elev',))
    dom = ImageDomain(d)
    with pytest.raises(ValueError) as e:
        ImageParams(p, dom)
    msg = str(e.value)
    assert 'elev' in msg and 'lon' in msg and 'lat' in msg


def test_a_transposed_three_dimensional_variable_is_refused(tmp_path):
    # With nlayer equal to nx the shapes match all the way through and the
    # misreading is total: a (lon, lat, nlayer) depth reads as if it were
    # (nlayer, lat, lon).
    import numpy as np
    d = _domain_file(tmp_path, np.ones((3, 3), dtype=int))
    p = _params_file(tmp_path, 3, 3, transpose=('depth',), nlayer=3)
    dom = ImageDomain(d)
    with pytest.raises(ValueError) as e:
        ImageParams(p, dom)
    assert 'depth' in str(e.value)


def test_the_untransposed_file_still_reads(tmp_path):
    # The check must not cost the ordinary case its readability.
    import numpy as np
    d = _domain_file(tmp_path, np.ones((2, 2), dtype=int))
    p = _params_file(tmp_path, 2, 2)
    par = ImageParams(p, ImageDomain(d))
    assert par.soil['elev'].shape == (4,)
    assert par.soil['depth'].shape == (3, 4)


# -- the day a daily model integrates is the local day -----------------------
def test_daily_binning_can_follow_the_local_day(tmp_path):
    # off_gmt is -7 at Walnut Gulch, so 00:00 UTC is 17:00 the previous
    # afternoon. Binning on the UTC day puts that afternoon's rain on the
    # wrong side of midnight; the annual mean cannot see it and the daily
    # series is shifted by up to 11 C in tmax.
    import numpy as np
    from torchvic.io.forcing import aggregate_to_daily
    t = _three_hourly(2)
    prcp = np.zeros(len(t))
    prcp[0] = 1.0                        # 2008-01-01 00:00 UTC = Dec 31 local
    fields = {'prcp': torch.as_tensor(np.repeat(prcp[:, None], 2, axis=1),
                                      dtype=torch.float64)}
    utc, utc_dates = aggregate_to_daily(fields, t, utc_offset=0.0)
    loc, loc_dates = aggregate_to_daily(fields, t, utc_offset=-7.0)
    assert str(utc_dates[0]) == '2008-01-01'
    assert str(loc_dates[0]) == '2007-12-31'
    assert float(utc['prcp'][0, 0]) == 1.0
    assert float(loc['prcp'][0, 0]) == 1.0
    assert float(loc['prcp'][1, 0]) == 0.0


def test_binning_without_an_offset_says_which_day_it_used(tmp_path):
    from torchvic.io.forcing import aggregate_to_daily
    t = _three_hourly(1)
    fields = {'tair': torch.ones((len(t), 2), dtype=torch.float64)}
    seen = []
    aggregate_to_daily(fields, t, log=seen.append)
    assert any('UTC' in m and 'utc_offset' in m for m in seen), seen


def test_the_utc_offset_is_read_from_the_parameter_file(domain):
    # off_gmt is in the parameter file and was never read; without it a caller
    # has nothing to pass.
    p = ImageParams(os.path.join(
        WG, 'params.WalnutGulch.L2015.0.0625_deg.nc'), domain)
    assert p.utc_offset_hours() == -7.0


def test_a_domain_spanning_time_zones_has_no_single_offset(tmp_path):
    import numpy as np
    d = _domain_file(tmp_path, np.ones((2, 2), dtype=int))
    off = np.full((2, 2), -7.0)
    off[0, 0] = -8.0
    import xarray as xr
    ds = xr.open_dataset(_params_file(tmp_path, 2, 2)).load()
    ds['off_gmt'] = (('lat', 'lon'), off)
    p2 = str(tmp_path / 'params_tz.nc')
    ds.to_netcdf(p2)
    par = ImageParams(p2, ImageDomain(d))
    with pytest.raises(ValueError) as e:
        par.utc_offset_hours()
    assert 'time zone' in str(e.value)


# -- the same weather twice -------------------------------------------------
def test_a_duplicated_forcing_file_is_refused_by_name(tmp_path):
    # The glob sorts and concatenates, and aggregate_to_daily folds the
    # repeated dates with np.unique: on Walnut Gulch a second copy of 2008
    # took precipitation from 414.09 to 828.17 mm while air temperature stayed
    # at 17.197 C and the day count stayed 366. Nothing was said.
    import shutil
    src = os.path.join(SD, 'image', 'WalnutGulch', 'forcings')
    one = [f for f in sorted(os.listdir(src)) if f.endswith('.nc')][0]
    shutil.copy(os.path.join(src, one), str(tmp_path / 'forc.2008.nc'))
    shutil.copy(os.path.join(src, one), str(tmp_path / 'forc.2008.copy.nc'))
    dom = ImageDomain(os.path.join(WG, 'domain.WalnutGulch.0.0625_deg.nc'))
    fc = ImageForcing(str(tmp_path / 'forc.'), dom,
                      force_type={'PREC': 'prec'})
    with pytest.raises(ValueError) as e:
        fc.read()
    msg = str(e.value)
    assert 'forc.2008.nc' in msg and 'forc.2008.copy.nc' in msg


# -- a tmax the file supplied ------------------------------------------------
def test_a_file_supplied_tmax_is_not_silently_replaced(tmp_path):
    # tmax and tmin were recomputed from sub-daily tair unconditionally, so a
    # file declaring tmax = 99 got 7.0 back and its own field vanished.
    import numpy as np
    from torchvic.io.forcing import aggregate_to_daily
    t = _three_hourly(1)
    tair = np.linspace(0.0, 14.0, len(t))
    fields = {
        'tair': torch.as_tensor(np.repeat(tair[:, None], 2, axis=1),
                                dtype=torch.float64),
        'tmax': torch.full((len(t), 2), 99.0, dtype=torch.float64),
        'tmin': torch.full((len(t), 2), -99.0, dtype=torch.float64),
    }
    seen = []
    out, _ = aggregate_to_daily(fields, t, utc_offset=0.0, log=seen.append)
    assert float(out['tmax'][0, 0]) == 99.0
    assert float(out['tmin'][0, 0]) == -99.0
    assert any('tmax' in m and 'NOT recomputed' in m for m in seen), seen


def test_tmax_is_still_derived_when_the_file_has_none(tmp_path):
    import numpy as np
    from torchvic.io.forcing import aggregate_to_daily
    t = _three_hourly(1)
    tair = np.linspace(0.0, 14.0, len(t))
    fields = {'tair': torch.as_tensor(np.repeat(tair[:, None], 2, axis=1),
                                      dtype=torch.float64)}
    out, _ = aggregate_to_daily(fields, t, utc_offset=0.0)
    assert abs(float(out['tmax'][0, 0]) - 14.0) < 1e-9
    assert abs(float(out['tmin'][0, 0]) - 0.0) < 1e-9


def test_a_wrongly_named_directory_is_named_in_the_error(tmp_path):
    """UW-Hydro's FindleyLake global file says `params/` where the published
    directory is `parameters/`. The earlier diagnostic suggested files in
    the directory named -- and there is no such directory, so it had
    nothing to suggest and fell to a bare "does not exist" pointing at a
    place that never was. The sibling directory is the useful fact.
    """
    from torchvic.io.domain import _open
    (tmp_path / 'parameters').mkdir()
    (tmp_path / 'parameters' / 'domain.nc').write_bytes(b'')
    wrong = tmp_path / 'params' / 'domain.nc'
    with pytest.raises(IOError) as e:
        _open(str(wrong))
    msg = str(e.value)
    assert 'params does not exist' in msg
    assert 'parameters does' in msg
    assert 'name the directory wrongly' in msg


FINDLEY_GLOBAL = os.path.join(SD, 'image', 'FindleyLake', 'parameters',
                              'global_param.FindleyLake.txt')


@pytest.mark.skipif(not os.path.exists(FINDLEY_GLOBAL),
                    reason='FindleyLake sample data not present; '
                           'set TORCHVIC_SAMPLE_DATA')
def test_an_unimplemented_option_on_real_data_is_announced_by_name():
    """The property the whole ingest rests on, on the first published dataset
    that exercises it. FindleyLake declares LAKES TRUE and LAKE_PROFILE TRUE;
    the lake model is not implemented. A reader that absorbed those would
    run a lake basin as dry land and say nothing. GlobalParam parses the
    global file without opening the domain, so the file's own wrong
    directory name (params/ for parameters/) does not get in the way here.
    """
    g = GlobalParam(FINDLEY_GLOBAL, sample_data=SD)
    assert 'LAKES TRUE' in g.unsupported
    assert 'LAKE_PROFILE TRUE' in g.unsupported
    assert g.unknown == [], 'a key VIC itself does not have: %s' % g.unknown
    # And what is implemented is not reported as missing.
    assert not any(u.startswith('DOMAIN') or u.startswith('PARAMETERS')
                   or u.startswith('FORCING1') for u in g.unsupported)
