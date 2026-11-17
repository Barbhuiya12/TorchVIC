"""A VIC dataset gives the same answer whichever driver reads it.

`scripts/compare_vic_drivers.py` established that for VIC itself on Stehekin
(evidence/vic_driver_equivalence.json). This establishes it for TorchVIC's two
readers: `torchvic.io.classic` on the ASCII files and `torchvic.io` on
the NetCDF ones, both feeding `image_run.build_soil / build_mosaic /
build_atmosphere` and `basin.simulate` unchanged.

Three layers, each with a stated tolerance and the reason for it:

* **Parameters.** The classic and image Stehekin files were written from the
  same numbers, so soil, bands and vegetation are required to agree
  *exactly*, except where VIC's own two drivers read the same file
  differently -- and those places are asserted, not excused: fractional cover
  is normalised by the classic driver (read_vegparam.c:628-645) and only
  checked by the image driver (vic_init.c:1231), 1e-6 on Stehekin; the
  bare-soil tile's roughness and displacement come from the soil file's
  `rough` in classic mode (read_soilparam.c:489-494) and from the parameter
  file in image mode (vic_init.c:236-258), where UW-Hydro stored 0; its
  reference wind height is the global file's WIND_H in classic mode
  (read_veglib.c:225) and the file's value, 2 m, in image mode.
* **Forcing.** The ASCII forcing is written to four decimals, so half a unit
  in the fourth place (5e-5 in the file's units, 0.05 Pa after the kPa to Pa
  conversion) is the honest floor, plus float32 rounding of the NetCDF copy
  (2e-7 of the field's magnitude). Measured: the two agree far inside that.
* **Fluxes.** Ten days at 24 steps a day through the same physics. With
  parameters and forcing agreeing as above the totals are required to agree
  to 1e-6 relative and every cell-step to 1e-5 mm; the measured agreement is
  ~1e-7 and better, and the residual is the two documented driver
  differences on the two cells with 1.8% bare soil plus the forcing floor.
  The bare tile's geometry is one of them and the water-balance path reads
  it where VIC does, in a thin pack's sublimation; as read that moves ET by
  2.0e-4 mm on those two cells and nothing else, which is pinned, and the
  fluxes are then compared with the bare tile's geometry made the same.

The cell matching is on latitude and longitude, and what is dropped on
either side is reported. The two Stehekin samples hold the same 16 cells (16
soil rows, 16 forcing files, 16 active image cells); a synthetic mismatch is
tested so the report is known to work when it matters.
"""
import contextlib
import copy
import datetime as dt
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('xarray')

from torchvic.model import basin  # noqa: E402
from torchvic.io import build as image_run
from torchvic.io.classic import (ClassicDomain, ClassicForcing, ClassicOptions,
                                 ClassicParams, MISSING, force_columns,
                                 match_cells, open_classic, read_soil_file,
                                 soil_columns)
from torchvic.io.global_param import GlobalParam
from torchvic.io.domain import ImageDomain
from torchvic.io.forcing import ImageForcing
from torchvic.io.params import ImageParams

import _paths                                           # noqa: E402
REPO = _paths.REPO
SAMPLES = _paths.SAMPLES
CLASSIC_GLOBAL = _paths.STEHE_CLASSIC_GLOBAL
IMAGE_GLOBAL = _paths.STEHE_IMAGE_GLOBAL
SD = _paths.SAMPLE_DATA
FINDLEY_CLASSIC = os.path.join(SD, 'classic', 'FindleyLake')
FINDLEY_IMAGE = os.path.join(SD, 'image', 'FindleyLake')

# The differentiable half runs on the GPU by default; cuda:1 is the card left
# idle on this machine. Override with TORCHVIC_DEVICE.
DEVICE = torch.device(os.environ.get('TORCHVIC_DEVICE') or (
    'cuda:1' if torch.cuda.device_count() > 1 else
    'cuda' if torch.cuda.is_available() else 'cpu'))
DT = torch.float64

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(SAMPLES, 'classic', 'Stehekin')),
    reason='samples/data/classic/Stehekin is not present')


@contextlib.contextmanager
def sample_env():
    """The tests/system global files locate their data through
    $test_data_dir, as VIC's own test harness does."""
    old = os.environ.get('test_data_dir')
    os.environ['test_data_dir'] = SAMPLES
    try:
        yield
    finally:
        if old is None:
            del os.environ['test_data_dir']
        else:
            os.environ['test_data_dir'] = old


@pytest.fixture(scope='module')
def classic():
    with sample_env():
        opts, dom, par, fc = open_classic(CLASSIC_GLOBAL)
        fields, stamps = fc.read(dtype=DT, device=DEVICE)
    return dict(gp=opts.gp, opts=opts, dom=dom, par=par, fc=fc,
                fields=fields, stamps=stamps)


@pytest.fixture(scope='module')
def image():
    with sample_env():
        gp = GlobalParam(IMAGE_GLOBAL)
        dom = ImageDomain(gp.domain_file, names=gp.domain_vars,
                          params=gp.params_file)
        par = ImageParams(gp.params_file, dom)
        fc = ImageForcing(gp.forcing_prefix, dom, force_type=gp.force_type)
        fields, stamps = fc.read(dtype=DT, device=DEVICE)
    return dict(gp=gp, dom=dom, par=par, fc=fc, fields=fields, stamps=stamps)


@pytest.fixture(scope='module')
def matched(classic, image):
    ic, ii, dc, di = match_cells(classic['dom'], image['dom'])
    return ic, ii, dc, di


# ---------------------------------------------------------------------------
# 3. The classic global file is parsed by the same GlobalParam.
# ---------------------------------------------------------------------------
def test_classic_global_keys_are_read_by_the_one_global_param_class():
    with sample_env():
        gp = GlobalParam(CLASSIC_GLOBAL)
    # SOIL/VEGLIB/VEGPARAM land in `raw`, and GlobalParam classifies them
    # from the inventory (evidence/vic5_global_options.json). When this test
    # was written the inventory still said the ASCII files were not read and
    # they came back as unsupported; the classic reader now honours all
    # three and the inventory says so, so they must NOT be reported as
    # unsupported -- a user's classic set-up would otherwise be told its
    # soil file is ignored while it is being read. The two-token SNOW_BAND
    # keeps both tokens.
    assert gp.raw['SNOW_BAND'] == [['5', '$test_data_dir/classic/Stehekin/'
                                    'parameters/Stehekin_snowbands.txt']]
    assert gp.get_int('SNOW_BAND') == 5
    for key in ('SOIL', 'VEGLIB', 'VEGPARAM'):
        assert gp.get(key), key
        assert gp.path_of(key).endswith('.txt')
        assert key not in gp.unsupported, (
            '%s is read by vic_classic but reported unsupported' % key)
    # A one-token FORCE_TYPE line is not a (name, variable) pair, so it goes
    # to `raw` in file order rather than to the force_type dict.
    assert gp.force_type == {}
    assert [r[0] for r in gp.raw['FORCE_TYPE']] == [
        'PREC', 'AIR_TEMP', 'SWDOWN', 'LWDOWN', 'SKIP', 'PRESSURE', 'VP',
        'WIND']
    assert gp.unknown == []
    # ClassicOptions interprets that same object; nothing is parsed twice
    # except the FORCE_TYPE column order, which GlobalParam cannot return.
    opts = ClassicOptions(gp)
    assert opts.gp is gp
    assert opts.force_columns == force_columns(CLASSIC_GLOBAL)
    assert opts.snow_bands == 5 and opts.snowband_file.endswith(
        'Stehekin_snowbands.txt')
    assert opts.lai_src == 'FROM_VEGPARAM' and opts.vegparam_lai
    assert opts.force_start == dt.datetime(1949, 1, 1)
    assert opts.force_steps_per_day == 24 and opts.grid_decimal == 4
    assert opts.refused == []


# ---------------------------------------------------------------------------
# Cell matching, and what is dropped.
# ---------------------------------------------------------------------------
def test_stehekin_holds_the_same_cells_in_both_formats(classic, image,
                                                       matched):
    ic, ii, dc, di = matched
    assert classic['dom'].n_active == 16
    assert image['dom'].n_active == 16
    assert len(classic['fc'].files) == 16
    assert ic.size == 16 and dc == [] and di == []
    np.testing.assert_array_equal(classic['dom'].lat[ic], image['dom'].lat[ii])
    np.testing.assert_array_equal(classic['dom'].lon[ic], image['dom'].lon[ii])


def test_a_soil_cell_outside_the_image_mask_is_reported(classic, image,
                                                        tmp_path):
    # One more soil row, at a coordinate the image mask does not have. The
    # extra classic cell must be named as dropped, and no image cell.
    src = classic['opts'].soil_file
    extra = tmp_path / 'soil_extra.txt'
    with open(src) as f, open(extra, 'w') as g:
        lines = f.read().splitlines()
        g.write('\n'.join(lines) + '\n')
        toks = lines[0].split()
        toks[1], toks[2], toks[3] = '99999', '48.6875', '-120.6875'
        g.write(' '.join(toks) + '\n')
    dom = ClassicDomain(str(extra), classic['opts'])
    assert dom.n_active == 17
    ic, ii, dc, di = match_cells(dom, image['dom'])
    assert ic.size == 16 and dc == [16] and di == []
    assert int(dom.gridcell[dc[0]]) == 99999
    # And the other way: a soil file missing a cell the image runs.
    fewer = tmp_path / 'soil_fewer.txt'
    with open(src) as f, open(fewer, 'w') as g:
        g.write('\n'.join(f.read().splitlines()[1:]) + '\n')
    dom2 = ClassicDomain(str(fewer), classic['opts'])
    ic, ii, dc, di = match_cells(dom2, image['dom'])
    assert ic.size == 15 and dc == [] and len(di) == 1
    lat_lost = float(image['dom'].lat[di[0]])
    assert abs(lat_lost - 48.3125) < 1e-9        # the row that was removed


def test_a_run_cell_zero_row_is_read_and_not_run(classic, tmp_path):
    src = classic['opts'].soil_file
    off = tmp_path / 'soil_off.txt'
    with open(src) as f, open(off, 'w') as g:
        lines = f.read().splitlines()
        lines[3] = '0' + lines[3][1:]
        g.write('\n'.join(lines) + '\n')
    said = []
    dom = ClassicDomain(str(off), classic['opts'], log=said.append)
    assert dom.n_active == 15 and dom.run_cell.size == 16
    assert any('run_cell 0' in s for s in said)


# ---------------------------------------------------------------------------
# 1. Parameters.
# ---------------------------------------------------------------------------
def test_soil_parameters_are_identical(classic, image, matched):
    ic, ii, _, _ = matched
    cp, ip = classic['par'], image['par']
    assert set(cp.soil) == set(ip.soil)
    worst = {}
    for k in sorted(cp.soil):
        a = np.asarray(cp.soil[k])[..., ic]
        b = np.asarray(ip.soil[k])[..., ii]
        assert a.shape == b.shape, k
        worst[k] = float(np.abs(a - b).max())
        np.testing.assert_array_equal(a, b, err_msg=k)
    assert max(worst.values()) == 0.0
    # The derived quantities go through ImageParams' own methods on both.
    np.testing.assert_array_equal(cp.max_moist()[..., ic],
                                  np.asarray(ip.max_moist())[..., ii])
    for a, b in zip(cp.critical_moisture(), ip.critical_moisture()):
        np.testing.assert_array_equal(a[..., ic], np.asarray(b)[..., ii])
    assert cp.n_layers == ip.n_layers == 3
    assert cp.n_root == ip.n_root == 3
    assert cp.n_bands == ip.n_bands == 5
    assert cp.n_veg == ip.n_veg == 12
    # off_gmt: the same -8 h; xarray decodes the image file's into a
    # timedelta because its units attribute says "hours".
    np.testing.assert_array_equal(cp.off_gmt[ic], -8.0)
    assert cp.utc_offset_hours() == -8.0


def test_snow_bands_are_identical(classic, image, matched):
    ic, ii, _, _ = matched
    cb, ib = classic['par'].bands, image['par'].bands
    assert set(cb) == set(ib) == {'AreaFract', 'elevation', 'Pfactor'}
    for k in cb:
        a, b = np.asarray(cb[k])[..., ic], np.asarray(ib[k])[..., ii]
        assert a.shape == b.shape == (5, 16)
        # Pfactor rows that sum to 1 only to floating-point precision are
        # divided by their sum (read_snowband.c:895-902): 1e-16, not 0.
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-15, err_msg=k)
    # And so the elevation VIC runs at is the same on both paths.
    np.testing.assert_allclose(
        image_run.band_mean_elevation(classic['par'])[ic],
        image_run.band_mean_elevation(image['par'])[ii], rtol=0, atol=1e-12)


def test_vegetation_is_identical_except_where_vics_drivers_differ(
        classic, image, matched):
    ic, ii, _, _ = matched
    cv_, iv = classic['par'].veg, image['par'].veg
    assert set(cv_) == set(iv), (set(cv_) ^ set(iv))
    bare = classic['par'].n_veg - 1
    assert classic['par'].veg_descr[bare] == 'Bare Soil'
    cover = np.asarray(cv_['Cv'])[..., ic]
    # Cover: the classic driver rescales sums of 1.000001 and 0.999999
    # (read_vegparam.c:628-645); the image driver keeps them. 1e-6, and the
    # bare-soil remainder on the two cells with 0.982222 of vegetation.
    icover = np.asarray(iv['Cv'])[..., ii]
    assert np.abs(cover - icover).max() <= 2e-6
    assert np.abs(cover.sum(0) - 1.0).max() <= 1e-12
    assert sorted(np.round(cover[bare], 6)) == [0.0] * 14 + [0.017778] * 2
    # Tile-level fields: identical wherever there is a tile, and zero on
    # both sides where there is not.
    for k in ('LAI', 'root_depth', 'root_fract'):
        a, b = np.asarray(cv_[k])[..., ic], np.asarray(iv[k])[..., ii]
        np.testing.assert_array_equal(a, b, err_msg=k)
    # Class-level scalars and monthly fields: the library's, for every
    # class, on both sides -- except the three bare-soil fields below.
    for k in ('rmin', 'rarc', 'RGL', 'overstory', 'trunk_ratio',
              'wind_atten', 'rad_atten', 'albedo'):
        a, b = np.asarray(cv_[k])[..., ic], np.asarray(iv[k])[..., ii]
        np.testing.assert_array_equal(a, b, err_msg=k)
    for k in ('veg_rough', 'displacement', 'wind_h'):
        a, b = np.asarray(cv_[k])[..., ic], np.asarray(iv[k])[..., ii]
        np.testing.assert_array_equal(a[:bare], b[:bare], err_msg=k)
    # The bare-soil tile, where VIC's two drivers genuinely disagree. The
    # classic driver's numbers are asserted so the disagreement is measured.
    rough = np.asarray(classic['par'].soil['rough'])[ic]
    np.testing.assert_array_equal(np.asarray(cv_['veg_rough'])[bare][..., ic],
                                  np.broadcast_to(rough, (12, ic.size)))
    np.testing.assert_allclose(np.asarray(cv_['displacement'])[bare][..., ic],
                               np.broadcast_to(rough * 0.667 / 0.123,
                                               (12, ic.size)), rtol=1e-15)
    np.testing.assert_array_equal(np.asarray(cv_['wind_h'])[bare][..., ic],
                                  10.0)                       # WIND_H
    assert np.all(np.asarray(iv['veg_rough'])[bare] == 0.0)
    assert np.all(np.asarray(iv['displacement'])[bare] == 0.0)
    assert np.all(np.asarray(iv['wind_h'])[bare] == 2.0)


def test_soil_column_order_follows_the_options(classic, tmp_path):
    """The Stehekin soil file has a 60th column the global file says not to
    read. Turning JULY_TAVG_SUPPLIED on reads it as the July temperature;
    turning SPATIAL_FROST on as well asks for a column the file lacks, and
    the failure names it, as read_soilparam.c does."""
    cols = soil_columns(classic['opts'])
    assert sum(c for _, c in cols) == 53
    assert [n for n, _ in cols][:5] == ['run_cell', 'gridcell', 'lat', 'lon',
                                         'infilt']
    assert [n for n, _ in cols][-3:] == ['annual_prec', 'resid_moist',
                                          'fs_active']
    said = []
    read_soil_file(classic['opts'].soil_file, classic['opts'],
                   log=said.append)
    assert any('1 trailing column' in s for s in said)

    def variant(**kw):
        g = tmp_path / ('global_%s.txt' % '_'.join(kw))
        text = open(CLASSIC_GLOBAL).read()
        for k, v in kw.items():
            if '%s  FALSE' % k in text:
                text = text.replace('%s  FALSE' % k, '%s  %s' % (k, v))
            else:
                # Stehekin's global does not mention SPATIAL_FROST at all.
                text += '\n%s  %s\n' % (k, v)
        g.write_text(text)
        with sample_env():
            return ClassicOptions(GlobalParam(str(g)))

    o = variant(JULY_TAVG_SUPPLIED='TRUE')
    assert soil_columns(o)[-1] == ('July_Tavg', 1)
    table = read_soil_file(o.soil_file, o)
    last = np.array([float(l.split()[-1]) for l in open(o.soil_file)
                     if l.strip()])
    np.testing.assert_array_equal(table['July_Tavg'], last)
    o2 = variant(JULY_TAVG_SUPPLIED='TRUE', SPATIAL_FROST='TRUE')
    with pytest.raises(ValueError) as e:
        read_soil_file(o2.soil_file, o2)
    assert "can't find values for July_Tavg" in str(e.value)


def test_options_the_reader_will_not_read_are_refused_by_name(tmp_path):
    g = tmp_path / 'global_refused.txt'
    text = open(CLASSIC_GLOBAL).read()
    text = text.replace('FORCE_FORMAT         ASCII', 'FORCE_FORMAT BINARY')
    text = text.replace('ORGANIC_FRACT       FALSE', 'ORGANIC_FRACT TRUE')
    # BASEFLOW NIJSSEN2001 is not among them: the d1..d4 the soil file
    # carries are converted in `image_run.build_soil`
    # (read_soilparam.c:693-707), which the classic parameters reach
    # like the image ones (tests/test_nijssen2001.py).
    g.write_text(text)
    with sample_env():
        opts = ClassicOptions(GlobalParam(str(g)))
    heads = sorted(r.split(':')[0] for r in opts.refused)
    assert heads == ['FORCE_FORMAT BINARY']
    with pytest.raises(ValueError) as e:
        opts.require_supported()
    for h in heads:
        assert h in str(e.value)
    # A ROOT_ZONES-less file is VIC's own stop (get_global_param.c:1037).
    g2 = tmp_path / 'global_noroot.txt'
    g2.write_text(open(CLASSIC_GLOBAL).read().replace('ROOT_ZONES          3',
                                                      ''))
    with sample_env(), pytest.raises(ValueError, match='ROOT_ZONES'):
        ClassicOptions(GlobalParam(str(g2)))


def test_missing_resid_moist_takes_vics_default(classic, tmp_path):
    src = classic['opts'].soil_file
    p = tmp_path / 'soil_missing.txt'
    with open(src) as f, open(p, 'w') as g:
        lines = f.read().splitlines()
        t = lines[0].split()
        t[50] = '%g' % MISSING                    # resid_moist, layer 1
        lines[0] = ' '.join(t)
        g.write('\n'.join(lines) + '\n')
    said = []
    table = read_soil_file(str(p), classic['opts'], log=said.append)
    assert table['resid_moist'][0, 0] == 0.0
    assert any('SOIL_RESID_MOIST' in s for s in said)


# ---------------------------------------------------------------------------
# 1. Forcing.
# ---------------------------------------------------------------------------
#: Half a unit in the fourth decimal of the ASCII file, in the canonical unit.
ASCII_HALF_UNIT = {'press': 5e-5 * 1000.0, 'vp': 5e-5 * 1000.0}
FLOAT32_REL = 2e-7


def test_forcing_agrees_to_the_precision_the_ascii_is_written_at(
        classic, image, matched):
    ic, ii, _, _ = matched
    cf, imf = classic['fields'], image['fields']
    assert set(cf) == set(imf) == {'prcp', 'tair', 'shortwave', 'longwave',
                                   'press', 'vp', 'wind'}
    assert np.array_equal(classic['stamps'], image['stamps'])
    assert classic['stamps'].dtype == image['stamps'].dtype
    assert str(classic['stamps'][0])[:19] == '1949-01-01T00:00:00'
    assert classic['stamps'].size == 240
    # Pressure is in Pa on both paths -- the classic reader multiplied by
    # 1000 as vic_force.c does, the image reader from the units attribute.
    assert 70e3 < float(cf['press'].mean()) < 100e3
    assert 70e3 < float(imf['press'].mean()) < 100e3
    report = []
    for k in sorted(cf):
        a = cf[k][:, ic].cpu().numpy()
        b = imf[k][:, ii].cpu().numpy()
        assert a.shape == b.shape == (240, 16)
        tol = ASCII_HALF_UNIT.get(k, 5e-5) + FLOAT32_REL * np.abs(b).max()
        worst = float(np.abs(a - b).max())
        report.append((k, worst, tol))
        assert worst <= tol, (k, worst, tol)
    # Well inside the floor, or the floor was the wrong explanation.
    assert max(w / t for _, w, t in report) < 0.5, report


def test_forcing_period_is_selected_like_read_atmos_data(classic):
    fc = classic['fc']
    f, s = fc.read(dtype=DT, model_start=dt.date(1949, 1, 3),
                   model_end=dt.date(1949, 1, 5))
    assert s.size == 72 and str(s[0])[:13] == '1949-01-03T00'
    assert f['tair'].shape == (72, 16)
    with pytest.raises(ValueError, match='not on a forcing record'):
        fc.read(dtype=DT, model_start=dt.datetime(1949, 1, 3, 0, 30))
    with pytest.raises(ValueError, match='not enough records'):
        fc.read(dtype=DT, model_start=dt.date(1949, 1, 3),
                model_end=dt.date(1949, 2, 1))


# ---------------------------------------------------------------------------
# 2. Fluxes.
# ---------------------------------------------------------------------------
def _defaults(device):
    """VIC's own defaults for what the parameter file does not carry --
    the same dict scripts/run_walnutgulch.py passes."""
    def t(v):
        return torch.tensor(v, dtype=DT, device=device)
    return dict(pcf=t(1.0), t_max_snow=t(0.5), t_min_rain=t(-0.5),
                rs=t(120.0), et_beta=t(1.0),
                new_snow_albedo=t(0.85), alb_accum_a=t(0.94),
                alb_accum_b=t(0.58), alb_thaw_a=t(0.82), alb_thaw_b=t(0.46),
                uh_shape=t(2.5), uh_scale=t(2.0),
                lai_scale=t(1.0), rmin_scale=t(1.0),
                w_crit=t(0.7), w_wilt=t(0.3))


def _simulate(gp, dom, par, fields, stamps):
    """Exactly the call scripts/run_walnutgulch.py makes, on either reader."""
    import pandas as pd
    steps = gp.get_int('MODEL_STEPS_PER_DAY')
    runoff_steps = gp.get_int('RUNOFF_STEPS_PER_DAY')
    dates = [pd.Timestamp(x).to_pydatetime() for x in stamps]
    elev = image_run.band_mean_elevation(par)
    batch = image_run.ImageBatch(dom, dates, dtype=DT, device=DEVICE,
                                 elevation=elev)
    atm = image_run.build_atmosphere(fields, batch)
    atm.pop('_sources')
    sp = image_run.build_soil(par, DT, DEVICE, runoff_steps_per_day=runoff_steps,
                              model_steps_per_day=steps)
    mosaic = image_run.build_mosaic(par, DT, DEVICE, log=lambda s: None)
    wind_h = float(gp.get('WIND_H') or basin.MEAS_HEIGHT)
    init_moist = torch.as_tensor(par.soil['init_moist'], dtype=DT,
                                 device=DEVICE).T.contiguous()
    out = basin.simulate(batch, atm, _defaults(DEVICE), sp, mosaic=mosaic,
                         wind_h=wind_h, init_moist=init_moist,
                         steps_per_day=steps)
    return out, batch


FLUX_REL = 1e-6       # basin total, relative
FLUX_CELL_ABS = 1e-5  # any cell, any step, mm


def test_fluxes_agree_through_the_unchanged_image_run_path(classic, image,
                                                           matched):
    ic, ii, _, _ = matched
    assert classic['gp'].get_int('MODEL_STEPS_PER_DAY') == 24
    assert image['gp'].get_int('MODEL_STEPS_PER_DAY') == 24
    oi, bi = _simulate(image['gp'], image['dom'], image['par'],
                       image['fields'], image['stamps'])
    # The one place VIC's two drivers give this run different inputs is the
    # bare-soil tile's geometry (test_vegetation_is_identical_except_where_
    # vics_drivers_differ): classic roughness, displacement and wind_h
    # rough, rough * 0.667 / 0.123 and WIND_H = 10 m, image 0, 0 and 2 m.
    # The water-balance path reads it where VIC does -- a thin pack's
    # vapour flux takes Ra[2] from the tile's reference height
    # (CalcAerodynamic.c:97-98 via func_surf_energy_bal.c:667-684), and so
    # does a full pack's balance (snow_melt via solve_snow.c 5.1.0:315-322)
    # -- so as read the two sublimate differently off the two cells with a
    # bare tile (1.8% cover): 2.0e-4 mm in the worst cell-step from thin
    # packs alone, 1.7e-3 mm since the full pack reads it too, nothing else
    # moving. That is VIC's own driver difference, pinned here; the
    # readers are then compared with the bare tile's geometry made the same.
    ic_, ii_, _, _ = matched
    oc_read, _ = _simulate(classic['gp'], classic['dom'], classic['par'],
                           classic['fields'], classic['stamps'])
    bare = classic['par'].n_veg - 1
    bare_cells = torch.as_tensor(
        np.asarray(classic['par'].veg['Cv'])[bare][ic_] > 0.0,
        device=oi['et'].device)
    assert int(bare_cells.sum()) == 2
    for k in ('et', 'sub_snow'):
        d = (oc_read[k][:, ic_] - oi[k][:, ii_]).abs()
        assert float(d[:, ~bare_cells].max()) <= FLUX_CELL_ABS, k
        # An upper bound only: the difference is VIC's own driver
        # difference, read from its source, not a run of vic_classic.exe
        # against vic_image.exe, so a reader change that closes it is not
        # a failure.
        assert float(d.max()) < 3e-3, (k, float(d.max()))
    for k in ('transp', 'evap_canop', 'runoff', 'baseflow'):
        d = (oc_read[k][:, ic_] - oi[k][:, ii_]).abs()
        assert float(d.max()) <= FLUX_CELL_ABS, (k, float(d.max()))
    par_c = copy.copy(classic['par'])
    par_c.veg = dict(classic['par'].veg)
    for k in ('veg_rough', 'displacement', 'wind_h'):
        a = np.array(par_c.veg[k], dtype=float)
        a[bare][..., ic_] = np.asarray(image['par'].veg[k])[bare][..., ii_]
        par_c.veg[k] = a
    oc, bc = _simulate(classic['gp'], classic['dom'], par_c,
                       classic['fields'], classic['stamps'])
    # The classic path has no domain file: its areas are ImageBatch's
    # spherical fallback on the inferred 0.125 degree grid, the image path's
    # are the domain file's. They differ by 7e-5 in total and the weights
    # by 2e-10, so the basin means below are weighted the same way.
    assert bc.area_source.startswith('spherical, 0.125')
    assert bi.area_source == 'domain file'
    wc = bc.cell_area / bc.cell_area.sum()
    wi = bi.cell_area / bi.cell_area.sum()
    assert float((wc[ic] - wi[ii]).abs().max()) < 1e-8
    assert abs(float(bi.cell_area.sum() / bc.cell_area.sum()) - 1) < 1e-3
    report = {}
    for k in ('et', 'runoff', 'baseflow', 'swe'):
        a, b = oc[k][:, ic], oi[k][:, ii]
        assert a.shape == b.shape == (240, 16)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        if k == 'swe':
            ta, tb = float((a[-1] * wc[ic]).sum()), float((b[-1] * wi[ii]).sum())
        else:
            ta, tb = float((a * wc[ic]).sum()), float((b * wi[ii]).sum())
        cell = float((a - b).abs().max())
        rel = abs(ta - tb) / max(abs(tb), 1e-12)
        report[k] = (ta, tb, rel, cell)
        assert cell <= FLUX_CELL_ABS, (k, cell)
        assert rel <= FLUX_REL, (k, ta, tb, rel)
    # January in the Cascades: snow accumulates, nothing runs off, baseflow
    # drains the initial store, and ET is small. The numbers themselves are
    # asserted loosely so a physics change does not fail an equivalence
    # test, but a run that produced nothing would.
    assert report['swe'][1] > 10.0
    assert report['baseflow'][1] > 1.0
    assert report['et'][1] > 0.01
    # A trace, not exactly zero. With one snow pack per vegetation tile --
    # which is what VIC keeps -- a tile whose canopy intercepted less snow
    # can melt while the cell as a whole does not, and 0.0102 mm over ten
    # January days is that. One pack per cell gave exactly 0. Both drivers
    # produce the same trace, which is what this test is actually for.
    assert report['runoff'][1] < 0.1, report['runoff']
    assert abs(report['runoff'][0] - report['runoff'][1]) < 1e-6
    print('\nclassic vs image, area-weighted basin totals over 240 h (mm):')
    for k, (ta, tb, rel, cell) in report.items():
        print('  %-9s classic %12.6f  image %12.6f  rel %.2e  '
              'max cell-step |diff| %.2e mm' % (k, ta, tb, rel, cell))


# ---------------------------------------------------------------------------
# A second dataset: FindleyLake exercises the tile lines Stehekin lacks.
# ---------------------------------------------------------------------------
findley = pytest.mark.skipif(
    not os.path.exists(os.path.join(FINDLEY_CLASSIC, 'parameters',
                                    'soil_param.FindleyLake'))
    or not os.path.exists(os.path.join(FINDLEY_IMAGE, 'parameters',
                                       'params.FindleyLake.nc')),
    reason='VIC_sample_data FindleyLake is not present')


@findley
def test_findley_lake_reads_fcanopy_and_albedo_lines_and_equal_area():
    """VEGPARAM_FCAN and VEGPARAM_ALB TRUE, FCAN_SRC/ALB_SRC FROM_VEGPARAM,
    two-token FORCE_TYPE lines, EQUAL_AREA with RESOLUTION 1.29 km, one
    snow band, a 17-class IGBP library. UW-Hydro's global file names a
    `params/` directory that ships as `parameters/`, which the reader names
    in its error; the files are then given explicitly."""
    gp = GlobalParam(os.path.join(FINDLEY_CLASSIC, 'parameters',
                                  'global_param.FindleyLake.txt'),
                     sample_data=SD)
    opts = ClassicOptions(gp)
    assert opts.force_columns == ['PREC', 'AIR_TEMP', 'SWDOWN', 'LWDOWN',
                                  'PRESSURE', 'VP', 'WIND']
    assert opts.vegparam_fcan and opts.vegparam_alb and opts.equal_area
    with pytest.raises(IOError, match='parameters/'):
        ClassicDomain(opts.soil_file, opts)
    P = os.path.join(FINDLEY_CLASSIC, 'parameters')
    dom = ClassicDomain(os.path.join(P, 'soil_param.FindleyLake'), opts)
    assert dom.n_active == 1
    np.testing.assert_array_equal(dom.area, [1.29e6])   # compute_cell_area.c:48
    par = ClassicParams(dom, opts,
                        veglib_path=os.path.join(P, 'veg_lib_IGBP'),
                        vegparam_path=os.path.join(P, 'veg_param.FindleyLake'))
    assert par.n_veg == 18            # 17 IGBP classes + VIC's bare soil
    assert 'fcanopy' in par.veg
    # Against its image twin. The image file has 17 classes: no separate
    # bare-soil class, because the image driver assumes the *last* class is
    # bare soil (vic_init.c:269), and here the classic bare tile has no cover.
    IP = os.path.join(FINDLEY_IMAGE, 'parameters')
    gi = GlobalParam(os.path.join(IP, 'global_param.FindleyLake.txt'),
                     sample_data=SD)
    idom = ImageDomain(os.path.join(IP, 'domain.FindleyLake.nc'),
                       names=gi.domain_vars,
                       params=os.path.join(IP, 'params.FindleyLake.nc'))
    ipar = ImageParams(os.path.join(IP, 'params.FindleyLake.nc'), idom)
    assert ipar.n_veg == 17
    np.testing.assert_array_equal(idom.area, dom.area)
    cv_, iv = par.veg, ipar.veg
    assert cv_['Cv'][17, 0] == 0.0
    np.testing.assert_array_equal(cv_['Cv'][:17], np.asarray(iv['Cv']))
    # Unlike Stehekin's, this image file pads every field of an absent class
    # with NaN -- roughness and resistances included -- where the classic
    # reader holds 0 for tile fields and the library's value for class
    # fields. build_mosaic treats both as padding. Compared where there is a
    # tile: classes 0, 1 and 16.
    has = np.asarray(iv['Cv'])[:, 0] > 0
    assert list(np.flatnonzero(has)) == [0, 1, 16]
    for k in ('veg_rough', 'displacement', 'root_depth', 'root_fract'):
        np.testing.assert_array_equal(cv_[k][:17][has], np.asarray(iv[k])[has],
                                      err_msg=k)
        assert np.all(np.isnan(np.asarray(iv[k])[~has])), k
    for k in ('root_depth', 'root_fract'):
        assert np.all(cv_[k][:17][~has] == 0.0)
    # The vegparam lines are written to four decimals; the image file holds
    # the unrounded values. Half a unit in the fourth place, where there is a
    # tile.
    for k in ('LAI', 'fcanopy', 'albedo'):
        a, b = cv_[k][:17][has], np.asarray(iv[k])[has]
        assert np.abs(a - b).max() <= 5e-5, k
    # Where UW-Hydro's two FindleyLake files disagree with each other. The
    # image file zeroes every class scalar of class 0 (Open Water, 8.85%
    # cover) and class 16 (Barren, 8.84%) -- rmin, rarc, RGL, wind_h, the
    # attenuations, the trunk ratio -- where the IGBP library the classic
    # global names gives both rarc 5, rmin 50, RGL 100, wind_h 5, 0.5, 0.5,
    # 0.2. The forest class carrying 82% of the cell is identical. Measured,
    # not hidden: this is a property of the sample data, not of a reader.
    scalars = ('rmin', 'rarc', 'RGL', 'wind_h', 'overstory', 'trunk_ratio',
               'wind_atten', 'rad_atten')
    for k in scalars:
        assert cv_[k][1, 0] == float(np.asarray(iv[k])[1, 0]), k
        for j in (0, 16):
            assert float(np.asarray(iv[k])[j, 0]) == 0.0, (k, j)
    for j in (0, 16):
        assert [cv_[k][j, 0] for k in ('rarc', 'rmin', 'RGL', 'wind_h')] == \
            [5.0, 50.0, 100.0, 5.0]
    # Soil: identical except init_moist and annual_prec, which the two
    # sample files simply do not share (12.3/36.8/122.8 vs 41.6/124.7/83.1
    # mm; -99 vs NaN).
    for k in sorted(par.soil):
        if k in ('init_moist', 'annual_prec'):
            continue
        np.testing.assert_array_equal(par.soil[k], np.asarray(ipar.soil[k]),
                                      err_msg=k)


@findley
def test_findley_lake_forcing_matches_the_netcdf_year():
    gp = GlobalParam(os.path.join(FINDLEY_CLASSIC, 'parameters',
                                  'global_param.FindleyLake.txt'),
                     sample_data=SD)
    opts = ClassicOptions(gp)
    P = os.path.join(FINDLEY_CLASSIC, 'parameters')
    dom = ClassicDomain(os.path.join(P, 'soil_param.FindleyLake'), opts)
    fc = ClassicForcing(opts.forcing_prefix, dom, opts.force_columns,
                        opts.force_steps_per_day, opts.force_start,
                        grid_decimal=opts.grid_decimal, fmt=opts.force_format)
    assert fc.files[0].endswith('FindleyLake_47.3188_-121.5853')
    f, s = fc.read(dtype=DT, model_start=dt.date(1970, 1, 1),
                   model_end=dt.date(1970, 12, 31))
    assert s.size == 8760
    gi = GlobalParam(os.path.join(FINDLEY_IMAGE, 'parameters',
                                  'global_param.FindleyLake.txt'),
                     sample_data=SD)
    IP = os.path.join(FINDLEY_IMAGE, 'parameters')
    idom = ImageDomain(os.path.join(IP, 'domain.FindleyLake.nc'),
                       names=gi.domain_vars,
                       params=os.path.join(IP, 'params.FindleyLake.nc'))
    ifc = ImageForcing(os.path.join(FINDLEY_IMAGE, 'forcings',
                                    'livneh_NAmerExt_15Oct2014.FindleyLake.'),
                       idom, force_type=gi.force_type, years=[1970])
    g, t = ifc.read(dtype=DT, log=lambda s: None)
    assert np.array_equal(s, t)
    for k in sorted(f):
        a, b = f[k].cpu().numpy(), g[k].cpu().numpy()
        # This NetCDF was written in float64 from the same numbers, so the
        # agreement is at 1e-7 relative rather than at the ASCII floor.
        assert np.abs(a - b).max() <= 1e-7 * max(np.abs(b).max(), 1.0), k
