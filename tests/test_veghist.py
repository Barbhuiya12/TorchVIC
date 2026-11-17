"""FORCING2 and *_SRC FROM_VEGHIST: the vegetation history forcing.

vic_force.c 5.1.0:241-262 (monthly values into veg_hist), :264-363 (the
file's LAI, fcanopy and albedo over them), :419-439 (leafless rule per
record), :466-474 (average over the model step). The end-to-end agreement
with pristine VIC is the verification matrix's igp_*_veghist rows.
"""
import numpy as np
import pytest
import torch

from torchvic.core.vegetation import VegMosaic
from torchvic.io import options as opt
from torchvic.io import veghist
from torchvic.io.global_param import GlobalParam



@pytest.fixture(autouse=True)
def _core_5_1_0():
    from torchvic.core import version
    old = version.CORE
    version.set_core('5.1.0')
    yield
    version.set_core(old)


def _gp(tmp_path, *lines):
    p = tmp_path / 'g.txt'
    p.write_text('\n'.join(('FORCING1 /d/f.', 'FORCE_TYPE AIR_TEMP t') +
                           lines) + '\n')
    return GlobalParam(str(p))


def test_force_type_after_forcing2_belongs_to_the_second_file(tmp_path):
    g = _gp(tmp_path, 'FORCING2 /d/veg.', 'FORCE_TYPE SKIP veg')
    assert g.force_type == {'AIR_TEMP': 't'}
    assert g.force2_type == {'SKIP': 'veg'}
    assert g.forcing2_prefix == '/d/veg.'


def test_forcing2_false_is_no_file(tmp_path):
    assert _gp(tmp_path, 'FORCING2 FALSE').forcing2_prefix is None


def test_no_forcing2_leaves_the_forcing_types_alone(tmp_path):
    g = _gp(tmp_path)
    assert g.force_type == {'AIR_TEMP': 't'} and not g.force2_type
    assert g.forcing2_prefix is None and veghist.wanted(g) == {}


def test_options_accept_a_history_with_a_force_type_line(tmp_path):
    g = _gp(tmp_path, 'LAI_SRC FROM_VEGHIST', 'ALB_SRC FROM_VEGHIST',
            'FORCING2 /d/veg.', 'FORCE_TYPE SKIP veg')
    refused, _ = opt.check(g)
    assert not [r for r in refused if 'VEGHIST' in r or 'FORCING2' in r]
    assert veghist.wanted(g) == {'lai': 'LAI', 'albedo': 'albedo'}


@pytest.mark.parametrize('src', ['LAI_SRC', 'FCAN_SRC', 'ALB_SRC'])
def test_options_refuse_a_history_without_a_file(tmp_path, src):
    g = _gp(tmp_path, '%s FROM_VEGHIST' % src)
    refused, _ = opt.check(g)
    assert any('without FORCING2' in r for r in refused)


def test_options_refuse_a_file_vic_never_opens(tmp_path):
    # no FORCE_TYPE after FORCING2: count_force_vars is 0, the file stays shut
    g = _gp(tmp_path, 'LAI_SRC FROM_VEGHIST', 'FORCING2 /d/veg.')
    refused, _ = opt.check(g)
    assert any('no FORCE_TYPE' in r for r in refused)


def test_forcing2_alone_is_announced_as_no_effect(tmp_path):
    g = _gp(tmp_path, 'FORCING2 /d/veg.', 'FORCE_TYPE SKIP veg')
    refused, announced = opt.check(g)
    assert not [r for r in refused if 'FORCING2' in r]
    assert any(a.startswith('FORCING2: read, no effect') for a in announced)


def test_record_steps_subdaily_and_daily():
    import datetime as dt
    dates = [dt.datetime(2004, 1, 1, h) for h in (0, 12)] + \
        [dt.datetime(2004, 1, 2, 0)]
    stamps = [np.datetime64('2004-01-01T00'), np.datetime64('2004-01-01T12'),
              np.datetime64('2004-01-02T00'), np.datetime64('2004-01-03T00')]
    assert veghist.record_steps(stamps, dates, daily=False) == [0, 1, 2, -1]
    d2 = [dt.datetime(2004, 1, 1), dt.datetime(2004, 1, 2)]
    assert veghist.record_steps(stamps, d2, daily=True) == [0, 0, 1, -1]


def _mosaic(cv, lai, fc, alb, disp, rough):
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)       # noqa: E731
    cv = t(cv)
    n, k = cv.shape
    m = VegMosaic(cv, t(lai), torch.ones(n, k), torch.ones(n, k),
                  torch.ones(n, k), torch.ones(n, k, 3), torch.float64,
                  torch.device('cpu'), veg_rough=t(rough),
                  displacement=t(disp), albedo=t(alb), fcanopy=t(fc))
    m.raw_monthly = (t(lai), t(fc))
    return m


def test_apply_replaces_and_averages_and_zeroes():
    import datetime as dt
    # one cell, two tiles (second has no cover), Jan 1 and Jan 2 daily steps,
    # four records a day
    lai12 = np.full((1, 2, 12), 2.0)
    fc12 = np.full((1, 2, 12), 0.8)
    alb12 = np.full((1, 2, 12), 0.15)
    disp12 = np.full((1, 2, 12), 1.0)
    rough12 = np.full((1, 2, 12), 0.1)
    m = _mosaic([[1.0, 0.0]], lai12, fc12, alb12, disp12, rough12)
    lai = np.zeros((8, 2, 1))
    lai[:, 0, 0] = [1, 2, 3, 4, 0, 4, 4, 4]       # day 2: one leafless record
    stamps = [np.datetime64('2004-01-01T00') + np.timedelta64(6 * i, 'h')
              for i in range(8)]
    dates = [dt.datetime(2004, 1, 1), dt.datetime(2004, 1, 2)]
    veghist.apply(m, {'lai': torch.as_tensor(lai)}, stamps, dates, True)
    assert m.step_axis and m.lai.shape == (1, 2, 2)
    assert m.lai[0, 0, 0] == pytest.approx(2.5)
    # the leafless record of day 2 sets LAI and fcanopy to 0 before averaging
    assert m.lai[0, 0, 1] == pytest.approx(3.0)
    assert m.fcanopy[0, 0, 0] == pytest.approx(0.8)
    assert m.fcanopy[0, 0, 1] == pytest.approx(0.6)
    # not from the file: the parameter file's month
    assert m.albedo[0, 0, 0] == pytest.approx(0.15)
    assert m.displacement.shape == (1, 2, 2)
    assert not hasattr(m, 'raw_monthly')
    # the tile without cover keeps the monthly values, not the file's
    assert m.lai[0, 1, 0] == pytest.approx(2.0)


def test_apply_one_record_per_subdaily_step_and_missing_step():
    import datetime as dt
    m = _mosaic([[1.0]], np.full((1, 1, 12), 2.0), np.full((1, 1, 12), 1.0),
                np.full((1, 1, 12), 0.2), np.zeros((1, 1, 12)),
                np.full((1, 1, 12), 0.1))
    dates = [dt.datetime(2004, 6, 1, h) for h in (0, 3)]
    h = {'albedo': torch.as_tensor([[[0.31]], [[0.33]]], dtype=torch.float64)}
    stamps = [np.datetime64('2004-06-01T00'), np.datetime64('2004-06-01T03')]
    veghist.apply(m, h, stamps, dates, False)
    assert m.albedo[0, 0].tolist() == pytest.approx([0.31, 0.33])
    m2 = _mosaic([[1.0]], np.full((1, 1, 12), 2.0), np.full((1, 1, 12), 1.0),
                 np.full((1, 1, 12), 0.2), np.zeros((1, 1, 12)),
                 np.full((1, 1, 12), 0.1))
    with pytest.raises(ValueError, match='no record for model step'):
        veghist.apply(m2, h, stamps[:1], dates, False)


def test_the_5_0_1_core_is_refused():
    from torchvic.core import version
    old = version.CORE
    try:
        version.set_core('5.0.1')
        m = _mosaic([[1.0]], np.ones((1, 1, 12)), np.ones((1, 1, 12)),
                    np.ones((1, 1, 12)), np.ones((1, 1, 12)),
                    np.ones((1, 1, 12)))
        with pytest.raises(ValueError, match='5.1.0 core'):
            veghist.apply(m, {}, [], [], False)
    finally:
        version.set_core(old)


def test_off_leaves_the_mosaic_alone():
    m = _mosaic([[1.0]], np.ones((1, 1, 12)), np.ones((1, 1, 12)),
                np.ones((1, 1, 12)), np.ones((1, 1, 12)), np.ones((1, 1, 12)))
    assert m.step_axis is False
    assert m.select(torch.tensor([0])).step_axis is False
