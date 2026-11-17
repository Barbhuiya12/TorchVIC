"""CalcAerodynamic.c, both branches, against the C compiled from the pristine
5.0.1 tree: the no-overstory Ra[0] the mosaic has used all along (never pinned
to compiled C until now) and the overstory Ra[0..2], U[0..2] with the three
trunk-space forms. CPU-pinned like the other oracles.
"""
import os
import shutil
import subprocess
import sys

import pytest
import torch

from torchvic.core import vegetation as V  # noqa: E402
from torchvic.core import energy as E  # noqa: E402

import _paths                                           # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')
VIC = _paths.VIC501_TREE
CPU = torch.device('cpu')
Z0_SNOW, Z0_SOIL, WIND_H = 0.03, 0.01, 10.0

SHIM = r'''
#include <stdio.h>
#include <vic_run.h>
#include <vic_driver_shared_all.h>
parameters_struct param; option_struct options; global_param_struct global_param;
int main(void){
  initialize_log();
  initialize_parameters();
  double d_v[]={0.4, 1.0, 3.0, 8.0}, z0_v[]={0.05, 0.5, 1.2}, tr_v[]={0.2, 0.5}, n_v[]={0.5, 1.5}, w_v[]={0.0, 0.5, 3.0};
  for(int a=0;a<4;a++)for(int b=0;b<3;b++)for(int c=0;c<2;c++)for(int e=0;e<2;e++)for(int f=0;f<3;f++){
    double d=d_v[a], z0=z0_v[b];
    double height = d / param.VEG_RATIO_DH_HEIGHT;
    double Ra[3], U[3]={w_v[f], -999., -999.}, disp[3]={d,0,0}, ref[3]={0,0,0}, rough[3]={z0,0,0};
    ref[0] = (d < %g) ? %g : d + %g + z0;
    for (int over=0; over<2; over++){
      double Ra_[3], U_[3]={w_v[f], -999., -999.}, disp_[3]={d,0,0}, ref_[3]={ref[0],0,0}, rough_[3]={z0,0,0};
      if (over && tr_v[c]*height < Z0_SOIL_) continue;
      CalcAerodynamic(over, height, tr_v[c], %g, %g, n_v[e], Ra_, U_, disp_, ref_, rough_);
      printf("%%d %%.17g %%.17g %%.17g %%.17g %%.17g %%.17g %%.17g %%.17g %%.17g %%.17g %%.17g %%.17g %%.17g\n", over, d, z0, tr_v[c], n_v[e], w_v[f],
             Ra_[0], Ra_[1], Ra_[2], U_[0], U_[1], U_[2], ref_[0], rough_[0]);
    }
  }
  return 0;}
'''.replace('Z0_SOIL_', '%g' % Z0_SOIL) % (WIND_H, WIND_H, WIND_H, Z0_SNOW, Z0_SOIL)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    if not os.path.isdir(VIC):
        pytest.skip('pristine VIC 5.0.1 not present under TORCHVIC_REFERENCE_DIR')
    gcc = shutil.which('gcc')
    if gcc is None:
        pytest.skip('no gcc on PATH')
    d = tmp_path_factory.mktemp('ref_aero')
    src = d / 'ref_aero.c'
    src.write_text(SHIM)
    exe = d / 'ref_aero'
    srcs = [os.path.join(VIC, 'vic_run', 'src', 'CalcAerodynamic.c')]
    srcs += [os.path.join(VIC, 'drivers', 'shared_all', 'src', f)
             for f in ('initialize_parameters.c', 'vic_log.c', 'open_file.c')]
    cmd = [gcc, '-fcommon', '-O2', '-std=c99', '-w',
           '-I' + os.path.join(VIC, 'vic_run', 'include'),
           '-I' + os.path.join(VIC, 'drivers', 'shared_all', 'include'),
           str(src)] + srcs + ['-o', str(exe), '-lm']
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1200:]
    out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
    rows = [tuple(map(float, ln.split())) for ln in out.split('\n') if ln.strip()]
    assert len(rows) > 100
    return rows


def _t(x):
    return torch.tensor(x, dtype=torch.float64, device=CPU)


def test_no_overstory_resistance_is_bit_exact_against_compiled_vic(oracle):
    worst = 0.0
    n = 0
    for over, d, z0, tr, natt, w, ra0, ra1, ra2, u0, u1, u2, rh0, rg0 in oracle:
        if over:
            continue
        got = float(V.aerodynamic_resistance(_t(z0), _t(d), _t(w), min_wind=0.0))
        worst = max(worst, abs(got - ra0) / max(abs(ra0), 1.0))
        n += 1
    assert n > 30
    assert worst < 1e-15, 'no-overstory Ra[0] differs from VIC by %.3e relative' % worst


def test_overstory_resistances_are_bit_exact_against_compiled_vic(oracle):
    worst = {k: 0.0 for k in ('ra0', 'ra1', 'ra2', 'u0', 'u1', 'u2')}
    branches = set()
    n = 0
    for over, d, z0, tr, natt, w, ra0, ra1, ra2, u0, u1, u2, rh0, rg0 in oracle:
        if not over:
            continue
        height = d / E.VEG_RATIO_DH_HEIGHT
        rh = WIND_H if d < WIND_H else d + WIND_H + z0
        r = V.aerodynamic_resistance_overstory(_t(height), _t(tr), _t(Z0_SNOW),
                                               _t(Z0_SOIL), _t(natt), _t(z0),
                                               _t(d), _t(rh), _t(w))
        zt = tr * height
        branches.add('top' if zt > 2 + Z0_SNOW else ('mid' if height > 2 + Z0_SNOW else 'low'))
        for k, ref, got in (('ra0', ra0, r['ra'][0]), ('ra1', ra1, r['ra'][1]),
                            ('ra2', ra2, r['ra'][2]), ('u0', u0, r['u'][0]),
                            ('u1', u1, r['u'][1]), ('u2', u2, r['u'][2])):
            worst[k] = max(worst[k], abs(float(got) - ref) / max(abs(ref), 1.0))
        assert float(r['ref_height'][0]) == rh0 and float(r['roughness'][0]) == rg0
        n += 1
    assert n > 60 and branches == {'top', 'mid', 'low'}, (n, branches)
    bad = {k: v for k, v in worst.items() if v > 1e-14}
    assert not bad, 'overstory terms differ from VIC (relative): %s' % bad
