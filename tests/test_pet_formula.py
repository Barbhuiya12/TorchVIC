"""Potential evapotranspiration against VIC's own compute_pot_evap, compiled
here from the pristine 5.0.1 sources.

compute_pot_evap.c forms rc with calc_rc at gsm_inv = 1 (unstressed) and puts
it through penman with the tile's architectural resistance; with fcanopy = 1
the soil-side Penman contributes nothing. TorchVIC's PET is the same two calls
per tile (basin.py, the energy block), so this pins the arithmetic and leaves
only the inputs -- resistance, net longwave -- as places a difference can be.

CPU-pinned for the same reason as test_energy_formulas: on CUDA float64
transcendental functions land an ulp off glibc.
"""
import os
import shutil
import subprocess
import sys

import pytest
import torch

from torchvic.core import vegetation as V  # noqa: E402

import _paths                                           # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')
VIC = _paths.VIC501_TREE
CPU = torch.device('cpu')

STEPS_PER_DAY = 8
ELEV = 1400.0
RARC = 60.0

SHIM = r'''
#include <stdio.h>
#include <vic_run.h>
#include <vic_driver_shared_all.h>
parameters_struct param; option_struct options; global_param_struct global_param;
int main(void){
  initialize_parameters();
  double tair_v[]={-5,4,12,25}, vpd_v[]={0,150,900}, sw_v[]={0,120,650},
         nl_v[]={-95,-40}, lai_v[]={0.225,0.488,2.0}, ra_v[]={25,70,180},
         rmin_v[]={50,135}, rgl_v[]={75,100};
  double ra[3]; double pe;
  for(int a=0;a<4;a++)for(int b=0;b<3;b++)for(int c=0;c<3;c++)for(int d=0;d<2;d++)
  for(int e=0;e<3;e++)for(int f=0;f<3;f++)for(int g=0;g<2;g++)for(int h=0;h<2;h++){
    ra[0]=ra_v[f]; ra[1]=ra_v[f]; ra[2]=ra_v[f];
    compute_pot_evap(%d, rmin_v[g], 0.2, sw_v[c], nl_v[d], rgl_v[h], tair_v[a], vpd_v[b],
                     lai_v[e], %g, ra, 0, %g, 1.0, ra_v[f], &pe);
    printf("%%g %%g %%g %%g %%g %%g %%g %%g %%.17g\n", tair_v[a], vpd_v[b], sw_v[c], nl_v[d],
           lai_v[e], ra_v[f], rmin_v[g], rgl_v[h], pe);}
  return 0;}
''' % (STEPS_PER_DAY, ELEV, RARC)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    if not os.path.isdir(VIC):
        pytest.skip('pristine VIC 5.0.1 not present under TORCHVIC_REFERENCE_DIR')
    gcc = shutil.which('gcc')
    if gcc is None:
        pytest.skip('no gcc on PATH')
    d = tmp_path_factory.mktemp('ref_pet')
    src = d / 'ref_pet.c'
    src.write_text(SHIM)
    exe = d / 'ref_pet'
    srcs = [os.path.join(VIC, 'vic_run', 'src', f)
            for f in ('compute_pot_evap.c', 'penman.c', 'svp.c', 'physics.c',
                      'canopy_assimilation.c', 'photosynth.c', )]  # calc_rc_ps, unused, links these
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
    assert len(rows) == 4 * 3 * 3 * 2 * 3 * 3 * 2 * 2
    return rows


def _t(x):
    return torch.tensor(x, dtype=torch.float64, device=CPU)


def test_pet_is_bit_exact_against_compiled_vic(oracle):
    worst = 0.0
    for tair, vpd, sw, nl, lai, ra, rmin, rgl, ref in oracle:
        ns = (1.0 - 0.2) * _t(sw)
        rc = V.calc_rc(_t(rmin), ns, _t(rgl), _t(tair), _t(vpd), _t(lai),
                       _t(1.0))
        got = float(V.penman(_t(tair), _t(ELEV), ns + _t(nl), _t(vpd), _t(ra),
                             rc, _t(RARC))) / STEPS_PER_DAY
        worst = max(worst, abs(got - ref))
    # Measured 4.441e-16 mm/step. Every intermediate -- scale height, pressure,
    # latent heat -- matched the compiled C to the bit; svp_slope at 25 C was
    # 2.8e-14 Pa/C off, one ulp of a 189 Pa/C value, from this CPU's exp
    # against glibc's. The line is the C's association exactly; the ulp is the
    # library's, so the bound is that ulp and not zero.
    assert worst <= 1e-15, 'PET differs from compiled VIC by %.3e mm/step' % worst
