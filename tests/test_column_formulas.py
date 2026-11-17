#!/usr/bin/env python3
"""Every formula in torchvic/core/column.py, against PRISTINE VIC 5.0.1.

The distinction in that sentence is the reason this file exists. The
repository carries its own copy of VIC under `vic/`, and that copy is not
upstream: its `runoff.c` calls a `calc_Q12()` that VIC 5.0.1 does not have.
`tests/oracle/build_ref_soil.sh` compiles the oracle against `vic/`, so the
column could agree with the oracle to machine precision and still disagree
with the model it is a port of -- which is exactly what was happening. The
oracle here is compiled against

    reference/VIC-VIC.5.0.1/vic

which is pristine upstream, and the C shim it compiles is embedded in this
file so that what was measured travels with the measurement.

What is checked, and what each check would have shown had the port been
wrong:

  * the full 480-step trajectory at two sub-step ratios, which is where the
    Brooks-Corey drainage form and the condensation branch showed up as a
    0.5-2.4 per cent drift in storage, runoff and baseflow;
  * asat, zwt and the per-layer water table over the trajectory and over a
    1000-state sweep that includes every combination of saturated layers,
    which is where an off-by-one in the water-table interpolation would show;
  * arno_evap over 126 combinations of infiltration shape, wetness and
    radiation, which the column previously did not compute at all;
  * and a set of checks that need no compiler, so the file still measures
    something on a machine without the reference tree.

Run: pytest tests/test_column_formulas.py
"""
import os
import subprocess
import sys
import tempfile

import pytest
import torch

import _paths  # noqa: E402

from torchvic.core import column, soil  # noqa: E402

#: these tests compare with VIC 5.0.1 (its source tree or image driver): the 5.0.1 core
pytestmark = pytest.mark.vic_core('5.0.1')

DT = torch.float64

# The pristine tree. Overridable, because a checkout elsewhere is a
# configuration question and not a reason to silently measure nothing.
REF_TREE = os.environ.get('VIC_REFERENCE_TREE', _paths.VIC501_TREE)

# The soil the shim uses. Kept here and in the C as two literals rather than
# passed across, so that a change to one and not the other fails loudly
# instead of moving both sides of the comparison together.
MAXM = [42.0, 126.0, 420.0]
DEPTH = [0.10, 0.30, 1.00]
EXPT = [8.0, 10.5, 12.0]
KSAT = [250.0, 120.0, 60.0]
RESIDF = [0.04, 0.05, 0.06]
BUBBLE = [20.0, 26.0, 30.0]
B_INFILT, DS, DSMAX, WS, CEXP = 0.25, 0.02, 10.0, 0.9, 2.0
RESID_MM = [RESIDF[i] * DEPTH[i] * 1000.0 for i in range(3)]

SHIM = r"""/* Oracle for torchvic/core/column.py, linked against PRISTINE VIC 5.0.1. */
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <vic_driver_shared_all.h>

global_param_struct global_param;
option_struct       options;
parameters_struct   param;
size_t              NR;
size_t              NF;

void compute_runoff_and_asat(soil_con_struct *soil_con, double *moist,
                             double inflow, double *A, double *runoff);

#define NL 3

static double MAXM[NL] = { 42.0, 126.0, 420.0 };
static double DEPTH[NL] = { 0.10, 0.30, 1.00 };
static double EXPT[NL] = { 8.0, 10.5, 12.0 };
static double KSAT[NL] = { 250.0, 120.0, 60.0 };
static double RESIDF[NL] = { 0.04, 0.05, 0.06 };   /* fraction */
static double BUBBLE[NL] = { 20.0, 26.0, 30.0 };
static double POROSITY[NL] = { 0.42, 0.42, 0.42 };  /* depth*por*1000 == MAXM */

static void
init_globals(int msd)
{
    initialize_options();
    initialize_global();
    initialize_parameters();
    options.Nlayer = NL;
    options.Nnode = 3;
    options.SNOW_BAND = 1;
    options.FROZEN_SOIL = false;
    options.FULL_ENERGY = false;
    options.Nfrost = 1;
    global_param.runoff_steps_per_day = 24;
    global_param.model_steps_per_day = msd;
}

static void
init_soil(soil_con_struct *s)
{
    memset(s, 0, sizeof(*s));
    s->b_infilt = 0.25;
    s->Ds = 0.02;
    s->Dsmax = 10.0;
    s->Ws = 0.9;
    s->c = 2.0;
    for (size_t i = 0; i < NL; i++) {
        s->max_moist[i] = MAXM[i];
        s->resid_moist[i] = RESIDF[i];
        s->expt[i] = EXPT[i];
        s->Ksat[i] = KSAT[i];
        s->depth[i] = DEPTH[i];
        s->bubble[i] = BUBBLE[i];
        s->porosity[i] = POROSITY[i];
        s->Wpwp[i] = 0.2 * s->max_moist[i];
        s->Wcr[i] = 0.5 * s->max_moist[i];
    }
    soil_moisture_from_water_table(s, NL);
}

static void
mode_column(int msd)
{
    soil_con_struct   soil;
    cell_data_struct  cell;
    energy_bal_struct energy;
    double            frost_fract[MAX_FROST_AREAS];
    int               nsteps = 480;

    init_globals(msd);
    init_soil(&soil);
    memset(&cell, 0, sizeof(cell));
    memset(&energy, 0, sizeof(energy));
    memset(frost_fract, 0, sizeof(frost_fract));
    frost_fract[0] = 1.0;

    for (size_t i = 0; i < NL; i++) {
        cell.layer[i].moist = 0.5 * soil.max_moist[i];
        for (size_t f = 0; f < MAX_FROST_AREAS; f++) {
            cell.layer[i].ice[f] = 0.0;
        }
    }

    printf("# source: runoff(), PRISTINE VIC 5.0.1 (called)\n");
    printf("# msd=%d rspd=%d b_infilt=%.17g Ds=%.17g Dsmax=%.17g Ws=%.17g c=%.17g\n",
           msd, (int)global_param.runoff_steps_per_day, soil.b_infilt, soil.Ds,
           soil.Dsmax, soil.Ws, soil.c);
    printf("step,ppt,evap0,evap1,evap2,moist0,moist1,moist2,runoff,baseflow,asat,zwt,zwt0,zwt1,zwt2\n");

    for (int t = 0; t < nsteps; t++) {
        double ppt = 0.0;
        if (t % 48 < 6) ppt = 2.0 + 0.5 * (t % 7);
        if (t >= 200 && t < 240) ppt = 0.0;
        if (t >= 300 && t < 306) ppt = 40.0;

        double evap[NL];
        evap[0] = 0.08 + 0.05 * ((t % 24) < 12 ? 1.0 : 0.0);
        evap[1] = 0.04;
        evap[2] = 0.01;
        /* a stretch with demand far beyond supply, to exercise the floors */
        if (t >= 240 && t < 260) { evap[0] = 6.0; evap[1] = 4.0; evap[2] = 2.0; }
        /* condensation onto a layer that the previous stretch left sitting
           at its residual content: evap <= 0 takes runoff.c's `else` branch
           (line 136-140), which does NOT zero it. */
        if (t >= 260 && t < 275) { evap[0] = -0.05; evap[1] = -0.02; evap[2] = -0.01; }
        for (size_t i = 0; i < NL; i++) cell.layer[i].evap = evap[i];

        cell.runoff = 0.0;
        cell.baseflow = 0.0;
        cell.inflow = ppt;

        int rc = runoff(&cell, &energy, &soil, ppt, frost_fract, 3);
        if (rc == ERROR) { fprintf(stderr, "ERROR at %d\n", t); exit(1); }

        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               t, ppt, evap[0], evap[1], evap[2],
               cell.layer[0].moist, cell.layer[1].moist, cell.layer[2].moist,
               cell.runoff, cell.baseflow, cell.asat, cell.zwt,
               cell.layer[0].zwt, cell.layer[1].zwt, cell.layer[2].zwt);
    }
}

/* Brooks-Corey drainage exactly as runoff.c:202-212 evaluates it, swept. */
static void
mode_q12(void)
{
    init_globals(24);
    printf("Ksat,tmp_liq,resid,max_moist,expt,liq,Q12\n");
    double ks[] = { 1.0, 10.0, 60.0, 250.0, 1257.0959 };
    double ex[] = { 4.0, 8.0, 11.7, 14.66, 25.0 };
    double fr[] = { 0.0, 1e-9, 1e-6, 0.001, 0.05, 0.3, 0.7, 0.999, 1.0 };
    double maxm = 126.0, resid = 6.3;
    for (size_t a = 0; a < sizeof(ks)/sizeof(*ks); a++)
    for (size_t b = 0; b < sizeof(ex)/sizeof(*ex); b++)
    for (size_t c = 0; c < sizeof(fr)/sizeof(*fr); c++) {
        double Ksat = ks[a] / global_param.runoff_steps_per_day;
        double tmp_liq = resid + fr[c] * (maxm - resid);
        double liq = tmp_liq;   /* gate is on liq, see runoff.c:202 */
        double q;
        if (liq > resid) {
            q = Ksat * pow(((tmp_liq - resid) / (maxm - resid)), ex[b]);
        } else {
            q = 0.;
        }
        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               Ksat, tmp_liq, resid, maxm, ex[b], liq, q);
    }
}

/* arno_evap(), called. */
static void
mode_arno_evap(void)
{
    init_globals(24);
    layer_data_struct layer[NL];
    double frost_fract[MAX_FROST_AREAS];
    memset(frost_fract, 0, sizeof(frost_fract));
    frost_fract[0] = 1.0;

    printf("# source: arno_evap(), PRISTINE VIC 5.0.1 (called)\n");
    printf("b_infilt,moist,max_moist,depth1,resid_frac,rad,air_temp,vpd,ra,delta_t,Epot,evap\n");
    double bv[] = { 0.01, 0.1, 0.25, 0.5, 1.0, 2.0 };
    double fv[] = { 0.0, 0.05, 0.2, 0.5, 0.8, 0.99, 1.0 };
    double rv[] = { 50.0, 200.0, 500.0 };
    double max_moist = 42.0, depth1 = 0.10, resid_frac = 0.04;
    double air_temp = 20.0, vpd = 1200.0, ra = 40.0, elev = 1400.0;
    double delta_t = 10800.0;
    for (size_t a = 0; a < sizeof(bv)/sizeof(*bv); a++)
    for (size_t b = 0; b < sizeof(fv)/sizeof(*fv); b++)
    for (size_t c = 0; c < sizeof(rv)/sizeof(*rv); c++) {
        memset(layer, 0, sizeof(layer));
        layer[0].moist = fv[b] * max_moist;
        double Epot = penman(air_temp, elev, rv[c], vpd, ra, 0.0,
                             param.SOIL_RARC) * delta_t / CONST_CDAY;
        arno_evap(layer, rv[c], air_temp, vpd, depth1, max_moist, elev,
                  bv[a], ra, delta_t, resid_frac, frost_fract);
        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               bv[a], layer[0].moist == 0 ? fv[b]*max_moist : fv[b]*max_moist,
               max_moist, depth1, resid_frac, rv[c], air_temp, vpd, ra,
               delta_t, Epot, layer[0].evap);
    }
}

/* wrap_compute_zwt() and compute_zwt(), called, over a moisture grid that
   includes full saturation of every combination of layers. */
static void
mode_zwt(void)
{
    soil_con_struct  soil;
    cell_data_struct cell;
    init_globals(24);
    init_soil(&soil);

    printf("moist0,moist1,moist2,zwt,zwt0,zwt1,zwt2,wetness\n");
    double f[] = { 0.0, 0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0 };
    size_t n = sizeof(f)/sizeof(*f);
    for (size_t a = 0; a < n; a++)
    for (size_t b = 0; b < n; b++)
    for (size_t c = 0; c < n; c++) {
        memset(&cell, 0, sizeof(cell));
        double ff[3] = { f[a], f[b], f[c] };
        for (size_t j = 0; j < NL; j++) {
            double rm = soil.resid_moist[j] * soil.depth[j] * MM_PER_M;
            cell.layer[j].moist = rm + ff[j] * (soil.max_moist[j] - rm);
        }
        wrap_compute_zwt(&soil, &cell);
        /* vic_run.c:365-371, transcribed (it is not a callable function). */
        double wet = 0;
        for (size_t j = 0; j < NL; j++) {
            wet += (cell.layer[j].moist - soil.Wpwp[j]) /
                   (soil.porosity[j] * soil.depth[j] * MM_PER_M - soil.Wpwp[j]);
        }
        wet /= NL;
        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               cell.layer[0].moist, cell.layer[1].moist, cell.layer[2].moist,
               cell.zwt, cell.layer[0].zwt, cell.layer[1].zwt,
               cell.layer[2].zwt, wet);
    }
}

int
main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: ref_col <mode> [msd]\n"); return 2; }
    if (strcmp(argv[1], "column") == 0) mode_column(argc > 2 ? atoi(argv[2]) : 24);
    else if (strcmp(argv[1], "q12") == 0) mode_q12();
    else if (strcmp(argv[1], "arno_evap") == 0) mode_arno_evap();
    else if (strcmp(argv[1], "zwt") == 0) mode_zwt();
    else { fprintf(stderr, "unknown mode\n"); return 2; }
    return 0;
}
"""


def _tensor(v, n=1):
    return torch.tensor([v] * n, dtype=DT)


def _rows(v, n=1):
    return torch.tensor([v] * n, dtype=DT)


def make_params(n=1, model_steps_per_day=8, with_zwt=True, with_wilt=True):
    p = column.SoilParams(
        b_infilt=_tensor(B_INFILT, n), Ds=_tensor(DS, n),
        Dsmax=_tensor(DSMAX, n), Ws=_tensor(WS, n), c=_tensor(CEXP, n),
        expt=_rows(EXPT, n), Ksat=_rows(KSAT, n),
        max_moist=_rows(MAXM, n), resid_moist=_rows(RESID_MM, n),
        runoff_steps_per_day=24, model_steps_per_day=model_steps_per_day,
        depth=_rows(DEPTH, n) if with_zwt else None,
        bubble=_rows(BUBBLE, n) if with_zwt else None)
    if with_wilt:
        # Wpwp = Wpwp_FRACT * max_moist, read_soilparam.c:646. The shim uses
        # Wpwp_FRACT = 0.2.
        p.w_wilt_mm = _rows([0.2 * m for m in MAXM], n)
    return p


_ORACLE = {}


def oracle():
    """Compile the embedded shim against the pristine tree; cache the binary.

    Skips rather than fails when the tree or a compiler is missing: a machine
    without them cannot make this measurement, and pretending otherwise is
    the failure mode this whole file exists to stop.
    """
    if 'path' in _ORACLE:
        if _ORACLE['path'] is None:
            pytest.skip(_ORACLE['why'])
        return _ORACLE['path']

    def give_up(why):
        _ORACLE['path'] = None
        _ORACLE['why'] = why
        pytest.skip(why)

    if not os.path.isdir(os.path.join(REF_TREE, 'vic_run', 'src')):
        give_up('pristine VIC 5.0.1 tree not found at %s '
                '(set VIC_REFERENCE_TREE)' % REF_TREE)

    tmp = tempfile.mkdtemp(prefix='ref_col_')
    src = os.path.join(tmp, 'ref_col.c')
    with open(src, 'w') as fh:
        fh.write(SHIM)
    out = os.path.join(tmp, 'ref_col')

    run_src = sorted(
        os.path.join(REF_TREE, 'vic_run', 'src', f)
        for f in os.listdir(os.path.join(REF_TREE, 'vic_run', 'src'))
        if f.endswith('.c'))
    shared = [os.path.join(REF_TREE, 'drivers', 'shared_all', 'src', f)
              for f in ('initialize_options.c', 'initialize_global.c',
                        'initialize_parameters.c', 'vic_log.c',
                        'open_file.c', 'soil_moisture_from_water_table.c')]
    inc = ['-I' + os.path.join(REF_TREE, 'drivers', 'classic', 'include'),
           '-I' + os.path.join(REF_TREE, 'drivers', 'shared_all', 'include'),
           '-I' + os.path.join(REF_TREE, 'vic_run', 'include')]
    # -fcommon: VIC 5.0.1 predates gcc 10's default of -fno-common and
    # declares several globals in more than one translation unit.
    cmd = ([os.environ.get('CC', 'cc'), '-o', out, src] + run_src + shared +
           inc + ['-DLOG_LVL=25', '-DGIT_VERSION="ref"', '-DUSERNAME="t"',
                  '-DHOSTNAME="t"', '-O2', '-std=c99', '-w', '-fcommon',
                  '-lm'])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        give_up('could not build the pristine oracle: %s'
                % proc.stderr.strip()[-400:])
    _ORACLE['path'] = out
    return out


def oracle_rows(*args):
    import csv
    out = subprocess.run([oracle()] + list(args), capture_output=True,
                         text=True, check=True).stdout
    body = [l for l in out.splitlines() if not l.startswith('#')]
    return list(csv.DictReader(body))


def colof(rows, name):
    return torch.tensor([float(r[name]) for r in rows], dtype=DT)


# ---------------------------------------------------------------------------
# The drainage law
# ---------------------------------------------------------------------------

def test_drainage_is_the_explicit_rate_not_the_4x_closed_form():
    """runoff.c:203-208 is Ksat * S**expt, evaluated once per sub-step.

    Fails if `column` goes back to `soil.calc_Q12`, which is VIC 4.0.x's
    analytic integral of the same law and is what the repository's modified
    `vic/` tree still calls. The second assertion is the one that matters:
    the two forms are not numerically close at a realistic Ksat, so an
    agreement test against either is a test against a specific VIC version.
    """
    Ksat = torch.tensor([250.0 / 24.0], dtype=DT)
    resid = torch.tensor([4.0], dtype=DT)
    maxm = torch.tensor([42.0], dtype=DT)
    expt = torch.tensor([8.0], dtype=DT)
    liq = torch.tensor([30.0], dtype=DT)

    want = Ksat * ((liq - resid) / (maxm - resid)) ** expt
    got = column._brooks_corey_drainage(Ksat, liq, resid, maxm, expt)
    assert torch.allclose(got, want, rtol=0, atol=1e-15)

    closed = soil.calc_Q12(Ksat, liq, resid, maxm, expt)
    assert abs(float(got - closed)) > 0.05 * float(got), (
        'the 4.0.x closed form and the 5.0.1 explicit rate must be '
        'distinguishable here, or this test proves nothing: got %.6g '
        'closed %.6g' % (float(got), float(closed)))


def test_drainage_gate_is_on_liq_not_on_the_evap_adjusted_moisture():
    """runoff.c:202 gates on `liq`, the moisture before evaporation.

    The two gates agree everywhere except under condensation: evap < 0 lifts
    tmp_liq above residual while liq is still at or below it, and VIC drains
    nothing there. Gating on tmp_liq instead drains a layer that has no water
    to give.
    """
    p = make_params(model_steps_per_day=24)
    # Layer 0 exactly at residual, condensation onto it.
    state = torch.tensor([[RESID_MM[0], 0.5 * MAXM[1], 0.5 * MAXM[2]]],
                         dtype=DT)
    evap = torch.tensor([[-2.0, 0.0, 0.0]], dtype=DT)
    ppt = torch.zeros(1, dtype=DT)
    new, ro, bf = column.step(state, ppt, evap, p)
    # All 2 mm of condensate stays in layer 0: no drainage, no runoff.
    assert float(ro) == 0.0
    assert abs(float(new[0, 0]) - (RESID_MM[0] + 2.0)) < 1e-12


def test_condensation_onto_a_dry_layer_is_not_discarded():
    """runoff.c:108 guards the availability limit with `if (evap > 0)`.

    A layer at its residual content has no water to give, so a positive
    demand is zeroed; a negative one is a deposit and is kept. Zeroing both
    silently loses the deposit.
    """
    p = make_params(model_steps_per_day=24)
    state = torch.tensor([[RESID_MM[0], RESID_MM[1], RESID_MM[2]]], dtype=DT)
    ppt = torch.zeros(1, dtype=DT)

    gained, _, _ = column.step(state, ppt,
                               torch.tensor([[-1.0, 0.0, 0.0]], dtype=DT), p)
    assert float(gained[0, 0]) > float(state[0, 0]) + 0.99

    # and the positive-demand half of the same branch still holds
    held, _, _ = column.step(state, ppt,
                             torch.tensor([[1.0, 0.0, 0.0]], dtype=DT), p)
    assert abs(float(held[0, 0]) - RESID_MM[0]) < 1e-12


def test_drainage_needs_no_floor_and_stays_differentiable_when_dry():
    """The 5.0.1 form has no negative power, so no floor can shift a value.

    `soil.calc_Q12` needed Q12_AVAIL_FLOOR and Q12_INNER_FLOOR because it
    raises (moist - resid) to 1 - expt, which is a large negative exponent.
    The explicit rate raises the same quantity to +expt, whose limit at the
    residual content is zero. The check is that the value is exactly the
    formula at a state 1e-12 mm above residual -- a floor of 1e-6 would
    change it by 48 orders of magnitude -- and that the gradient is finite.
    """
    Ksat = torch.tensor([250.0 / 24.0], dtype=DT)
    resid = torch.tensor([4.0], dtype=DT)
    maxm = torch.tensor([42.0], dtype=DT)
    expt = torch.tensor([8.0], dtype=DT, requires_grad=True)
    liq = torch.tensor([4.0 + 1e-12], dtype=DT)

    q = column._brooks_corey_drainage(Ksat, liq, resid, maxm, expt)
    # The available water is taken from the tensor rather than written as
    # 1e-12, because 4.0 + 1e-12 is not representable and the eighth power
    # magnifies the rounding into the fourth decimal place.
    avail = float(liq) - 4.0
    want = float(Ksat) * (avail / (42.0 - 4.0)) ** 8.0
    assert want < 1e-100, 'the state must be deep in the range a floor ruins'
    assert abs(float(q) - want) <= 1e-12 * want

    # At the residual content itself the value is the limit, zero, and the
    # gradient exists rather than being NaN.
    liq0 = torch.tensor([4.0], dtype=DT, requires_grad=True)
    q0 = column._brooks_corey_drainage(Ksat, liq0, resid, maxm, expt)
    q0.backward()
    assert float(q0) == 0.0
    assert torch.isfinite(liq0.grad).all()
    assert torch.isfinite(expt.grad).all()


# ---------------------------------------------------------------------------
# The sub-step loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model_steps_per_day', [8, 24])
def test_column_trajectory_matches_pristine_vic(model_steps_per_day):
    """480 steps of runoff() from the pristine tree, state carried forward.

    runoff.c:97-98 sets runoff_steps_per_dt = runoff_steps_per_day /
    model_steps_per_day, and 8 steps a day is what the image driver actually
    runs, so the 3-sub-step case is the one the model uses and the
    1-sub-step case is the degenerate control.
    """
    rows = oracle_rows('column', str(model_steps_per_day))
    p = make_params(model_steps_per_day=model_steps_per_day)
    ppt = colof(rows, 'ppt')
    evap = torch.stack([colof(rows, 'evap%d' % l) for l in range(3)], dim=-1)

    state = torch.tensor([[0.5 * m for m in MAXM]], dtype=DT)
    got = {'moist': [], 'runoff': [], 'baseflow': []}
    for t in range(len(rows)):
        state, ro, bf = column.step(state, ppt[t:t + 1], evap[t:t + 1], p)
        got['moist'].append(state[0])
        got['runoff'].append(ro[0])
        got['baseflow'].append(bf[0])
    moist = torch.stack(got['moist'])

    for l in range(3):
        d = (moist[:, l] - colof(rows, 'moist%d' % l)).abs().max()
        assert float(d) < 1e-9, 'layer %d drifts by %.3e mm' % (l, float(d))
    for name, series in (('runoff', got['runoff']),
                         ('baseflow', got['baseflow'])):
        d = (torch.stack(series) - colof(rows, name)).abs().max()
        assert float(d) < 1e-9, '%s differs by %.3e mm' % (name, float(d))


def test_substep_count_and_scaling_follow_runoff_steps_per_day():
    """runoff.c:151-152 and :185 divide Ksat and Dsmax by runoff_steps_per_day.

    Not by runoff_steps_per_dt: the sub-step loop runs
    runoff_steps_per_day / model_steps_per_day times, each with a rate scaled
    to one twenty-fourth of a day, so a model step covers exactly its own
    share of the day however the day is divided. The check is that a dry day
    of pure drainage ends at the same state whether it is delivered as 8
    model steps of 3 sub-steps or as 24 of 1 -- which holds only if both the
    rate scaling and the sub-step count are VIC's.
    """
    p8 = make_params(model_steps_per_day=8)
    p24 = make_params(model_steps_per_day=24)
    assert p8.runoff_steps_per_day // p8.model_steps_per_day == 3
    assert p24.runoff_steps_per_day // p24.model_steps_per_day == 1

    start = torch.tensor([[0.9 * m for m in MAXM]], dtype=DT)
    zero = torch.zeros(1, dtype=DT)
    ev = torch.zeros(1, 3, dtype=DT)

    s = start
    for _ in range(8):
        s, _, _ = column.step(s, zero, ev, p8)
    a = s
    s = start
    for _ in range(24):
        s, _, _ = column.step(s, zero, ev, p24)
    assert torch.allclose(a, s, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_step_signature_stays_a_three_tuple_by_default():
    """basin.py unpacks three values; adding diagnostics must not break it."""
    p = make_params()
    state = torch.tensor([[0.5 * m for m in MAXM]], dtype=DT)
    out = column.step(state, torch.ones(1, dtype=DT),
                      torch.zeros(1, 3, dtype=DT), p)
    assert len(out) == 3
    out4 = column.step(state, torch.ones(1, dtype=DT),
                       torch.zeros(1, 3, dtype=DT), p, diagnostics=True)
    assert len(out4) == 4
    for key in ('asat', 'inflow', 'soil_wet', 'rootmoist', 'zwt',
                'zwt_layer'):
        assert key in out4[3], 'missing diagnostic %s' % key


def test_missing_inputs_give_None_rather_than_a_plausible_number():
    """A diagnostic without its parameter is absent, not invented."""
    p = make_params(with_zwt=False, with_wilt=False)
    state = torch.tensor([[0.5 * m for m in MAXM]], dtype=DT)
    _, _, _, d = column.step(state, torch.zeros(1, dtype=DT),
                             torch.zeros(1, 3, dtype=DT), p,
                             diagnostics=True)
    assert d['zwt'] is None and d['zwt_layer'] is None
    assert d['soil_wet'] is None
    assert d['rootmoist'] is None
    assert d['asat'] is not None and d['inflow'] is not None


def test_inflow_is_the_water_arriving_at_the_surface():
    """surface_fluxes.c:1087 -- cell->inflow = ppt, and OUT_INFLOW is that."""
    p = make_params()
    state = torch.tensor([[0.5 * m for m in MAXM]], dtype=DT)
    ppt = torch.tensor([3.75], dtype=DT)
    _, _, _, d = column.step(state, ppt, torch.zeros(1, 3, dtype=DT), p,
                             diagnostics=True)
    assert float(d['inflow']) == 3.75


def test_rootmoist_sums_whole_layers_not_root_weighted_ones():
    """vic_run.c:358-364 sums layer moisture wherever root[l] > 0.

    It is not a root-fraction weighting. A layer holding one per cent of the
    roots contributes its whole moisture.
    """
    moist = torch.tensor([[10.0, 20.0, 40.0]], dtype=DT)
    root = torch.tensor([[0.99, 0.01, 0.0]], dtype=DT)
    got = column.root_zone_moisture(moist, root)
    assert float(got) == 30.0


def test_asat_and_zwt_match_pristine_vic_over_the_trajectory():
    """runoff.c:380-392 recomputes asat from the state the step ends at.

    Reporting the start-of-step asat instead -- the one the surface runoff
    was computed from -- is the easy mistake, and it is what this catches.
    """
    rows = oracle_rows('column', '8')
    p = make_params(model_steps_per_day=8)
    ppt = colof(rows, 'ppt')
    evap = torch.stack([colof(rows, 'evap%d' % l) for l in range(3)], dim=-1)

    state = torch.tensor([[0.5 * m for m in MAXM]], dtype=DT)
    asat, zwt, zl = [], [], [[], [], []]
    for t in range(len(rows)):
        state, _, _, d = column.step(state, ppt[t:t + 1], evap[t:t + 1], p,
                                     diagnostics=True)
        asat.append(d['asat'][0])
        zwt.append(d['zwt'][0])
        for l in range(3):
            zl[l].append(d['zwt_layer'][0, l])

    assert float((torch.stack(asat) - colof(rows, 'asat')).abs().max()) < 1e-12
    assert float((torch.stack(zwt) - colof(rows, 'zwt')).abs().max()) < 1e-9
    for l in range(3):
        d = (torch.stack(zl[l]) - colof(rows, 'zwt%d' % l)).abs().max()
        assert float(d) < 1e-9, 'layer %d zwt differs by %.3e cm' % (
            l, float(d))


def test_water_table_and_wetness_over_a_saturation_sweep():
    """wrap_compute_zwt(), compute_zwt.c:80-126, over 1000 column states.

    The sweep includes every combination of layers at exactly their capacity,
    which is the branch the C reaches with `max_moist - moist <= DBL_EPSILON`
    and where a batched implementation is most likely to pick the wrong
    layer.
    """
    rows = oracle_rows('zwt')
    n = len(rows)
    p = make_params(n=n)
    moist = torch.stack([colof(rows, 'moist%d' % l) for l in range(3)],
                        dim=-1)
    zwt, layer_zwt = column.wrap_compute_zwt(moist, p)
    assert float((zwt - colof(rows, 'zwt')).abs().max()) < 1e-9
    for l in range(3):
        d = (layer_zwt[:, l] - colof(rows, 'zwt%d' % l)).abs().max()
        assert float(d) < 1e-9

    wet = column.soil_wetness(moist, p)
    assert float((wet - colof(rows, 'wetness')).abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# Bare soil evaporation
# ---------------------------------------------------------------------------

def test_bare_soil_evap_matches_compiled_arno_evap():
    """arno_evap(), called, over b_infilt x wetness x radiation.

    Fails outright if the column has no bare soil evaporation, which is the
    state this file was written in. Walnut Gulch's Cv sums to 1 so VIC
    reports OUT_EVAP_BARE 0.000 there; that is the dataset, not the model.
    """
    rows = oracle_rows('arno_evap')
    got = column.bare_soil_evap(
        colof(rows, 'Epot'), colof(rows, 'moist'), colof(rows, 'max_moist'),
        colof(rows, 'b_infilt'), colof(rows, 'depth1'),
        colof(rows, 'resid_frac'))
    ref = colof(rows, 'evap')
    d = (got - ref).abs().max()
    assert float(d) < 1e-12, 'bare soil evap differs by %.3e mm' % float(d)
    assert float(ref.max()) > 0.1, 'the sweep must reach non-trivial evap'


def test_bare_soil_evap_including_penman_matches_the_c():
    """The whole path, Epot included: arno_evap.c:82-84 with rc = 0.

    param.SOIL_RARC = 100.0 (initialize_parameters.c:58) and the canopy
    resistance is zero, which is the only difference from the transpiration
    call into the same penman().
    """
    rows = oracle_rows('arno_evap')
    n = len(rows)
    got = column.arno_evap(
        colof(rows, 'moist'), colof(rows, 'max_moist'),
        colof(rows, 'b_infilt'), colof(rows, 'depth1'),
        colof(rows, 'resid_frac'), colof(rows, 'air_temp'),
        torch.full((n,), 1400.0, dtype=DT), colof(rows, 'rad'),
        colof(rows, 'vpd'), colof(rows, 'ra'), colof(rows, 'delta_t'))
    d = (got - colof(rows, 'evap')).abs().max()
    assert float(d) < 1e-12


def test_bare_soil_evap_is_capped_by_available_water():
    """arno_evap.c:171-181 -- withdrawal cannot exceed the water above resid.

    And condensation is explicitly not capped (arno_evap.c:170).
    """
    one = torch.ones(1, dtype=DT)
    b = one * 0.25
    depth1 = one * 0.10
    resid_frac = one * 0.04
    max_moist = one * 42.0
    resid_mm = 0.04 * 0.10 * 1000.0

    dry = column.bare_soil_evap(one * 50.0, one * resid_mm, max_moist, b,
                                depth1, resid_frac)
    assert float(dry) == 0.0

    near = column.bare_soil_evap(one * 50.0, one * (resid_mm + 0.5),
                                 max_moist, b, depth1, resid_frac)
    assert abs(float(near) - 0.5) < 1e-12

    # Condensation onto a layer with nothing to give is not capped: it stays
    # negative and stays proportional to Epot, which is what "no limits on
    # condensation" means. Note it is still scaled by the ARNO beta, as the
    # C does -- arno_evap.c:162 applies beta before the sign test at line
    # 171 -- so the value is not Epot itself.
    cond1 = column.bare_soil_evap(one * -1.0, one * resid_mm, max_moist, b,
                                  depth1, resid_frac)
    cond2 = column.bare_soil_evap(one * -2.0, one * resid_mm, max_moist, b,
                                  depth1, resid_frac)
    assert float(cond1) < 0.0
    assert abs(float(cond2) - 2.0 * float(cond1)) < 1e-15


def test_bare_soil_evap_reaches_potential_at_saturation():
    """r0 = 0 at capacity, so beta = 1 and evap = Epot (arno_evap.c:117-119).

    The C takes a separate branch there; this checks the branch-free form
    lands on the same value, which is the whole justification for removing
    it.
    """
    one = torch.ones(1, dtype=DT)
    got = column.bare_soil_evap(one * 3.0, one * 42.0, one * 42.0,
                                one * 0.25, one * 0.10, one * 0.04)
    assert abs(float(got) - 3.0) < 1e-12


def test_a_runoff_step_that_does_not_divide_the_model_step_is_refused():
    """get_global_param.c:809-813 refuses the pair; so does SoilParams.

    runoff.c:97-98 forms the sub-step count by integer division. A remainder
    there is not a rounding, it is a sixth of the day's drainage vanishing
    with nothing said.
    """
    with pytest.raises(ValueError) as exc:
        column.SoilParams(
            b_infilt=_tensor(B_INFILT), Ds=_tensor(DS), Dsmax=_tensor(DSMAX),
            Ws=_tensor(WS), c=_tensor(CEXP), expt=_rows(EXPT),
            Ksat=_rows(KSAT), max_moist=_rows(MAXM),
            resid_moist=_rows(RESID_MM),
            runoff_steps_per_day=20, model_steps_per_day=8)
    assert 'whole multiple' in str(exc.value)
