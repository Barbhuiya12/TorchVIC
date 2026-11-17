/******************************************************************************
 * @section DESCRIPTION
 *
 * Reference oracle for the differentiable soil core.
 *
 * This program calls VIC's own runoff functions -- compute_runoff_and_asat()
 * and calc_Q12() from vic_run/src/runoff.c of pristine VIC 5.1.0 -- over a
 * grid of inputs, and writes the results as CSV.
 *
 * It exists so that torchvic can be checked against VIC's arithmetic rather
 * than against a second reading of VIC's papers. Re-deriving the variable
 * infiltration curve from Wood et al. (1992) and comparing two derivations
 * would test the derivation; calling the shipped function tests the port.
 *
 * The ARNO baseflow block is not a separate function in runoff.c -- it is
 * inline in runoff() -- so it is transcribed here from the same source, with
 * the source lines quoted, and is the one place this oracle is a reading
 * rather than a call. That is called out in the output header so it cannot be
 * mistaken for a call.
 *
 * Build: see tests/oracle/build_ref_soil.sh
 * Usage: ref_soil <mode>
 *          infilt    variable infiltration curve / saturated area
 *          q12       Brooks-Corey drainage between layers
 *          baseflow  ARNO baseflow
 *          column    full runoff() driven over a time series
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

#include <vic_driver_shared_all.h>

/* The globals the physics reads. The classic driver defines these in
   vic_classic.c, which has a main(); this harness is linked without it. */
global_param_struct global_param;
option_struct       options;
parameters_struct   param;
size_t              NR;
size_t              NF;

/* runoff.c's helpers are not declared in a public header. */
void compute_runoff_and_asat(soil_con_struct *soil_con, double *moist,
                             double inflow, double *A, double *runoff);
double calc_Q12(double Ksat, double init_moist, double resid_moist,
                double max_moist, double expt);

#define NLAYER 3

/* Pristine VIC 5.1.0 passes the root-finder residuals their arguments as a
   va_list (soil_thermal_eqn.c, SnowPackEnergyBalance.c). These two structs
   carry the same arguments by name, and the wrappers below hand them over in
   the order the C reads them with va_arg. */
typedef struct {
    double TL, TU, T0, moist, max_moist, bubble, expt, ice0;
    double A, B, C, D, E;
    int EXP_TRANS, node;
} vic_soil_thermal_eqn_args_t;

typedef struct {
    double Dt, Ra; double *Ra_used;
    double Z; double *Z0;
    double AirDens, EactAir, LongSnowIn, Lv, Press, Rain, NetShortUnder, Vpd,
           Wind;
    double OldTSurf, SnowCoverFract, SnowDepth, SnowDensity,
           SurfaceLiquidWater, SweSurfaceLayer;
    double Tair, TGrnd;
    double *AdvectedEnergy, *AdvectedSensibleHeat, *DeltaColdContent,
           *GroundFlux, *LatentHeat, *LatentHeatSub, *NetLongUnder,
           *RefreezeEnergy, *SensibleHeat, *vapor_flux, *blowing_flux,
           *surface_flux;
} vic_snow_pack_energy_bal_args_t;

static double
va_soil_thermal_eqn(double T, ...)
{
    va_list ap;
    double  r;
    va_start(ap, T);
    r = soil_thermal_eqn(T, ap);
    va_end(ap);
    return r;
}

static double
call_soil_thermal_eqn(double T, const vic_soil_thermal_eqn_args_t *a)
{
    return va_soil_thermal_eqn(T, a->TL, a->TU, a->T0, a->moist,
                               a->max_moist, a->bubble, a->expt, a->ice0,
                               a->A, a->B, a->C, a->D, a->E, a->EXP_TRANS,
                               a->node);
}

static double
va_snow_pack_energy_balance(double T, ...)
{
    va_list ap;
    double  r;
    va_start(ap, T);
    r = SnowPackEnergyBalance(T, ap);
    va_end(ap);
    return r;
}

static double
call_snow_pack_energy_balance(double T,
                              const vic_snow_pack_energy_bal_args_t *a)
{
    return va_snow_pack_energy_balance(
        T, a->Dt, a->Ra, a->Ra_used, a->Z, a->Z0, a->AirDens, a->EactAir,
        a->LongSnowIn, a->Lv, a->Press, a->Rain, a->NetShortUnder, a->Vpd,
        a->Wind, a->OldTSurf, a->SnowCoverFract, a->SnowDepth,
        a->SnowDensity, a->SurfaceLiquidWater, a->SweSurfaceLayer, a->Tair,
        a->TGrnd, a->AdvectedEnergy, a->AdvectedSensibleHeat,
        a->DeltaColdContent, a->GroundFlux, a->LatentHeat, a->LatentHeatSub,
        a->NetLongUnder, a->RefreezeEnergy, a->SensibleHeat, a->vapor_flux,
        a->blowing_flux, a->surface_flux);
}


static void
init_globals(void)
{
    initialize_options();
    initialize_global();
    initialize_parameters();
    options.Nlayer = NLAYER;
    options.Nnode = 3;
    options.SNOW_BAND = 1;
    options.FROZEN_SOIL = false;
    options.Nfrost = 1;
    global_param.runoff_steps_per_day = 24;
    global_param.model_steps_per_day = 24;
}

/* A soil column with the shape of the Stehekin sample: three layers of
   increasing depth. Values are the ones make_synthetic_case.py emits. */
static void
init_soil(soil_con_struct *s, double b_infilt, double Ds, double Dsmax,
          double Ws, double c)
{
    static double max_moist[NLAYER] = { 15.4420, 46.3270, 154.4250 };
    static double resid[NLAYER] = { 0.0, 0.0, 0.0 };
    static double expt[NLAYER] = { 14.660, 14.660, 14.660 };
    static double Ksat[NLAYER] = { 1257.0959, 1257.0959, 1257.0959 };
    static double depth[NLAYER] = { 0.1000, 0.3000, 1.5000 };

    memset(s, 0, sizeof(*s));
    s->b_infilt = b_infilt;
    s->Ds = Ds;
    s->Dsmax = Dsmax;
    s->Ws = Ws;
    s->c = c;
    for (size_t i = 0; i < NLAYER; i++) {
        s->max_moist[i] = max_moist[i];
        s->resid_moist[i] = resid[i];
        s->expt[i] = expt[i];
        s->Ksat[i] = Ksat[i];
        s->depth[i] = depth[i];
    }
}

static void
mode_infilt(void)
{
    soil_con_struct s;
    double          moist[NLAYER];
    double          A, runoff;

    printf("# source: compute_runoff_and_asat(), vic/vic_run/src/runoff.c "
           "(called, not transcribed)\n");
    printf("b_infilt,sat_frac_top,inflow,moist0,moist1,moist2,A,runoff\n");

    double bvals[] = { 0.001, 0.05, 0.2, 0.5, 1.0, 2.0 };
    double fvals[] = { 0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999, 1.0 };
    double ivals[] = { 0.0, 0.1, 1.0, 5.0, 20.0, 100.0, 500.0 };

    for (size_t bi = 0; bi < sizeof(bvals) / sizeof(*bvals); bi++) {
        init_soil(&s, bvals[bi], 0.02, 10.0, 0.9, 2.0);
        for (size_t fi = 0; fi < sizeof(fvals) / sizeof(*fvals); fi++) {
            /* Fill the top Nlayer-1 layers to the given fraction; the bottom
               layer does not enter the infiltration curve. */
            moist[0] = fvals[fi] * s.max_moist[0];
            moist[1] = fvals[fi] * s.max_moist[1];
            moist[2] = 0.5 * s.max_moist[2];
            for (size_t ii = 0; ii < sizeof(ivals) / sizeof(*ivals); ii++) {
                compute_runoff_and_asat(&s, moist, ivals[ii], &A, &runoff);
                printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                       bvals[bi], fvals[fi], ivals[ii],
                       moist[0], moist[1], moist[2], A, runoff);
            }
        }
    }
}

static void
mode_q12(void)
{
    printf("# source: calc_Q12(), vic/vic_run/src/runoff.c "
           "(called, not transcribed)\n");
    printf("Ksat,init_moist,resid_moist,max_moist,expt,Q12\n");

    double ksat[] = { 1.0, 10.0, 100.0, 1257.0959 };
    double expt[] = { 3.0, 7.5, 14.66, 25.0 };
    double maxm = 46.3270;
    double resid[] = { 0.0, 1.0, 5.0 };
    double frac[] = { 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0 };

    for (size_t k = 0; k < sizeof(ksat) / sizeof(*ksat); k++) {
        for (size_t e = 0; e < sizeof(expt) / sizeof(*expt); e++) {
            for (size_t r = 0; r < sizeof(resid) / sizeof(*resid); r++) {
                for (size_t f = 0; f < sizeof(frac) / sizeof(*frac); f++) {
                    double im = resid[r] + frac[f] * (maxm - resid[r]);
                    if (im <= resid[r]) {
                        continue;
                    }
                    double q = calc_Q12(ksat[k], im, resid[r], maxm, expt[e]);
                    printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                           ksat[k], im, resid[r], maxm, expt[e], q);
                }
            }
        }
    }
}

/******************************************************************************
 * ARNO baseflow, transcribed from runoff.c. The block is inline in runoff()
 * and has no callable entry point, so this is the one quantity the oracle
 * reads rather than calls. Source, verbatim:
 *
 *     frac = Dsmax * soil_con->Ds / soil_con->Ws;
 *     dt_baseflow = frac * rel_moist;
 *     if (rel_moist > soil_con->Ws) {
 *         frac = (rel_moist - soil_con->Ws) / (1 - soil_con->Ws);
 *         dt_baseflow += Dsmax * (1 - soil_con->Ds / soil_con->Ws) *
 *                        pow(frac, soil_con->c);
 *     }
 *     if (dt_baseflow < 0) { dt_baseflow = 0; }
 *
 * with Dsmax = soil_con->Dsmax / global_param.runoff_steps_per_day, from the
 * assignment above the sub-step loop.
 *****************************************************************************/
static void
mode_baseflow(void)
{
    printf("# source: ARNO block inside runoff(), vic/vic_run/src/runoff.c "
           "(TRANSCRIBED -- no callable entry point)\n");
    printf("Dsmax_day,Ds,Ws,c,rel_moist,baseflow\n");

    double dsmax[] = { 1.0, 10.0, 30.0 };
    double ds[] = { 0.001, 0.02, 0.3, 1.0 };
    double ws[] = { 0.3, 0.7, 0.9, 1.0 };
    double cc[] = { 1.0, 2.0, 3.0 };
    double rel[] = { 0.0, 0.1, 0.3, 0.5, 0.69, 0.7, 0.71, 0.9, 0.99, 1.0 };

    for (size_t a = 0; a < sizeof(dsmax) / sizeof(*dsmax); a++) {
        for (size_t b = 0; b < sizeof(ds) / sizeof(*ds); b++) {
            for (size_t w = 0; w < sizeof(ws) / sizeof(*ws); w++) {
                for (size_t cidx = 0; cidx < sizeof(cc) / sizeof(*cc); cidx++) {
                    for (size_t r = 0; r < sizeof(rel) / sizeof(*rel); r++) {
                        double Dsmax = dsmax[a] /
                                       global_param.runoff_steps_per_day;
                        double Ds = ds[b], Ws = ws[w], C = cc[cidx];
                        double rm = rel[r];
                        double frac = Dsmax * Ds / Ws;
                        double q = frac * rm;
                        if (rm > Ws) {
                            frac = (rm - Ws) / (1 - Ws);
                            q += Dsmax * (1 - Ds / Ws) * pow(frac, C);
                        }
                        if (q < 0) {
                            q = 0;
                        }
                        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                               dsmax[a], Ds, Ws, C, rm, q);
                    }
                }
            }
        }
    }
}


/******************************************************************************
 * Column mode: drive VIC's own runoff() over a time series.
 *
 * This is the strongest check available for torchvic/column.py. The helper
 * comparisons above test three expressions; this tests the whole layer update
 * -- drainage, the saturation cascade that pushes excess water back up
 * through the layers, the residual-moisture floor, the sub-step loop and the
 * ARNO extraction from the bottom layer -- against the function VIC actually
 * runs, with its state carried forward exactly as the model carries it.
 *
 * FROZEN_SOIL is off and Nfrost is 1, so ice is zero throughout and
 * frost_fract is a single 1.0. That is the configuration torchvic implements
 * (the soil column is checked for gradients elsewhere); the frozen path is not claimed.
 *
 * The forcing is deterministic and written to stdout with the state, so the
 * comparison needs nothing but this program's output.
 *****************************************************************************/
/******************************************************************************
 * Snow mode: VIC's closed-form snow functions, called not transcribed.
 *
 * Covers new_snow_density() and snow_albedo() from vic/vic_run/src/
 * snow_utility.c. Both SNOW_DENSITY options are swept, because
 * initialize_options.c defaults to DENS_BRAS but DENS_SNTHRM is selectable
 * and torchvic implements both.
 *
 * snow_melt(), snow_intercept() and SnowPackEnergyBalance() are deliberately
 * absent: they solve surface temperature with root_brent and are not a
 * closed-form comparison.
 *****************************************************************************/
/******************************************************************************
 * Frozen mode: VIC's closed-form soil thermal properties, called not
 * transcribed. soil_conduction.c.
 *
 * calc_soil_thermal_fluxes() and newt_raph() are deliberately absent: they
 * iterate, and docs/IMPLICIT_SOLVER_COST.md records the implicit solver
 * failing to converge on 17.58% of calls in upstream VIC.
 *****************************************************************************/
static void
mode_frozen(void)
{
    printf("# source: soil_conductivity(), volumetric_heat_capacity(), "
           "maximum_unfrozen_water(), vic/vic_run/src/soil_conduction.c "
           "(called, not transcribed)\n");
    printf("fn,arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,result\n");

    double temps[] = { -20.0, -5.0, -1.0, -0.01, 0.0, 3.0 };
    double maxm[] = { 0.2, 0.4 };
    double bubs[] = { 8.69, 14.66 };
    double expts[] = { 10.978, 12.676 };
    for (size_t a = 0; a < 6; a++)
     for (size_t b = 0; b < 2; b++)
      for (size_t c = 0; c < 2; c++)
       for (size_t d = 0; d < 2; d++) {
           printf("maximum_unfrozen_water,%.17g,%.17g,%.17g,%.17g,0,0,0,0,"
                  "%.17g\n", temps[a], maxm[b], bubs[c], expts[d],
                  maximum_unfrozen_water(temps[a], maxm[b], bubs[c],
                                         expts[d]));
       }

    /* soil_thermal_eqn(), the residual root_brent drives to zero at a frozen
       node. Called, not transcribed. */
    {
        vic_soil_thermal_eqn_args_t sa;
        double Ts[] = { -8.0, -2.0, -0.1, 0.0, 1.0 };
        double TLs[] = { -10.0, 0.0, 4.0 };
        double TUs[] = { -1.0, 6.0 };
        int nodes[] = { 1, 3 };
        int et;

        for (et = 0; et < 2; et++)
         for (size_t a = 0; a < 5; a++)
          for (size_t b = 0; b < 3; b++)
           for (size_t c = 0; c < 2; c++)
            for (size_t d = 0; d < 2; d++) {
                memset(&sa, 0, sizeof(sa));
                sa.TL = TLs[b]; sa.TU = TUs[c]; sa.T0 = -1.5;
                sa.moist = 0.3; sa.max_moist = 0.4;
                sa.bubble = 14.66; sa.expt = 12.676; sa.ice0 = 0.05;
                sa.A = 1.7e5; sa.B = 7.2e2; sa.C = 2.3e4;
                sa.D = 1.9e4; sa.E = 2.7e7;
                sa.EXP_TRANS = et; sa.node = nodes[d];
                printf("soil_thermal_eqn,%d,%d,%.17g,%.17g,%.17g"
                       "|%.17g\n", et, nodes[d], Ts[a], TLs[b], TUs[c],
                       call_soil_thermal_eqn(Ts[a], &sa));
            }
    }

    /* Unfrozen thermal-node Gauss-Seidel update, frozen_soil.c:389 and
       :459. Inline in calc_soil_thermal_fluxes and so transcribed. */
    {
        double Ts[] = { -3.0, 0.0, 2.0, 8.0 };
        double Tps[] = { -1.0, 1.0, 5.0 };
        double Tms[] = { -6.0, 0.5, 4.0 };
        double T0s[] = { -2.0, 3.0 };
        double ices[] = { 0.0, 0.08 };
        double Av = 1.7e5, Bv = 7.2e2, Cv = 2.3e4, Dv = 1.9e4, Ev = 2.7e7;
        int et, nfn;

        for (et = 0; et < 2; et++)
         for (nfn = 0; nfn < 2; nfn++)
          for (size_t a = 0; a < 4; a++)
           for (size_t b = 0; b < 3; b++)
            for (size_t c = 0; c < 3; c++)
             for (size_t d = 0; d < 2; d++)
              for (size_t e = 0; e < 2; e++) {
                  double Tj = Ts[a], Tjp1 = Tps[b], Tjm1 = Tms[c];
                  double T0j = T0s[d], icej = ices[e], out;
                  double up = nfn ? Tj : Tjp1;
                  if (et) {
                      out = (Av*T0j + Bv*(up - Tjm1) + Cv*(up + Tjm1)
                             - Dv*(up - Tjm1) + Ev*(0. - icej))
                            / (Av + 2.*Cv);
                  }
                  else {
                      out = (Av*T0j + Bv*(up - Tjm1) + Cv*up + Dv*Tjm1
                             + Ev*(0. - icej)) / (Av + Cv + Dv);
                  }
                  printf("node_update,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g"
                         "|%.17g\n", et, nfn, Tj, Tjp1, Tjm1, T0j, icej, out);
              }
    }

    /* Heat-equation coefficients, frozen_soil.c:125. Inline in
       solve_T_profile and so transcribed; both EXP_TRANS branches are
       covered, and NOFLUX for the uniform-grid branch. */
    {
        int N = 5;
        double deltat = 3600.0, Dp = 4.0;
        double Cs[5], kappa[5], Zsum[5], alpha[4], beta[4], gamma[4];
        double A[5], B[5], C[5], D[5], E[5];
        double Bexp;
        int j, et, nf;

        for (j = 0; j < N; j++) {
            Cs[j] = 1.8e6 + 1.0e5 * j;
            kappa[j] = 1.2 + 0.1 * j;
            Zsum[j] = 0.05 * j * j + 0.1 * j;
        }
        for (j = 0; j < N - 1; j++) {
            alpha[j] = 0.3 + 0.05 * j;
            beta[j] = 0.15 + 0.02 * j;
            gamma[j] = 0.12 + 0.03 * j;
        }

        for (et = 0; et < 2; et++) {
         for (nf = 0; nf < 2; nf++) {
            for (j = 0; j < N; j++) { A[j]=B[j]=C[j]=D[j]=E[j]=0.0; }
            if (et) {
                Bexp = logf(Dp + 1.) / (double) (N - 1);
                for (j = 1; j < N - 1; j++) {
                    A[j] = 4*Bexp*Bexp*Cs[j]*(Zsum[j]+1)*(Zsum[j]+1);
                    B[j] = (kappa[j+1]-kappa[j-1])*deltat;
                    C[j] = 4*deltat*kappa[j];
                    D[j] = 2*deltat*kappa[j]*Bexp;
                    E[j] = 4*Bexp*Bexp*CONST_RHOICE*CONST_LATICE*
                           (Zsum[j]+1)*(Zsum[j]+1);
                }
                /* frozen_soil.c:218-226. This block was missing, and the
                   Python it is compared against was missing it too, so the
                   pair agreed on zero where VIC has a coefficient. That is
                   what a hand transcription costs when it stands in for the
                   function it is meant to check: `ref_snow tprof`
                   calls VIC's own solve_T_profile instead. */
                if (nf) {
                    j = N - 1;
                    A[j] = 4*Bexp*Bexp*Cs[j]*(Zsum[j]+1)*(Zsum[j]+1);
                    B[j] = (kappa[j]-kappa[j-1])*deltat;
                    C[j] = 4*deltat*kappa[j];
                    D[j] = 2*deltat*kappa[j]*Bexp;
                    E[j] = 4*Bexp*Bexp*CONST_RHOICE*CONST_LATICE*
                           (Zsum[j]+1)*(Zsum[j]+1);
                }
            }
            else {
                for (j = 1; j < N - 1; j++) {
                    A[j] = Cs[j]*alpha[j-1]*alpha[j-1];
                    B[j] = (kappa[j+1]-kappa[j-1])*deltat;
                    C[j] = 2*deltat*kappa[j]*alpha[j-1]/gamma[j-1];
                    D[j] = 2*deltat*kappa[j]*alpha[j-1]/beta[j-1];
                    E[j] = CONST_RHOICE*CONST_LATICE*alpha[j-1]*alpha[j-1];
                }
                if (nf) {
                    j = N - 1;
                    A[j] = Cs[j]*alpha[j-1]*alpha[j-1];
                    B[j] = (kappa[j]-kappa[j-1])*deltat;
                    C[j] = 2*deltat*kappa[j]*alpha[j-1]/gamma[j-1];
                    D[j] = 2*deltat*kappa[j]*alpha[j-1]/beta[j-1];
                    E[j] = CONST_RHOICE*CONST_LATICE*alpha[j-1]*alpha[j-1];
                }
            }
            for (j = 0; j < N; j++) {
                printf("thermal_coeff,%d,%d,%d"
                       "|%.17g|%.17g|%.17g|%.17g|%.17g\n",
                       et, nf, j, A[j], B[j], C[j], D[j], E[j]);
            }
         }
        }
    }

    double sf[] = { 0.3, 0.5, 0.6 };
    double wf[] = { 0.0, 0.1, 0.3 };
    double if_[] = { 0.0, 0.05, 0.2 };
    double of[] = { 0.0, 0.1 };
    for (size_t a = 0; a < 3; a++)
     for (size_t b = 0; b < 3; b++)
      for (size_t c = 0; c < 3; c++)
       for (size_t d = 0; d < 2; d++) {
           printf("volumetric_heat_capacity,%.17g,%.17g,%.17g,%.17g,0,0,0,0,"
                  "%.17g\n", sf[a], wf[b], if_[c], of[d],
                  volumetric_heat_capacity(sf[a], wf[b], if_[c], of[d]));
       }

    double moists[] = { 0.0, 0.1, 0.35 };
    double quartzes[] = { 0.1, 0.693, 0.852 };
    double orgs[] = { 0.0, 0.15 };
    double sdm = 2650.0, bdm = 1569.1, sd = 2619.1, bd = 1569.1;
    for (size_t a = 0; a < 3; a++)
     for (size_t b = 0; b < 3; b++)
      for (size_t c = 0; c < 2; c++) {
          /* Wu == moist is the no-ice case the C tests by exact equality;
             both it and a partially frozen case are swept. */
          printf("soil_conductivity,%.17g,%.17g,%.17g,%.17g,0,0,0,0,%.17g\n",
                 moists[a], moists[a], quartzes[b], orgs[c],
                 soil_conductivity(moists[a], moists[a], sdm, bdm,
                                   quartzes[b], sd, bd, orgs[c]));
          double wu = moists[a] * 0.4;
          printf("soil_conductivity_frozen,%.17g,%.17g,%.17g,%.17g,0,0,0,0,"
                 "%.17g\n", moists[a], wu, quartzes[b], orgs[c],
                 soil_conductivity(moists[a], wu, sdm, bdm, quartzes[b],
                                   sd, bd, orgs[c]));
      }
}

static void
mode_snow(void)
{
    extern option_struct options;

    printf("# source: new_snow_density(), snow_albedo(), "
           "vic/vic_run/src/snow_utility.c (called, not transcribed)\n");
    printf("fn,arg0,arg1,arg2,arg3,arg4,arg5,result\n");

    double temps[] = { -40.0, -25.0, -15.0, -10.0, -5.0, -2.0, -0.5, 0.0,
                       0.5, 2.0, 5.0, 10.0, 17.7, 20.0, 30.0 };
    size_t nt = sizeof(temps) / sizeof(*temps);

    options.SNOW_DENSITY = DENS_BRAS;
    for (size_t i = 0; i < nt; i++) {
        printf("new_snow_density_bras,%.17g,0,0,0,0,0,%.17g\n",
               temps[i], new_snow_density(temps[i]));
    }
    options.SNOW_DENSITY = DENS_SNTHRM;
    for (size_t i = 0; i < nt; i++) {
        printf("new_snow_density_snthrm,%.17g,0,0,0,0,0,%.17g\n",
               temps[i], new_snow_density(temps[i]));
    }
    options.SNOW_DENSITY = DENS_BRAS;

    /* snow_albedo(new_snow, new_snow_albedo, swq, albedo, cold_content,
                   dt, last_snow, MELTING) */
    double newsnow[] = { 0.0, 0.01, 0.03, 0.5, 10.0 };
    double swqs[] = { 0.0, 0.001, 0.05, 1.0 };
    double ccs[] = { -5000.0, -1.0, 0.0, 100.0 };
    int    lasts[] = { 0, 1, 5, 30, 200 };
    double dt = 3600.0;
    double nsa = 0.85;

    /* Canopy energy balance residual, snow-laden branch,
       func_canopy_energy_bal.c. Transcribed: the C reaches it only through a
       large argument struct and a veg_var the harness does not build. Only
       the IntSnow > 0 branch is covered; the snow-free branch calls
       canopy_evap and is out of scope. */
    {
        double dt = 3600.0;
        double tfs[] = { -12.0, -3.0, -0.5, 0.0, 1.5 };
        double tcs[] = { -8.0, 0.0, 4.0 };
        double ras[] = { 30.0, 200.0 };
        double irs[] = { 0.0, 0.0005 };
        double rfs[] = { 0.0, 0.0004 };
        double AirDens = 1.2, EactAir = 400.0, Press = 101300.0;
        double LongOverIn = 260.0, LongUnderOut = 300.0, NetShortOver = 70.0;
        double EPSv = CONST_MWWV / CONST_MWAIR;

        for (size_t a = 0; a < 5; a++)
         for (size_t b = 0; b < 3; b++)
          for (size_t c = 0; c < 2; c++)
           for (size_t d = 0; d < 2; d++)
            for (size_t e = 0; e < 2; e++) {
                double Tf = tfs[a], Tc = tcs[b], Ra = ras[c];
                double IntRain = irs[d], Rainfall = rfs[e];
                double LongOverOut, NetRadiation, EsSnow, VaporMassFlux;
                double Ls, LatentHeatSub, SensibleHeat, AdvectedEnergy;
                double RestTerm, RefreezeEnergy;

                LongOverOut = calc_outgoing_longwave(Tf + CONST_TKFRZ, 0.97);
                NetRadiation = NetShortOver + LongOverIn + LongUnderOut -
                               2 * LongOverOut;
                EsSnow = svp(Tf);
                VaporMassFlux = AirDens * (EPSv / Press) *
                                (EactAir - EsSnow) / Ra / CONST_RHOFW;
                Ls = calc_latent_heat_of_sublimation(Tf);
                LatentHeatSub = Ls * VaporMassFlux * CONST_RHOFW;
                SensibleHeat = calc_sensible_heat(AirDens, Tc, Tf, Ra);
                AdvectedEnergy = (CONST_CPFW * CONST_RHOFW * Tc * Rainfall) /
                                 dt;
                RestTerm = SensibleHeat + 0.0 + LatentHeatSub +
                           NetRadiation + AdvectedEnergy;
                RefreezeEnergy = (IntRain * CONST_LATICE * CONST_RHOFW) / dt;
                if (Tf == 0.0 && RestTerm > -RefreezeEnergy) {
                    RefreezeEnergy = -RestTerm;
                    RestTerm = 0.0;
                }
                else { RestTerm += RefreezeEnergy; }

                printf("canopy_energy_bal,%.17g,%.17g,%.17g,%.17g,%.17g"
                       "|%.17g|%.17g\n", tfs[a], tcs[b], ras[c], irs[d],
                       rfs[e], RestTerm, RefreezeEnergy);
            }
    }

    /* Canopy interception mass balance, snow_intercept.c:140-270. Inline
       in snow_intercept() and so transcribed, same caveat as the other
       inline blocks. The canopy energy balance that follows it in the C is
       a root_brent solve and is not covered here. */
    {
        double mult = 0.0005, lwc = 0.035, thres = 0.0010;
        double tfs[] = { -8.0, -2.0, -0.5, 1.0 };
        double sfs[] = { 0.0, 0.0005, 0.004 };
        double rfs[] = { 0.0, 0.0008 };
        double isn[] = { 0.0, 0.0005, 0.006 };
        double irn[] = { 0.0, 0.0004 };
        double wnd[] = { 0.5, 3.0 };
        double LAI = 4.0, F = 0.8, MaxInt = 0.0006;

        for (size_t a = 0; a < 4; a++)
         for (size_t b = 0; b < 3; b++)
          for (size_t c = 0; c < 2; c++)
           for (size_t d = 0; d < 3; d++)
            for (size_t e = 0; e < 2; e++)
             for (size_t f2 = 0; f2 < 2; f2++) {
                double Tf = tfs[a], SnowFall = sfs[b], RainFall = rfs[c];
                double IntSnow = isn[d], IntRain = irn[e], Wind = wnd[f2];
                double Imax1, MaxSnowInt, DeltaSnowInt, BlownSnow = 0.0;
                double SnowThroughFall, RainThroughFall, MaxWaterInt;
                double Overload, IntRainFract, IntSnowFract;

                Imax1 = 4.0 * mult * LAI;
                if (Tf < -1.0 && Tf > -3.0) {
                    MaxSnowInt = (Tf * 3.0 / 2.0) + (11.0 / 2.0);
                }
                else if (Tf > -1.0) { MaxSnowInt = 4.0; }
                else { MaxSnowInt = 1.0; }
                MaxSnowInt *= mult * LAI;

                if (MaxSnowInt > 0) {
                    DeltaSnowInt = (1 - IntSnow / MaxSnowInt) * SnowFall;
                    if (DeltaSnowInt + IntSnow > MaxSnowInt) {
                        DeltaSnowInt = MaxSnowInt - IntSnow;
                    }
                    if (DeltaSnowInt < 0.0) { DeltaSnowInt = 0.0; }
                }
                else { DeltaSnowInt = 0.0; }

                if (Tf < -3.0 && DeltaSnowInt > 0.0 && Wind > 1.0) {
                    BlownSnow = (0.2 * Wind - 0.2) * DeltaSnowInt;
                    if (BlownSnow >= DeltaSnowInt) { BlownSnow = DeltaSnowInt; }
                    DeltaSnowInt -= BlownSnow;
                }
                else { BlownSnow = 0.0; }

                if (IntSnow + DeltaSnowInt > Imax1) { DeltaSnowInt = 0.0; }
                SnowThroughFall = (SnowFall - DeltaSnowInt) * F +
                                  SnowFall * (1 - F);
                if (SnowFall == 0 && IntSnow < thres) {
                    SnowThroughFall += IntSnow;
                    DeltaSnowInt -= IntSnow;
                }
                IntSnow += DeltaSnowInt;

                MaxWaterInt = lwc * IntSnow + MaxInt;
                if ((IntRain + RainFall) <= MaxWaterInt) {
                    /* snow_intercept.c:243-248. IntRain is a PHYSICAL depth
                       and RainThroughFall a PIXEL depth -- the C says so in
                       the comments on those two lines -- so only the
                       throughfall carries the (1 - F) fraction and the
                       store takes the whole rainfall. This line used to
                       read `IntRain + RainFall * F`, which is 25% low at
                       F = 0.75 and was the only thing saying our
                       torchvic.canopy was wrong. It was not; this was. */
                    RainThroughFall = RainFall * (1 - F);
                    IntRain = IntRain + RainFall;
                }
                else {
                    RainThroughFall = (IntRain + RainFall - MaxWaterInt) * F +
                                      (RainFall * (1 - F));
                    IntRain = MaxWaterInt;
                }
                if (RainFall == 0 && IntRain < thres) {
                    RainThroughFall += IntRain;
                    IntRain = 0.0;
                }

                if (IntRain + IntSnow > Imax1) {
                    Overload = (IntSnow + IntRain) - Imax1;
                    IntRainFract = IntRain / (IntRain + IntSnow);
                    IntSnowFract = IntSnow / (IntRain + IntSnow);
                    RainThroughFall += (Overload * IntRainFract) * F;
                    SnowThroughFall += (Overload * IntSnowFract) * F;
                    IntRain -= Overload * IntRainFract;
                    IntSnow -= Overload * IntSnowFract;
                }

                printf("canopy_intercept,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g"
                       "|%.17g|%.17g|%.17g|%.17g|%.17g\n",
                       tfs[a], sfs[b], rfs[c], isn[d], irn[e], wnd[f2],
                       IntSnow, IntRain, SnowThroughFall, RainThroughFall,
                       BlownSnow);
             }
    }

    /* Layer split and snowfall accumulation, snow_melt.c:94. Inline,
       transcribed, same caveat. */
    {
        double MAXSWE = 0.125;
        double ices[] = { 0.0, 0.05, 0.125, 0.5 };
        double stemps[] = { -12.0, -1.0, 0.0 };
        double ptemps[] = { -7.0, -0.5, 0.0 };
        double sfs[] = { 0.0, 0.001, 0.05, 0.3 };
        double airs[] = { -10.0, -0.5, 0.0, 3.0 };

        for (size_t a = 0; a < 4; a++) {
         for (size_t b = 0; b < 3; b++) {
          for (size_t c = 0; c < 3; c++) {
           for (size_t d = 0; d < 4; d++) {
            for (size_t e = 0; e < 4; e++) {
                double Ice = ices[a], surf_temp = stemps[b];
                double pack_temp = ptemps[c], SnowFall = sfs[d];
                double air_temp = airs[e];
                double SurfaceSwq, PackSwq, SurfaceCC, PackCC, SnowFallCC;
                double DeltaPackSwq, DeltaPackCC;

                if (Ice > MAXSWE) { SurfaceSwq = MAXSWE; }
                else { SurfaceSwq = Ice; }
                PackSwq = Ice - SurfaceSwq;
                SurfaceCC = CONST_VCPICE_WQ * SurfaceSwq * surf_temp;
                PackCC = CONST_VCPICE_WQ * PackSwq * pack_temp;
                if (air_temp > 0.0) { SnowFallCC = 0.0; }
                else { SnowFallCC = CONST_VCPICE_WQ * SnowFall * air_temp; }

                if (SnowFall > (MAXSWE - SurfaceSwq) &&
                    (MAXSWE - SurfaceSwq) > DBL_EPSILON) {
                    DeltaPackSwq = SurfaceSwq + SnowFall - MAXSWE;
                    if (DeltaPackSwq > SurfaceSwq) {
                        DeltaPackCC = SurfaceCC +
                                      (SnowFall - MAXSWE) / SnowFall * SnowFallCC;
                    }
                    else {
                        DeltaPackCC = DeltaPackSwq / SurfaceSwq * SurfaceCC;
                    }
                    SurfaceSwq = MAXSWE;
                    SurfaceCC += SnowFallCC - DeltaPackCC;
                    PackSwq += DeltaPackSwq;
                    PackCC += DeltaPackCC;
                }
                else {
                    SurfaceSwq += SnowFall;
                    SurfaceCC += SnowFallCC;
                }
                if (SurfaceSwq > 0.0) {
                    surf_temp = SurfaceCC / (CONST_VCPICE_WQ * SurfaceSwq);
                }
                else { surf_temp = 0.0; }
                if (PackSwq > 0.0) {
                    pack_temp = PackCC / (CONST_VCPICE_WQ * PackSwq);
                }
                else { pack_temp = 0.0; }

                printf("accumulate_snowfall,%.17g,%.17g,%.17g,%.17g,%.17g"
                       "|%.17g|%.17g|%.17g|%.17g\n",
                       ices[a], stemps[b], ptemps[c], sfs[d], airs[e],
                       SurfaceSwq, PackSwq, surf_temp, pack_temp);
            }
           }
          }
         }
        }
    }

    /* Surface melt/refreeze, snow_melt.c:169, the Qnet == 0 branch. Inline
       and transcribed, same caveat as the other two. The vapour-flux
       rescaling at snow_melt.c:194-205 is omitted on both sides: it belongs
       to the blowing-snow option, which torchvic does not implement. */
    {
        double dt = 3600.0;
        double LATV = CONST_LATICE * CONST_RHOFW;
        double res[] = { -5000.0, -50.0, 0.0, 50.0, 5000.0 };
        double sws[] = { 0.0, 0.002, 0.02 };
        double sswq[] = { 0.0, 0.05, 0.125 };
        double pswq[] = { 0.0, 0.02, 0.5 };
        double pws[] = { 0.0, 0.004 };

        for (size_t a = 0; a < 5; a++) {
         for (size_t b = 0; b < 3; b++) {
          for (size_t c = 0; c < 3; c++) {
           for (size_t d = 0; d < 3; d++) {
            for (size_t e = 0; e < 2; e++) {
                double RefreezeEnergy = res[a];
                double surf_water = sws[b], SurfaceSwq = sswq[c];
                double PackSwq = pswq[d], pack_water = pws[e];
                double Ice = SurfaceSwq + PackSwq;
                double SnowMelt = 0.0, melt_energy = 0.0, RefrozenWater;

                if (RefreezeEnergy >= 0.0) {
                    RefrozenWater = RefreezeEnergy / LATV * dt;
                    if (RefrozenWater > surf_water) {
                        RefrozenWater = surf_water;
                        RefreezeEnergy = RefrozenWater * LATV / dt;
                    }
                    melt_energy += RefreezeEnergy;
                    SurfaceSwq += RefrozenWater;
                    Ice += RefrozenWater;
                    surf_water -= RefrozenWater;
                    if (surf_water < 0.0) { surf_water = 0.0; }
                    SnowMelt = 0.0;
                }
                else {
                    SnowMelt = fabs(RefreezeEnergy) / LATV * dt;
                    melt_energy += RefreezeEnergy;
                }

                if (SnowMelt < Ice) {
                    if (SnowMelt <= PackSwq) {
                        surf_water += SnowMelt;
                        PackSwq -= SnowMelt;
                        Ice -= SnowMelt;
                    }
                    else {
                        surf_water += SnowMelt + pack_water;
                        pack_water = 0.0;
                        PackSwq = 0.0;
                        Ice -= SnowMelt;
                        SurfaceSwq = Ice;
                    }
                }
                else {
                    SnowMelt = Ice;
                    surf_water += Ice;
                    SurfaceSwq = 0.0;
                    PackSwq = 0.0;
                    Ice = 0.0;
                    melt_energy -= RefreezeEnergy;
                    if (RefreezeEnergy != 0.0) {
                        RefreezeEnergy = RefreezeEnergy / fabs(RefreezeEnergy) *
                                         SnowMelt * LATV / dt;
                    }
                    melt_energy += RefreezeEnergy;
                }

                printf("surface_melt,%.17g,%.17g,%.17g,%.17g,%.17g"
                       "|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g\n",
                       res[a], sws[b], sswq[c], pswq[d], pws[e],
                       SnowMelt, surf_water, SurfaceSwq, PackSwq, pack_water,
                       Ice, melt_energy);
            }
           }
          }
         }
        }
    }

    /* Surface/pack layer rebalance, snow_melt.c:435. Also inline and so
       also transcribed; the same caveat as pack_refreeze applies. */
    {
        double MAXSWE = 0.125;
        double sswq[] = { 0.0, 0.02, 0.125, 0.4 };
        double pswq[] = { 0.0, 0.1, 0.6 };
        double stemps[] = { -10.0, -1.0, 0.0 };
        double ptemps[] = { -6.0, -0.5, 0.0 };
        double pw = 0.003, sw = 0.001;

        /* Ice is not swept independently. The caller maintains
           Ice = PackSwq + SurfaceSwq, and breaking that invariant generates
           states the model cannot reach -- specifically Ice > MAXSWE with an
           empty pack, where snow_melt.c divides by PackSwq and returns NaN.
           Sweeping it independently tests arithmetic that never runs. */
        for (size_t b = 0; b < 4; b++) {
         for (size_t c = 0; c < 3; c++) {
          {
           for (size_t d = 0; d < 3; d++) {
            for (size_t e = 0; e < 3; e++) {
                double SurfaceSwq = sswq[b], PackSwq = pswq[c];
                double Ice = SurfaceSwq + PackSwq;
                double surf_temp = stemps[d], pack_temp = ptemps[e];
                double SurfaceCC, PackCC, swq;

                if (Ice > MAXSWE) {
                    SurfaceCC = CONST_VCPICE_WQ * surf_temp * SurfaceSwq;
                    PackCC = CONST_VCPICE_WQ * pack_temp * PackSwq;
                    if (SurfaceSwq > MAXSWE) {
                        PackCC += SurfaceCC * (SurfaceSwq - MAXSWE) / SurfaceSwq;
                        SurfaceCC -= SurfaceCC * (SurfaceSwq - MAXSWE) / SurfaceSwq;
                        PackSwq += SurfaceSwq - MAXSWE;
                        SurfaceSwq -= SurfaceSwq - MAXSWE;
                    }
                    else if (SurfaceSwq < MAXSWE) {
                        PackCC -= PackCC * (MAXSWE - SurfaceSwq) / PackSwq;
                        SurfaceCC += PackCC * (MAXSWE - SurfaceSwq) / PackSwq;
                        PackSwq -= MAXSWE - SurfaceSwq;
                        SurfaceSwq += MAXSWE - SurfaceSwq;
                    }
                    pack_temp = PackCC / (CONST_VCPICE_WQ * PackSwq);
                    surf_temp = SurfaceCC / (CONST_VCPICE_WQ * SurfaceSwq);
                }
                else {
                    PackSwq = 0.0;
                    PackCC = 0.0;
                    pack_temp = 0.0;
                }
                swq = Ice + pw + sw;
                if (swq == 0.0) { surf_temp = 0.0; pack_temp = 0.0; }

                printf("rebalance_layers,%.17g,%.17g,%.17g,%.17g,%.17g"
                       "|%.17g|%.17g|%.17g|%.17g|%.17g\n",
                       sswq[b] + pswq[c], sswq[b], pswq[c], stemps[d],
                       ptemps[e],
                       SurfaceSwq, PackSwq, surf_temp, pack_temp, swq);
            }
           }
          }
         }
        }
    }

    /* Pack water routing and refreeze. Unlike every other snow function
       checked here this block is *inline* in snow_melt() and cannot be
       called, so it is transcribed. The transcription is line for line from
       snow_melt.c and is the only part of this file that is not a direct
       call; a divergence here could therefore be a transcription error
       rather than a model disagreement, which is why the torch side is held
       to the same sequence rather than a tidier equivalent. */
    {
        double lwc = 0.035;
        double sws[] = { 0.0, 0.001, 0.01 };
        double pws[] = { 0.0, 0.002, 0.02 };
        double sswq[] = { 0.01, 0.125 };
        double pswq[] = { 0.0, 0.05, 0.4 };
        double pccs[] = { -1.0e5, -1.0e3, -1.0, 0.0 };
        double ptemps[] = { -8.0, -0.5, 0.0 };

        for (size_t a = 0; a < 3; a++) {
         for (size_t b = 0; b < 3; b++) {
          for (size_t c = 0; c < 2; c++) {
           for (size_t d = 0; d < 3; d++) {
            for (size_t e = 0; e < 4; e++) {
             for (size_t f = 0; f < 3; f++) {
                double surf_water = sws[a], pack_water = pws[b];
                double SurfaceSwq = sswq[c], PackSwq = pswq[d];
                double PackCC = pccs[e], pack_temp = ptemps[f];
                double MaxLiquidWater, melt, PackRefreezeEnergy, DeltaPackSwq;
                double Ice = 0.0;

                MaxLiquidWater = lwc * SurfaceSwq;
                if (surf_water > MaxLiquidWater) {
                    melt = surf_water - MaxLiquidWater;
                    surf_water = MaxLiquidWater;
                }
                else { melt = 0.0; }

                pack_water += melt;
                PackRefreezeEnergy = pack_water * CONST_LATICE * CONST_RHOFW;

                if (PackCC < -PackRefreezeEnergy) {
                    PackSwq += pack_water;
                    Ice += pack_water;
                    pack_water = 0.0;
                    if (PackSwq > 0.0) {
                        PackCC = PackSwq * CONST_VCPICE_WQ * pack_temp +
                                 PackRefreezeEnergy;
                        pack_temp = PackCC / (CONST_VCPICE_WQ * PackSwq);
                        if (pack_temp > 0.) { pack_temp = 0.; }
                    }
                    else { pack_temp = 0.0; }
                }
                else {
                    pack_temp = 0.0;
                    DeltaPackSwq = -PackCC / (CONST_LATICE * CONST_RHOFW);
                    pack_water -= DeltaPackSwq;
                    PackSwq += DeltaPackSwq;
                    Ice += DeltaPackSwq;
                }

                MaxLiquidWater = lwc * PackSwq;
                if (pack_water > MaxLiquidWater) {
                    melt = pack_water - MaxLiquidWater;
                    pack_water = MaxLiquidWater;
                }
                else { melt = 0.0; }

                printf("pack_refreeze,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g"
                       "|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g\n",
                       sws[a], pws[b], sswq[c], pswq[d], pccs[e], ptemps[f],
                       surf_water, pack_water, PackSwq, Ice, pack_temp, melt);
             }
            }
           }
          }
         }
        }
    }

    /* SnowPackEnergyBalance(), the residual root_brent drives to zero.
       Called, not transcribed. SPATIAL_SNOW stays false, VIC's default, so
       the advected-sensible-heat term is zero as the C sets it. */
    {
        vic_snow_pack_energy_bal_args_t ap;
        double Ra_used, Z0arr[3];
        double advE, advSH, dCC, gf, lh, lhs, nlu, rfe, sh, vf, bf, sf;

        double tsurfs[] = { -20.0, -8.0, -2.0, -0.5, 0.0 };
        double tairs[] = { -12.0, -3.0, 1.0 };
        double winds[] = { 0.8, 3.0 };
        double depths[] = { 0.05, 0.6 };
        double oldts[] = { -10.0, -1.0 };
        double liq[] = { 0.0, 0.004 };

        for (size_t a = 0; a < 5; a++) {
         for (size_t b = 0; b < 3; b++) {
          for (size_t c = 0; c < 2; c++) {
           for (size_t d = 0; d < 2; d++) {
            for (size_t e = 0; e < 2; e++) {
             for (size_t f = 0; f < 2; f++) {
                Z0arr[0] = 0.01; Z0arr[1] = 0.01; Z0arr[2] = 0.01;
                Ra_used = 0.0;
                advE = advSH = dCC = gf = lh = lhs = nlu = rfe = sh = 0.0;
                vf = bf = sf = 0.0;
                memset(&ap, 0, sizeof(ap));
                ap.Dt = 3600.0; ap.Ra = 60.0; ap.Ra_used = &Ra_used;
                ap.Z = 2.0; ap.Z0 = Z0arr;
                ap.AirDens = 1.2; ap.EactAir = 400.0; ap.LongSnowIn = 250.0;
                ap.Lv = 2.5e6; ap.Press = 101300.0; ap.Rain = 0.0;
                ap.NetShortUnder = 80.0; ap.Vpd = 200.0; ap.Wind = winds[c];
                ap.OldTSurf = oldts[e]; ap.SnowCoverFract = 1.0;
                ap.SnowDepth = depths[d]; ap.SnowDensity = 250.0;
                ap.SurfaceLiquidWater = liq[f]; ap.SweSurfaceLayer = 0.05;
                ap.Tair = tairs[b]; ap.TGrnd = 0.0;
                ap.AdvectedEnergy = &advE;
                ap.AdvectedSensibleHeat = &advSH;
                ap.DeltaColdContent = &dCC; ap.GroundFlux = &gf;
                ap.LatentHeat = &lh; ap.LatentHeatSub = &lhs;
                ap.NetLongUnder = &nlu; ap.RefreezeEnergy = &rfe;
                ap.SensibleHeat = &sh; ap.vapor_flux = &vf;
                ap.blowing_flux = &bf; ap.surface_flux = &sf;

                printf("snow_pack_energy_balance,%.17g,%.17g,%.17g,%.17g,"
                       "%.17g,%.17g,%.17g\n",
                       tsurfs[a], tairs[b], winds[c], depths[d], oldts[e],
                       liq[f], call_snow_pack_energy_balance(tsurfs[a], &ap));
             }
            }
           }
          }
         }
        }
    }

    /* svp() and latent_heat_from_snow(), the remaining inputs to the snow
       surface energy balance. */
    {
        double ts[] = { -30.0, -10.0, -0.001, 0.0, 5.0, 20.0 };
        for (size_t i = 0; i < 6; i++) {
            printf("svp,%.17g,0,0,0,0,0,%.17g\n", ts[i], svp(ts[i]));
        }

        double lh, lhs, vmf, bmf, smf;
        double dens[] = { 1.0, 1.3 };
        double eacts[] = { 100.0, 400.0, 900.0 };
        double press[] = { 80000.0, 101300.0 };
        double ras[] = { 20.0, 200.0 };
        double tms[] = { -15.0, -1.0, 0.0, 3.0 };
        double vpds[] = { 0.0, 250.0 };
        double lv = 2.5e6;
        for (size_t a = 0; a < 2; a++) {
         for (size_t b = 0; b < 3; b++) {
          for (size_t c = 0; c < 2; c++) {
           for (size_t d = 0; d < 2; d++) {
            for (size_t e = 0; e < 4; e++) {
             for (size_t f = 0; f < 2; f++) {
                bmf = 0.0;
                vmf = 0.0;
                smf = 0.0;
                latent_heat_from_snow(dens[a], eacts[b], lv, press[c],
                                      ras[d], tms[e], vpds[f],
                                      &lh, &lhs, &vmf, &bmf, &smf);
                printf("latent_heat_from_snow,%.17g,%.17g,%.17g,%.17g,"
                       "%.17g,%.17g,%.17g\n",
                       dens[a], eacts[b], press[c], ras[d], tms[e], vpds[f],
                       lh + lhs);
             }
            }
           }
          }
         }
        }
    }

    /* Surface-flux helpers used by SnowPackEnergyBalance: physics.c and
       StabilityCorrection.c. These are the pieces the snow energy balance is
       assembled from, checked separately so a disagreement in the assembly
       can be told apart from a disagreement in a part. */
    {
        double temps[] = { 233.15, 258.15, 273.15, 283.15, 300.0 };
        for (size_t i = 0; i < 5; i++) {
            printf("calc_outgoing_longwave,%.17g,0.97,0,0,0,0,%.17g\n",
                   temps[i], calc_outgoing_longwave(temps[i], 0.97));
        }

        double dens[] = { 0.9, 1.2, 1.4 };
        double t1s[] = { -20.0, -2.0, 0.0, 5.0 };
        double t0s[] = { -25.0, -5.0, 0.0, 2.0 };
        double ras[] = { 10.0, 60.0, 300.0 };
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 4; b++) {
                for (size_t c = 0; c < 4; c++) {
                    for (size_t d = 0; d < 3; d++) {
                        printf("calc_sensible_heat,%.17g,%.17g,%.17g,%.17g,"
                               "0,0,%.17g\n", dens[a], t1s[b], t0s[c],
                               ras[d],
                               calc_sensible_heat(dens[a], t1s[b], t0s[c],
                                                  ras[d]));
                    }
                }
            }
        }

        double zs[] = { 2.0, 10.0 };
        double tsurfs[] = { -20.0, -5.0, -1.0, 0.0, 2.0 };
        double tairs[] = { -15.0, -5.0, 0.0, 5.0 };
        double winds[] = { 0.5, 2.0, 10.0 };
        double z0s[] = { 0.001, 0.01 };
        for (size_t a = 0; a < 2; a++) {
            for (size_t b = 0; b < 5; b++) {
                for (size_t c = 0; c < 4; c++) {
                    for (size_t d = 0; d < 3; d++) {
                        for (size_t e = 0; e < 2; e++) {
                            printf("stability_correction,%.17g,%.17g,%.17g,"
                                   "%.17g,%.17g,0,%.17g\n",
                                   zs[a], tsurfs[b], tairs[c], winds[d],
                                   z0s[e],
                                   StabilityCorrection(zs[a], 0.0, tsurfs[b],
                                                       tairs[c], winds[d],
                                                       z0s[e]));
                        }
                    }
                }
            }
        }
    }

    /* snow_density(), DENS_BRAS only -- VIC's default. The pack state is
       passed through a snow_data_struct because that is the signature; only
       depth, swq and surf_temp are read by the BRAS branch. */
    {
        snow_data_struct sn;
        double depths[] = { 0.05, 0.2, 0.8, 2.0 };
        double sswqs[] = { 0.01, 0.05, 0.3, 0.9 };
        double news[] = { 0.0, 0.5, 5.0, 40.0 };
        double stemps[] = { -20.0, -5.0, -0.5, 0.0 };
        double airs[] = { -15.0, -3.0, 1.0 };

        for (size_t a = 0; a < 4; a++) {
            for (size_t b = 0; b < 4; b++) {
                for (size_t c = 0; c < 4; c++) {
                    for (size_t d = 0; d < 4; d++) {
                        for (size_t e = 0; e < 3; e++) {
                            memset(&sn, 0, sizeof(sn));
                            sn.depth = depths[a];
                            sn.swq = sswqs[b];
                            sn.surf_temp = stemps[d];
                            double r = snow_density(&sn, news[c], sswqs[b],
                                                    airs[e], 3600.0);
                            printf("snow_density_bras,%.17g,%.17g,%.17g,"
                                   "%.17g,%.17g,0,%.17g\n",
                                   depths[a], sswqs[b], news[c], stemps[d],
                                   airs[e], r);
                        }
                    }
                }
            }
        }
    }

    /* snow_density(), DENS_SNTHRM -- the SNTHERM89 branch,
       snow_utility.c:61-130. Unlike BRAS it reads the pack's carried density
       (snow->density seeds the update) and its liquid water (the wet-snow
       c4 factor), so the sweep carries those too: wetfrac is the liquid
       fraction of the depth, below (0.005) and above (0.03) the
       MIN_SNOW_WETFRAC = 0.01 threshold. Rows carry seven inputs, one more
       than the header's six. */
    options.SNOW_DENSITY = DENS_SNTHRM;
    {
        snow_data_struct sn;
        double dens[] = { 0.0, 60.0, 150.0, 300.0 };
        double depths[] = { 0.0, 0.05, 0.5, 1.5 };
        double sswqs[] = { 0.0, 0.02, 0.2, 0.6 };
        double news[] = { 0.0, 0.5, 8.0, 40.0 };
        double stemps[] = { -20.0, -5.0, -0.5, 0.0 };
        double airs[] = { -15.0, -3.0, 0.5 };
        double wetfracs[] = { 0.0, 0.005, 0.03 };

        for (size_t a = 0; a < 4; a++) {
            for (size_t b = 0; b < 4; b++) {
                for (size_t c = 0; c < 4; c++) {
                    for (size_t d = 0; d < 4; d++) {
                        for (size_t e = 0; e < 4; e++) {
                            for (size_t f = 0; f < 3; f++) {
                                for (size_t g = 0; g < 3; g++) {
                                    memset(&sn, 0, sizeof(sn));
                                    sn.density = dens[a];
                                    sn.depth = depths[b];
                                    sn.swq = sswqs[c];
                                    sn.surf_temp = stemps[e];
                                    sn.surf_water = wetfracs[g] * depths[b];
                                    double r = snow_density(&sn, news[d],
                                                            sswqs[c], airs[f],
                                                            3600.0);
                                    printf("snow_density_snthrm,%.17g,%.17g,"
                                           "%.17g,%.17g,%.17g,%.17g,%.17g,"
                                           "%.17g\n",
                                           dens[a], depths[b], sswqs[c],
                                           news[d], stemps[e], airs[f],
                                           wetfracs[g], r);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    options.SNOW_DENSITY = DENS_BRAS;

    for (size_t a = 0; a < sizeof(newsnow) / sizeof(*newsnow); a++) {
        for (size_t b = 0; b < sizeof(swqs) / sizeof(*swqs); b++) {
            for (size_t c = 0; c < sizeof(ccs) / sizeof(*ccs); c++) {
                for (size_t d = 0; d < sizeof(lasts) / sizeof(*lasts); d++) {
                    for (int m = 0; m < 2; m++) {
                        double r = snow_albedo(newsnow[a], nsa, swqs[b], 0.0,
                                               ccs[c], dt, lasts[d],
                                               (bool) m);
                        printf("snow_albedo,%.17g,%.17g,%.17g,%.17g,%d,%d,"
                               "%.17g\n",
                               newsnow[a], nsa, swqs[b], ccs[c], lasts[d], m,
                               r);
                    }
                }
            }
        }
    }
}

static void
mode_column(void)
{
    soil_con_struct   soil;
    cell_data_struct  cell;
    energy_bal_struct energy;
    double            frost_fract[MAX_FROST_AREAS];
    int               nsteps = 480;

    init_soil(&soil, 0.2, 0.02, 10.0, 0.9, 2.0);
    memset(&cell, 0, sizeof(cell));
    memset(&energy, 0, sizeof(energy));
    memset(frost_fract, 0, sizeof(frost_fract));
    frost_fract[0] = 1.0;

    /* Start each layer half full, as generate_default_state would. */
    for (size_t i = 0; i < NLAYER; i++) {
        cell.layer[i].moist = 0.5 * soil.max_moist[i];
        for (size_t f = 0; f < MAX_FROST_AREAS; f++) {
            cell.layer[i].ice[f] = 0.0;
        }
    }

    printf("# source: runoff(), vic/vic_run/src/runoff.c (called, not "
           "transcribed)\n");
    printf("# soil: b_infilt=%.17g Ds=%.17g Dsmax=%.17g Ws=%.17g c=%.17g\n",
           soil.b_infilt, soil.Ds, soil.Dsmax, soil.Ws, soil.c);
    printf("# max_moist=%.17g,%.17g,%.17g expt=%.17g Ksat=%.17g resid=%.17g\n",
           soil.max_moist[0], soil.max_moist[1], soil.max_moist[2],
           soil.expt[0], soil.Ksat[0], soil.resid_moist[0]);
    printf("step,ppt,evap0,evap1,evap2,moist0,moist1,moist2,runoff,baseflow\n");

    for (int t = 0; t < nsteps; t++) {
        /* A deterministic wet/dry cycle with a dry spell and two storms, so
           the trajectory visits saturation, free drainage and the residual
           floor rather than sitting in the middle of the curve. */
        double ppt = 0.0;
        if (t % 48 < 6) {
            ppt = 2.0 + 0.5 * (t % 7);
        }
        if (t >= 200 && t < 240) {
            ppt = 0.0;                       /* dry spell */
        }
        if (t >= 300 && t < 306) {
            ppt = 40.0;                      /* storm, drives saturation */
        }

        double evap[NLAYER];
        evap[0] = 0.02 + 0.01 * ((t % 24) < 12 ? 1.0 : 0.0);
        evap[1] = 0.01;
        evap[2] = 0.005;
        for (size_t i = 0; i < NLAYER; i++) {
            cell.layer[i].evap = evap[i];
        }

        cell.runoff = 0.0;
        cell.baseflow = 0.0;
        cell.inflow = ppt;

        int rc = runoff(&cell, &energy, &soil, ppt, frost_fract, 3);
        if (rc == ERROR) {
            fprintf(stderr, "runoff() returned ERROR at step %d\n", t);
            exit(1);
        }

        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               t, ppt, evap[0], evap[1], evap[2],
               cell.layer[0].moist, cell.layer[1].moist, cell.layer[2].moist,
               cell.runoff, cell.baseflow);
    }
}


/* ---------------------------------------------------------------------------
 * atmos: the forcing derivations an energy balance needs and Daymet does not
 * publish. Daymet gives precipitation, temperature, shortwave and vapour
 * pressure; VIC needs pressure, air density and latent heat as well. These
 * are VIC's own routines from vic/vic_run/src/physics.c and penman.c, called
 * here so torchvic.atmos can be measured against them rather than against a
 * reading of the Handbook of Hydrology.
 * ------------------------------------------------------------------------ */
static void
mode_atmos(void)
{
    double tair, elev, h, pz, lv, r_air, s, e;

    printf("tair|elev|scale_height|pressure|latent_heat|air_density|svp|"
           "svp_slope\n");
    for (tair = -40.0; tair <= 40.0; tair += 5.0) {
        for (elev = 0.0; elev <= 3000.0; elev += 250.0) {
            h = calc_scale_height(tair, elev);
            pz = CONST_PSTD * exp(-elev / h);
            lv = calc_latent_heat_of_vaporization(tair);
            /* VIC's OWN air_density (forcing_utils.c:88), which is what
               every driver writes into force[].density and what the whole
               energy balance sees. This line used to transcribe penman.c's
               local `r_air` -- 0.003486 * p / (275 + T) -- by hand, under
               the column name `air_density`, and so checked torchvic against
               a different function of VIC's than the one it implements. The
               port was corrected long ago; this was not, and the script
               that would have said so is not collected by pytest. */
            r_air = air_density(tair, pz);
            e = svp(tair);
            s = svp_slope(tair);
            printf("%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g\n",
                   tair, elev, h, pz, lv, r_air, e, s);
        }
    }
}


/* ---------------------------------------------------------------------------
 * canopy: VIC's canopy resistance and Penman-Monteith, the physics that
 * torchvic's single `rs` and `et_beta` were standing in for. A study
 * measured those two stand-ins as the leading controls on evapotranspiration,
 * which is the argument for porting what they replace.
 * ------------------------------------------------------------------------ */
static void
mode_canopy(void)
{
    double ns, tair, vpd, lai, gsm, rs, rgl, rc, ra, rarc, rad, elev, ev;

    printf("rs|net_short|RGL|tair|vpd|lai|gsm_inv|rc|"
           "elev|rad|ra|rarc|penman\n");
    for (rs = 0.0; rs <= 200.0; rs += 100.0) {
        for (ns = 0.0; ns <= 400.0; ns += 200.0) {
            for (tair = -10.0; tair <= 30.0; tair += 10.0) {
                for (vpd = 0.0; vpd <= 3000.0; vpd += 1500.0) {
                    for (lai = 0.0; lai <= 6.0; lai += 3.0) {
                        for (gsm = 0.25; gsm <= 1.0; gsm += 0.75) {
                            rgl = 100.0;
                            rc = calc_rc(rs, ns, rgl, tair, vpd, lai, gsm, 0);
                            elev = 1200.0;
                            rad = ns * 0.7;
                            ra = 60.0;
                            rarc = 25.0;
                            ev = penman(tair, elev, rad, vpd, ra, rc, rarc);
                            printf("%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|"
                                   "%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|"
                                   "%.17g\n",
                                   rs, ns, rgl, tair, vpd, lai, gsm, rc,
                                   elev, rad, ra, rarc, ev);
                        }
                    }
                }
            }
        }
    }
}

int
main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s infilt|q12|baseflow|column|snow|frozen\n", argv[0]);
        return 2;
    }
    init_globals();

    if (strcmp(argv[1], "infilt") == 0) {
        mode_infilt();
    }
    else if (strcmp(argv[1], "q12") == 0) {
        mode_q12();
    }
    else if (strcmp(argv[1], "baseflow") == 0) {
        mode_baseflow();
    }
    else if (strcmp(argv[1], "column") == 0) {
        mode_column();
    }
    else if (strcmp(argv[1], "snow") == 0) {
        mode_snow();
    }
    else if (strcmp(argv[1], "canopy") == 0) {
        mode_canopy();
    }
    else if (strcmp(argv[1], "atmos") == 0) {
        mode_atmos();
    }
    else if (strcmp(argv[1], "frozen") == 0) {
        mode_frozen();
    }
    else {
        fprintf(stderr, "unknown mode: %s\n", argv[1]);
        return 2;
    }
    return 0;
}
