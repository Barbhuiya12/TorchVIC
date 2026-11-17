/******************************************************************************
 * @section DESCRIPTION
 *
 * Reference oracle for torchvic/mtclim.py, linked against PRISTINE VIC 4.2.d.
 *
 * VIC 5 removed MTCLIM, so the daily-to-sub-daily disaggregation this audits
 * does not exist in the 5.0.1 tree the rest of the oracles link. It links
 * reference/VIC-VIC.4.2.d/ instead -- see tests/oracle/build_ref_mtclim.sh --
 * and calls mtclim_wrapper(), set_max_min_hour() and HourlyT() as upstream
 * ships them.
 *
 * Every mode writes CSV: a header naming the columns, then one row per day
 * or per hour, at %.17g. Seventeen significant digits round-trip an IEEE
 * double and fourteen do not, so an oracle printed at fourteen measures its
 * own printf as much as the code under test -- which in the lake oracle
 * (tests/oracle/ref_snow.c, mode_lake7) came to a 2000 m2 disagreement in
 * a surface area.
 *
 * Build: tests/oracle/build_ref_mtclim.sh
 * Usage: ref_mtclim <mode> <seed>
 *          daily   mtclim_wrapper() -- per-day cloud fraction, vapour
 *                  pressure and direct fraction
 *          hourly  the hourly shortwave mtclim_wrapper() leaves, and the
 *                  hourly air temperature HourlyT() splines through it
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <vicNl.h>

/* The globals VIC's physics reads. vicNl.c defines these and has a main(). */
option_struct       options;
global_param_struct global_param;
param_set_struct    param_set;
int                 NR;
int                 NF;
char                ref_veg_over[] = { 0 };
double              ref_veg_rarc[] = { 0 };
double              ref_veg_rmin[] = { 0 };
double              ref_veg_lai[] = { 0 };
double              ref_veg_albedo[] = { 0 };
double              ref_veg_rough[] = { 0 };
double              ref_veg_displ[] = { 0 };
double              ref_veg_wind_h[] = { 0 };
double              ref_veg_RGL[] = { 0 };
double              ref_veg_rad_atten[] = { 0 };
double              ref_veg_wind_atten[] = { 0 };
double              ref_veg_trunk_ratio[] = { 0 };
char                ref_veg_ref_crop[] = { 0 };
char                *version = "ref-4.2.d";
char                *optstring = "";

/* vicerror.c closes the model's open files before exiting; this
   harness has none. MTCLIM reaches it only on a zero annual
   precipitation with a non-zero site isohyet, which cannot happen
   here because base and site are the same value. */
void
close_files(filep_struct *fp, out_data_file_struct *o,
            filenames_struct *s)
{
    (void)fp; (void)o; (void)s;
}

/* Likewise: the model's error exit writes a state file first. */
void
Error(char *s)
{
    fprintf(stderr, "%s\n", s);
    exit(1);
}

#define NDAYS 120

/* A deterministic pseudo-random sequence, so the Python side can be handed
   the same inputs without a file passing between them. Linear congruential,
   written out rather than taken from the library so both sides agree. */
static unsigned long rng_state;
static double
rnd(void)
{
    rng_state = rng_state * 6364136223846793005UL + 1442695040888963407UL;
    return (double)((rng_state >> 33) % 1000000) / 1000000.0;
}

static void
make_inputs(int seed, double *prec, double *tmax, double *tmin)
{
    int i;
    double t;

    rng_state = (unsigned long)seed * 2654435761UL + 12345UL;
    for (i = 0; i < NDAYS; i++) {
        /* A seasonal cycle plus noise, and a precipitation series with dry
           spells, so the 90-day window and the rain-day scalar both bite. */
        t = 8.0 - 14.0 * cos(2.0 * 3.14159265 * (double)i / 365.0);
        tmax[i] = t + 4.0 + 6.0 * rnd();
        tmin[i] = t - 4.0 - 6.0 * rnd();
        prec[i] = (rnd() < 0.35) ? 25.0 * rnd() : 0.0;
    }
}

int
main(int argc, char **argv)
{
    dmy_struct *dmy;
    double     *prec, *tmax, *tmin, *tskc, *vp, *hourlyrad, *fdir, *tair;
    int        *tmaxhour, *tminhour;
    double      lat, lng, elev, annual_prcp, hour_offset;
    int         i, j, seed, start_yday;
    double      sum;

    if (argc != 3) {
        fprintf(stderr, "usage: ref_mtclim <daily|hourly> <seed>\n");
        return 2;
    }
    seed = atoi(argv[2]);

    initialize_global();
    options.SW_PREC_THRESH = 0;
    options.MTCLIM_SWE_CORR = FALSE;
    options.LW_CLOUD = LW_CLOUD_DEARDORFF;
    options.VP_ITER = VP_ITER_ALWAYS;

    lat = 44.0 + 8.0 * (double)(seed % 5);      /* 44 .. 76 N */
    lng = -121.5;
    elev = 300.0 + 400.0 * (double)(seed % 4);
    annual_prcp = 900.0;
    hour_offset = 0.0;
    start_yday = 1 + 40 * (seed % 8);

    dmy = (dmy_struct *)calloc(NDAYS * 24, sizeof(dmy_struct));
    prec = (double *)calloc(NDAYS, sizeof(double));
    tmax = (double *)calloc(NDAYS, sizeof(double));
    tmin = (double *)calloc(NDAYS, sizeof(double));
    tskc = (double *)calloc(NDAYS, sizeof(double));
    vp = (double *)calloc(NDAYS, sizeof(double));
    fdir = (double *)calloc(NDAYS, sizeof(double));
    hourlyrad = (double *)calloc(NDAYS * 24, sizeof(double));
    tair = (double *)calloc(NDAYS * 24, sizeof(double));
    tmaxhour = (int *)calloc(NDAYS, sizeof(int));
    tminhour = (int *)calloc(NDAYS, sizeof(int));

    for (i = 0; i < NDAYS; i++) {
        for (j = 0; j < 24; j++) {
            dmy[i * 24 + j].day_in_year = ((start_yday - 1 + i) % 365) + 1;
            dmy[i * 24 + j].hour = j;
        }
    }
    make_inputs(seed, prec, tmax, tmin);

    mtclim_wrapper(0, 0, hour_offset, elev, 0.0, 0.0, 0.0, 0.0,
                   annual_prcp, lat, NDAYS, dmy, prec, tmax, tmin, tskc,
                   vp, hourlyrad, fdir);

    if (!strcmp(argv[1], "daily")) {
        printf("# lat,lng,elev,annual_prcp,start_yday,seed\n");
        printf("%.17g,%.17g,%.17g,%.17g,%d,%d\n", lat, lng, elev,
               annual_prcp, start_yday, seed);
        printf("# day,yday,prec,tmax,tmin | tskc,vp,fdir,swdaily\n");
        for (i = 0; i < NDAYS; i++) {
            sum = 0;
            for (j = 0; j < 24; j++) {
                sum += hourlyrad[i * 24 + j];
            }
            printf("%d,%d,%.17g,%.17g,%.17g | %.17g,%.17g,%.17g,%.17g\n",
                   i, dmy[i * 24].day_in_year, prec[i], tmax[i], tmin[i],
                   tskc[i], vp[i], fdir[i], sum / 24.0);
        }
    }
    else if (!strcmp(argv[1], "hourly")) {
        set_max_min_hour(hourlyrad, NDAYS, tmaxhour, tminhour);
        HourlyT(1, NDAYS, tmaxhour, tmax, tminhour, tmin, tair);
        printf("# lat,lng,elev,annual_prcp,start_yday,seed\n");
        printf("%.17g,%.17g,%.17g,%.17g,%d,%d\n", lat, lng, elev,
               annual_prcp, start_yday, seed);
        printf("# day,hour | shortwave,tair,tmaxhour,tminhour\n");
        for (i = 0; i < NDAYS; i++) {
            for (j = 0; j < 24; j++) {
                printf("%d,%d | %.17g,%.17g,%d,%d\n", i, j,
                       hourlyrad[i * 24 + j], tair[i * 24 + j],
                       tmaxhour[i], tminhour[i]);
            }
        }
    }
    else if (!strcmp(argv[1], "longwave")) {
        /* calc_longwave.c, every clear-sky formulation against both cloud
           methods, over a grid of temperature, humidity and cloud fraction
           that spans what MTCLIM produces. */
        int         ti, vi, ci, lw, cl;
        double      lwout, at, vpp, tk;
        static int  lwtypes[] = { LW_TVA, LW_ANDERSON, LW_BRUTSAERT,
                                  LW_SATTERLUND, LW_IDSO, LW_PRATA };

        printf("# lw_type,lw_cloud,air_temp,vp,tskc | longwave\n");
        for (lw = 0; lw < 6; lw++) {
            for (cl = 0; cl < 2; cl++) {
                options.LW_TYPE = lwtypes[lw];
                options.LW_CLOUD = cl ? LW_CLOUD_DEARDORFF : LW_CLOUD_BRAS;
                for (ti = 0; ti < 9; ti++) {
                    at = -40.0 + 8.0 * (double)ti;
                    for (vi = 0; vi < 7; vi++) {
                        tk = at + KELVIN;
                        vpp = 20.0 + 700.0 * (double)vi;
                        for (ci = 0; ci <= 10; ci++) {
                            calc_longwave(&lwout, 0.1 * (double)ci, at, vpp);
                            printf("%d,%d,%.17g,%.17g,%.17g | %.17g\n",
                                   lw, cl, at, vpp, 0.1 * (double)ci, lwout);
                        }
                    }
                }
            }
        }
    }
    else {
        fprintf(stderr, "unknown mode %s\n", argv[1]);
        return 2;
    }
    return 0;
}
