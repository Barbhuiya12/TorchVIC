/******************************************************************************
 * @section DESCRIPTION
 *
 * Reference oracle for torchvic/snow.py, linked against PRISTINE VIC 5.0.1.
 *
 * ref_soil.c links the `vic/` tree inside this repository, which carries our
 * own CUDA modifications. For a formula audit that is the wrong authority: a
 * defect we introduced in the C would be reproduced on both sides of the
 * comparison and would look like agreement. This oracle therefore links
 * reference/VIC-VIC.5.0.1/ instead -- see tests/oracle/build_ref_snow.sh --
 * and calls snow_melt(), SnowPackEnergyBalance(), snow_density(),
 * new_snow_density() and StabilityCorrection() as upstream ships them.
 *
 * Every mode writes CSV: inputs, then '|', then outputs.
 *
 * Build: tests/oracle/build_ref_snow.sh
 * Usage: ref_snow <mode>
 *          dens   new_snow_density() over an air-temperature sweep
 *          pack   snow_density() with SNOW_DENSITY = DENS_BRAS
 *          eb     SnowPackEnergyBalance(), including the vapour mass flux
 *          melt   snow_melt() -- one whole snow step, state in and state out
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>

#include <vic_driver_shared_all.h>

/* The globals the physics reads. The classic driver defines these in
   vic_classic.c, which has a main(); this harness is linked without it. */
global_param_struct global_param;
option_struct       options;
parameters_struct   param;
size_t              NR;
size_t              NF;

static void
init_globals(void)
{
    initialize_options();
    initialize_global();
    initialize_parameters();
    options.Nlayer = 3;
    options.Nnode = 3;
    options.SNOW_BAND = 1;
    options.FROZEN_SOIL = false;
    options.Nfrost = 1;
    /* The configuration torchvic.snow claims to implement. */
    options.SNOW_DENSITY = DENS_BRAS;
    options.SPATIAL_SNOW = false;
    options.BLOWING = false;
    options.TFALLBACK = true;
    global_param.model_steps_per_day = 8;
    global_param.snow_steps_per_day = 8;
    global_param.dt = SEC_PER_DAY / 8;
}

/*****************************************************************************
 * new_snow_density() over a temperature sweep. The sweep deliberately runs
 * past +20 C: VIC 5.0.1 applies no upper limit to the Bras density, so this
 * is where a transcription that caps it shows up.
 *****************************************************************************/
static void
mode_dens(void)
{
    double t;
    int    i;

    printf("# new_snow_density(air_temp) from snow_utility.c:207\n");
    printf("air_temp|bras|snthrm\n");
    for (i = 0; i <= 60; i++) {
        t = -30.0 + i;
        options.SNOW_DENSITY = DENS_BRAS;
        double b = new_snow_density(t);
        options.SNOW_DENSITY = DENS_SNTHRM;
        double s = new_snow_density(t);
        options.SNOW_DENSITY = DENS_BRAS;
        printf("%.17g|%.17g|%.17g\n", t, b, s);
    }
}

/*****************************************************************************
 * snow_density() -- pack compaction, DENS_BRAS.
 *****************************************************************************/
static void
mode_pack(void)
{
    static double depth0[] = { 0.0, 0.02, 0.2, 0.2, 1.0, 0.5, 0.35 };
    static double sswq[] = { 0.0, 0.004, 0.05, 0.05, 0.4, 0.15, 0.09 };
    static double stemp[] = { -5.0, -1.0, 0.0, -20.0, -10.0, -0.5, -3.0 };
    static double nsnow[] = { 5.0, 0.0, 10.0, 2.0, 0.0, 30.0, 1.0 };
    static double tair[] = { -5.0, -1.0, -2.0, -20.0, -10.0, 1.0, -3.0 };
    double        dt = 3600.0;
    size_t        i;

    printf("# snow_density(snow, new_snow, sswq, Tair, dt), "
           "snow_utility.c:40, DENS_BRAS\n");
    printf("depth0,sswq,surf_temp,new_snow,air_temp,dt|density\n");
    for (i = 0; i < sizeof(depth0) / sizeof(depth0[0]); i++) {
        snow_data_struct sn;
        memset(&sn, 0, sizeof(sn));
        sn.depth = depth0[i];
        sn.swq = sswq[i];
        sn.surf_temp = stemp[i];
        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g|%.17g\n",
               depth0[i], sswq[i], stemp[i], nsnow[i], tair[i], dt,
               snow_density(&sn, nsnow[i], sswq[i], tair[i], dt));
    }
}

/* A representative snow forcing. Index selects one of a handful of cases. */
struct ebcase {
    double t_surf, old_t_surf, t_air, t_grnd, wind, net_short, long_in;
    double air_dens, eact, lv, press, vpd, rain, swe_surf, surf_water;
    double depth, density, ra, z, z0, dt;
};

static struct ebcase cases[] = {
    /* cold, dry, windy: the sublimation case */
    { -5.0, -4.0, -8.0, -2.0, 3.0, 40.0, 220.0, 1.30, 180.0, 2.5e6,
      88000.0, 120.0, 0.0, 0.05, 0.0, 0.30, 250.0, 60.0, 2.0, 0.010, 10800.0 },
    /* at freezing, rain on snow */
    { 0.0, -0.5, 2.0, 0.0, 2.0, 120.0, 300.0, 1.25, 600.0, 2.5e6,
      95000.0, 100.0, 0.002, 0.10, 0.004, 0.40, 300.0, 80.0, 2.0, 0.010,
      10800.0 },
    /* very cold, calm */
    { -15.0, -14.0, -20.0, -5.0, 0.5, 10.0, 150.0, 1.40, 60.0, 2.5e6,
      80000.0, 40.0, 0.0, 0.125, 0.0, 0.80, 320.0, 200.0, 2.0, 0.010,
      10800.0 },
    /* saturated air (Vpd == 0): the condensation guard */
    { -2.0, -2.0, -1.0, -1.0, 1.5, 20.0, 260.0, 1.28, 550.0, 2.5e6,
      90000.0, 0.0, 0.0, 0.03, 0.001, 0.12, 200.0, 70.0, 2.0, 0.010,
      10800.0 },
    /* warm air over a melting pack */
    { 0.0, 0.0, 6.0, 1.0, 4.0, 200.0, 330.0, 1.20, 800.0, 2.5e6,
      97000.0, 150.0, 0.001, 0.125, 0.010, 0.50, 350.0, 45.0, 2.0, 0.010,
      10800.0 },
    /* no wind: HUGE_RESIST substitution */
    { -3.0, -3.0, -6.0, -2.0, 0.0, 30.0, 210.0, 1.30, 200.0, 2.5e6,
      88000.0, 110.0, 0.0, 0.06, 0.0, 0.25, 240.0, 500.0, 2.0, 0.010,
      10800.0 },
};
#define NCASE (sizeof(cases) / sizeof(cases[0]))

/*****************************************************************************
 * SnowPackEnergyBalance(), through the CalcSnowPackEnergyBalance wrapper.
 * The vapour flux is an in-out and is what OUT_SUB_SNOW is built from, so it
 * is reported alongside the residual.
 *****************************************************************************/
static void
mode_eb(void)
{
    size_t i;

    printf("# CalcSnowPackEnergyBalance(), SnowPackEnergyBalance.c\n");
    printf("t_surf,old_t_surf,t_air,t_grnd,wind,net_short,long_in,air_dens,"
           "eact,lv,press,vpd,rain,swe_surf,surf_water,depth,density,ra,z,z0,"
           "dt|Qnet|refreeze_energy|vapor_flux|latent|latent_sub|sensible|"
           "net_long|ground_flux|advection|deltaCC|ra_used\n");
    for (i = 0; i < NCASE; i++) {
        struct ebcase c = cases[i];
        double Z0[3] = { c.z0, c.z0, c.z0 };
        double ra_used = 0.0;
        double advection, advected_sensible, deltaCC, grnd_flux, latent;
        double latent_sub, net_long, refreeze, sensible;
        double vapor_flux = 0.0, blowing_flux = 0.0, surface_flux = 0.0;
        double Qnet;

        Qnet = CalcSnowPackEnergyBalance(
            c.t_surf, c.dt, c.ra, &ra_used, c.z, Z0,
            c.air_dens, c.eact, c.long_in, c.lv, c.press,
            c.rain, c.net_short, c.vpd, c.wind, c.old_t_surf, 1.0,
            c.depth, c.density, c.surf_water, c.swe_surf,
            c.t_air, c.t_grnd,
            &advection, &advected_sensible, &deltaCC, &grnd_flux, &latent,
            &latent_sub, &net_long, &refreeze, &sensible,
            &vapor_flux, &blowing_flux, &surface_flux);

        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|"
               "%.17g|%.17g\n",
               c.t_surf, c.old_t_surf, c.t_air, c.t_grnd, c.wind, c.net_short,
               c.long_in, c.air_dens, c.eact, c.lv, c.press, c.vpd, c.rain,
               c.swe_surf, c.surf_water, c.depth, c.density, c.ra, c.z, c.z0,
               c.dt,
               Qnet, refreeze, vapor_flux, latent, latent_sub, sensible,
               net_long, grnd_flux, advection, deltaCC, ra_used);
    }
}

/* A whole snow step. `swq`, `pack_water`, `surf_water` are the entry state. */
struct meltcase {
    double swq, pack_water, surf_water, surf_temp, pack_temp, depth, density;
    double snowfall, rainfall;      /* mm */
    double t_air, t_grnd, wind, net_short, long_in, air_dens, eact, lv;
    double press, vpd, ra, z, z0, dt;
};

static struct meltcase mcases[] = {
    /* deep cold pack, dry windy air: sublimation with no melt */
    { 0.30, 0.0, 0.0, -5.0, -6.0, 1.0, 300.0, 0.0, 0.0,
      -8.0, -2.0, 3.0, 40.0, 220.0, 1.30, 180.0, 2.5e6, 88000.0, 120.0,
      60.0, 2.0, 0.010, 10800.0 },
    /* shallow pack, warm air: melting */
    { 0.05, 0.0, 0.0, 0.0, 0.0, 0.2, 250.0, 0.0, 0.0,
      6.0, 1.0, 4.0, 200.0, 330.0, 1.20, 800.0, 2.5e6, 97000.0, 150.0,
      45.0, 2.0, 0.010, 10800.0 },
    /* snowfall onto an existing cold pack */
    { 0.20, 0.0, 0.0, -3.0, -4.0, 0.7, 285.0, 12.0, 0.0,
      -4.0, -2.0, 2.0, 30.0, 240.0, 1.29, 300.0, 2.5e6, 90000.0, 60.0,
      70.0, 2.0, 0.010, 10800.0 },
    /* rain on a ripe pack */
    { 0.10, 0.005, 0.003, 0.0, 0.0, 0.35, 300.0, 0.0, 5.0,
      3.0, 0.5, 2.5, 100.0, 310.0, 1.24, 700.0, 2.5e6, 95000.0, 80.0,
      55.0, 2.0, 0.010, 10800.0 },
    /* thin pack that melts out completely */
    { 0.004, 0.0, 0.0, 0.0, 0.0, 0.02, 200.0, 0.0, 0.0,
      10.0, 4.0, 3.0, 250.0, 340.0, 1.20, 900.0, 2.5e6, 97000.0, 200.0,
      40.0, 2.0, 0.010, 10800.0 },
    /* very cold, calm, with a little liquid in the surface layer */
    { 0.50, 0.0, 0.002, -12.0, -14.0, 1.6, 310.0, 0.0, 0.0,
      -20.0, -6.0, 1.0, 10.0, 150.0, 1.40, 60.0, 2.5e6, 80000.0, 40.0,
      120.0, 2.0, 0.010, 10800.0 },
    /* snowfall onto bare ground */
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0,
      -6.0, -3.0, 2.0, 20.0, 200.0, 1.31, 150.0, 2.5e6, 88000.0, 90.0,
      70.0, 2.0, 0.010, 10800.0 },
};
#define NMCASE (sizeof(mcases) / sizeof(mcases[0]))

/*****************************************************************************
 * snow_melt() -- the whole step. Note the sign convention on the reported
 * sublimation: snow_melt() flips snow->vapor_flux on the way out
 * (snow_melt.c:499) so that put_data.c's OUT_SUB_SNOW is positive when mass
 * leaves the pack. The flipped value is what is printed here.
 *****************************************************************************/
static void
mode_melt(void)
{
    size_t i;

    printf("# snow_melt(), snow_melt.c:33.  sub_snow is snow->vapor_flux "
           "AFTER the sign flip at snow_melt.c:499 (positive = mass lost)\n");
    printf("swq,pack_water,surf_water,surf_temp,pack_temp,depth,density,"
           "snowfall,rainfall,t_air,t_grnd,wind,net_short,long_in,air_dens,"
           "eact,lv,press,vpd,ra,z,z0,dt"
           "|melt|swq|pack_water|surf_water|surf_temp|pack_temp|sub_snow"
           "|Qnet|refreeze_energy|mass_error|depth|density\n");
    for (i = 0; i < NMCASE; i++) {
        struct meltcase c = mcases[i];
        snow_data_struct sn;
        double Z0[3] = { c.z0, c.z0, c.z0 };
        double ra_used = 0.0, melt = 0.0, net_long = 0.0, old_tsurf = 0.0;
        double Qnet = 0.0, adv_sens = 0.0, advection = 0.0, deltaCC = 0.0;
        double gflux = 0.0, latent = 0.0, latent_sub = 0.0, refreeze = 0.0;
        double sensible = 0.0;

        memset(&sn, 0, sizeof(sn));
        sn.swq = c.swq;
        sn.pack_water = c.pack_water;
        sn.surf_water = c.surf_water;
        sn.surf_temp = c.surf_temp;
        sn.pack_temp = c.pack_temp;
        sn.depth = c.depth;
        sn.density = c.density;

        snow_melt(c.lv, c.net_short, c.t_air, c.t_grnd, Z0, c.ra, &ra_used,
                  c.t_air, 1.0, c.dt, c.air_dens, 0.0, c.long_in, c.press,
                  c.rainfall, c.snowfall, c.eact, c.vpd, c.wind, c.z,
                  &net_long, &old_tsurf, &melt, &Qnet, &adv_sens, &advection,
                  &deltaCC, &gflux, &latent, &latent_sub, &refreeze,
                  &sensible, 0, 0, 0, &sn);

        /* solve_snow.c:355-372 and 470-472 update the density and the depth
           after snow_melt() returns; snow_melt() itself never touches them.
           Those seven lines are reproduced here around a call to VIC's own
           snow_density(), so that torchvic.snow.snow_step -- which does the
           same update inside itself -- has something to be checked against.
           The surf_temp > 0 branch is unreachable after snow_melt(). */
        if (sn.swq > 0.) {
            if (sn.surf_temp <= 0) {
                sn.density = snow_density(&sn, c.snowfall, c.swq, c.t_air,
                                          c.dt);
            }
            sn.depth = CONST_RHOFW * sn.swq / sn.density;
        }
        else {
            sn.density = 0.;
            sn.depth = 0.;
        }

        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g"
               "|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g"
               "|%.17g|%.17g\n",
               c.swq, c.pack_water, c.surf_water, c.surf_temp, c.pack_temp,
               c.depth, c.density, c.snowfall, c.rainfall, c.t_air, c.t_grnd,
               c.wind, c.net_short, c.long_in, c.air_dens, c.eact, c.lv,
               c.press, c.vpd, c.ra, c.z, c.z0, c.dt,
               melt, sn.swq, sn.pack_water, sn.surf_water, sn.surf_temp,
               sn.pack_temp, sn.vapor_flux, Qnet, refreeze, sn.mass_error,
               sn.depth, sn.density);
    }
}

/******************************************************************************
 * @brief    CalcBlowingSnow() over a sweep of wind, air temperature, snow
 *           depth and surface liquid water. BLOWING must be on for the
 *           option tests inside it to take the branches torchvic.blowing
 *           implements; the defaults of the four sub-options are VIC's own
 *           (initialize_options.c:45-49).
 *****************************************************************************/
static void
mode_blow(bool spatial)
{
    double wind, tair, depth, slw, transport, flux;
    double Ls, AirDens, EactAir, ZO, Zrh, lag_one, sigma_slope, Tsnow;
    double fe, displacement, roughness, dt;
    unsigned last_snow;
    int     iw, it, id, is;

    options.BLOWING = true;
    options.BLOWING_SPATIAL_WIND = spatial;
    dt = SEC_PER_DAY / 24.;

    printf("# Dt,Tair,LastSnow,SurfaceLiquidWater,Wind,Ls,AirDens,EactAir,"
           "ZO,Zrh,snowdepth,lag_one,sigma_slope,Tsnow,fe,displacement,"
           "roughness | flux,transport\n");
    for (iw = 0; iw < 6; iw++) {
        wind = 1.0 + 3.0 * iw;
        for (it = 0; it < 5; it++) {
            tair = -25.0 + 6.0 * it;
            for (id = 0; id < 3; id++) {
                depth = 0.05 + 0.45 * id;
                for (is = 0; is < 2; is++) {
                    slw = is * 0.004;
                    last_snow = 3 + 20 * it;
                    Ls = calc_latent_heat_of_sublimation(tair);
                    AirDens = 1.2;
                    EactAir = 0.7 * svp(tair);
                    ZO = 0.005;
                    Zrh = 2.0;
                    lag_one = 0.8;
                    sigma_slope = 0.06;
                    Tsnow = tair < 0 ? tair : 0.0;
                    fe = 1000.0;
                    displacement = 0.2;
                    roughness = 0.05;
                    transport = 0.0;
                    flux = CalcBlowingSnow(dt, tair, last_snow, slw, wind, Ls,
                                           AirDens, EactAir, ZO, Zrh, depth,
                                           lag_one, sigma_slope, Tsnow, 0, 1,
                                           fe, displacement, roughness,
                                           &transport);
                    printf("%.10g,%.10g,%u,%.10g,%.10g,%.10g,%.10g,%.10g,"
                           "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,"
                           "%.10g | %.12g,%.12g\n",
                           dt, tair, last_snow, slw, wind, Ls, AirDens,
                           EactAir, ZO, Zrh, depth, lag_one, sigma_slope,
                           Tsnow, fe, displacement, roughness, flux,
                           transport);
                }
            }
        }
    }
    options.BLOWING = false;
    options.BLOWING_SPATIAL_WIND = true;
}


/******************************************************************************
 * @brief    The carbon cycle: faparl(), calc_Nscale_factors(),
 *           canopy_assimilation() in both of its modes, and
 *           compute_soil_resp(). Ncanopy is VIC's default of 3.
 *****************************************************************************/
static void
mode_carbon(void)
{
    double CanopLayerBnd[3] = { 1. / 3., 2. / 3., 1. };
    double LAIlayer[3], faPAR[3], NscaleFactor[3], rsLayer[3];
    double rc, Ci, GPP, Rdark, Rphoto, Rmaint, Rgrowth, Raut, NPP;
    double LAI, CosZen, AlbSoilPAR, Fdir, Tfoliage, SWdown, elevation, Catm;
    double MaxCarboxRate, MaxETransport, CO2Specificity;
    int    il, iz, it, ic, ctype;
    size_t cidx;

    options.Ncanopy = 3;
    options.CARBON = true;

    printf("# LAI,CosZen,AlbSoilPAR,Fdir,Tfoliage,SWdown,elevation,Catm,"
           "Ctype,MaxCarboxRate,MaxETransport,CO2Specificity | "
           "faPAR0,faPAR1,faPAR2,Nscale0,Nscale1,Nscale2,"
           "rc,Ci,GPP,Rdark,Rphoto,Rmaint,Rgrowth,Raut,NPP,"
           "rs0,rs1,rs2\n");
    for (il = 0; il < 4; il++) {
        LAI = 0.5 + 2.0 * il;
        for (iz = 0; iz < 3; iz++) {
            CosZen = 0.05 + 0.4 * iz;
            for (it = 0; it < 4; it++) {
                Tfoliage = -5.0 + 11.0 * it;
                for (ic = 0; ic < 2; ic++) {
                    ctype = ic == 0 ? PHOTO_C3 : PHOTO_C4;
                    AlbSoilPAR = 0.08;
                    Fdir = 0.7;
                    SWdown = 60.0 + 120.0 * iz;
                    elevation = 800.0;
                    Catm = 3.7e-4;
                    MaxCarboxRate = ic == 0 ? 33.0e-6 : 14.0e-6;
                    MaxETransport = ic == 0 ? 75.0e-6 : 0.0;
                    CO2Specificity = ic == 0 ? 0.0 : 9.0e-3;
                    faparl(CanopLayerBnd, LAI, AlbSoilPAR, CosZen, Fdir,
                           LAIlayer, faPAR);
                    calc_Nscale_factors(true, CanopLayerBnd, LAI, CosZen,
                                        NscaleFactor);
                    for (cidx = 0; cidx < options.Ncanopy; cidx++) {
                        rsLayer[cidx] = param.HUGE_RESIST;
                    }
                    canopy_assimilation(ctype, MaxCarboxRate, MaxETransport,
                                        CO2Specificity, NscaleFactor,
                                        Tfoliage, SWdown, faPAR, elevation,
                                        Catm, CanopLayerBnd, LAI, "ci",
                                        rsLayer, &rc, &Ci, &GPP, &Rdark,
                                        &Rphoto, &Rmaint, &Rgrowth, &Raut,
                                        &NPP);
                    printf("%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
                           "%d,%.12g,%.12g,%.12g | ",
                           LAI, CosZen, AlbSoilPAR, Fdir, Tfoliage, SWdown,
                           elevation, Catm, ic, MaxCarboxRate, MaxETransport,
                           CO2Specificity);
                    printf("%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
                           "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
                           "%.12g,%.12g,%.12g,%.12g\n",
                           faPAR[0], faPAR[1], faPAR[2], NscaleFactor[0],
                           NscaleFactor[1], NscaleFactor[2], rc, Ci, GPP,
                           Rdark, Rphoto, Rmaint, Rgrowth, Raut, NPP,
                           rsLayer[0], rsLayer[1], rsLayer[2]);
                }
            }
        }
    }
    options.CARBON = false;
}


/******************************************************************************
 * @brief    compute_soil_resp() over three thermal nodes.
 *****************************************************************************/
static void
mode_soilresp(void)
{
    double dZ[3] = { 100., 300., 1000. };   /* mm */
    double dZTot = 1400.;
    double T[3], w[3], wsave[3];
    double RhLitter, RhInter, RhSlow;
    double CLitter, CInter, CSlow, dt;
    int    it, iw, ip;
    size_t i;

    printf("# dt,T0,T1,T2,w0,w1,w2,CLitter,CInter,CSlow | "
           "RhLitter,RhInter,RhSlow\n");
    for (it = 0; it < 5; it++) {
        for (iw = 0; iw < 4; iw++) {
            for (ip = 0; ip < 2; ip++) {
                for (i = 0; i < 3; i++) {
                    T[i] = -12.0 + 8.0 * it + 2.0 * (double) i;
                    w[i] = 0.1 + 0.28 * iw;
                    wsave[i] = w[i];
                }
                CLitter = 500.0 + 300.0 * ip;
                CInter = 3000.0 + 1000.0 * ip;
                CSlow = 9000.0 + 2000.0 * ip;
                dt = 3600.0;
                compute_soil_resp(3, dZ, dZTot, dt, T, w, CLitter, CInter,
                                  CSlow, &RhLitter, &RhInter, &RhSlow);
                printf("%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
                       "%.12g,%.12g,%.12g | %.12g,%.12g,%.12g\n",
                       dt, T[0], T[1], T[2], wsave[0], wsave[1], wsave[2],
                       CLitter, CInter, CSlow, RhLitter, RhInter, RhSlow);
            }
        }
    }
}


/******************************************************************************
 * @brief    calc_rc_ps(), the canopy resistance under RC_MODE = RC_PHOTO.
 *****************************************************************************/
static void
mode_rcps(void)
{
    double CanopLayerBnd[3] = { 1. / 3., 2. / 3., 1. };
    double LAIlayer[3], faPAR[3], NscaleFactor[3], rsLayer[3], aPARLayer[3];
    double rc, LAI, CosZen, Tair, SWdown, PAR, vpd, gsm_inv, Catm;
    double MaxCarboxRate, MaxETransport, CO2Specificity;
    int    il, iz, iv, ig, ic, ctype;
    size_t cidx;

    options.Ncanopy = 3;
    options.CARBON = true;
    options.RC_MODE = RC_PHOTO;

    printf("# LAI,CosZen,Tair,SWdown,PAR,vpd,gsm_inv,Catm,Ctype,"
           "MaxCarboxRate,MaxETransport,CO2Specificity | rc,rs0,rs1,rs2,"
           "aPAR0,aPAR1,aPAR2\n");
    for (il = 0; il < 3; il++) {
        LAI = 1.0 + 2.5 * il;
        for (iz = 0; iz < 3; iz++) {
            CosZen = 0.1 + 0.35 * iz;
            for (iv = 0; iv < 3; iv++) {
                vpd = 200.0 + 1500.0 * iv;
                for (ig = 0; ig < 2; ig++) {
                    gsm_inv = 0.35 + 0.65 * ig;
                    for (ic = 0; ic < 2; ic++) {
                        ctype = ic == 0 ? PHOTO_C3 : PHOTO_C4;
                        Tair = 18.0;
                        SWdown = 80.0 + 200.0 * iz;
                        PAR = 0.45 * SWdown;
                        Catm = 3.7e-4;
                        MaxCarboxRate = ic == 0 ? 33.0e-6 : 14.0e-6;
                        MaxETransport = ic == 0 ? 75.0e-6 : 0.0;
                        CO2Specificity = ic == 0 ? 0.0 : 9.0e-3;
                        faparl(CanopLayerBnd, LAI, 0.08, CosZen, 0.7,
                               LAIlayer, faPAR);
                        calc_Nscale_factors(true, CanopLayerBnd, LAI, CosZen,
                                            NscaleFactor);
                        for (cidx = 0; cidx < options.Ncanopy; cidx++) {
                            aPARLayer[cidx] = (PAR / param.PHOTO_EPAR) *
                                              faPAR[cidx] / LAIlayer[cidx];
                            rsLayer[cidx] = param.HUGE_RESIST;
                        }
                        rc = 0.0;
                        calc_rc_ps(ctype, MaxCarboxRate, MaxETransport,
                                   CO2Specificity, NscaleFactor, Tair,
                                   SWdown, aPARLayer, 800.0, Catm,
                                   CanopLayerBnd, LAI, gsm_inv, vpd,
                                   rsLayer, &rc);
                        printf("%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
                               "%.12g,%d,%.12g,%.12g,%.12g | "
                               "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                               LAI, CosZen, Tair, SWdown, PAR, vpd, gsm_inv,
                               Catm, ic, MaxCarboxRate, MaxETransport,
                               CO2Specificity, rc, rsLayer[0], rsLayer[1],
                               rsLayer[2], aPARLayer[0], aPARLayer[1],
                               aPARLayer[2]);
                    }
                }
            }
        }
    }
    options.CARBON = false;
    options.RC_MODE = RC_JARVIS;
}


/******************************************************************************
 * @brief    compute_coszen() over latitude, day of year and time of day.
 *****************************************************************************/
static void
mode_coszen(void)
{
    double lat, lng, tzlng, cz;
    int    il, id, ih;
    unsigned short doy;
    unsigned sec;

    printf("# lat,lng,time_zone_lng,day_in_year,second | coszen\n");
    for (il = 0; il < 5; il++) {
        lat = -60.0 + 30.0 * il;
        lng = -121.5853;
        tzlng = -120.0;
        for (id = 0; id < 6; id++) {
            doy = (unsigned short) (15 + 60 * id);
            for (ih = 0; ih < 8; ih++) {
                sec = (unsigned) (ih * 10800);
                cz = compute_coszen(lat, lng, tzlng, doy, sec);
                printf("%.12g,%.12g,%.12g,%u,%u | %.12g\n",
                       lat, lng, tzlng, doy, sec, cz);
            }
        }
    }
}


/* soil_moisture_from_water_table.c:155-224 -- the LUMPED column curve, the
   last of the Nlayer+2 rows, which is what OUT_ZWT_LUMPED and the soil
   carbon cycle read. The per-layer rows have their own coverage; this row
   fills the whole column from the bottom up and is the only one that
   propagates an effective water table from one layer into the next. */
static void
mode_zwtlump(void)
{
    soil_con_struct soil;
    int             c, i, j;
    double          d[3], e[3], b[3], mm[3], rm[3];

    printf("# case,layer,depth,expt,bubble,max_moist,resid_moist\n");
    printf("# case,i | zwt_cm,moist_mm\n");
    for (c = 0; c < 6; c++) {
        /* Layer thicknesses, pore-size distributions and bubbling pressures
           spanning what a soil file holds: thin sand over thick clay, a
           uniform column, and an inverted one. */
        for (j = 0; j < 3; j++) {
            d[j] = 0.1 + 0.35 * (double)j + 0.07 * (double)c;
            e[j] = 4.0 + 2.0 * (double)((j + c) % 4);
            b[j] = 20.0 + 25.0 * (double)((j + 2 * c) % 3);
            mm[j] = d[j] * 1000.0 * (0.35 + 0.05 * (double)((j + c) % 3));
            rm[j] = 0.04 + 0.01 * (double)((j + c) % 3);
        }
        memset(&soil, 0, sizeof(soil));
        for (j = 0; j < 3; j++) {
            soil.depth[j] = d[j];
            soil.expt[j] = e[j];
            soil.bubble[j] = b[j];
            soil.max_moist[j] = mm[j];
            soil.resid_moist[j] = rm[j];
            printf("%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                   c, j, d[j], e[j], b[j], mm[j], rm[j]);
        }
        soil_moisture_from_water_table(&soil, 3);
        for (i = 0; i < MAX_ZWTVMOIST; i++) {
            printf("%d,%d | %.17g,%.17g\n", c, i,
                   soil.zwtvmoist_zwt[3 + 1][i],
                   soil.zwtvmoist_moist[3 + 1][i]);
        }
        /* compute_zwt.c:41-70 read off that row, which is exactly what
           compute_zwt.c:133 does for cell->zwt_lumped. The lumped curve is
           not monotonic in moisture, so the C's downward walk is not the
           same as a count of covered samples, and this is where that shows.
           Query it across the whole range, including outside it. */
        {
            double lo, hi, q;
            int    k;
            /* Span the row's own range, so the walk is actually exercised
               rather than answering "no water table" almost everywhere. A
               little past each end, to cover both out-of-range branches. */
            lo = hi = soil.zwtvmoist_moist[3 + 1][0];
            for (k = 1; k < MAX_ZWTVMOIST; k++) {
                if (soil.zwtvmoist_moist[3 + 1][k] < lo) {
                    lo = soil.zwtvmoist_moist[3 + 1][k];
                }
                if (soil.zwtvmoist_moist[3 + 1][k] > hi) {
                    hi = soil.zwtvmoist_moist[3 + 1][k];
                }
            }
            lo -= 0.05 * (hi - lo);
            hi += 0.05 * (hi - lo);
            for (k = 0; k <= 40; k++) {
                q = lo + (hi - lo) * (double)k / 40.0;
                printf("%d,%d ; %.17g,%.17g\n", c, k, q,
                       compute_zwt(&soil, 3 + 1, q));
            }
        }
    }
}


/* lakes.eb.c's closed-form layer: the state functions of water, the drag
   and turbulent fluxes at the lake surface, and the three albedos. Each is
   called with the same deterministic sequence the Python side builds.

   Every lake mode prints %.17g. Seventeen significant digits round-trip an
   IEEE double and fourteen do not, and a comparison fed fourteen measures
   the printf as much as the physics. It is not a formality: at %.17g the
   lake's outflow mode disagreed by 2000 m2 of surface area, because the
   lost bits flipped the sign of a 1e-10 m3 residual and VIC treats a
   negative one as a dry lake. See mode_lake7. */
static void
mode_lake(void)
{
    int      i;
    double   t, p, ts, ta, w, vp, rho, hice, evap, qsen, tfsp;
    double   snowalb, albs, albi, albw;
    unsigned last_snow;
    bool     melting;

    printf("# calc_density,specheat,freezing_point\n");
    printf("# i,t,pressure | density_diff,specheat,tfsp\n");
    for (i = 0; i <= 40; i++) {
        t = -5.0 + 1.0 * (double)i;
        p = 70000.0 + 800.0 * (double)i;
        rhoinit(&tfsp, p / PA_PER_KPA);
        printf("%d,%.17g,%.17g | %.17g,%.17g,%.17g\n", i, t, p,
               calc_density(t), specheat(t), tfsp);
    }

    printf("# lkdrag\n");
    printf("# i,Tsurf,Tair,wind,roughness,Z1 | cdr\n");
    for (i = 0; i <= 60; i++) {
        ts = 253.0 + 1.0 * (double)i;
        ta = 273.15 + 12.0 * sin(0.37 * (double)i);
        w = 0.2 + 0.31 * (double)(i % 23);
        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g | %.17g\n", i, ts, ta, w,
               (i % 2) ? param.LAKE_ZSNOW : param.LAKE_ZWATER, 10.0,
               lkdrag(ts, ta, w,
                      (i % 2) ? param.LAKE_ZSNOW : param.LAKE_ZWATER, 10.0));
    }

    printf("# latsens\n");
    printf("# i,Tsurf,Tcut,hice,tair,wind,pressure,vp,rho,wind_h "
           "| evap,qsen\n");
    for (i = 0; i <= 60; i++) {
        ts = 258.0 + 0.6 * (double)i;
        ta = -15.0 + 0.55 * (double)i;
        w = 0.3 + 0.27 * (double)(i % 19);
        p = 88000.0 + 130.0 * (double)i;
        vp = 120.0 + 38.0 * (double)(i % 17);
        rho = 1.15 - 0.004 * (double)(i % 11);
        hice = (i % 3 == 0) ? 0.0 : 0.02 * (double)(i % 7);
        latsens(ts, CONST_TKFRZ, hice, ta, w, p, vp, rho, &evap, &qsen, 10.0);
        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g "
               "| %.17g,%.17g\n", i, ts, CONST_TKFRZ, hice, ta, w, p, vp,
               rho, 10.0, evap, qsen);
    }

    printf("# alblake\n");
    printf("# i,Tcut,tair,snowalb_in,newsnow,coldcontent,dt,last_snow_in,"
           "swq,melting_in,day,lat | snowalb,albs,albi,albw,last_snow,"
           "melting\n");
    for (i = 0; i <= 80; i++) {
        double tcut = 0.0, newsnow, cc, dt, swq, lat;
        int    day;

        snowalb = 0.2 + 0.008 * (double)(i % 75);
        newsnow = (i % 5 == 0) ? 0.0 : 1.0e-5 * (double)(i % 9);
        cc = -2.0e5 + 6.0e3 * (double)i;
        dt = 3600.0;
        last_snow = (unsigned)(i % 40);
        swq = (i % 4 == 0) ? 0.0 : 0.01 * (double)(i % 13);
        melting = (i % 2) ? true : false;
        day = 1 + (i * 17) % 365;
        lat = (i % 3 == 0) ? -42.0 : 47.0;
        ta = -12.0 + 0.4 * (double)i;
        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%u,%.17g,%d,%d,%.17g",
               i, tcut, ta, snowalb, newsnow, cc, dt, last_snow, swq,
               (int)melting, day, lat);
        alblake(tcut, ta, &snowalb, &albs, &albi, &albw, newsnow, cc, dt,
                &last_snow, swq, &melting, (unsigned short)day, lat);
        printf(" | %.17g,%.17g,%.17g,%.17g,%u,%d\n", snowalb, albs, albi,
               albw, last_snow, (int)melting);
    }
}


/* lakes.eb.c's thermal-profile helpers: the tridiagonal solve the
   temperature equation runs through, the eddy diffusivity profile, the
   ice/water column average, and the column's heat content. */
static void
mode_lake2(void)
{
    int    c, k, nn;
    double a[MAX_LAKE_NODES], b[MAX_LAKE_NODES], cc[MAX_LAKE_NODES];
    double y[MAX_LAKE_NODES], x[MAX_LAKE_NODES];
    double de[MAX_LAKE_NODES], rho[MAX_LAKE_NODES];
    double Tw[MAX_LAKE_NODES], Ti[MAX_LAKE_NODES];
    double ft[MAX_LAKE_NODES], dens[MAX_LAKE_NODES];
    double cp[MAX_LAKE_NODES], surf[MAX_LAKE_NODES + 1];
    double dz, surfdz, wind, lat, sumjoule, frac;

    printf("# tridia\n");
    printf("# case,numnod | a...,b...,c...,y...,x...\n");
    for (c = 0; c < 6; c++) {
        nn = 4 + c;
        printf("%d,%d", c, nn);
        for (k = 0; k < nn; k++) {
            a[k] = -0.3 - 0.05 * (double)((k + c) % 5);
            b[k] = 2.5 + 0.2 * (double)((k + 2 * c) % 7);
            cc[k] = -0.4 - 0.06 * (double)((k + 3 * c) % 4);
            y[k] = 1.0 + 0.7 * sin(0.9 * (double)(k + c));
        }
        tridia(nn, a, b, cc, y, x);
        for (k = 0; k < nn; k++) {
            printf(",%.17g", a[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", b[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", cc[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", y[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", x[k]);
        }
        printf("\n");
    }

    printf("# eddy\n");
    printf("# case,freezeflag,wind,lat,numnod,dz,surfdz | rho...,de...\n");
    for (c = 0; c < 8; c++) {
        nn = 6;
        wind = 0.4 + 1.3 * (double)(c % 6);
        lat = -55.0 + 17.0 * (double)c;
        dz = 0.5 + 0.25 * (double)(c % 3);
        surfdz = 0.2 + 0.1 * (double)(c % 4);
        for (k = 0; k < nn; k++) {
            /* calc_density's offset form, warm at the top and cold below,
               so the Brunt-Vaisala frequency is a real stratification. */
            rho[k] = calc_density(18.0 - 2.2 * (double)k);
        }
        printf("%d,%d,%.17g,%.17g,%d,%.17g,%.17g", c, (c % 2), wind, lat, nn,
               dz, surfdz);
        eddy((c % 2), wind, rho, de, lat, nn, dz, surfdz);
        for (k = 0; k < nn; k++) {
            printf(",%.17g", rho[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", de[k]);
        }
        printf("\n");
    }

    printf("# colavg\n");
    printf("# case,lakeprv,numnod,dz,surfdz | Tw...,Ti...,final...,dens...\n");
    for (c = 0; c < 8; c++) {
        nn = 6;
        frac = 0.05 + 0.12 * (double)c;
        dz = 0.4 + 0.2 * (double)(c % 4);
        surfdz = 0.15 + 0.09 * (double)(c % 5);
        for (k = 0; k < nn; k++) {
            Tw[k] = 14.0 - 1.9 * (double)k + 0.4 * (double)c;
            Ti[k] = 1.5 - 0.22 * (double)k;
        }
        printf("%d,%.17g,%d,%.17g,%.17g", c, frac, nn, dz, surfdz);
        colavg(ft, Tw, Ti, frac, dens, nn, dz, surfdz);
        for (k = 0; k < nn; k++) {
            printf(",%.17g", Tw[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", Ti[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", ft[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", dens[k]);
        }
        printf("\n");
    }

    printf("# energycalc\n");
    printf("# case,numnod,dz,surfdz | final...,cp...,dens...,surface...,"
           "sumjoule\n");
    for (c = 0; c < 8; c++) {
        nn = 6;
        dz = 0.4 + 0.2 * (double)(c % 4);
        surfdz = 0.15 + 0.09 * (double)(c % 5);
        for (k = 0; k < nn; k++) {
            ft[k] = 12.0 - 1.7 * (double)k + 0.3 * (double)c;
            cp[k] = specheat(ft[k]);
            dens[k] = calc_density(ft[k]);
        }
        for (k = 0; k <= nn; k++) {
            surf[k] = 1.0e6 * (1.0 - 0.13 * (double)k) + 1.0e4 * (double)c;
        }
        printf("%d,%d,%.17g,%.17g", c, nn, dz, surfdz);
        energycalc(ft, &sumjoule, nn, dz, surfdz, surf, cp, dens);
        for (k = 0; k < nn; k++) {
            printf(",%.17g", ft[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", cp[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", dens[k]);
        }
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", surf[k]);
        }
        printf(",%.17g\n", sumjoule);
    }
}


/* lakes.eb.c:1294-1560, the lake's temperature solve: Crank-Nicolson
   diffusion on a column whose area shrinks with depth, forced by a
   two-band shortwave and by whatever the surface balance hands down. */
static void
mode_lake3(void)
{
    int    c, k, nn;
    double T[MAX_LAKE_NODES], Tnew[MAX_LAKE_NODES], rho[MAX_LAKE_NODES];
    double de[MAX_LAKE_NODES], cp[MAX_LAKE_NODES];
    double surf[MAX_LAKE_NODES + 1];
    double dz, surfdz, dt, swv, swn, force, temph, eob;

    printf("# temp_area\n");
    printf("# case,numnod,swv,swn,force,dt,dz,surfdz | T...,rho...,de...,"
           "cp...,surface...,Tnew...,eob,joulenew\n");
    for (c = 0; c < 10; c++) {
        nn = 3 + (c % 5);
        swv = 30.0 + 24.0 * (double)(c % 7);
        swn = 18.0 + 11.0 * (double)(c % 5);
        force = -80.0 + 35.0 * (double)(c % 6);
        dt = 3600.0;
        dz = 0.4 + 0.22 * (double)(c % 4);
        surfdz = 0.12 + 0.08 * (double)(c % 5);
        for (k = 0; k < nn; k++) {
            T[k] = 16.0 - 2.1 * (double)k + 0.5 * (double)c;
            rho[k] = calc_density(T[k]);
            cp[k] = specheat(T[k]);
            de[k] = param.LAKE_DM + 1.0e-5 * (double)((k + c) % 6);
        }
        for (k = 0; k <= nn; k++) {
            surf[k] = 1.0e6 * (1.0 - 0.11 * (double)k) + 2.0e4 * (double)c;
        }
        printf("%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               c, nn, swv, swn, force, dt, dz, surfdz);
        for (k = 0; k < nn; k++) {
            printf(",%.17g", T[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", rho[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", de[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", cp[k]);
        }
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", surf[k]);
        }
        eob = 0.0;
        temp_area(swv, swn, force, T, Tnew, rho, de, dt, surf, nn, dz,
                  surfdz, &temph, cp, &eob);
        for (k = 0; k < nn; k++) {
            printf(",%.17g", Tnew[k]);
        }
        printf(",%.17g,%.17g\n", eob, temph);
    }

    {
        int mixdepth;

        printf("# tracer_mixer\n");
        printf("# case,numnod,dz,surfdz,mixdepth_in | T...,cp...,surface...,"
               "Tout...,mixdepth_out\n");
        for (c = 0; c < 14; c++) {
            nn = 4 + (c % 6);
            dz = 0.35 + 0.19 * (double)(c % 4);
            surfdz = 0.11 + 0.07 * (double)(c % 5);
            mixdepth = (c % 3);
            for (k = 0; k < nn; k++) {
                /* Profiles that are stable, inverted at the top, inverted
                   in the middle, and inverted more than once, so the
                   restart at :1674-1678 is reached. */
                switch (c % 4) {
                case 0:
                    T[k] = 18.0 - 2.0 * (double)k;
                    break;
                case 1:
                    T[k] = 3.0 + 2.0 * (double)k;
                    break;
                case 2:
                    T[k] = (k < nn / 2) ? 8.0 - (double)k : 14.0 - (double)k;
                    break;
                default:
                    T[k] = 10.0 + 4.0 * sin(1.7 * (double)k) - 0.6 *
                           (double)c;
                    break;
                }
                cp[k] = specheat(T[k]);
            }
            for (k = 0; k <= nn; k++) {
                surf[k] = 1.0e6 * (1.0 - 0.09 * (double)k) + 3.0e4 *
                          (double)c;
            }
            printf("%d,%d,%.17g,%.17g,%d", c, nn, dz, surfdz, mixdepth);
            for (k = 0; k < nn; k++) {
                printf(",%.17g", T[k]);
            }
            for (k = 0; k < nn; k++) {
                printf(",%.17g", cp[k]);
            }
            for (k = 0; k <= nn; k++) {
                printf(",%.17g", surf[k]);
            }
            tracer_mixer(T, &mixdepth, surf, nn, dz, surfdz, cp);
            for (k = 0; k < nn; k++) {
                printf(",%.17g", T[k]);
            }
            printf(",%d\n", mixdepth);
        }
    }
}


/* lakes.eb.c's ice: new ice forming on open water, shortwave through the
   snow-and-ice cover, and growth or melt at the ice base. */
static void
mode_lake4(void)
{
    int    c, k, nn;
    double T[MAX_LAKE_NODES], cp[MAX_LAKE_NODES], surf[MAX_LAKE_NODES + 1];
    double dz, surfdz, dt, tcut, fracprv, lvolume;
    double qfusion, areaadd, nih, niwe;
    double sw, hi, hs, avgcond, swnet, swunder;
    double fracice, snowflux, qw, eimb, swabs, qf, iwe, volume, sarea;

    printf("# iceform\n");
    printf("# case,numnod,tcut,fracprv,dt,dz,surfdz,lvolume | T...,cp...,"
           "surface...,Tout...,qfusion,areaadd,new_ice_height,"
           "new_ice_water_eq\n");
    for (c = 0; c < 12; c++) {
        nn = 4 + (c % 4);
        tcut = -0.01 * (double)(c % 3);
        fracprv = 0.05 + 0.13 * (double)(c % 7);
        dt = 3600.0;
        dz = 0.4 + 0.2 * (double)(c % 4);
        surfdz = 0.12 + 0.07 * (double)(c % 5);
        /* volumes chosen to reach all three branches at :1002-1022 */
        lvolume = (c % 3 == 0) ? 0.0 : ((c % 3 == 1) ? 1.0e-2 : 1.0e7);
        for (k = 0; k < nn; k++) {
            T[k] = 1.5 - 0.9 * (double)k - 0.3 * (double)c;
            cp[k] = specheat(T[k]);
        }
        for (k = 0; k <= nn; k++) {
            surf[k] = 1.0e6 * (1.0 - 0.1 * (double)k) + 1.5e4 * (double)c;
        }
        printf("%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               c, nn, tcut, fracprv, dt, dz, surfdz, lvolume);
        for (k = 0; k < nn; k++) {
            printf(",%.17g", T[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", cp[k]);
        }
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", surf[k]);
        }
        iceform(&qfusion, T, tcut, fracprv, &areaadd, nn, dt, dz, surfdz,
                cp, surf, &nih, &niwe, lvolume);
        for (k = 0; k < nn; k++) {
            printf(",%.17g", T[k]);
        }
        printf(",%.17g,%.17g,%.17g,%.17g\n", qfusion, areaadd, nih, niwe);
    }

    printf("# icerad\n");
    printf("# case,sw,hi,hs | avgcond,SWnet,SW_under_ice\n");
    for (c = 0; c < 24; c++) {
        sw = 5.0 + 31.0 * (double)(c % 11);
        hi = 0.02 + 0.17 * (double)(c % 9);
        hs = 0.001 * (double)(c % 7) + 0.03 * (double)(c % 5);
        icerad(sw, hi, hs, &avgcond, &swnet, &swunder);
        printf("%d,%.17g,%.17g,%.17g | %.17g,%.17g,%.17g\n", c, sw, hi, hs,
               avgcond, swnet, swunder);
    }

    printf("# lakeice\n");
    printf("# case,sw_ice,fracice,dt,snowflux,qw,SWabsorbed,ice_water_eq_in,"
           "volume,sarea | qf,energy_ice_melt_bot,ice_water_eq_out\n");
    for (c = 0; c < 20; c++) {
        sw = 3.0 + 9.0 * (double)(c % 8);
        fracice = 0.1 + 0.09 * (double)(c % 10);
        dt = 3600.0;
        snowflux = -30.0 + 12.0 * (double)(c % 7);
        qw = -20.0 + 9.0 * (double)(c % 6);
        swabs = 1.0 + 2.0 * (double)(c % 5);
        iwe = 100.0 * (double)(c % 6);
        /* volumes that both allow and restrict freezing */
        volume = (c % 4 == 0) ? iwe : iwe + 50.0 * (double)(c % 9);
        sarea = 1.0e5 + 1.0e4 * (double)c;
        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               c, sw, fracice, dt, snowflux, qw, swabs, iwe, volume, sarea);
        lakeice(sw, fracice, dt, snowflux, qw, &eimb, swabs, &qf, &iwe,
                volume, sarea);
        printf(" | %.17g,%.17g,%.17g\n", qf, eimb, iwe);
    }
}


/* lake_utils.c -- the basin's stage-area-volume relation, which every other
   lake routine reads. Three functions that must invert one another. */
static void
mode_lake5(void)
{
    lake_con_struct lc;
    int             c, k, nn, st;
    double          depth, sarea, volume, back;

    printf("# hypsometry\n");
    printf("# case,numnod,maxdepth,maxvolume | z...,basin...\n");
    printf("# query,case,depth,volume_in | sarea,volume,depth_back\n");
    for (c = 0; c < 6; c++) {
        nn = 4 + (c % 4);
        lc.numnod = nn;
        /* A basin that narrows with depth, at varying rates. */
        for (k = 0; k <= nn; k++) {
            lc.z[k] = (double)(nn - k) * (1.1 + 0.4 * (double)(c % 3));
            lc.basin[k] = 1.0e6 * pow((double)(nn - k) / (double)nn,
                                      1.0 + 0.5 * (double)(c % 4))
                          + 1.0e3;
        }
        lc.maxdepth = lc.z[0];
        lc.maxvolume = 0.0;
        for (k = 0; k < nn; k++) {
            lc.maxvolume += (lc.basin[k] + lc.basin[k + 1]) *
                            (lc.z[k] - lc.z[k + 1]) / 2.;
        }
        printf("%d,%d,%.17g,%.17g", c, nn, lc.maxdepth, lc.maxvolume);
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", lc.z[k]);
        }
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", lc.basin[k]);
        }
        printf("\n");
        for (k = 0; k <= 20; k++) {
            depth = lc.maxdepth * (double)k / 18.0;   /* past the top too */
            st = get_sarea(lc, depth, &sarea);
            (void)st;
            st = get_volume(lc, depth, &volume);
            (void)st;
            st = get_depth(lc, volume, &back);
            (void)st;
            printf("q,%d,%.17g,%.17g | %.17g,%.17g,%.17g\n", c, depth,
                   volume, sarea, volume, back);
        }
    }
}


/* compute_derived_lake_dimensions.c:34-104 -- a lake depth turned into the
   node grid and the areas the thermal solve runs on. */
static void
mode_lake6(void)
{
    lake_con_struct lc;
    lake_var_struct lv;
    int             c, k, nn, d;

    printf("# derived\n");
    printf("# case,numnod,maxdepth,maxvolume | z...,basin...\n");
    printf("# d,case,ldepth,ice_water_eq | surfdz,dz,activenod,sarea,volume,"
           "surface...\n");
    for (c = 0; c < 5; c++) {
        nn = 4 + (c % 4);
        lc.numnod = nn;
        for (k = 0; k <= nn; k++) {
            lc.z[k] = (double)(nn - k) * (1.6 + 0.9 * (double)(c % 3));
            lc.basin[k] = 1.0e6 * pow((double)(nn - k) / (double)nn,
                                      1.0 + 0.4 * (double)(c % 3)) + 2.0e3;
        }
        lc.maxdepth = lc.z[0];
        lc.maxvolume = 0.0;
        for (k = 0; k < nn; k++) {
            lc.maxvolume += (lc.basin[k] + lc.basin[k + 1]) *
                            (lc.z[k] - lc.z[k + 1]) / 2.;
        }
        printf("%d,%d,%.17g,%.17g", c, nn, lc.maxdepth, lc.maxvolume);
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", lc.z[k]);
        }
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", lc.basin[k]);
        }
        printf("\n");
        for (d = 0; d <= 22; d++) {
            memset(&lv, 0, sizeof(lv));
            /* from dry, through one layer, two, and many */
            lv.ldepth = lc.maxdepth * (double)d / 22.0;
            lv.ice_water_eq = 100.0 * (double)(d % 4);
            compute_derived_lake_dimensions(&lv, lc);
            printf("d,%d,%.17g,%.17g | %.17g,%.17g,%d,%.17g,%.17g",
                   c, lv.ldepth, lv.ice_water_eq, lv.surfdz, lv.dz,
                   lv.activenod, lv.sarea, lv.volume);
            for (k = 0; k <= MAX_LAKE_NODES; k++) {
                printf(",%.17g", (k <= lv.activenod) ? lv.surface[k] : 0.0);
            }
            printf("\n");
        }
    }
}


/* lakes.eb.c:1873-1897 and :2010-2118 -- the lake's own water balance and
   what leaves it. water_balance() reaches into all_vars for the wetland
   tile, so this builds a one-by-one of each struct and fills only what
   those two sections read; lakefrac is 1 so the recharge section between
   them is a no-op and the comparison isolates them.

   Printed at %.17g, not the %.17g the other modes use, and the difference
   matters here. Seventeen digits round-trip an IEEE double and fourteen do
   not; the lost bits move this mode's Dsmax by one part in 1e14, which is
   nothing in the baseflow itself. But the lake can end the step with a
   liquid volume of about 1e-10 m3 -- the residual of `a - (a - b)` -- and a
   change in the last bits flips its SIGN. VIC zeroes a negative residual
   (lake_utils.c:127-130) and reports a depth for a positive one, so a tenth
   of a cubic millimetre of water decides whether the lake has a surface
   area at all. */
static void
mode_lake7(void)
{
    lake_con_struct   lc;
    lake_var_struct   lv;
    all_vars_struct   av;
    soil_con_struct   sc;
    veg_con_struct    vc;
    cell_data_struct  cellmem, *cellrow[1], **cellp;
    veg_var_struct    vvmem, *vvrow[1], **vvp;
    snow_data_struct  snmem, *snrow[1], **snp;
    energy_bal_struct enmem, *enrow[1], **enp;
    int               c, k, nn;

    cellrow[0] = &cellmem;
    cellp = cellrow;
    vvrow[0] = &vvmem;
    vvp = vvrow;
    snrow[0] = &snmem;
    snp = snrow;
    enrow[0] = &enmem;
    enp = enrow;

    printf("# outflow\n");
    printf("# case,numnod,maxdepth,maxvolume | z...,basin...\n");
    printf("# o,case,volume_in,ice_water_eq,evapw_in,ice_throughfall,"
           "inflow,snowmlt,mindepth,minvolume,wfrac,dt,Dsmax_step,Ds,Ws,c,"
           "max_moist_bot,resid_moist_bot,ice_bot,new_ice_area "
           "| volume,ldepth,sarea,baseflow_out,runoff_out,evapw\n");
    for (c = 0; c < 4; c++) {
        nn = 4 + (c % 3);
        memset(&lc, 0, sizeof(lc));
        memset(&sc, 0, sizeof(sc));
        memset(&vc, 0, sizeof(vc));
        lc.numnod = nn;
        for (k = 0; k <= nn; k++) {
            lc.z[k] = (double)(nn - k) * (1.4 + 0.5 * (double)(c % 3));
            lc.basin[k] = 1.0e6 * pow((double)(nn - k) / (double)nn,
                                      1.0 + 0.3 * (double)(c % 3)) + 2.0e3;
        }
        lc.maxdepth = lc.z[0];
        lc.maxvolume = 0.0;
        for (k = 0; k < nn; k++) {
            lc.maxvolume += (lc.basin[k] + lc.basin[k + 1]) *
                            (lc.z[k] - lc.z[k + 1]) / 2.;
        }
        lc.mindepth = 0.2 + 0.15 * (double)c;
        lc.minvolume = lc.maxvolume * 0.01 * (double)(c % 3);
        lc.wfrac = 0.01 + 0.02 * (double)(c % 4);
        printf("%d,%d,%.17g,%.17g", c, nn, lc.maxdepth, lc.maxvolume);
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", lc.z[k]);
        }
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", lc.basin[k]);
        }
        printf("\n");

        for (k = 0; k < 14; k++) {
            int    j;
            double dtv = 3600.0;

            memset(&lv, 0, sizeof(lv));
            memset(&cellmem, 0, sizeof(cellmem));
            memset(&vvmem, 0, sizeof(vvmem));
            memset(&snmem, 0, sizeof(snmem));
            memset(&enmem, 0, sizeof(enmem));
            av.cell = cellp;
            av.veg_var = vvp;
            av.snow = snp;
            av.energy = enp;

            lv.volume = lc.maxvolume * (0.05 + 0.065 * (double)k);
            lv.ice_water_eq = lv.volume * 0.05 * (double)(k % 4);
            lv.evapw = (k % 3 == 0) ? 1.0e5 * (double)(k % 5) : 10.0;
            lv.ice_throughfall = 1.0e3 * (double)(k % 3);
            lv.runoff_in = 2.0e4 * (double)(k % 6);
            lv.baseflow_in = 1.0e3 * (double)(k % 4);
            lv.channel_in = 0.0;
            lv.snowmlt = 5.0e2 * (double)(k % 5);
            lv.new_ice_area = 1.0e5 * (double)(k % 3);
            lv.ldepth = 0.0;
            get_depth(lc, lv.volume - lv.ice_water_eq, &(lv.ldepth));
            compute_derived_lake_dimensions(&lv, lc);

            sc.Dsmax = 20.0 + 5.0 * (double)(k % 4);
            sc.Ds = 0.02 + 0.03 * (double)(k % 3);
            sc.Ws = 0.6 + 0.1 * (double)(k % 3);
            sc.c = 2.0;
            for (j = 0; j < (int)options.Nlayer; j++) {
                sc.depth[j] = 0.3 + 0.2 * (double)j;
                sc.max_moist[j] = sc.depth[j] * 1000.0 * 0.45;
                sc.resid_moist[j] = 0.04;
                cellmem.layer[j].moist = sc.max_moist[j] * 0.8;
            }
            for (j = 0; j < (int)options.Nfrost; j++) {
                sc.frost_fract[j] = 1.0 / (double)options.Nfrost;
                cellmem.layer[options.Nlayer - 1].ice[j] =
                    sc.max_moist[options.Nlayer - 1] * 0.05 * (double)(k % 4);
            }

            printf("o,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                   "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                   "%.17g",
                   c, lv.volume, lv.ice_water_eq, lv.evapw,
                   lv.ice_throughfall,
                   lv.runoff_in + lv.baseflow_in + lv.channel_in, lv.snowmlt,
                   lc.mindepth, lc.minvolume, lc.wfrac, dtv,
                   sc.Dsmax / global_param.model_steps_per_day, sc.Ds, sc.Ws,
                   sc.c,
                   sc.max_moist[options.Nlayer - 1],
                   sc.resid_moist[options.Nlayer - 1] *
                   sc.depth[options.Nlayer - 1] * MM_PER_M,
                   cellmem.layer[options.Nlayer - 1].ice[0],
                   lv.new_ice_area);
            water_balance(&lv, lc, dtv, &av, 0, 0, 1.0, sc, vc);
            printf(" | %.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                   lv.volume, lv.ldepth, lv.sarea, lv.baseflow_out,
                   lv.runoff_out, lv.evapw);
        }
    }
}


/* lakes.eb.c:2313-2683 -- the machinery that moves storage between the lake
   and the wetland beside it as the lake's area changes. */
static void
mode_lake8(void)
{
    lake_con_struct   lc;
    lake_var_struct   lv;
    soil_con_struct   sc;
    veg_con_struct    vc;
    cell_data_struct  cell;
    veg_var_struct    vv;
    snow_data_struct  sn;
    energy_bal_struct en;
    int               c, j;
    double            lakefrac, maxnew, newfrac, dm[MAX_LAYERS];

    printf("# advect_soil_veg\n");
    printf("# case,lakefrac,max_newfraction,newfraction,basin0,baseflow_in "
           "| moist...,delta_moist...,max_moist...,Wdew_in "
           "| moist_out...,Wdew_out,baseflow_out\n");
    for (c = 0; c < 18; c++) {
        memset(&sc, 0, sizeof(sc));
        memset(&vc, 0, sizeof(vc));
        memset(&cell, 0, sizeof(cell));
        memset(&vv, 0, sizeof(vv));
        lc.basin[0] = 1.0e6 + 1.0e4 * (double)c;
        lakefrac = 0.05 + 0.05 * (double)(c % 18);
        maxnew = lakefrac + 0.06 * (double)((c % 5) - 2);
        newfrac = lakefrac + 0.04 * (double)((c % 7) - 3);
        if (maxnew < 0) {
            maxnew = 0;
        }
        if (maxnew > 0.98) {
            maxnew = 0.98;
        }
        if (newfrac < 0) {
            newfrac = 0;
        }
        if (newfrac > 0.98) {
            newfrac = 0.98;
        }
        for (j = 0; j < (int)options.Nlayer; j++) {
            sc.depth[j] = 0.3 + 0.2 * (double)j;
            sc.max_moist[j] = sc.depth[j] * 1000.0 * 0.45;
            sc.porosity[j] = 0.45;
            sc.Wpwp[j] = sc.max_moist[j] * 0.2;
            sc.expt[j] = 6.0;
            sc.bubble[j] = 30.0;
            sc.resid_moist[j] = 0.04;
            sc.init_moist[j] = sc.max_moist[j];
            cell.layer[j].moist = sc.max_moist[j] *
                                  (0.55 + 0.04 * (double)((j + c) % 8));
            /* recharge that will overflow some columns */
            dm[j] = sc.max_moist[j] * 0.15 * (double)((j + c) % 4);
            vc.root[j] = 1.0 / (double)options.Nlayer;
        }
        vv.Wdew = 1.0 + 0.7 * (double)(c % 6);
        cell.baseflow = 3.0 * (double)(c % 5);
        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g", c, lakefrac, maxnew,
               newfrac, lc.basin[0], cell.baseflow);
        printf(" |");
        for (j = 0; j < (int)options.Nlayer; j++) {
            printf(",%.17g", cell.layer[j].moist);
        }
        for (j = 0; j < (int)options.Nlayer; j++) {
            printf(",%.17g", dm[j]);
        }
        for (j = 0; j < (int)options.Nlayer; j++) {
            printf(",%.17g", sc.max_moist[j]);
        }
        printf(",%.17g", vv.Wdew);
        advect_soil_veg_storage(lakefrac, maxnew, newfrac, dm, &sc, &vc,
                                &cell, &vv, lc);
        printf(" |");
        for (j = 0; j < (int)options.Nlayer; j++) {
            printf(",%.17g", cell.layer[j].moist);
        }
        printf(",%.17g,%.17g\n", vv.Wdew, cell.baseflow);
    }

    printf("# rescale\n");
    printf("# case,oldfrac,newfrac | evap_in,baseflow_in,runoff_in,"
           "swq_in,melt_in,grnd_in | evap,baseflow,runoff,swq,melt,grnd\n");
    for (c = 0; c < 16; c++) {
        double oldfrac = 0.05 * (double)(c % 16);
        double newfrac2 = 0.04 * (double)((c + 3) % 20);

        memset(&cell, 0, sizeof(cell));
        memset(&vv, 0, sizeof(vv));
        memset(&sn, 0, sizeof(sn));
        memset(&en, 0, sizeof(en));
        cell.layer[0].evap = 1.5 + 0.3 * (double)c;
        cell.baseflow = 2.0 + 0.5 * (double)c;
        cell.runoff = 4.0 + 0.7 * (double)c;
        sn.swq = 0.1 * (double)(c % 7);
        sn.melt = 0.05 * (double)(c % 5);
        en.grnd_flux = -20.0 + 4.0 * (double)c;
        printf("%d,%.17g,%.17g | %.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               c, oldfrac, newfrac2, cell.layer[0].evap, cell.baseflow,
               cell.runoff, sn.swq, sn.melt, en.grnd_flux);
        rescale_soil_veg_fluxes(oldfrac, newfrac2, &cell, &vv);
        rescale_snow_storage(oldfrac, newfrac2, &sn);
        rescale_snow_energy_fluxes(oldfrac, newfrac2, &sn, &en);
        printf(" | %.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               cell.layer[0].evap, cell.baseflow, cell.runoff, sn.swq,
               sn.melt, en.grnd_flux);
    }

    printf("# advect_snow\n");
    printf("# case,lakefrac,max_newfraction,newfraction | swq_in,depth_in "
           "| swq,depth\n");
    for (c = 0; c < 16; c++) {
        lakefrac = 0.06 * (double)(c % 17);
        maxnew = lakefrac + 0.05 * (double)((c % 5) - 2);
        newfrac = lakefrac + 0.03 * (double)((c % 7) - 3);
        if (maxnew < 0) {
            maxnew = 0;
        }
        if (newfrac < 0) {
            newfrac = 0;
        }
        if (maxnew > 0.99) {
            maxnew = 0.99;
        }
        if (newfrac > 0.99) {
            newfrac = 0.99;
        }
        memset(&sn, 0, sizeof(sn));
        sn.swq = 0.05 + 0.03 * (double)(c % 9);
        sn.depth = sn.swq * 3.0;
        printf("%d,%.17g,%.17g,%.17g | %.17g,%.17g", c, lakefrac, maxnew,
               newfrac, sn.swq, sn.depth);
        advect_snow_storage(lakefrac, maxnew, newfrac, &sn);
        printf(" | %.17g,%.17g\n", sn.swq, sn.depth);
    }

    printf("# advect_carbon\n");
    printf("# case,lakefrac,newfraction | lake_in,cell_in | lake,cell\n");
    for (c = 0; c < 16; c++) {
        memset(&lv, 0, sizeof(lv));
        memset(&cell, 0, sizeof(cell));
        lakefrac = 0.06 * (double)(c % 16);
        newfrac = lakefrac + 0.05 * (double)((c % 5) - 2);
        if (newfrac < 0) {
            newfrac = 0;
        }
        if (newfrac > 0.99) {
            newfrac = 0.99;
        }
        lv.soil.CLitter = 100.0 + 20.0 * (double)c;
        cell.CLitter = 300.0 + 15.0 * (double)c;
        printf("%d,%.17g,%.17g | %.17g,%.17g", c, lakefrac, newfrac,
               lv.soil.CLitter, cell.CLitter);
        advect_carbon_storage(lakefrac, newfrac, &lv, &cell);
        printf(" | %.17g,%.17g\n", lv.soil.CLitter, cell.CLitter);
    }
}


/* water_energy_balance.c:33-205 and water_under_ice.c:31-142 -- the two
   fixed-point loops the lake's energy balance turns on, including the
   branches where they do not converge. */
static void
mode_lake9(void)
{
    int    c, k, nn, mixdepth, ff;
    double T[MAX_LAKE_NODES], rho[MAX_LAKE_NODES], cp[MAX_LAKE_NODES];
    double surf[MAX_LAKE_NODES + 1];
    double dz, surfdz, dt, lat, tcut, tair, wind, press, vp, rhoa;
    double lwin, swin, wind_h, lvolume, fracprv;
    double evapw, Qh, Qle, LWnet, deltaH, eif, nia, nih, niwe, eob;
    double qw, sdepth, hice, sw_ice;

    printf("# web\n");
    printf("# case,numnod,dt,dz,surfdz,lat,Tcutoff,Tair,wind,pressure,vp,"
           "air_density,longwave,shortwave,wind_h,fracprv,lvolume "
           "| T...,rho...,cp...,surface... "
           "| Tout...,evapw,Qh,Qle,LWnet,deltaH,eif,nia,nih,niwe,eob\n");
    for (c = 0; c < 14; c++) {
        nn = 3 + (c % 5);
        dt = 3600.0;
        dz = 0.4 + 0.2 * (double)(c % 4);
        surfdz = 0.12 + 0.08 * (double)(c % 5);
        lat = 40.0 + 6.0 * (double)(c % 6);
        tcut = -0.01 * (double)(c % 3);
        tair = -12.0 + 3.0 * (double)(c % 9);
        wind = 0.4 + 1.1 * (double)(c % 7);
        press = 92000.0 + 400.0 * (double)c;
        vp = 150.0 + 60.0 * (double)(c % 8);
        rhoa = 1.25 - 0.01 * (double)(c % 5);
        lwin = 200.0 + 13.0 * (double)(c % 11);
        swin = 20.0 + 45.0 * (double)(c % 9);
        wind_h = 10.0;
        fracprv = 0.02 * (double)(c % 6);
        lvolume = 1.0e5 * (double)(1 + c % 7);
        for (k = 0; k < nn; k++) {
            T[k] = 6.0 - 1.4 * (double)k + 0.7 * (double)c - 3.0;
            rho[k] = calc_density(T[k]);
            cp[k] = specheat(T[k]);
        }
        for (k = 0; k <= nn; k++) {
            surf[k] = 1.0e6 * (1.0 - 0.1 * (double)k) + 2.0e4 * (double)c;
        }
        printf("%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               c, nn, dt, dz, surfdz, lat, tcut, tair, wind, press, vp, rhoa,
               lwin, swin, wind_h, fracprv, lvolume);
        printf(" |");
        for (k = 0; k < nn; k++) {
            printf(",%.17g", T[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", rho[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", cp[k]);
        }
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", surf[k]);
        }
        evapw = Qh = Qle = LWnet = deltaH = eif = nia = nih = niwe = eob = 0;
        water_energy_balance(nn, surf, &evapw, dt, dz, surfdz, lat, tcut,
                             tair, wind, press, vp, rhoa, lwin, swin, wind_h,
                             &Qh, &Qle, &LWnet, T, rho, &deltaH, &eif,
                             fracprv, &nia, cp, &nih, &eob, &niwe, lvolume);
        printf(" |");
        for (k = 0; k < nn; k++) {
            printf(",%.17g", T[k]);
        }
        printf(",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g\n", evapw, Qh, Qle, LWnet, deltaH, eif, nia, nih,
               niwe, eob);
    }

    printf("# wui\n");
    printf("# case,freezeflag,numnod,dt,dz,surfdz,lat,Tcutoff,wind,sw_ice,"
           "mixdepth,hice,sdepth | Ti...,rho...,cp...,surface... "
           "| Tiout...,qw,deltaH,eob,mixdepth_out\n");
    for (c = 0; c < 14; c++) {
        nn = 3 + (c % 5);
        ff = (c % 2);
        dt = 3600.0;
        dz = 0.4 + 0.2 * (double)(c % 4);
        surfdz = 0.12 + 0.08 * (double)(c % 5);
        lat = 44.0 + 5.0 * (double)(c % 6);
        tcut = -0.01 * (double)(c % 3);
        wind = 0.4 + 1.1 * (double)(c % 7);
        sw_ice = 3.0 + 12.0 * (double)(c % 8);
        mixdepth = c % 3;
        hice = 0.05 + 0.12 * (double)(c % 6);
        sdepth = 0.01 * (double)(c % 5);
        for (k = 0; k < nn; k++) {
            T[k] = 2.5 - 0.6 * (double)k + 0.2 * (double)c;
            rho[k] = calc_density(T[k]);
            cp[k] = specheat(T[k]);
        }
        for (k = 0; k <= nn; k++) {
            surf[k] = 1.0e6 * (1.0 - 0.1 * (double)k) + 1.0e4 * (double)c;
        }
        printf("%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,"
               "%.17g", c, ff, nn, dt, dz, surfdz, lat, tcut, wind, sw_ice,
               mixdepth, hice, sdepth);
        printf(" |");
        for (k = 0; k < nn; k++) {
            printf(",%.17g", T[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", rho[k]);
        }
        for (k = 0; k < nn; k++) {
            printf(",%.17g", cp[k]);
        }
        for (k = 0; k <= nn; k++) {
            printf(",%.17g", surf[k]);
        }
        qw = deltaH = eob = 0;
        water_under_ice(ff, sw_ice, wind, T, rho, lat, nn, dz, surfdz, tcut,
                        &qw, surf, &deltaH, cp, mixdepth, hice, sdepth, dt,
                        &eob);
        printf(" |");
        for (k = 0; k < nn; k++) {
            printf(",%.17g", T[k]);
        }
        printf(",%.17g,%.17g,%.17g,%d\n", qw, deltaH, eob, mixdepth);
    }
}


/* IceEnergyBalance.c:34-214 -- the surface energy balance over lake ice.
   Called through the same dummy wrapper ice_melt.c uses (:613-630), so the
   variable-argument list is assembled exactly as the model assembles it. */
static void
mode_lake10(void)
{
    int    c;
    double Tsurf, Dt, Ra, Ra_used, Z, Z0, Wind, ShortRad, LongRadIn;
    double AirDens, Lv, Tair, Press, Vpd, EactAir, Rain, SurfLiq;
    double RefreezeEnergy, vapor_flux, blowing_flux, surface_flux;
    double AdvectedEnergy, Tfreeze, AvgCond, SWconducted, qf;
    double LatentHeat, LatentHeatSub, SensibleHeat, LongRadOut, rest;

    printf("# iceeb\n");
    printf("# case,Tsurf,Dt,Ra,Z,Z0,Wind,ShortRad,LongRadIn,AirDens,Lv,Tair,"
           "Press,Vpd,EactAir,Rain,SurfaceLiquidWater,blowing_flux,Tfreeze,"
           "AvgCond,SWconducted "
           "| rest,Ra_used,LongRadOut,SensibleHeat,LatentHeat,LatentHeatSub,"
           "vapor_flux,surface_flux,AdvectedEnergy,qf,RefreezeEnergy\n");
    for (c = 0; c < 40; c++) {
        Tsurf = (c % 7 == 0) ? 0.0 : (-14.0 + 0.7 * (double)c);
        Dt = 3600.0;
        Ra = 20.0 + 9.0 * (double)(c % 8);
        Z = 10.0;
        Z0 = param.LAKE_ZSNOW;
        Wind = (c % 11 == 0) ? 0.0 : (0.3 + 0.9 * (double)(c % 9));
        ShortRad = 5.0 + 17.0 * (double)(c % 10);
        LongRadIn = 190.0 + 9.0 * (double)(c % 12);
        AirDens = 1.2 + 0.01 * (double)(c % 6);
        Tair = -13.0 + 0.8 * (double)c;
        Lv = calc_latent_heat_of_vaporization(Tair);
        Press = 90000.0 + 300.0 * (double)c;
        EactAir = 120.0 + 33.0 * (double)(c % 9);
        Vpd = svp(Tair) - EactAir;
        if (Vpd < 0) {
            Vpd = 0;
        }
        Rain = 1.0e-4 * (double)(c % 5);
        SurfLiq = 1.0e-3 * (double)(c % 6);
        blowing_flux = 0.0;
        Tfreeze = -0.01 * (double)(c % 3);
        AvgCond = 0.02 + 0.05 * (double)(c % 7);
        SWconducted = 1.0 + 3.0 * (double)(c % 8);
        RefreezeEnergy = vapor_flux = surface_flux = AdvectedEnergy = 0;
        qf = LatentHeat = LatentHeatSub = SensibleHeat = LongRadOut = 0;
        Ra_used = 0;

        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g",
               c, Tsurf, Dt, Ra, Z, Z0, Wind, ShortRad, LongRadIn, AirDens,
               Lv, Tair, Press, Vpd, EactAir, Rain, SurfLiq, blowing_flux,
               Tfreeze, AvgCond, SWconducted);

        rest = CalcIcePackEnergyBalance(
            Tsurf, Dt, Ra, &Ra_used, Z, Z0, Wind, ShortRad, LongRadIn,
            AirDens, Lv, Tair, Press, Vpd, EactAir, Rain, SurfLiq,
            &RefreezeEnergy, &vapor_flux, &blowing_flux, &surface_flux,
            &AdvectedEnergy, Tfreeze, AvgCond, SWconducted, &qf, &LatentHeat,
            &LatentHeatSub, &SensibleHeat, &LongRadOut);

        printf(" | %.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g\n", rest, Ra_used, LongRadOut, SensibleHeat,
               LatentHeat, LatentHeatSub, vapor_flux, surface_flux,
               AdvectedEnergy, qf, RefreezeEnergy);
    }
}


/* ice_melt() -- one whole step of the snow-and-ice pack on a lake.
   ice_melt.c:34-611. State in, state out, plus every saved flux. */
static void
mode_lake11(void)
{
    int              c;
    snow_data_struct snow;
    lake_var_struct  lake;
    double           z2, aero_resist, aero_resist_used, Le, delta_t;
    double           displacement, Z0, surf_atten, rainfall, snowfall, wind;
    double           Tcutoff, air_temp, net_short, longwave, density;
    double           pressure, vpd, vp, melt;
    double           advection, deltaCC, SnowFlux, latent, sensible, Qnet;
    double           refreeze_energy, LWnet;
    int              status;

    printf("# icemelt\n");
    printf("# case,swq,surf_water,pack_water,surf_temp,pack_temp,"
           "ice_water_eq,areai,hice,volume,dt,rainfall_mm,snowfall_mm,"
           "air_temp,wind,net_short,longwave,air_dens,Lv,press_Pa,vpd_Pa,"
           "vp_Pa,Ra,z2,Z0,Tcutoff "
           "| status,swq,surf_water,pack_water,surf_temp,pack_temp,"
           "coldcontent,vapor_flux,surface_flux,mass_error,coverage,"
           "ice_water_eq,volume,melt_mm,advection,deltaCC,SnowFlux,latent,"
           "sensible,Qnet,refreeze_energy,LWnet,fbflag\n");

    for (c = 0; c < 44; c++) {
        memset(&snow, 0, sizeof(snow));
        memset(&lake, 0, sizeof(lake));

        delta_t = 3600.0;
        z2 = 10.0;
        Z0 = param.LAKE_ZSNOW;
        displacement = 0.0;
        surf_atten = 1.0;
        aero_resist = 20.0 + 12.0 * (double)(c % 7);
        aero_resist_used = 0.0;

        air_temp = -12.0 + 4.0 * (double)(c % 6);
        wind = 0.5 + 1.5 * (double)(c % 4);
        net_short = 10.0 + 60.0 * (double)(c % 5);
        longwave = 200.0 + 20.0 * (double)(c % 7);
        density = 1.20 + 0.01 * (double)(c % 6);
        pressure = 95.0;                       /* kPa, as VIC's callers pass */
        vp = 0.15 + 0.06 * (double)(c % 5);    /* kPa */
        vpd = svp(air_temp) / PA_PER_KPA - vp;
        if (vpd < 0.0) {
            vpd = 0.0;
        }
        Le = calc_latent_heat_of_vaporization(air_temp);
        Tcutoff = -0.02 * (double)(c % 3);

        snowfall = 2.0 * (double)(c % 5);      /* mm */
        rainfall = (c % 7 == 0) ? 1.5 : 0.0;   /* mm */

        snow.pack_water = 0.001 * (double)(c % 3);
        snow.surf_water = 0.002 * (double)(c % 4);
        snow.swq = 0.02 + 0.15 * (double)(c % 3) +
                   snow.pack_water + snow.surf_water;
        snow.surf_temp = -2.0 * (double)(c % 5);
        snow.pack_temp = -1.5 * (double)(c % 4);
        snow.last_snow = 1 + (unsigned int)(c % 9);
        snow.density = param.LAKE_RHOSNOW;
        snow.depth = snow.swq * CONST_RHOFW / param.LAKE_RHOSNOW;

        lake.areai = 10000.0;
        lake.ice_water_eq = 100.0 + 4000.0 * (double)(c % 6);
        lake.hice = lake.ice_water_eq / lake.areai * CONST_RHOFW /
                    CONST_RHOICE;
        lake.volume = 500000.0;

        /* A pack too thin for the surface energy balance: ice_melt sets
           surf_temp to the 999 flag and leaves the layer alone
           (ice_melt.c:347, 415). */
        if (c % 11 == 10) {
            snow.swq = 0.0004;
            snow.surf_water = 0.0;
            snow.pack_water = 0.0;
            lake.ice_water_eq = 0.0;
            lake.hice = 0.0;
            snowfall = 0.0;
        }
        /* A pack the step melts away completely, ice and all
           (ice_melt.c:325-343). */
        if (c % 13 == 12) {
            snow.swq = 0.001;
            snow.surf_water = 0.0;
            snow.pack_water = 0.0;
            snow.surf_temp = 0.0;
            snow.pack_temp = 0.0;
            lake.ice_water_eq = 20.0;
            lake.hice = lake.ice_water_eq / lake.areai * CONST_RHOFW /
                        CONST_RHOICE;
            air_temp = 12.0;
            net_short = 400.0;
            longwave = 340.0;
            snowfall = 0.0;
            wind = 3.0;
        }

        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               c, snow.swq, snow.surf_water, snow.pack_water, snow.surf_temp,
               snow.pack_temp, lake.ice_water_eq, lake.areai, lake.hice,
               lake.volume, delta_t, rainfall, snowfall, air_temp, wind,
               net_short, longwave, density, Le, pressure * PA_PER_KPA,
               vpd * PA_PER_KPA, vp * PA_PER_KPA, aero_resist, z2, Z0,
               Tcutoff);

        melt = advection = deltaCC = SnowFlux = latent = sensible = 0.0;
        Qnet = refreeze_energy = LWnet = 0.0;

        status = ice_melt(z2, aero_resist, &aero_resist_used, Le, &snow,
                          &lake, delta_t, displacement, Z0, surf_atten,
                          rainfall, snowfall, wind, Tcutoff, air_temp,
                          net_short, longwave, density, pressure, vpd, vp,
                          &melt, &advection, &deltaCC, &SnowFlux, &latent,
                          &sensible, &Qnet, &refreeze_energy, &LWnet);

        printf(" | %d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%d\n",
               status, snow.swq, snow.surf_water, snow.pack_water,
               snow.surf_temp, snow.pack_temp, snow.coldcontent,
               snow.vapor_flux, snow.surface_flux, snow.mass_error,
               snow.coverage, lake.ice_water_eq, lake.volume, melt,
               advection, deltaCC, SnowFlux, latent, sensible, Qnet,
               refreeze_energy, LWnet, (int)snow.surf_temp_fbflag);
    }
}


/* solve_lake() -- one whole lake step, lakes.eb.c:33-532. Every piece this
   module ports, called in the order the C calls them. */
static void
mode_lake12(void)
{
    int             c, k;
    lake_var_struct lake;
    lake_con_struct lake_con;
    soil_con_struct soil_con;
    dmy_struct      dmy;
    double          snowfall, rainfall, tair, wind, vp, shortin, longin;
    double          vpd, pressure, air_density, dt, wind_h, fracprv;
    int             status;

    printf("# solvelake\n");
    printf("# case,activenod,dz,surfdz,sarea,volume,ice_water_eq,areai,hice,"
           "SAlbedo,swq,surf_water,pack_water,surf_temp,pack_temp,"
           "coldcontent,last_snow,MELTING,swe,snowfall_mm,rainfall_mm,tair,"
           "wind,vp_kPa,shortin,longin,vpd_kPa,press_kPa,air_density,dt,"
           "wind_h,day_in_year,fracprv,lat,snow_rough,"
           "surface0..surface20,temp0..temp19 "
           "| status,volume,ice_water_eq,areai,new_ice_area,hice,SAlbedo,"
           "tempavg,tempi,swe,sdepth,snowmlt,evapw,vapor_flux,pack_water,"
           "surf_water,ice_throughfall,aero_resist,snow_swq,snow_surf_water,"
           "snow_pack_water,snow_surf_temp,snow_pack_temp,snow_coldcontent,"
           "snow_depth,snow_coverage,snow_vapor_flux,snow_blowing_flux,"
           "snow_surface_flux,snow_melt,last_snow,MELTING,AlbedoLake,"
           "AtmosLatent,AtmosSensible,NetLongAtmos,NetShortAtmos,deltaH,"
           "grnd_flux,refreeze_energy,advection,deltaCC,snow_flux,Tsurf,"
           "error,temp0..temp19\n");

    /* One basin, VIC's own default depth-area profile. */
    memset(&soil_con, 0, sizeof(soil_con));
    soil_con.cell_area = 1.0e7;
    soil_con.lat = 47.5;
    soil_con.snow_rough = 0.001;
    memset(&lake_con, 0, sizeof(lake_con));
    lake_con.numnod = 10;
    lake_con.z[0] = 12.0;
    lake_con.Cl[0] = 0.10;
    lake_con.rpercent = 1.0;
    lake_con.mindepth = 0.0;
    options.LAKE_PROFILE = false;
    compute_lake_params(&lake_con, soil_con);

    for (c = 0; c < 40; c++) {
        memset(&lake, 0, sizeof(lake));

        dt = 3600.0;
        wind_h = 10.0;
        dmy.day_in_year = 1 + 9 * (c % 40);

        lake.ldepth = 0.4 + 1.3 * (double)(c % 9);
        fracprv = 0.25 * (double)(c % 5);          /* 0, .25, .5, .75, 1 */
        if (c % 17 == 16) {
            fracprv = 0.01;                        /* below LAKE_FRACLIM */
        }

        tair = -14.0 + 2.0 * (double)(c % 15);
        wind = 0.4 + 1.3 * (double)(c % 6);
        shortin = 5.0 + 55.0 * (double)(c % 8);
        longin = 195.0 + 18.0 * (double)(c % 9);
        pressure = 95.0;
        vp = 0.15 + 0.09 * (double)(c % 7);
        vpd = svp(tair) / PA_PER_KPA - vp;
        if (vpd < 0.0) {
            vpd = 0.0;
        }
        air_density = 1.2 + 0.012 * (double)(c % 6);
        snowfall = 1.5 * (double)(c % 4);
        rainfall = (c % 5 == 0) ? 2.0 : 0.0;

        lake.ice_water_eq = (fracprv > 0.0) ? 400.0 * (double)(1 + c % 6) : 0.;
        compute_derived_lake_dimensions(&lake, lake_con);
        lake.sarea = lake.surface[0];
        lake.areai = fracprv * lake.sarea;
        lake.hice = (lake.areai > 0.0) ?
                    lake.ice_water_eq / lake.areai * CONST_RHOFW /
                    CONST_RHOICE : 0.0;

        for (k = 0; k < MAX_LAKE_NODES; k++) {
            lake.temp[k] = (k < (int)lake.activenod) ?
                           (0.6 + 0.35 * (double)(c % 11) +
                            0.12 * (double)k) : 0.0;
        }
        lake.SAlbedo = 0.55 + 0.02 * (double)(c % 6);
        lake.snow.swq = 0.004 * (double)(c % 7);
        lake.snow.surf_water = 0.0005 * (double)(c % 3);
        lake.snow.pack_water = 0.0003 * (double)(c % 4);
        if (lake.snow.swq < lake.snow.surf_water + lake.snow.pack_water) {
            lake.snow.swq = lake.snow.surf_water + lake.snow.pack_water;
        }
        lake.snow.surf_temp = -1.5 * (double)(c % 5);
        lake.snow.pack_temp = -1.2 * (double)(c % 4);
        lake.snow.coldcontent = -900.0 * (double)(c % 5);
        lake.snow.last_snow = 1 + (unsigned int)(c % 12);
        lake.snow.MELTING = (c % 3 == 0);
        lake.snow.density = param.LAKE_RHOSNOW;
        lake.swe = lake.snow.swq * lake.sarea;

        printf("%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%u,%d,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%u,%.17g,%.17g,%.17g",
               c, (int)lake.activenod, lake.dz, lake.surfdz, lake.sarea,
               lake.volume, lake.ice_water_eq, lake.areai, lake.hice,
               lake.SAlbedo, lake.snow.swq, lake.snow.surf_water,
               lake.snow.pack_water, lake.snow.surf_temp, lake.snow.pack_temp,
               lake.snow.coldcontent, lake.snow.last_snow,
               (int)lake.snow.MELTING, lake.swe, snowfall, rainfall, tair,
               wind, vp, shortin, longin, vpd, pressure, air_density, dt,
               wind_h, dmy.day_in_year, fracprv, soil_con.lat,
               soil_con.snow_rough);
        for (k = 0; k <= MAX_LAKE_NODES; k++) {
            printf(",%.17g", lake.surface[k]);
        }
        for (k = 0; k < MAX_LAKE_NODES; k++) {
            printf(",%.17g", lake.temp[k]);
        }

        status = solve_lake(snowfall, rainfall, tair, wind, vp, shortin,
                            longin, vpd, pressure, air_density, &lake,
                            soil_con, dt, wind_h, dmy, fracprv);

        printf(" | %d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%u,%d,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               status, lake.volume, lake.ice_water_eq, lake.areai,
               lake.new_ice_area, lake.hice, lake.SAlbedo, lake.tempavg,
               lake.tempi, lake.swe, lake.sdepth, lake.snowmlt, lake.evapw,
               lake.vapor_flux, lake.pack_water, lake.surf_water,
               lake.ice_throughfall, lake.aero_resist, lake.snow.swq,
               lake.snow.surf_water, lake.snow.pack_water,
               lake.snow.surf_temp, lake.snow.pack_temp,
               lake.snow.coldcontent, lake.snow.depth, lake.snow.coverage,
               lake.snow.vapor_flux, lake.snow.blowing_flux,
               lake.snow.surface_flux, lake.snow.melt, lake.snow.last_snow,
               (int)lake.snow.MELTING, lake.energy.AlbedoLake,
               lake.energy.AtmosLatent, lake.energy.AtmosSensible,
               lake.energy.NetLongAtmos, lake.energy.NetShortAtmos,
               lake.energy.deltaH, lake.energy.grnd_flux,
               lake.energy.refreeze_energy, lake.energy.advection,
               lake.energy.deltaCC, lake.energy.snow_flux,
               lake.energy.Tsurf, lake.energy.error);
        for (k = 0; k < MAX_LAKE_NODES; k++) {
            printf(",%.17g", lake.temp[k]);
        }
        printf("\n");
    }
}


/* water_balance() -- the lake's water balance and the area it trades with
   the wetland. lakes.eb.c:1808-2300. */
static void
mode_lake13(void)
{
    int             c, k;
    size_t          j;
    lake_var_struct lake;
    lake_con_struct lake_con;
    soil_con_struct soil_con;
    veg_con_struct  veg_con;
    all_vars_struct all_vars;
    double          dt, lakefrac;
    int             isave_n, status;
    size_t          NL = 3;

    options.Nlayer = NL;
    options.Nnode = 3;
    options.SNOW_BAND = 1;
    options.CARBON = false;
    options.LAKE_PROFILE = false;
    options.FROZEN_SOIL = false;
    options.Nfrost = 1;
    global_param.model_steps_per_day = 24;

    all_vars = make_all_vars(1);

    printf("# waterbal\n");
    printf("# case,activenod,dz,surfdz,sarea,volume,ice_water_eq,"
           "new_ice_area,evapw,ice_throughfall,snowmlt,runoff_in,"
           "baseflow_in,channel_in,lakefrac,isave_n,dt,Dsmax,Ds,Ws,cexp,"
           "wet_moist0,wet_moist1,wet_moist2,max_moist0,max_moist1,"
           "max_moist2,resid_bot,ice_bot,Wdew,snow_canopy,wet_swq,"
           "wet_depth,wet_packw,wet_surfw,wet_baseflow,wet_runoff,"
           "wet_inflow,wet_evap0,wet_evap1,wet_evap2,canopyevap,"
           "throughfall,lake_swq,lake_depth,lake_packw,lake_surfw,"
           "lake_canopy,basin0,maxvolume,maxdepth,mindepth,minvolume,wfrac,"
           "surface0..surface20,temp0..temp19,z0..z10,basin0..basin10 "
           "| status,volume,ldepth,sarea,activenod,dz,surfdz,evapw,recharge,"
           "baseflow_out,runoff_out,lk_swq,lk_depth,lk_packw,lk_surfw,"
           "lk_canopy,lk_coverage,lk_soil_runoff,lk_soil_baseflow,"
           "lk_soil_inflow,lk_soil_evap0,wet_moist0,wet_moist1,wet_moist2,"
           "Wdew,wet_swq,wet_depth,wet_packw,wet_surfw,snow_canopy,"
           "wet_baseflow,wet_runoff,wet_inflow,wet_evap0,wet_evap1,"
           "wet_evap2,canopyevap,throughfall,"
           "surface0..surface20,temp0..temp19\n");

    memset(&soil_con, 0, sizeof(soil_con));
    soil_con.cell_area = 1.0e7;
    soil_con.lat = 47.5;
    soil_con.snow_rough = 0.001;
    soil_con.frost_fract[0] = 1.0;
    soil_con.FS_ACTIVE = false;
    for (j = 0; j < NL; j++) {
        soil_con.depth[j] = 0.1 + 0.4 * (double)j;
        soil_con.max_moist[j] = 120.0 + 60.0 * (double)j;
        soil_con.resid_moist[j] = 0.02;
        soil_con.expt[j] = 10.0;
        soil_con.bubble[j] = 30.0;
        soil_con.quartz[j] = 0.4;
        soil_con.soil_density[j] = 2650.0;
        soil_con.bulk_density[j] = 1400.0;
        soil_con.soil_dens_min[j] = 2650.0;
        soil_con.bulk_dens_min[j] = 1400.0;
        soil_con.organic[j] = 0.0;
    }
    for (k = 0; k < 3; k++) {
        soil_con.Zsum_node[k] = 0.1 * (double)k;
        soil_con.expt_node[k] = 10.0;
        soil_con.bubble_node[k] = 30.0;
        soil_con.max_moist_node[k] = 0.4;
    }
    soil_con.Ds = 0.02;
    soil_con.Dsmax = 12.0;
    soil_con.Ws = 0.8;
    soil_con.c = 2.0;
    memset(&veg_con, 0, sizeof(veg_con));
    veg_con.Cv = 1.0;

    memset(&lake_con, 0, sizeof(lake_con));
    lake_con.numnod = 10;
    lake_con.z[0] = 12.0;
    lake_con.Cl[0] = 0.10;
    lake_con.rpercent = 1.0;
    lake_con.mindepth = 0.2;
    compute_lake_params(&lake_con, soil_con);
    lake_con.wfrac = 0.02;   /* compute_lake_params does not set it */

    for (c = 0; c < 36; c++) {
        memset(&lake, 0, sizeof(lake));
        memset(&(all_vars.cell[0][0]), 0, sizeof(cell_data_struct));
        memset(&(all_vars.veg_var[0][0]), 0, sizeof(veg_var_struct));
        memset(&(all_vars.snow[0][0]), 0, sizeof(snow_data_struct));
        memset(&(all_vars.energy[0][0]), 0, sizeof(energy_bal_struct));

        dt = 3600.0;
        lakefrac = 0.25 * (double)(c % 5);         /* 0 .. 1 */
        lake.ldepth = 0.5 + 1.4 * (double)(c % 9);
        lake.ice_water_eq = (c % 4 == 0) ? 0.0 : 300.0 * (double)(1 + c % 5);
        compute_derived_lake_dimensions(&lake, lake_con);
        isave_n = (int)lake.activenod;
        lake.sarea = lake.surface[0];
        lake.new_ice_area = (lake.ice_water_eq > 0.0) ?
                            lakefrac * lake.sarea : 0.0;
        lake.hice = (lake.new_ice_area > 0.0) ?
                    lake.ice_water_eq / lake.new_ice_area * CONST_RHOFW /
                    CONST_RHOICE : 0.0;
        for (k = 0; k < MAX_LAKE_NODES; k++) {
            lake.temp[k] = (k < isave_n) ? (1.0 + 0.2 * (double)(c % 8) +
                                            0.1 * (double)k) : 0.0;
        }
        lake.evapw = 0.5 * (double)(c % 6) * lake.sarea / 1000.0;
        lake.ice_throughfall = 0.4 * (double)(c % 7) * lake.sarea / 1000.0;
        lake.snowmlt = 0.3 * (double)(c % 5) * lake.sarea / 1000.0;
        lake.runoff_in = 200.0 * (double)(c % 9);
        lake.baseflow_in = 80.0 * (double)(c % 6);
        lake.channel_in = 50.0 * (double)(c % 4);
        lake.snow.swq = 0.01 * (double)(c % 5);
        lake.snow.depth = lake.snow.swq * CONST_RHOFW / param.LAKE_RHOSNOW;
        lake.snow.pack_water = 0.0004 * (double)(c % 3);
        lake.snow.surf_water = 0.0006 * (double)(c % 4);
        lake.snow.snow_canopy = 0.0;

        for (j = 0; j < NL; j++) {
            all_vars.cell[0][0].layer[j].moist =
                40.0 + 25.0 * (double)((c + j) % 5);
            all_vars.cell[0][0].layer[j].ice[0] = 0.0;
            all_vars.cell[0][0].layer[j].evap = 0.2 * (double)((c + j) % 4);
        }
        all_vars.cell[0][0].baseflow = 1.5 * (double)(c % 5);
        all_vars.cell[0][0].runoff = 2.5 * (double)(c % 6);
        all_vars.cell[0][0].inflow = 3.5 * (double)(c % 4);
        all_vars.veg_var[0][0].Wdew = 0.6 * (double)(c % 7);
        all_vars.veg_var[0][0].canopyevap = 0.15 * (double)(c % 5);
        all_vars.veg_var[0][0].throughfall = 0.9 * (double)(c % 3);
        all_vars.snow[0][0].swq = 0.02 * (double)(c % 6);
        all_vars.snow[0][0].depth = all_vars.snow[0][0].swq * 4.0;
        all_vars.snow[0][0].pack_water = 0.0005 * (double)(c % 4);
        all_vars.snow[0][0].surf_water = 0.0007 * (double)(c % 3);
        all_vars.snow[0][0].snow_canopy = 0.001 * (double)(c % 5);
        all_vars.energy[0][0].Tsurf = 2.0;
        for (k = 0; k < 3; k++) {
            all_vars.energy[0][0].T[k] = 3.0 + (double)k;
        }

        printf("%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,"
               "%.17g",
               c, isave_n, lake.dz, lake.surfdz, lake.sarea, lake.volume,
               lake.ice_water_eq, lake.new_ice_area, lake.evapw,
               lake.ice_throughfall, lake.snowmlt, lake.runoff_in,
               lake.baseflow_in, lake.channel_in, lakefrac, isave_n, dt,
               soil_con.Dsmax, soil_con.Ds, soil_con.Ws, soil_con.c);
        for (j = 0; j < NL; j++) {
            printf(",%.17g", all_vars.cell[0][0].layer[j].moist);
        }
        for (j = 0; j < NL; j++) {
            printf(",%.17g", soil_con.max_moist[j]);
        }
        printf(",%.17g,%.17g", soil_con.resid_moist[NL - 1] *
               soil_con.depth[NL - 1] * MM_PER_M,
               all_vars.cell[0][0].layer[NL - 1].ice[0]);
        printf(",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               all_vars.veg_var[0][0].Wdew, all_vars.snow[0][0].snow_canopy,
               all_vars.snow[0][0].swq, all_vars.snow[0][0].depth,
               all_vars.snow[0][0].pack_water,
               all_vars.snow[0][0].surf_water);
        printf(",%.17g,%.17g,%.17g", all_vars.cell[0][0].baseflow,
               all_vars.cell[0][0].runoff, all_vars.cell[0][0].inflow);
        for (j = 0; j < NL; j++) {
            printf(",%.17g", all_vars.cell[0][0].layer[j].evap);
        }
        printf(",%.17g,%.17g", all_vars.veg_var[0][0].canopyevap,
               all_vars.veg_var[0][0].throughfall);
        printf(",%.17g,%.17g,%.17g,%.17g,%.17g", lake.snow.swq,
               lake.snow.depth, lake.snow.pack_water, lake.snow.surf_water,
               lake.snow.snow_canopy);
        printf(",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g", lake_con.basin[0],
               lake_con.maxvolume, lake_con.maxdepth, lake_con.mindepth,
               lake_con.minvolume, lake_con.wfrac);
        for (k = 0; k <= MAX_LAKE_NODES; k++) {
            printf(",%.17g", lake.surface[k]);
        }
        for (k = 0; k < MAX_LAKE_NODES; k++) {
            printf(",%.17g", lake.temp[k]);
        }
        /* the basin profile itself, so the comparison does not have to
           rebuild it and guess param.LAKE_BETA */
        for (k = 0; k <= (int)lake_con.numnod; k++) {
            printf(",%.17g", lake_con.z[k]);
        }
        for (k = 0; k <= (int)lake_con.numnod; k++) {
            printf(",%.17g", lake_con.basin[k]);
        }

        status = water_balance(&lake, lake_con, dt, &all_vars, 0, 0,
                               lakefrac, soil_con, veg_con);

        printf(" | %d,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g",
               status, lake.volume, lake.ldepth, lake.sarea,
               (int)lake.activenod, lake.dz, lake.surfdz, lake.evapw,
               lake.recharge, lake.baseflow_out, lake.runoff_out,
               lake.snow.swq, lake.snow.depth, lake.snow.pack_water,
               lake.snow.surf_water, lake.snow.snow_canopy,
               lake.snow.coverage, lake.soil.runoff, lake.soil.baseflow,
               lake.soil.inflow, lake.soil.layer[0].evap);
        for (j = 0; j < NL; j++) {
            printf(",%.17g", all_vars.cell[0][0].layer[j].moist);
        }
        printf(",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
               all_vars.veg_var[0][0].Wdew, all_vars.snow[0][0].swq,
               all_vars.snow[0][0].depth, all_vars.snow[0][0].pack_water,
               all_vars.snow[0][0].surf_water,
               all_vars.snow[0][0].snow_canopy);
        printf(",%.17g,%.17g,%.17g", all_vars.cell[0][0].baseflow,
               all_vars.cell[0][0].runoff, all_vars.cell[0][0].inflow);
        for (j = 0; j < NL; j++) {
            printf(",%.17g", all_vars.cell[0][0].layer[j].evap);
        }
        printf(",%.17g,%.17g", all_vars.veg_var[0][0].canopyevap,
               all_vars.veg_var[0][0].throughfall);
        for (k = 0; k <= MAX_LAKE_NODES; k++) {
            printf(",%.17g", lake.surface[k]);
        }
        for (k = 0; k < MAX_LAKE_NODES; k++) {
            printf(",%.17g", lake.temp[k]);
        }
        printf("\n");
    }
}


/* compute_lake_params() -- the basin geometry a lake parameter file
   implies. compute_lake_params.c:34-90, both profile modes. */
static void
mode_lake14(void)
{
    int             c, k;
    lake_con_struct lc;
    soil_con_struct sc;

    printf("# lakeparams\n");
    printf("# case,profile,numnod,cell_area,mindepth,rpercent,"
           "z_in0..z_in20,Cl_in0..Cl_in20 "
           "| maxdepth,maxvolume,minvolume,bpercent,basin0,"
           "z0..z20,Cl0..Cl20,basin0..basin20\n");

    for (c = 0; c < 24; c++) {
        memset(&sc, 0, sizeof(sc));
        memset(&lc, 0, sizeof(lc));
        options.LAKE_PROFILE = (c % 2 == 1);
        sc.cell_area = 1.0e6 * (double)(1 + c % 5);
        lc.numnod = 2 + (c % 9);
        lc.mindepth = 0.3 * (double)(c % 6);
        lc.rpercent = 0.1 * (double)(1 + c % 9);
        lc.z[0] = 4.0 + 3.0 * (double)(c % 7);
        lc.Cl[0] = 0.02 + 0.03 * (double)(c % 6);
        if (options.LAKE_PROFILE) {
            /* a monotone curve from the surface down to a point */
            for (k = 1; k < (int)lc.numnod; k++) {
                lc.z[k] = lc.z[0] * (1.0 - (double)k / (double)lc.numnod);
                lc.Cl[k] = lc.Cl[0] *
                           pow(1.0 - (double)k / (double)lc.numnod, 1.7);
            }
        }

        printf("%d,%d,%d,%.17g,%.17g,%.17g", c, (int)options.LAKE_PROFILE,
               (int)lc.numnod, sc.cell_area, lc.mindepth, lc.rpercent);
        for (k = 0; k <= MAX_LAKE_NODES; k++) {
            printf(",%.17g", lc.z[k]);
        }
        for (k = 0; k <= MAX_LAKE_NODES; k++) {
            printf(",%.17g", lc.Cl[k]);
        }

        compute_lake_params(&lc, sc);

        printf(" | %.17g,%.17g,%.17g,%.17g,%.17g", lc.maxdepth, lc.maxvolume,
               lc.minvolume, lc.bpercent, lc.basin[0]);
        for (k = 0; k <= MAX_LAKE_NODES; k++) {
            printf(",%.17g", lc.z[k]);
        }
        for (k = 0; k <= MAX_LAKE_NODES; k++) {
            printf(",%.17g", lc.Cl[k]);
        }
        for (k = 0; k <= MAX_LAKE_NODES; k++) {
            printf(",%.17g", lc.basin[k]);
        }
        printf("\n");
    }
}


/* One lake driven for 720 hourly steps -- vic_run.c:155-482 in sequence,
   solve_lake then water_balance, with the state carried between steps. */
static void
mode_lake15(void)
{
    int             t, k, isave_n, status;
    size_t          j;
    lake_var_struct lake;
    lake_con_struct lake_con;
    soil_con_struct soil_con;
    veg_con_struct  veg_con;
    all_vars_struct all_vars;
    dmy_struct      dmy;
    double          dt, wind_h, lakefrac, fraci, rainonly, prec;
    double          tair, wind, vp, vpd, shortin, longin, air_density, press;
    double          snowprec, rainprec, sum_runoff, sum_baseflow;
    size_t          NL = 3;
    /* the FindleyLake basin, the one lake profile VIC ships */
    double zin[11]  = {24.4, 23.4, 20.8, 18.2, 15.6, 13.0, 10.4, 7.8, 5.2,
                       2.6, 0.0};
    double clin[11] = {0.0885, 0.0884, 0.0698, 0.0535, 0.0393, 0.0273,
                       0.0175, 0.0098, 0.0044, 0.0011, 0.0};

    options.Nlayer = NL;
    options.Nnode = 3;
    options.SNOW_BAND = 1;
    options.CARBON = false;
    options.LAKE_PROFILE = true;
    options.FROZEN_SOIL = false;
    options.Nfrost = 1;
    global_param.model_steps_per_day = 24;

    all_vars = make_all_vars(1);

    printf("# lakerun\n");
    printf("# step,tair,wind,vp,vpd,shortin,longin,air_density,press,prec,"
           "sum_runoff,sum_baseflow,day_in_year "
           "| status,volume,ldepth,sarea,activenod,areai,new_ice_area,hice,"
           "ice_water_eq,swq,surf_temp,tempavg,tempi,evapw,recharge,"
           "baseflow_out,runoff_out,SAlbedo,coverage,error,fraci,lakefrac,"
           "wm0,wm1,wm2,Wdew,wet_swq\n");

    memset(&soil_con, 0, sizeof(soil_con));
    soil_con.cell_area = 1.0e7;
    soil_con.lat = 47.32;
    soil_con.snow_rough = 0.001;
    soil_con.frost_fract[0] = 1.0;
    soil_con.FS_ACTIVE = false;
    for (j = 0; j < NL; j++) {
        soil_con.depth[j] = 0.1 + 0.4 * (double)j;
        soil_con.max_moist[j] = 120.0 + 60.0 * (double)j;
        soil_con.resid_moist[j] = 0.02;
        soil_con.expt[j] = 10.0;
        soil_con.bubble[j] = 30.0;
        soil_con.quartz[j] = 0.4;
        soil_con.soil_density[j] = 2650.0;
        soil_con.bulk_density[j] = 1400.0;
        soil_con.soil_dens_min[j] = 2650.0;
        soil_con.bulk_dens_min[j] = 1400.0;
        soil_con.porosity[j] = 0.47;
        soil_con.organic[j] = 0.0;
    }
    for (k = 0; k < 3; k++) {
        soil_con.Zsum_node[k] = 0.1 * (double)k;
        soil_con.expt_node[k] = 10.0;
        soil_con.bubble_node[k] = 30.0;
        soil_con.max_moist_node[k] = 0.4;
    }
    soil_con.Ds = 0.02;
    soil_con.Dsmax = 12.0;
    soil_con.Ws = 0.8;
    soil_con.c = 2.0;
    memset(&veg_con, 0, sizeof(veg_con));
    veg_con.Cv = 0.0885;

    memset(&lake_con, 0, sizeof(lake_con));
    lake_con.numnod = 10;
    for (k = 0; k <= 10; k++) {
        lake_con.z[k] = zin[k];
        lake_con.Cl[k] = clin[k];
    }
    lake_con.rpercent = 1.0;
    lake_con.mindepth = 2.0;
    lake_con.wfrac = 0.001671;
    compute_lake_params(&lake_con, soil_con);

    memset(&lake, 0, sizeof(lake));
    memset(&(all_vars.cell[0][0]), 0, sizeof(cell_data_struct));
    memset(&(all_vars.veg_var[0][0]), 0, sizeof(veg_var_struct));
    memset(&(all_vars.snow[0][0]), 0, sizeof(snow_data_struct));
    memset(&(all_vars.energy[0][0]), 0, sizeof(energy_bal_struct));
    lake.ldepth = 6.0;
    lake.ice_water_eq = 0.0;
    compute_derived_lake_dimensions(&lake, lake_con);
    lake.sarea = lake.surface[0];
    lake.new_ice_area = 0.0;
    for (k = 0; k < MAX_LAKE_NODES; k++) {
        lake.temp[k] = (k < (int)lake.activenod) ? 6.0 : 0.0;
    }
    lake.snow.density = param.LAKE_RHOSNOW;
    for (j = 0; j < NL; j++) {
        all_vars.cell[0][0].layer[j].moist = 60.0 + 30.0 * (double)j;
    }
    all_vars.energy[0][0].Tsurf = 4.0;
    for (k = 0; k < 3; k++) {
        all_vars.energy[0][0].T[k] = 4.0;
    }

    dt = 3600.0;
    wind_h = 10.0;

    for (t = 0; t < 720; t++) {
        dmy.day_in_year = 1 + t / 24;
        /* a month cooling to well below freezing and back */
        tair = 8.0 - 13.0 * sin(CONST_PI * (double)t / 720.0)
               + 3.0 * sin(2.0 * CONST_PI * (double)(t % 24) / 24.0);
        wind = 1.5 + 1.0 * sin(CONST_PI * (double)t / 90.0);
        if (wind < 0.2) {
            wind = 0.2;
        }
        shortin = 240.0 * sin(CONST_PI * (double)(t % 24) / 24.0);
        if (shortin < 0.0) {
            shortin = 0.0;
        }
        longin = 240.0 + 3.0 * tair;
        air_density = 1.25;
        press = 95.0;
        vp = 0.4 + 0.25 * sin(CONST_PI * (double)t / 240.0);
        vpd = svp(tair) / PA_PER_KPA - vp;
        if (vpd < 0.0) {
            vpd = 0.0;
        }
        prec = (t % 17 == 0) ? 2.5 : 0.0;
        sum_runoff = 0.3 + 0.2 * sin(CONST_PI * (double)t / 120.0);
        sum_baseflow = 0.1;

        /* vic_run.c:156-172 */
        lake.areai = lake.new_ice_area;
        if (lake.areai < 0) {
            lake.areai = 0;
        }
        fraci = (lake.sarea > 0) ? lake.areai / lake.sarea : 0.0;
        if (fraci > 1.0) {
            fraci = 1.0;
        }
        lakefrac = lake.sarea / lake_con.basin[0];

        /* vic_run.c:435-447 */
        lake.runoff_in = sum_runoff * lake_con.rpercent *
                         soil_con.cell_area / MM_PER_M;
        lake.baseflow_in = sum_baseflow * lake_con.rpercent *
                           soil_con.cell_area / MM_PER_M;
        lake.channel_in = 0.0;
        lake.prec = prec * lake.sarea / MM_PER_M;
        rainonly = calc_rainonly(tair, prec, param.SNOW_MAX_SNOW_TEMP,
                                 param.SNOW_MIN_RAIN_TEMP);
        snowprec = prec - rainonly;
        rainprec = rainonly;

        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%u",
               t, tair, wind, vp, vpd, shortin, longin, air_density, press,
               prec, sum_runoff, sum_baseflow, dmy.day_in_year);

        status = solve_lake(snowprec, rainprec, tair, wind, vp, shortin,
                            longin, vpd, press, air_density, &lake, soil_con,
                            dt, wind_h, dmy, fraci);
        if (status != 0) {
            printf(" | solve_lake failed\n");
            return;
        }
        isave_n = (int)lake.activenod;
        status = water_balance(&lake, lake_con, dt, &all_vars, 0, 0,
                               lakefrac, soil_con, veg_con);

        printf(" | %d,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               status, lake.volume, lake.ldepth, lake.sarea,
               (int)lake.activenod, lake.areai, lake.new_ice_area, lake.hice,
               lake.ice_water_eq, lake.snow.swq, lake.snow.surf_temp,
               lake.tempavg, lake.tempi, lake.evapw, lake.recharge,
               lake.baseflow_out, lake.runoff_out, lake.SAlbedo,
               lake.snow.coverage, lake.energy.error, fraci, lakefrac,
               all_vars.cell[0][0].layer[0].moist,
               all_vars.cell[0][0].layer[1].moist,
               all_vars.cell[0][0].layer[2].moist,
               all_vars.veg_var[0][0].Wdew, all_vars.snow[0][0].swq);
        (void)isave_n;
    }
}


/* solve_T_profile() -- the soil temperature profile, frozen_soil.c:125-246
   and its Gauss-Seidel sweep at :346-560. Both grid branches, with and
   without the no-flux bottom, frozen and unfrozen. */
static void
mode_tprof(void)
{
    int      c, j, k, Nnodes, FIRST_SOLN, NOFLUX, EXP_TRANS, FS_ACTIVE;
    int      status;
    double   T[MAX_NODES], T0[MAX_NODES], Zsum[MAX_NODES], kappa[MAX_NODES];
    double   Cs[MAX_NODES], moist[MAX_NODES], max_moist[MAX_NODES];
    double   bubble[MAX_NODES], expt[MAX_NODES], ice[MAX_NODES];
    double   alpha[MAX_NODES], beta[MAX_NODES], gamma[MAX_NODES];
    char     Tfbflag[MAX_NODES];
    unsigned Tfbcount[MAX_NODES];
    double   deltat, Dp, dz;

    options.FROZEN_SOIL = true;
    options.TFALLBACK = true;
    options.Nlayer = 3;

    printf("# tprofile\n");
    printf("# case,Nnodes,NOFLUX,EXP_TRANS,FS_ACTIVE,deltat,Dp,"
           "T0_0..T0_%d,Zsum,kappa,Cs,moist,max_moist,bubble,expt,ice,"
           "alpha,beta,gamma | status,T_0..T_%d,fbflag_0..fbflag_%d\n",
           MAX_NODES - 1, MAX_NODES - 1, MAX_NODES - 1);

    for (c = 0; c < 48; c++) {
        Nnodes = 3 + (c % 6);                       /* 3 to 8 */
        NOFLUX = (c / 6) % 2;
        EXP_TRANS = (c / 12) % 2;
        FS_ACTIVE = (c / 24) % 2;
        deltat = 3600.0;
        Dp = 4.0;
        dz = Dp / (double)(Nnodes - 1);

        memset(T, 0, sizeof(T));
        memset(Tfbflag, 0, sizeof(Tfbflag));
        memset(Tfbcount, 0, sizeof(Tfbcount));
        for (j = 0; j < MAX_NODES; j++) {
            Zsum[j] = 0;
            kappa[j] = 0;
            Cs[j] = 0;
            moist[j] = 0;
            max_moist[j] = 0;
            bubble[j] = 0;
            expt[j] = 0;
            ice[j] = 0;
            alpha[j] = 0;
            beta[j] = 0;
            gamma[j] = 0;
            T0[j] = 0;
        }
        for (j = 0; j < Nnodes; j++) {
            if (EXP_TRANS) {
                Zsum[j] = (exp(log(Dp + 1.) * (double)j /
                               (double)(Nnodes - 1)) - 1.);
            }
            else {
                Zsum[j] = dz * (double)j;
            }
            /* a profile that crosses freezing in the middle for half the
               cases and stays above it for the other half */
            T0[j] = (c % 2 == 0)
                    ? (-6.0 + 1.6 * (double)j + 0.4 * (double)(c % 5))
                    : (3.0 + 0.5 * (double)j + 0.3 * (double)(c % 4));
            kappa[j] = 1.2 + 0.05 * (double)j + 0.02 * (double)(c % 7);
            Cs[j] = 2.0e6 + 4.0e4 * (double)j;
            /* the thermal nodes carry VOLUMETRIC moisture, not millimetres:
               max_moist_node is the porosity and moist[] the water fraction
               (distribute_node_moisture_properties). Feeding millimetres
               here makes the fusion term E*(0-ice) swamp the equation and
               the explicit scheme diverges -- faithfully, in VIC as much as
               here, to thousands of degrees. */
            max_moist[j] = 0.40 + 0.01 * (double)j;
            moist[j] = 0.45 * max_moist[j] + 0.02 * (double)(c % 4);
            bubble[j] = 20.0 + 5.0 * (double)(c % 6);
            expt[j] = 8.0 + 1.0 * (double)(c % 5);
            ice[j] = (T0[j] < 0.0) ? 0.2 * moist[j] : 0.0;
        }
        for (j = 0; j < Nnodes - 1; j++) {
            alpha[j] = Zsum[j + 2 <= Nnodes - 1 ? j + 2 : Nnodes - 1] -
                       Zsum[j];
            beta[j] = Zsum[j + 1] - Zsum[j];
            gamma[j] = Zsum[j + 1 <= Nnodes - 1 ? j + 1 : Nnodes - 1] -
                       Zsum[j];
            if (alpha[j] <= 0) {
                alpha[j] = 2.0 * beta[j];
            }
            if (gamma[j] <= 0) {
                gamma[j] = beta[j];
            }
        }

        printf("%d,%d,%d,%d,%d,%.17g,%.17g", c, Nnodes, NOFLUX, EXP_TRANS,
               FS_ACTIVE, deltat, Dp);
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", T0[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", Zsum[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", kappa[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", Cs[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", moist[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", max_moist[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", bubble[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", expt[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", ice[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", alpha[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", beta[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", gamma[k]);
        }

        FIRST_SOLN = true;
        status = solve_T_profile(T, T0, Tfbflag, Tfbcount, Zsum, kappa, Cs,
                                 moist, deltat, max_moist, bubble, expt, ice,
                                 alpha, beta, gamma, Dp, Nnodes, &FIRST_SOLN,
                                 FS_ACTIVE, NOFLUX, EXP_TRANS);

        printf(" | %d", status);
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", T[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%d", (int)Tfbflag[k]);
        }
        printf("\n");
    }
}


/* solve_T_profile_implicit() -- the soil temperature profile by
   Newton-Raphson, frozen_soil.c:235-322 with fda_heat_eqn at :624-1010.
   Both grid branches, with and without the no-flux bottom. The three-node
   grid without the no-flux bottom is deliberately absent: there VIC reads
   T_2[1] past its single unknown (frozen_soil.c:793-796), uninitialized
   memory, and torchvic refuses that grid instead of reproducing it. The
   soil arithmetic here is identical in VIC 5.0.1 and 5.1.0 (only the
   OpenMP threadprivate pragmas differ), so this 5.0.1 oracle also checks
   the 5.1.0 core.
   `only` runs a single case: fda_heat_eqn keeps its trial conductivities
   in statics that each evaluation resets only over 0..n (:745), while the
   bottom row without the no-flux bottom reads kappa_new[n + 1] (:811).
   With a fixed node count -- every real run -- that slot is never
   written and stays 0.0, but cases of mixed sizes in one process would
   poison it, so each case gets a fresh process. */
static void
mode_tprof_implicit(int only)
{
    int      c, j, k, Nnodes, FIRST_SOLN[2], NOFLUX, EXP_TRANS;
    int      status;
    double   T[MAX_NODES], T0[MAX_NODES], Zsum[MAX_NODES], kappa[MAX_NODES];
    double   Cs[MAX_NODES], moist[MAX_NODES], max_moist[MAX_NODES];
    double   bubble[MAX_NODES], expt[MAX_NODES], ice[MAX_NODES];
    double   alpha[MAX_NODES], beta[MAX_NODES], gamma[MAX_NODES];
    double   depth[3] = { 0.5, 1.5, 4.0 };
    double   bdmin[3] = { 1300.0, 1300.0, 1300.0 };
    double   sdmin[3] = { 2650.0, 2650.0, 2650.0 };
    double   qz[3] = { 0.5, 0.5, 0.5 };
    double   bd[3] = { 1300.0, 1300.0, 1300.0 };
    double   sd[3] = { 2650.0, 2650.0, 2650.0 };
    double   org[3] = { 0.0, 0.0, 0.0 };
    char     Tfbflag[MAX_NODES];
    unsigned Tfbcount[MAX_NODES];
    double   deltat, Dp, dz;

    options.FROZEN_SOIL = true;
    options.TFALLBACK = true;
    options.Nlayer = 3;

    printf("# tprofile_implicit\n");
    printf("# case,Nnodes,NOFLUX,EXP_TRANS,deltat,Dp,"
           "T0_0..T0_%d,Zsum,kappa,Cs,moist,max_moist,bubble,expt,ice,"
           "alpha,beta,gamma,depth_0..2,bdmin,sdmin,qz,bd,sd,org"
           " | status,T_0..T_%d,fbflag_0..fbflag_%d\n",
           MAX_NODES - 1, MAX_NODES - 1, MAX_NODES - 1);

    for (c = 0; c < 48; c++) {
        if (only >= 0 && c != only) {
            continue;
        }
        if (c < 8) {
            /* the three-node grid is only defined with the no-flux
               bottom (then n = Nnodes - 1 = 2 unknowns) */
            Nnodes = 3;
            NOFLUX = 1;
            EXP_TRANS = (c / 4) % 2;
        }
        else {
            Nnodes = 4 + ((c - 8) % 5);             /* 4 to 8 */
            NOFLUX = ((c - 8) / 5) % 2;
            EXP_TRANS = ((c - 8) / 10) % 2;
        }
        deltat = 10800.0;
        Dp = 4.0;
        dz = Dp / (double)(Nnodes - 1);

        memset(T, 0, sizeof(T));
        memset(Tfbflag, 0, sizeof(Tfbflag));
        memset(Tfbcount, 0, sizeof(Tfbcount));
        for (j = 0; j < MAX_NODES; j++) {
            Zsum[j] = 0;
            kappa[j] = 0;
            Cs[j] = 0;
            moist[j] = 0;
            max_moist[j] = 0;
            bubble[j] = 0;
            expt[j] = 0;
            ice[j] = 0;
            alpha[j] = 0;
            beta[j] = 0;
            gamma[j] = 0;
            T0[j] = 0;
        }
        for (j = 0; j < Nnodes; j++) {
            if (EXP_TRANS) {
                Zsum[j] = (exp(log(Dp + 1.) * (double)j /
                               (double)(Nnodes - 1)) - 1.);
            }
            else {
                Zsum[j] = dz * (double)j;
            }
            /* smooth gradients, crossing freezing in half the cases */
            T0[j] = (c % 2 == 0)
                    ? (-4.0 + 1.2 * (double)j + 0.3 * (double)(c % 5))
                    : (4.0 + 0.6 * (double)j + 0.2 * (double)(c % 4));
            kappa[j] = 1.2 + 0.05 * (double)j + 0.02 * (double)(c % 7);
            Cs[j] = 2.0e6 + 4.0e4 * (double)j;
            /* volumetric moisture, as in mode_tprof */
            max_moist[j] = 0.40 + 0.01 * (double)j;
            moist[j] = 0.45 * max_moist[j] + 0.02 * (double)(c % 4);
            bubble[j] = 20.0 + 5.0 * (double)(c % 6);
            expt[j] = 8.0 + 1.0 * (double)(c % 5);
            ice[j] = (T0[j] < 0.0) ? 0.2 * moist[j] : 0.0;
        }
        for (j = 0; j < Nnodes - 1; j++) {
            alpha[j] = Zsum[j + 2 <= Nnodes - 1 ? j + 2 : Nnodes - 1] -
                       Zsum[j];
            beta[j] = Zsum[j + 1] - Zsum[j];
            gamma[j] = Zsum[j + 1 <= Nnodes - 1 ? j + 1 : Nnodes - 1] -
                       Zsum[j];
            if (alpha[j] <= 0) {
                alpha[j] = 2.0 * beta[j];
            }
            if (gamma[j] <= 0) {
                gamma[j] = beta[j];
            }
        }

        printf("%d,%d,%d,%d,%.17g,%.17g", c, Nnodes, NOFLUX, EXP_TRANS,
               deltat, Dp);
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", T0[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", Zsum[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", kappa[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", Cs[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", moist[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", max_moist[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", bubble[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", expt[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", ice[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", alpha[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", beta[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", gamma[k]);
        }
        for (k = 0; k < 3; k++) {
            printf(",%.17g", depth[k]);
        }
        for (k = 0; k < 3; k++) {
            printf(",%.17g", bdmin[k]);
        }
        for (k = 0; k < 3; k++) {
            printf(",%.17g", sdmin[k]);
        }
        for (k = 0; k < 3; k++) {
            printf(",%.17g", qz[k]);
        }
        for (k = 0; k < 3; k++) {
            printf(",%.17g", bd[k]);
        }
        for (k = 0; k < 3; k++) {
            printf(",%.17g", sd[k]);
        }
        for (k = 0; k < 3; k++) {
            printf(",%.17g", org[k]);
        }

        FIRST_SOLN[0] = true;
        FIRST_SOLN[1] = true;
        status = solve_T_profile_implicit(T, T0, Tfbflag, Tfbcount, Zsum,
                                          kappa, Cs, moist, deltat,
                                          max_moist, bubble, expt, ice,
                                          alpha, beta, gamma, Dp, Nnodes,
                                          FIRST_SOLN, NOFLUX, EXP_TRANS,
                                          bdmin, sdmin, qz, bd, sd, org,
                                          depth);

        printf(" | %d", status);
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%.17g", T[k]);
        }
        for (k = 0; k < MAX_NODES; k++) {
            printf(",%d", (int)Tfbflag[k]);
        }
        printf("\n");
    }
}


int
main(int argc, char **argv)
{
    if (argc != 2 && (argc != 3 || strcmp(argv[1], "tprof_implicit") != 0)) {
        fprintf(stderr, "usage: ref_snow <dens|pack|eb|melt|blow|blow1|carbon|soilresp|rcps|coszen>\n");
        return 2;
    }
    init_globals();
    if (strcmp(argv[1], "dens") == 0) {
        mode_dens();
    }
    else if (strcmp(argv[1], "pack") == 0) {
        mode_pack();
    }
    else if (strcmp(argv[1], "eb") == 0) {
        mode_eb();
    }
    else if (strcmp(argv[1], "melt") == 0) {
        mode_melt();
    }
    else if (strcmp(argv[1], "blow") == 0) {
        mode_blow(true);
    }
    else if (strcmp(argv[1], "blow1") == 0) {
        mode_blow(false);
    }
    else if (strcmp(argv[1], "carbon") == 0) {
        mode_carbon();
    }
    else if (strcmp(argv[1], "soilresp") == 0) {
        mode_soilresp();
    }
    else if (strcmp(argv[1], "rcps") == 0) {
        mode_rcps();
    }
    else if (strcmp(argv[1], "coszen") == 0) {
        mode_coszen();
    }
    else if (strcmp(argv[1], "zwtlump") == 0) {
        mode_zwtlump();
    }
    else if (strcmp(argv[1], "lake") == 0) {
        mode_lake();
    }
    else if (strcmp(argv[1], "lake2") == 0) {
        mode_lake2();
    }
    else if (strcmp(argv[1], "lake3") == 0) {
        mode_lake3();
    }
    else if (strcmp(argv[1], "lake4") == 0) {
        mode_lake4();
    }
    else if (strcmp(argv[1], "lake5") == 0) {
        mode_lake5();
    }
    else if (strcmp(argv[1], "lake6") == 0) {
        mode_lake6();
    }
    else if (strcmp(argv[1], "lake7") == 0) {
        mode_lake7();
    }
    else if (strcmp(argv[1], "lake8") == 0) {
        mode_lake8();
    }
    else if (strcmp(argv[1], "lake9") == 0) {
        mode_lake9();
    }
    else if (strcmp(argv[1], "lake10") == 0) {
        mode_lake10();
    }
    else if (strcmp(argv[1], "lake11") == 0) {
        mode_lake11();
    }
    else if (strcmp(argv[1], "lake12") == 0) {
        mode_lake12();
    }
    else if (strcmp(argv[1], "lake13") == 0) {
        mode_lake13();
    }
    else if (strcmp(argv[1], "lake14") == 0) {
        mode_lake14();
    }
    else if (strcmp(argv[1], "lake15") == 0) {
        mode_lake15();
    }
    else if (strcmp(argv[1], "tprof") == 0) {
        mode_tprof();
    }
    else if (strcmp(argv[1], "tprof_implicit") == 0) {
        mode_tprof_implicit(argc > 2 ? atoi(argv[2]) : -1);
    }
    else {
        fprintf(stderr, "unknown mode %s\n", argv[1]);
        return 2;
    }
    return 0;
}
