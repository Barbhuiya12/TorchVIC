/* Oracle of the VIC-WUR vic_run hooks and core.
 * Links the unchanged VIC-WUR 3.0 vic_run (<data root>/build/vicwur3/src_tree/vic,
 * every vic_run/src/*.c) and VIC-WUR 3.0's drivers/shared_all calc_root_fraction.c,
 * and calls:
 *   root    calc_root_fractions()   VIC-WUR 3.0 calc_root_fraction.c:34-184, after
 *           VIC-WUR 3.0 vic_init.c:1255-1267's zone_fract normalisation (copied)
 *   transp  transpiration()         VIC-WUR 3.0 canopy_evap.c:168-539: layer Wcr,
 *           root, LAI and veg_lib per call, b_co2 with Catm, SHARE on/off;
 *           returns layertransp (after the C's wilting clamp), the stored
 *           layer[i].water_stress and veg_var->rc
 *   column  runoff()                VIC-WUR 3.0 runoff.c:35-443 over a series, with
 *           layer[l].Ksat per step (VIC-WUR 3.0 runoff.c:156-157)
 * tests/test_vicwur_core.py writes the input and compares
 * torchvic.core.vegetation.root_fractions_w3 / transpiration and
 * torchvic.core.column.step with the Ksat of the hook.
 * Little-endian float64 in and out.
 *   root:   in  n nz nl, per record zone_depth[nz] zone_fract[nz] depth[nl]
 *           out per record root[nl]
 *   transp: in  n share, per record moist[3] ice[3] Wmax[3] Wcr[3] Wpwp[3]
 *           root[3] LAI rmin RGL rarc b_co2 Catm rad vpd net_short air_temp
 *           ra dryFrac delta_t elevation   (32)
 *           out per record layertransp[3] water_stress[3] rc   (7)
 *   column: in  nsteps runoff_steps b_infilt Ds Dsmax Ws c, max_moist[3]
 *           expt[3] Ksat[3] resid_frac[3] depth[3] moist0[3], then per step
 *           ppt evap[3] layer_Ksat[3]   (7)
 *           out per step moist[3] runoff baseflow   (5)
 *   gwm     runoff_GWM()            VIC-WUR 3.0 runoff.c:445-773 with options.GWM
 *           TRUE (compute_runoff_and_asat over every layer, :797-807), the
 *           column input as above
 *           out per step moist[3] runoff baseflow recharge GWrecharge (7)
 */
#include <vic_driver_shared_all.h>

global_param_struct global_param;
option_struct       options;
parameters_struct   param;
size_t              NR;
size_t              NF;

#define NL 3
/* VIC-WUR 3.0 vic_driver_shared_image.h:37 (the image driver header, not included) */
#define AREA_SUM_ERROR_THRESH 1e-5

static int
rd(FILE *f, double *x, size_t n)
{
    return fread(x, 8, n, f) == n;
}

static void
init_globals(void)
{
    LOG_DEST = stderr;
    initialize_options();
    initialize_global();
    initialize_parameters();
    options.Nlayer = NL;
    options.Nnode = 3;
    options.SNOW_BAND = 1;
    options.FROZEN_SOIL = false;
    options.Nfrost = 1;
    options.CARBON = false;
    options.GWM = false;
    NR = 0;
    NF = 1;
}

static int
mode_root(FILE *fi, FILE *fo)
{
    double         h[3], zd[64], zf[64], dep[MAX_LAYERS], sum;
    size_t         n, nz, nl, r, k;
    veg_con_struct vc;
    soil_con_struct sc;

    if (!rd(fi, h, 3)) {
        return 3;
    }
    n = (size_t) h[0];
    nz = (size_t) h[1];
    nl = (size_t) h[2];
    options.ROOT_ZONES = nz;
    options.Nlayer = nl;
    for (r = 0; r < n; r++) {
        if (!rd(fi, zd, nz) || !rd(fi, zf, nz) || !rd(fi, dep, nl)) {
            return 4;
        }
        /* VIC-WUR 3.0 vic_init.c:1255-1267 */
        sum = 0;
        for (k = 0; k < nz; k++) {
            sum += zf[k];
        }
        if (!assert_close_double(sum, 1.0, 0., AREA_SUM_ERROR_THRESH)) {
            for (k = 0; k < nz; k++) {
                zf[k] /= sum;
            }
        }
        memset(&vc, 0, sizeof(vc));
        memset(&sc, 0, sizeof(sc));
        vc.vegetat_type_num = 1;
        vc.zone_depth = zd;
        vc.zone_fract = zf;
        for (k = 0; k < nl; k++) {
            sc.depth[k] = dep[k];
        }
        calc_root_fractions(&vc, &sc);
        fwrite(vc.root, 8, nl, fo);
    }
    return 0;
}

static int
mode_transp(FILE *fi, FILE *fo)
{
    double            h[2], x[32], o[7], Wmax[NL], Wcr[NL], Wpwp[NL];
    double            root[NL], lt[NL], frost_fract[MAX_FROST_AREAS];
    double            bnd[MAX_LAYERS];
    size_t            n, r, i;
    layer_data_struct layer[MAX_LAYERS];
    veg_var_struct    vv;
    veg_lib_struct    vl;

    if (!rd(fi, h, 2)) {
        return 3;
    }
    n = (size_t) h[0];
    options.SHARE_LAYER_MOIST = h[1] != 0;
    memset(frost_fract, 0, sizeof(frost_fract));
    frost_fract[0] = 1.0;
    memset(bnd, 0, sizeof(bnd));
    for (r = 0; r < n; r++) {
        if (!rd(fi, x, 32)) {
            return 4;
        }
        memset(layer, 0, sizeof(layer));
        memset(&vv, 0, sizeof(vv));
        memset(&vl, 0, sizeof(vl));
        for (i = 0; i < NL; i++) {
            layer[i].moist = x[i];
            layer[i].ice[0] = x[3 + i];
            Wmax[i] = x[6 + i];
            Wcr[i] = x[9 + i];
            Wpwp[i] = x[12 + i];
            root[i] = x[15 + i];
            layer[i].water_stress = -1.0;     /* shows an untouched layer */
            lt[i] = 0.;
        }
        vv.LAI = x[18];
        vl.rmin = x[19];
        vl.RGL = x[20];
        vl.rarc = x[21];
        vl.b_co2 = x[22];
        vl.Wnpp_inhib = 1.0;
        vl.NPPfactor_sat = 1.0;
        transpiration(layer, &vv, &vl, x[24], x[25], x[26], x[27], x[28],
                      x[29], x[30], x[31], Wmax, Wcr, Wpwp, lt, frost_fract,
                      root, 0.0, x[23], bnd);
        for (i = 0; i < NL; i++) {
            o[i] = lt[i];
            o[3 + i] = layer[i].water_stress;
        }
        o[6] = vv.rc;
        fwrite(o, 8, 7, fo);
    }
    return 0;
}

static int
mode_column(FILE *fi, FILE *fo, int gwm)
{
    double            h[7 + 6 * NL], s[7], o[7];
    double            frost_fract[MAX_FROST_AREAS];
    size_t            nsteps, t, i;
    soil_con_struct   soil;
    cell_data_struct  cell;
    energy_bal_struct energy;

    if (!rd(fi, h, 7 + 6 * NL)) {
        return 3;
    }
    nsteps = (size_t) h[0];
    global_param.runoff_steps_per_day = (unsigned short) h[1];
    global_param.model_steps_per_day = 1;
    memset(&soil, 0, sizeof(soil));
    soil.b_infilt = h[2];
    soil.Ds = h[3];
    soil.Dsmax = h[4];
    soil.Ws = h[5];
    soil.c = h[6];
    for (i = 0; i < NL; i++) {
        soil.max_moist[i] = h[7 + i];
        soil.expt[i] = h[7 + NL + i];
        soil.Ksat[i] = h[7 + 2 * NL + i];
        soil.resid_moist[i] = h[7 + 3 * NL + i];     /* a fraction in VIC-WUR 3.0 */
        soil.depth[i] = h[7 + 4 * NL + i];
    }
    memset(&cell, 0, sizeof(cell));
    memset(&energy, 0, sizeof(energy));
    memset(frost_fract, 0, sizeof(frost_fract));
    frost_fract[0] = 1.0;
    for (i = 0; i < NL; i++) {
        cell.layer[i].moist = h[7 + 5 * NL + i];
    }
    for (t = 0; t < nsteps; t++) {
        if (!rd(fi, s, 7)) {
            return 4;
        }
        for (i = 0; i < NL; i++) {
            cell.layer[i].evap = s[1 + i];
            cell.layer[i].Ksat = s[4 + i];
        }
        cell.runoff = 0.;
        cell.baseflow = 0.;
        options.GWM = gwm ? true : false;
        if ((gwm ? runoff_GWM(&cell, &energy, &soil, s[0], frost_fract, 3)
             : runoff(&cell, &energy, &soil, s[0], frost_fract, 3))
            == ERROR) {
            return 5;
        }
        for (i = 0; i < NL; i++) {
            o[i] = cell.layer[i].moist;
        }
        o[3] = cell.runoff;
        o[4] = cell.baseflow;
        o[5] = cell.recharge;
        o[6] = cell.GWrecharge;
        fwrite(o, 8, gwm ? 7 : 5, fo);
    }
    return 0;
}

int
main(int argc, char **argv)
{
    FILE *fi, *fo;
    int   rc;

    if (argc != 4) {
        fprintf(stderr, "usage: %s root|transp|column|gwm in out\n", argv[0]);
        return 1;
    }
    fi = fopen(argv[2], "rb");
    fo = fopen(argv[3], "wb");
    if (!fi || !fo) {
        return 2;
    }
    init_globals();
    if (strcmp(argv[1], "root") == 0) {
        rc = mode_root(fi, fo);
    }
    else if (strcmp(argv[1], "transp") == 0) {
        rc = mode_transp(fi, fo);
    }
    else if (strcmp(argv[1], "column") == 0) {
        rc = mode_column(fi, fo, 0);
    }
    else if (strcmp(argv[1], "gwm") == 0) {
        rc = mode_column(fi, fo, 1);
    }
    else {
        rc = 1;
    }
    fclose(fi);
    fclose(fo);
    return rc;
}
