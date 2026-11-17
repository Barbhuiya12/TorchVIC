/* Oracle of the VIC-WUR CO2 forcing and b_co2 hook.
 *
 * Links the unchanged VIC-WUR 3.0 vic_run/src/penman.c (calc_rc with the co2factor,
 * :34-92) compiled against the real vic_run headers, and copies the three
 * pieces of arithmetic that live in files too entangled with the driver to
 * include:
 *   plugins/co2/src/co2_update_step_vars.c:14-17   Catm[i<NF] = CO2 * 1e-6,
 *                                                  Catm[NR] = average(Catm,NF)
 *   drivers/shared_all/src/forcing_utils.c:33-51   average()
 *   drivers/image/src/vic_force.c:263-267,489-500  default Catm
 * and the call convention of canopy_evap.c:302 / surface_fluxes.c:793
 * (calc_rc receives Catm / PPM_to_MIXRATIO).
 * tests/test_vicwur_co2.py writes the input and compares
 * torchvic.vicwur.co2 and torchvic.core.vegetation.calc_rc(co2factor=).
 *
 * Input (little endian float64):  n nf co2ref  then per record
 *   rs net_short rgl tair vpd co2 lai b_co2 gsm_inv ref_crop use_default
 * Output per record:
 *   rc_raw catm[0..3] catm_nr ppm_back rc_via_catm   (8)
 *   rc_raw      calc_rc with the forcing ppm as co2
 *   catm[0..3]  Catm[i<nf] (rest 0)
 *   (use_default != 0: no CO2 forcing, Catm = co2ref * 1e-6)
 */
#include <vic_run.h>

parameters_struct param;
option_struct     options;

#define NO 8

static double
average_c(double *ar, size_t n)     /* forcing_utils.c:33-51 */
{
    size_t i;
    double sum = 0.;

    if (n == 1) {
        return ar[0];
    }
    for (i = 0; i < n; i++) {
        sum += ar[i];
    }
    return sum / n;
}

int
main(int argc, char **argv)
{
    FILE  *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    double h[3], r[11], o[NO], catm[5];
    size_t n, c, nf, i;

    if (!fi || !fo || fread(h, 8, 3, fi) != 3) {
        return 2;
    }
    n = (size_t) h[0];
    nf = (size_t) h[1];
    /* initialize_parameters.c:67-80 */
    param.CANOPY_CLOSURE = 4000.;
    param.CANOPY_RSMAX = 5000.;
    param.CANOPY_VPDMINFACTOR = 0.1;
    param.CANOPY_CO2REF = h[2];
    for (c = 0; c < n; c++) {
        double rs, ns, rgl, tair, vpd, co2, lai, b, gsm, rc_raw, ppm;
        int    ref_crop, use_default;

        if (fread(r, 8, 11, fi) != 11) {
            return 3;
        }
        rs = r[0]; ns = r[1]; rgl = r[2]; tair = r[3]; vpd = r[4];
        co2 = r[5]; lai = r[6]; b = r[7]; gsm = r[8];
        ref_crop = r[9] != 0; use_default = r[10] != 0;
        rc_raw = calc_rc(rs, ns, rgl, tair, vpd, co2, lai, b, gsm, ref_crop);
        for (i = 0; i < 5; i++) {
            catm[i] = 0.;
        }
        for (i = 0; i < nf; i++) {
            if (use_default) {
                catm[i] = (double) param.CANOPY_CO2REF * 1.0e-6;
            }
            else {
                catm[i] = (double) co2 * 1.0e-6;
            }
        }
        catm[nf] = average_c(catm, nf);
        ppm = catm[0] / 1.0e-6;
        o[0] = rc_raw;
        for (i = 0; i < 4; i++) {
            o[1 + i] = (i < nf) ? catm[i] : 0.;
        }
        o[5] = catm[nf];
        o[6] = ppm;
        o[7] = calc_rc(rs, ns, rgl, tair, vpd, ppm, lai, b, gsm, ref_crop);
        fwrite(o, 8, NO, fo);
    }
    fclose(fo);
    fclose(fi);
    return 0;
}
