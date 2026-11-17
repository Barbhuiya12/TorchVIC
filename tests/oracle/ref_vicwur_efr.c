/* Oracle for VIC-WUR environmental flow requirements.
 *
 * Standalone single-cell copy of the arithmetic of VIC-WUR 3.0
 *   plugins/efr/src/efr_force.c:34-126     efr_derived_forcing
 *   plugins/efr/src/efr_output.c:107-126   efr_put_data (OUT_EFR_MOIST)
 * with the C structs replaced by arrays of fixed size.  The only edit is that
 * `soil_con->Ws`, `->Ds` and `->c` of the non-linear term (cell 0 of the
 * local domain in C, efr_force.c:94-98) come from the record (Ds0, Ws0, c0)
 * when `bug` is 1, and from the cell itself when `bug` is 0.  The `min`
 * macro is vic_def.h:91.  Reads records of doubles from argv[1], writes
 * records of doubles to argv[2].  Layout: tests/test_vicwur_efr.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define min(a, b) (a < b) ? a : b
#define NV 3
#define NB 3
#define NF 2
#define NG 100
#define MM_PER_M 1000.
#define NIN (15 + NF + NB + NV + NV*NB + NV*NB*NF)
#define NOUT (3 + NV*NB + NG)

static int nv_active, bug;
static double Ds_i, Ws_i, c_i, Ds_0, Ws_0, c_0, dsmax, res_moist, max_moist;
static double AF[NB], Cv[NV], ff[NF], moist[NV][NB], ice_[NV][NB][NF];

/* efr_force.c:72-103: calculated_baseflow at one frac */
static double calc_bf(double frac)
{
    double calculated_baseflow = 0.0, veg_frac, area_frac, mo, ice, liq;
    double rel_liq, bflow, ws, ds, c;
    int j, k, f;

    ds = bug ? Ds_0 : Ds_i; ws = bug ? Ws_0 : Ws_i; c = bug ? c_0 : c_i;
    for (j = 0; j < nv_active; j++) {
        veg_frac = Cv[j];
        for (k = 0; k < NB; k++) {
            area_frac = AF[k];
            mo = moist[j][k];
            ice = 0.0;
            for (f = 0; f < NF; f++) {
                ice += ice_[j][k][f] * ff[f];
            }
            liq = mo - ice;
            rel_liq = (liq * frac - res_moist) / (max_moist - res_moist);
            bflow = rel_liq * dsmax * Ds_i / Ws_i;
            if (rel_liq > ws) {
                bflow += dsmax * (1 - ds / ws) *
                         pow((rel_liq - ws) / (1 - ws), c);
            }
            if (bflow < 0) {
                bflow = 0.0;
            }
            calculated_baseflow += bflow * veg_frac * area_frac;
        }
    }
    return calculated_baseflow;
}

int main(int argc, char **argv)
{
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    double n;
    int c;

    if (!fi || !fo || fread(&n, 8, 1, fi) != 1) return 1;
    for (c = 0; c < (int) n; c++) {
        double in[NIN], *p = in, out[NOUT], *q = out;
        double resid, depth, Dsmax, spd, req, disch, frac, cellm = 0, mo, ice, liq;
        double em[NV][NB];
        int i, j, k, f;

        if (fread(in, 8, NIN, fi) != NIN) return 2;
        bug = (int) *p++; nv_active = (int) *p++; spd = *p++;
        resid = *p++; depth = *p++; max_moist = *p++; Dsmax = *p++;
        Ds_i = *p++; Ws_i = *p++; c_i = *p++; Ds_0 = *p++; Ws_0 = *p++;
        c_0 = *p++; req = *p++; disch = *p++;
        for (f = 0; f < NF; f++) ff[f] = *p++;
        for (k = 0; k < NB; k++) AF[k] = *p++;
        for (j = 0; j < NV; j++) Cv[j] = *p++;
        for (j = 0; j < NV; j++) for (k = 0; k < NB; k++) moist[j][k] = *p++;
        for (j = 0; j < NV; j++) for (k = 0; k < NB; k++)
            for (f = 0; f < NF; f++) ice_[j][k][f] = *p++;

        /* efr_force.c:64-69 */
        res_moist = resid * depth * MM_PER_M;
        dsmax = Dsmax / spd;

        /* efr_force.c:71-110, verbatim loop */
        for (frac = 1.0; frac >= 0.0; frac -= 0.01) {
            if (calc_bf(frac) < req) {
                frac = min(1, frac + 0.01);
                break;
            }
        }
        /* efr_force.c:112-125 */
        for (j = 0; j < NV; j++) for (k = 0; k < NB; k++) em[j][k] = 0.0;
        for (j = 0; j < nv_active; j++) {
            for (k = 0; k < NB; k++) {
                mo = moist[j][k];
                ice = 0.0;
                for (f = 0; f < NF; f++) ice += ice_[j][k][f] * ff[f];
                liq = mo - ice;
                em[j][k] = liq * frac + ice;
            }
        }
        /* efr_output.c:111-120 */
        for (j = 0; j < nv_active; j++)
            for (k = 0; k < NB; k++)
                cellm += em[j][k] * Cv[j] * AF[k];

        *q++ = frac;
        *q++ = cellm;
        *q++ = disch;
        for (j = 0; j < NV; j++) for (k = 0; k < NB; k++) *q++ = em[j][k];
        /* the full scan of the grid (the C loop stops at the first hit) */
        {
            double fr = 1.0;
            for (i = 0; i < NG; i++, fr -= 0.01) *q++ = calc_bf(fr);
        }
        fwrite(out, 8, NOUT, fo);
    }
    fclose(fi); fclose(fo);
    return 0;
}
