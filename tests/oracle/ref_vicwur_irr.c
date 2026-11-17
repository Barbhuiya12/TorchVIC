/* Oracle for VIC-WUR irrigation per-tile kernels.
 *
 * Standalone single-cell copy of the arithmetic of VIC-WUR 3.0
 *   plugins/irrigation/src/irr_run.c:39-70      irr_update_step_vars
 *   plugins/irrigation/src/irr_run.c:107-221    irr_run_requirement
 *   plugins/irrigation/src/irr_run.c:260-372    irr_run_shortage
 *   plugins/irrigation/src/irr_use.c:62-105     irr_set_demand
 *   plugins/irrigation/src/irr_use.c:217-269    irr_apply
 *   plugins/irrigation/src/irr_use.c:274-324    irr_leftover
 *   plugins/irrigation/src/irr_use.c:329-380    irr_potential
 *   plugins/irrigation/src/irr_use.c:386-526    irr_wateruse (without log_err)
 *   plugins/irrigation/src/irr_use.c:531-608    irr_wofost (applying part)
 *   plugins/irrigation/src/irr_use.c:650-687    irr_get_withdrawn
 * with the C structs replaced by fixed-size arrays and the C `min`/`max`
 * macros kept (vic_def.h:91-92).  Reads records of doubles from argv[1]
 * (first double = number of records), writes records of doubles to argv[2].
 * Layout: see tests/test_vicwur_irrigation.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define min(a, b) (a < b) ? a : b
#define max(a, b) (a > b) ? a : b
#define NI 3
#define NB 3
#define NL 3
#define NF 2
#define PADDY_FLOOD_HEIGHT 0
#define NIN (8 + NI*3 + NI*NB*2 + NI*2 + NB + 1 + NF + NI*NB*NL*4 + 2*NL + \
             NI*NB*NL*NF + NI*NB*2 + 1)
#define NOUT ((13 + NL)*NI*NB + 4)

typedef struct {
    double moist[NL], ice[NL][NF], Wcr[NL], Ksat0, ws[NL], ws_cell;
    double root[NL];
} cell_t;
typedef struct {
    double requirement, received, applied, leftover, shortage, prev_Cv;
    int flag_req, offset;
} irr_t;

static int    EFFICIENT, POTENTIAL, SHARE, WATERUSE, WOFOST_ON;
static double Wfc[NL], Wpwp[NL], max_moist0, frost[NF], AF[NB];
static double Cv[NI], gw[NI], effc[NI];
static int    paddy[NI], offmax[NI];
static cell_t cell[NI][NB];
static irr_t  var[NI][NB];

static void irr_apply(int i, int b, double received, double *applied,
                      double *leftover)
{
    double applied_tmp, leftover_tmp, max_added;
    cell_t *cv = &cell[i][b];

    if (received <= 0.) return;
    if (EFFICIENT && !paddy[i]) max_added = Wfc[0] - cv->moist[0];
    else max_added = max_moist0 - cv->moist[0];
    max_added = max(0., max_added);
    if (received < max_added) applied_tmp = received;
    else applied_tmp = max_added;
    leftover_tmp = received - applied_tmp;
    cv->moist[0] += applied_tmp;
    (*applied) += applied_tmp;
    (*leftover) += leftover_tmp;
}

int main(int argc, char **argv)
{
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    double n;
    int c;

    if (!fi || !fo || fread(&n, 8, 1, fi) != 1) return 1;
    for (c = 0; c < (int) n; c++) {
        double in[NIN], *p = in, out[NOUT], *q = out;
        int new_day, i, b, k, l;
        double ksat0, expt, available, wofost[NI][NB], ws_in[NI][NB][NL];
        double prev_left[NI][NB];
        double D = 0, G = 0, E = 0, D0 = 0;
        double ksat_pre[NI][NB], ksat_req[NI][NB];

        if (fread(in, 8, NIN, fi) != NIN) return 2;
        EFFICIENT = (int) *p++; POTENTIAL = (int) *p++; SHARE = (int) *p++;
        WATERUSE = (int) *p++; WOFOST_ON = (int) *p++;
        new_day = (int) *p++; ksat0 = *p++; expt = *p++;
        for (i = 0; i < NI; i++) paddy[i] = (int) *p++;
        for (i = 0; i < NI; i++) Cv[i] = *p++;
        for (i = 0; i < NI; i++) offmax[i] = (int) *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) var[i][b].prev_Cv = *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) var[i][b].offset = (int) *p++;
        for (i = 0; i < NI; i++) gw[i] = *p++;
        for (i = 0; i < NI; i++) effc[i] = *p++;
        for (b = 0; b < NB; b++) AF[b] = *p++;
        max_moist0 = *p++;
        for (l = 0; l < NF; l++) frost[l] = *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) for (k = 0; k < NL; k++) cell[i][b].moist[k] = *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) for (k = 0; k < NL; k++) cell[i][b].root[k] = *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) for (k = 0; k < NL; k++) cell[i][b].Wcr[k] = *p++;
        for (k = 0; k < NL; k++) Wfc[k] = *p++;
        for (k = 0; k < NL; k++) Wpwp[k] = *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) for (k = 0; k < NL; k++) for (l = 0; l < NF; l++) cell[i][b].ice[k][l] = *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) var[i][b].leftover = *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) wofost[i][b] = *p++;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) for (k = 0; k < NL; k++) ws_in[i][b][k] = *p++;
        available = *p++;

        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) {
            prev_left[i][b] = var[i][b].leftover;
            for (k = 0; k < NL; k++) cell[i][b].ws[k] = ws_in[i][b][k];
            cell[i][b].ws_cell = 0;
        }

        /* irr_update_step_vars, irr_run.c:56-69 */
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) {
            cell[i][b].Ksat0 = ksat0;
            if (paddy[i]) cell[i][b].Ksat0 = pow(ksat0, expt);
            ksat_pre[i][b] = cell[i][b].Ksat0;
        }

        /* irr_run_requirement, irr_run.c:107-221 */
        for (i = 0; i < NI; i++) {
            double veg_fract = Cv[i];
            for (b = 0; b < NB; b++) {
                var[i][b].requirement = 0.0;
                var[i][b].flag_req = 0;
                cell[i][b].Ksat0 = ksat0;
            }
            if (new_day) {
                for (b = 0; b < NB; b++) {
                    if (veg_fract != var[i][b].prev_Cv) var[i][b].offset = 0;
                    if (var[i][b].offset < offmax[i]) var[i][b].offset++;
                    var[i][b].prev_Cv = veg_fract;
                }
            }
            if (veg_fract > 0) {
                for (b = 0; b < NB; b++) {
                    irr_t *cv = &var[i][b];
                    cell_t *cc = &cell[i][b];
                    double area_fract = AF[b];
                    if (area_fract > 0) {
                        double moist[NL], tm = 0, tw = 0, tf = 0;
                        if (paddy[i]) cc->Ksat0 = pow(ksat0, expt);
                        for (k = 0; k < NL; k++) moist[k] = 0.0;
                        for (k = 0; k < NL; k++) {
                            for (l = 0; l < NF; l++)
                                moist[k] += (cc->moist[k] - cc->ice[k][l]) * frost[l];
                            if (cc->root[k] > 0.) {
                                tm += moist[k]; tw += cc->Wcr[k]; tf += Wfc[k];
                            }
                        }
                        if (paddy[i])
                            cv->requirement = (max_moist0 + PADDY_FLOOD_HEIGHT) - (moist[0] + cv->leftover);
                        else
                            cv->requirement = tf - (tm + cv->leftover);
                        if (cv->offset >= offmax[i]) {
                            if (paddy[i] && max_moist0 - (moist[0] + cv->leftover) > 0 && cv->requirement > 0)
                                cv->flag_req = 1;
                            else if (tw - (tm + cv->leftover) > 0 && cv->requirement > 0)
                                cv->flag_req = 1;
                        }
                    }
                }
            }
            for (b = 0; b < NB; b++) ksat_req[i][b] = cell[i][b].Ksat0;
        }

        /* irr_set_demand, irr_use.c:62-105 */
        for (i = 0; i < NI; i++) if (Cv[i] > 0) for (b = 0; b < NB; b++) {
            if (AF[b] > 0 && var[i][b].flag_req) {
                double dc = var[i][b].requirement * Cv[i] * AF[b];
                D0 += dc; D += dc; G += dc * gw[i]; E += dc * effc[i];
            }
        }
        if (D > 0) { G = G / D; E = E / D; D *= (1 / E); }
        else { E = 0; G = 0; }
        D0 = D0; /* D0 is the crop demand before D *= 1/E */
        {
            double d0 = 0;
            for (i = 0; i < NI; i++) if (Cv[i] > 0) for (b = 0; b < NB; b++)
                if (AF[b] > 0 && var[i][b].flag_req)
                    d0 += var[i][b].requirement * Cv[i] * AF[b];
            D0 = d0;
        }

        /* irr_get_withdrawn, irr_use.c:650-687 */
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) {
            var[i][b].received = 0.0; var[i][b].applied = 0.0;
        }
        for (i = 0; i < NI; i++) if (Cv[i] > 0) for (b = 0; b < NB; b++) {   /* irr_leftover */
            if (AF[b] > 0 && var[i][b].leftover > 0) {
                double lt = var[i][b].leftover;
                var[i][b].leftover = 0.0;
                irr_apply(i, b, lt, &var[i][b].applied, &var[i][b].leftover);
            }
        }
        if (POTENTIAL) {
            for (i = 0; i < NI; i++) if (Cv[i] > 0) for (b = 0; b < NB; b++) {
                if (AF[b] > 0) {
                    double rt = 0.0;
                    if (var[i][b].flag_req) {
                        rt = var[i][b].requirement;
                        irr_apply(i, b, rt, &var[i][b].applied, &var[i][b].leftover);
                        var[i][b].received += rt;
                    }
                }
            }
        }
        else {
            if (WATERUSE) {                                   /* irr_wateruse */
                double demand = 0.0, avail_frac;
                for (i = 0; i < NI; i++) if (Cv[i] > 0) for (b = 0; b < NB; b++)
                    if (AF[b] > 0 && var[i][b].flag_req)
                        demand += var[i][b].requirement * Cv[i] * AF[b];
                if (available > 0) avail_frac = available / demand;
                else avail_frac = 0;
                if (available > 0) {
                    for (i = 0; i < NI; i++) if (Cv[i] > 0) for (b = 0; b < NB; b++) {
                        if (AF[b] > 0) {
                            double rt = 0.0;
                            if (var[i][b].flag_req && avail_frac > 0) {
                                rt = var[i][b].requirement * avail_frac;
                                irr_apply(i, b, rt, &var[i][b].applied, &var[i][b].leftover);
                                var[i][b].received += rt;
                            }
                        }
                    }
                }
            }
            if (WOFOST_ON) {                                  /* irr_wofost */
                for (i = 0; i < NI; i++) if (Cv[i] > 0) for (b = 0; b < NB; b++) {
                    if (AF[b] > 0) {
                        irr_apply(i, b, wofost[i][b], &var[i][b].applied, &var[i][b].leftover);
                        var[i][b].received += wofost[i][b];
                    }
                }
            }
        }

        /* irr_run_shortage, irr_run.c:260-372 */
        for (i = 0; i < NI; i++) {
            for (b = 0; b < NB; b++) var[i][b].shortage = 0.0;
            if (Cv[i] > 0) for (b = 0; b < NB; b++) {
                irr_t *cv = &var[i][b];
                cell_t *cc = &cell[i][b];
                if (AF[b] > 0) {
                    double moist[NL], tm = 0, tw = 0, lm = 0, lw = 0, lr = 0;
                    for (k = 0; k < NL; k++) moist[k] = 0.0;
                    for (k = 0; k < NL; k++) {
                        for (l = 0; l < NF; l++)
                            moist[k] += (cc->moist[k] - cc->ice[k][l]) * frost[l];
                        if (k < NL - 1) {
                            if (cc->root[k] > 0.) { tm += moist[k]; tw += cc->Wcr[k]; }
                        }
                        else { lm += moist[k]; lw += cc->Wcr[k]; lr += cc->root[k]; }
                    }
                    if (tm < tw && cv->flag_req) cv->shortage = tw - tm;
                    if (cv->shortage < 0.0) cv->shortage = 0.0;
                    if (SHARE &&
                        ((tm >= tw && lm >= lw && tw > 0.) ||
                         (tm >= tw && (1 - lr) >= 0.5) ||
                         (lm >= lw && lr >= 0.5))) {
                        for (k = 0; k < NL; k++) cc->ws[k] = 1.;
                    }
                    else {
                        for (k = 0; k < NL; k++) {
                            if (moist[k] >= cc->Wcr[k]) cc->ws[k] = 1.0;
                            else if (moist[k] >= Wpwp[k] && moist[k] < cc->Wcr[k])
                                cc->ws[k] = (moist[k] - Wpwp[k]) / (cc->Wcr[k] - Wpwp[k]);
                            else cc->ws[k] = 0.0;
                        }
                    }
                    cc->ws_cell = 0;
                    for (k = 0; k < NL; k++) cc->ws_cell += cc->ws[k] * cc->root[k];
                }
            }
        }

        /* output */
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = ksat_pre[i][b];
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = ksat_req[i][b];
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = var[i][b].offset;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = var[i][b].prev_Cv;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = var[i][b].requirement;
        *q++ = D; *q++ = G; *q++ = E; *q++ = D0;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = var[i][b].flag_req;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = cell[i][b].moist[0];
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = var[i][b].leftover;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = var[i][b].received;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = var[i][b].applied;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = var[i][b].shortage;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) for (k = 0; k < NL; k++) *q++ = cell[i][b].ws[k];
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = cell[i][b].ws_cell;
        for (i = 0; i < NI; i++) for (b = 0; b < NB; b++) *q++ = prev_left[i][b];
        fwrite(out, 8, q - out, fo);
        if (q - out != NOUT) { fprintf(stderr, "NOUT %d vs %d\n", (int)(q - out), NOUT); return 3; }
    }
    fclose(fo);
    return 0;
}
