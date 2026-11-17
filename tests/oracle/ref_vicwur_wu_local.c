/* Oracle for VIC-WUR water use core + local.
 *
 * Standalone single-cell copy of the arithmetic of VIC-WUR 3.0
 *   plugins/wateruse/src/wu_run.c:77-111           calculate_demand
 *   plugins/wateruse/src/wu_run_local.c:33-54      calculate_demand_local
 *   plugins/wateruse/src/wu_run_local.c:59-171     calculate_availability_local
 *   plugins/wateruse/src/wu_run_local.c:176-249    calculate_division_local
 *   plugins/wateruse/src/wu_run_local.c:254-346    calculate_use_local
 *   plugins/wateruse/src/wu_run_local.c:351-563    calculate_hydrology_local
 *                                                  (pumping, payback, dam;
 *                                                   the ring edit is 1.1)
 *   plugins/wateruse/src/wu_nonrenewable.c:54-65   unmet demand
 * with the C structs replaced by arrays of fixed size and the C `min` macro
 * kept (vic_def.h:91).  Reads records of doubles from argv[1], writes
 * records of doubles to argv[2].  Layout: see tests/test_vicwur_wateruse.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define min(a, b) (a < b) ? a : b
#define NS 5
#define NV 3
#define NB 3
#define NF 2
#define NK 8
#define ND 3
#define MM_PER_M 1000.
#define M3_PER_HM3 1e6
#define NIN (2 + 5 + 5*5 + 2*NV*NB + NV*NB*NF + NF + 1 + NB + NV + NK + 4 + 3*ND + 1)
#define NOUT (5*2 + 3 + 3 + NV*NB + ND + 5*8 + 4 + NV*NB + ND + 2 + 5)

int main(int argc, char **argv)
{
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    double n;
    int c, i, v, b, f, d;

    if (!fi || !fo || fread(&n, 8, 1, fi) != 1) return 1;
    for (c = 0; c < (int) n; c++) {
        double in[NIN], *p = in, out[NOUT], *q = out;
        int efr, gwm, act[NS];
        double dem[NS], gfr[NS], cf[NS], ret0[NS], con0[NS];
        double moist[NV][NB], efrm[NV][NB], ice[NV][NB][NF], ff[NF];
        double resid_mm, AF[NB], Cv[NV], dtq[NK], dt, area, efr_q, disch;
        double stor[ND], dact[ND], dloc[ND], deficit;
        double dgw[NS], dsf[NS], avgw[NV][NB], avdam[ND];
        double A_gw = 0, A_sf = 0, A_dam = 0, Dgw = 0, Dsf = 0;
        double avail_gw[NS], avail_surf[NS], avail_dam[NS];
        double wgw[NS], wsf[NS], wdm[NS], ret[NS], con[NS];
        double Wgw = 0, Wsf = 0, Wdm = 0, R = 0, frac, ice_t;
        double tgw, tsf, tdm, tot, nr[NS];

        if (fread(in, 8, NIN, fi) != NIN) return 2;
        efr = (int) *p++; gwm = (int) *p++;
        for (i = 0; i < NS; i++) act[i] = (int) *p++;
        for (i = 0; i < NS; i++) dem[i] = *p++;
        for (i = 0; i < NS; i++) gfr[i] = *p++;
        for (i = 0; i < NS; i++) cf[i] = *p++;
        for (i = 0; i < NS; i++) ret0[i] = *p++;
        for (i = 0; i < NS; i++) con0[i] = *p++;
        for (v = 0; v < NV; v++) for (b = 0; b < NB; b++) moist[v][b] = *p++;
        for (v = 0; v < NV; v++) for (b = 0; b < NB; b++) efrm[v][b] = *p++;
        for (v = 0; v < NV; v++) for (b = 0; b < NB; b++) for (f = 0; f < NF; f++) ice[v][b][f] = *p++;
        for (f = 0; f < NF; f++) ff[f] = *p++;
        resid_mm = *p++;
        for (b = 0; b < NB; b++) AF[b] = *p++;
        for (v = 0; v < NV; v++) Cv[v] = *p++;
        for (i = 0; i < NK; i++) dtq[i] = *p++;
        dt = *p++; area = *p++; efr_q = *p++; disch = *p++;
        for (d = 0; d < ND; d++) stor[d] = *p++;
        for (d = 0; d < ND; d++) dact[d] = *p++;
        for (d = 0; d < ND; d++) dloc[d] = *p++;
        deficit = *p++;

        /* wu_run.c:88-109 */
        for (i = 0; i < NS; i++) {
            dgw[i] = dsf[i] = 0;
            avail_gw[i] = avail_surf[i] = avail_dam[i] = 0;
            wgw[i] = wsf[i] = wdm[i] = 0; ret[i] = ret0[i]; con[i] = con0[i];
            if (!act[i]) continue;
            if (gwm) { dgw[i] = 0.0; dsf[i] = dem[i]; }
            else {
                dgw[i] = dem[i] * gfr[i];
                dsf[i] = dem[i] * (1 - gfr[i]);
            }
        }
        /* wu_run_local.c:45-53 */
        for (i = 0; i < NS; i++) {
            if (!act[i]) continue;
            Dgw += dgw[i]; Dsf += dsf[i];
        }
        /* :95-136 */
        for (v = 0; v < NV; v++) for (b = 0; b < NB; b++) avgw[v][b] = 0.0;
        if (gwm == 0) {
            for (v = 0; v < NV; v++) {
                if (Cv[v] <= 0.0) continue;
                for (b = 0; b < NB; b++) {
                    if (AF[b] <= 0.0) continue;
                    avgw[v][b] = moist[v][b];
                    if (efr) avgw[v][b] -= efrm[v][b];
                    avgw[v][b] -= resid_mm;
                    ice_t = 0;
                    for (f = 0; f < NF; f++) ice_t += ice[v][b][f] * ff[f];
                    avgw[v][b] -= ice_t;
                    avgw[v][b] *= AF[b] * Cv[v];
                    if (avgw[v][b] < 0) avgw[v][b] = 0;
                    A_gw += avgw[v][b];
                }
            }
        }
        /* :140-153 */
        for (i = 0; i < NK; i++) A_sf += dtq[i] * dt / area * MM_PER_M;
        if (efr) {
            frac = min(efr_q / disch, 1.0);
            A_sf *= (1 - frac);
        }
        if (A_sf < 0) A_sf = 0;
        /* :156-170 (DAMS on) */
        for (d = 0; d < ND; d++) {
            avdam[d] = 0;
            if (dact[d] != 0 && dloc[d] != 0) {
                avdam[d] = stor[d] * M3_PER_HM3 / area * MM_PER_M;
                A_dam += avdam[d];
            }
        }
        if (A_dam < 0) A_dam = 0;
        /* :196-248 */
        {
            double gt = A_gw, st = A_sf, dtm = A_dam, stot;
            for (i = 0; i < NS; i++) {
                if (!act[i]) continue;
                if (Dgw > 0) {
                    if (dgw[i] >= gt) { avail_gw[i] = gt; gt = 0.0; }
                    else { avail_gw[i] = dgw[i]; gt -= dgw[i]; }
                }
                stot = st + dtm;
                if (Dsf > 0) {
                    if (dsf[i] >= st + dtm) {
                        avail_surf[i] = st; avail_dam[i] = dtm;
                        st = 0.0; dtm = 0.0;
                    }
                    else {
                        avail_surf[i] = dsf[i] * (st / stot);
                        avail_dam[i] = dsf[i] * (dtm / stot);
                        st -= dsf[i] * (st / stot);
                        dtm -= dsf[i] * (dtm / stot);
                    }
                }
            }
        }
        /* :271-345 */
        for (i = 0; i < NS; i++) {
            if (!act[i]) continue;
            if (avail_gw[i] > 0) {
                frac = dgw[i] / avail_gw[i];
                frac = min(frac, 1);
                wgw[i] = avail_gw[i] * frac;
            }
            else wgw[i] = 0;
            ret[i] += wgw[i] * (1 - cf[i]);
            con[i] += wgw[i] * cf[i];
            if (avail_surf[i] + avail_dam[i] > 0) {
                frac = dsf[i] * (avail_surf[i] / (avail_surf[i] + avail_dam[i]));
                frac = frac / avail_surf[i];
                frac = min(frac, 1);
                wsf[i] = avail_surf[i] * frac;
                frac = dsf[i] * (avail_dam[i] / (avail_surf[i] + avail_dam[i]));
                frac = frac / avail_dam[i];
                frac = min(frac, 1);
                wdm[i] = avail_dam[i] * frac;
            }
            else { wsf[i] = 0; wdm[i] = 0; }
            ret[i] += wsf[i] * (1 - cf[i]);
            ret[i] += wdm[i] * (1 - cf[i]);
            con[i] += wsf[i] * cf[i];
            con[i] += wdm[i] * cf[i];
            Wgw += wgw[i]; Wsf += wsf[i]; Wdm += wdm[i]; R += ret[i];
        }
        /* unmet (wu_nonrenewable.c:55-64) */
        for (i = 0; i < NS; i++) {
            nr[i] = 0;
            if (!act[i]) continue;
            nr[i] = (dsf[i] + dgw[i]) - (wsf[i] + wgw[i] + wdm[i] + 0.0);
            if (nr[i] < 0.) nr[i] = 0.;
        }
        /* :398-436 pumping */
        if (Wgw > 0. && gwm == 0) {
            for (v = 0; v < NV; v++) {
                if (Cv[v] <= 0.0) continue;
                for (b = 0; b < NB; b++) {
                    if (AF[b] <= 0.0) continue;
                    moist[v][b] -= Wgw / (AF[b] * Cv[v]) * avgw[v][b] / A_gw;
                    if (moist[v][b] < 0) moist[v][b] = 0.;
                }
            }
        }
        /* :439-469 payback */
        if (R > 0. && gwm == 0) {
            for (i = 0; i < NS; i++) {
                double avn, rn;
                if (!act[i]) continue;
                avn = deficit;
                rn = ret[i];
                rn = min(avn, rn);
                if (avn > 0) {
                    deficit -= rn;
                    if (deficit < 0) deficit = 0;
                }
                R -= rn;
                if (R < 0) R = 0;
            }
        }
        /* :544-561 dam */
        if (Wdm > 0.) {
            for (d = 0; d < ND; d++) {
                if (dact[d] != 0 && dloc[d] != 0) {
                    stor[d] -= Wdm * (avdam[d] / A_dam) / MM_PER_M * area / M3_PER_HM3;
                    if (stor[d] < 0) stor[d] = 0.;
                }
            }
        }
        (void) tgw; (void) tsf; (void) tdm; (void) tot;
        /* output */
        for (i = 0; i < NS; i++) *q++ = dgw[i];
        for (i = 0; i < NS; i++) *q++ = dsf[i];
        *q++ = A_gw; *q++ = A_sf; *q++ = A_dam;
        *q++ = Dgw; *q++ = Dsf; *q++ = 0;
        for (v = 0; v < NV; v++) for (b = 0; b < NB; b++) *q++ = avgw[v][b];
        for (d = 0; d < ND; d++) *q++ = avdam[d];
        for (i = 0; i < NS; i++) *q++ = avail_gw[i];
        for (i = 0; i < NS; i++) *q++ = avail_surf[i];
        for (i = 0; i < NS; i++) *q++ = avail_dam[i];
        for (i = 0; i < NS; i++) *q++ = wgw[i];
        for (i = 0; i < NS; i++) *q++ = wsf[i];
        for (i = 0; i < NS; i++) *q++ = wdm[i];
        for (i = 0; i < NS; i++) *q++ = ret[i];
        for (i = 0; i < NS; i++) *q++ = con[i];
        *q++ = Wgw; *q++ = Wsf; *q++ = Wdm; *q++ = R;
        for (v = 0; v < NV; v++) for (b = 0; b < NB; b++) *q++ = moist[v][b];
        for (d = 0; d < ND; d++) *q++ = stor[d];
        *q++ = deficit; *q++ = 0;
        for (i = 0; i < NS; i++) *q++ = nr[i];
        fwrite(out, 8, q - out, fo);
        if (q - out != NOUT) { fprintf(stderr, "NOUT %ld != %d\n", (long)(q - out), NOUT); return 3; }
    }
    fclose(fo);
    return 0;
}
