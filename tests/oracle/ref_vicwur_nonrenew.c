/* Oracle of the VIC-WUR non-renewable withdrawal.
 *
 * Compiles the unchanged VIC-WUR 3.0 sources
 *   plugins/wateruse/src/wu_nonrenewable.c   (demand, availability, division,
 *                                             use, hydrology, balance check)
 *   plugins/routing/src/rout_run.c           (deficit payback from baseflow
 *                                             and runoff, :49-76)
 * in one translation unit (stub headers in vicwur_shim_nr/ and vicwur_shim/),
 * and drives them as plugin_simulation.c:97-162 does for one cell: rout_run in
 * P1, then wu_nonrenew in P3.  The sub-functions of wu_nonrenew are called one
 * by one, in the order of wu_nonrenew (wu_nonrenewable.c:378-406), so that the
 * cell sums are observable.
 * tests/test_vicwur_nonrenew.py writes the input and compares
 * torchvic.vicwur.nonrenew with the output.
 *
 * Input (little endian float64):
 *   ncell gwm nonrenew_runoff uh_length n_sub nren_lim dt     (7)
 *   per cell: area deficit0 baseflow runoff                    (4)
 *             act[5] dgw[5] dsf[5] wgw[5] wsf[5] wdm[5] wtr[5] cf[5]
 *             ret0[5] con0[5]                                  (50)
 *             ring[K] (K = uh_length + n_sub - 1) discharge0 stream0
 * Output per cell:
 *   deficit_after_rout_run runoff_m3s                          (2)
 *   demand_nr[5] avail_nr[5] withdrawn_nr[5] returned[5] consumed[5] (25)
 *   cell_demand cell_available cell_withdrawn cell_returned    (4)
 *   deficit_end discharge stream ring[K]                       (3+K)
 * Any log_err of the sources exits with status 1.
 */
#include <vic_driver_image.h>
#include <plugin.h>

#define min(a, b) (a < b) ? a : b              /* vic_def.h:91 */

domain_struct       local_domain;
global_param_struct global_param;
option_struct       options;
double           ***out_data;

#include "wu_nonrenewable.c"
#include "rout_run.c"

bool
assert_close_double(double x, double y, double rtol, double abs_tol)
{
    return fabs(x - y) <= abs_tol + rtol * fabs(y);
}

void
sprint_location(char *str, location_struct *loc)
{
    sprintf(str, "cell %zu", loc->global_idx);
}

#define NS 5

int
main(int argc, char **argv)
{
    FILE   *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    double  h[7];
    size_t  n, k, c, s, i;
    double *buf, *out;
    int     sidx[NS];
    wu_var_struct   *wv;
    wu_force_struct *wf;

    if (!fi || !fo || fread(h, 8, 7, fi) != 7) {
        return 2;
    }
    n = (size_t) h[0];
    options.GWM = (h[1] != 0);
    plugin_options.NONRENEW_RUNOFF = (h[2] != 0);
    plugin_options.UH_LENGTH = (short unsigned int) h[3];
    plugin_options.NWUTYPES = NS;
    plugin_global_param.rout_steps_per_day = (size_t) h[4];
    global_param.model_steps_per_day = 1;
    plugin_param.NREN_LIM = h[5];
    global_param.dt = h[6];
    k = plugin_options.UH_LENGTH + plugin_global_param.rout_steps_per_day - 1;

    local_domain.ncells_active = 1;
    local_domain.locations = calloc(1, sizeof(location_struct));
    rout_var = calloc(1, sizeof(rout_var_struct));
    rout_var[0].dt_discharge = calloc(k, sizeof(double));
    wu_con_map = calloc(1, sizeof(wu_con_map_struct));
    wu_con_map[0].sidx = sidx;
    wu_var = calloc(1, sizeof(wu_var_struct *));
    wu_var[0] = calloc(NS, sizeof(wu_var_struct));
    wu_force = calloc(1, sizeof(wu_force_struct *));
    wu_force[0] = calloc(NS, sizeof(wu_force_struct));
    out_data = calloc(1, sizeof(double **));
    out_data[0] = calloc(N_OUTVAR_TYPES, sizeof(double *));
    for (i = 0; i < N_OUTVAR_TYPES; i++) {
        out_data[0][i] = calloc(1, sizeof(double));
    }
    wv = wu_var[0];
    wf = wu_force[0];
    buf = malloc(sizeof(double) * (4 + 50 + k + 2));
    out = malloc(sizeof(double) * (2 + 25 + 4 + 3 + k));

    for (c = 0; c < n; c++) {
        double *p = buf, *q = out;
        double  dem_nr = 0., av_nr = 0.;

        if (fread(buf, 8, 4 + 50 + k + 2, fi) != 4 + 50 + k + 2) {
            return 3;
        }
        local_domain.locations[0].area = *p++;
        rout_var[0].nonrenew_deficit = *p++;
        out_data[0][OUT_BASEFLOW][0] = *p++;
        out_data[0][OUT_RUNOFF][0] = *p++;
        for (s = 0; s < NS; s++) {
            sidx[s] = (*p++ != 0) ? (int) s : NODATA_WU;
        }
        for (s = 0; s < NS; s++) wv[s].demand_gw = *p++;
        for (s = 0; s < NS; s++) wv[s].demand_surf = *p++;
        for (s = 0; s < NS; s++) wv[s].withdrawn_gw = *p++;
        for (s = 0; s < NS; s++) wv[s].withdrawn_surf = *p++;
        for (s = 0; s < NS; s++) wv[s].withdrawn_dam = *p++;
        for (s = 0; s < NS; s++) wv[s].withdrawn_tremote = *p++;
        for (s = 0; s < NS; s++) wf[s].consumption_frac = *p++;
        for (s = 0; s < NS; s++) wv[s].returned = *p++;
        for (s = 0; s < NS; s++) wv[s].consumed = *p++;
        for (s = 0; s < NS; s++) {      /* wu_run.c:67-69 */
            wv[s].available_nonrenew = 0.0;
            wv[s].demand_nonrenew = 0.0;
            wv[s].withdrawn_nonrenew = 0.0;
        }
        for (i = 0; i < k; i++) rout_var[0].dt_discharge[i] = *p++;
        rout_var[0].discharge = *p++;
        rout_var[0].stream = *p++;

        /* P1: rout_run.c */
        rout_run(0);
        *q++ = rout_var[0].nonrenew_deficit;
        *q++ = rout_var[0].runoff;

        /* P3: wu_nonrenew, wu_nonrenewable.c:359-407 */
        calculate_demand_nonrenew(0, &dem_nr);
        calculate_availability_nonrenew(0, &av_nr);
        {
            double a_cell = av_nr, d_cell = dem_nr, w_cell = 0., r_cell = 0.;
            calculate_division_nonrenew(0, av_nr, dem_nr);
            calculate_use_nonrenew(0, &w_cell, &r_cell);
            calculate_hydrology_nonrenew(0, w_cell, r_cell);
            check_water_use_balance_nonrenew(0, a_cell, d_cell, w_cell);
            for (s = 0; s < NS; s++) {
                *q++ = (sidx[s] == NODATA_WU) ? 0. : wv[s].demand_nonrenew;
            }
            for (s = 0; s < NS; s++) {
                *q++ = (sidx[s] == NODATA_WU) ? 0. : wv[s].available_nonrenew;
            }
            for (s = 0; s < NS; s++) {
                *q++ = (sidx[s] == NODATA_WU) ? 0. : wv[s].withdrawn_nonrenew;
            }
            for (s = 0; s < NS; s++) *q++ = wv[s].returned;
            for (s = 0; s < NS; s++) *q++ = wv[s].consumed;
            *q++ = d_cell; *q++ = a_cell; *q++ = w_cell; *q++ = r_cell;
        }
        *q++ = rout_var[0].nonrenew_deficit;
        *q++ = rout_var[0].discharge;
        *q++ = rout_var[0].stream;
        for (i = 0; i < k; i++) *q++ = rout_var[0].dt_discharge[i];
        fwrite(out, 8, q - out, fo);
    }
    fclose(fo);
    fclose(fi);
    return 0;
}
