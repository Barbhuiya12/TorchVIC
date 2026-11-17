/* Oracle of VIC-WUR remote water use.
 *
 * Compiles the unchanged VIC-WUR 3.0 source plugins/wateruse/src/wu_remote.c (the
 * whole of wu_remote(iCell): reset, demand, availability, division, use,
 * account, hydrology incl. the deficit payback and the ring edit, balance
 * check) in one translation unit (stub headers in vicwur_shim_rm/ and
 * vicwur_shim/) and drives it as plugin_simulation.c:138 does: wu_remote(i)
 * for the cells of a given routing order, one after the other. Local use
 * (wu_run_local) is not part of it: its results are the input.
 * tests/test_vicwur_wu_remote.py writes the input and compares
 * torchvic.vicwur.wateruse_remote with the output.
 *
 * Input (little endian float64):
 *   ncell nk uh_length n_sub efr gwm dt norder                   (8)
 *   order[norder]
 *   per cell: area deficit efr_q discharge nrec rec[nk]          (5+nk)
 *             act[5] dgw[5] dsf[5] wgw[5] wsf[5] wdm[5]
 *             wtr0[5] cf[5] ret0[5] con0[5]                      (50)
 *             ring[K] (K = uh_length + n_sub - 1)
 * Output per cell:
 *   for sector 0..4: the 24 fields of wu_var (tests/test_vicwur_wu_remote.py
 *   FIELDS), 0 for an inactive sector                            (120)
 *   deficit discharge stream ring[K]                             (3+K)
 * Any log_err of the sources exits with status 1.
 */
#include <vic_driver_image.h>
#include <plugin.h>

#define min(a, b) (a < b) ? a : b              /* vic_def.h:91 */
#define max(a, b) (a > b) ? a : b              /* vic_def.h:92 */

typedef int MPI_Comm;                          /* only wu_remote_report_issue_counts */
#define MPI_UNSIGNED_LONG_LONG 0
#define MPI_SUM 0
#define check_mpi_status(A, M, ...) if (A != 0) {log_err(M, ## __VA_ARGS__); }
int      mpi_rank = 0;
MPI_Comm MPI_COMM_VIC = 0;
static int
MPI_Reduce(const void *s, void *r, int c, int t, int o, int root, MPI_Comm cm)
{
    memcpy(r, s, sizeof(unsigned long long));
    return 0;
}

domain_struct       local_domain;
global_param_struct global_param;
option_struct       options;

#include "wu_remote.c"

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
    double  h[8];
    size_t  n, nk, k, c, s, i, j, no;
    double *buf, *out;
    size_t *order;
    double *tmp;

    if (!fi || !fo || fread(h, 8, 8, fi) != 8) {
        return 2;
    }
    n = (size_t) h[0];
    nk = (size_t) h[1];
    plugin_options.UH_LENGTH = (short unsigned int) h[2];
    plugin_options.NWUTYPES = NS;
    plugin_global_param.rout_steps_per_day = (size_t) h[3];
    global_param.model_steps_per_day = 1;
    plugin_options.EFR = (h[4] != 0);
    options.GWM = (h[5] != 0);
    global_param.dt = h[6];
    no = (size_t) h[7];
    k = plugin_options.UH_LENGTH + plugin_global_param.rout_steps_per_day - 1;

    order = malloc(no * sizeof(size_t));
    tmp = malloc(no * sizeof(double));
    if (fread(tmp, 8, no, fi) != no) {
        return 2;
    }
    for (i = 0; i < no; i++) {
        order[i] = (size_t) tmp[i];
    }
    local_domain.ncells_active = n;
    local_domain.locations = calloc(n, sizeof(location_struct));
    rout_var = calloc(n, sizeof(rout_var_struct));
    efr_force = calloc(n, sizeof(efr_force_struct));
    wu_con = calloc(n, sizeof(wu_con_struct));
    wu_con_map = calloc(n, sizeof(wu_con_map_struct));
    wu_var = calloc(n, sizeof(wu_var_struct *));
    wu_force = calloc(n, sizeof(wu_force_struct *));
    for (c = 0; c < n; c++) {
        rout_var[c].dt_discharge = calloc(k, sizeof(double));
        wu_con[c].receiving = calloc(nk, sizeof(size_t));
        wu_con_map[c].sidx = calloc(NS, sizeof(int));
        wu_var[c] = calloc(NS, sizeof(wu_var_struct));
        wu_force[c] = calloc(NS, sizeof(wu_force_struct));
    }
    buf = malloc(sizeof(double) * (5 + nk + 50 + k));
    out = malloc(sizeof(double) * (NS * 24 + 3 + k));

    for (c = 0; c < n; c++) {
        double *p = buf;
        double  a[NS];

        if (fread(buf, 8, 5 + nk + 50 + k, fi) != 5 + nk + 50 + k) {
            return 3;
        }
        local_domain.locations[c].area = *p++;
        rout_var[c].nonrenew_deficit = *p++;
        efr_force[c].discharge = *p++;
        rout_var[c].discharge = *p++;
        wu_con[c].nreceiving = (size_t) *p++;
        for (j = 0; j < nk; j++) {
            wu_con[c].receiving[j] = (size_t) *p++;
        }
        for (s = 0; s < NS; s++) {
            wu_con_map[c].sidx[s] = (*p++ != 0) ? (int) s : NODATA_WU;
        }
        for (s = 0; s < NS; s++) wu_var[c][s].demand_gw = *p++;
        for (s = 0; s < NS; s++) wu_var[c][s].demand_surf = *p++;
        for (s = 0; s < NS; s++) wu_var[c][s].withdrawn_gw = *p++;
        for (s = 0; s < NS; s++) wu_var[c][s].withdrawn_surf = *p++;
        for (s = 0; s < NS; s++) wu_var[c][s].withdrawn_dam = *p++;
        for (s = 0; s < NS; s++) wu_var[c][s].withdrawn_tremote = *p++;
        for (s = 0; s < NS; s++) wu_force[c][s].consumption_frac = *p++;
        for (s = 0; s < NS; s++) wu_var[c][s].returned = *p++;
        for (s = 0; s < NS; s++) wu_var[c][s].consumed = *p++;
        for (i = 0; i < k; i++) rout_var[c].dt_discharge[i] = *p++;
        (void) a;
    }

    /* P2: wu_remote in the routing order (plugin_simulation.c:138) */
    for (i = 0; i < no; i++) {
        wu_remote(order[i]);
    }

    for (c = 0; c < n; c++) {
        double *q = out;

        for (s = 0; s < NS; s++) {
            wu_var_struct *w = &wu_var[c][s];
            if (wu_con_map[c].sidx[s] == NODATA_WU) {
                for (i = 0; i < 24; i++) *q++ = 0.;
                continue;
            }
            *q++ = w->available_gw; *q++ = w->available_surf;
            *q++ = w->available_dam; *q++ = w->available_tremote;
            *q++ = w->available_remote; *q++ = w->available_nonrenew;
            *q++ = w->demand_gw; *q++ = w->demand_surf;
            *q++ = w->demand_tremote; *q++ = w->demand_remote;
            *q++ = w->demand_nonrenew; *q++ = w->withdrawn_gw;
            *q++ = w->withdrawn_surf; *q++ = w->withdrawn_dam;
            *q++ = w->withdrawn_tremote; *q++ = w->withdrawn_remote;
            *q++ = w->withdrawn_nonrenew; *q++ = w->returned;
            *q++ = w->consumed; *q++ = w->available_remote_tmp;
            *q++ = w->demand_remote_tmp; *q++ = w->withdrawn_remote_tmp;
            *q++ = w->returned_remote_tmp; *q++ = w->consumed_remote_tmp;
        }
        *q++ = rout_var[c].nonrenew_deficit;
        *q++ = rout_var[c].discharge;
        *q++ = rout_var[c].stream;
        for (i = 0; i < k; i++) *q++ = rout_var[c].dt_discharge[i];
        fwrite(out, 8, q - out, fo);
    }
    fclose(fo);
    fclose(fi);
    return 0;
}
