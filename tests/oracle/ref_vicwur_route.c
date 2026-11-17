/* Oracle of the VIC-WUR routing cascade.
 *
 * Compiles the unchanged VIC-WUR 3.0 sources rout_init.c, rout_run.c,
 * rout_basin_run.c, convolute.c, cshift.c and wu_init.c in one translation
 * unit (stub headers in vicwur_shim/), and drives them as the image driver
 * does: rout_init (UH renormalisation, downstream, upstream, BASIN order),
 * optionally wu_init (REMOTE_WITH order override and check), then per model
 * step rout_run for every cell and rout_basin_run in routing_order
 * (plugin_simulation.c:97-138). tests/test_vicwur_routing.py writes the
 * input and compares torchvic.model.routing_cascade.CascadeRouter with the output.
 *
 * Input (little endian, all int64 or float64):
 *   N L nsub nt force_routing nonrenew_runoff gwm remote nrecv_max
 *   dt, area[N]
 *   id[N] downstream[N]                  (downstream_id, downstream fields)
 *   uh_inflow[L][N] uh_runoff[L][N]       (as in the file, not normalised)
 *   if remote: receiving_id[N] nreceiving[N] receiving[nrecv_max][N]
 *   ring0[N][R] deficit0[N]               (R = L + nsub - 1)
 *   per step: out_runoff[N] out_baseflow[N] force_discharge[N]
 * Output:
 *   routing_order[N] nupstream[N] upstream[N][8] (-1 padded)
 *   uh_inflow[N][L] uh_runoff[N][L]       (normalised)
 *   per step: runoff[N] inflow[N] discharge[N] stream[N] deficit[N]
 *             ring[N][R]
 * Any log_err of the sources exits with status 1.
 */
#include <vic_driver_image.h>
#include <plugin.h>

domain_struct              global_domain;
domain_struct              local_domain;
global_param_struct        global_param;
option_struct              options;
plugin_option_struct       plugin_options;
plugin_global_param_struct plugin_global_param;
plugin_filenames_struct    plugin_filenames;
wu_con_struct             *wu_con;
double                  ***out_data;
int                        mpi_rank = 0;

static int64_t            *f_id, *f_down, *f_recv_id, *f_recv;
static double             *f_uhi, *f_uhr;

#include "convolute.c"
#include "cshift.c"
#include "rout_init.c"
#include "rout_run.c"
#include "rout_basin_run.c"
#include "wu_init.c"

bool
assert_close_double(double x, double y, double rtol, double abs_tol)
{
    return fabs(x - y) <= abs_tol + rtol * fabs(y);   /* comparisons.c:48-57 */
}

void
sprint_location(char *str, location_struct *loc)
{
    sprintf(str, "cell %zu", loc->global_idx);
}

int
nc_open(const char *path, int mode, int *ncidp)
{
    (void) path; (void) mode; *ncidp = 0; return 0;
}

int
nc_close(int ncid)
{
    (void) ncid; return 0;
}

static void
field_int(char *name, size_t *start, int *out)
{
    size_t  n = local_domain.ncells_active, i;
    int64_t *src = NULL;

    if (strcmp(name, "downstream_id") == 0) {
        src = f_id;
    }
    else if (strcmp(name, "downstream") == 0) {
        src = f_down;
    }
    else if (strcmp(name, "receiving_id") == 0) {
        src = f_recv_id;
    }
    else if (strcmp(name, "receiving") == 0) {
        src = f_recv + start[0] * n;
    }
    if (src == NULL) {
        fprintf(stderr, "no field %s\n", name); exit(2);
    }
    for (i = 0; i < n; i++) {
        out[i] = (int) src[i];
    }
}

void
get_scatter_nc_field_int(nc_file_struct *f, char *name, size_t *start,
                         size_t *count, int *out)
{
    (void) f; (void) count; field_int(name, start, out);
}

void
get_active_nc_field_int(nc_file_struct *f, char *name, size_t *start,
                        size_t *count, int *out)
{
    (void) f; (void) count; field_int(name, start, out);
}

void
get_scatter_nc_field_double(nc_file_struct *f, char *name, size_t *start,
                            size_t *count, double *out)
{
    size_t  n = local_domain.ncells_active;
    double *src = strcmp(name, "uh_inflow") == 0 ? f_uhi : f_uhr;

    (void) f; (void) count;
    memcpy(out, src + start[0] * n, n * sizeof(double));
}

/* one MPI rank: scatter and gather are copies */
void
scatter_int(int *g, int *l)
{
    memcpy(l, g, local_domain.ncells_active * sizeof(int));
}

void
scatter_size_t(size_t *g, size_t *l)
{
    memcpy(l, g, local_domain.ncells_active * sizeof(size_t));
}

void
gather_size_t(size_t *g, size_t *l)
{
    memcpy(g, l, local_domain.ncells_active * sizeof(size_t));
}

void
scatter_size_t_2d(size_t **g, size_t **l, int m)
{
    size_t i;
    for (i = 0; i < local_domain.ncells_active; i++) {
        memcpy(l[i], g[i], m * sizeof(size_t));
    }
}

void
gather_size_t_2d(size_t **g, size_t **l, int m)
{
    size_t i;
    for (i = 0; i < local_domain.ncells_active; i++) {
        memcpy(g[i], l[i], m * sizeof(size_t));
    }
}

static FILE *fin, *fout;

static void
rd(void *p, size_t sz, size_t n)
{
    if (fread(p, sz, n, fin) != n) {
        fprintf(stderr, "ref_vicwur_route: short input\n"); exit(2);
    }
}

static int64_t
rdi(void)
{
    int64_t v; rd(&v, 8, 1); return v;
}

static void
wr(const void *p, size_t sz, size_t n)
{
    fwrite(p, sz, n, fout);
}

int
main(int argc, char **argv)
{
    size_t  N, L, nsub, nt, R, nrecv_max, i, j, t;
    int     remote;
    double *rbuf;
    int64_t *ibuf;

    if (argc < 3) {
        fprintf(stderr, "usage: ref_vicwur_route in out\n"); return 2;
    }
    fin = fopen(argv[1], "rb");
    fout = fopen(argv[2], "wb");
    if (!fin || !fout) {
        fprintf(stderr, "ref_vicwur_route: cannot open files\n"); return 2;
    }
    N = rdi(); L = rdi(); nsub = rdi(); nt = rdi();
    plugin_options.FORCE_ROUTING = rdi();
    plugin_options.NONRENEW_RUNOFF = rdi();
    options.GWM = rdi();
    remote = (int) rdi();
    nrecv_max = rdi();
    R = L + nsub - 1;
    rd(&global_param.dt, 8, 1);

    local_domain.ncells_active = global_domain.ncells_active = N;
    local_domain.n_ny = global_domain.n_ny = 1;
    local_domain.n_nx = global_domain.n_nx = N;
    local_domain.locations = calloc(N, sizeof(location_struct));
    global_domain.locations = local_domain.locations;
    for (i = 0; i < N; i++) {
        rd(&local_domain.locations[i].area, 8, 1);
        local_domain.locations[i].global_idx = i;
    }
    global_param.model_steps_per_day = 1;
    plugin_global_param.rout_steps_per_day = nsub;
    plugin_options.UH_LENGTH = (short unsigned int) L;
    plugin_options.DECOMPOSITION = BASIN_DECOMPOSITION;
    plugin_options.REMOTE_WITH = remote;
    plugin_options.NWURECEIVING = nrecv_max;

    f_id = malloc(N * 8); f_down = malloc(N * 8);
    rd(f_id, 8, N); rd(f_down, 8, N);
    f_uhi = malloc(L * N * 8); f_uhr = malloc(L * N * 8);
    rd(f_uhi, 8, L * N); rd(f_uhr, 8, L * N);

    /* rout_alloc + rout_initialize_local_structures (rout_alloc_free.c) */
    routing_order = malloc(N * sizeof(size_t));
    rout_con = calloc(N, sizeof(rout_con_struct));
    rout_var = calloc(N, sizeof(rout_var_struct));
    rout_force = calloc(N, sizeof(rout_force_struct));
    for (i = 0; i < N; i++) {
        rout_con[i].inflow_uh = malloc(L * sizeof(double));
        rout_con[i].runoff_uh = malloc(L * sizeof(double));
        rout_var[i].dt_discharge = calloc(R, sizeof(double));
    }
    out_data = malloc(N * sizeof(double **));
    for (i = 0; i < N; i++) {
        out_data[i] = malloc(N_OUTVAR_TYPES * sizeof(double *));
        for (j = 0; j < N_OUTVAR_TYPES; j++) {
            out_data[i][j] = calloc(1, sizeof(double));
        }
    }

    rout_init();

    if (remote) {
        f_recv_id = malloc(N * 8);
        f_recv = malloc((nrecv_max ? nrecv_max : 1) * N * 8);
        rd(f_recv_id, 8, N);
        wu_con = calloc(N, sizeof(wu_con_struct));
        for (i = 0; i < N; i++) {      /* wu_alloc_free.c:92-96 */
            wu_con[i].nreceiving = (size_t) rdi();
            wu_con[i].receiving = malloc((wu_con[i].nreceiving + 1) *
                                         sizeof(size_t));
        }
        rd(f_recv, 8, nrecv_max * N);
        wu_init();
    }

    for (i = 0; i < N; i++) {
        rd(rout_var[i].dt_discharge, 8, R);
    }
    for (i = 0; i < N; i++) {
        rd(&rout_var[i].nonrenew_deficit, 8, 1);
    }
    /* rout_compute_derived_state_vars (rout_restore.c:194-205) */
    for (i = 0; i < N; i++) {
        rout_var[i].discharge = 0.0;
        rout_var[i].stream = 0.0;
        for (j = 0; j < R; j++) {
            if (j < nsub) {
                rout_var[i].discharge += rout_var[i].dt_discharge[j];
            }
            else {
                rout_var[i].stream += rout_var[i].dt_discharge[j];
            }
        }
    }

    ibuf = malloc(N * 8 * 8);
    for (i = 0; i < N; i++) {
        ibuf[i] = (int64_t) routing_order[i];
    }
    wr(ibuf, 8, N);
    for (i = 0; i < N; i++) {
        ibuf[i] = (int64_t) rout_con[i].Nupstream;
    }
    wr(ibuf, 8, N);
    for (i = 0; i < N; i++) {
        for (j = 0; j < MAX_UPSTREAM; j++) {
            ibuf[i * MAX_UPSTREAM + j] = j < rout_con[i].Nupstream ?
                                         (int64_t) rout_con[i].upstream[j] : -1;
        }
    }
    wr(ibuf, 8, N * MAX_UPSTREAM);
    for (i = 0; i < N; i++) {
        wr(rout_con[i].inflow_uh, 8, L);
    }
    for (i = 0; i < N; i++) {
        wr(rout_con[i].runoff_uh, 8, L);
    }

    rbuf = malloc(N * 8);
    for (t = 0; t < nt; t++) {
        rd(rbuf, 8, N);
        for (i = 0; i < N; i++) {
            out_data[i][OUT_RUNOFF][0] = rbuf[i];
        }
        rd(rbuf, 8, N);
        for (i = 0; i < N; i++) {
            out_data[i][OUT_BASEFLOW][0] = rbuf[i];
        }
        rd(rbuf, 8, N);
        for (i = 0; i < N; i++) {
            rout_force[i].discharge = rbuf[i];
        }
        for (i = 0; i < N; i++) {          /* P1, plugin_simulation.c:117 */
            rout_run(i);
        }
        for (i = 0; i < N; i++) {          /* P2, :125-130 */
            rout_basin_run(routing_order[i]);
        }
        for (i = 0; i < N; i++) {
            wr(&rout_var[i].runoff, 8, 1);
        }
        for (i = 0; i < N; i++) {
            wr(&rout_var[i].inflow, 8, 1);
        }
        for (i = 0; i < N; i++) {
            wr(&rout_var[i].discharge, 8, 1);
        }
        for (i = 0; i < N; i++) {
            wr(&rout_var[i].stream, 8, 1);
        }
        for (i = 0; i < N; i++) {
            wr(&rout_var[i].nonrenew_deficit, 8, 1);
        }
        for (i = 0; i < N; i++) {
            wr(rout_var[i].dt_discharge, 8, R);
        }
    }
    fclose(fout);
    return 0;
}
