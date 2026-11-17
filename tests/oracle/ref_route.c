/* Reference oracle for the differentiable router.
 *
 * Links VIC's own convolution() and cshift() and steps them exactly as
 * rout_run() does: one call per time step, runoff for that step in, discharge
 * for that step out. tests/test_routing_vs_rvic.py drives this and
 * compares the result against torchvic.routing.route.
 *
 * Input is read from a file so the C and the Python see byte-identical
 * numbers rather than two roundings of the same generator:
 *
 *   n_sources n_outlets n_uh nt dt
 *   source2outlet_ind      n_sources ints
 *   source_time_offset     n_sources ints
 *   unit_hydrograph        n_uh * n_sources doubles, [uh_step][source]
 *   outlet_area            n_outlets doubles
 *   runoff                 nt * n_sources doubles, [t][source]
 *
 * Output is nt lines of n_outlets values at full double precision.
 */
#include <rout.h>

rout_struct         rout;
global_param_struct global_param;
domain_struct       global_domain;

static void
die(const char *m)
{
    fprintf(stderr, "ref_route: %s\n", m);
    exit(1);
}

int
main(int argc, char **argv)
{
    FILE   *fp;
    size_t  ns, no, nuh, i, t;
    int     nt;
    double *runoff, *discharge;

    if (argc < 2) {
        die("usage: ref_route <input file>");
    }
    fp = fopen(argv[1], "r");
    if (!fp) {
        die("cannot open input");
    }
    if (fscanf(fp, "%zu %zu %zu %d %lf", &ns, &no, &nuh, &nt,
               &global_param.dt) != 5) {
        die("bad header");
    }

    rout.rout_param.n_sources = ns;
    rout.rout_param.n_outlets = no;
    rout.rout_param.n_timesteps = nuh;
    /* The ring must hold the furthest write any source offset can reach. */
    rout.rout_param.full_time_length = nuh + nt + 1;

    rout.rout_param.source2outlet_ind = malloc(ns * sizeof(size_t));
    rout.rout_param.source_time_offset = malloc(ns * sizeof(int));
    rout.rout_param.source_VIC_index = malloc(ns * sizeof(int));
    rout.rout_param.outlet_VIC_index = malloc(no * sizeof(int));
    rout.rout_param.unit_hydrograph = malloc(nuh * ns * sizeof(double));
    rout.ring = calloc(rout.rout_param.full_time_length * no, sizeof(double));
    global_domain.ncells_total = (ns > no ? ns : no);
    global_domain.locations = malloc(global_domain.ncells_total *
                                     sizeof(location_struct));
    runoff = malloc(ns * sizeof(double));
    discharge = calloc(global_domain.ncells_total, sizeof(double));

    for (i = 0; i < ns; i++) {
        if (fscanf(fp, "%zu", &rout.rout_param.source2outlet_ind[i]) != 1) {
            die("bad source2outlet");
        }
        /* Identity mapping between routing indices and VIC indices; the
         * mapping is a plain indirection and is not what this test is for. */
        rout.rout_param.source_VIC_index[i] = (int)i;
    }
    for (i = 0; i < ns; i++) {
        if (fscanf(fp, "%d", &rout.rout_param.source_time_offset[i]) != 1) {
            die("bad offset");
        }
    }
    for (i = 0; i < nuh * ns; i++) {
        if (fscanf(fp, "%lf", &rout.rout_param.unit_hydrograph[i]) != 1) {
            die("bad uh");
        }
    }
    for (i = 0; i < no; i++) {
        rout.rout_param.outlet_VIC_index[i] = (int)i;
        if (fscanf(fp, "%lf", &global_domain.locations[i].area) != 1) {
            die("bad area");
        }
    }

    for (t = 0; t < (size_t)nt; t++) {
        for (i = 0; i < ns; i++) {
            if (fscanf(fp, "%lf", &runoff[i]) != 1) {
                die("bad runoff");
            }
        }
        convolution(runoff, discharge);
        for (i = 0; i < no; i++) {
            printf(i ? " %.17g" : "%.17g",
                   discharge[rout.rout_param.outlet_VIC_index[i]]);
        }
        printf("\n");
    }
    fclose(fp);
    return 0;
}
