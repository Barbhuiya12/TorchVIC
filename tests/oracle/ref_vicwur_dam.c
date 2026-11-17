/* Oracle for VIC-WUR global and local dams.
 *
 * Unlike the single-cell copies in the other VIC-WUR oracles this one
 * #includes the UNMODIFIED VIC-WUR 3.0 sources
 *   plugins/dams/src/dam_operation.c   Hanasaki release, k, c, correction,
 *                                      optimal storage, dam_corr_release
 *   plugins/dams/src/dam_register.c    history, op_month, plan, inflow,
 *                                      demand, EFR, activation
 *   plugins/dams/src/dam_run.c         dam_operate, write-back, local/global run
 *   plugins/dams/src/dam_populate.c    default storage alpha C
 *   plugins/dams/src/dam_init_library.c  initial dam_var
 *   plugins/general/src/{array_average,cshift,sort,plugin_time_utils}.c
 * against the stub driver in vicwur_dam_shim/ (see there), and drives them.
 * leap_year is copied from drivers/shared_all/src/vic_time.c:652-673.
 *
 * Input: file of doubles, [mode, n, records...] (argv[1]); output: doubles
 * (argv[2]).  Layouts: tests/test_vicwur_dams.py.
 *   mode 1  PLAN  one dam, one month change: dam_register_operation_start,
 *                 dam_register_operation and their intermediates
 *   mode 2  CORR  dam_corr_release
 *   mode 3  OPT   dam_calc_opt_release, dam_corr_opt_release, dam_calc_opt_
 *                 storage on given 12-month arrays
 *   mode 4  SIM   a set of dams over a run of steps: global_dam_run and
 *                 local_dam_run exactly as plugin_simulation.c:127-139 calls
 *                 them (local dams of every cell, then global dams)
 */
#include <vic_driver_image.h>
#include <plugin.h>

#include <array_average.c>
#include <cshift.c>
#include <sort.c>
#include <plugin_time_utils.c>

/* drivers/shared_all/src/vic_time.c:652-673, verbatim */
bool
leap_year(unsigned short int year,
          unsigned short int calendar)
{
    bool leap = false;

    if ((calendar == CALENDAR_JULIAN ||
         calendar == CALENDAR_STANDARD ||
         calendar == CALENDAR_GREGORIAN ||
         calendar == CALENDAR_PROLEPTIC_GREGORIAN) &&
        (year % 4 == 0)) {
        leap = true;
        if ((calendar == CALENDAR_PROLEPTIC_GREGORIAN) &&
            (year % 100 == 0) && (year % 400 != 0)) {
            leap = false;
        }
        else if ((calendar == CALENDAR_STANDARD ||
                  calendar == CALENDAR_GREGORIAN) &&
                 (year % 100 == 0) && (year % 400 != 0) && (year > 1583)) {
            leap = false;
        }
    }
    return leap;
}

dmy_struct          *dmy;
size_t               current;
global_param_struct  global_param;
domain_struct        local_domain;
double            ***out_data;

#include <dam_operation.c>
#include <dam_register.c>
#include <dam_run.c>
#include <dam_populate.c>
#include <dam_init_library.c>

static double *rd;
#define NEXT (*rd++)

static void
set_globals(int cal, double spd, double alpha, double beta, double gamma)
{
    global_param.calendar = (unsigned short int) cal;
    global_param.model_steps_per_day = (size_t) spd;
    global_param.dt = 86400. / spd;
    plugin_param.DAM_ALPHA = alpha;
    plugin_param.DAM_BETA = beta;
    plugin_param.DAM_GAMMA = gamma;
}

static void
mode_plan(int n, FILE *fo)
{
    int c, i;
    for (c = 0; c < n; c++) {
        dam_con_struct con;
        dam_var_struct var;
        dmy_struct     d0;
        double         cal = NEXT, year = NEXT, month = NEXT, spd = NEXT;
        double         alpha = NEXT, beta = NEXT, gamma = NEXT;
        double         inflow[12], demand[12], efr[12], release[12];
        double         steps_d[12], out[12 * 8 + 3], *q = out, k, cc;
        size_t         steps[12], yr;

        set_globals((int) cal, spd, alpha, beta, gamma);
        initialize_dam_var(&var);
        memset(&con, 0, sizeof(con));
        con.capacity = NEXT;
        var.storage = NEXT;
        var.months_running = (size_t) NEXT;
        for (i = 0; i < 60; i++) var.history_inflow[i] = NEXT;
        for (i = 0; i < 60; i++) var.history_demand[i] = NEXT;
        for (i = 0; i < 60; i++) var.history_efr[i] = NEXT;
        d0.year = (int) year; d0.month = (unsigned short) month;
        dmy = &d0; current = 0;

        dam_register_operation_start(&var);
        /* the intermediates of dam_register_operation, dam_register.c:121-147 */
        yr = var.months_running / MONTHS_PER_YEAR;
        if (yr > DAM_HIST_YEARS) yr = DAM_HIST_YEARS;
        for (i = 0; i < 12; i++) {
            steps[i] = days_per_month(i + 1, d0.year, global_param.calendar) *
                       global_param.model_steps_per_day;
            steps_d[i] = (double) steps[i];
            inflow[i] = array_average(var.history_inflow, yr, 1, i, 11);
            demand[i] = array_average(var.history_demand, yr, 1, i, 11);
            efr[i] = array_average(var.history_efr, yr, 1, i, 11);
        }
        double_flip(inflow, 12); double_flip(demand, 12); double_flip(efr, 12);
        dam_calc_opt_release(inflow, demand, efr, release, 12);
        k = dam_calc_k_factor(con.capacity, var.storage);
        cc = dam_calc_c_factor(inflow, con.capacity, 12, steps);

        dam_register_operation(&con, &var);

        *q++ = (double) var.op_month;
        for (i = 0; i < 12; i++) *q++ = var.op_release[i];
        for (i = 0; i < 12; i++) *q++ = var.op_storage[i];
        *q++ = k; *q++ = cc;
        for (i = 0; i < 12; i++) *q++ = inflow[i];
        for (i = 0; i < 12; i++) *q++ = demand[i];
        for (i = 0; i < 12; i++) *q++ = efr[i];
        for (i = 0; i < 12; i++) *q++ = release[i];
        for (i = 0; i < 12; i++) *q++ = steps_d[i];
        fwrite(out, 8, q - out, fo);
    }
}

static void
mode_corr(int n, FILE *fo)
{
    int c;
    for (c = 0; c < n; c++) {
        double alpha = NEXT, beta = NEXT, gamma = NEXT, spd = NEXT;
        double rel = NEXT, sto = NEXT, cap = NEXT, out;
        set_globals(0, spd, alpha, beta, gamma);
        out = dam_corr_release(rel, sto, cap);
        fwrite(&out, 8, 1, fo);
    }
}

static void
mode_opt(int n, FILE *fo)
{
    int c, i;
    for (c = 0; c < n; c++) {
        double inflow[12], demand[12], efr[12], rel[12], sto[12];
        double out[12 * 3 + 2], *q = out, k, cc, cur;
        size_t steps[12];
        for (i = 0; i < 12; i++) inflow[i] = NEXT;
        for (i = 0; i < 12; i++) demand[i] = NEXT;
        for (i = 0; i < 12; i++) efr[i] = NEXT;
        for (i = 0; i < 12; i++) steps[i] = (size_t) NEXT;
        k = NEXT; cc = NEXT; cur = NEXT;
        dam_calc_opt_release(inflow, demand, efr, rel, 12);
        for (i = 0; i < 12; i++) *q++ = rel[i];
        dam_corr_opt_release(inflow, rel, 12, k, cc);
        for (i = 0; i < 12; i++) *q++ = rel[i];
        dam_calc_opt_storage(inflow, rel, sto, 12, steps, cur);
        for (i = 0; i < 12; i++) *q++ = sto[i];
        fwrite(out, 8, q - out, fo);
    }
}

#define DW (12 + 180 + 24)

static void
mode_sim(FILE *fo)
{
    int       i, j, k, t, s;
    size_t    ND = (size_t) NEXT, NC = (size_t) NEXT, NT = (size_t) NEXT;
    double    spd = NEXT, rspd = NEXT;
    size_t    LR = (size_t) NEXT;
    int       cal = (int) NEXT;
    int       y = (int) NEXT, m = (int) NEXT, d = (int) NEXT;
    double    alpha = NEXT, beta = NEXT, gamma = NEXT;
    size_t    NS, MAXS;
    int      *cell_of;

    plugin_options.ROUTING = true;
    plugin_options.DAMS = true;
    plugin_options.WATERUSE = NEXT > 0.5;
    plugin_options.EFR = NEXT > 0.5;
    NS = (size_t) NEXT; MAXS = (size_t) NEXT;
    set_globals(cal, spd, alpha, beta, gamma);
    plugin_global_param.rout_steps_per_day = (size_t) rspd;

    local_domain.ncells_active = NC;
    local_domain.locations = calloc(NC, sizeof(location_struct));
    wu_con_map = calloc(NC, sizeof(wu_con_map_struct));
    wu_var = calloc(NC, sizeof(wu_var_struct *));
    efr_force = calloc(NC, sizeof(efr_force_struct));
    rout_var = calloc(NC, sizeof(rout_var_struct));
    out_data = calloc(NC, sizeof(double **));
    dam_con_map = calloc(NC, sizeof(dam_con_map_struct));
    dam_con = calloc(NC, sizeof(dam_con_struct *));
    dam_var = calloc(NC, sizeof(dam_var_struct *));
    for (i = 0; i < (int) NC; i++) {
        local_domain.locations[i].area = NEXT;
        wu_con_map[i].ns_active = (size_t) NEXT;
        wu_var[i] = calloc(NS, sizeof(wu_var_struct));
        rout_var[i].dt_discharge = calloc(LR, sizeof(double));
        out_data[i] = calloc(N_OUTVAR_TYPES, sizeof(double *));
        out_data[i][OUT_RUNOFF] = calloc(1, sizeof(double));
    }
    /* dams, listed cell by cell in class order */
    cell_of = malloc(ND * sizeof(int));
    {
        double *save = rd;
        for (i = 0; i < (int) ND; i++) {
            cell_of[i] = (int) NEXT;
            rd += 5 + 2 * MAXS;
            dam_con_map[cell_of[i]].nd_active++;
        }
        rd = save;
    }
    for (i = 0; i < (int) NC; i++) {
        dam_con[i] = calloc(dam_con_map[i].nd_active + 1, sizeof(dam_con_struct));
        dam_var[i] = calloc(dam_con_map[i].nd_active + 1, sizeof(dam_var_struct));
        dam_con_map[i].nd_active = 0;   /* refilled below */
    }
    for (i = 0; i < (int) ND; i++) {
        int c = (int) NEXT, j2 = (int) dam_con_map[c].nd_active++;
        dam_con_struct *dc = &dam_con[c][j2];
        dc->type = (unsigned short) NEXT;
        dc->year = (unsigned short) NEXT;
        dc->capacity = NEXT;
        dc->inflow_frac = NEXT;
        dc->nservice = (size_t) NEXT;
        dc->service = calloc(MAXS, sizeof(size_t));
        dc->service_frac = calloc(MAXS, sizeof(double));
        for (k = 0; k < (int) MAXS; k++) dc->service[k] = (size_t) NEXT;
        for (k = 0; k < (int) MAXS; k++) dc->service_frac[k] = NEXT;
        initialize_dam_var(&dam_var[c][j2]);
    }
    {   /* initialize_dam_con wipes type etc, so do not call it; defaults: */
        dam_generate_default_state();
    }

    dmy = calloc(NT, sizeof(dmy_struct));
    dmy[0].year = y; dmy[0].month = m; dmy[0].day = d;
    for (t = 1; t < (int) NT; t++) {
        dmy[t] = dmy[t - 1];
        if (t % (int) spd == 0) {
            dmy[t].day++;
            if (dmy[t].day > days_per_month(dmy[t].month, dmy[t].year,
                                            global_param.calendar)) {
                dmy[t].day = 1;
                dmy[t].month++;
                if (dmy[t].month > 12) {
                    dmy[t].month = 1;
                    dmy[t].year++;
                }
            }
        }
    }

    for (t = 0; t < (int) NT; t++) {
        double *buf = malloc((ND * DW + NC * (2 + LR)) * sizeof(double)), *q = buf;
        current = (size_t) t;
        for (i = 0; i < (int) NC; i++) {
            *out_data[i][OUT_RUNOFF] = NEXT;
            rout_var[i].discharge = NEXT;
            for (j = 0; j < (int) LR; j++) rout_var[i].dt_discharge[j] = NEXT;
            efr_force[i].discharge = NEXT;
            for (s = 0; s < (int) NS; s++) wu_var[i][s].demand_surf = NEXT;
        }
        for (i = 0; i < (int) NC; i++) local_dam_run(i);
        for (i = 0; i < (int) NC; i++) global_dam_run(i);
        for (i = 0; i < (int) NC; i++) {
            for (j = 0; j < (int) dam_con_map[i].nd_active; j++) {
                dam_var_struct *v = &dam_var[i][j];
                *q++ = (double) v->active;
                *q++ = v->inflow; *q++ = v->demand; *q++ = v->efr;
                *q++ = v->release; *q++ = v->storage;
                *q++ = (double) v->op_month;
                *q++ = (double) v->months_running;
                *q++ = v->total_inflow; *q++ = v->total_demand;
                *q++ = v->total_efr; *q++ = (double) v->register_steps;
                for (k = 0; k < 60; k++) *q++ = v->history_inflow[k];
                for (k = 0; k < 60; k++) *q++ = v->history_demand[k];
                for (k = 0; k < 60; k++) *q++ = v->history_efr[k];
                for (k = 0; k < 12; k++) *q++ = v->op_release[k];
                for (k = 0; k < 12; k++) *q++ = v->op_storage[k];
            }
        }
        for (i = 0; i < (int) NC; i++) {
            *q++ = *out_data[i][OUT_RUNOFF];
            *q++ = rout_var[i].discharge;
            for (j = 0; j < (int) LR; j++) *q++ = rout_var[i].dt_discharge[j];
        }
        fwrite(buf, 8, q - buf, fo);
        free(buf);
    }
}

int
main(int argc, char **argv)
{
    FILE   *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    long    sz;
    double *all, mode, n;

    if (!fi || !fo) return 1;
    fseek(fi, 0, SEEK_END); sz = ftell(fi); fseek(fi, 0, SEEK_SET);
    all = malloc(sz);
    if (fread(all, 1, sz, fi) != (size_t) sz) return 2;
    rd = all;
    mode = NEXT;
    if (mode == 4) {
        mode_sim(fo);
    }
    else {
        n = NEXT;
        if (mode == 1) mode_plan((int) n, fo);
        else if (mode == 2) mode_corr((int) n, fo);
        else if (mode == 3) mode_opt((int) n, fo);
    }
    fclose(fi); fclose(fo);
    return 0;
}
