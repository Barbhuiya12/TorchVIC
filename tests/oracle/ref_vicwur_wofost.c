/* Oracle for the VIC-WUR WOFOST-NPK crop kernel.
 *
 * The unmodified VIC-WUR 3.0 sources are #included: every file of
 * plugins/wofost/src that the daily crop step reaches, plus
 * plugins/crops/src/crop_run.c, wofost_run.c, wofost_update_step_vars.c and
 * wofost_const_library.c, and plugins/general/src/cshift.c.  The driver
 * globals they read (force, all_vars, soil_con, veg_con, co2_force, dmy, ...)
 * are owned here and filled from the input file; tests/oracle/wofost_shim
 * supplies the VIC types.  Two builds:
 *   ref_vicwur_wofost      float, as the C model (fp32 bound)
 *   ref_vicwur_wofost_f64  -DWOFOST_DOUBLE: `#define float double` before
 *                          the sources, the fp64 reference of the 1e-12 gate
 *
 * Only the input-file readers are replaced (their fscanf("%f") cannot run in
 * the double build): the crop/management scalars go through the real
 * FillCropVariables / FillManageVariables (Fillvar.c); the table assignment,
 * MaxOptimumTemp and the initial plant states are a copy of
 * Cropdata.c:112-285 (marked below); GetSimInput.c:35-39 and the season set
 * up of wofost_init.c:147-195,500-529 are copied as well.
 *
 * Modes (first value of the input file):
 *   1  season run: N cells x B bands x NC crop classes for T days, the order
 *      of a VIC-WUR daily step: U wofost_update_step_vars, P3 crop_run.
 *      Writes the unit fields every day (names to <out>.names) and the full
 *      leaf lists every `leafevery` days to <out>.leaves.
 *   2  function records: Afgen, Astro + DailyTotalAssimilation + Correct,
 *      Partioning, DyingLeaves, NutritionINDX on random states.
 * Layout of the files: tests/wofost_npk_cases.py.
 */
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WOFOST_DOUBLE
#define float double
#endif

#include <vic_driver_image.h>
#include <plugin.h>

/* driver globals read by the included sources */
global_param_struct        global_param;
option_struct              options;
soil_con_struct           *soil_con;
force_data_struct         *force;
co2_force_struct          *co2_force;
all_vars_struct           *all_vars;
veg_con_map_struct        *veg_con_map;
veg_con_struct           **veg_con;
size_t                     NR = 0;
dmy_struct                *dmy;
size_t                     current;
plugin_option_struct       plugin_options;
plugin_global_param_struct plugin_global_param;

/* unchanged VIC-WUR 3.0 sources */
#include "cshift.c"
#include "wofost_const_library.c"
#include "Afgen.c"
#include "Functions.c"
#include "List.c"
#include "Sweaf.c"
#include "Astro.c"
#include "Assim.c"
#include "IfSowing.c"
#include "Develop.c"
#include "DM-Partioning.c"
#include "Conversion.c"
#include "Respiration.c"
#include "Growth.c"
#include "Lai.c"
#include "LeaveGrowth.c"
#include "Leaves.c"
#include "IntegrationCrop.c"
#include "RateCalculationCrop.c"
#include "NutrientLoss.c"
#include "Nutrients.c"
#include "NutrientPartitioning.c"
#include "NutrientTranslocation.c"
#include "CropNutrientRates.c"
#include "NutritionIndex.c"
#include "SoilNutrientRates.c"
#include "IntegrationNutrients.c"
#include "InitializeCrop.c"
#include "InitializeNutrients.c"
#include "RatesToZero.c"
#include "Clean.c"
#include "Fillvar.c"
#include "wofost_run.c"
#include "crop_run.c"
#include "wofost_update_step_vars.c"

static FILE *fin;

static double
rd(void)
{
    double v;
    if (fread(&v, sizeof(v), 1, fin) != 1) {
        fprintf(stderr, "ref_vicwur_wofost: short input\n");
        exit(2);
    }
    return v;
}

static int
rdi(void)
{
    return (int) rd();
}

static TABLE *
rd_table(int P)
{
    int    n = rdi(), i;
    double x[64], y[64];
    TABLE *head = NULL, *t = NULL;
    for (i = 0; i < P; i++) {
        x[i] = rd();
    }
    for (i = 0; i < P; i++) {
        y[i] = rd();
    }
    for (i = 0; i < n; i++) {
        TABLE *node = malloc(sizeof(TABLE));
        node->x = x[i];
        node->y = y[i];
        node->next = NULL;
        if (head == NULL) {
            head = node;
        }
        else {
            t->next = node;
        }
        t = node;
    }
    return head;
}

static TABLE_D *
rd_table_d(int PM)
{
    int      n = rdi(), i;
    double   m[64], d[64], a[64];
    TABLE_D *head = NULL, *t = NULL;
    for (i = 0; i < PM; i++) {
        m[i] = rd();
    }
    for (i = 0; i < PM; i++) {
        d[i] = rd();
    }
    for (i = 0; i < PM; i++) {
        a[i] = rd();
    }
    for (i = 0; i < n; i++) {
        TABLE_D *node = malloc(sizeof(TABLE_D));
        node->month = (int) m[i];
        node->day = (int) d[i];
        node->amount = a[i];
        node->next = NULL;
        if (head == NULL) {
            head = node;
        }
        else {
            t->next = node;
        }
        t = node;
    }
    return head;
}

static TABLE *
copy_table(TABLE *s)
{
    TABLE *head = NULL, *t = NULL;
    while (s) {
        TABLE *node = malloc(sizeof(TABLE));
        *node = *s;
        node->next = NULL;
        if (head == NULL) {
            head = node;
        }
        else {
            t->next = node;
        }
        t = node;
        s = s->next;
    }
    return head;
}

/* One crop class template: the parsed crop and management file. */
typedef struct {
    float    var[100];
    TABLE   *tab[NR_TABLES_CRP];
    float    mvar[100];
    TABLE_D *mtab[NR_TABLES_MANAGEMENT];
    int      emergence;
} template_t;

static void
init_unit(SimUnit *g, template_t *tp)
{
    int   i;
    float tmp[100];
    TABLE *head;
    float XValue, YValue;

    memset(g->crp, 0, sizeof(*g->crp));
    memset(g->ste, 0, sizeof(*g->ste));
    memset(g->mng, 0, sizeof(*g->mng));
    memset(g->soil, 0, sizeof(*g->soil));
    memset(g->met, 0, sizeof(*g->met));
    g->emergence = 0;
    g->file = 0;
    g->cultivating = 0;
    g->growing = 0;
    memset(&g->start, 0, sizeof(g->start));
    memset(&g->end, 0, sizeof(g->end));

    for (i = 0; i < 100; i++) {
        tmp[i] = tp->var[i];
    }
    FillCropVariables(g->crp, tmp);
    for (i = 0; i < 100; i++) {
        tmp[i] = tp->mvar[i];
    }
    FillManageVariables(g->mng, tmp);
    g->mng->N_Fert_table = tp->mtab[0];
    g->mng->P_Fert_table = tp->mtab[1];
    g->mng->K_Fert_table = tp->mtab[2];
    g->mng->Irrigation = tp->mtab[3];

    /* ---- copy of Cropdata.c:112-285 ---- */
    if (g->crp->prm.IdentifyAnthesis < 2) {
        g->crp->prm.VernalizationRate = NULL;
    }
    else {
        g->crp->prm.VernalizationRate = copy_table(tp->tab[0]);
    }
    g->crp->prm.DeltaTempSum = copy_table(tp->tab[1]);
    g->crp->prm.SpecificLeaveArea = copy_table(tp->tab[2]);
    g->crp->prm.SpecificStemArea = copy_table(tp->tab[3]);
    g->crp->prm.KDiffuseTb = copy_table(tp->tab[4]);
    g->crp->prm.EFFTb = copy_table(tp->tab[5]);
    g->crp->prm.MaxAssimRate = copy_table(tp->tab[6]);
    g->crp->prm.FactorAssimRateTemp = copy_table(tp->tab[7]);
    g->crp->prm.FactorGrossAssimTemp = copy_table(tp->tab[8]);
    g->crp->prm.CO2AMAXTB = copy_table(tp->tab[9]);
    g->crp->prm.CO2EFFTB = copy_table(tp->tab[10]);
    g->crp->prm.CO2TRATB = copy_table(tp->tab[11]);
    g->crp->prm.FactorSenescence = copy_table(tp->tab[12]);
    g->crp->prm.Roots = copy_table(tp->tab[13]);
    g->crp->prm.Leaves = copy_table(tp->tab[14]);
    g->crp->prm.Stems = copy_table(tp->tab[15]);
    g->crp->prm.Storage = copy_table(tp->tab[16]);
    g->crp->prm.DeathRateStems = copy_table(tp->tab[17]);
    g->crp->prm.DeathRateRoots = copy_table(tp->tab[18]);
    g->crp->prm.N_MaxLeaves = copy_table(tp->tab[19]);
    g->crp->prm.P_MaxLeaves = copy_table(tp->tab[20]);
    g->crp->prm.K_MaxLeaves = copy_table(tp->tab[21]);

    head = g->crp->prm.FactorAssimRateTemp;
    XValue = 0;
    YValue = 0;
    while (head) {
        if (head->y >= YValue) {
            XValue = head->x;
            YValue = head->y;
        }
        head = head->next;
    }
    g->crp->prm.MaxOptimumTemp = XValue;
    g->crp->Emergence = 0;
    g->crp->TSumEmergence = 0.;
    g->crp->CultivateDay = 0.;
    g->crp->st.RootDepth = 0.;
    g->crp->st.RootDepth_prev = 0.;
    g->crp->st.Development = 0.;
    g->crp->DaysOxygenStress = 0;
    g->crp->NutrientStress = 1.;
    g->crp->NPK_Indx = 1.;
    g->crp->N_st.Indx = 1.;
    g->crp->P_st.Indx = 1.;
    g->crp->K_st.Indx = 1.;
    g->crp->LeaveProperties = NULL;   /* other states: zero (memset) */
    /* ---- end of the Cropdata.c copy ---- */

    /* GetSimInput.c:35-39 (start month/day set by the caller) */
    g->emergence = tp->emergence;
    g->crp->Sowing = 0;
    g->crp->Emergence = 0;
}

/* ------------------------------------------------------------- dump */
#define NUTRIENT_ST(X, P, s) \
    X(P ## _st_roots, s.roots) X(P ## _st_stems, s.stems) \
    X(P ## _st_leaves, s.leaves) X(P ## _st_storage, s.storage) \
    X(P ## _st_Demand, s.Demand) X(P ## _st_Max_lv, s.Max_lv) \
    X(P ## _st_Max_st, s.Max_st) X(P ## _st_Max_ro, s.Max_ro) \
    X(P ## _st_Max_so, s.Max_so) X(P ## _st_Optimum_lv, s.Optimum_lv) \
    X(P ## _st_Optimum_st, s.Optimum_st) X(P ## _st_Indx, s.Indx) \
    X(P ## _st_Uptake, s.Uptake) X(P ## _st_death_lv, s.death_lv) \
    X(P ## _st_death_st, s.death_st) X(P ## _st_death_ro, s.death_ro) \
    X(P ## _st_Avail, s.Avail) X(P ## _st_Avail_lv, s.Avail_lv) \
    X(P ## _st_Avail_st, s.Avail_st) X(P ## _st_Avail_ro, s.Avail_ro)
#define NUTRIENT_RT(X, P, r) \
    X(P ## _rt_roots, r.roots) X(P ## _rt_stems, r.stems) \
    X(P ## _rt_leaves, r.leaves) X(P ## _rt_storage, r.storage) \
    X(P ## _rt_Demand, r.Demand) X(P ## _rt_Demand_lv, r.Demand_lv) \
    X(P ## _rt_Demand_st, r.Demand_st) X(P ## _rt_Demand_ro, r.Demand_ro) \
    X(P ## _rt_Demand_so, r.Demand_so) X(P ## _rt_Supply, r.Supply) \
    X(P ## _rt_Transloc, r.Transloc) X(P ## _rt_Transloc_lv, r.Transloc_lv) \
    X(P ## _rt_Transloc_st, r.Transloc_st) \
    X(P ## _rt_Transloc_ro, r.Transloc_ro) X(P ## _rt_Uptake, r.Uptake) \
    X(P ## _rt_Uptake_lv, r.Uptake_lv) X(P ## _rt_Uptake_st, r.Uptake_st) \
    X(P ## _rt_Uptake_ro, r.Uptake_ro) X(P ## _rt_death_lv, r.death_lv) \
    X(P ## _rt_death_st, r.death_st) X(P ## _rt_death_ro, r.death_ro)

#define FIELDS(X) \
    X(Sowing, g->crp->Sowing) X(Emergence, g->crp->Emergence) \
    X(CultivateDay, g->crp->CultivateDay) X(GrowthDay, g->crp->GrowthDay) \
    X(cultivating, g->cultivating) X(growing, g->growing) \
    X(NPK_Indx, g->crp->NPK_Indx) X(NutrientStress, g->crp->NutrientStress) \
    X(TSumEmergence, g->crp->TSumEmergence) \
    X(fac_ro, g->crp->fac_ro) X(fac_lv, g->crp->fac_lv) \
    X(fac_st, g->crp->fac_st) X(fac_so, g->crp->fac_so) \
    X(prm_TempSum1, g->crp->prm.TempSum1) \
    X(prm_TempSum2, g->crp->prm.TempSum2) \
    X(prm_MaxRootingDepth, g->crp->prm.MaxRootingDepth) \
    X(rt_roots, g->crp->rt.roots) X(rt_stems, g->crp->rt.stems) \
    X(rt_leaves, g->crp->rt.leaves) X(rt_LAIExp, g->crp->rt.LAIExp) \
    X(rt_storage, g->crp->rt.storage) \
    X(rt_Development, g->crp->rt.Development) \
    X(rt_RootDepth, g->crp->rt.RootDepth) \
    X(rt_vernalization, g->crp->rt.vernalization) \
    X(st_roots, g->crp->st.roots) X(st_stems, g->crp->st.stems) \
    X(st_leaves, g->crp->st.leaves) X(st_LAI, g->crp->st.LAI) \
    X(st_LAIExp, g->crp->st.LAIExp) X(st_storage, g->crp->st.storage) \
    X(st_Development, g->crp->st.Development) \
    X(st_Development_prev, g->crp->st.Development_prev) \
    X(st_RootDepth, g->crp->st.RootDepth) \
    X(st_RootDepth_prev, g->crp->st.RootDepth_prev) \
    X(st_vernalization, g->crp->st.vernalization) \
    X(drt_roots, g->crp->drt.roots) X(drt_stems, g->crp->drt.stems) \
    X(drt_leaves, g->crp->drt.leaves) \
    X(dst_roots, g->crp->dst.roots) X(dst_stems, g->crp->dst.stems) \
    X(dst_leaves, g->crp->dst.leaves) \
    NUTRIENT_ST(X, N, g->crp->N_st) NUTRIENT_ST(X, P, g->crp->P_st) \
    NUTRIENT_ST(X, K, g->crp->K_st) \
    NUTRIENT_RT(X, N, g->crp->N_rt) NUTRIENT_RT(X, P, g->crp->P_rt) \
    NUTRIENT_RT(X, K, g->crp->K_rt) \
    X(ste_st_N_tot, g->ste->st_N_tot) X(ste_st_P_tot, g->ste->st_P_tot) \
    X(ste_st_K_tot, g->ste->st_K_tot) X(ste_st_N_mins, g->ste->st_N_mins) \
    X(ste_st_P_mins, g->ste->st_P_mins) X(ste_st_K_mins, g->ste->st_K_mins) \
    X(ste_rt_N_tot, g->ste->rt_N_tot) X(ste_rt_P_tot, g->ste->rt_P_tot) \
    X(ste_rt_K_tot, g->ste->rt_K_tot) X(ste_rt_N_mins, g->ste->rt_N_mins) \
    X(ste_rt_P_mins, g->ste->rt_P_mins) X(ste_rt_K_mins, g->ste->rt_K_mins) \
    X(mng_N_Mins, g->mng->N_Mins) X(mng_NRecoveryFrac, g->mng->NRecoveryFrac) \
    X(mng_N_external, g->mng->N_external) \
    X(mng_P_external, g->mng->P_external) \
    X(mng_K_external, g->mng->K_external) \
    X(soil_WaterStress, g->soil->WaterStress) \
    X(met_Tmin0, g->met->Tmin[0]) X(met_Tmin1, g->met->Tmin[1]) \
    X(met_Tmin2, g->met->Tmin[2]) X(met_Tmin3, g->met->Tmin[3]) \
    X(met_Tmin4, g->met->Tmin[4]) X(met_Tmin5, g->met->Tmin[5]) \
    X(met_Tmin6, g->met->Tmin[6]) X(met_Tmax, g->met->Tmax) \
    X(met_Temp, g->met->Temp) X(met_DayTemp, g->met->DayTemp) \
    X(met_Radiation, g->met->Radiation) X(met_CO2, g->met->CO2) \
    X(met_PotEvaptrans, g->met->PotEvaptrans) \
    X(met_MeteoDay, g->met->MeteoDay) \
    X(met_AtmosphTransm, g->met->AtmosphTransm) \
    X(met_AngotRadiation, g->met->AngotRadiation) \
    X(met_Daylength, g->met->Daylength) \
    X(met_PARDaylength, g->met->PARDaylength) \
    X(met_SinLD, g->met->SinLD) X(met_CosLD, g->met->CosLD) \
    X(met_DiffRadPP, g->met->DiffRadPP) X(met_DSinBE, g->met->DSinBE) \
    X(met_TempStress, g->met->TempStress) \
    X(met_TminStress, g->met->TminStress)

#define X_COUNT(name, expr) +1
#define NFIELDS (0 FIELDS(X_COUNT))
#define X_VAL(name, expr) buf[k++] = (double) (expr);
#define X_NAME(name, expr) fprintf(fn, "%s\n", # name);

static void
dump_unit(SimUnit *g, FILE *fo, double con_tsum1, double con_tsum2)
{
    double buf[NFIELDS + 8];
    size_t k = 0;
    double sw = 0., swa = 0.;
    int    n = 0;
    Green *l;
    if (g == NULL) {
        for (k = 0; k < NFIELDS + 6; k++) {
            buf[k] = NAN;
        }
        fwrite(buf, sizeof(double), NFIELDS + 6, fo);
        return;
    }
    FIELDS(X_VAL)
    for (l = g->crp->LeaveProperties; l; l = l->next) {
        n++;
        sw += l->weight;
        swa += l->weight * l->area;
    }
    buf[k++] = n;
    buf[k++] = sw;
    buf[k++] = swa;
    buf[k++] = g->crp->LeaveProperties ? g->crp->LeaveProperties->age : NAN;
    buf[k++] = con_tsum1;
    buf[k++] = con_tsum2;
    fwrite(buf, sizeof(double), k, fo);
}

static void
dump_names(const char *path)
{
    FILE *fn = fopen(path, "w");
    FIELDS(X_NAME)
    fprintf(fn, "leaf_n\nleaf_sumw\nleaf_sumwa\nleaf_headage\n"
            "con_tsum1\ncon_tsum2\n");
    fclose(fn);
}

static void
dump_leaves(SimUnit *g, FILE *fl, int t, int u)
{
    Green *l;
    double h[3];
    int    n = 0;
    if (g) {
        for (l = g->crp->LeaveProperties; l; l = l->next) {
            n++;
        }
    }
    h[0] = t;
    h[1] = u;
    h[2] = n;
    fwrite(h, sizeof(double), 3, fl);
    if (g) {
        for (l = g->crp->LeaveProperties; l; l = l->next) {
            double v[3] = {l->weight, l->age, l->area};
            fwrite(v, sizeof(double), 3, fl);
        }
    }
}

/* ------------------------------------------------------------- mode 1 */
static int
season_run(const char *outp)
{
    int         N, B, NC, T, NF, P, PM, leafevery;
    int         i, j, k, b, t, f, u;
    template_t *tp;
    char        path[4096];
    FILE       *fo, *fl;
    SimUnit ****unit;    /* [cell][band][class] or NULL */
    int       **active;

    N = rdi();
    B = rdi();
    NC = rdi();
    T = rdi();
    NF = rdi();
    P = rdi();
    PM = rdi();
    leafevery = rdi();
    plugin_options.WOFOST_PIRR = rdi();
    plugin_options.WOFOST_PFERT = rdi();
    plugin_options.WOFOST_PTEMP = rdi();
    plugin_options.WOFOST_DIST_TSUM = rdi();
    plugin_options.WOFOST_DIST_FERT = rdi();
    plugin_options.WOFOST_CALC_MIN = rdi();
    plugin_options.WOFOST_CONTINUE = rdi();
    plugin_options.WOFOST_FORCE_TSUM = rdi();
    plugin_options.WOFOST_FORCE_FERT = rdi();
    plugin_options.NFERTTIMES = NF;
    plugin_options.NCROPTYPES = NC;
    plugin_global_param.wofost_steps_per_day = 1;
    global_param.atmos_steps_per_day = 1;
    global_param.model_steps_per_day = 1;
    global_param.calendar = rdi();
    options.SNOW_BAND = B;

    tp = calloc(NC, sizeof(*tp));
    for (k = 0; k < NC; k++) {
        for (j = 0; j < NR_VARIABLES_CRP; j++) {
            tp[k].var[j] = rd();
        }
        for (j = 0; j < NR_TABLES_CRP; j++) {
            tp[k].tab[j] = rd_table(P);
        }
        for (j = 0; j < NR_VARIABLES_MANAGEMENT; j++) {
            tp[k].mvar[j] = rd();
        }
        for (j = 0; j < NR_TABLES_MANAGEMENT; j++) {
            tp[k].mtab[j] = rd_table_d(PM);
        }
        tp[k].emergence = rdi();
    }

    soil_con = calloc(N, sizeof(*soil_con));
    force = calloc(N, sizeof(*force));
    co2_force = calloc(N, sizeof(*co2_force));
    all_vars = calloc(N, sizeof(*all_vars));
    veg_con_map = calloc(N, sizeof(*veg_con_map));
    veg_con = calloc(N, sizeof(*veg_con));
    crop_con_map = calloc(N, sizeof(*crop_con_map));
    crop_con = calloc(N, sizeof(*crop_con));
    crop_force = calloc(N, sizeof(*crop_force));
    Grid = calloc(N, sizeof(*Grid));
    unit = calloc(N, sizeof(*unit));
    active = calloc(N, sizeof(*active));
    for (i = 0; i < N; i++) {
        double lat = rd();
        soil_con[i].ph = rd();
        soil_con[i].carbon = rd();
        soil_con[i].AreaFract = calloc(B, sizeof(double));
        soil_con[i].Tfactor = calloc(B, sizeof(double));
        for (b = 0; b < B; b++) {
            soil_con[i].AreaFract[b] = rd();
        }
        for (b = 0; b < B; b++) {
            soil_con[i].Tfactor[b] = rd();
        }
        force[i].air_temp = calloc(1, sizeof(double));
        force[i].shortwave = calloc(1, sizeof(double));
        /* every crop class is its own vegetation tile: veg class = class */
        all_vars[i].cell = calloc(NC, sizeof(cell_data_struct *));
        for (k = 0; k < NC; k++) {
            all_vars[i].cell[k] = calloc(B, sizeof(cell_data_struct));
        }
        veg_con_map[i].vidx = calloc(NC, sizeof(int));
        veg_con[i] = calloc(NC, sizeof(veg_con_struct));
        crop_con_map[i].nc_types = NC;
        crop_con_map[i].veg_class = calloc(NC, sizeof(int));
        crop_con_map[i].cidx = calloc(NC, sizeof(int));
        crop_con_map[i].Cc = calloc(NC, sizeof(double *));
        crop_con[i] = calloc(NC, sizeof(crop_con_struct));
        crop_force[i].tsum1 = calloc(NC, sizeof(double));
        crop_force[i].tsum2 = calloc(NC, sizeof(double));
        crop_force[i].DVS_point = calloc(NC, sizeof(double *));
        crop_force[i].N_amount = calloc(NC, sizeof(double *));
        crop_force[i].P_amount = calloc(NC, sizeof(double *));
        crop_force[i].K_amount = calloc(NC, sizeof(double *));
        Grid[i] = calloc(B, sizeof(SimUnit *));
        unit[i] = calloc(B, sizeof(SimUnit **));
        active[i] = calloc(NC, sizeof(int));
        for (b = 0; b < B; b++) {
            unit[i][b] = calloc(NC, sizeof(SimUnit *));
        }
        u = 0;
        for (k = 0; k < NC; k++) {
            int    act = rdi();
            double cv = rd(), smon, sday, sdoy, edoy;
            crop_con_map[i].Cc[k] = calloc(12, sizeof(double));
            for (j = 0; j < 12; j++) {
                crop_con_map[i].Cc[k][j] = rd();
            }
            smon = rd();
            sday = rd();
            sdoy = rd();
            edoy = rd();
            veg_con_map[i].vidx[k] = k;
            veg_con[i][k].Cv = cv;
            crop_con_map[i].veg_class[u] = k;
            active[i][k] = act;
            /* crop_con, crop_force are per crop tile (cidx order) */
            crop_con[i][u].DVS_point = calloc(NF, sizeof(double));
            crop_con[i][u].N_amount = calloc(NF, sizeof(double));
            crop_con[i][u].P_amount = calloc(NF, sizeof(double));
            crop_con[i][u].K_amount = calloc(NF, sizeof(double));
            crop_force[i].DVS_point[u] = calloc(NF, sizeof(double));
            crop_force[i].N_amount[u] = calloc(NF, sizeof(double));
            crop_force[i].P_amount[u] = calloc(NF, sizeof(double));
            crop_force[i].K_amount[u] = calloc(NF, sizeof(double));
            crop_con[i][u].tsum1 = rd();
            crop_con[i][u].tsum2 = rd();
            for (f = 0; f < NF; f++) {
                crop_con[i][u].DVS_point[f] = rd();
                crop_con[i][u].N_amount[f] = rd();
                crop_con[i][u].P_amount[f] = rd();
                crop_con[i][u].K_amount[f] = rd();
            }
            {
                double nmins = rd(), nrec = rd(), pmins = rd(), prec = rd();
                double kmins = rd(), krec = rd();
                if (!act) {
                    continue;
                }
                crop_con_map[i].cidx[k] = u;
                for (b = 0; b < B; b++) {
                    SimUnit *g = calloc(1, sizeof(SimUnit));
                    g->crp = calloc(1, sizeof(Plant));
                    g->ste = calloc(1, sizeof(Field));
                    g->mng = calloc(1, sizeof(Management));
                    g->soil = calloc(1, sizeof(Soil));
                    g->met = calloc(1, sizeof(Meteo));
                    init_unit(g, &tp[k]);
                    g->met->crop_class = k;
                    g->met->Latitude = lat;
                    g->start.month = (unsigned short) smon;
                    g->start.day = (unsigned short) sday;
                    g->start.day_in_year = (unsigned short) sdoy;
                    g->end.day_in_year = (unsigned short) edoy;
                    /* wofost_init.c:376-497 (DIST_MIN values, or the
                       management file ones, chosen by the caller) */
                    g->mng->N_Mins = nmins;
                    g->mng->NRecoveryFrac = nrec;
                    g->mng->P_Mins = pmins;
                    g->mng->PRecoveryFrac = prec;
                    g->mng->K_Mins = kmins;
                    g->mng->KRecoveryFrac = krec;
                    /* wofost_init.c:511-525 */
                    if (!plugin_options.WOFOST_CONTINUE) {
                        if (g->start.day_in_year < g->end.day_in_year) {
                            g->met->CycleLength = g->end.day_in_year -
                                                  g->start.day_in_year;
                        }
                        else {
                            g->met->CycleLength = DAYS_PER_YEAR -
                                                  g->start.day_in_year +
                                                  g->end.day_in_year;
                        }
                    }
                    else {
                        g->met->CycleLength = DAYS_PER_YEAR - 10;
                    }
                    unit[i][b][k] = g;
                    if (Grid[i][b] == NULL) {
                        Grid[i][b] = g;
                    }
                    else {
                        SimUnit *q = Grid[i][b];
                        while (q->next) {
                            q = q->next;
                        }
                        q->next = g;
                    }
                }
                u++;
            }
        }
        for (k = u; k < NC; k++) {
            crop_con_map[i].cidx[k] = NODATA_VEG;
        }
        /* cidx must be per class: rebuild it (crop_init.c:93-113) */
        u = 0;
        for (k = 0; k < NC; k++) {
            if (active[i][k]) {
                crop_con_map[i].cidx[k] = u;
                crop_con_map[i].veg_class[k] = k;
                u++;
            }
            else {
                crop_con_map[i].cidx[k] = NODATA_VEG;
                crop_con_map[i].veg_class[k] = k;
            }
        }
        crop_con_map[i].nc_active = u;
    }

    dmy = calloc(T, sizeof(dmy_struct));
    snprintf(path, sizeof(path), "%s.names", outp);
    dump_names(path);
    fo = fopen(outp, "wb");
    snprintf(path, sizeof(path), "%s.leaves", outp);
    fl = fopen(path, "wb");
    for (t = 0; t < T; t++) {
        current = t;
        dmy[t].year = rdi();
        dmy[t].month = rdi();
        dmy[t].day = rdi();
        dmy[t].day_in_year = rdi();
        dmy[t].dayseconds = 0;
        for (i = 0; i < N; i++) {
            force[i].air_temp[0] = rd();
            force[i].shortwave[0] = rd();
            co2_force[i].CO2 = rd();
            for (b = 0; b < B; b++) {
                for (k = 0; k < NC; k++) {
                    all_vars[i].cell[k][b].pot_evap = rd();
                    all_vars[i].cell[k][b].water_stress = rd();
                }
            }
            for (k = 0; k < NC; k++) {
                double t1 = rd(), t2 = rd();
                int    c = crop_con_map[i].cidx[k];
                if (c != NODATA_VEG) {
                    crop_force[i].tsum1[c] = t1;
                    crop_force[i].tsum2[c] = t2;
                }
                for (f = 0; f < NF; f++) {
                    double v0 = rd(), v1 = rd(), v2 = rd(), v3 = rd();
                    if (c != NODATA_VEG) {
                        crop_force[i].DVS_point[c][f] = v0;
                        crop_force[i].N_amount[c][f] = v1;
                        crop_force[i].P_amount[c][f] = v2;
                        crop_force[i].K_amount[c][f] = v3;
                    }
                }
            }
        }
        /* U: plugin_update_step_vars (crop_update_step_vars, the feedback,
           only reads the crop states and belongs to the crop feedback) */
        for (i = 0; i < N; i++) {
            wofost_update_step_vars(i);
        }
        /* P3 */
        for (i = 0; i < N; i++) {
            crop_run(i);
        }
        for (i = 0; i < N; i++) {
            for (b = 0; b < B; b++) {
                for (k = 0; k < NC; k++) {
                    int    c = crop_con_map[i].cidx[k];
                    double t1 = c != NODATA_VEG ? crop_con[i][c].tsum1 : NAN;
                    double t2 = c != NODATA_VEG ? crop_con[i][c].tsum2 : NAN;
                    dump_unit(unit[i][b][k], fo, t1, t2);
                    if (leafevery > 0 && (t % leafevery == leafevery - 1)) {
                        dump_leaves(unit[i][b][k], fl, t,
                                    (i * B + b) * NC + k);
                    }
                }
            }
        }
    }
    fclose(fo);
    fclose(fl);
    return 0;
}

/* ------------------------------------------------------------- mode 2 */
/* Function records.  Each record: kind, then the inputs of that kind; the
   outputs are written in the same order.  See oracle.py. */
static int
function_run(const char *outp)
{
    int         nrec = rdi(), P = rdi(), r, j, i;
    FILE       *fo = fopen(outp, "wb");
    template_t  tp;
    SimUnit     g;
    Plant       crp;
    Field       ste;
    Management  mng;
    Soil        soil;
    Meteo       met;
    double      out[64];
    int         no;

    memset(&tp, 0, sizeof(tp));
    for (j = 0; j < NR_VARIABLES_CRP; j++) {
        tp.var[j] = rd();
    }
    for (j = 0; j < NR_TABLES_CRP; j++) {
        tp.tab[j] = rd_table(P);
    }
    for (j = 0; j < NR_VARIABLES_MANAGEMENT; j++) {
        tp.mvar[j] = rd();
    }
    plugin_global_param.wofost_steps_per_day = 1;
    plugin_options.WOFOST_PTEMP = rdi();
    memset(&g, 0, sizeof(g));
    g.crp = &crp;
    g.ste = &ste;
    g.mng = &mng;
    g.soil = &soil;
    g.met = &met;
    for (r = 0; r < nrec; r++) {
        int kind = rdi();
        init_unit(&g, &tp);
        no = 0;
        if (kind == 0) {
            /* Afgen of table j at x */
            int   tj = rdi();
            float x = rd();
            TABLE *tb = tp.tab[tj];
            out[no++] = Afgen(tb, &x);
        }
        else if (kind == 1) {
            /* Astro + DailyTotalAssimilation + Correct */
            met.MeteoDay = rdi();
            met.Latitude = rd();
            met.Radiation = rd();
            met.DayTemp = rd();
            met.CO2 = rd();
            crp.st.Development = rd();
            crp.st.LAI = rd();
            crp.GrowthDay = rdi();
            for (i = 0; i < DAYS_PER_WEEK; i++) {
                met.Tmin[i] = rd();
            }
            Astro(&g);
            out[no++] = met.Daylength;
            out[no++] = met.PARDaylength;
            out[no++] = met.SinLD;
            out[no++] = met.CosLD;
            out[no++] = met.DSinBE;
            out[no++] = met.AngotRadiation;
            out[no++] = met.AtmosphTransm;
            out[no++] = met.DiffRadPP;
            {
                float a = DailyTotalAssimilation(&g);
                out[no++] = a;
                out[no++] = met.TempStress;
                out[no++] = Correct(&g, a);
                out[no++] = met.TminStress;
            }
        }
        else if (kind == 2) {
            /* Partioning */
            soil.WaterStress = rd();
            crp.N_st.Indx = rd();
            crp.st.Development = rd();
            Partioning(&g);
            out[no++] = crp.fac_ro;
            out[no++] = crp.fac_lv;
            out[no++] = crp.fac_st;
            out[no++] = crp.fac_so;
        }
        else if (kind == 3) {
            /* DyingLeaves on a random leaf list */
            int    n = rdi();
            Green *head = NULL, *tl = NULL, *l;
            crp.st.leaves = rd();
            crp.st.LAI = rd();
            crp.st.Development = rd();
            soil.WaterStress = rd();
            crp.NPK_Indx = rd();
            for (i = 0; i < n; i++) {
                Green *node = malloc(sizeof(Green));
                node->weight = rd();
                node->age = rd();
                node->area = rd();
                node->next = NULL;
                if (head == NULL) {
                    head = node;
                }
                else {
                    tl->next = node;
                }
                tl = node;
            }
            crp.LeaveProperties = head;
            out[no++] = DyingLeaves(&g);
            i = 0;
            for (l = crp.LeaveProperties; l; l = l->next) {
                i++;
            }
            out[no++] = i;
            out[no++] = crp.LeaveProperties ? crp.LeaveProperties->weight :
                        NAN;
            Harvest(&crp.LeaveProperties);
        }
        else if (kind == 4) {
            /* NutrientMax/Optimum + NutritionINDX */
            crp.st.Development = rd();
            crp.st.leaves = rd();
            crp.st.stems = rd();
            crp.N_st.leaves = rd();
            crp.N_st.stems = rd();
            crp.P_st.leaves = rd();
            crp.P_st.stems = rd();
            crp.K_st.leaves = rd();
            crp.K_st.stems = rd();
            crp.N_rt.Uptake_lv = rd();
            crp.N_rt.Uptake_st = rd();
            crp.P_rt.Uptake_lv = rd();
            crp.P_rt.Uptake_st = rd();
            crp.K_rt.Uptake_lv = rd();
            crp.K_rt.Uptake_st = rd();
            NutrientMax(&g);
            NutrientOptimum(&g);
            NutritionINDX(&g);
            out[no++] = crp.N_st.Indx;
            out[no++] = crp.P_st.Indx;
            out[no++] = crp.K_st.Indx;
            out[no++] = crp.NPK_Indx;
            out[no++] = crp.NutrientStress;
        }
        fwrite(&no, sizeof(int), 1, fo);
        fwrite(out, sizeof(double), no, fo);
    }
    fclose(fo);
    return 0;
}

int
main(int argc, char **argv)
{
    int mode;
    if (argc != 3) {
        fprintf(stderr, "usage: %s input output\n", argv[0]);
        return 2;
    }
    fin = fopen(argv[1], "rb");
    if (fin == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    mode = rdi();
    if (mode == 1) {
        return season_run(argv[2]);
    }
    if (mode == 2) {
        return function_run(argv[2]);
    }
    fprintf(stderr, "unknown mode %d\n", mode);
    return 2;
}
