/******************************************************************************
 * Oracle for the CropSyst crop backend (E6 thermal time and phenology).
 *
 * The harness is linked against the COMPILED REFERENCE: the VIC-CropSyst objects and
 * libraries of the CPU build (<data root>/build/vic_cropsyst, same flags, -O0 -g). These are
 * the real reference objects, driven day by day:
 *   - CropSyst::Thermal_time_daily_linear / _nonlinear (thermal_time_daily.cpp:22-86) with
 *     their Vernalization and Photoperiodization limits (thermal_time_common.cpp:54-130,
 *     158-272, 346-363), as owned by Crop_complete (crop_cropsyst.cpp:265-295);
 *   - CropSyst::Phenology_2013 with its Period_thermal accrescence, culminescence,
 *     senescence and root elongation periods (phenology_2013.cpp, phenology_A.cpp:79-150,
 *     172-375) and its own Vernalization and Photoperiodization (crop_cropsyst.cpp:297-318);
 *   - CropSyst::calc_stress_adjusted_temperature (crop_cropsyst.cpp:119-141).
 *
 * The day ORDER and the event tests live inside Crop_complete, which cannot be built
 * alone (it needs a soil, a weather provider, a canopy and an event scheduler). They are
 * written here from the cited lines and are marked TRANSCRIBED:
 *   - Crop_complete::start_day / update_temperatures order (crop_cropsyst.cpp:1938-1948,
 *     2638-2672): stress temperature, crop thermal time, phenology.start_day, events;
 *   - Crop_complete::thermal_time_event and thermal_time_matchs (crop_cropsyst.cpp:2957-3015)
 *     with the initiate_* bodies reduced to their phenology calls (crop_cropsyst.cpp:2269-2420)
 *     and create_roots' root elongation period (crop_cropsyst.cpp:1534-1535);
 *   - the parameter adjustment event (event_scheduler_crop.cpp:120-150): the thermal time
 *     temperature is overwritten on (synchronised stage day + offset) after the day's
 *     accumulation (Crop_common::start_day runs last, crop_cropsyst.cpp:1997-2001);
 *   - Crop_complete::end_day order (crop_cropsyst.cpp:2026-2029).
 * The LAI-based canopy's know_senescence(0) returns false (canopy_growth_LAI_based.h:235-236),
 * so Crop_complete::culminate_senescence never reaches phenology.culminate_senescence:
 * the senescence period stays, expired ("senescence(expired)" in the crop csv).
 *
 * stdin, one case per line (reals as C99 hex floats or decimals):
 *   N nonlinear base cutoff opt max emergence tuberization flowering filling maturity
 *   dur_acc dur_culm dur_sen dur_root maturity_significant has_leaf_senescence is_tuber
 *   fruit_harvest vern_enabled vern_low vern_high vern_start vern_end vern_min
 *   photoperiodism stop_day_length unconstrained_day_length sens_water_stress
 *   n_adj [stage offset param value] * n_adj
 *   then N days: tmax tmin day_length wsi vpd_max senescence_reduction_factor
 * stdout, one line per day (C99 hex floats, %a; integers as decimals):
 *   tmax_adj gdd GDDs GDDs_yesterday growth_stage
 *   [present tt rel_elapsed rel_remaining expired] for accrescence culminescence senescence root
 *   vern_factor_crop vern_factor_phenology season_duration_days
 *   init day for PLANTING EMERGENCE ACCRESCENCE END_CANOPY_GROWTH ANTHESIS FILLING MATURITY,
 *   tuber initiation day (day index, -1 if never)
 * A first line "ENUM ..." lists the Normal_crop_event_sequence values used.
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

#include "crop/phenology_2013.h"
#include "crop/thermal_time_daily.h"
#include "crop/crop_param_struct.h"
#include "common/weather/parameter/WP_temperature.h"

namespace CropSyst {
float64 calc_stress_adjusted_temperature                // crop_cropsyst.cpp:119 (no header)
   (float64, float64, float64, float64, float64, bool, bool);
}
using namespace CropSyst;

namespace {
double rd(char *&s) {
   char *e; double v = strtod(s, &e);
   if (e == s) { fprintf(stderr, "parse error\n"); exit(2); }
   s = e; return v;
}
// date32 = year*1000 + doy, day 0 = 2001001 (2001 is not a leap year, nor is 2002)
CORN::date32 date_of(int d) {
   int y = 2001, doy = d + 1;
   while (doy > 365) { doy -= 365; y++; }
   return (CORN::date32)(y * 1000 + doy);
}
int day_of(CORN::date32 dt) {
   if (dt == 0) return -1;
   int y = dt / 1000, doy = dt % 1000;
   return (y - 2001) * 365 + doy - 1;
}
void period_out(const Phenology_abstract::Period_thermal *p) {
   if (!p) { printf("0 0x0p+0 0x0p+0 0x0p+0 0 "); return; }
   printf("1 %a %a %a %d ", p->get_thermal_time_accum(), p->get_thermal_time_relative_elapsed(),
          (double)p->get_thermal_time_relative_remaining(), (int)p->has_expired());
}
// Phenology_2013 keeps the period pointers protected; this subclass only reads the root
// elongation period, which has no public getter in the 2013 build.
class Phenology_probe : public Phenology_2013 {
 public:
   using Phenology_2013::Phenology_2013;
   const Period_thermal *root() const { return root_elongation; }
};
}  // namespace

int main() {
   printf("ENUM %d %d %d %d %d %d %d %d %d %d\n", (int)NGS_NONE, (int)NGS_PLANTING,
          (int)NGS_GERMINATION, (int)NGS_EMERGENCE, (int)NGS_ACCRESCENCE,
          (int)NGS_END_CANOPY_GROWTH, (int)NGS_ANTHESIS, (int)NGS_FILLING, (int)NGS_MATURITY,
          (int)NGS_QUIESCENCE);
   static char line[1 << 20];
   while (fgets(line, sizeof line, stdin)) {
      char *s = line;
      int N = (int)rd(s);
      // parameters, float32 as in the reference structs (crop_param_struct.h:64-75, 147-160,
      // 420-435)
      Crop_parameters_struct::Thermal_time ttp;
      ttp.thermal_response = ((int)rd(s)) ? nonlinear : linear;
      ttp.temperature.base = (float32)rd(s);
      ttp.temperature.cutoff = (float32)rd(s);
      ttp.temperature.opt = (float32)rd(s);
      ttp.temperature.max = (float32)rd(s);
      float32 emergence = (float32)rd(s), tuberization = (float32)rd(s);
      float32 flowering = (float32)rd(s), filling = (float32)rd(s), maturity = (float32)rd(s);
      float32 dur_acc = (float32)rd(s), dur_culm = (float32)rd(s), dur_sen = (float32)rd(s);
      float32 dur_root = (float32)rd(s);
      bool maturity_significant = (int)rd(s), has_leaf_sen = (int)rd(s);
      bool is_tuber = (int)rd(s), fruit_harvest = (int)rd(s);
      Crop_parameters_struct::Vernalization vp;
      vp.enabled = (int)rd(s);
      vp.low_temp = (float32)rd(s); vp.high_temp = (float32)rd(s);
      vp.start = (float32)rd(s); vp.end = (float32)rd(s); vp.min_factor = (float32)rd(s);
      Crop_parameters_struct::Photoperiod pp;
      pp.photoperiodism = (Photoperiodism)(int)rd(s);
      pp.stop_day_length = (float32)rd(s);
      pp.unconstrained_day_length = (float32)rd(s);
      float32 sens = (float32)rd(s);
      int n_adj = (int)rd(s);
      std::vector<int> adj_stage(n_adj), adj_off(n_adj), adj_par(n_adj);
      std::vector<float32> adj_val(n_adj);
      std::vector<bool> adj_done(n_adj, false);     // reschedule=no: an event fires once
      for (int a = 0; a < n_adj; a++) {
         adj_stage[a] = (int)rd(s); adj_off[a] = (int)rd(s); adj_par[a] = (int)rd(s);
         adj_val[a] = (float32)rd(s);
      }
      // phenologic periods composed as crop_param_class.cpp:975-1019 (all share ttp)
      Crop_parameters_struct::Phenology php;
      php.initiation.emergence = emergence;
      php.initiation.tuberization = tuberization;
      php.initiation.flowering = flowering;
      php.initiation.filling = filling;
      php.initiation.maturity = maturity;
      php.maturity_significant = maturity_significant;
      nat8 relevant = (vp.enabled ? RELEVANT_VERNALIZATION : 0)
                    | ((pp.photoperiodism != no_photoperiod) ? RELEVANT_PHOTOPERIOD : 0);
      php.accrescence_period = new Crop_parameters_struct::Phenologic_period
         ("phenology/accrescence", "", Crop_parameters_struct::Phenologic_period::initiation_basis,
          emergence, dur_acc, true, relevant, &ttp);
      php.culminescence_period = new Crop_parameters_struct::Phenologic_period
         ("phenology/culminescence", "phenology/accrescence",
          Crop_parameters_struct::Phenologic_period::culmination_basis, 0, dur_culm, true,
          relevant, &ttp);
      php.senescence_period = new Crop_parameters_struct::Phenologic_period
         ("phenology/senescence", "phenology/culminescence",
          Crop_parameters_struct::Phenologic_period::culmination_basis, 0, dur_sen, true,
          relevant, &ttp);
      php.root_elongation_period = new Crop_parameters_struct::Phenologic_period
         ("phenology/root_elongation", "phenology/accrescence",
          Crop_parameters_struct::Phenologic_period::initiation_basis, 0, dur_root, true,
          relevant, &ttp);

      // state the crop owns (crop_cropsyst.cpp:258-318); heap and never deleted, as in ref_cropsyst_uptake
      float64 stress_adj_temp = 0.0, tmin = 0.0, day_length = 12.0;
      CS::Temperature &tavg = *new CS::Temperature(0.0);
      CORN::date32 simdate = date_of(0);
      Thermal_time_common::Vernalization *vern_tt = vp.enabled
         ? new Thermal_time_common::Vernalization(vp, tavg) : 0;
      Thermal_time_common::Photoperiodization *photo_tt =
         (pp.photoperiodism > no_photoperiod)
         ? new Thermal_time_common::Photoperiodization(pp, day_length) : 0;
      Thermal_time_common *tt = (ttp.thermal_response == linear)
         ? (Thermal_time_common *)new Thermal_time_daily_linear
              (ttp, stress_adj_temp, tmin, false, vern_tt, photo_tt)
         : (Thermal_time_common *)new Thermal_time_daily_nonlinear
              (ttp, stress_adj_temp, tmin, false, vern_tt, photo_tt);
      Thermal_time_common::Vernalization *vern_ph = vp.enabled
         ? new Thermal_time_common::Vernalization(vp, tavg) : 0;
      Thermal_time_common::Photoperiodization *photo_ph =
         (pp.photoperiodism != no_photoperiod)
         ? new Thermal_time_common::Photoperiodization(pp, day_length) : 0;
      Phenology_probe &ph = *new Phenology_probe
         (php, tt->ref_thermal_time_cumulative(), stress_adj_temp, tmin, vern_ph, photo_ph,
          false, false, simdate);
      bool canopy = false, roots = false;
      std::vector<int> sync_day(NGS_COUNT, -1);
      int tuber_day = -1;

      // planting: Crop_complete::plant / start (crop_cropsyst.cpp:619, 652, 657-658)
      ph.activate_sowing();
      ph.start_season();
      sync_day[NGS_PLANTING] = 0;

      for (int d = 0; d < N; d++) {
         simdate = date_of(d);
         double tmax = rd(s); tmin = rd(s); day_length = rd(s);
         double wsi = rd(s), vpd = rd(s), srf = rd(s);
         // TRANSCRIBED update_temperatures (crop_cropsyst.cpp:2640-2653, 2667)
         tavg.set_Celcius((tmax + tmin) / 2.0, CORN::measured_quality);  // VIC_land_unit_simulation.cpp:252
         stress_adj_temp = calc_stress_adjusted_temperature
            (sens, tmax, srf, wsi, vpd, !ph.is_matured(), has_leaf_sen);
         tt->accumulate_degree_days();
         ph.start_day();                                        // crop_cropsyst.cpp:1947
         // TRANSCRIBED thermal_time_event (crop_cropsyst.cpp:2968-3015)
         float64 y = tt->get_accum_degree_days_yesterday(), t = tt->get_accum_degree_days();
         auto match = [&](float64 x) { return (y <= x) && (t >= x); };
         if (!ph.is_in_quiescence()) {
            if (ph.is_in_germination() && match(emergence)) {
               // initiate_accrescence (crop_cropsyst.cpp:2269-2310)
               canopy = true;
               if (!roots) {                                    // create_roots :1534-1535
                  roots = true;
                  ph.activate_root_elongation()->start_day();
               }
               ph.activate_emergence();
               ph.activate_accrescence();
               sync_day[NGS_EMERGENCE] = d; sync_day[NGS_ACCRESCENCE] = d;
            }
            if (has_leaf_sen && match(tuberization) && is_tuber) {   // :2319-2328
               ph.activate_tuberization();
               tuber_day = d;
            }
            if (canopy) {
               if (flowering && match(flowering)) {             // initiate_flowering :2331-2346
                  sync_day[NGS_ANTHESIS] = d;
                  if (!ph.is_in_yield_formation()) ph.activate_anthesis();
               }
               const Phenology_abstract::Period_thermal *c = ph.get_culminescence_period();
               if (c && c->has_expired()) ph.activate_senescence();   // :2359-2366
               const Phenology_abstract::Period_thermal *a = ph.get_accrescence_period();
               if (a && a->has_expired()) {                     // :2374-2381
                  ph.activate_culminescence();
                  sync_day[NGS_END_CANOPY_GROWTH] = d;
               }
               // senescence expiry: culminate_senescence stops at know_senescence(0)=false
            }
            if (!fruit_harvest && filling && match(filling)) {   // initiate_filling :2383-2396
               ph.activate_yield_formation();
               sync_day[NGS_FILLING] = d;
            }
            if (maturity_significant && match(maturity)) {      // initiate_maturity :2397-2420
               ph.activate_maturity();
               sync_day[NGS_MATURITY] = d;
            }
         }
         // TRANSCRIBED adjustment events (event_scheduler_crop.cpp:120-150)
         for (int a = 0; a < n_adj; a++)
            if (!adj_done[a] && sync_day[adj_stage[a]] >= 0
                && d == sync_day[adj_stage[a]] + adj_off[a]) {
               adj_done[a] = true;
               float32 *slot = adj_par[a] == 0 ? &ttp.temperature.base
                             : adj_par[a] == 1 ? &ttp.temperature.cutoff
                             : adj_par[a] == 2 ? &ttp.temperature.opt : &ttp.temperature.max;
               *slot = adj_val[a];
            }
         // output (before end_day: the day's values)
         printf("%a %a %a %a %d ", stress_adj_temp, tt->get_growing_degree_day(),
                tt->get_accum_degree_days(), tt->get_accum_degree_days_yesterday(),
                (int)ph.growth_stage);
         period_out(ph.get_accrescence_period());
         period_out(ph.get_culminescence_period());
         period_out(ph.get_senescence_period());
         period_out(ph.root());
         printf("%a %a %d ", vern_tt ? vern_tt->get_factor() : 1.0,
                vern_ph ? vern_ph->get_factor() : 1.0, (int)ph.get_season_duration_days());
         const int st[7] = {NGS_PLANTING, NGS_EMERGENCE, NGS_ACCRESCENCE, NGS_END_CANOPY_GROWTH,
                            NGS_ANTHESIS, NGS_FILLING, NGS_MATURITY};
         for (int k = 0; k < 7; k++) printf("%d ", day_of(ph.initialization_date32[st[k]]));
         printf("%d\n", tuber_day);
         // TRANSCRIBED end_day (crop_cropsyst.cpp:2027-2029)
         tt->end_day();
         ph.end_day();
      }
      fflush(stdout);
   }
   _exit(0);
}
