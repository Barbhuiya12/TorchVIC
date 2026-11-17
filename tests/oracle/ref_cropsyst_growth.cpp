/******************************************************************************
 * Oracle for the CropSyst crop backend (E4 potential transpiration and
 * CO2, E5 biomass growth, E7 canopy, E8 roots, E9 yield), driven together with the
 * E6 phenology over whole seasons.
 *
 * Linked against the COMPILED REFERENCE (<data root>/build/vic_cropsyst, -O0 -g, same
 * flags). Real reference objects used day by day:
 *   - Thermal_time_daily_linear/_nonlinear, Vernalization, Phenology_2013 (as in ref_cropsyst_phenology);
 *   - calc_stress_adjusted_temperature (crop_cropsyst.cpp:119-141);
 *   - Stress_adjusted_temperature_estimator / Temperature_estimator hourly
 *     temperatures (crop_cropsyst.h:106-119, temperature_est.cpp,
 *     min_max_to_timestep_estimator.cpp);
 *   - Crop_CO2_response (CO2_response.cpp);
 *   - Biomass_growth_RUE_TUE + Biomass_production_transpiration_dependent_V4
 *     (biomass_growth_RUE_TUE.cpp, transpiration_dependent_growth*.cpp);
 *   - Canopy_growth_leaf_area_index_based_V5 (canopy_growth_LAI_based*.cpp,
 *     canopy_growth_portioned.cpp);
 *   - Crop_root_vital_V5 on a VIC::Soil_layers profile (crop_root*.cpp);
 *   - Yield_grain with Yield_grain_temperature_extreme_response and
 *     calc_source_limited_harvest_index (yield.cpp, harvest_grain_asymptotic.c,
 *     yield_temperature_extreme_response.cpp);
 *   - CORN::statistical::Sample_cumulative water stress statistics.
 * TRANSCRIBED (members of Crop_complete/Crop_common/Land_unit that cannot be built
 * alone): the day order and glue of Land_unit_simulation_VX::start_day ET max
 * (land_unit_sim.cpp:1296-1322, crop_common.cpp update_evapotranspiration_max,
 * crop_cropsyst.cpp:3606-3619), Crop_complete::start_day/update_temperatures/
 * thermal_time_event/initiate_* (crop_cropsyst.cpp:1937-2003, 2638-2672, 2968-3015,
 * 2286-2427, 1514-1548), process_day/actual_growth/calc_act_biomass_growth
 * (crop_cropsyst.cpp:2701-2828, 1622-1738, 2829-2947), process_transpiration_m
 * (crop_common.cpp) with the E1 uptake replaced by act = uptake_frac * limited
 * potential transpiration (E1 is covered by ref_cropsyst_uptake), the grain stress period events
 * (event_scheduler_crop.cpp:96-112), end_day (crop_cropsyst.cpp:2005-2117) and
 * harvest_crop's yield (crop_cropsyst.cpp:1830-1872). Nitrogen is disabled.
 *
 * stdin: "CRP <path>" prints the reference-parsed parameters of a crop file;
 * otherwise one case per line, see tests/test_cropsyst_growth.py (make_cases).
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

#include "crop/phenology_2013.h"
#include "crop/thermal_time_daily.h"
#include "crop/crop_param.h"
#include "crop/crop_cropsyst.h"
#include "crop/canopy_growth_LAI_based_V5.h"
#include "crop/crop_root_V5.h"
#include "crop/biomass_growth_RUE_TUE.h"
#include "crop/transpiration_dependent_growth_V4.h"
#include "crop/CO2_response.h"
#include "crop/yield.h"
#include "agronomic/VIC_soil.h"
#include "common/weather/parameter/WP_temperature.h"
#include "common/weather/parameter/WP_air_temperature.h"
#include "common/weather/parameter/WP_solar_radiation.h"
#include "common/weather/parameter/WP_vapor_pressure_deficit.h"
#include "common/weather/parameter/WP_vapor_pressure_saturation_slope.h"
#include "common/weather/parameter/WP_psychrometric_const.h"
#include "common/biometeorology/parameter/CO2_atmospheric_conc.h"
#include "corn/data_source/vv_file.h"

extern option_struct options;
namespace CropSyst {
float64 calc_stress_adjusted_temperature
   (float64, float64, float64, float64, float64, bool, bool);
}
using namespace CropSyst;

namespace {
double rd(char *&s) {
   char *e; double v = strtod(s, &e);
   if (e == s) { fprintf(stderr, "parse error\n"); exit(2); }
   s = e; return v;
}
CORN::date32 date_of(int d) {
   int y = 2001, doy = d + 1;
   while (doy > 365) { doy -= 365; y++; }
   return (CORN::date32)(y * 1000 + doy);
}
class Phenology_probe : public Phenology_2013 {
 public:
   using Phenology_2013::Phenology_2013;
   const Period_thermal *root() const { return root_elongation; }
};
class Psychro : public Psychrometric_constant {
 public:
   void set(float64 v) { update_kPa_per_C(v); }
};
const unsigned VITAL = 1, EFFETE = 2;

void print_params(const char *path) {
   Crop_parameters &p = *new Crop_parameters(true, false, true, true);
   CORN::VV_File f(path, true);
   f.get(p);
   const auto &b = p.biomass_production;
   printf("%d %a %a %a %a ", (int)b.efficiency_model, (double)b.TUE_at_1kPa_VPD[0],
          (double)b.TUE_scaling_coef[0], (double)b.RUE[0], (double)b.continuous_cultivation_impact_factor);
   printf("%a %a %a %a ", (double)b.temperature_limitation.min, (double)b.temperature_limitation.low_optimum,
          (double)b.temperature_limitation.high_optimum, (double)b.temperature_limitation.max);
   const auto &m = p.morphology;
   printf("%a %a %a %a %a ", (double)p.transpiration.ET_coef, (double)m.light_extinction_coef,
          (double)m.reduce_canopy_expansion_LWP, (double)m.stop_canopy_expansion_LWP, (double)m.max_canopy_height);
   const auto &c = p.canopy_growth_LAI_based;
   printf("%a %a %a %a %a %a ", (double)c.initial_GAI, (double)c.regrowth_GAI, (double)c.max_LAI,
          (double)c.specific_leaf_area, (double)c.fract_LAI_mature, (double)c.stem_leaf_partition);
   const auto &r = p.root;
   printf("%a %a %a %a %a %a %a ", (double)r.max_depth, (double)r.surface_density,
          (double)r.density_distribution_curvature, (double)r.sensitivity_to_water_stress,
          (double)r.length_at_emergence_cm, (double)r.root_shoot_emergence_ratio, (double)r.root_shoot_full_ratio);
   const auto &y = p.yield;
   printf("%a %d %d %a ", (double)y.harvest_index_unstressed, (int)y.apply_adjustment_to_unstressed_harvest_index,
          (int)y.apply_temperature_extreme_response, (double)y.translocation_fraction_max);
   const auto &co = p.CO2_response;
   printf("%a %a %a %a %a ", (double)co.CO2_concentration_when_measured.TUE, (double)co.CO2_concentration_when_measured.RUE,
          (double)co.non_rectagular_hyperbola_solution.alpha, (double)co.non_rectagular_hyperbola_solution.theta,
          (double)co.non_rectagular_hyperbola_solution.growth_ratio_asymptotic);
   int n = 0;
   FOR_EACH_IN(sp, Crop_parameters_class::Yield::Grain_stress_sensitivity_period, y.grain_stress_sensitivity_periods, it)
   {  n++; } FOR_EACH_END(it)
   printf("%d ", n);
   FOR_EACH_IN(sp, Crop_parameters_class::Yield::Grain_stress_sensitivity_period, y.grain_stress_sensitivity_periods, it)
   {  printf("%a %a %a %d %a %a %a %a %d %a ",
         (double)sp->cold.no_stress_threshold_temperature, (double)sp->cold.max_stress_threshold_temperature,
         (double)sp->cold.sensitivity, (int)sp->cold.exposure_initial_hour, (double)sp->cold.exposure_damaging_hours,
         (double)sp->heat.no_stress_threshold_temperature, (double)sp->heat.max_stress_threshold_temperature,
         (double)sp->heat.sensitivity, (int)sp->heat.exposure_initial_hour, (double)sp->heat.exposure_damaging_hours);
   } FOR_EACH_END(it)
   printf("%d %d %d\n", (int)p.has_leaf_senescence(), (int)p.harvested_part_clad.get(),
          (int)(!p.continuation_mode_clad.get() == single_season));
}
}  // namespace

int main() {
   static char line[1 << 22];
   while (fgets(line, sizeof line, stdin)) {
      if (!strncmp(line, "CRP ", 4)) {
         line[strcspn(line, "\n")] = 0;
         print_params(line + 4); fflush(stdout); continue;
      }
      char *s = line;
      int N = (int)rd(s), L = (int)rd(s);
      bool co2_on = (int)rd(s);
      Crop_parameters &p = *new Crop_parameters(false, false, true, true);
      // E6 parameters (float32 structs, as in ref_cropsyst_phenology)
      Crop_parameters_struct::Thermal_time &ttp = p.thermal_time;
      ttp.thermal_response = ((int)rd(s)) ? nonlinear : linear;
      ttp.temperature.base = (float32)rd(s); ttp.temperature.cutoff = (float32)rd(s);
      ttp.temperature.opt = (float32)rd(s);  ttp.temperature.max = (float32)rd(s);
      float32 emergence = (float32)rd(s), flowering = (float32)rd(s);
      float32 filling = (float32)rd(s), maturity = (float32)rd(s);
      float32 dur_acc = (float32)rd(s), dur_culm = (float32)rd(s), dur_sen = (float32)rd(s);
      float32 dur_root = (float32)rd(s);
      Crop_parameters_struct::Vernalization &vp = p.vernalization;
      vp.enabled = (int)rd(s);
      vp.low_temp = (float32)rd(s); vp.high_temp = (float32)rd(s);
      vp.start = (float32)rd(s); vp.end = (float32)rd(s); vp.min_factor = (float32)rd(s);
      p.photoperiod.photoperiodism = no_photoperiod;
      float32 sens = (float32)rd(s);
      // E4/E5/E7 parameters
      auto &bp = p.biomass_production;
      bp.efficiency_model_raw = "TUE"; bp.efficiency_model = TUE_efficiency; bp.WUE_changes = false;
      bp.TUE_at_1kPa_VPD[0] = (float32)rd(s); bp.TUE_scaling_coef[0] = (float32)rd(s);
      bp.RUE[0] = bp.RUE[1] = (float32)rd(s);
      bp.temperature_limitation.min = (float32)rd(s); bp.temperature_limitation.low_optimum = (float32)rd(s);
      bp.temperature_limitation.high_optimum = (float32)rd(s); bp.temperature_limitation.max = (float32)rd(s);
      bp.continuous_cultivation_impact_factor = 1.0;
      p.transpiration.ET_coef = (float32)rd(s);
      p.morphology.light_extinction_coef = (float32)rd(s);
      p.morphology.reduce_canopy_expansion_LWP = (float32)rd(s);
      p.morphology.stop_canopy_expansion_LWP = (float32)rd(s);
      p.morphology.max_canopy_height = (float32)rd(s);
      auto &cp = p.canopy_growth_LAI_based;
      cp.initial_GAI = (float32)rd(s); cp.regrowth_GAI = (float32)rd(s); cp.max_LAI = (float32)rd(s);
      cp.specific_leaf_area = (float32)rd(s); cp.fract_LAI_mature = (float32)rd(s);
      cp.stem_leaf_partition = (float32)rd(s);
      // E8
      auto &rp = p.root;
      rp.max_depth = (float32)rd(s); rp.surface_density = (float32)rd(s);
      rp.density_distribution_curvature = (float32)rd(s); rp.sensitivity_to_water_stress = (float32)rd(s);
      rp.length_at_emergence_cm = (float32)rd(s); rp.root_shoot_emergence_ratio = (float32)rd(s);
      rp.root_shoot_full_ratio = (float32)rd(s);
      // E9
      float32 HI = (float32)rd(s); bool adjust = (int)rd(s); bool extreme = (int)rd(s);
      float32 transl = (float32)rd(s);
      // CO2
      auto &co = p.CO2_response;
      co.CO2_concentration_when_measured.TUE = (float32)rd(s);
      co.CO2_concentration_when_measured.RUE = (float32)rd(s);
      co.non_rectagular_hyperbola_solution.alpha = (float32)rd(s);
      co.non_rectagular_hyperbola_solution.theta = (float32)rd(s);
      co.non_rectagular_hyperbola_solution.growth_ratio_asymptotic = (float32)rd(s);
      float64 co2_ppm = rd(s);
      int n_gsp = (int)rd(s);
      std::vector<int> g_stage(n_gsp), g_off(n_gsp);
      std::vector<bool> g_done(n_gsp, false);
      std::vector<Crop_parameters_struct::Temperature_sensitivity *> g_par(n_gsp);
      for (int g = 0; g < n_gsp; g++) {
         g_stage[g] = (int)rd(s); g_off[g] = (int)rd(s);
         auto *t = new Crop_parameters_struct::Temperature_sensitivity;
         t->cold.no_stress_threshold_temperature = (float32)rd(s);
         t->cold.max_stress_threshold_temperature = (float32)rd(s);
         t->cold.sensitivity = (float32)rd(s);
         t->cold.exposure_initial_hour = (int16)rd(s);
         t->cold.exposure_damaging_hours = (float32)rd(s);
         t->heat.no_stress_threshold_temperature = (float32)rd(s);
         t->heat.max_stress_threshold_temperature = (float32)rd(s);
         t->heat.sensitivity = (float32)rd(s);
         t->heat.exposure_initial_hour = (int16)rd(s);
         t->heat.exposure_damaging_hours = (float32)rd(s);
         g_par[g] = t;
      }
      // soil profile (VIC_land_unit_simulation.cpp:200-222)
      options.Nlayer = L;
      VIC::Soil_layers &layers = *new VIC::Soil_layers;
      for (int l = 1; l <= L; l++) layers.set_thickness_m(l, rd(s));
      layers.set_thickness_provided(true);
      layers.set_depth_by_thickness(L);
      const bool has_leaf_sen = true;    // annual crops (crop_param_class.cpp:595-600)

      // phenologic periods (crop_param_class.cpp:975-1019)
      Crop_parameters_struct::Phenology &php = p.phenology;
      php.initiation.emergence = emergence; php.initiation.tuberization = 99999.0;
      php.initiation.flowering = flowering; php.initiation.filling = filling;
      php.initiation.maturity = maturity; php.maturity_significant = true;
      nat8 relevant = (vp.enabled ? RELEVANT_VERNALIZATION : 0);
      php.accrescence_period = new Crop_parameters_struct::Phenologic_period
         ("phenology/accrescence", "", Crop_parameters_struct::Phenologic_period::initiation_basis,
          emergence, dur_acc, true, relevant, &ttp);
      php.culminescence_period = new Crop_parameters_struct::Phenologic_period
         ("phenology/culminescence", "phenology/accrescence",
          Crop_parameters_struct::Phenologic_period::culmination_basis, 0, dur_culm, true, relevant, &ttp);
      php.senescence_period = new Crop_parameters_struct::Phenologic_period
         ("phenology/senescence", "phenology/culminescence",
          Crop_parameters_struct::Phenologic_period::culmination_basis, 0, dur_sen, true, relevant, &ttp);
      php.root_elongation_period = new Crop_parameters_struct::Phenologic_period
         ("phenology/root_elongation", "phenology/accrescence",
          Crop_parameters_struct::Phenologic_period::initiation_basis, 0, dur_root, true, relevant, &ttp);

      // weather the crop references
      float64 stress_adj_temp = 0.0, tmin = 0.0, day_length = 12.0;
      CS::Temperature &tavg_v = *new CS::Temperature(0.0);
      Air_temperature_average &air_avg = *new Air_temperature_average;
      Solar_radiation &srad = *new Solar_radiation;
      Vapor_pressure_saturation_slope &slope = *new Vapor_pressure_saturation_slope;
      Psychro &psychro = *new Psychro;
      CS::CO2_atmospheric_concentration &co2c = *new CS::CO2_atmospheric_concentration;
      Vapor_pressure_deficit_daytime &vpd_day = *new Vapor_pressure_deficit_daytime;
      float64 ra = 0.001;
      co2c.set_ppm(co2_ppm, CORN::measured_quality);
      CORN::date32 simdate = date_of(0);

      Thermal_time_common::Vernalization *vern_tt = vp.enabled
         ? new Thermal_time_common::Vernalization(vp, tavg_v) : 0;
      Thermal_time_common *tt = (ttp.thermal_response == linear)
         ? (Thermal_time_common *)new Thermal_time_daily_linear(ttp, stress_adj_temp, tmin, false, vern_tt, 0)
         : (Thermal_time_common *)new Thermal_time_daily_nonlinear(ttp, stress_adj_temp, tmin, false, vern_tt, 0);
      Thermal_time_common::Vernalization *vern_ph = vp.enabled
         ? new Thermal_time_common::Vernalization(vp, tavg_v) : 0;
      Phenology_probe &ph = *new Phenology_probe
         (php, tt->ref_thermal_time_cumulative(), stress_adj_temp, tmin, vern_ph, 0, false, false, simdate);

      // crop state (crop_cropsyst.cpp:155-360), created on the sowing day = day 0
      Crop_CO2_response *CO2 = 0;
      if (co2_on) {                                   // perform_op.cpp:368-385
         CO2 = new Crop_CO2_response(co, slope, psychro, ra, co2c);
         CO2->update_with_new_atmphospheric_CO2();
      }
      float64 CO2_TUE_adjustment = 1.0;
      bool before_WUE_change = false;
      Biomass_production_transpiration_dependent_V4 &tdg = *new
         Biomass_production_transpiration_dependent_V4(p, vpd_day, CO2_TUE_adjustment, before_WUE_change);
      CORN::statistical::Sample_cumulative<float64,float64> &ws_stats =
         *new CORN::statistical::Sample_cumulative<float64,float64>(STAT_arithmetic_mean_bit);
      CORN::statistical::Sample_cumulative<float64,float64> &ts_stats =
         *new CORN::statistical::Sample_cumulative<float64,float64>(STAT_arithmetic_mean_bit);
      Stress_adjusted_temperature_estimator &est = *new Stress_adjusted_temperature_estimator(60);
      CORN::Dynamic_array<float32> &hourly = *new CORN::Dynamic_array<float32>(24);
      CORN::Local_time &sun_rise = *new CORN::Local_time;
      Yield_grain &yield = *new Yield_grain(HI, adjust, extreme, transl, 0, 0, ph, sun_rise, hourly);
      Canopy_growth_leaf_area_index_based_V5 *canopy = 0;
      Crop_root_vital_V5 *roots = 0;
      Biomass_growth_RUE_TUE *bg = 0;
      bool transpiration = false;
      nat8 emergence_count = 0;
      float64 pot_unadj = 0, pot_co2 = 0, intercepted = 0, reported_peak_LAI = 0;
      float64 plant_height = 0, overall = 0, attainable = 0, limited = 0, act_tr = 0;
      float64 bm_fill = 0, bm_mat = 0;
      std::vector<int> sync_day(NGS_COUNT, -1);

      // sowing (crop_cropsyst.cpp:646-658, 1887-1905)
      ph.activate_sowing(); ph.start_season();
      sync_day[NGS_PLANTING] = 0;
      ws_stats.clear(1.0); ts_stats.clear(1.0);
      bool first = true;

      for (int d = 0; d < N; d++) {
         simdate = date_of(d);
         double tmax = rd(s); tmin = rd(s);
         double tmin_tom = rd(s), sr = rd(s), vpdmax = rd(s), et_ref = rd(s);
         double delta = rd(s), gamma = rd(s); ra = rd(s);
         double ufrac = rd(s), lwp = rd(s); int sunrise_h = (int)rd(s);
         if (first) {   // constructor: hourly estimator reset (crop_cropsyst.cpp:378-387)
            est.reset(tmax, tmax, tmin, tmin_tom); first = false;
         }
         air_avg.set_Celcius((tmax + tmin) / 2.0, CORN::measured_quality);
         srad.set_MJ_m2(sr, CORN::measured_quality);
         // Vapor_pressure_deficit_daytime_calculator (WC_vapor_pressure_deficit.cpp:183-191)
         float64 vday = vpdmax * 0.666666; if (vday <= 0.0) vday = 0.00001;
         vpd_day.update_value_qualified_code(vday, CORN::measured_quality);
         slope.set_kPa_per_C(delta, CORN::measured_quality);
         psychro.set(gamma);
         sun_rise.set_S((CORN::Seconds)(sunrise_h * 3600));

         // TRANSCRIBED Land_unit_simulation_VX::start_day ET max (land_unit_sim.cpp:1296-1322)
         // Crop_common::update_evapotranspiration_max, crop exists from sowing
         float64 adj = (p.transpiration.ET_coef == 0.0) ? 1.0
                     : canopy ? canopy->calc_adjusted_ET_crop_coefficient() : 1.0;
         float64 ET_max = et_ref * adj;
         if (ET_max < 0.0) ET_max = 0.0;
         pot_unadj = (canopy ? canopy->get_interception_global_green() : 0.0) * ET_max;
         pot_co2 = pot_unadj;

         // TRANSCRIBED Crop_complete::start_day (crop_cropsyst.cpp:1937-2003)
         overall = 1.0; attainable = 0.0; act_tr = 0.0;
         tavg_v.set_Celcius((tmax + tmin) / 2.0, CORN::measured_quality);
         float64 srf = canopy ? canopy->get_senescence_reduction_factor() : 1.0;
         stress_adj_temp = calc_stress_adjusted_temperature
            (sens, tmax, srf, ws_stats.index, vpdmax, !ph.is_matured(), has_leaf_sen);
         est.set_max_today_min_tomorrow(tmax, tmin_tom);
         est.set_stress_adjusted_temperature(stress_adj_temp);
         est.get_temperature_by_interval(hourly);
         tt->accumulate_degree_days();
         ph.start_day();
         float64 y = tt->get_accum_degree_days_yesterday(), t = tt->get_accum_degree_days();
         auto match = [&](float64 x) { return (y <= x) && (t >= x); };
         if (!ph.is_in_quiescence()) {
            if (ph.is_in_germination() && match(emergence)) {
               // initiate_accrescence (crop_cropsyst.cpp:2286-2325)
               emergence_count++;
               transpiration = true;
               bg = new Biomass_growth_RUE_TUE(false, &p, ph.growth_stage, emergence_count, pot_co2, tt, CO2,
                                               canopy, air_avg, srad, intercepted, tdg, ts_stats);
               bg->start();
               delete canopy;
               canopy = new Canopy_growth_leaf_area_index_based_V5(p, cp, 0, 0, reported_peak_LAI, CO2);
               canopy->initialize();
               canopy->start();
               if (!roots) {   // create_roots (crop_cropsyst.cpp:1514-1548)
                  float64 initial_root_length = cm_to_m(rp.length_at_emergence_cm);
                  Phenology::Period_thermal *rep = ph.activate_root_elongation();
                  rep->start_day();
                  roots = new Crop_root_vital_V5(rp, *rep, layers, initial_root_length);
               }
               float64 initial_biomass = cp.initial_GAI / cp.specific_leaf_area;
               canopy->restart_with(initial_biomass, cp.initial_GAI, true);
               roots->know_above_ground_biomass(initial_biomass);
               ph.activate_emergence();
               ph.activate_accrescence();
               canopy->know_accrescence(ph.get_accrescence_period());
               sync_day[NGS_EMERGENCE] = d; sync_day[NGS_ACCRESCENCE] = d;
            }
            if (canopy) {
               if (flowering && match(flowering)) {       // :2339-2357
                  sync_day[NGS_ANTHESIS] = d;
                  if (!ph.is_in_yield_formation()) ph.activate_anthesis();
                  yield.know_flowering();
               }
               const Phenology_abstract::Period_thermal *c = ph.get_culminescence_period();
               if (c && c->has_expired()) {               // initiate_senescence :2358-2367
                  ph.activate_senescence();
                  canopy->know_senescence(ph.get_senescence_period());
               }
               const Phenology_abstract::Period_thermal *a = ph.get_accrescence_period();
               if (a && a->has_expired()) {               // initiate_culminescence :2373-2383
                  ph.activate_culminescence();
                  canopy->know_culminescence(ph.get_culminescence_period());
                  sync_day[NGS_END_CANOPY_GROWTH] = d;
               }
               const Phenology_abstract::Period_thermal *sp = ph.get_senescence_period();
               if (sp && sp->has_expired())               // culminate_senescence :2368-2372
                  canopy->know_senescence(0);             // pointer cleared; phenology call not reached
            }
            if (filling && match(filling)) {              // initiate_filling :2384-2398
               bm_fill = canopy ? canopy->get_biomass_current(VITAL) : 0.0;
               yield.know_biomass_at_filling(bm_fill);
               sync_day[NGS_FILLING] = d;
               ph.activate_yield_formation();
               yield.know_filling();
            }
            if (match(maturity)) {                        // initiate_maturity :2399-2427
               bm_mat = canopy ? canopy->get_biomass_current(VITAL) : 0.0;
               yield.know_biomass_at_maturity(bm_mat);
               sync_day[NGS_MATURITY] = d;
               delete bg; bg = 0;
               ph.activate_maturity();
               if (canopy) canopy->know_maturity_initiation();
            }
         }
         if (CO2) CO2->update();                          // :1959-1964
         if (roots) roots->start_day();                   // :1971
         before_WUE_change = !has_leaf_sen
            || (bp.WUE_changes && (tt->get_accum_degree_days() <= p.phenology.WUE_change));
         // grain stress period events (event_scheduler_crop.cpp:96-112)
         for (int g = 0; g < n_gsp; g++)
            if (!g_done[g] && sync_day[g_stage[g]] >= 0 && d == sync_day[g_stage[g]] + g_off[g]) {
               g_done[g] = true;
               if (yield.temperature_extreme_reponse) yield.temperature_extreme_reponse->add_period(*g_par[g]);
            }

         // TRANSCRIBED Crop_complete::process_day (crop_cropsyst.cpp:2701-2828)
         float64 act_growth = 0, gai_rel = 0, wcef = -1, green_pd = 0;
         if (roots) {
            roots->know_above_ground_biomass(canopy ? canopy->get_biomass_current(VITAL) : 0.0);
            roots->update(ws_stats.get_current(), false);
         }
         if (canopy) {
            canopy->process_day();
            plant_height = canopy->calc_plant_development(p.morphology.max_canopy_height, plant_height, overall);
            if (CO2) {
               CO2_TUE_adjustment = (canopy->get_interception_global_green() > 0.0000001)
                  ? CO2->get_TUE_adjustment() : 1.0;
               pot_co2 *= CO2->get_transpiration_adjustment_factor();
            }
            green_pd = canopy->get_interception_global_green();
            intercepted = (float64)srad * green_pd;
            if (bg) { bg->process_day(); attainable = bg->calc_attainable_top_growth(); }
            if (transpiration) {
               limited = mm_to_m(attainable / tdg.ref_transpiration_use_efficiency_CO2_adjusted());
               // process_transpiration_m (crop_common.cpp) with E1 replaced
               act_tr = ufrac * limited; if (act_tr < 0.0) act_tr = 0.0;
               float64 ws = (limited > 0.0) ? act_tr / limited : 1.0;
               ws_stats.append((float32)ws);
               // actual_growth (crop_cropsyst.cpp:1622-1738)
               if (!ph.has_matured_new()) {
                  act_growth = attainable;               // calc_act_biomass_growth :2847-2947
                  if (limited > 0.0) {
                     float64 ratio = act_tr / limited;
                     float64 tlg = attainable * ratio;
                     act_growth = std::min<float64>(tlg, attainable);
                     wcef = (lwp - p.morphology.stop_canopy_expansion_LWP)
                          / (p.morphology.reduce_canopy_expansion_LWP - p.morphology.stop_canopy_expansion_LWP);
                     wcef = CORN::must_be_between<float64>(wcef, 0.0, 1.0);
                     gai_rel = act_growth * wcef;
                  } else act_growth = 0.0;
                  overall = (attainable > 0.000001) ? act_growth / attainable : 1.0;
               } else overall = 1.0;
               float32 mws = (float32)ws_stats.get_current();
               yield.know_stress(mws);
               float64 gdd = tt->get_growing_degree_day();
               canopy->age_leaf(gdd, ws_stats.index);
               if (!ph.is_in_quiescence()) {
                  bool cont = (ph.is_in_accrescence() || !has_leaf_sen);
                  canopy->develop_leaf(cont, act_growth, gai_rel, gdd);
               }
               canopy->update_LAI_for_self_shading_response
                  (ph.has_expired_accrescence() || ph.is_in_culminescence() || ph.is_in_senescence(),
                   !ph.is_matured());
            }
         }
         // output
         printf("%d %a %a %a ", (int)ph.growth_stage, tt->get_accum_degree_days(),
                tt->get_growing_degree_day(), stress_adj_temp);
         printf("%a %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a ", ET_max, pot_unadj, pot_co2,
                CO2 ? CO2->get_transpiration_adjustment_factor() : 1.0, CO2_TUE_adjustment, intercepted,
                attainable, limited, act_tr, act_growth, gai_rel, wcef, ws_stats.get_current(),
                ws_stats.index, overall, tdg.ref_transpiration_use_efficiency_CO2_adjusted());
         if (canopy)
            printf("1 %a %a %a %a %a %a %a %a %a %a %a ", canopy->get_GAI(VITAL), canopy->get_GAI(VITAL | EFFETE),
                   canopy->get_LAI(), canopy->get_LAI(false), canopy->peak_LAI, canopy->cover_total_fract,
                   canopy->get_interception_global_green(), canopy->get_biomass_current(VITAL),
                   canopy->get_biomass_current(VITAL | EFFETE), plant_height, srf);
         else printf("0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 %a %a ", plant_height, srf);
         if (roots) {
            printf("1 %a %a %a %a ", roots->get_length(), roots->get_apparent_depth(), roots->get_biomass(),
                   roots->get_growth_biomass());
            for (int l = 1; l <= L; l++) printf("%a ", roots->get_total_fract_length_m()[l]);
         } else {
            printf("0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 ");
            for (int l = 1; l <= L; l++) printf("0x0p+0 ");
         }
         printf("%a %a %a ", bm_fill, bm_mat,
                yield.temperature_extreme_reponse ? yield.temperature_extreme_reponse->get_factor() : 1.0);
         for (int h = 0; h < 24; h++) printf("%a ", (double)hourly.get(h));
         printf("\n");
         // TRANSCRIBED end_day (crop_cropsyst.cpp:2005-2117)
         tt->end_day();
         ph.end_day();
         if (roots) ((Crop_root *)roots)->update_root_depth(false);
         yield.end_day();
      }
      // harvest_crop's yield (crop_cropsyst.cpp:1849-1866)
      float64 bfh = bm_mat;
      if (CORN::is_zero(bfh)) bfh = canopy ? canopy->get_biomass_current(VITAL | EFFETE) : 0.0;
      float64 yld = yield.calculate(bfh);
      printf("Y %a %a %a\n", bfh, yld, yield.get_reported_harvest_index());
      fflush(stdout);
   }
   _exit(0);
}
