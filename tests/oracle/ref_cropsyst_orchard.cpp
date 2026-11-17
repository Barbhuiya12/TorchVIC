/******************************************************************************
 * Oracle for the orchard / perennial fruit part of the CropSyst crop backend
 * Linked against the COMPILED REFERENCE (<data root>/build/
 * vic_cropsyst, -O0 -g, the shipped defines incl. CROP_ORCHARD).
 *
 * stdin, one request per line:
 *   CRP <path>      the reference parser's view of a crop file (orchard fields)
 *   SEASON <path> N L south co2_on thick[L] h_stage h_offset h_min_lai h_min_bm
 *          then per day: doy tmax tmin tmin_tom srad_MJ vpd_max et_ref delta gamma ra
 *          ufrac lwp
 *                   a multi-year orchard run of the crop file with the replayed water
 *                   stress (act = ufrac * limited potential transpiration), one output
 *                   line per day (see test_cropsyst_orchard.py, ORACLE_FIELDS).
 *
 * Real reference objects used day by day (SEASON): Thermal_time_daily_linear/
 * _nonlinear and Phenology_2013 with the parsed crop file (fruit tree flags), the
 * land unit's hourly Temperature_estimator (temperature_est.cpp), Fruit_chill_
 * requirement, Fruit_bud_break_requirement, Fruit_harvest_requirements
 * (crop_fruit.cpp:17-79), Quiescence_start/_end_requirement (quiescence.cpp),
 * CS::Seasons (seasons.cpp), Canopy_growth_leaf_area_index_based_V5 (with the fruit
 * self-shading branch), Crop_root_vital_V5, Biomass_growth_RUE_TUE (fruit: no
 * temperature limitation; perennial season adjustment), Biomass_production_
 * transpiration_dependent_V4, Crop_CO2_response, the water stress statistics.
 * TRANSCRIBED (members of Crop_complete / Crop_orchard_fruit / the land unit that
 * cannot be built alone), each block cites its lines: Crop_orchard_fruit::start and
 * Crop_complete::start, the day order of the VIC land unit (start_day, performable
 * operations, fruit harvestable, process_day, end_day), Crop_complete::start_day,
 * Crop_orchard_fruit::thermal_time_event and Crop_complete::thermal_time_event,
 * initiate_accrescence, consider_inactive_period (fruit and complete),
 * restart_after_inactive_period (fruit and complete), senesce_at_dormancy and
 * start_inactive_period, process_day, actual_growth with
 * distribute_todays_non_leaf_stem_growth, Crop_orchard_fruit::harvest.
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>
#include "crop/crop_param.h"
#include "crop/crop_types.h"
#include "crop/phenology_2013.h"
#include "crop/thermal_time_daily.h"
#include "crop/crop_cropsyst.h"
#include "crop/crop_fruit.h"
#include "crop/quiescence.h"
#include "crop/canopy_growth_LAI_based_V5.h"
#include "crop/crop_root_V5.h"
#include "crop/biomass_growth_RUE_TUE.h"
#include "crop/transpiration_dependent_growth_V4.h"
#include "crop/CO2_response.h"
#include "agronomic/VIC_soil.h"
#include "common/weather/seasons.h"
#include "common/weather/any_timestep/temperature_est.h"
#include "common/weather/parameter/WP_temperature.h"
#include "common/weather/parameter/WP_air_temperature.h"
#include "common/weather/parameter/WP_solar_radiation.h"
#include "common/weather/parameter/WP_vapor_pressure_deficit.h"
#include "common/weather/parameter/WP_vapor_pressure_saturation_slope.h"
#include "common/weather/parameter/WP_psychrometric_const.h"
#include "common/biometeorology/parameter/CO2_atmospheric_conc.h"
#include "corn/chronometry/date_32.h"
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
class Qstart : public Quiescence_start_requirement {
 public:
   using Quiescence_start_requirement::Quiescence_start_requirement;
   int n() const { return consecutive_days; }
};
class Qend : public Quiescence_end_requirement {
 public:
   using Quiescence_end_requirement::Quiescence_end_requirement;
   int n() const { return consecutive_days; }
};
const unsigned VITAL = 1, EFFETE = 2;
double tt_of(const Phenology_abstract::Period_thermal *p) { return p ? p->get_thermal_time_accum() : -1.0; }

void print_params(const char *path) {
   Crop_parameters &p = *new Crop_parameters(true, false, true, true);
   CORN::VV_File f(path, true);
   f.get(p);
   printf("%d %d %d %d %d %d ", (int)p.crop_model_clad.get(), (int)p.harvested_part_clad.get(),
          (int)p.life_cycle_clad.get(), (int)p.continuation_mode_clad.get(),
          (int)p.abscission_clad.get(), (int)p.linger_days);
   printf("%d %d %d %d %d ", (int)p.is_fruit_tree(), (int)p.is_perennial(), (int)p.is_annual(),
          (int)p.has_leaf_senescence(), (int)(!p.continuation_mode_clad.get() == single_season));
   printf("%d %d ", (int)p.morphology.canopy_growth_model, (int)p.morphology.canopy_architecture);
   const auto &d = p.dormancy;
   printf("%a %a %a %a ", (double)d.chill_hours, (double)d.deg_day_bud_break_chill_req_sat,
          (double)d.deg_day_bud_break_chill_req_not_sat, (double)d.senesce);
   printf("%a %d ", (double)p.quiescence.inducement_temperature, (int)p.quiescence.consider_inactive_days);
   const auto &fr = p.fruit;
   printf("%d %a %a %d %a %d %d %d %a %a ", (int)fr.grape, (double)fr.fract_total_solids,
          (double)fr.max_fruit_load_kg_ha, (int)fr.harvest_condition_mode, (double)fr.harvest_temperature,
          (int)fr.harvest_temperature_days, (int)fr.harvest_DOY_earliest, (int)fr.harvest_DOY_latest,
          (double)fr.initial_growth_fraction_to_fruit, (double)fr.rapid_growth_fraction_to_fruit);
   printf("%a %a ", (double)p.phenology.initiation.rapid_fruit_growth, (double)p.phenology.initiation.veraison);
   const auto &b = p.biomass_production;
   for (int i = 0; i < 6; i++) printf("%a ", (double)b.season_biomass_adjustment[i]);
   printf("%a ", (double)b.continuous_cultivation_impact_factor);
   const auto &t = p.tolerance;
   printf("%d %a %a %a %a ", (int)t.enabled, (double)t.temperature_cold_fruit_damage,
          (double)t.temperature_cold_fruit_lethal, (double)t.temperature_cold_leaf_damage,
          (double)t.temperature_cold_leaf_lethal);
   printf("%a %a %a\n", (double)p.transpiration.ET_coef, (double)p.canopy_growth_LAI_based.max_LAI,
          (double)p.phenology.initiation.emergence);
}
void season(char *s) {
   while (*s == ' ') s++;
   char path[4096]; int k = 0;
   while (*s && *s != ' ') path[k++] = *s++;
   path[k] = 0;
   Crop_parameters &p = *new Crop_parameters(true, false, true, true);
   { CORN::VV_File f(path, true); f.get(p); }
   int N = (int)rd(s), L = (int)rd(s);
   bool south = (int)rd(s), co2_on = (int)rd(s);
   options.Nlayer = L;
   VIC::Soil_layers &layers = *new VIC::Soil_layers;
   for (int l = 1; l <= L; l++) layers.set_thickness_m(l, rd(s));
   layers.set_thickness_provided(true);
   layers.set_depth_by_thickness(L);
   int h_stage = (int)rd(s), h_offset = (int)rd(s);
   double h_min_lai = rd(s), h_min_bm_kg_ha = rd(s);
   const bool fruit = p.is_fruit_tree(), perennial = p.is_perennial();
   const bool has_leaf_sen = p.has_leaf_senescence();
   const int cont = p.continuation_mode_clad.get();
   const bool established = !p.continuation_mode_clad.get() == single_season;   // :2843-2845
   Crop_parameters_struct::Phenology &php = p.phenology;
   if (!php.accrescence_period || !php.root_elongation_period) { fprintf(stderr, "no periods\n"); exit(3); }
   Crop_parameters_struct::Thermal_time &ttp = p.thermal_time;
   auto &cp = p.canopy_growth_LAI_based;
   auto &rp = p.root;
   auto &bp = p.biomass_production;
   bp.efficiency_model_raw = "TUE"; bp.efficiency_model = TUE_efficiency;   // Crop_complete::start :599-605

   // weather the crop references
   float64 stress_adj_temp = 0.0, tmin = 0.0;
   Air_temperature_average &air_avg = *new Air_temperature_average;
   Solar_radiation &srad = *new Solar_radiation;
   Vapor_pressure_saturation_slope &slope = *new Vapor_pressure_saturation_slope;
   Psychro &psychro = *new Psychro;
   CS::CO2_atmospheric_concentration &co2c = *new CS::CO2_atmospheric_concentration;
   Vapor_pressure_deficit_daytime &vpd_day = *new Vapor_pressure_deficit_daytime;
   float64 ra = 0.001;
   co2c.set_ppm(370.0, CORN::measured_quality);
   CORN::date32 simdate = date_of(0);
   CORN::Date_clad_32 today(simdate);
   CS::Seasons seasons(today, south);

   Thermal_time_common *tt = (ttp.thermal_response == linear)
      ? (Thermal_time_common *)new Thermal_time_daily_linear(ttp, stress_adj_temp, tmin, fruit, 0, 0)
      : (Thermal_time_common *)new Thermal_time_daily_nonlinear(ttp, stress_adj_temp, tmin, fruit, 0, 0);
   // crop_cropsyst.cpp:303-321 passes (is_fruit_tree, is_perennial) to (is_perennial_, is_fruit_tree_)
   Phenology_probe &ph = *new Phenology_probe
      (php, tt->ref_thermal_time_cumulative(), stress_adj_temp, tmin, 0, 0, fruit, perennial, simdate);
   Crop_CO2_response *CO2 = 0;
   if (co2_on) {
      CO2 = new Crop_CO2_response(p.CO2_response, slope, psychro, ra, co2c);
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
   Temperature_estimator *air_est = 0;                 // the land unit's hourly estimator
   CORN::Dynamic_array<float32> &hourly = *new CORN::Dynamic_array<float32>(24);
   Canopy_growth_leaf_area_index_based_V5 *canopy = 0;
   Crop_root_vital_V5 *roots = 0;
   Biomass_growth_RUE_TUE *bg = 0;
   bool transpiration = false;
   nat8 emergence_count = 0;
   float64 pot_co2 = 0, intercepted = 0, reported_peak_LAI = 0, plant_height = 0, overall = 0;
   float64 attainable = 0, limited = 0, act_tr = 0, et_prev = 0;
   // Crop_orchard_fruit members (crop_fruit.h:90-140)
   Fruit_chill_requirement *chill = 0;
   Fruit_bud_break_requirement *bud = 0;
   Fruit_harvest_requirements *fh = 0;
   Qstart *qstart = 0;
   Qend *qend = 0;
   float64 fruit_bm = 0, reserves = 0, latest = 0, mfl_adj = 0;
   bool max_load = false, limit_lai = false, harvested = false, harvestable = false;
   std::vector<int> sync_day(NGS_COUNT, -1);
   Crop_parameters_struct::Fruit fruit_par = p.fruit;
   auto max_dry = [&]() { return (mfl_adj / 10000.0) * fruit_par.fract_total_solids; };   // crop_fruit.cpp:140-142
   auto canopy_lai = [&]() { return canopy ? canopy->get_LAI() : 0.0; };
   auto canopy_bm = [&]() { return canopy ? canopy->get_biomass_current(VITAL | EFFETE) : 0.0; };

   std::vector<double> W(12);
   for (int d = 0; d < N; d++) {
      for (int i = 0; i < 12; i++) W[i] = rd(s);
      int doy = (int)W[0];
      double tmax = W[1]; tmin = W[2];
      double tmin_tom = W[3], sr = W[4], vpdmax = W[5], et_ref = W[6];
      double delta = W[7], gamma = W[8]; ra = W[9];
      double ufrac = W[10];
      simdate = (CORN::date32)(2001000 + doy); today.set_date32(simdate);   // the season tests use the DOY
      air_avg.set_Celcius((tmax + tmin) / 2.0, CORN::measured_quality);
      srad.set_MJ_m2(sr, CORN::measured_quality);
      float64 vday = vpdmax * 0.666666; if (vday <= 0.0) vday = 0.00001;
      vpd_day.update_value_qualified_code(vday, CORN::measured_quality);
      slope.set_kPa_per_C(delta, CORN::measured_quality);
      psychro.set(gamma);
      float64 et_used = d ? et_prev : et_ref;
      bool harvest_today = false;
      overall = 1.0; attainable = 0.0; act_tr = 0.0;   // clear_day (:1925-1934)

      // ---- ET max (land_unit_sim.cpp:1296-1322; crop_common.cpp update_evapotranspiration_max),
      // fruit trees: the parameter ET coefficient (crop_cropsyst.cpp:3606-3618)
      float64 kadj = (p.transpiration.ET_coef == 0.0) ? 1.0 : fruit ? (float64)p.transpiration.ET_coef
                   : canopy ? canopy->calc_adjusted_ET_crop_coefficient() : 1.0;
      float64 ET_max = et_used * kadj; if (ET_max < 0.0) ET_max = 0.0;
      float64 pot_unadj = (canopy ? canopy->get_interception_global_green() : 0.0) * ET_max;
      pot_co2 = pot_unadj;

      auto trigger = [&](int code) { sync_day[code] = d; };
      // initiate_accrescence (crop_cropsyst.cpp:2286-2330)
      auto create_roots = [&]() {                    // :1514-1548
         float64 initial_root_length = cm_to_m(rp.length_at_emergence_cm);
         Phenology::Period_thermal *rep = ph.activate_root_elongation();
         rep->start_day();
         if (established) initial_root_length = rp.max_depth;
         roots = new Crop_root_vital_V5(rp, *rep, layers, initial_root_length);
      };
      auto initiate_accrescence = [&]() {
         emergence_count++;
         transpiration = true;
         if (!bg) bg = new Biomass_growth_RUE_TUE(fruit, &p, ph.growth_stage, emergence_count, pot_co2, tt, CO2,
                                                  canopy, air_avg, srad, intercepted, tdg, ts_stats);
         bg->start();
         delete canopy;
         canopy = new Canopy_growth_leaf_area_index_based_V5(p, cp, 0, 0, reported_peak_LAI, CO2);
         canopy->initialize();
         canopy->start();
         if (!roots) create_roots();
         float64 initial_biomass = cp.initial_GAI / cp.specific_leaf_area;
         canopy->restart_with(initial_biomass, cp.initial_GAI, true);
         roots->know_above_ground_biomass(initial_biomass);
         ph.activate_emergence();
         ph.activate_accrescence();
         canopy->know_accrescence(ph.get_accrescence_period());
         trigger(NGS_EMERGENCE); trigger(NGS_ACTIVE_GROWTH);
      };
      // Crop_complete::start_day (crop_cropsyst.cpp:1937-2003)
      auto start_day = [&]() {
         float64 srf = canopy ? canopy->get_senescence_reduction_factor() : 1.0;
         stress_adj_temp = calc_stress_adjusted_temperature
            (p.phenology.phenologic_sensitivity_water_stress, tmax, srf, ws_stats.index, vpdmax,
             !ph.is_matured(), has_leaf_sen);
         tt->accumulate_degree_days();
         ph.start_day();
         float64 y = tt->get_accum_degree_days_yesterday(), t = tt->get_accum_degree_days();
         auto match = [&](float64 x) { return (y <= x) && (t >= x); };
         if (!ph.is_in_quiescence()) {
            // Crop_orchard_fruit::thermal_time_event (crop_fruit.cpp:27-39)
            if (fruit) {
               if (match(php.initiation.filling)) {             // initiate_fruit_growth :298-307
                  harvestable = false; fruit_bm = 0.0;
                  ph.activate_fructescence(); trigger(FGS_FRUCTESCENCE);
               }
               if (match(php.initiation.rapid_fruit_growth)) {  // initiate_rapid_fruit_growth :318-341
                  if (fruit_par.grape) { trigger(FGS_VERAISON); ph.activate_veraison(); }
                  ph.activate_rapid_fruit_development();
                  trigger(FGS_RAPID_FRUIT_DEVELOPMENT);
                  delete fh; fh = new Fruit_harvest_requirements(p.fruit);
               }
            }
            // Crop_complete::thermal_time_event (crop_cropsyst.cpp:2967-3016)
            if (ph.is_in_germination() && match(php.initiation.emergence)) initiate_accrescence();
            if (canopy) {
               if (php.initiation.flowering && match(php.initiation.flowering)) {
                  trigger(NGS_ANTHESIS);
                  if (!ph.is_in_yield_formation()) { trigger(NGS_ANTHESIS); ph.activate_anthesis(); }
               }
               const Phenology_abstract::Period_thermal *c = ph.get_culminescence_period();
               if (c && c->has_expired()) { ph.activate_senescence(); canopy->know_senescence(ph.get_senescence_period()); }
               const Phenology_abstract::Period_thermal *a = ph.get_accrescence_period();
               if (a && a->has_expired()) {
                  ph.activate_culminescence(); canopy->know_culminescence(ph.get_culminescence_period());
                  trigger(NGS_END_CANOPY_GROWTH);
               }
               const Phenology_abstract::Period_thermal *sp = ph.get_senescence_period();
               if (sp && sp->has_expired()) canopy->know_senescence(0);   // returns false: phenology not reached
            }
            if (!(fruit && p.harvested_part_clad.get() == fruit_part)
                && php.initiation.filling && match(php.initiation.filling)) {
               trigger(NGS_FILLING); ph.activate_yield_formation();
            }
            if (php.maturity_significant && match(php.initiation.maturity)) {  // initiate_maturity :2399-2427
               trigger(NGS_MATURITY);
               if (!fruit) { delete bg; bg = 0; }
               ph.activate_maturity();
               if (canopy) canopy->know_maturity_initiation();
            }
         }
         if (CO2) CO2->update();
         if (roots) roots->start_day();
      };

      if (d == 0) {
         // ---- sowing (perform_op.cpp:430-512; land_unit_sim.cpp:1026-1029): the land unit's hourly
         // estimator is created (subdaily.cpp:120-140, dates fixed at the sowing day)
         air_est = new Temperature_estimator(60);
         air_est->reset(tmax, tmax, tmin, tmin);
         air_est->get_by_interval(hourly);
         // Crop_complete::initialize (:493-560), plant (:646-658), Crop_orchard_fruit::start (crop_fruit.cpp:191-226)
         ws_stats.clear(1.0); ts_stats.clear(1.0);
         ph.activate_sowing();
         if (fruit) {
            mfl_adj = fruit_par.max_fruit_load_kg_ha;
            if (seasons.is_winter_time()) {
               qend = new Qend(p.quiescence);
               chill = new Fruit_chill_requirement(p.dormancy.chill_hours);
            }
         }
         // Crop_complete::start (:597-640)
         if (perennial) create_roots();
         if (!qend && cont > single_season) qstart = new Qstart(p.quiescence, fruit);
         if (established) plant_height = p.morphology.max_canopy_height;
         else { trigger(NGS_PLANTING); trigger(NGS_GERMINATION); }
         ws_stats.clear(1.0); ts_stats.clear(1.0);    // start_season (:1887-1923)
         start_day();
      } else {
         // ---- Land_unit_simulation_VX::start_day: crop->start_day, then the weather's hourly update
         start_day();
         air_est->set_max_today_min_tomorrow(tmax, tmin_tom);
         air_est->get_by_interval(hourly);
         // ---- land unit process_day: the harvest event (perform_op.cpp:1063-1170), then a harvestable fruit crop
         auto fruit_harvest = [&]() {                  // Crop_orchard_fruit::harvest (crop_fruit.cpp:462-479)
            harvested = true; harvestable = false; latest = fruit_bm; fruit_bm = 0; harvest_today = true;
         };
         if (fruit && h_stage >= 0 && sync_day[h_stage] >= 0 && d == sync_day[h_stage] + h_offset
             && canopy_lai() > h_min_lai && canopy_bm() >= h_min_bm_kg_ha / 10000.0)
            fruit_harvest();
         if (fruit && harvestable && canopy_lai() > 0.0 && canopy_bm() >= 0.0) fruit_harvest();
      }
      // ---- Crop_orchard_fruit::process_day (crop_fruit.cpp:317-360)
      if (fh && fh->is_satisfied((CORN::DOY)doy, tmin, (tmax + tmin) / 2.0) && fruit_bm > 1e-6) harvestable = true;
      // consider_inactive_period (crop_fruit.cpp:54-130 around crop_cropsyst.cpp:2119-2178)
      auto restart = [&]() {
         // Crop_orchard_fruit::restart_after_inactive_period (crop_fruit.cpp:228-276)
         mfl_adj = p.fruit.max_fruit_load_kg_ha;
         delete bud; bud = 0; delete chill; chill = 0;
         // Crop_complete::restart_after_inactive_period (crop_cropsyst.cpp:2179-2284)
         ws_stats.clear(1.0); ts_stats.clear(1.0); harvestable = false;   // start_season
         delete qstart; qstart = new Qstart(p.quiescence, fruit);
         trigger(NGS_RESTART);
         ph.activate_restart();
         ph.activate_accrescence();
         ph.reset(NGS_EMERGENCE);
         ph.reset(NGS_RESTART);
         tt->clear_accum_degree_days();
         initiate_accrescence();
         limit_lai = false; max_load = false;
         roots->know_above_ground_biomass(2.5);
         roots->update(ws_stats.get_current(), false);
         harvested = false; reserves = 0;
      };
      if (fruit && bud && bud->is_satisfied(!chill, (float32)tt->get_accum_degree_days())) {
         trigger(FGS_BUD_BREAK);
         delete qend; qend = 0;
         restart();
         latest = 0.0;
      }
      if (qend) {
         if (qend->is_satisfied()) {
            delete qend; qend = 0;
            ph.activate_accrescence();
            restart();
         } else qend->add_daily_temperature((float32)((tmax + tmin) / 2.0));
      }
      if (qstart && canopy && ((fruit && !ph.is_in_accrescence()) || !fruit)
          && (ph.is_matured() || ph.is_in_rapid_fruit_development() || ph.is_in_culminescence()
              || ph.is_in_accrescence())
          && doy > 182) {
         if (qstart->is_satisfied() || doy >= 349) {
            delete qstart; qstart = 0;
            bool dormancy = cont == dormancy_season;
            if (dormancy) { harvested = false; canopy->know_senescence(0); }   // senesce_at_dormancy (crop_fruit.cpp:131-136)
            if (cont >= multiple_season) { delete chill; chill = new Fruit_chill_requirement(p.dormancy.chill_hours); }
            // start_inactive_period (crop_cropsyst.cpp:737-777): leaves dropped at 100 % senescence
            trigger(NGS_QUIESCENCE);
            intercepted = 0.0;
            if (canopy) { delete canopy; canopy = 0; }
            tt->clear_accum_degree_days();
            ph.activate_quiescence();
         } else qstart->add_daily_temperature((CORN::DOY)doy, (float32)((tmax + tmin) / 2.0));
      }
      if (chill) {
         if (chill->is_satisfied()) {
            delete chill; chill = 0;
            bud = new Fruit_bud_break_requirement(p.dormancy.deg_day_bud_break_chill_req_sat,
                                                  p.dormancy.deg_day_bud_break_chill_req_not_sat);
         } else {
            for (int hr = 0; hr < 24; hr++) chill->add_hour_temperature((float64)hourly.get(hr));
            // crop_fruit.cpp:123-125. set_chill_hours_reminning_negtive (crop_fruit.h:30) has no return
            // type (implicit int) and no return: g++ ends it with ud2, so the compiled reference stops
            // with SIGILL right after its store. The oracle performs the store (the first and only data
            // member, no vtable) and goes on, as the port does.
            if (CS::is_spring_or_summer_time((CORN::DOY)doy, south) && chill->get_chill_hours_remaining() > 0)
               *reinterpret_cast<float32 *>(chill) = -1.0f;
         }
      }
      // ---- Crop_complete::process_day, the rest (crop_cropsyst.cpp:2716-2826)
      float64 act_growth = 0, gai_rel = 0, leaf = 0;
      if (roots) {
         if (fruit) roots->root_length = rp.max_depth;
         else {
            roots->know_above_ground_biomass(canopy ? canopy->get_biomass_current(VITAL) : 0.0);
            roots->update(ws_stats.get_current(), false);
         }
      }
      if (canopy) {
         canopy->process_day();
         plant_height = canopy->calc_plant_development(p.morphology.max_canopy_height, plant_height, overall);
         if (CO2) {
            CO2_TUE_adjustment = (canopy->get_interception_global_green() > 0.0000001) ? CO2->get_TUE_adjustment() : 1.0;
            pot_co2 *= CO2->get_transpiration_adjustment_factor();
         }
         intercepted = (float64)srad * canopy->get_interception_global_green();
         if (bg) { bg->process_day(); attainable = bg->calc_attainable_top_growth(); }
         if (transpiration) {
            limited = mm_to_m(attainable / tdg.ref_transpiration_use_efficiency_CO2_adjusted());
            act_tr = ufrac * limited; if (act_tr < 0.0) act_tr = 0.0;
            float64 ws = (limited > 0.0) ? act_tr / limited : 1.0;
            ws_stats.append((float32)ws);
            // actual_growth (crop_cropsyst.cpp:1622-1738)
            bool allow = ph.is_maturing() && fruit;
            if (!ph.has_matured_new() || allow) {
               act_growth = attainable;                  // calc_act_biomass_growth :2847-2947
               if (limited > 0.0) {
                  float64 tlg = attainable * (act_tr / limited);
                  act_growth = std::min<float64>(tlg, attainable);
                  float64 wcef = (W[11] - p.morphology.stop_canopy_expansion_LWP)
                       / (p.morphology.reduce_canopy_expansion_LWP - p.morphology.stop_canopy_expansion_LWP);
                  wcef = CORN::must_be_between<float64>(wcef, 0.0, 1.0);
                  gai_rel = act_growth * wcef;
               } else act_growth = 0.0;
               overall = (attainable > 0.000001) ? act_growth / attainable : 1.0;
            } else overall = 1.0;
            float64 gdd = tt->get_growing_degree_day();
            canopy->age_leaf(gdd, ws_stats.index);
            if (!ph.is_in_quiescence()) {
               leaf = act_growth;
               if (fruit) {                              // distribute (crop_fruit.cpp:146-268)
                  if (max_load || harvested) { reserves += leaf; leaf = 0; }
                  else {
                     float64 frac = 0.0;
                     switch (ph.growth_stage) {
                        case FGS_FRUCTESCENCE: frac = fruit_par.initial_growth_fraction_to_fruit; break;
                        case FGS_VERAISON: case FGS_RAPID_FRUIT_DEVELOPMENT:
                           frac = fruit_par.rapid_growth_fraction_to_fruit; break;
                        default: frac = 0.0; break;
                     }
                     float64 today_f = act_growth * frac;
                     fruit_bm += today_f;
                     if (fruit_bm > max_dry()) { today_f = 0.0; fruit_bm = max_dry(); }
                     leaf -= today_f;
                  }
                  if (canopy->get_LAI() > cp.max_LAI) limit_lai = true;
               }
               bool cont_growth = (ph.is_in_accrescence() || !has_leaf_sen) && !(fruit && limit_lai);
               canopy->develop_leaf(cont_growth, leaf, gai_rel, gdd);
            }
            canopy->update_LAI_for_self_shading_response
               (ph.has_expired_accrescence() || ph.is_in_culminescence() || ph.is_in_senescence(), !ph.is_matured());
         }
      }
      if (fruit && fruit_bm >= max_dry()) max_load = true;   // crop_fruit.cpp:350-351
      // ---- output (the day record, before end_day)
      const Phenology_abstract::Period_thermal *pa = ph.get_accrescence_period(), *pc = ph.get_culminescence_period(),
         *ps = ph.get_senescence_period();
      printf("%d %a %a %a %a %a %a %a %a %a %a %a ", (int)ph.growth_stage, tt->get_accum_degree_days(),
             tt->get_growing_degree_day(), stress_adj_temp, ET_max, pot_unadj, attainable, limited, act_tr,
             act_growth, ws_stats.get_current(), ws_stats.index);
      if (canopy)
         printf("1 %a %a %a %a %a %a ", canopy->get_GAI(VITAL), canopy->get_GAI(VITAL | EFFETE), canopy->get_LAI(),
                canopy->get_LAI(false), canopy->get_interception_global_green(), canopy->get_biomass_current(VITAL));
      else printf("0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 ");
      printf("%a %a %a ", roots ? roots->get_length() : 0.0, roots ? roots->get_apparent_depth() : 0.0,
             roots ? roots->get_biomass() : 0.0);
      printf("%a %a %a %d %d %d %d %d %d %d %d %a %d %d %d ", fruit_bm, reserves,
             chill ? (double)chill->get_chill_hours_remaining() : 0.0, chill ? 1 : 0, bud ? 1 : 0,
             qend ? 1 : 0, qend ? qend->n() : 0, qstart ? 1 : 0, qstart ? qstart->n() : 0,
             (int)harvestable, (int)harvested, latest, (int)max_load, (int)limit_lai, (int)emergence_count);
      printf("%d %a %d %a %d %a %a %d %d %d ", pa ? 1 : 0, tt_of(pa), pc ? 1 : 0, tt_of(pc), ps ? 1 : 0, tt_of(ps),
             tt_of(ph.root()), (int)harvest_today, fh ? 1 : 0,
             fh ? (int)fh->consecutive_days_below_threshold_temperature : 0);
      for (int h = 0; h < 24; h++) printf("%a ", (double)hourly.get(h));
      if (roots) for (int l = 1; l <= L; l++) printf("%a ", roots->get_total_fract_length_m()[l]);
      else for (int l = 1; l <= L; l++) printf("0x0p+0 ");
      printf("\n");
      // ---- end_day (crop_cropsyst.cpp:2005-2117)
      tt->end_day();
      ph.end_day();
      if (roots) ((Crop_root *)roots)->update_root_depth(false);
      et_prev = et_ref;
   }
   fflush(stdout);
}
}  // namespace

int main() {
   static char line[1 << 22];
   while (fgets(line, sizeof line, stdin)) {
      line[strcspn(line, "\n")] = 0;
      if (!strncmp(line, "CRP ", 4)) { print_params(line + 4); fflush(stdout); continue; }
      if (!strncmp(line, "SEASON ", 7)) { season(line + 7); continue; }
      fprintf(stderr, "unknown request: %.40s\n", line); return 2;
   }
   _exit(0);
}
