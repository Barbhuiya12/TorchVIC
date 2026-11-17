/******************************************************************************
 * Oracle for the CropSyst crop backend (crop and soil nitrogen,
 * residues and organic matter).
 *
 * Linked against the COMPILED REFERENCE (<data root>/build/vic_cropsyst, -O0 -g, same
 * flags as the shipped Makefile). Modes, one request per stdin line:
 *
 *   SDR            the NRCS/RUSLE2 field operation table as the tillage code queries it
 *                  (USDA/NRCS/RUSLE2/SDR_field_ops.cpp: lookup_operation_number,
 *                  get_disturbance_depth_avg_m, get_mixing_fraction,
 *                  get_disturbance_rating, get_soil_disturbance_effect,
 *                  get_residue_flatten_ratio, get_residue_burial_ratio) and the
 *                  composite operations; one line per operation:
 *                  "OP <number> <depth_avg_m %a> <mixing %a> <rating> <effect>
 *                   <flatten[6] %a> <burial[6] %a> <name>"
 *                  then "COMPOSITE <name>|<op>|<op>..." lines and "END".
 *   CRP <path>     the reference-parsed [nitrogen], [residue] and phenology values of a
 *                  crop file (one line of %a values, see tests/test_cropsyst_nitrogen.py).
 *   SOIL ...       the soil N / residue day loop on real reference objects (VIC soil,
 *                  Chemicals_profile, Abiotic_environment_profile,
 *                  Organic_matter_pools_multiple, Organic_matter_residues_profile_common),
 *                  see run_soil below.
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unistd.h>
// Members are read and set directly on the reference objects; opening the access
// specifiers in this translation unit does not change any layout.
#define private public
#define protected public
#include "crop/crop_param.h"
#include "crop/crop_param_V4.h"
#include "USDA/NRCS/RUSLE2/SDR_field_ops.h"
#include "corn/data_source/vv_file.h"
#include "agronomic/VIC_soil.h"
#include "soil.h"
#include "soil/chemicals_profile.h"
#include "soil/nitrogen_profile.h"
#include "soil/abiotic_environment.h"
#include "organic_matter/OM_residues_profile_common.h"
#include "organic_matter/OM_pools_common.h"
#include "organic_matter/OM_common.h"
#include "organic_matter/multiple_pool/OM_pools_multiple.h"
#include "organic_matter/multiple_pool/OM_multiple_pool.h"
#include "common/residue/residues_cycling.h"
#include "common/residue/residue_decomposition_param.h"
#include "soil/tillage_effect.h"
#include "organic_matter/OM_params.h"
#include "soil/disturbance.h"
#include "management/management_param_struct.h"
#include "common/weather/parameter/WP_air_temperature.h"
#include "common/weather/parameter/WP_wind_speed.h"

extern option_struct options;
namespace USDA { namespace NRCS { namespace RUSLE2 {
extern Field_operation_parameters field_operation_parameters[];
extern Field_operation_composite_parameters field_operation_composite_parameters[];
}}}
using namespace CropSyst;

namespace {
double rd(char *&s) {
   char *e; double v = strtod(s, &e);
   if (e == s) { fprintf(stderr, "parse error at '%.20s'\n", s); exit(2); }
   s = e; return v;
}

void print_sdr() {
   using namespace USDA::NRCS::RUSLE2;
   for (nat16 i = 1; field_operation_parameters[i].number_dont_use; i++) {
      const char *name = field_operation_parameters[i].operation_name;
      nat16 found = SDR_field_operations.lookup_operation_number(name);
      if (found != i) continue;           // a duplicate name: lookup returns the first one
      float32 depth = SDR_field_operations.get_disturbance_depth_avg_m(i);
      float32 mixing = SDR_field_operations.get_mixing_fraction(i);
      int rating = (int)SDR_field_operations.get_disturbance_rating(i);
      int effect = (int)SDR_field_operations.get_soil_disturbance_effect(i);
      const float32 *fl = SDR_field_operations.get_residue_flatten_ratio(i);
      const float32 *bu = SDR_field_operations.get_residue_burial_ratio(i);
      printf("OP %d %a %a %d %d", (int)i, (double)depth, (double)mixing, rating, effect);
      for (int k = 0; k < RUSLE2_RESIDUE_TYPE_COUNT; k++) printf(" %a", (double)fl[k]);
      for (int k = 0; k < RUSLE2_RESIDUE_TYPE_COUNT; k++) printf(" %a", (double)bu[k]);
      printf(" %s\n", name);
   }
   for (nat16 i = 0; field_operation_composite_parameters[i].number_dont_use; i++) {
      printf("COMPOSITE %s", field_operation_composite_parameters[i].operation_name);
      for (int k = 0; k < 12 && field_operation_composite_parameters[i].operations[k]; k++)
         printf("|%s", field_operation_composite_parameters[i].operations[k]);
      printf("\n");
   }
   printf("END\n");
}

void print_crp(const char *path) {
   Crop_parameters &p = *new Crop_parameters(true, false, true, true);
   CORN::VV_File f(path, true);
   f.get(p);
   const auto &n = p.nitrogen;
   printf("%d %a %a %a %a %a %a %a %a %a %a %a %a %d %a ", (int)n.fixating, (double)n.legume_fraction,
          (double)n.max_uptake_daily_rate_kg_ha, (double)n.residual_N_not_available_for_uptake,
          (double)n.soil_N_conc_where_N_uptake_decreases, (double)n.PAW_where_N_uptake_rate_decreases,
          (double)n.stubble_max_conc, (double)n.root_conc, (double)n.dilution_curve_max_conc_mature,
          (double)n.dilution_curve_slope, (double)n.emergence_max_conc, (double)n.emergence_crit_conc,
          (double)n.biomass_to_start_dilution_crit_N_conc_T_ha, (int)n.end_dilution_curve_deg_day,
          (double)n.leaf_expansion_sensitivity);
   printf("%d %d %d ", (int)p.photosynthetic_pathway_clad.get(), (int)p.is_perennial(),
          (int)p.phenology.maturity_significant);
   const Residue_decomposition_parameters &r = p.residue_decomposition;
   printf("%d %a %a %a %a %a %a %a %a %a %a %a %a %a\n", (int)r.RUSLE2_type_clad.get(),
          (double)r.fast_cycling_top_fraction, (double)r.slow_cycling_top_fraction, (double)r.lignified_top_fraction,
          (double)r.fast_cycling_root_fraction, (double)r.slow_cycling_root_fraction, (double)r.lignified_root_fraction,
          (double)r.fast_cycling_halflife, (double)r.slow_cycling_halflife, (double)r.lignified_halflife,
          (double)r.detrition_halflife, (double)r.carbon_fraction, (double)r.stubble_area_covered_to_mass_ratio,
          (double)r.surface_area_covered_to_mass_ratio);
}

/* SOIL: the soil N / residue day loop on real reference objects, built as
   Land_unit_simulation_VX::initialize_soil and instanciate_organic_matter_and_residues
   (land_unit_sim.cpp:589-692, 759-854) do on a VIC soil, then each day:
   OM start_day; the day's operations in order (residue additions as
   distribute_any_crop_biomass_fate, NRCS field operation responses as
   respond_to_field_operation, inorganic fertilizer as perform_inorganic_nitrogen);
   OM process_day; crop uptake requests; chemical_transformations(3600);
   residue balance; soil process_day (tillage effect) and end_day, in the order of
   land_unit_sim.cpp:1413-2004. */
void read_decomp(char *&s, Residue_decomposition_parameters &p) {
   p.fast_cycling_top_fraction = (float32)rd(s); p.slow_cycling_top_fraction = (float32)rd(s);
   p.lignified_top_fraction = (float32)rd(s);
   p.fast_cycling_root_fraction = (float32)rd(s); p.slow_cycling_root_fraction = (float32)rd(s);
   p.lignified_root_fraction = (float32)rd(s);
   p.fast_cycling_halflife = (float32)rd(s); p.slow_cycling_halflife = (float32)rd(s);
   p.lignified_halflife = (float32)rd(s); p.detrition_halflife = (float32)rd(s);
   p.carbon_fraction = (float32)rd(s); p.stubble_area_covered_to_mass_ratio = (float32)rd(s);
   p.surface_area_covered_to_mass_ratio = (float32)rd(s);
}
void run_soil(char *s) {
   int L = (int)rd(s), D = (int)rd(s);
   static soil_con_struct soil_con; static layer_data_struct layer[MAX_LAYERS];
   std::memset(&soil_con, 0, sizeof(soil_con)); std::memset(layer, 0, sizeof(layer));
   options.Nlayer = L;
   CS::Annual_temperature &annual = *new CS::Annual_temperature;
   annual.avg = 10.0; annual.avg_half_amplitude = 10.0; annual.phase = 200;
   static CORN::date32 simdate = 2000001;
   for (int l = 0; l < L; l++) {
      soil_con.depth[l] = rd(s); soil_con.bulk_density[l] = rd(s); soil_con.max_moist[l] = rd(s);
      soil_con.Wcr[l] = rd(s); soil_con.Wpwp[l] = rd(s); soil_con.quartz[l] = rd(s); soil_con.VCS.clay[l] = rd(s);
   }
   VIC::Soil_components &soil = *new VIC::Soil_components(simdate, annual, layer, soil_con);
   // setup_soil_property_from_VIC (VIC_land_unit_simulation.cpp:202-226)
   for (int i = 1; i <= L; i++) {
      soil.mod_structure()->set_bulk_density_kg_m3(i, soil_con.bulk_density[i - 1]);
      soil.layers.set_thickness_m(i, soil_con.depth[i - 1]);
      soil.mod_texture()->set_sand_clay(i, 100. * soil_con.quartz[i - 1], 100. * soil_con.VCS.clay[i - 1]);
   }
   soil.layers.set_thickness_provided(true);
   soil.layers.set_depth_by_thickness(L);
   // the land unit holds the soil as a CropSyst::Soil_components pointer (land_unit_sim.cpp:748-757):
   // the accessors are called through it, which reaches the hydrology the reference uses
   CropSyst::Soil_components &base = soil;
   Soil::Abiotic_environment_profile &abiotic = *new Soil::Abiotic_environment_profile
      (base.ref_layers(), base.ref_temperature_profile_hourly(), base.ref_hydrology());
   CropSyst::Denitrification_parameters denit;
   soil_horizon_array32(hn); for (int h = 0; h < MAX_soil_horizons_alloc; h++) hn[h] = 0;
   soil.enable_nitrogen(true, CASCADE_HOURLY_INFILTRATION, &abiotic, denit, hn, hn, 0, 0);
   Soil::Chemicals_profile &ch = *soil.chemicals;
   ch.know_NH4_Langmuir(soil.NH4_Langmuir);
   ch.initialize();
   for (int l = 1; l <= L; l++) ch.set_NO3_N_mass_kg_m2(l, rd(s));
   for (int l = 1; l <= L; l++) ch.set_NH4_N_mass_kg_m2(l, rd(s));
   Organic_matter_initialization_parameters &init = *new Organic_matter_initialization_parameters;
   Organic_matter_decomposition_parameters &decp = *new Organic_matter_decomposition_parameters;
   CORN::Dynamic_array<float64> &evap = *new CORN::Dynamic_array<float64>(1, 1, 0);
   evap.set(0, 0.0);
   Air_temperature_average &tavg = *new Air_temperature_average;
   Wind_speed &wind = *new Wind_speed;
   soil_horizon_array32(pom); for (int h = 0; h < MAX_soil_horizons_alloc; h++) pom[h] = 0;
   Organic_matter_pools_multiple *pools = new Organic_matter_pools_multiple
      (decp, abiotic, *base.ref_layers(), *base.ref_structure(), base.mod_nitrogen(), *base.ref_texture());
   Organic_matter_residues_profile_common &om = *new Organic_matter_residues_profile_common
      (&soil, abiotic, evap, tavg, wind, pom, pools, init, decp, simdate);
   ch.NO3->know_CO2_C_loss_subsurface(om.get_CO2_C_loss_profile_residue_and_OM());
   om.clear_residues();
   Soil_tillage_effect *tillage = 0;
   for (int d = 0; d < D; d++) {
      for (int l = 0; l < L; l++) layer[l].moist = rd(s);
      for (int l = 0; l < L; l++) layer[l].ice = rd(s);
      for (int l = 0; l < L; l++) layer[l].T = rd(s);
      om.start_day();                    // land_unit_sim.cpp:1413
                                         // (no soil start_day in the VIC build, :1490-1492)
      int nops = (int)rd(s);
      std::vector<double> up3(L + 1, 0.0), up4(L + 1, 0.0);
      bool uptake = false;
      for (int k = 0; k < nops; k++) {
         int code = (int)rd(s);
         if (code == 1) {
            int pos = (int)rd(s); double bm = rd(s), nc = rd(s);
            Residue_decomposition_parameters &rp = *new Residue_decomposition_parameters; read_decomp(s, rp);
            om.residues.add_surface_plant_pool((Organic_matter_position)pos, OM_ORIGINAL_CYCLING, bm, nc, rp);
         } else if (code == 2) {
            soil_sublayer_array_64(roots); clear_sublayer_array(roots);
            for (int l = 1; l <= L; l++) roots[l] = rd(s);
            double nc = rd(s);
            Residue_decomposition_parameters &rp = *new Residue_decomposition_parameters; read_decomp(s, rp);
            om.residues.add_subsurface_plant_pool(roots, nc, rp);
         } else if (code == 3) {
            nat16 op = (nat16)rd(s); double oxs = rd(s), oxc = rd(s); int reps = (int)rd(s);
            CropSyst::Tillage_operation_struct::Oxidation_effect ox; ox.sand = (float32)oxs; ox.clay = (float32)oxc;
            using namespace USDA::NRCS::RUSLE2;
            for (int r = 0; r < reps; r++) {   // "Seedbed finisher" twice, the second without oxidation
               float64 depth = (float64)SDR_field_operations.get_disturbance_depth_avg_m(op);
               float64 mixing = SDR_field_operations.get_mixing_fraction(op);
               CropSyst::Soil_components &cs = soil;
               if (!tillage) tillage = new Soil_tillage_effect(&cs, cs.layers, cs.hydraulic_properties, cs.erosion);
               tillage->respond_to_field_operation(depth, mixing, SDR_field_operations.get_disturbance_rating(op),
                                                   (oxs >= 0 && r == 0) ? &ox : 0);
               soil.tillage_effect = tillage;
               om.respond_to_field_operation(depth, mixing, SDR_field_operations.get_soil_disturbance_effect(op),
                  soil.get_disturbance(), SDR_field_operations.get_residue_flatten_ratio(op),
                  SDR_field_operations.get_residue_burial_ratio(op));
            }
         } else if (code == 4) {
            double no3 = rd(s), nh4 = rd(s), fr = rd(s); int lay = (int)rd(s);
            double nh4_m2 = nh4 * over_ha_to_m2, non_vol = nh4_m2 * fr, vol = nh4_m2 - non_vol;
            ch.NO3->application_at_layer(no3 * over_ha_to_m2, lay, true);
            ch.NH4->volatilized_application(non_vol, vol, lay);
            ch.commit_diffs();
         } else if (code == 5) {
            uptake = true;
            for (int l = 1; l <= L; l++) up3[l] = rd(s);
            for (int l = 1; l <= L; l++) up4[l] = rd(s);
         }
      }
      om.know_soil_disturbance(soil.get_disturbance());
      om.process_day();
      Organic_matter_pools_common *pp = om.organic_matter_pools_profile_owned;
      std::vector<double> co2(L + 1), mn(L + 1);
      for (int l = 1; l <= L; l++) { co2[l] = pp->CO2_C_loss_subsurface_residue_and_OM[l]; mn[l] = pp->mineralized_N_residue[l]; }
      if (uptake) for (int l = 2; l <= L; l++) { ch.uptake_N_mass_from_NO3(l, up3[l]); ch.uptake_N_mass_from_NH4(l, up4[l]); }
      ch.commit_diffs();
      ch.chemical_transformations(3600);
      om.residues.balance(true);
      soil.process_day();                // :1907 (tillage effect and disturbance, soil.cpp:124-131)
      tillage = soil.tillage_effect;
      soil.end_day();                    // :2004 (records the day's water content)
      printf("D %d", d);
      for (int l = 1; l <= L; l++) printf(" %a", ch.get_NO3_N_mass_kg_m2(l));
      for (int l = 1; l <= L; l++) printf(" %a", ch.get_NH4_N_mass_kg_m2(l));
      for (int l = 1; l <= L; l++) printf(" %a", co2[l]);
      for (int l = 1; l <= L; l++) printf(" %a", mn[l]);
      printf(" %a", om.residues.flat_surface_contact_fraction);
      int n = 0;
      FOR_EACH_IN(p, Organic_biomatter_pool_common, (*pp), it) { if (p->type <= MANURE_RESIDUE) n++; } FOR_EACH_END(it)
      printf(" %d", n);
      FOR_EACH_IN(p, Organic_biomatter_pool_common, (*pp), it2) {
         if (p->type > MANURE_RESIDUE) continue;
         int env = -1;
         for (int l = 0; l < MAX_soil_layers_alloc; l++) if (p->soil_abiotic_env == &abiotic.soil_abiotic_environment_layer[l]) env = l;
         printf(" %d %d %d %d %d %a %a %a %a %a %a", (int)p->type, (int)p->position, (int)p->layer, (int)p->carbon_cycling, env,
                p->kg(), p->carbon_fraction, p->carbon_nitrogen_ratio, p->decomposition_constant, p->detrition_constant, p->area_per_mass);
      } FOR_EACH_END(it2)
      const Soil::Disturbance *dist = soil.get_disturbance();
      for (int l = 1; l <= L; l++) printf(" %a %a", soil.ref_texture()->get_sand_percent(l), soil.ref_texture()->get_clay_percent(l));
      for (int l = 1; l <= L; l++) printf(" %a", dist ? dist->rating[l] : 0.0);
      printf("\n");
   }
   printf("END\n");
}
}  // namespace

int main() {
   static char line[1 << 22];
   while (fgets(line, sizeof line, stdin)) {
      line[strcspn(line, "\n")] = 0;
      if (!strcmp(line, "SDR")) { print_sdr(); fflush(stdout); continue; }
      if (!strncmp(line, "CRP ", 4)) { print_crp(line + 4); fflush(stdout); continue; }
      if (!strncmp(line, "SOIL ", 5)) { run_soil(line + 5); fflush(stdout); continue; }
      fprintf(stderr, "unknown request\n"); return 2;
   }
   return 0;
}
