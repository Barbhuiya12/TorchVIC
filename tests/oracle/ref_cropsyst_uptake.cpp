/******************************************************************************
 * Oracle for the CropSyst crop backend (E1-E3):
 *   E1 root water uptake (Crop_transpiration_2::calc_transpiration)
 *   E2 soil water potential per VIC layer (VIC::Soil_hydrology::get_water_potential)
 *   E3 uptake into the VIC layers (VIC::Soil_hydrology::extract_water)
 *
 * The harness is linked against the COMPILED REFERENCE: the VIC-CropSyst objects
 * and libraries of the CPU build (<data root>/build/vic_cropsyst, same flags,
 * -O0 -g). E1, E2 and E3 above are calls into those objects, not re-readings.
 * The soil is the real VIC::Soil_components built on a soil_con_struct and a
 * layer_data_struct array, exactly as VIC_land_unit_simulation.cpp builds it.
 *
 * Three small pieces sit inside larger functions that cannot be called alone;
 * they are written here from the cited lines and are marked TRANSCRIBED:
 *   - the water stress ratio of Crop_common::process_transpiration_m
 *     (crop_common.cpp:157-164),
 *   - the per-step layer evap of a crop tile (canopy_evap.c:1008-1009),
 *   - the top-layer evap seen by runoff() (runoff.c:370-374).
 *
 * stdin, one case per line (all reals as C99 hex floats or decimals):
 *   L pot_transp_m fract_green max_uptake_mm interception_m co2_factor
 *   lwp_closure lwp_wilt raf_dry raf_sat lwp_prev dt_sec evap_bare_soil_mm
 *   then per layer l=1..L:
 *     depth_m max_moist_mm AE b psi_fc moist_mm ice_mm fract_root transp_daily_prev_mm
 * stdout, one line per case (C99 hex floats, %a):
 *   psi[L] act_transp_m uptake_act_m leaf_water_pot interception_after_m
 *   uptake_m[L] transp_daily_mm[L] extracted_m water_stress layer_evap_mm[L]
 *   top_evap_runoff_mm
 * The crop parameters (LWP, max uptake, RAF coefficients) are float32 in the
 * reference (crop_param_struct.h:293-295, cs_scenario.h:206-207); the harness
 * stores them as float32 like the reference does.
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

#include "agronomic/VIC_soil.h"
#include "crop/crop_param.h"          // as transpiration.cpp:4
#include "crop/transpiration.h"

extern option_struct options;   // vicNl.o / global.o

namespace {
// leaf_water_pot is a protected member that persists between days in the
// reference object; this subclass only lets the harness set yesterday's value.
class Transp_probe : public Crop_transpiration_2 {
 public:
   Transp_probe(CropSyst::Crop_parameters_struct::Transpiration &p,
                const Soil::Soil_interface *soil, float64 dry, float64 sat)
   : Crop_transpiration_2(p, soil, 0.0, 0.0, dry, sat) {}
   void set_leaf_water_pot(float64 v) { leaf_water_pot = v; }
};

double rd(char *&s) {
   char *e; double v = strtod(s, &e);
   if (e == s) { fprintf(stderr, "parse error\n"); exit(2); }
   s = e; return v;
}
}  // namespace

int main() {
   static soil_con_struct   soil_con;
   static layer_data_struct layer[MAX_LAYERS];
   std::memset(&soil_con, 0, sizeof(soil_con));
   std::memset(layer, 0, sizeof(layer));
   options.Nlayer = 3;
   CS::Annual_temperature annual_temperature;
   annual_temperature.avg = 10.0; annual_temperature.avg_half_amplitude = 10.0;
   annual_temperature.phase = 200;
   CORN::date32 simdate = 2000001;
   // Heap and never deleted: the reference Soil_base destructor crashes in
   // CS::Emanators::forget when no emanator was registered (CS_emanator.cpp:107).
   VIC::Soil_components &soil =
      *new VIC::Soil_components(simdate, annual_temperature, layer, soil_con);

   static char line[1 << 16];
   while (fgets(line, sizeof line, stdin)) {
      char *s = line;
      int L = (int)rd(s);
      if (L < 1 || L > MAX_LAYERS) { fprintf(stderr, "bad L\n"); return 2; }
      double pot = rd(s), green = rd(s), maxup = rd(s), icpt = rd(s), co2 = rd(s);
      double closure = rd(s), wilt = rd(s), dry = rd(s), sat = rd(s), lwp_prev = rd(s);
      double dt = rd(s), evap_bare = rd(s);
      soil_sublayer_array_64(fract_root); clear_sublayer_array(fract_root);
      for (int l = 0; l < L; l++) {
         soil_con.depth[l] = rd(s);
         soil_con.max_moist[l] = rd(s);
         soil_con.VCS.AE[l] = rd(s);
         soil_con.VCS.b_campbell[l] = rd(s);
         soil_con.VCS.water_pot_at_FC[l] = rd(s);
         layer[l].moist = rd(s);
         layer[l].ice = rd(s);
         fract_root[l + 1] = rd(s);
         layer[l].VCS.transpiration_daily = rd(s);
      }
      options.Nlayer = L;

      // E2: real call, 1-based CropSyst layer numbering (VIC_soil.cpp:102-127)
      std::vector<double> psi(L);
      for (int l = 0; l < L; l++)
         psi[l] = soil.ref_hydrology()->get_water_potential((nat8)(l + 1));

      // E1: real object and call (transpiration.cpp:41-117)
      CropSyst::Crop_parameters_struct::Transpiration par;
      std::memset(&par, 0, sizeof(par));
      par.LWP_stomatal_closure = (float32)closure;
      par.LWP_wilt_point = (float32)wilt;
      par.normal_max_water_uptake_mm = (float32)maxup;
      Transp_probe tr(par, &soil, (float32)dry, (float32)sat);
      tr.set_leaf_water_pot(lwp_prev);
      soil_sublayer_array_64(uptake); clear_sublayer_array(uptake);  // crop_common.cpp:128-129
      float64 icpt_io = icpt;
      float64 act = tr.calc_transpiration(pot, green, (float64)par.normal_max_water_uptake_mm,
                                          icpt_io, fract_root, uptake, co2);
      // TRANSCRIBED crop_common.cpp:157-164
      float64 act0 = act > 0.0 ? act : 0.0;
      float64 stress = (pot > 0.0) ? act0 / pot : 1.0;

      // E3: real call, start layer 2 as in crop_common.cpp:193 (VIC_soil.cpp:225-236)
      float64 extracted = soil.mod_hydrology()->extract_water(uptake, 2);
      // TRANSCRIBED canopy_evap.c:1008-1009 and runoff.c:370-374
      std::vector<double> evap(L);
      for (int l = 0; l < L; l++) evap[l] = layer[l].VCS.transpiration_daily * dt / 86400.0;
      double top_evap = evap_bare + evap[0];

      for (int l = 0; l < L; l++) printf("%a ", psi[l]);
      printf("%a %a %a %a ", act, tr.get_uptake_actual(), tr.get_leaf_water_pot(), icpt_io);
      for (int l = 0; l < L; l++) printf("%a ", uptake[l + 1]);
      for (int l = 0; l < L; l++) printf("%a ", layer[l].VCS.transpiration_daily);
      printf("%a %a ", extracted, stress);
      for (int l = 0; l < L; l++) printf("%a ", evap[l]);
      printf("%a\n", top_evap);
      fflush(stdout);
   }
   _exit(0);
}
