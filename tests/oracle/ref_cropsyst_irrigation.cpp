/******************************************************************************
 * Oracle for the CropSyst irrigation kernels:
 *   E12 irrigation trigger and amount   VIC_soil_hydrology.c:44-204 (need_irrigation),
 *                                       canopy_evap.c:319-497, :560-641, irrigation_lib.c (tables)
 *   E13 mechanistic Ed and Ro           VIC_soil_hydrology.c:542-640 (Ed), :655-890 (Ro),
 *                                       nozzle table irrigation_lib.c:55-120
 *   E14 Saxton PTF                      read_soilparam.c:723-813
 *   E15 PET_SHORT rc/rarc               compute_pot_evap.c:70-75
 * (E1/E2 are covered by ref_cropsyst_uptake.)
 *
 * Linked against the COMPILED reference (<data root>/build/vic_cropsyst, -O0 -g, same flags):
 *   REAL calls:   need_irrigation, closest_layer_at, calc_irrigation_runoff,
 *                 calc_runoff_from_center_pivot, calc_infiltration_capacity,
 *                 calc_wetted_percentage, calc_Maximum_sorptivity, get_irrig_lib_index,
 *                 set_irrigation_efficiency_of_irrigation_library, the Irrigation_library table,
 *                 compute_pot_evap (PET_SHORT).
 *   REAL source, compiled here (the reference build has MECHANISTIC_IRRIGATION off, so the
 *   function is not in its objects): evaporation_from_irrigation_systems (Ed) and the Saxton
 *   block of read_soilparam. The build script cuts those line ranges out of the shipped source
 *   into ed_ref.inc / saxton_ref.inc and this file #includes them, so the code is not re-read.
 *   TRANSCRIBED (inside the 1600-line canopy_evap(), not callable alone), marked below:
 *   the gross-amount formula, the Ed/Ro/Dp table partition, the depletion-observation depth.
 *
 * stdin: one case per line, first token the mode; reals as C99 hex floats or decimals.
 *   N  L MAD obs_mm root_mm fill max_cap top_frac wsi tol  then per layer: depth_m FC Wpwp moist
 *   G  code mgmt demand_mm delta_t_sec
 *   E  code mgmt amount ET0 ET0w isPerennial  ksat porosity wpwp fc bulk S_max moist0 MAD
 *   P  amount d_spk ks s_philip t_rot  irrig_time  (calc_runoff_from_center_pivot, infiltration)
 *   S  L  then per layer: clay quartz depth_m
 *   V  dt_h shortwave net_lw tair vpd elev ra albedo run_time_hours
 *   O  mode const_m root_mm root_frac total_thickness_mm   (depletion observation depth)
 *   B  basin_wide_proration
 *   Q  rotation rec amount proration                (add one irrigation-pattern record)
 *   R  path year month day                          (read_irrig_pattern on a file; dumps the map)
 *   D  L MAD obs_mm root_mm fill max_cap top_frac wsi tol rotation rec pre_total  then per layer as N
 *   T  mgmt            (dump the whole table for that management factor)
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unistd.h>

#include "vicNl.h"
#include "irrigation_lib.h"
#include "crop/VIC_soil_hydrology.h"
#include "corn/math/compare.hpp"
using namespace CORN;

extern option_struct options;
extern veg_lib_struct *veg_lib;
extern char ref_veg_ref_crop[];
extern "C" int get_irrig_lib_index(const int);
double calc_irrigation_runoff(const int, double, double, double, const soil_con_struct &);
double calc_runoff_from_center_pivot(const double, const double, const double, const double, const double);
double calc_infiltration_capacity(const double, const double, const double);
double calc_Maximum_sorptivity(double, double, double, double);
int closest_layer_at(const soil_con_struct &, float64);
General_Irrigation_Type identify_general_irrigation_type(const Irrigation_Type);
bool is_surface_irrigation(const General_Irrigation_Type);
void set_irrigation_efficiency_of_irrigation_library(const int);

// The reference reads the root depth from the live CropSyst land unit
// (VIC_soil_hydrology.c:78-82, VIC_CropSyst_get(VC_root_depth_mm)); the link wraps that one call
// (-Wl,--wrap=VIC_CropSyst_get) so the harness can set it.
static double g_root_depth_mm = 0.0;
extern "C" double __wrap_VIC_CropSyst_get(unsigned long) { return g_root_depth_mm; }

// ---- deficit-irrigation path (the reference build has FULL_IRRIGATION defined; the shipped source of
// the other path is cut out by the build script and compiled here with FULL_IRRIGATION undefined)
#include <map>
#include <string>
int words(const char *s);                          // read_croplib.c, linked
extern global_param_struct global_param;
#undef FULL_IRRIGATION
#include "pattern_struct.inc"                      // VCS_Nl_def.h irrigation_pattern_struct
std::map<int, std::map<int, irrigation_pattern_struct>> girrig_patt;
// global_param.VCS.basin_wide_proration exists only in the non-FULL struct layout, so the build script
// points the cut source at this variable instead (one text substitution, nothing else changes)
static double g_basin_proration = 1.0;
double amount_of_deficit_irrigation(int rec, int rotation_code);
#define need_irrigation need_irrigation_deficit
#include "deficit_ref.inc"                         // need_irrigation (+rotation_code), amount_of_deficit_irrigation
#undef need_irrigation
#include "readpat_ref.inc"                         // read_irrig_pattern
#define FULL_IRRIGATION

// ---- reference source cut out by the build script ------------------------------------------
#define MECHANISTIC_IRRIGATION
#include "ed_ref.inc"

namespace {
double rd(char *&s) {
   char *e; double v = strtod(s, &e);
   if (e == s) { fprintf(stderr, "parse error\n"); exit(2); }
   s = e; return v;
}
bool tag(char *&s, char t) {
   while (*s == ' ') s++;
   if (*s != t) return false;
   s++; return true;
}
}  // namespace

int main() {
   static soil_con_struct   soil_con;
   static layer_data_struct layer[MAX_LAYERS];
   std::memset(&soil_con, 0, sizeof(soil_con));
   std::memset(layer, 0, sizeof(layer));
   options.Nlayer = 3;
   initialize_parameters();

   static char line[1 << 16];
   while (fgets(line, sizeof line, stdin)) {
      char *s = line;
      if (tag(s, 'N')) {                        // ---- E12 need_irrigation (real)
         int L = (int)rd(s);
         double MAD = rd(s), obs = rd(s), root = rd(s);
         bool fill = rd(s) != 0.0;
         double maxcap = rd(s), topf = rd(s), wsi = rd(s), tol = rd(s);
         options.Nlayer = L;
         for (int l = 0; l < L; l++) {
            soil_con.depth[l] = rd(s);
            soil_con.VCS.Field_Capacity[l] = rd(s);
            soil_con.Wpwp[l] = rd(s);
            layer[l].moist = rd(s);
         }
         g_root_depth_mm = root;
         double demand = -1.0, added = -7.0;
         bool irr = need_irrigation(MAD, obs, 0, 0.0, layer, soil_con, demand, fill, maxcap,
                                    &added, topf, wsi, tol);
         int cr = closest_layer_at(soil_con, mm_to_m(root));
         int co = closest_layer_at(soil_con, mm_to_m(obs));
         printf("%a %a %a %a %a ", (double)irr, demand, added, (double)cr, (double)co);
         for (int l = 0; l < L; l++) printf("%a ", layer[l].moist);
         printf("\n");
      } else if (tag(s, 'G')) {                 // ---- E12 gross amount + table partition
         int code = (int)rd(s), mg = (int)rd(s);
         double demand = rd(s), dt = rd(s);
         set_irrigation_efficiency_of_irrigation_library(mg);
         int idx = get_irrig_lib_index(code);
         if (idx < 0) { printf("-1\n"); fflush(stdout); continue; }
         const Irrigation_Efficiency &er = Irrigation_library[idx].irrigation_efficiency;
         double cap = er.maximum_capacity * dt / 86400.0;                    // canopy_evap.c:420-421
         // TRANSCRIBED canopy_evap.c:489-497 (IRRIGATION_FROM_DEMAND, table mode)
         double total = demand / er.table_irrigation_efficiency;
         if (total > cap) total = cap;
         // TRANSCRIBED canopy_evap.c:489-497 (MECHANISTIC_IRRIGATION branch, :490-492)
         double total_m = demand / (er.uniformity * er.presumed_efficiency);
         if (total_m > cap) total_m = cap;
         // TRANSCRIBED table partition canopy_evap.c:541-545, :583-585, :591-602, :607-611
         double Ed = er.table_irrigation_evap_loss * total;
         double aft = total - Ed;
         double Ro = total * er.table_irrigation_runoff_loss;
         double Dp = total * er.table_irrigation_deep_percolation_loss;
         if (Dp > (total - Ro)) Dp = total - Ro;
         double net = aft - Ro - Dp;
         int gt = (int)identify_general_irrigation_type(Irrigation_library[idx].IrrigationType_code);
         printf("%a %a %a %a %a %a %a %a %a %a %a %a %a\n", (double)idx, (double)gt, cap, total, total_m, Ed, aft, Ro,
                Dp, net, er.max_allowable_depletion, er.table_irrigation_efficiency,
                er.maximum_capacity);
      } else if (tag(s, 'E')) {                 // ---- E13 Ed (real source), Ro and wetted % (real)
         int code = (int)rd(s), mg = (int)rd(s);
         double amount = rd(s), et0 = rd(s), et0w = rd(s);
         bool per = rd(s) != 0.0;
         soil_con.Ksat[0] = rd(s); soil_con.porosity[0] = rd(s); soil_con.Wpwp[0] = rd(s);
         soil_con.VCS.Field_Capacity[0] = rd(s); soil_con.bulk_density[0] = rd(s);
         soil_con.VCS.S_max = rd(s);
         double moist0 = rd(s), MAD = rd(s);
         set_irrigation_efficiency_of_irrigation_library(mg);
         int idx = get_irrig_lib_index(code);
         if (idx < 0) { printf("-1\n"); fflush(stdout); continue; }
         double Ed = evaporation_from_irrigation_systems(idx, amount, et0, et0w, 0.5, soil_con, per);
         double wet = calc_wetted_percentage(Irrigation_library[idx].IrrigationType_code, soil_con,
                                             amount, per);
         double Ro = calc_irrigation_runoff(idx, amount, moist0, MAD, soil_con);
         printf("%a %a %a\n", Ed, wet, Ro);
      } else if (tag(s, 'P')) {                 // ---- Philip pieces (real)
         double amount = rd(s), d = rd(s), ks = rd(s), sp = rd(s), trot = rd(s), t = rd(s);
         printf("%a %a\n", calc_runoff_from_center_pivot(amount, d, ks, sp, trot),
                calc_infiltration_capacity(sp, t, ks));
      } else if (tag(s, 'S')) {                 // ---- E14 Saxton (real source block)
         int L = (int)rd(s);
         options.Nlayer = L;
         options.VCS.clay_input = 1;
         static soil_con_struct temp;
         std::memset(&temp, 0, sizeof(temp));
         for (int l = 0; l < L; l++) {
            temp.VCS.clay[l] = rd(s); temp.quartz[l] = rd(s); temp.depth[l] = rd(s);
         }
         int layer_i;
         { int layer = 0; (void)layer;
#include "saxton_ref.inc"
         }
         (void)layer_i;
         for (int l = 0; l < L; l++)
            printf("%a %a %a %a %a %a %a %a %a %a %a ", temp.VCS.silt[l], temp.soil_density[l],
                   temp.porosity[l], temp.bulk_density[l], temp.VCS.b_campbell[l], temp.VCS.AE[l],
                   temp.Ksat[l], temp.VCS.water_pot_at_FC[l], temp.VCS.Field_Capacity[l],
                   temp.Wpwp[l], temp.Wcr[l]);
         printf("%a\n", temp.VCS.S_max);
      } else if (tag(s, 'V')) {                 // ---- E15 compute_pot_evap, PET_SHORT (real)
         double dt = rd(s), sw = rd(s), lw = rd(s), ta = rd(s), vpd = rd(s), el = rd(s), ra = rd(s);
         double alb = rd(s), rt = rd(s);
         static veg_lib_struct vl[64];
         std::memset(vl, 0, sizeof(vl));
         int nveg = 3;
         vl[0].NVegLibTypes = nveg;
         for (int i = 0; i < N_PET_TYPES_NON_NAT; i++) {
            for (int m = 0; m < 12; m++) { vl[nveg + i].LAI[m] = 1.5; vl[nveg + i].albedo[m] = alb; }
            vl[nveg + i].rmin = 25.0 + i; vl[nveg + i].rarc = 2.0 + i; vl[nveg + i].RGL = 100.0;
         }
         veg_lib = vl;
         dmy_struct dmy; std::memset(&dmy, 0, sizeof(dmy)); dmy.month = 6;
         static double r0[N_PET_TYPES], r1[N_PET_TYPES];
         for (int i = 0; i < N_PET_TYPES; i++) { r0[i] = ra; r1[i] = ra; }
         const double *aero[N_PET_TYPES];
         static double pair[N_PET_TYPES][2];
         for (int i = 0; i < N_PET_TYPES; i++) { pair[i][0] = ra; pair[i][1] = ra; aero[i] = pair[i]; }
         double pe[N_PET_TYPES] = {0}, ped[N_PET_TYPES] = {0};
         compute_pot_evap(0, &dmy, 0, (int)dt, sw, lw, ta, vpd, el, aero, pe, ped, rt);
         printf("%a %a\n", pe[PET_SHORT], ped[PET_SHORT]);
      } else if (tag(s, 'B')) {
         g_basin_proration = rd(s);
         printf("0x0p+0\n");
      } else if (tag(s, 'Q')) {
         int rot = (int)rd(s), rec = (int)rd(s);
         irrigation_pattern_struct ip; ip.irrig_amount = rd(s); ip.proration_rate = rd(s);
         girrig_patt[rot][rec] = ip;
         printf("0x0p+0\n");
      } else if (tag(s, 'R')) {
         while (*s == ' ') s++;
         char *e = s; while (*e && *e != ' ' && *e != '\n') e++;
         std::string path(s, e - s); s = e;
         global_param.startyear = (int)rd(s); global_param.startmonth = (int)rd(s);
         global_param.startday = (int)rd(s);
         FILE *f = fopen(path.c_str(), "r");
         int rc = read_irrig_pattern(f, girrig_patt, 0);
         if (f) fclose(f);
         printf("%a", (double)rc);
         for (auto &r : girrig_patt) for (auto &q : r.second)
            printf(" %a %a %a %a", (double)r.first, (double)q.first, q.second.irrig_amount, q.second.proration_rate);
         printf("\n");
      } else if (tag(s, 'D')) {                 // ---- deficit-irrigation path (real source)
         int L = (int)rd(s);
         double MAD = rd(s), obs = rd(s), root = rd(s);
         bool fill = rd(s) != 0.0;
         double maxcap = rd(s), topf = rd(s), wsi = rd(s), tol = rd(s);
         int rot = (int)rd(s), rec = (int)rd(s);
         double pre = rd(s);
         options.Nlayer = L;
         for (int l = 0; l < L; l++) {
            soil_con.depth[l] = rd(s);
            soil_con.VCS.Field_Capacity[l] = rd(s);
            soil_con.Wpwp[l] = rd(s);
            layer[l].moist = rd(s);
         }
         g_root_depth_mm = root;
         double demand = -1.0, added = -7.0;
         bool irr = need_irrigation_deficit(MAD, obs, rec, 0.0, layer, soil_con, demand, rot, fill,
                                            maxcap, &added, topf, wsi, tol);
         double amt = amount_of_deficit_irrigation(rec, rot);
         // TRANSCRIBED canopy_evap.c:510-511
         double total = pre;
         if (amt >= 0) total = amt;
         printf("%a %a %a %a %a ", (double)irr, demand, added, amt, total);
         for (int l = 0; l < L; l++) printf("%a ", layer[l].moist);
         printf("\n");
      } else if (tag(s, 'O')) {                 // ---- depletion observation depth
         int mode = (int)rd(s);
         double cm = rd(s), root = rd(s), fr = rd(s), tot = rd(s), d;
         // TRANSCRIBED canopy_evap.c:328-349 (the enum is the reference's, mgmt_types.h:38)
         switch (mode) {
         case 0: d = m_to_mm(cm); break;                  // constant_fixed_depth_mode
         case 1: d = root * fr; break;                    // fraction_of_root_depth_mode
         case 2: d = tot; break;                          // soil_profile_depth_mode
         default: d = m_to_mm(cm); break;
         }
         d = d < 0.3 ? 0.3 : d;
         printf("%a\n", d);
      } else if (tag(s, 'T')) {                 // ---- table dump
         int mg = (int)rd(s);
         set_irrigation_efficiency_of_irrigation_library(mg);
         for (int i = 0; i < IRRIGATION_TYPE_ACCOUNTS; i++) {
            const VIC_Irrigation_library &t = Irrigation_library[i];
            const Irrigation_Facility_Charactor &f = t.irrigation_facility_charactor;
            const Irrigation_Efficiency &e = t.irrigation_efficiency;
            printf("%a %a %a %a %a %a %a %a %a %a %a %a %a %a %a  %a %a %a %a %a %a %a %a %a ", (double)i,
                   (double)t.IrrigationType_code, f.Runoff, f.D, f.h_op, f.H, f.cd, f.y0_spk, f.x_dm_spk,
                   f.Q, f.q_nuzzle, f.v, f.A_Ap, f.tirrig, f.actual_droplet_size, e.time_irrig,
                   e.uniformity, e.presumed_efficiency, e.table_irrigation_efficiency,
                   e.table_irrigation_runoff_loss, e.table_irrigation_evap_loss,
                   e.table_irrigation_deep_percolation_loss, e.maximum_capacity,
                   e.max_allowable_depletion);
         }
         for (int c = 0; c <= 80; c++) printf("%a ", (double)get_irrig_lib_index(c));
         printf("\n");
      } else {
         fprintf(stderr, "bad mode: %s\n", line); return 2;
      }
      fflush(stdout);
   }
   _exit(0);
}
