/******************************************************************************
 * Oracle for the clipping of the CropSyst crop backend.
 * Linked against the COMPILED REFERENCE (<data root>/build/vic_cropsyst, -O0 -g,
 * the shipped defines).
 *
 * stdin, one request per line; one state line on stdout after each request:
 *   NEW sla slp regrowth_gai initial_gai kc
 *        a new Canopy_growth_leaf_area_index_based_V5 at emergence (provide_canopy,
 *        initialize, start, restart_with(initial_GAI/SLA, initial_GAI), crop_cropsyst.cpp:
 *        779-832, 2296-2307)
 *   D cont act gai_rel   develop_leaf (canopy_growth_LAI_based.cpp:328-351)
 *   K rate gdd           the V5 die-back: die_back_to_GAI(GAI - rate * gdd)
 *                        (canopy_growth_LAI_based_V5.cpp:28-48)
 *   F term mode newest remove retain_gai retain_bm yield conc canopy_N p0..p9
 *                        Crop_complete::process_biomass_fate (crop_cropsyst.cpp:3210-3482,
 *                        TRANSCRIBED glue around the real canopy calls; nitrogen:
 *                        remove_canopy_nitrogen_for, crop_N_common.cpp:1126-1145)
 *   R ia                 the canopy side of respond_to_clipping: when ia, initiate_accrescence's
 *                        new canopy (provide_canopy + restart_with, crop_cropsyst.cpp:2296-2307),
 *                        then Canopy_growth_leaf_area_index_based::respond_to_clipping (:295-300)
 * State line: "S n_portions [GAI dead bm gbm]* U has GAI dead bm gbm lai_ss peak GAIve BMve cover"
 * after F also "Fate yield useful manure gdisp disposal chaff stubble uncut removed canopy_N".
 * All floats printed with %a.
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <list>
#include <map>
#include <unistd.h>
#define private public
#define protected public
#include "crop/crop_param.h"
#include "crop/canopy_growth_LAI_based_V5.h"
#include "crop/crop_biomass.h"
#include "crop/CO2_response.h"
#undef private
#undef protected

using namespace CropSyst;

namespace {
double rd(char *&s) {
   char *e; double v = strtod(s, &e);
   if (e == s) { fprintf(stderr, "parse error\n"); exit(2); }
   s = e; return v;
}
typedef Canopy_growth_leaf_area_index_based::Canopy_accumulation::Portion LPortion;
Crop_parameters *P = 0;
Canopy_growth_leaf_area_index_based_V5 *canopy = 0;
float64 reported_peak_LAI = 0;

void new_canopy() {
   delete canopy;
   canopy = new Canopy_growth_leaf_area_index_based_V5(*P, P->canopy_growth_LAI_based, 0, 0, reported_peak_LAI, 0);
   canopy->initialize();
   canopy->start();
   auto &cp = P->canopy_growth_LAI_based;
   float64 initial_biomass = cp.initial_GAI / cp.specific_leaf_area;           // crop_cropsyst.cpp:2302
   canopy->restart_with(initial_biomass, cp.initial_GAI, true);
}

void print_state() {
   std::vector<LPortion *> v;
   FOR_EACH_IN(portion, LPortion, canopy->canopy_vital, each)
   {  v.push_back(portion);
   } FOR_EACH_END(each)
   printf("S %d", (int)v.size());
   for (auto *p : v) printf(" %a %a %a %a", p->GAI, p->dead_AI, p->biomass_current, p->GAI_related_biomass);
   LPortion *u = (LPortion *)canopy->untouchable_portion;
   if (u) printf(" U 1 %a %a %a %a", u->GAI, u->dead_AI, u->biomass_current, u->GAI_related_biomass);
   else printf(" U 0 0x0p+0 0x0p+0 0x0p+0 0x0p+0");
   printf(" %a %a %a %a %a\n", canopy->get_LAI(false), canopy->peak_LAI,
          canopy->get_GAI(include_vital | include_effete), canopy->get_biomass_current(include_vital | include_effete),
          canopy->cover_total_fract);
}

// remove_canopy_nitrogen_for (crop_N_common.cpp:1126-1145), vital only (no effete canopy here)
double canopy_N = 0, conc = 0;
void remove_N(double respective_biomass) {
   double r = respective_biomass * conc;
   r = std::min<float64>(r, canopy_N);
   canopy_N -= r;
}

void fate(char *s) {
   int term = (int)rd(s), mode = (int)rd(s), newest = (int)rd(s);
   double remove_amount_kg_m2 = rd(s), management_retain_GAI = rd(s), management_retain_biomass_kg_m2 = rd(s);
   double yield_now = rd(s);
   conc = rd(s); canopy_N = rd(s);
   int16 pct[10];
   for (int k = 0; k < 10; k++) pct[k] = (int16)rd(s);
   // use graze residue dispose dead live root_dead root_loss root_live return_as_manure
   int16 use = pct[0], graze = pct[1], resid = pct[2], disp = pct[3], dead = pct[4], live = pct[5], ret = pct[9];
   Crop_biomass canopy_removed;
   // crop_cropsyst.cpp:3225
   if (term) canopy->terminate();
   float64 curr_canopy_biomass_kg_m2 = canopy->get_biomass_current(include_vital | include_effete | include_canopy_dead);
   float64 unharvested = curr_canopy_biomass_kg_m2;
   float32 total_act_biomass_removed = 0;
   double f_yield = 0, useful = 0, manure = 0, gdisp = 0, disposal = 0, chaff = 0, stubble = 0, uncut = 0;
   if (yield_now > 0.0000001) {                                                   // :3241-3249
      unharvested -= yield_now;
      float64 r = canopy->remove_biomass(yield_now, management_retain_GAI, true, !term, canopy_removed);
      remove_N(r);
      total_act_biomass_removed = (float32)r;
   }
   f_yield = yield_now;
   int32 actual_remove_to_standing_dead = (int32)dead;                            // :3257-3264
   int32 actual_remain_as_standing_live = (int32)live;
   if (term) {
      actual_remove_to_standing_dead += actual_remain_as_standing_live;
      actual_remove_to_standing_dead = std::min<int32>(actual_remove_to_standing_dead, 100);
      actual_remain_as_standing_live = 0;
   }
   float64 act_biomass_removed = 0;
   float64 removable_biomass = unharvested - management_retain_biomass_kg_m2;
   if (mode == 1) {                                                               // :3270-3279
      remove_amount_kg_m2 = std::min<float64>(remove_amount_kg_m2, removable_biomass);
      float64 r = canopy->remove_biomass(remove_amount_kg_m2, management_retain_GAI,
                                         newest, !term, canopy_removed);
      remove_N(r);
      act_biomass_removed += r;
   } else {                                                                        // :3280-3313
      useful = unharvested * (float32)use / 100.0;
      manure = unharvested * (float32)graze / 100.0 * ((float32)(ret) / 100.0);
      gdisp = unharvested * (float32)graze / 100.0 * (1.0 - (float32)(ret) / 100.0);
      disposal = unharvested * (float32)disp / 100.0;
      chaff = unharvested * (float32)resid / 100.0;
      float64 to_remove = 0.0;
      to_remove += useful; to_remove += manure; to_remove += gdisp; to_remove += disposal; to_remove += chaff;
      float64 killed = unharvested * (float32)actual_remove_to_standing_dead / 100.0;
      to_remove += killed;
      stubble = killed;
      uncut = unharvested * (float32)actual_remain_as_standing_live / 100.0;
      if (uncut < management_retain_biomass_kg_m2) {
         uncut = management_retain_biomass_kg_m2;
         to_remove = removable_biomass;
      }
      float64 r = canopy->remove_biomass(to_remove, management_retain_GAI, newest, !term, canopy_removed);
      remove_N(r);
      act_biomass_removed += (float32)r;
   }
   if (mode == 1 || mode == 3) {                                                  // :3325-3353
      useful = act_biomass_removed * (float32)use / 100.0;
      float32 removed_for_grazing = (float32)(act_biomass_removed * graze / 100.0);
      manure = removed_for_grazing * ((float32)(ret) / 100.0);
      gdisp = removed_for_grazing - manure;
      disposal = act_biomass_removed * (float32)disp / 100.0;
      chaff = act_biomass_removed * (float32)resid / 100.0;
      stubble = act_biomass_removed * actual_remove_to_standing_dead / 100.0;
      uncut = unharvested - act_biomass_removed;
   }
   total_act_biomass_removed += (float32)act_biomass_removed;
   if (term) {                                                                    // :3359-3376
      float64 killed_biomass = canopy->remove_all_biomass(canopy_removed);
      stubble += killed_biomass;
      uncut = 0;
      remove_N(999999.0);
      total_act_biomass_removed += (float32)killed_biomass;
   }
   print_state();
   printf("Fate %a %a %a %a %a %a %a %a %a %a\n", f_yield, useful, manure, gdisp, disposal, chaff, stubble, uncut,
          (double)total_act_biomass_removed, canopy_N);
}
}  // namespace

int main() {
   static char line[1 << 20];
   while (fgets(line, sizeof line, stdin)) {
      line[strcspn(line, "\n")] = 0;
      char *s = line + 1;
      if (!strncmp(line, "NEW ", 4)) {
         s = line + 3;
         delete P;
         P = new Crop_parameters(true, false, true, true);
         auto &cp = P->canopy_growth_LAI_based;
         cp.specific_leaf_area = (float32)rd(s);
         cp.stem_leaf_partition = (float32)rd(s);
         cp.regrowth_GAI = (float32)rd(s);
         cp.initial_GAI = (float32)rd(s);
         P->morphology.light_extinction_coef = (float32)rd(s);
         new_canopy();
         print_state();
      } else if (line[0] == 'D') {
         bool cont = (int)rd(s);
         double act = rd(s), rel = rd(s);
         canopy->develop_leaf(cont, act, rel, 0.0);
         print_state();
      } else if (line[0] == 'K') {
         double rate = rd(s), gdd = rd(s);
         float64 old_GAI = canopy->get_GAI(include_vital | include_effete);
         canopy->die_back_to_GAI(old_GAI - rate * gdd);
         print_state();
      } else if (line[0] == 'L') {                 // a canopy given portion by portion (test aid)
         int n = (int)rd(s);
         canopy->canopy_vital.delete_all();
         for (int i = 0; i < n; i++) {
            double g = rd(s), dd = rd(s), b = rd(s), gb = rd(s);
            LPortion *q = new LPortion(0, g, b, gb);
            q->dead_AI = dd;
            canopy->canopy_vital.append(q);
         }
         double g = rd(s), dd = rd(s), b = rd(s), gb = rd(s);
         LPortion *u = new LPortion(0, g, b, gb);
         u->dead_AI = dd;
         canopy->untouchable_portion = u;
         canopy->invalidate();
         canopy->get_GAI(include_vital | include_effete);      // the cached totals of a day
         print_state();
      } else if (line[0] == 'F') {
         fate(s);
      } else if (line[0] == 'R') {
         int ia = (int)rd(s);
         if (ia) new_canopy();
         canopy->respond_to_clipping();
         print_state();
      } else {
         fprintf(stderr, "unknown request: %.40s\n", line); return 2;
      }
      fflush(stdout);
   }
   _exit(0);
}
