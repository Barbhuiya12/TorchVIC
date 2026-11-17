/******************************************************************************
 * Oracle for the orchard canopy architecture of the CropSyst crop backend
 * (torchvic/crop/cropsyst/orchard_arch.py). Linked against
 * the COMPILED REFERENCE (<data root>/build/vic_cropsyst, -O0 -g, the shipped
 * defines incl. CROP_ORCHARD).
 *
 * stdin, one request per line:
 *   ARCH <crop path> <model 0 none|1 Kemanian-Stockle|2 Oyarzun-Stockle> <lat> <lon>
 *        <slope %> <aspect deg N0_E90> <N>
 *        then N day lines: year doy srad_MJ tt restart tt_after
 *     one output line per day: width breadth height interception (C99 hex floats;
 *     the extent is 0 0 0 without an interception model)
 *   CRP <path>  the parsed architecture: canopy_architecture w0 w1 b0 b1 h0 h1 bih
 *        row col azimuth flowering rapid_fruit_growth
 *
 * Real reference objects: the crop parameter parser (Crop_parameters + VV_File),
 * Geocoordinate, Sun (sunrise/sunset), Sun_days, Sun_hours,
 * CS::Solar_radiation_extraterrestrial_hourly, Solar_radiation_hour, Sky (cloud
 * cover of the first day), Orchard_common with its Tree_canopy_extent,
 * Interception_architectural_Oyarzun_Stockle and _Kemanian_Stockle.
 * TRANSCRIBED glue (cited): the land unit's Weather_hours_calculated getters
 * (weather_hours.cpp:55-73) over the real Solar_radiation_hour and extraterrestrial
 * objects; the meteorology's object wiring (CS_land_unit_meteorological.cpp:110-
 * 137, 221-227, 335-352; perform_op.cpp:1761-1826); the order of the day: the
 * land unit's start_day (Sun::update, Sky::update_clouds, the hourly update,
 * CS_land_unit_meteorological.cpp:384-396), then Crop_orchard_fruit::process_day
 * (expand_canopy, update_solid_canopy_interception; crop_fruit.cpp:659-671), a
 * restart after it (set_deg_days_begin_canopy_development, crop_fruit.cpp:262-270).
 * The Orchard_common is told of a canopy before the first day (know_canopy_leaf,
 * crop_fruit.cpp:782-789), which the shipped build does only once the crop has
 * provided one (patch P5 of the reference runs).
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "crop/crop_param.h"
#include "crop/crop_orchard_common.h"
#include "crop/crop_orchard_Oyarzun_Stockle.h"
#include "crop/crop_orchard_Kemanian_Stockle.h"
#include "common/geodesy/geocoordinate.h"
#include "common/weather/sun.h"
#include "common/weather/sky.h"
#include "common/weather/day/sun_days.h"
#include "common/weather/hour/sun_hours.h"
#include "common/weather/hour/weather_hours.h"
#include "common/weather/hour/solar_irrad_hour.h"
#include "common/solar/irradiance_extraterrestrial.h"
#include "common/weather/parameter/WP_solar_radiation.h"
#include "common/weather/parameter/WP_air_temperature.h"
#include "common/weather/parameter/WP_precipitation.h"
#include "corn/chronometry/date_32.h"
#include "corn/data_source/vv_file.h"

using namespace CropSyst;

namespace {
double rd(char *&s) {
   char *e; double v = strtod(s, &e);
   if (e == s) { fprintf(stderr, "parse error\n"); exit(2); }
   s = e; return v;
}
char *word(char *&s) {
   while (*s == ' ') s++;
   char *b = s;
   while (*s && *s != ' ' && *s != '\n') s++;
   if (*s) *s++ = 0;
   return b;
}
// weather_hours.cpp:55-73, the solar part of Weather_hours_calculated
class Weather_hours_glue : public CS::Weather_hours_store {
   const CS::Solar_radiation_extraterrestrial_hourly &et;
   const Solar_radiation_hour &sr;
   const Sun_hours &sh;
   const CORN::Date_clad_32 &today;
 public:
   Weather_hours_glue(const CS::Solar_radiation_extraterrestrial_hourly &et_, const Solar_radiation_hour &sr_,
                      const Sun_hours &sh_, const CORN::Date_clad_32 &today_)
      : CS::Weather_hours_store(), et(et_), sr(sr_), sh(sh_), today(today_) {}
   float64 get_solar_zenith_angle_rad(CORN::Hour h) const { return sh.get_solar_zenith_angle_rad(today.get_DOY(), h); }
   float64 get_solar_radiation_MJ_m2(CORN::Hour h) const { return sr.get_solar_irradiation(h); }
   float64 get_solar_direct_radiation_MJ_m2(CORN::Hour h) const { return sr.get_direct_solar_radiation(h); }
   float64 get_solar_diffuse_radiation_MJ_m2(CORN::Hour h) const { return sr.get_diffuse_solar_radiation(h); }
   float64 get_insolation_fraction(CORN::Hour h) const { return et.get_insolation_fraction_today(h); }
   const Sun_hours &ref_sun_hours() const { return sh; }
};
// access to the protected tree canopy of the interception models
class Oy : public Interception_architectural_Oyarzun_Stockle {
 public:
   using Interception_architectural_Oyarzun_Stockle::Interception_architectural_Oyarzun_Stockle;
   void extent(double &w, double &b, double &h) const
   {  w = tree_canopy->width.get(); b = tree_canopy->breadth.get(); h = tree_canopy->height.get(); }
};
class Ks : public Interception_architectural_Kemanian_Stockle {
 public:
   using Interception_architectural_Kemanian_Stockle::Interception_architectural_Kemanian_Stockle;
   void extent(double &w, double &b, double &h) const
   {  w = tree_canopy->width.get(); b = tree_canopy->breadth.get(); h = tree_canopy->height.get(); }
};

Crop_parameters &parse(const char *path) {
   Crop_parameters &p = *new Crop_parameters(true, false, true, true);
   CORN::VV_File f(path, true);
   f.get(p);
   return p;
}

void print_params(char *s) {
   Crop_parameters &p = parse(word(s));
   const auto &a = p.canopy_architecture;
   printf("%d %a %a %a %a %a %a %a %a %a %a %a %a\n", (int)p.morphology.canopy_architecture,
          (double)a.initial_width, (double)a.final_width, (double)a.initial_breadth, (double)a.final_breadth,
          (double)a.initial_height, (double)a.final_height, (double)a.branch_insertion_height,
          (double)p.layout.spacing.row, (double)p.layout.spacing.col, (double)p.layout.row_azimuth_true_north_deg,
          (double)p.phenology.initiation.flowering, (double)p.phenology.initiation.rapid_fruit_growth);
}

void run(char *s) {
   Crop_parameters &p = parse(word(s));
   int model = (int)rd(s);
   double lat = rd(s), lon = rd(s), slope = rd(s), aspect = rd(s);
   int N = (int)rd(s);
   Geocoordinate &geo = *new Geocoordinate;
   geo.set_latitude_dec_deg_f32((float32)lat);
   geo.set_longitude_dec_deg_f32((float32)lon);
   CORN::Date_clad_32 &today = *new CORN::Date_clad_32((CORN::date32)2001001);
   // CS_land_unit_meteorological.cpp:110-137, 221-227
   Sun &sun = *new Sun(today, geo);
   Sun_days &sun_days = *new Sun_days(geo.get_longitude_dec_deg(), geo.get_longitude_dec_deg());
   Sun_hours &sun_hours = *new Sun_hours(sun_days, geo.get_latitude_dec_deg());
   CS::Solar_radiation_extraterrestrial_hourly &et =
      *new CS::Solar_radiation_extraterrestrial_hourly(geo, sun_days, sun_hours, today);
   et.start();
   Solar_radiation &srad = *new Solar_radiation;
   Air_temperature_average &tavg = *new Air_temperature_average;
   tavg.set_Celcius(10.0, CORN::measured_quality);
   Precipitation &prec = *new Precipitation;
   Sky &sky = *new Sky(et, tavg, srad, prec);
   Solar_radiation_hour &srh = *new Solar_radiation_hour(srad, et);
   Weather_hours_glue &wh = *new Weather_hours_glue(et, srh, sun_hours, today);
   // perform_op.cpp:1761-1826
   Orchard_common::Interception_architectural *icpt = 0;
   Oy *oy = 0; Ks *ks = 0;
   if (model == 2)
      icpt = oy = new Oy(p.layout, p.canopy_architecture, slope, aspect, sun_days, sun, sun_hours, wh,
                         sky.curr_cloud_cover_fraction, today);
   else if (model == 1)
      icpt = ks = new Ks(p.layout, p.canopy_architecture, slope, aspect, sun_days, sky.curr_cloud_cover_fraction,
                         today, geo.get_latitude_radians());
   Orchard_common &orchard = *new Orchard_common(p.morphology.light_extinction_coef, icpt, p.canopy_architecture,
                                                 p.phenology.initiation.flowering,
                                                 p.phenology.initiation.rapid_fruit_growth);
   orchard.know_canopy_leaf(0);
   char line[512];
   for (int d = 0; d < N; d++) {
      if (!fgets(line, sizeof line, stdin)) exit(3);
      char *q = line;
      int year = (int)rd(q), doy = (int)rd(q);
      double sr = rd(q), tt = rd(q);
      int restart = (int)rd(q);
      double tt_after = rd(q);
      today.set_date32((CORN::date32)(year * 1000 + doy));
      sun.update();
      srad.set_MJ_m2(sr, CORN::measured_quality);
      sky.update_clouds();
      srh.update();
      orchard.expand_canopy(tt);
      double sci = orchard.update_solid_canopy_interception();
      double w = 0.0, b = 0.0, h = 0.0;
      if (oy) oy->extent(w, b, h);
      if (ks) ks->extent(w, b, h);
      printf("%a %a %a %a\n", w, b, h, sci);
      if (restart) orchard.set_deg_days_begin_canopy_development(tt_after);
   }
}
}  // namespace

int main() {
   char line[4096];
   while (fgets(line, sizeof line, stdin)) {
      if (!strncmp(line, "CRP ", 4)) { print_params(line + 4); fflush(stdout); continue; }
      if (!strncmp(line, "ARCH ", 5)) { run(line + 5); fflush(stdout); continue; }
   }
   return 0;
}
