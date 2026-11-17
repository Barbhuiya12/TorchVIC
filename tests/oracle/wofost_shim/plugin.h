/* Stand-in for VIC-WUR 3.0 plugins/general/include/plugin.h in the WOFOST-NPK
 * oracle: the WOFOST options and steps of plugin_option_struct /
 * plugin_global_param_struct, then the real crop.h (which includes the real
 * wofost.h).
 */
#ifndef WOFOST_SHIM_PLUGIN_H
#define WOFOST_SHIM_PLUGIN_H

#include <vic_driver_image.h>

typedef struct {
    bool WOFOST_PIRR;
    bool WOFOST_PFERT;
    bool WOFOST_PTEMP;
    bool WOFOST_DIST_SEASON;
    bool WOFOST_DIST_TSUM;
    bool WOFOST_DIST_FERT;
    bool WOFOST_DIST_MIN;
    bool WOFOST_CALC_MIN;
    bool WOFOST_CONTINUE;
    bool WOFOST_FORCE_TSUM;
    bool WOFOST_FORCE_FERT;
    size_t NFERTTIMES;
    int NCROPTYPES;
} plugin_option_struct;

typedef struct {
    size_t wofost_steps_per_day;
} plugin_global_param_struct;

extern plugin_option_struct plugin_options;
extern plugin_global_param_struct plugin_global_param;

#include <crop.h>

#endif
