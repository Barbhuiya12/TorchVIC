/* Minimal stand-in for VIC-WUR plugins/general/include/plugin.h: the option,
 * global-parameter and parameter structs the dam sources read (field names
 * and order of VIC-WUR 3.0 plugin_def.h:239-284 and :301-308), then the real VIC-WUR 3.0
 * headers of the plugins whose structs they touch.  The include paths
 * (build_ref_vicwur_dam.sh) point at the VIC-WUR 3.0 tree.
 */
#ifndef VICWUR_DAM_SHIM_PLUGIN_H
#define VICWUR_DAM_SHIM_PLUGIN_H

#include <vic_driver_image.h>

typedef struct {
    bool ROUTING;
    bool DAMS;
    bool WATERUSE;
    bool EFR;
} plugin_option_struct;

typedef struct {
    size_t rout_steps_per_day;
} plugin_global_param_struct;

typedef struct {
    double DAM_ALPHA;
    double DAM_BETA;
    double DAM_GAMMA;
} plugin_parameters_struct;

#include <routing.h>
#include <efr.h>
#include <wateruse.h>
#include <dams.h>

plugin_option_struct       plugin_options;
plugin_global_param_struct plugin_global_param;
plugin_parameters_struct   plugin_param;

#endif
