/* Stand-in for VIC-WUR plugins/general/include/plugin.h for the
 * non-renewable oracle (ref_vicwur_nonrenew.c). It shadows
 * vicwur_shim/plugin.h (put this directory first on the include path) and adds
 * what wu_nonrenewable.c and rout_run.c touch: the sector enum
 * (plugin_def.h:225-233), NWUTYPES, UH_LENGTH, NONRENEW_RUNOFF
 * (plugin_def.h:251,:259), NREN_LIM (:307) and rout_steps_per_day. The real
 * wateruse.h and routing.h are used unchanged.
 */
#ifndef VICWUR_SHIM_NR_PLUGIN_H
#define VICWUR_SHIM_NR_PLUGIN_H

#include <vic_driver_image.h>

enum {
    WU_MUNICIPAL,
    WU_ENERGY,
    WU_MANUFACTURING,
    WU_LIVESTOCK,
    WU_IRRIGATION,
    WU_NSECTORS
};

typedef struct {
    short unsigned int UH_LENGTH;
    short unsigned int NWUTYPES;
    bool               NONRENEW_RUNOFF;
} plugin_option_struct;

typedef struct {
    size_t rout_steps_per_day;
} plugin_global_param_struct;

typedef struct {
    double NREN_LIM;
} plugin_parameters_struct;

plugin_option_struct       plugin_options;
plugin_global_param_struct plugin_global_param;
plugin_parameters_struct   plugin_param;

/* opaque: only named in the prototypes of the NetCDF output code */
typedef struct stream_struct stream_struct;
typedef struct nc_var_struct nc_var_struct;

#include <wateruse.h>
#include <routing.h>

#endif
