/* Stand-in for VIC-WUR plugins/general/include/plugin.h for the remote
 * water-use oracle (ref_vicwur_wu_remote.c). It shadows vicwur_shim/plugin.h
 * (put this directory first on the include path) and adds what wu_remote.c
 * touches: NWUTYPES, UH_LENGTH and EFR (plugin_def.h:251,:259), the
 * rout_steps_per_day parameter, the efr_force table (efr.h) and the real
 * wateruse.h and routing.h, unchanged.
 */
#ifndef VICWUR_SHIM_RM_PLUGIN_H
#define VICWUR_SHIM_RM_PLUGIN_H

#include <vic_driver_image.h>

typedef struct {
    short unsigned int UH_LENGTH;
    short unsigned int NWUTYPES;
    bool               EFR;
} plugin_option_struct;

typedef struct {
    size_t rout_steps_per_day;
} plugin_global_param_struct;

plugin_option_struct       plugin_options;
plugin_global_param_struct plugin_global_param;

/* opaque: only named in the prototypes of the NetCDF output code */
typedef struct stream_struct stream_struct;
typedef struct nc_var_struct nc_var_struct;

#include <wateruse.h>
#include <routing.h>
#include <efr.h>

#endif
