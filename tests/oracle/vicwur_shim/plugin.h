/* Minimal stand-in for VIC-WUR plugins/general/include/plugin.h (see
 * vic_driver_image.h). Field names follow VIC-WUR 3.0 plugin_def.h:37-42 (decomposition
 * enum), :239-284 (options and global parameters); wu_con_struct follows
 * plugins/wateruse/include/wateruse.h:59-62. routing.h is the real VIC-WUR 3.0 header.
 */
#ifndef VICWUR_SHIM_PLUGIN_H
#define VICWUR_SHIM_PLUGIN_H

#include <vic_driver_image.h>

enum {
    RANDOM_DECOMPOSITION,
    BASIN_DECOMPOSITION,
    FILE_DECOMPOSITION,
    PLUGIN_N_DECOMPOSITION
};

typedef struct {
    short unsigned int DECOMPOSITION;
    short unsigned int UH_LENGTH;
    bool               FORCE_ROUTING;
    bool               NONRENEW_RUNOFF;
    bool               REMOTE_WITH;
    size_t             NWURECEIVING;
} plugin_option_struct;

typedef struct {
    size_t rout_steps_per_day;
} plugin_global_param_struct;

typedef struct {
    nc_file_struct routing;
    nc_file_struct wateruse;
} plugin_filenames_struct;

typedef struct {
    size_t  nreceiving;
    size_t *receiving;
} wu_con_struct;

#include <routing.h>

void convolve(const double *, size_t, const double *, size_t, double *);
void cshift(double *, int, int, int, int);

#endif
