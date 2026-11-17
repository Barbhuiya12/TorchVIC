/* Stand-in for plugins/general/include/plugin.h (VIC-WUR 3.0) for ref_vicwur_lu.c:
 * the real plugin_def.h (options, global parameters) and the real landuse.h,
 * routing.h and irrigation.h; the other plugin headers are not needed by
 * lu_apply.c / lu_support.c. */
#ifndef VICWUR_LU_SHIM_PLUGIN_H
#define VICWUR_LU_SHIM_PLUGIN_H

#include <vic_driver_image.h>
#include <plugin_def.h>
#include <landuse.h>
#include <routing.h>
#include <irrigation.h>

#endif

/* The oracle records a log_err of lu_apply.c (the water, carbon and energy
 * balance errors :477,600,779 and the Cv sum check :1013-1018) per cell
 * instead of exiting, so one fatal cell does not hide the others; the cell
 * flags are written out and the kernels' `fatal` must match them.  Only
 * lu_apply.c / lu_support.c see this definition (vic_run keeps the real
 * vic_log.h one). */
extern int *lu_fatal_cell;
#undef log_err
#define log_err(M, ...) do { \
        fprintf(LOG_DEST, "[ERROR] " M "\n", ## __VA_ARGS__); \
        lu_fatal_cell[iCell] = 1; \
} while (0)
