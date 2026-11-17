/* Stand-in for the VIC image driver header (drivers/image/include/
 * vic_driver_image.h) for the land-use oracle (ref_vicwur_lu.c).
 *
 * Unlike vicwur_shim/ this one includes the REAL VIC-WUR 3.0 vic_run and shared_all
 * headers (vic_def.h, vic_run.h, vic_driver_shared_all.h), so lu_apply.c and
 * lu_support.c see the real all_vars / soil_con / veg_con structs, and the
 * real VIC-WUR 3.0 vic_run sources they call are linked in.  Only the image-driver
 * types are copied: location_struct, domain_struct and veg_con_map_struct
 * (drivers/shared_image/include/vic_driver_shared_image.h:79-120,166-175,
 * fields lu_*.c touches) and AREA_SUM_ERROR_THRESH (:37).  NetCDF types are
 * opaque (they only appear in prototypes).
 */
#ifndef VICWUR_LU_SHIM_DRIVER_H
#define VICWUR_LU_SHIM_DRIVER_H

#include <vic_driver_shared_all.h>

#define AREA_SUM_ERROR_THRESH 1e-5

typedef struct {
    double area;
    double frac;
    double latitude;
    double longitude;
    size_t global_idx;
    size_t io_idx;
    size_t local_idx;
} location_struct;

typedef struct {
    size_t           ncells_total;
    size_t           ncells_active;
    size_t           n_nx;
    size_t           n_ny;
    location_struct *locations;
} domain_struct;

typedef struct {
    size_t  nv_types;
    size_t  nv_active;
    int    *vidx;
    double *Cv;
} veg_con_map_struct;

typedef struct nc_file_struct nc_file_struct;
typedef struct nc_var_struct nc_var_struct;

void sprint_location(char *str, location_struct *loc);

#endif
