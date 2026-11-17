/* Minimal declarations so VIC's own rout_convolution.c and cshift.c compile
 * outside the image driver.
 *
 * The real rout.h pulls in vic_driver_shared_image.h, which needs NetCDF and
 * MPI. Only the surrounding type declarations are supplied here. The two
 * translation units under test, extensions/rout_rvic/src/rout_convolution.c
 * and cshift.c, are compiled verbatim from pristine VIC 5.1.0, so the
 * algorithm being measured is VIC's, not a second reading of it.
 *
 * rout_param_struct and rout_struct are copied field for field from VIC
 * 5.1.0's extensions/rout_rvic/include/rout.h.
 */
#ifndef ROUT_SHIM_H
#define ROUT_SHIM_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#define MM_PER_M 1000

typedef struct {
    size_t  full_time_length;
    size_t  n_timesteps;
    size_t  n_sources;
    size_t  n_outlets;
    size_t *source2outlet_ind;
    int    *source_y_ind;
    int    *source_x_ind;
    double *source_lat;
    double *source_lon;
    double *outlet_lat;
    double *outlet_lon;
    int    *source_VIC_index;
    int    *outlet_VIC_index;
    int    *source_time_offset;
    double *unit_hydrograph;
    double *aggrunin;
} rout_param_struct;

typedef struct {
    rout_param_struct rout_param;
    double           *ring;
    double           *discharge;
} rout_struct;

typedef struct {
    double area;
} location_struct;

typedef struct {
    size_t           ncells_total;
    location_struct *locations;
} domain_struct;

typedef struct {
    double dt;
} global_param_struct;

void convolution(double *, double *);
void cshift(double *, int, int, int, int);

#endif
