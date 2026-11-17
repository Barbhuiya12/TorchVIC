/* Minimal stand-in for the VIC image driver headers, so that VIC-WUR plugin
 * sources (<data root>/build/vicwur3/src_tree, VIC-WUR 3.0) compile unchanged in the
 * oracles (tests/oracle/ref_vicwur_*.c) without NetCDF or MPI.
 *
 * Only the fields the included sources touch are declared. Constants are
 * copied from VIC-WUR 3.0: MM_PER_M and M3_PER_HM3 (vic_run/include/
 * vic_physical_constants.h:51,55), AREA_SUM_ERROR_THRESH (drivers/
 * shared_image/include/vic_driver_shared_image.h:37), VIC_MPI_ROOT
 * (vic_mpi.h:38). log_err exits, as in vic_run/include/vic_log.h:111-114.
 * The NetCDF readers and MPI scatter/gather are supplied by the oracle.
 */
#ifndef VICWUR_SHIM_DRIVER_H
#define VICWUR_SHIM_DRIVER_H

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MM_PER_M 1000
#define M3_PER_HM3 (100 * 100 * 100)
#define AREA_SUM_ERROR_THRESH 1e-5
#define VIC_MPI_ROOT 0
#define MAXSTRING 2048
#define NC_NOWRITE 0
#define LOG_DEST stderr

#define log_err(M, ...) fprintf(LOG_DEST, "[ERROR] " M "\n", ## __VA_ARGS__); \
    exit(EXIT_FAILURE);
#define log_warn(M, ...) fprintf(LOG_DEST, "[WARN] " M "\n", ## __VA_ARGS__)
#define check_alloc_status(A, M, ...) \
    if (A == NULL) {log_err(M, ## __VA_ARGS__); }
#define check_nc_status(A, M, ...) \
    if (A != 0) {log_err(M, ## __VA_ARGS__); }

typedef struct {
    double area;
    double latitude;
    double longitude;
    size_t global_idx;
} location_struct;

typedef struct {
    size_t           ncells_active;
    size_t           n_nx;
    size_t           n_ny;
    location_struct *locations;
} domain_struct;

typedef struct {
    double dt;
    size_t model_steps_per_day;
} global_param_struct;

typedef struct {
    bool GWM;
} option_struct;

typedef struct {
    char nc_filename[MAXSTRING];
    int  nc_id;
} nc_file_struct;

enum {
    OUT_RUNOFF,
    OUT_BASEFLOW,
    N_OUTVAR_TYPES
};

bool assert_close_double(double x, double y, double rtol, double abs_tol);
void sprint_location(char *str, location_struct *loc);
int nc_open(const char *path, int mode, int *ncidp);
int nc_close(int ncid);
void get_scatter_nc_field_int(nc_file_struct *, char *, size_t *, size_t *,
                              int *);
void get_scatter_nc_field_double(nc_file_struct *, char *, size_t *,
                                 size_t *, double *);
void get_active_nc_field_int(nc_file_struct *, char *, size_t *, size_t *,
                             int *);
void scatter_int(int *, int *);
void scatter_size_t(size_t *, size_t *);
void scatter_size_t_2d(size_t **, size_t **, int);
void gather_size_t(size_t *, size_t *);
void gather_size_t_2d(size_t **, size_t **, int);

#endif
