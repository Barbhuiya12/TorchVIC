/* Minimal stand-in for the VIC image driver headers, so that the VIC-WUR dam
 * plugin (<data root>/build/vicwur3/src_tree/vic/plugins/dams, VIC-WUR 3.0) compiles
 * unchanged in ref_vicwur_dam.c without NetCDF or MPI.  Separate from
 * vicwur_shim/ (the routing oracle's) because the dams need more of the
 * driver: the date array dmy, the calendar and the global parameters.
 *
 * Constants and types copied from VIC-WUR 3.0: MM_PER_M, M3_PER_HM3
 * (vic_run/include/vic_physical_constants.h:51,55), MONTHS_PER_YEAR (:37),
 * min/max (vic_run/include/vic_def.h:91-92), dmy_struct (vic_def.h:766-772),
 * the calendars enum (drivers/shared_all/include/vic_driver_shared_all.h:
 * 428-438), leap_year (drivers/shared_all/src/vic_time.c:652-673).
 * log_err exits, as in vic_run/include/vic_log.h:111-114.
 */
#ifndef VICWUR_DAM_SHIM_DRIVER_H
#define VICWUR_DAM_SHIM_DRIVER_H

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
#define MONTHS_PER_YEAR 12
#define DAYS_PER_YEAR 365
#define DAYS_PER_JYEAR 365.25
#define SEC_PER_DAY 86400
#define MISSING_USI 99999
#define MAXSTRING 2048
#define min(a, b) (a < b) ? a : b
#define max(a, b) (a > b) ? a : b

#define log_err(M, ...) fprintf(stderr, "[ERROR] " M "\n", ## __VA_ARGS__); \
    exit(EXIT_FAILURE);
#define log_warn(M, ...) fprintf(stderr, "[WARN] " M "\n", ## __VA_ARGS__)

enum calendars {
    CALENDAR_STANDARD,
    CALENDAR_GREGORIAN,
    CALENDAR_PROLEPTIC_GREGORIAN,
    CALENDAR_NOLEAP,
    CALENDAR_365_DAY,
    CALENDAR_360_DAY,
    CALENDAR_JULIAN,
    CALENDAR_ALL_LEAP,
    CALENDAR_366_DAY
};

typedef struct {
    unsigned short int day;
    unsigned short int day_in_year;
    unsigned short int month;
    int year;
    unsigned int dayseconds;
} dmy_struct;

typedef struct {
    double area;
} location_struct;

typedef struct {
    size_t           ncells_active;
    location_struct *locations;
} domain_struct;

typedef struct {
    double             dt;
    size_t             model_steps_per_day;
    unsigned short int calendar;
} global_param_struct;

typedef struct nc_file_struct nc_file_struct;
typedef struct stream_struct stream_struct;
typedef struct nc_var_struct nc_var_struct;

enum {
    OUT_RUNOFF,
    N_OUTVAR_TYPES
};

bool leap_year(unsigned short int, unsigned short int);
unsigned short int days_per_month(unsigned short int, unsigned short int,
                                  unsigned short int);
double array_average(double *, size_t, size_t, size_t, size_t);
void cshift(double *, int, int, int, int);
void double_flip(double *, size_t);
double between_jday(double, double, double);

#endif
