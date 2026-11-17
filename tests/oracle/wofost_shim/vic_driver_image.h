/* Stand-in for the VIC-WUR 3.0 image-driver headers in the WOFOST-NPK oracle
 * (tests/oracle/ref_vicwur_wofost.c).  Only the fields that
 * plugins/crops/src/crop_run.c, wofost_run.c, wofost_update_step_vars.c and
 * plugins/wofost/src read are declared; the oracle owns the globals.
 * leap_year is VIC-WUR 3.0 drivers/shared_all/src/vic_time.c:652-676 for the
 * calendars the oracle uses (standard / proleptic gregorian / noleap).
 */
#ifndef WOFOST_SHIM_DRIVER_H
#define WOFOST_SHIM_DRIVER_H

#include <vic_def.h>

#define CALENDAR_STANDARD 0
#define CALENDAR_GREGORIAN 1
#define CALENDAR_PROLEPTIC_GREGORIAN 2
#define CALENDAR_NOLEAP 3

typedef struct {
    size_t atmos_steps_per_day;
    size_t model_steps_per_day;
    unsigned short int calendar;
} global_param_struct;

typedef struct {
    size_t SNOW_BAND;
} option_struct;

typedef struct {
    double *Tfactor;
    double *AreaFract;
    double ph;
    double carbon;
} soil_con_struct;

typedef struct {
    double *air_temp;
    double *shortwave;
} force_data_struct;

typedef struct {
    double CO2;
} co2_force_struct;

typedef struct {
    double pot_evap;
    double water_stress;
} cell_data_struct;

typedef struct {
    cell_data_struct **cell;
} all_vars_struct;

typedef struct {
    int *vidx;
} veg_con_map_struct;

typedef struct {
    double Cv;
} veg_con_struct;

static inline bool
leap_year(unsigned short int year, unsigned short int calendar)
{
    bool leap = false;
    if ((calendar == CALENDAR_STANDARD || calendar == CALENDAR_GREGORIAN ||
         calendar == CALENDAR_PROLEPTIC_GREGORIAN) && (year % 4 == 0)) {
        leap = true;
        if ((calendar == CALENDAR_PROLEPTIC_GREGORIAN) &&
            (year % 100 == 0) && (year % 400 != 0)) {
            leap = false;
        }
        else if ((calendar == CALENDAR_STANDARD ||
                  calendar == CALENDAR_GREGORIAN) &&
                 (year % 100 == 0) && (year % 400 != 0) && (year > 1583)) {
            leap = false;
        }
    }
    return leap;
}

void cshift(double *, int, int, int, int);

/* opaque types named in crop.h prototypes */
typedef struct nc_file_struct nc_file_struct;
typedef struct nc_var_struct nc_var_struct;
typedef struct stream_struct stream_struct;

#endif
