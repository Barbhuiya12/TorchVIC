/* Stand-in for VIC-WUR 3.0 vic_run/include/vic_def.h in the WOFOST-NPK oracle
 * (tests/oracle/ref_vicwur_wofost.c).  Only what plugins/wofost and
 * plugins/crops touch.  The physical constants are the real VIC-WUR 3.0 header
 * (vic_run/include/vic_physical_constants.h, on the include path after this
 * directory).  min/max are copied verbatim from vic_def.h:91-92, without the
 * outer parentheses, because the WOFOST sources rely on that exact macro.
 * dmy_struct follows vic_def.h:762-772.
 */
#ifndef WOFOST_SHIM_VIC_DEF_H
#define WOFOST_SHIM_VIC_DEF_H

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vic_physical_constants.h>

#define MAXSTRING 2048
#define MISSING_USI 99999
#define NODATA_VEG -1
#define LOG_DEST stderr

#define min(a, b) (a < b) ? a : b
#define max(a, b) (a > b) ? a : b

/* vic_log.h:111-114: log_err exits */
#define log_err(M, ...) { fprintf(LOG_DEST, "[ERROR] " M "\n", ## __VA_ARGS__); \
                          exit(EXIT_FAILURE); }
#define log_warn(M, ...) fprintf(LOG_DEST, "[WARN] " M "\n", ## __VA_ARGS__)
#define check_alloc_status(A, M, ...) \
    if (A == NULL) {log_err(M, ## __VA_ARGS__); }

typedef struct {
    unsigned short int year;
    unsigned short int month;
    unsigned short int day;
    unsigned short int day_in_year;
    unsigned int dayseconds;
} dmy_struct;

#endif
