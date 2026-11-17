#!/bin/bash
# Build ref_snow, the oracle of test_snow_formulas.py, test_blowing_snow.py,
# test_carbon.py, test_lake.py, test_t_profile.py and test_zwt_lumped.py,
# against PRISTINE VIC 5.0.1. -fcommon is required because VIC 5.0.1 declares
# its globals tentatively, which gcc 10 and later reject by default.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
REF="$REFERENCE/VIC-VIC.5.0.1/vic"
need_tree "$REF/vic_run/src"
INC="-I$REF/drivers/classic/include -I$REF/drivers/shared_all/include \
     -I$REF/vic_run/include"
SRCS="$(ls $REF/vic_run/src/*.c) \
  $REF/drivers/shared_all/src/initialize_options.c \
  $REF/drivers/shared_all/src/initialize_global.c \
  $REF/drivers/shared_all/src/initialize_parameters.c \
  $REF/drivers/shared_all/src/vic_log.c \
  $REF/drivers/shared_all/src/open_file.c \
  $REF/drivers/shared_all/src/soil_moisture_from_water_table.c \
  $REF/drivers/shared_all/src/compute_lake_params.c \
  $REF/drivers/shared_all/src/make_all_vars.c \
  $REF/drivers/shared_all/src/make_cell_data.c \
  $REF/drivers/shared_all/src/make_energy_bal.c \
  $REF/drivers/shared_all/src/make_snow_data.c \
  $REF/drivers/shared_all/src/make_veg_var.c"
${CC:-cc} -o "$OUT/ref_snow" "$HERE/ref_snow.c" $SRCS $INC \
   -DLOG_LVL=25 -DGIT_VERSION='"ref-5.0.1"' -DUSERNAME="\"${USER:-unknown}\"" \
   -DHOSTNAME="\"$(uname -n)\"" -O2 -std=c99 -fcommon -w -lm
echo "built $OUT/ref_snow against $REF"
