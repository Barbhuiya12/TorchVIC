#!/bin/bash
# Build ref_soil, the oracle of test_soil_vs_vic.py, test_vic_column.py,
# test_column_formulas.py, test_vic_snow.py, test_vic_frozen.py,
# test_vegetation.py and test_atmos.py.
#
# It calls VIC's own runoff.c, snow_utility.c, frozen_soil.c, penman.c and the
# rest of vic_run, compiled from PRISTINE VIC 5.1.0 -- the release whose
# vic_run (calc_Q12, the explicit new-snow albedo argument of snow_albedo)
# ref_soil.c is written against. soil_thermal_eqn and SnowPackEnergyBalance
# take their arguments as a va_list in the release; ref_soil.c passes them
# through two small variadic wrappers (see the comment above them), which is
# the only change from the harness as first written.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
REF="$REFERENCE/VIC-5.1.0/vic"
need_tree "$REF/vic_run/src/runoff.c"
INC="-I$REF/drivers/classic/include -I$REF/drivers/shared_all/include \
     -I$REF/vic_run/include"
SRCS="$(ls $REF/vic_run/src/*.c) \
  $REF/drivers/shared_all/src/initialize_options.c \
  $REF/drivers/shared_all/src/initialize_global.c \
  $REF/drivers/shared_all/src/initialize_parameters.c \
  $REF/drivers/shared_all/src/vic_log.c \
  $REF/drivers/shared_all/src/open_file.c \
  $REF/drivers/shared_all/src/forcing_utils.c"
${CC:-cc} -o "$OUT/ref_soil" "$HERE/ref_soil.c" $SRCS $INC \
   -DLOG_LVL=25 -DGIT_VERSION='"ref-5.1.0"' -DUSERNAME="\"${USER:-unknown}\"" \
   -DHOSTNAME="\"$(uname -n)\"" -O2 -std=c99 -fcommon -w -lm
echo "built $OUT/ref_soil against $REF"
