#!/bin/bash
# Build ref_mtclim, the oracle of test_mtclim.py and test_daily_forcing.py,
# against PRISTINE VIC 4.2.d: VIC 5 removed MTCLIM, so 4.2.d is the only
# authority for the daily-to-sub-daily disaggregation. -fcommon is required
# because VIC declares its globals tentatively.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
REF="$REFERENCE/VIC-VIC.4.2.d/src"
need_tree "$REF/mtclim_vic.c"
SRCS="$REF/mtclim_vic.c $REF/mtclim_wrapper.c $REF/svp.c \
      $REF/calc_air_temperature.c $REF/initialize_global.c \
      $REF/nrerror.c $REF/vicerror.c $REF/calc_longwave.c"
${CC:-cc} -o "$OUT/ref_mtclim" "$HERE/ref_mtclim.c" $SRCS -I"$REF" \
   -O2 -std=c99 -fcommon -w -lm
echo "built $OUT/ref_mtclim against $REF"
