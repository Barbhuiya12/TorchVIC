#!/bin/bash
# Build ref_vicwur_co2, the oracle of the VIC-WUR CO2 forcing and
# b_co2 hook (tests/test_vicwur_co2.py). The VIC-WUR 3.0 vic_run/src/penman.c is
# compiled unchanged against the real vic_run headers; the functions it calls
# that the oracle does not exercise (photosynthesis, svp_slope, ...) are
# dropped by --gc-sections.
#
#   TORCHVIC_VICWUR_SRC  the VIC-WUR 3.0 source tree (VICWUR_SRC also read; default
#                        $TORCHVIC_DATA_ROOT/build/vicwur3/src_tree/vic)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SRC="${TORCHVIC_VICWUR_SRC:-${VICWUR_SRC:-${TORCHVIC_DATA_ROOT:-.}/build/vicwur3/src_tree/vic}}"
need_tree "$SRC/vic_run/src/penman.c"
# -std=c99 keeps fp-contract off, as in the VIC-WUR 3.0 Makefile.
${CC:-cc} -o "$OUT/ref_vicwur_co2" "$HERE/ref_vicwur_co2.c" \
  "$SRC/vic_run/src/penman.c" \
  -I"$SRC/vic_run/include" -I"$SRC/drivers/shared_all/include" \
  -O0 -std=c99 -ffp-contract=off -fcommon -ffunction-sections \
  -Wl,--gc-sections -Wno-unused-variable -lm
echo "built $OUT/ref_vicwur_co2 against $SRC"
