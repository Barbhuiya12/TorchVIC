#!/bin/bash
# Build ref_vicwur_core, the oracle of the VIC-WUR vic_run hooks and
# the vicwur3 core (tests/test_vicwur_core.py): VIC-WUR 3.0's unchanged vic_run
# (every vic_run/src/*.c) and calc_root_fraction.c, with the shared_all
# initialisers the harness calls.
#
#   TORCHVIC_VICWUR_SRC  the VIC-WUR 3.0 source tree (VICWUR_SRC also read; default
#                        $TORCHVIC_DATA_ROOT/build/vicwur3/src_tree/vic)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SRC="${TORCHVIC_VICWUR_SRC:-${VICWUR_SRC:-${TORCHVIC_DATA_ROOT:-.}/build/vicwur3/src_tree/vic}}"
need_tree "$SRC/vic_run/src/canopy_evap.c"
S="$SRC/drivers/shared_all/src"
# -std=c99 keeps fp-contract off, as in the VIC-WUR 3.0 Makefile.
${CC:-cc} -o "$OUT/ref_vicwur_core" "$HERE/ref_vicwur_core.c" \
  $(ls "$SRC"/vic_run/src/*.c) "$S/calc_root_fraction.c" \
  "$S/initialize_options.c" "$S/initialize_global.c" \
  "$S/initialize_parameters.c" "$S/vic_log.c" "$S/open_file.c" \
  "$S/forcing_utils.c" \
  -I"$SRC/vic_run/include" -I"$SRC/drivers/shared_all/include" \
  -DLOG_LVL=25 -DGIT_VERSION='"ref-w3"' -DUSERNAME="\"${USER:-unknown}\"" \
  -DHOSTNAME="\"$(uname -n)\"" -O0 -std=c99 -ffp-contract=off -fcommon \
  -ffunction-sections -Wl,--gc-sections -w -lm
echo "built $OUT/ref_vicwur_core against $SRC"
