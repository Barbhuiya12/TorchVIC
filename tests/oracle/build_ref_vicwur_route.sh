#!/bin/bash
# Build ref_vicwur_route, the oracle of the VIC-WUR routing cascade
# (tests/test_vicwur_routing.py). The VIC-WUR 3.0 sources are #included unchanged;
# vicwur_shim/ supplies the driver types and the NetCDF/MPI stubs.
#
#   TORCHVIC_VICWUR_SRC  the VIC-WUR 3.0 source tree (VICWUR_SRC also read; default
#                        $TORCHVIC_DATA_ROOT/build/vicwur3/src_tree/vic)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SRC="${TORCHVIC_VICWUR_SRC:-${VICWUR_SRC:-${TORCHVIC_DATA_ROOT:-.}/build/vicwur3/src_tree/vic}}"
P="$SRC/plugins"
need_tree "$P/routing/src/rout_basin_run.c"
# -std=c99 keeps fp-contract off, as in the VIC-WUR 3.0 Makefile.
${CC:-cc} -o "$OUT/ref_vicwur_route" "$HERE/ref_vicwur_route.c" \
  -I"$HERE/vicwur_shim" -I"$P/routing/include" \
  -I"$P/routing/src" -I"$P/general/src" -I"$P/wateruse/src" \
  -O3 -std=c99 -D_DEFAULT_SOURCE -Wall -Wno-unused-variable \
  -Wno-unused-but-set-variable -Wno-absolute-value -Wno-maybe-uninitialized -lm
echo "built $OUT/ref_vicwur_route against $SRC"
