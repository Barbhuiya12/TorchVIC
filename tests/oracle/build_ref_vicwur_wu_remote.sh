#!/bin/bash
# Build ref_vicwur_wu_remote, the oracle of the VIC-WUR remote water
# use (tests/test_vicwur_wu_remote.py). The VIC-WUR 3.0 source wu_remote.c is
# #included unchanged; vicwur_shim_rm/ and vicwur_shim/ supply the driver
# types and the MPI stub.
#
#   TORCHVIC_VICWUR_SRC  the VIC-WUR 3.0 source tree (VICWUR_SRC also read; default
#                        $TORCHVIC_DATA_ROOT/build/vicwur3/src_tree/vic)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SRC="${TORCHVIC_VICWUR_SRC:-${VICWUR_SRC:-${TORCHVIC_DATA_ROOT:-.}/build/vicwur3/src_tree/vic}}"
P="$SRC/plugins"
need_tree "$P/wateruse/src/wu_remote.c"
# -std=c99 -ffp-contract=off keeps the arithmetic as in the VIC-WUR 3.0 Makefile.
${CC:-cc} -o "$OUT/ref_vicwur_wu_remote" "$HERE/ref_vicwur_wu_remote.c" \
  -I"$HERE/vicwur_shim_rm" -I"$HERE/vicwur_shim" -I"$P/routing/include" \
  -I"$P/wateruse/include" -I"$P/efr/include" -I"$P/wateruse/src" \
  -O0 -std=c99 -ffp-contract=off -D_DEFAULT_SOURCE -Wall \
  -Wno-unused-variable -Wno-unused-but-set-variable -Wno-absolute-value \
  -Wno-maybe-uninitialized -Wno-unused-function -lm
echo "built $OUT/ref_vicwur_wu_remote against $SRC"
