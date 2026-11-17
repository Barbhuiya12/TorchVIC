#!/bin/bash
# Build ref_vicwur_nonrenew, the oracle of the VIC-WUR non-renewable
# withdrawal (tests/test_vicwur_nonrenew.py). The VIC-WUR 3.0 sources are #included
# unchanged; vicwur_shim_nr/ and vicwur_shim/ supply the driver types.
#
#   TORCHVIC_VICWUR_SRC  the VIC-WUR 3.0 source tree (VICWUR_SRC also read; default
#                        $TORCHVIC_DATA_ROOT/build/vicwur3/src_tree/vic)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SRC="${TORCHVIC_VICWUR_SRC:-${VICWUR_SRC:-${TORCHVIC_DATA_ROOT:-.}/build/vicwur3/src_tree/vic}}"
P="$SRC/plugins"
need_tree "$P/wateruse/src/wu_nonrenewable.c"
# -std=c99 keeps fp-contract off, as in the VIC-WUR 3.0 Makefile.
${CC:-cc} -o "$OUT/ref_vicwur_nonrenew" "$HERE/ref_vicwur_nonrenew.c" \
  -I"$HERE/vicwur_shim_nr" -I"$HERE/vicwur_shim" -I"$P/routing/include" \
  -I"$P/wateruse/include" -I"$P/wateruse/src" -I"$P/routing/src" \
  -O0 -std=c99 -ffp-contract=off -D_DEFAULT_SOURCE -Wall \
  -Wno-unused-variable -Wno-unused-but-set-variable -Wno-absolute-value \
  -Wno-maybe-uninitialized -lm
echo "built $OUT/ref_vicwur_nonrenew against $SRC"
