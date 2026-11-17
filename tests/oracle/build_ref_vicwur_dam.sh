#!/bin/bash
# Build ref_vicwur_dam, the oracle of the VIC-WUR dams
# (tests/test_vicwur_dams.py). The VIC-WUR 3.0 sources are #included unchanged;
# vicwur_dam_shim/ supplies the driver types and constants.
#
#   TORCHVIC_VICWUR_SRC  the VIC-WUR 3.0 source tree (VICWUR_SRC also read; default
#                        $TORCHVIC_DATA_ROOT/build/vicwur3/src_tree/vic)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SRC="${TORCHVIC_VICWUR_SRC:-${VICWUR_SRC:-${TORCHVIC_DATA_ROOT:-.}/build/vicwur3/src_tree/vic}}"
P="$SRC/plugins"
need_tree "$P/dams/src/dam_operation.c"
# -std=c99 keeps fp-contract off, as in the VIC-WUR 3.0 Makefile.
${CC:-cc} -o "$OUT/ref_vicwur_dam" "$HERE/ref_vicwur_dam.c" \
  -I"$HERE/vicwur_dam_shim" -I"$P/dams/include" -I"$P/routing/include" \
  -I"$P/efr/include" -I"$P/wateruse/include" -I"$P/dams/src" \
  -I"$P/general/src" \
  -O0 -std=c99 -ffp-contract=off -D_DEFAULT_SOURCE -Wall -Wno-unused-variable \
  -Wno-unused-but-set-variable -Wno-absolute-value -Wno-maybe-uninitialized \
  -Wno-unused-function -lm
echo "built $OUT/ref_vicwur_dam against $SRC"
