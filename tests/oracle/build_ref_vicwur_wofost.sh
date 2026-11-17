#!/bin/bash
# Build ref_vicwur_wofost (float, as the C model) and ref_vicwur_wofost_f64
# (-DWOFOST_DOUBLE: every float of the WOFOST sources is double), the
# oracles of the VIC-WUR WOFOST-NPK kernel (tests/test_vicwur_wofost.py).
# The VIC-WUR 3.0 sources are #included unchanged; wofost_shim/ supplies the driver
# types.
#
#   TORCHVIC_VICWUR_SRC  the VIC-WUR 3.0 source tree (VICWUR_SRC also read; default
#                        $TORCHVIC_DATA_ROOT/build/vicwur3/src_tree/vic)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SRC="${TORCHVIC_VICWUR_SRC:-${VICWUR_SRC:-${TORCHVIC_DATA_ROOT:-.}/build/vicwur3/src_tree/vic}}"
P="$SRC/plugins"
need_tree "$P/wofost/src/Assim.c"
# -std=c99 keeps fp-contract off, as in the VIC-WUR 3.0 Makefile (-O3 -std=c99).
for v in f32 f64; do
    name=ref_vicwur_wofost; def=
    if [ $v = f64 ]; then name=ref_vicwur_wofost_f64; def=-DWOFOST_DOUBLE; fi
    ${CC:-cc} -o "$OUT/$name" $def "$HERE/ref_vicwur_wofost.c" \
      -I"$HERE/wofost_shim" -I"$SRC/vic_run/include" \
      -I"$P/wofost/include" -I"$P/crops/include" \
      -I"$P/wofost/src" -I"$P/crops/src" -I"$P/general/src" \
      -O3 -std=c99 -fcommon -D_DEFAULT_SOURCE -Wall -Wno-unused-variable \
      -Wno-unused-but-set-variable -Wno-unused-function \
      -Wno-maybe-uninitialized -lm
    echo "built $OUT/$name against $SRC"
done
