#!/bin/bash
# Build ref_route, the oracle of test_routing_vs_rvic.py.
#
# rout_convolution.c and cshift.c are compiled verbatim from PRISTINE VIC
# 5.1.0's RVIC extension (VIC 5.0.1 does not ship it). Only the type
# declarations around them come from rout_shim/rout.h, because the real
# rout.h pulls in the image driver's NetCDF and MPI headers, which the two
# translation units under test do not use.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
REF="$REFERENCE/VIC-5.1.0/vic/extensions/rout_rvic/src"
need_tree "$REF/rout_convolution.c"
${CC:-cc} -o "$OUT/ref_route" "$HERE/ref_route.c" \
  "$REF/rout_convolution.c" "$REF/cshift.c" \
  -I"$HERE/rout_shim" -O2 -std=c99 -Wall -Wextra -lm
echo "built $OUT/ref_route against $REF"
