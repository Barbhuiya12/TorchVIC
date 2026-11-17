#!/bin/bash
# Build ref_vicwur_lu, the oracle of VIC-WUR land use
# (tests/test_vicwur_landuse.py).  The VIC-WUR 3.0 lu_apply.c, lu_support.c and the
# whole VIC-WUR 3.0 vic_run are compiled unchanged against the real VIC-WUR 3.0 headers;
# vicwur_lu_shim/ supplies the image-driver types.
#
#   TORCHVIC_VICWUR_SRC  the VIC-WUR 3.0 source tree (VICWUR_SRC also read; default
#                        $TORCHVIC_DATA_ROOT/build/vicwur3/src_tree/vic)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SRC="${TORCHVIC_VICWUR_SRC:-${VICWUR_SRC:-${TORCHVIC_DATA_ROOT:-.}/build/vicwur3/src_tree/vic}}"
P="$SRC/plugins"
need_tree "$P/landuse/src/lu_apply.c"
INC="-I$HERE/vicwur_lu_shim -I$SRC/vic_run/include \
  -I$SRC/drivers/shared_all/include -I$P/general/include \
  -I$P/landuse/include -I$P/routing/include -I$P/irrigation/include"
SRCS="$P/landuse/src/lu_apply.c $P/landuse/src/lu_support.c \
  $(ls $SRC/vic_run/src/*.c) \
  $SRC/drivers/shared_all/src/initialize_options.c \
  $SRC/drivers/shared_all/src/initialize_parameters.c \
  $SRC/drivers/shared_all/src/vic_log.c \
  $SRC/drivers/shared_all/src/open_file.c"
# -std=c99 keeps fp-contract off, as in the VIC-WUR 3.0 Makefile; -fcommon for the
# tentative definitions in the VIC-WUR 3.0 headers.
${CC:-cc} -o "$OUT/ref_vicwur_lu" "$HERE/ref_vicwur_lu.c" $SRCS $INC \
  -DLOG_LVL=25 -DGIT_VERSION='"ref-vicwur3"' -DUSERNAME="\"${USER:-unknown}\"" \
  -DHOSTNAME="\"$(uname -n)\"" -O0 -std=c99 -ffp-contract=off -fcommon \
  -D_DEFAULT_SOURCE -w -lm
echo "built $OUT/ref_vicwur_lu against $SRC"
