#!/bin/bash
# Build ref_cropsyst_orchard, the oracle of tests/test_cropsyst_orchard.py (orchard and perennial fruit crops).
#
# It links ref_cropsyst_orchard.cpp against the COMPILED VIC-CropSyst reference
# build (docs/connectors/cropsyst.md): every object of
# VIC_CropSyst/build/gcc/Release except that vicNl.o's main() is renamed, plus
# the five static libraries the shipped Makefile links. Compiler flags and
# defines are those of VIC_CropSyst/build/Xcc/Release/Makefile_Kamiak.
#
#   TORCHVIC_CROPSYST_PKG  the built VIC-CropSyst-Package (default
#                          $TORCHVIC_DATA_ROOT/build/vic_cropsyst/VIC-CropSyst-Package)
#   TORCHVIC_ORACLE_DIR    where the binary goes (default <repository>/build/oracle)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PKG="${TORCHVIC_CROPSYST_PKG:-}"
if [ -z "$PKG" ] && [ -n "${TORCHVIC_DATA_ROOT:-}" ]; then
    c="$TORCHVIC_DATA_ROOT/build/vic_cropsyst/VIC-CropSyst-Package"
    if [ -d "$c/VIC_CropSyst" ]; then PKG="$(cd "$c" && pwd)"; fi
fi
REL="$PKG/VIC_CropSyst/build/gcc/Release"
need_tree "$REL/vicNl.o"
need_tree "$PKG/VIC_CropSyst/build/library/gcc/Release/libCropSyst_element_V5_for_VIC_V3.a"
MK="$PKG/VIC_CropSyst/build/Xcc/Release/Makefile_Kamiak"
DEFINES="$(sed -n 's/^DEFINES *= *//p' "$MK")"
INCPATH="$(sed -n 's/^INCPATH *= *//p' "$MK")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
objcopy --redefine-sym main=vicnl_main_unused "$REL/vicNl.o" "$TMP/vicNl_nomain.o"
OBJS="$(ls "$REL"/*.o | grep -v '/vicNl\.o$') $TMP/vicNl_nomain.o"
# INCPATH is relative to the Release directory, as in the shipped build.
cd "$REL"
g++ -pipe -fpermissive -w -g -O0 -std=gnu++0x -D_REENTRANT -fPIC $DEFINES $INCPATH \
    -I"$PKG/VIC_CropSyst" -c "$HERE/ref_cropsyst_orchard.cpp" -o "$TMP/ref_cropsyst_orchard.o"
g++ -o "$OUT/ref_cropsyst_orchard" "$TMP/ref_cropsyst_orchard.o" $OBJS -ldl \
    -L"$PKG/CS_suite/build/gcc/Release" -lCS_suite_static \
    -L"$PKG/UED/library/build/gcc/Release" -lUED_static \
    -L"$PKG/corn/build/gcc/Release" -lcorn_complete_static \
    -L"$PKG/VIC_CropSyst/build/library/gcc/Release" -lCropSyst_element_V5_for_VIC_V3 \
    -L"$PKG/common/weather/build/gcc/Release" -lweather_static
echo "built $OUT/ref_cropsyst_orchard against $PKG"
