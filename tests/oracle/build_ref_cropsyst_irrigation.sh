#!/bin/bash
# Build ref_cropsyst_irrigation, the oracle of tests/test_cropsyst_irrigation.py
# (E12 irrigation trigger/amount, E13 mechanistic Ed/Ro and nozzle table, E14 Saxton PTF, E15 PET_SHORT).
#
# Same link recipe as build_ref_cropsyst_uptake.sh (compiled VIC-CropSyst reference).
# Two ranges of the shipped source are cut out and compiled into the harness, because the reference
# build does not contain them: Ed (evaporation_from_irrigation_systems, behind MECHANISTIC_IRRIGATION,
# off in the build) and the Saxton block of read_soilparam (inside a file-reading function).
# The cut is anchored on text, so a changed source fails the build instead of being cut wrongly.
#
#   TORCHVIC_CROPSYST_PKG  the built VIC-CropSyst-Package (default $TORCHVIC_DATA_ROOT/build/vic_cropsyst/VIC-CropSyst-Package)
#   TORCHVIC_CROPSYST_SRC  the pristine package source (default $TORCHVIC_DATA_ROOT/static/upstream/cropsyst/VIC-CropSyst-Package)
#   TORCHVIC_ORACLE_DIR    where the binary goes (default <repository>/build/oracle)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PKG="${TORCHVIC_CROPSYST_PKG:-}"
SRC="${TORCHVIC_CROPSYST_SRC:-}"
if [ -z "$PKG" ] && [ -n "${TORCHVIC_DATA_ROOT:-}" ]; then
    c="$TORCHVIC_DATA_ROOT/build/vic_cropsyst/VIC-CropSyst-Package"
    if [ -d "$c/VIC_CropSyst" ]; then PKG="$(cd "$c" && pwd)"; fi
fi
if [ -z "$SRC" ] && [ -n "${TORCHVIC_DATA_ROOT:-}" ]; then
    c="$TORCHVIC_DATA_ROOT/static/upstream/cropsyst/VIC-CropSyst-Package"
    if [ -d "$c/VIC_CropSyst" ]; then SRC="$(cd "$c" && pwd)"; fi
fi
REL="$PKG/VIC_CropSyst/build/gcc/Release"
need_tree "$REL/vicNl.o"
need_tree "$SRC/VIC_CropSyst/crop/VIC_soil_hydrology.c"
MK="$PKG/VIC_CropSyst/build/Xcc/Release/Makefile_Kamiak"
DEFINES="$(sed -n 's/^DEFINES *= *//p' "$MK")"
INCPATH="$(sed -n 's/^INCPATH *= *//p' "$MK")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# Ed: the one '#ifdef MECHANISTIC_IRRIGATION' ... '#endif //MECHANISTIC_IRRIGATION' range
SH="$SRC/VIC_CropSyst/crop/VIC_soil_hydrology.c"
[ "$(grep -c '^#ifdef MECHANISTIC_IRRIGATION$' "$SH")" = 1 ] || { echo "Ed anchor changed" >&2; exit 1; }
sed -n '/^#ifdef MECHANISTIC_IRRIGATION$/,/^#endif \/\/MECHANISTIC_IRRIGATION/p' "$SH" > "$TMP/ed_ref.inc"
# Saxton block: from 'if(options.VCS.clay_input==1){' to the S_max assignment (read_soilparam.c:723-813)
RS="$SRC/VIC_CropSyst/read_soilparam.c"
a="$(grep -n '^     if(options.VCS.clay_input==1){' "$RS" | head -1 | cut -d: -f1)"
b="$(grep -n 'temp.VCS.b_campbell\[0\]);' "$RS" | head -1 | cut -d: -f1)"
[ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ] || { echo "Saxton anchor changed" >&2; exit 1; }
sed -n "${a},${b}p" "$RS" > "$TMP/saxton_ref.inc"
# Deficit-irrigation path (FULL_IRRIGATION undefined): need_irrigation, amount_of_deficit_irrigation,
# read_irrig_pattern and the pattern struct, cut from the shipped source by text anchors.
sed -n '/^bool need_irrigation(/,/^}/p' "$SH" > "$TMP/deficit_ref.inc"
sed -n '/^double amount_of_deficit_irrigation(/,/^};/p' "$SH" >> "$TMP/deficit_ref.inc"
sed -i 's/global_param\.VCS\.basin_wide_proration/g_basin_proration/' "$TMP/deficit_ref.inc"
sed -n '/^int read_irrig_pattern(/,/^}/p' "$SRC/VIC_CropSyst/read_croplib.c" > "$TMP/readpat_ref.inc"
sed -n '/^typedef struct {$/{N;N;/irrig_amount;.*\n.*proration_rate/{N;p}}' "$SRC/VIC_CropSyst_common/VCS_Nl_def.h" > "$TMP/pattern_struct.inc"
for f in deficit_ref readpat_ref pattern_struct; do [ -s "$TMP/$f.inc" ] || { echo "$f anchor changed" >&2; exit 1; }; done
objcopy --redefine-sym main=vicnl_main_unused "$REL/vicNl.o" "$TMP/vicNl_nomain.o"
OBJS="$(ls "$REL"/*.o | grep -v '/vicNl\.o$') $TMP/vicNl_nomain.o"
cd "$REL"
g++ -pipe -fpermissive -w -g -O0 -std=gnu++0x -D_REENTRANT -fPIC $DEFINES $INCPATH \
    -I"$PKG/VIC_CropSyst" -I"$TMP" -c "$HERE/ref_cropsyst_irrigation.cpp" -o "$TMP/ref_cropsyst_irrigation.o"
g++ -o "$OUT/ref_cropsyst_irrigation" "$TMP/ref_cropsyst_irrigation.o" $OBJS -ldl \
    -Wl,--wrap=VIC_CropSyst_get \
    -L"$PKG/CS_suite/build/gcc/Release" -lCS_suite_static \
    -L"$PKG/UED/library/build/gcc/Release" -lUED_static \
    -L"$PKG/corn/build/gcc/Release" -lcorn_complete_static \
    -L"$PKG/VIC_CropSyst/build/library/gcc/Release" -lCropSyst_element_V5_for_VIC_V3 \
    -L"$PKG/common/weather/build/gcc/Release" -lweather_static
echo "built $OUT/ref_cropsyst_irrigation against $PKG"
