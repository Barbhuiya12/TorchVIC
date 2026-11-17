#!/usr/bin/env bash
# Fill the test-data directories the test suite reads (all git-ignored):
#
#   reference/            pristine VIC releases with their image drivers built
#                         (VIC-VIC.5.0.1, VIC-5.1.0, VIC-VIC.4.2.d)
#   data/VIC_sample_data  UW-Hydro's sample data (WalnutGulch, FindleyLake, ...)
#   data/samples          VIC's own Stehekin samples
#   data/India_basin37_Nandgaon  optional: a 100-cell gauged basin of the
#                         Indian 1/16 deg dataset (tests skip without it)
#
# Usage:
#   tools/get_test_data.sh                      download and build
#   tools/get_test_data.sh --link REF SD SAMP [B37]   link existing copies
#
# Building the image drivers needs nc-config (NetCDF-C) and mpicc on PATH.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO/data"

if [ "${1:-}" = "--link" ]; then
    ln -sfn "$(readlink -f "$2")" "$REPO/reference"
    ln -sfn "$(readlink -f "$3")" "$REPO/data/VIC_sample_data"
    ln -sfn "$(readlink -f "$4")" "$REPO/data/samples"
    [ -n "${5:-}" ] && ln -sfn "$(readlink -f "$5")" "$REPO/data/India_basin37_Nandgaon"
    echo "linked reference/, data/VIC_sample_data, data/samples"
    exit 0
fi

for TOOL in nc-config mpicc curl git; do
    command -v "$TOOL" >/dev/null 2>&1 || {
        echo "get_test_data.sh: '$TOOL' not found on PATH." >&2
        echo "  Building VIC's image driver needs nc-config (NetCDF-C" >&2
        echo "  development files) and mpicc (an MPI C compiler); the" >&2
        echo "  downloads need curl and git. Install them (or load the" >&2
        echo "  modules) and run again." >&2
        exit 1
    }
done

REF="$REPO/reference"
mkdir -p "$REF"
cd "$REF"
# upstream tags: VIC.5.0.1 and VIC.4.2.d, but 5.1.0 (no "VIC." prefix)
for TAG in VIC.5.0.1 5.1.0 VIC.4.2.d; do
    DIR="VIC-$TAG"
    [ "$TAG" = 5.1.0 ] && DIR="VIC-5.1.0"
    if [ ! -d "$DIR" ]; then
        curl -fsSL "https://github.com/UW-Hydro/VIC/archive/refs/tags/$TAG.tar.gz" | tar xz
        [ "$TAG" = 5.1.0 ] && [ -d VIC-VIC.5.1.0 ] && mv VIC-VIC.5.1.0 VIC-5.1.0
    fi
done
# image drivers, flags as used for the paper's reference runs:
# -O2, -fcommon (VIC 5.0.1 predates GCC 10's -fno-common default)
for DIR in VIC-VIC.5.0.1 VIC-5.1.0; do
    (cd "$DIR/vic/drivers/image" && [ -x vic_image.exe ] || make CFLAGS="\
-I ./include -I ../../vic_run/include -I ../shared_all/include \
-I ../shared_image/include -I ../../extensions/rout_stub/include \
$(nc-config --cflags) -O2 -std=c99 -fcommon -DLOG_LVL=5")
done

[ -d "$REPO/data/VIC_sample_data" ] || git clone --depth 1 \
    https://github.com/UW-Hydro/VIC_sample_data.git "$REPO/data/VIC_sample_data"
[ -e "$REPO/data/samples" ] || ln -s VIC_sample_data "$REPO/data/samples"
echo "test data ready under $REPO/reference and $REPO/data"
