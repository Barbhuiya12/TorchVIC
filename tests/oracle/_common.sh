# Shared by the build_ref_*.sh scripts: where the pristine VIC trees are and
# where the oracle binaries go.
#
#   TORCHVIC_REFERENCE_DIR  the directory holding VIC-VIC.5.0.1, VIC-5.1.0 and
#                           VIC-VIC.4.2.d as released (default
#                           <repository>/reference)
#   TORCHVIC_ORACLE_DIR     where the binaries are written (default
#                           <repository>/build/oracle)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
REFERENCE="${TORCHVIC_REFERENCE_DIR:-$REPO/reference}"
OUT="${TORCHVIC_ORACLE_DIR:-$REPO/build/oracle}"
mkdir -p "$OUT"

need_tree() {
    if [ ! -e "$1" ]; then
        echo "pristine VIC source not found: $1 (set TORCHVIC_REFERENCE_DIR or TORCHVIC_DATA_ROOT)" >&2
        exit 1
    fi
}
