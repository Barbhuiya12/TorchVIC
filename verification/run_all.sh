#!/bin/bash
# Rerun the whole verification: ISC windows, the matrix, the refusal probes,
# the scoreboard. CPU only; the GPU equality check is gpu_check.py.
#
#   TORCHVIC_DATA_ROOT=/path/to/data bash verification/run_all.sh
#
# On a batch cluster submit it with your own partition and limits, e.g.
#   sbatch -p <cpu-partition> -c 32 --mem=128G -t 360 verification/run_all.sh
# Environment: TORCHVIC_DATA_ROOT (required; windows and snapshots are
# written below it), TORCHVIC_PYTHON (default python).
set -euo pipefail
: "${TORCHVIC_DATA_ROOT:?set TORCHVIC_DATA_ROOT}"
cd "$(dirname "$0")/.."
source verification/snapshot.sh
export PYTHONNOUSERSITE=1
# Official rows: VIC's glibc pow/exp/sqrt (torchvic/core/xmath.py, CPU).
# The default and GPU paths may differ from them in the last bit.
export TORCHVIC_EXACT_MATH=${TORCHVIC_EXACT_MATH:-1}
[ "$TORCHVIC_EXACT_MATH" = 1 ] && export ATEN_CPU_CAPABILITY=default
PY=${TORCHVIC_PYTHON:-python}
W=$TORCHVIC_DATA_ROOT/verification/windows
[ -f $W/tntip/isc_3h.2004.nc ] || /usr/bin/env $PY verification/make_isc_windows.py
/usr/bin/env $PY verification/make_variants.py
/usr/bin/env $PY verification/run_matrix.py --jobs 16 --threads 2 "$@"
/usr/bin/env $PY verification/classic_rows.py
/usr/bin/env $PY verification/refusals.py
/usr/bin/env $PY verification/sensitivity.py
/usr/bin/env $PY verification/scoreboard.py
