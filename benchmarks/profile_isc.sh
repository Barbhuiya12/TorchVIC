#!/bin/bash
# Profiling sweep of TorchVIC's time loop on the ISC domain (see
# profile_isc_torchvic.py). Run inside a one-GPU allocation:
#   TORCHVIC_DATA_ROOT=/path/to/data benchmarks/profile_isc.sh TAG [extra args]
# (under srun/sbatch with one GPU, e.g. --gres=gpu:1 -c 16, on your cluster)
# Environment: TORCHVIC_DATA_ROOT (required), TORCHVIC_PYTHON (default python),
# TORCHVIC_COMPILE_CACHE (default $HOME/.cache/torchvic), GLOBAL, CACHE.
# Writes benchmarks/results_isc/profile/TAG_*.{log,json}.
set -uo pipefail
TAG=$1; shift
: "${TORCHVIC_DATA_ROOT:?set TORCHVIC_DATA_ROOT}"
PY=${TORCHVIC_PYTHON:-python}
CC_DIR=${TORCHVIC_COMPILE_CACHE:-$HOME/.cache/torchvic}
B=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$B/results_isc/profile}
GLOBAL=${GLOBAL:-$TORCHVIC_DATA_ROOT/benchmark_isc/runs/tv_3h_r1/global.txt}
CACHE=${CACHE:-$TORCHVIC_DATA_ROOT/benchmark_isc/profile_cache/jan2004_3h_all.pt}
CELLS=${CELLS:-"4096 16384 65536 0"}
DAYS=${DAYS:-10}
mkdir -p $OUT
# the torchvic of the tree this script is in
export PYTHONPATH=$(dirname "$B") PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=$CC_DIR/inductor TRITON_CACHE_DIR=$CC_DIR/triton
for n in $CELLS; do
    $PY $B/profile_isc_torchvic.py $GLOBAL --cache $CACHE \
        --cells $n --days $DAYS --start-day 1 --json $OUT/${TAG}_c$n.json "$@" \
        > $OUT/${TAG}_c$n.log 2>&1
    tail -3 $OUT/${TAG}_c$n.log | head -1
done
