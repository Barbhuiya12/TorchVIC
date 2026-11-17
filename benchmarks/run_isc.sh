#!/bin/bash
# One timed ISC benchmark run, end to end, under /usr/bin/time -v.
# Run on the machine being measured (inside a batch allocation on a cluster).
# Environment: TORCHVIC_DATA_ROOT (required), TORCHVIC_PYTHON (default python),
# NETCDF_HOME (netCDF/MPI install, default /usr), TORCHVIC_REFERENCE_DIR
# (pristine VIC trees, default reference), TORCHVIC_COMPILE_CACHE.
#
#   run_isc.sh vic      NAME NRANKS NTHREADS [EXE]      pristine VIC 5.1.0
#                       (SNOW_BAND=FALSE in the environment: one band per cell)
#   run_isc.sh torchvic NAME 3h|daily DEVICE [ARGS...]  benchmarks/run_isc_torchvic.py
#
# END=YYYY-MM-DD in the environment ends the run early (a labelled short run).
# ISC_GLOBAL=FILE (in benchmarks/isc) replaces the global file of either kind
# (isc_vic5_3h_eb.txt: the energy balance).
# Model output goes to $TORCHVIC_DATA_ROOT/benchmark_isc/runs/NAME/ (overwritten); the
# log, /usr/bin/time report, timing JSON and hardware description go to
# benchmarks/results_isc/logs/NAME.*
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${TORCHVIC_DATA_ROOT:?set TORCHVIC_DATA_ROOT}"
ENVNC=${NETCDF_HOME:-/usr}
PY=${TORCHVIC_PYTHON:-python}
REF=${TORCHVIC_REFERENCE_DIR:-reference}
CC_DIR=${TORCHVIC_COMPILE_CACHE:-$HOME/.cache/torchvic}
KIND=$1; NAME=$2; shift 2
RUN=$TORCHVIC_DATA_ROOT/benchmark_isc/runs/$NAME
LOGS=${LOGS:-$HERE/results_isc/logs}
mkdir -p "$LOGS"
rm -rf "$RUN"; mkdir -p "$RUN/out" "$RUN/log"
ENDSED=()
if [ -n "${END:-}" ]; then
    ENDSED=(-e "s|^ENDYEAR .*|ENDYEAR     ${END:0:4}|"
            -e "s|^ENDMONTH .*|ENDMONTH    $((10#${END:5:2}))|"
            -e "s|^ENDDAY .*|ENDDAY      $((10#${END:8:2}))|")
fi

{ echo "host $(hostname)  date $(date -Is)"; lscpu | grep -E 'Model name|^CPU\(s\)|Socket|Core|Thread'
  free -g | head -2; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true
} > "$LOGS/$NAME.hw"

case $KIND in
vic)
    NP=$1; NT=$2; EXE=${3:-$REF/VIC-5.1.0/vic/drivers/image/vic_image_O3.exe}
    sed -e "s|@DATA_ROOT@|$TORCHVIC_DATA_ROOT|" -e "s|@RESULT_DIR@|$RUN/out/|" -e "s|@LOG_DIR@|$RUN/log/|" \
        -e "s|^SNOW_BAND .*|SNOW_BAND    ${SNOW_BAND:-TRUE}|" \
        -e "s|vic5_3h/isc_3h\.\$|vic5_3h/isc_3h${FORC_SUFFIX:-}.|" \
        "${ENDSED[@]}" \
        "$HERE/isc/${ISC_GLOBAL:-isc_vic5_3h.txt}" > "$RUN/global.txt"
    export LD_LIBRARY_PATH=$ENVNC/lib OMP_NUM_THREADS=$NT OMP_PROC_BIND=spread OMP_PLACES=cores
    echo "exe $EXE  ranks $NP  threads $NT  SNOW_BAND ${SNOW_BAND:-TRUE}  END ${END:-2005-12-31}" >> "$LOGS/$NAME.hw"
    if [ "$NP" -gt 1 ]; then
        # mpirun inside the Slurm step would take the step's one task as its
        # only slot; without the SLURM_* variables it uses the node's cores
        for v in $(env | grep -o '^SLURM[A-Z_]*'); do unset "$v"; done
        CMD=("$ENVNC/bin/mpirun" -np "$NP" --host "localhost:$NP" --bind-to core --map-by core "$EXE" -g "$RUN/global.txt")
    else
        CMD=("$EXE" -g "$RUN/global.txt")
    fi
    ;;
torchvic)
    FORC=$1; DEV=$2; shift 2
    sed -e "s|@DATA_ROOT@|$TORCHVIC_DATA_ROOT|" -e "s|@RESULT_DIR@|$RUN/out/|" -e "s|@LOG_DIR@|$RUN/log/|" \
        -e "s|vic5_3h/isc_3h\.\$|vic5_3h/isc_3h${FORC_SUFFIX:-}.|" \
        "${ENDSED[@]}" \
        "$HERE/isc/${ISC_GLOBAL:-$( [ "$FORC" = daily ] && echo isc_torchvic_daily.txt || echo isc_vic5_3h.txt )}" \
        > "$RUN/global.txt"
    export PYTHONNOUSERSITE=1
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CC_DIR/inductor} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CC_DIR/triton}
    CMD=("$PY" "$HERE/${RUNNER:-run_isc_torchvic.py}" "$RUN/global.txt" --out "$RUN/out" --device "$DEV"
         --timing "$LOGS/$NAME.json" "$@")
    ;;
*) echo "unknown kind $KIND" >&2; exit 2 ;;
esac

echo "${CMD[*]}" > "$LOGS/$NAME.log"
T0=$(date +%s.%N)
/usr/bin/time -v -o "$LOGS/$NAME.time" "${CMD[@]}" >> "$LOGS/$NAME.log" 2>&1
T1=$(date +%s.%N)
echo "wall_s $(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.2f", b-a}')" | tee -a "$LOGS/$NAME.time"
