#!/bin/bash
# Source this: export a read-only copy of the committed TorchVIC package
# (git HEAD, or $VERIFY_REV) on PYTHONPATH, so that a verification run is
# tied to one commit and not to whatever is modified in the working
# tree. The copy lives under $TORCHVIC_DATA_ROOT/verification/ and is
# removed when the calling script exits.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${TORCHVIC_DATA_ROOT:?set TORCHVIC_DATA_ROOT}"
REV=${VERIFY_REV:-HEAD}
VERIFY_SHA=$(git -C $REPO rev-parse --short "$REV")
SNAP=$TORCHVIC_DATA_ROOT/verification/snapshot-$VERIFY_SHA-${SLURM_JOB_ID:-$$}
mkdir -p $SNAP
git -C $REPO archive "$REV" torchvic | tar -x -C $SNAP
export PYTHONPATH=$SNAP
export VERIFY_SHA
trap 'rm -rf "$SNAP"' EXIT
echo "TorchVIC code: commit $VERIFY_SHA ($SNAP)"
