#!/bin/bash
# Build every oracle (see each build_ref_*.sh).
set -e
D="$(dirname "${BASH_SOURCE[0]}")"
for s in soil snow mtclim route vicwur_route vicwur_nonrenew vicwur_co2 vicwur_dam vicwur_wofost vicwur_lu vicwur_wu_remote vicwur_core; do
    bash "$D/build_ref_$s.sh"
done
