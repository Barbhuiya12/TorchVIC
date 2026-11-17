#!/bin/bash
# ISC input pipeline.  extract (bsdtar for the MIRCA rars, unzip GLHYMPS, Huang
# 7z already extracted with <data root>/build/7zip/7z) then the four generators
# on the cpu partition.  Python: conda env geospatial (rasterio, GDAL CLI).
PY=/apps/compilers/miniforge3/envs/geospatial/bin/python
D=$(dirname "$0")
for s in demand co2 landuse modflow; do
  sbatch ${TORCHVIC_SBATCH_ARGS:-} -c 8 --mem=64G -t 08:00:00 -J w52_$s --wrap "$PY -I $D/make_$s.py"
done
# second batch (needs pdftotext on the login node once: python parse_nrsd.py <pdf> <csv>)
for s in wofost dams aquifer; do
  sbatch ${TORCHVIC_SBATCH_ARGS:-} -c 8 --mem=96G -t 08:00:00 -J w52b_$s --wrap "$PY -I $D/make_$s.py"
done
