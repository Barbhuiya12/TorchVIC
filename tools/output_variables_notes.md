**Where each row comes from.** `torchvic/model/outvars.py` forms the
variables that are functions of rows the step loop already returns (put_data.c
"derived variables", the forcing echo, the stores VIC zeroes where an option
is off); `basin.simulate` collects the rest per tile (aerodynamic
conductances of the two surfaces, the thermal-node temperatures, the frost
cover, the `OUT_*_BAND` rows of the understory) in lists registered like the
older rows, outside the fused regions, so no row that existed is touched.
Each is weighted by `Cv * AreaFract * lakefactor` as put_data.c does; the band
rows are placed on the band axis (`snowbands.to_bands`), not area-weighted.

**Option-dependent values.**

* `OUT_ENERGY_ERROR`: VIC forms it from the cell's reported rows
  (`calc_energy_balance_error`, put_data.c:538-552) and reports up to
  10-190 W/m2 where its own snow terms do not close. TorchVIC's default is
  the surface balance's residual at the solved temperature (about 1e-6), a
  deliberate difference that predates this work; `--energy-error vic`
  (`basin.simulate(energy_error='vic')`) forms VIC's number, to 2e-13 of it.
  With FULL_ENERGY FALSE VIC writes MISSING (-99999); so does TorchVIC.
* Variables that are zero or MISSING by construction in a configuration
  (BLOWING FALSE, LAKES FALSE, SPATIAL_SNOW FALSE, CARBON FALSE, no RVIC
  extension) are written as VIC writes them.
* `OUT_*_FBFLAG`: written as zero when the run took no temperature-solve
  fallback (TFALLBACK TRUE counts them, `torchvic.core.tfallback`), left out
  of a run that did. A run on the CUDA-graph path records no fallbacks, so its
  flags are written as zero.

**Water-balance mode.** FULL_ENERGY FALSE still reports the energy rows, because
VIC evaluates the surface balance once at Tsurf = Tair
(calc_surf_energy_bal.c:542-545). `torchvic/model/wbenergy.py` forms them with
the energy-balance path's formulas at that temperature (OUT_SWNET, LWNET,
R_NET, LATENT, SENSIBLE, SURF_TEMP, RAD_TEMP, VEGT, SNOW_FLUX, PET,
SOIL_TEMP, SOIL_TNODE, AERO_COND/RESIST and their band rows).

**Known gaps.**

* `OUT_BARESOILT` is not written where the mosaic leaves a bare tile
  (cover below 1) that TorchVIC does not simulate (as before).
* BLOWING TRUE: where TorchVIC's own blowing flux is zero and VIC's is
  5e-7 mm (Stehekin), OUT_SUB_BLOWING follows (a difference of the existing
  CalcBlowingSnow port, not of these rows).

* `OUT_TIME_VICRUN_WALL`, `OUT_TIME_VICRUN_CPU`: timers of the VIC process.
* CARBON TRUE: `OUT_GPP`, `OUT_NPP`, `OUT_RAUT`, ... are mapped from the
  carbon rows with put_data.c's unit conversions, but pristine VIC 5.1.0's
  image driver writes zero for all of them on the Walnut Gulch carbon rows
  while TorchVIC's carbon cycle (which matches VIC's hydrology under
  RC_PHOTO) is nonzero. The discrepancy is open; the mapped values are not
  verified against VIC.
* LAKES TRUE: `OUT_SOIL_TNODE_WL` and the lake/wetland terms of the other
  rows are not formed (zero is written for the wetland nodes).
