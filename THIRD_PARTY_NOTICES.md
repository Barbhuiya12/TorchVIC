# Third-party notices and credits

TorchVIC is Copyright (C) 2026 Siddik Barbhuiya, licensed GPL-3.0-or-later
(see `LICENSE`, `NOTICE`). Both upstream VIC licences are compatible with
GPL-3.0-or-later (MIT; GPL-2.0-or-later).

## 1. VIC model (algorithms re-implemented and logic ported)

### VIC 5.1.0 - MIT License
Source: https://github.com/UW-Hydro/VIC (tag 5.1.0), `LICENSE.txt`.

```
MIT License

Copyright (c) 2021 VIC Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### VIC 5.0.1 - GNU GPL v2 or later
Source file headers: "Copyright (C) 2016 The Computational Hydrology Group,
Department of Civil and Environmental Engineering, University of Washington.
The VIC model is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version." TorchVIC takes the "or any later version" option and distributes the
combined work under GPL-3.0-or-later.

### VIC 4.2.d - GNU GPL v2
The VIC 4.2.d output convention in `torchvic/io/history.py` (OUT_STEP aggregation,
SKIPYEAR, ALMA_OUTPUT, MOISTFRACT, OUTPUT_FORCE, and the related part of
`torchvic/case.py`) and the output-variable table `torchvic/data/vic42d_outvars.json`
follow VIC 4.2.d (`put_data.c`, `write_forcing_file.c`, `vicNl_def.h`); the code cites
the VIC 4.2.d file and line it follows. Credit: the VIC developers, Computational
Hydrology Group, Department of Civil and Environmental Engineering, University of
Washington (https://github.com/UW-Hydro/VIC, release VIC.4.2.d). VIC 4.2.d is
distributed under the GNU General Public License, version 2.

### Scientific references
- Liang, X., Lettenmaier, D. P., Wood, E. F., Burges, S. J. (1994), J. Geophys.
  Res. 99(D7), 14415-14428, doi:10.1029/94JD00483.
- Hamman, J. J., Nijssen, B., Bohn, T. J., Gergel, D. R., Mao, Y. (2018),
  Geosci. Model Dev. 11, 3481-3496, doi:10.5194/gmd-11-3481-2018.

## 2. Data and tools used by the TorchVIC data-preparation tools
Data are not part of the TorchVIC package and are not redistributed with it
(the small test fixtures derived from third-party software are listed in
section 7); each remains under its own terms. Licence fields below were read from the
cited source on 2026-10-14.

| Dataset | Credit | Licence (source) |
|---|---|---|
| HydroSHEDS v1 | Lehner, B., Verdin, K., Jarvis, A. (2008), Eos 89(10), 93-94, doi:10.1029/2008eo100001 | HydroSHEDS Version 1 License Agreement (World Wildlife Fund, Inc.), Appendix A of the HydroSHEDS Technical Documentation v1.4 (https://data.hydrosheds.org/file/technical-documentation/HydroSHEDS_TechDoc_v1_4.pdf); free for non-commercial and commercial use; required attribution in section 4 below. |
| MERIT-Hydro IHU (30 arcsec upscaled) | Eilander, D., Winsemius, H. C., van Verseveld, W., Yamazaki, D., Weerts, A., Ward, P. J. (2021), HESS 25, 5287-5313, doi:10.5194/hess-25-5287-2021; data doi:10.5281/zenodo.5166932 | Zenodo metadata: ODC-By 1.0. The LICENSE file in the same record says "Creative Commons CC-BY-NC 4.0 or Open Data Commons Open Database License (ODbL) v1.0" (dual); TorchVIC uses the ODbL option only (attribution; section 5 below). Underlying MERIT Hydro: Yamazaki et al. (2019), Water Resour. Res. 55, 5053-5073, doi:10.1029/2019WR024873. |
| GRIT river topology | Wortmann, M., Slater, L., Hawker, L., Liu, Y., Neal, J., Global River Topology (GRIT), doi:10.5281/zenodo.7629908 and vector datasets doi:10.5281/zenodo.11219313; Wortmann et al. (2025), Water Resour. Res. | CC BY-NC 4.0 (Zenodo records 7629908 and 11219313): non-commercial use of the data only. Not used for any TorchVIC data product since 2026-10-14 (isc_fdir.nc is built from HydroSHEDS and MERIT-Hydro only); only tools/build_isc_fdir_grit.py can read it. |
| ERA5 reanalysis (via Earth Data Hub / Copernicus) | Hersbach, H., et al. (2020), Q. J. R. Meteorol. Soc. 146, 1999-2049, doi:10.1002/qj.3803 | Licence to Use Copernicus Products (v1.2, https://apps.ecmwf.int/datasets/licences/copernicus/): free, worldwide, perpetual; attribution "Generated using Copernicus Climate Change Service information [year]" / "Contains modified Copernicus Climate Change Service information [year]"; and the statement that neither the European Commission nor ECMWF is responsible for any use of the information. |
| VICGlobal parameters | Schaperow, J., Li, D., Margulis, S., Lettenmaier, D. P. (2021), Sci. Data 8, 216, doi:10.1038/s41597-021-00999-4; data doi:10.5281/zenodo.3475602 (v1.6d: 10.5281/zenodo.5038653) | CC BY 4.0 (Zenodo record metadata) |

## 3. Software dependencies (not bundled; licences from installed package metadata)
- PyTorch (`torch>=2.6,<2.8`; 2.6.0 is the tested version): BSD-3-Clause (https://github.com/pytorch/pytorch)
- Triton 3.2 (installed with torch 2.6) or 3.3 (torch 2.7): MIT (optional; installed with the PyTorch CUDA packages and used by `torch.compile` on GPU; not a direct dependency in `pyproject.toml`)
- NumPy: BSD-3-Clause and others (see its metadata); pandas: BSD-3-Clause; SciPy (optional, the `flood` and `test` extras): BSD-3-Clause
- xarray: Apache-2.0; zarr (optional): MIT; netCDF4-python: MIT; cftime: MIT; PyYAML: MIT; tqdm (optional): MPL-2.0 and MIT
Further dependencies are listed in `pyproject.toml`; each keeps its own licence.

## 4. Required attribution: HydroSHEDS v1 (Exhibit B of the License Agreement, TechDoc v1.4)
Quoted verbatim; to be displayed with the documentation or metadata of any product
that incorporates the data (TorchVIC flow-direction map `isc_fdir.nc`; "[Product]" = TorchVIC ISC flow-direction map):

> This product [TorchVIC ISC flow-direction map] incorporates data from the HydroSHEDS version 1 database which is (c) World Wildlife Fund, Inc. (2006-2022) and has been used herein under license. WWF has not evaluated the data as altered and incorporated within [TorchVIC ISC flow-direction map], and therefore gives no warranty regarding its accuracy, completeness, currency or suitability for any particular purpose. Portions of the HydroSHEDS v1 database incorporate data which are the intellectual property rights of (c) USGS (2006-2008), NASA (2000-2005), ESRI (1992-1998), CIAT (2004-2006), UNEP-WCMC (1993), WWF (2004), Commonwealth of Australia (2007), and Her Royal Majesty and the British Crown and are used under license. The HydroSHEDS v1 database and more information are available at https://www.hydrosheds.org.

Scientific citation: Lehner, B., Verdin, K., Jarvis, A. (2008): New global hydrography derived from spaceborne elevation data. Eos, Transactions, AGU, 89(10): 93-94.

(The v1.3 text says "2006-2013" for the WWF copyright years; v1.4 says "2006-2022" and is used here.)
Licence conditions to respect (section 2 of the Agreement): the Licensed Materials may not be licensed or
distributed as a stand-alone product, and any redistribution to end users must be under terms at least as
protective as the Agreement; no use of WWF names or logos in publicity (section 8); no warranty (section 4).
Consequence: ship `isc_fdir.nc` only as part of the TorchVIC data bundle with this notice, not as a
stand-alone HydroSHEDS-derived dataset; if in doubt, ask WWF/HydroSHEDS (hydrosheds.org) before public release.

## 5. Required attribution: MERIT-Hydro IHU (ODbL 1.0 option)
The Zenodo record 5166932 has no prescribed wording; ODbL/ODC-By require attribution and, under ODbL,
share-alike of a publicly used derived database. Text to use:

> Contains information from MERIT-Hydro IHU (Eilander, D., Winsemius, H. C., van Verseveld, W., Yamazaki, D., Weerts, A., Ward, P. J., 2021, doi:10.5281/zenodo.5166932), upscaled from MERIT Hydro (Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., Pavelsky, T. M., 2019, Water Resour. Res. 55, 5053-5073, doi:10.1029/2019WR024873), which is made available under the Open Database License (ODbL) v1.0, https://opendatacommons.org/licenses/odbl/1-0/.

Cited papers: Eilander et al. (2021), Hydrol. Earth Syst. Sci. 25, 5287-5313, doi:10.5194/hess-25-5287-2021.
Note: the record metadata (ODC-By) and its LICENSE file (CC BY-NC 4.0 or ODbL) differ; the ODbL option is the
one compatible with an unrestricted data product.

## 6. Connectors
Format: name, version, copyright holder, licence, how used, citation. Licences were
read from the cited sources on 2026-10-14.

- RVIC (UW-Hydro, https://github.com/UW-Hydro/RVIC). Licence: GNU GPL v3.0 (README:
  "RVIC is available under the GNU GPL v3.0 License"; LICENSE.txt is the GPL v3 text;
  "only" or "or later" is not stated). Copyright holder: the RVIC developers (UW-Hydro).
  No version pinned. How used: the Lohmann unit-hydrograph routing and the RVIC
  convolution are re-implemented in PyTorch (`torchvic/model/routing.py`, following
  `rout_convolution.c` of VIC 5.0.1 and the RVIC method); the RVIC package is not
  imported or bundled. Cite: Lohmann, D., Nolte-Holube, R., Raschke, E. (1996), Tellus
  48A, 708-721; Lohmann, D., Raschke, E., Nijssen, B., Lettenmaier, D. P. (1998), Hydrol.
  Sci. J. 43(1), 131-141; and the RVIC release on Zenodo (DOI badge on the repository).
- VIC-WUR (Water Systems and Global Change group, Wageningen University,
  https://github.com/wur-wsg/VIC; release 3.0.0 on Zenodo). Licence: GNU GPL v2 (repository
  page: "GNU GPL v2.0"); every source header read in `reference/VIC-WUR` (224 C files
  carrying the notice) says "version 2 ... or (at your option) any later version", which
  TorchVIC takes as GPL-3.0-or-later. How used: the water-use, irrigation, reservoir and
  environmental-flow methods are ported/reimplemented in PyTorch (`torchvic/wateruse/`,
  `torchvic/reservoir/`, `torchvic/coupling/connectors/`); the C program is a validation
  reference only. Cite: Droppers, B., Franssen, W. H. P., van Vliet, M. T. H., Nijssen,
  B., Ludwig, F. (2020), Simulating human impacts on global water resources using VIC-5,
  Geosci. Model Dev. 13, 5029-5052, doi:10.5194/gmd-13-5029-2020 (read from the paper).
- VICRes (Critical Infrastructure Systems Lab, https://github.com/Critical-Infrastructure-Systems-Lab/VICRes).
  Licence: none. The repository has no LICENSE file and states no licence, so no permission
  to copy or redistribute exists. Cite-only: no VICRes code or files are copied or
  bundled; its reservoir-operation methods are reimplemented from the papers. Cite:
  Dang, T. D., Chowdhury, A. F. M. K., Galelli, S. (2020), Hydrol. Earth Syst. Sci. 24,
  397-416; Dang, T. D., Vu, T. D., Chowdhury, K., Galelli, S. (2020), A software package
  for the representation and optimization of water reservoir operations in the VIC
  hydrologic model, Environ. Model. Softw. 126, 104673; Eldardiry, H., Mahto, S. S.,
  Fatichi, S., Galelli, S. (2025), VIC-Res Mekong, Environ. Model. Softw. (the three
  papers the VICRes README asks users to cite).
- MODFLOW 6 (USGS / MODFLOW-ORG, https://github.com/MODFLOW-ORG/modflow6). Licence:
  public domain in the United States, CC0 1.0 Universal (`LICENSE.md`: "Unless otherwise
  noted, This project is in the public domain in the United States"). Reference only:
  the groundwater connector is a PyTorch implementation of the same equations; MODFLOW 6
  is run only to validate it and is not bundled or linked. Cite: Langevin, C. D., et al.
  (2017), Documentation for the MODFLOW 6 Groundwater Flow Model, USGS Techniques and
  Methods 6-A55 (reference recalled, not re-read for this notice).
- LISFLOOD-FP v8.2 (LISFLOOD-FP developers; contact J. Neal, University of Bristol; P.
  Bates, G. Kesserwani, M. K. Sharifian; Zenodo doi:10.5281/zenodo.13121102, 2024-07-29).
  Licence: GNU GPL v2.0 only (Zenodo record). External reference backend only: it runs as a
  separate process (files in, files out), is not linked, imported or bundled, because
  GPL-2.0-only cannot be combined with GPL-3.0-or-later. TorchVIC's own flood engine
  (`torchvic/flood/`) is the default backend; LISFLOOD-FP serves as the comparison
  reference and optional backend. Cite the Zenodo record above.
- WOFOST 7.2 via PCSE 6.0.13 (crop model): Copyright (c) 2004-2024 Wageningen
  Environmental Research, Wageningen-UR (A. de Wit, H. Berghuijs). Licence EUPL-1.2
  (GPL-3.0 listed as compatible in its appendix). How used: ported to PyTorch in
  `torchvic/crop/` (full notice in `torchvic/crop/NOTICE_PCSE.txt`); PCSE itself used
  only as the validation reference. Crop parameters: WOFOST_crop_parameters, branch
  wofost72 (Wageningen). Cite: de Wit et al. (2019), Agricultural Systems 168, 154-167,
  doi:10.1016/j.agsy.2018.06.018.

- CropSyst, with the VIC-CropSyst coupling (C. O. Stockle, R. L. Nelson, Biological
  Systems Engineering, Washington State University; VIC-CropSyst by K. Malek, C. Stockle,
  K. Chinnayakanahalli, R. Nelson, M. Liu, K. Rajagopalan, J. C. Adam and co-workers).
  Licence: Academic Free License, as stated by the CropSyst project (the version is that
  of the distributed package; it is not public-domain software). The authors ask to be
  informed of derivative works. How used: the crop growth, phenology, water and nitrogen
  uptake, irrigation and orchard-architecture methods are re-implemented in PyTorch from
  the published equations and the reference source (`torchvic/crop/cropsyst/`,
  `torchvic/coupling/connectors/crop.py`, and the CropSyst hooks in `torchvic/core/`);
  source text is not copied. The compiled VIC-CropSyst package is used only as a
  validation reference and is not bundled. Cite: Stockle, C. O., Donatelli, M., Nelson,
  R. (2003), CropSyst, a cropping systems simulation model, Eur. J. Agron. 18, 289-307;
  Malek, K., et al. (2017), VIC-CropSyst-v2: a regional-scale modeling platform to
  simulate the nexus of climate, hydrology, cropping systems, and human decisions,
  Geosci. Model Dev. 10, 3059-3084, doi:10.5194/gmd-10-3059-2017.

### Data used by the connectors
Not redistributed with TorchVIC; each keeps its own terms (see also section 2).

| Dataset | Credit | Licence (source) |
|---|---|---|
| GDW v1 (Global Dam Watch consensus database) and GRanD v1.3 | Lehner, B., et al. (2024), Global Dam Watch database, Sci. Data, doi:10.1038/s41597-024-03752-9 (the DOI given on globaldamwatch.org/database); Lehner, B., Reidy Liermann, C., Revenga, C., et al. (2011), Front. Ecol. Environ. 9(9), 494-502 | GDW: "distributed under a free CC BY license" (globaldamwatch.org/database; CC version not stated there, check the download form before release). GRanD: the page globaldamwatch.org/grand states no licence (only a no-warranty disclaimer) and says GRanD is superseded by GDW; treat as terms unknown until confirmed with Global Dam Watch. |
| GMIA v5 (Global Map of Irrigation Areas) | Siebert, S., Henrich, V., Frenken, K., Burke, J. (2013), Global Map of Irrigation Areas version 5, Rheinische Friedrich-Wilhelms-University, Bonn / FAO, Rome | The FAO page (fao.org/aquastat, latest-version) states no licence and only links FAO's general Terms and Conditions; the FAO terms page names only the Open Data Licensing Policy for statistical databases. Terms not confirmed: do not redistribute; use of the data needs the FAO terms checked. |
| Copernicus DEM GLO-30 | European Space Agency; doi:10.5270/ESA-c5d3d65 | Copernicus DEM licence, free worldwide (Copernicus Data Space page). Required notice when distributing: "(c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved"; for modified data begin with "produced using Copernicus WorldDEM-30". |

### MTCLIM daily-to-sub-daily disaggregation (MetSim)
`torchvic/forcing/mtclim.py` and `torchvic/forcing/daily.py` re-implement the MTCLIM
algorithm and disaggregation of MetSim, the Meteorology Simulator (UW-Hydro,
https://github.com/UW-Hydro/MetSim), release v2.0.1, commit
`742ef6897f7c3bb35649568f63cb30451e9ba4b9`, as batched PyTorch tensor code (reading its
`metsim/physics.py`, `metsim/methods/mtclim.py`, `metsim/disaggregate.py` and
`metsim/constants.py`). Copyright (C) 2017 The Computational Hydrology Group, Department of
Civil and Environmental Engineering, University of Washington (principal author of the
package: Andrew Bennett). Licence: GNU GPL v3 or later, compatible with TorchVIC's
GPL-3.0-or-later. MetSim is not redistributed or executed by TorchVIC; the tests compare
against a numpy re-reading of its algorithms.
Credits for the method: Thornton, P. E., Running, S. W. (1999), Agric. For. Meteorol. 93,
211-228 (transmittance from the daily temperature range); Kimball, J. S., Running, S. W.,
Nemani, R. (1997), Agric. For. Meteorol. 85, 87-98 (humidity from potential
evapotranspiration); Bohn, T. J., Livneh, B., Oyler, J. W., Running, S. W., Nijssen, B.,
Lettenmaier, D. P. (2013), Agric. For. Meteorol. 176, 38-49, doi:10.1016/j.agrformet.2013.03.003
(the disaggregation and the longwave and cloud options MetSim implements); the original
MT-CLIM of Running, S. W., Nemani, R. R., Hungerford, R. D. (1987), Can. J. For. Res. 17,
472-483, and Glassy, J. M., Running, S. W. (1994), Ecol. Appl. 4, 248-257. The direct
fraction used by `carbon: true` (MetSim has none) is the daily correlation of Erbs, D. G.,
Klein, S. A., Duffie, J. A. (1982), Solar Energy 28, 293-302. The PCHIP spline is that of
Fritsch, F. N., Carlson, R. E. (1980), SIAM J. Numer. Anal. 17, 238-246.
Earlier versions of this module (up to 0.4.6) transcribed the MTCLIM code in VIC 4.2.d
(`mtclim_vic.c`, `mtclim_wrapper.c`, `initialize_atmos.c`, `calc_air_temperature.c`,
`calc_longwave.c`), which is "GNU GPL v2.0" with no "or later" grant and so cannot be
combined with GPL-3.0-or-later. That code has been removed and replaced; the numerical
differences from VIC 4.2.d are listed in `CHANGELOG.md`. The VIC 4.2.d oracle in
`tests/oracle/ref_mtclim.c` links VIC 4.2.d only to measure that distance and is not part of
the distributed package. The VIC 4.2.d reference tree and its compiled oracle are test
inputs, not shipped. (Citations from the papers' reference lists as quoted in MetSim's
README; the papers themselves were not re-read for this notice.)

## 7. Fixtures, oracle drivers and patches kept in the repository
The repository carries a few small files that are derived from third-party software or data. They are test and
development material. The installed package (the wheel) contains `torchvic/`, its three `torchvic/data/*.json`
tables, `torchvic/crop/NOTICE_PCSE.txt` and `torchvic/crop/cropsyst/sdr_field_ops.txt`, and none of the others.

- `tests/data/crop_pcse/*.npz`: the daily weather, the crop parameters and PCSE's daily output for the PCSE
  demonstration cases, written by `torchvic/crop/pcse_reference.py` with PCSE 6.0.13 (WOFOST 7.2). PCSE, its
  demonstration weather and database and the WOFOST crop parameter sets are EUPL-1.2 (see
  `torchvic/crop/NOTICE_PCSE.txt`).
- `tests/data/metsim_2_0_1_fixture.npz`: output of MetSim 2.0.1 (UW-Hydro, GNU GPL v3 or later; see the MetSim
  section above) for two seeded synthetic daily series made by `tests/oracle/ref_mtclim.c`; no observations.
- `tests/data/cropsyst_growth_l1.txt.gz`, `cropsyst_irrigation_l1.txt.gz`, `cropsyst_phenology_l1.txt.gz`,
  `cropsyst_uptake_l1.txt`: numbers returned by the compiled VIC-CropSyst reference build (CropSyst, Academic
  Free License, see section 6) for generated inputs. Numbers only; no CropSyst source text.
- `torchvic/crop/cropsyst/sdr_field_ops.txt`: the NRCS/RUSLE2 (USDA) field-operation table that the VIC-CropSyst
  build queries (operation names, soil mixing and burial fractions, disturbance ratings), extracted by running
  that build (`tests/oracle/ref_cropsyst_nitrogen.cpp`). Factual data, read at run time by the nitrogen code.
- `torchvic/data/vic5_global_options.json` and `vic5_outvars.json`: inventories of the global-file options and the
  output variables of VIC 5.0.1 and 5.1.0 (names, defaults and the C lines that handle them), extracted from the
  VIC sources by `tools/extract_vic_options.py` and `tools/extract_vic_outvars.py`. VIC 5.1.0 is MIT, VIC 5.0.1 is
  GPL-2.0-or-later (section 1); `vic42d_outvars.json` is covered there too.
- `tests/oracle/`: small C and C++ drivers written for TorchVIC. At test time they are compiled against source
  trees that the user supplies (VIC 5.0.1, 5.1.0 and 4.2.d, VIC-WUR 3.0.0, the CropSyst package) and call their
  routines; none of those sources is in the repository. Where a driver reproduces a few lines of upstream code it
  says so and cites the file and lines; the headers under `*_shim/` declare only the names the upstream sources use.
- `tools/vicwur/*.patch`: instrumentation patches for VIC-WUR 3.0.0 (GPL-2.0-or-later) that the user applies to a
  copy of it; they add dump code and change no model equation.
- `tools/data/*.csv`: gauge-review tables written for TorchVIC (CWC gauge identifiers and names, the grid
  corrections chosen and the evidence for each). No gauge records, catchment polygons or river networks are included.
