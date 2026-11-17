# How to cite

If you use TorchVIC in published work, please cite the software, the VIC model
it implements, and the codes and data listed under [Credits](#credits) that
apply to the features you used.

## TorchVIC

Cite the software as below. `CITATION.cff` at the repository root holds the
same title, authors and license, and GitHub reads it for its *Cite this
repository* button:

> Barbhuiya, S. and Gupta, V.: TorchVIC: the VIC 5 land surface model in PyTorch, version
> 0.4.8, GPL-3.0-or-later, <https://github.com/Barbhuiya12/TorchVIC>.

```bibtex
@software{barbhuiya_torchvic,
  author  = {Barbhuiya, Siddik and Gupta, Vivek},
  title   = {{TorchVIC}: the {VIC} 5 land surface model in {PyTorch}},
  version = {0.4.8},
  license = {GPL-3.0-or-later},
  url     = {https://github.com/Barbhuiya12/TorchVIC}
}
```

No DOI has been issued for the software.

A paper that describes the model and its evaluation is in preparation:

> Barbhuiya, S. and Gupta, V.: TorchVIC 0.4.7: the Variable Infiltration Capacity model
> version 5 on graphics processing units, manuscript in preparation.

Until it appears, cite the software as above. This page will give the paper's
reference when it is available.

## VIC

TorchVIC re-implements the physics of VIC. Credit for the model belongs to its
creators and developers. Please acknowledge both references below, as the VIC
project asks, together with the references of the VIC features you use (they
are cited in [Model overview](overview.md) and on the
[references](references.md) page):

* Liang, X., Lettenmaier, D. P., Wood, E. F., and Burges, S. J.: A simple
  hydrologically based model of land surface water and energy fluxes for
  general circulation models, J. Geophys. Res., 99(D7), 14415-14428, 1994.
  <https://doi.org/10.1029/94JD00483>
* Hamman, J. J., Nijssen, B., Bohn, T. J., Gergel, D. R., and Mao, Y.: The
  Variable Infiltration Capacity model version 5 (VIC-5): infrastructure
  improvements for new applications and reproducibility, Geosci. Model Dev.,
  11, 3481-3496, 2018. <https://doi.org/10.5194/gmd-11-3481-2018>

VIC is developed by the Computational Hydrology Group, University of
Washington: <https://github.com/UW-Hydro/VIC>.

## Credits

TorchVIC is free software under the GNU General Public License, version 3 or
(at your option) any later version. It re-implements, or is checked against,
the codes below. The original sources of these codes are not distributed with
TorchVIC. The license texts and the full notices are in
[`NOTICE`](https://github.com/Barbhuiya12/TorchVIC/blob/main/NOTICE) and
[`THIRD_PARTY_NOTICES.md`](https://github.com/Barbhuiya12/TorchVIC/blob/main/THIRD_PARTY_NOTICES.md).

### Models and codes

| code | license | what TorchVIC takes from it | cite |
|---|---|---|---|
| [VIC](https://github.com/UW-Hydro/VIC) 5.1.0 and VIC 5.0.1 | MIT; GPL-2.0-or-later | the physics, re-implemented and ported; every process function cites the C file and lines it follows | Liang et al. (1994); Hamman et al. (2018) |
| VIC 4.2.d | GPL-2.0 | the output convention of the history writer (`OUT_STEP` aggregation, `SKIPYEAR`, `ALMA_OUTPUT`, `MOISTFRACT`, `OUTPUT_FORCE`) | VIC developers |
| [MetSim](https://github.com/UW-Hydro/MetSim) 2.0.1 | GPL-3.0-or-later | the MTCLIM daily-to-sub-daily disaggregation, re-implemented as tensor code | Thornton and Running (1999); Bohn et al. (2013); Kimball et al. (1997); Running et al. (1987); Glassy and Running (1994); Erbs et al. (1982); Fritsch and Carlson (1980) |
| [RVIC](https://github.com/UW-Hydro/RVIC) | GPL-3.0 | the Lohmann unit-hydrograph routing and the RVIC convolution, re-implemented; RVIC itself is neither imported nor bundled | Lohmann et al. (1996, 1998) |
| [VIC-WUR](https://github.com/wur-wsg/VIC) 3.0.0 | GPL-2.0 (source headers: or any later version) | the water-use, irrigation, reservoir and environmental-flow methods, re-implemented in PyTorch | Droppers et al. (2020) |
| [VICRes](https://github.com/Critical-Infrastructure-Systems-Lab/VICRes) | none stated | reservoir-operation methods re-implemented from the papers; no VICRes code or file is used | Dang et al. (2020, HESS 24, 397-416); Dang et al. (2020, Environ. Model. Softw. 126, 104673); Eldardiry et al. (2025, Environ. Model. Softw.) |
| [MODFLOW 6](https://github.com/MODFLOW-ORG/modflow6) | public domain in the United States (CC0 1.0) | the groundwater connector implements the same equations in PyTorch; MODFLOW 6 is used only to validate it | Langevin et al. (2017), USGS Techniques and Methods 6-A55 |
| LISFLOOD-FP 8.2 | GPL-2.0 only | an optional external flood backend and the reference of the flood engine, run as a separate process and never linked | Zenodo, [doi:10.5281/zenodo.13121102](https://doi.org/10.5281/zenodo.13121102) |
| WOFOST 7.2 (PCSE 6.0.13) | EUPL-1.2 | the crop model, ported to PyTorch; the license notice is in `torchvic/crop/NOTICE_PCSE.txt` | de Wit et al. (2019); van Diepen et al. (1989) |
| CropSyst and VIC-CropSyst | Academic Free License | crop growth, phenology, water and nitrogen uptake, irrigation and orchard methods, re-implemented from the published equations | Stöckle et al. (2003); Malek et al. (2017) |

### Data used by the tools

The data below are not part of TorchVIC and are not redistributed with it;
each keeps its own terms. The tools in `tools/` read them to build inputs for
the Indian-subcontinent example, and the connectors name them as sources.

| data | cite |
|---|---|
| VICGlobal parameters | Schaperow et al. (2021), Sci. Data 8, 216 |
| ERA5 reanalysis | Hersbach et al. (2020), Q. J. R. Meteorol. Soc. 146, 1999-2049. Generated using Copernicus Climate Change Service information |
| HydroSHEDS v1 | Lehner et al. (2008), Eos 89(10), 93-94 |
| MERIT-Hydro IHU | Eilander et al. (2021), Hydrol. Earth Syst. Sci. 25, 5287-5313; Yamazaki et al. (2019), Water Resour. Res. 55, 5053-5073 |
| Global Dam Watch (GDW) and GRanD | Lehner et al. (2024), Sci. Data; Lehner et al. (2011), Front. Ecol. Environ. 9(9), 494-502 |
| Global Map of Irrigation Areas v5 | Siebert et al. (2013) |
| Copernicus DEM GLO-30 | European Space Agency, doi:10.5270/ESA-c5d3d65 |
| GRIT river topology (CC BY-NC 4.0; read only by `tools/build_isc_fdir_grit.py`) | Wortmann et al. (2025), Water Resour. Res.; doi:10.5281/zenodo.7629908 |

The attribution that HydroSHEDS and MERIT-Hydro require for products that
incorporate their data is in `THIRD_PARTY_NOTICES.md`.

### Software

TorchVIC is built on PyTorch (BSD-3-Clause), with Triton (MIT) used by
`torch.compile` on GPUs, NumPy, pandas and SciPy (BSD-3-Clause; SciPy is used by
the DEM conditioning of the flood connector), xarray
(Apache-2.0), and zarr, netCDF4-python and cftime (MIT). The example data of the
tutorials are UW-Hydro's
[VIC_sample_data](https://github.com/UW-Hydro/VIC_sample_data).
