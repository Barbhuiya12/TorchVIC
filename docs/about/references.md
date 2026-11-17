# References

Please cite the references that apply to the features you use. For every
TorchVIC run these are Liang et al. (1994) and Hamman et al. (2018) for VIC,
and TorchVIC itself ([How to cite](citing.md)).

## VIC

* Liang, X., Lettenmaier, D. P., Wood, E. F., and Burges, S. J.: A simple
  hydrologically based model of land surface water and energy fluxes for
  general circulation models, J. Geophys. Res., 99(D7), 14415-14428, 1994.
  <https://doi.org/10.1029/94JD00483>
* Hamman, J. J., Nijssen, B., Bohn, T. J., Gergel, D. R., and Mao, Y.: The
  Variable Infiltration Capacity model version 5 (VIC-5): infrastructure
  improvements for new applications and reproducibility, Geosci. Model Dev.,
  11, 3481-3496, 2018. <https://doi.org/10.5194/gmd-11-3481-2018>
* The Computational Hydrology Group, University of Washington: VIC source
  code and sample data, <https://github.com/UW-Hydro/VIC>.

## VIC physics

* Liang, X., Wood, E. F., and Lettenmaier, D. P.: Surface soil moisture
  parameterization of the VIC-2L model: Evaluation and modification, Global
  Planet. Change, 13, 195-206, 1996.
  <https://doi.org/10.1016/0921-8181(95)00046-1>
* Cherkauer, K. A. and Lettenmaier, D. P.: Hydrologic effects of frozen soils
  in the upper Mississippi River basin, J. Geophys. Res., 104(D16),
  19599-19610, 1999. <https://doi.org/10.1029/1999JD900337>
* Andreadis, K. M., Storck, P., and Lettenmaier, D. P.: Modeling snow
  accumulation and ablation processes in forested environments, Water Resour.
  Res., 45, W05429, 2009. <https://doi.org/10.1029/2008WR007042>

The VIC project lists further references for the individual features of VIC
on its [references page](https://vic.readthedocs.io/en/master/Documentation/References/).

## Forcing disaggregation and routing

* Thornton, P. E. and Running, S. W.: An improved algorithm for estimating
  incident daily solar radiation from measurements of temperature, humidity,
  and precipitation, Agric. Forest Meteorol., 93, 211-228, 1999.
  <https://doi.org/10.1016/S0168-1923(98)00126-9>
* Bohn, T. J., Livneh, B., Oyler, J. W., Running, S. W., Nijssen, B., and
  Lettenmaier, D. P.: Global evaluation of MTCLIM and related algorithms for
  forcing of ecological and hydrological models, Agric. Forest Meteorol.,
  176, 38-49, 2013. <https://doi.org/10.1016/j.agrformet.2013.03.003>
* MetSim, the Meteorology Simulator of the UW-Hydro group, whose algorithms
  the MTCLIM disaggregation re-implements (release 2.0.1),
  <https://github.com/UW-Hydro/MetSim>.
* Lohmann, D., Nolte-Holube, R., and Raschke, E.: A large-scale horizontal
  routing model to be coupled to land surface parametrization schemes,
  Tellus A, 48, 708-721, 1996.
* RVIC, the routing model of the UW-Hydro VIC project, whose conventions the
  routing follows, <https://github.com/UW-Hydro/RVIC>.

## Data

* Schaperow, J. R., Li, D., Margulis, S. A., and Lettenmaier, D. P.: A
  near-global, high resolution land surface parameter dataset for the variable
  infiltration capacity model, Sci. Data, 8, 216, 2021.
  <https://doi.org/10.1038/s41597-021-00999-4>
* Hersbach, H., et al.: The ERA5 global reanalysis, Q. J. R. Meteorol. Soc.,
  146, 1999-2049, 2020. <https://doi.org/10.1002/qj.3803>

## Coupled system

* Hanasaki, N., Kanae, S., and Oki, T.: A reservoir operation scheme for
  global river routing models, J. Hydrol., 327, 22-41, 2006.
  <https://doi.org/10.1016/j.jhydrol.2005.11.011>
* Droppers, B., Franssen, W. H. P., van Vliet, M. T. H., Nijssen, B., and
  Ludwig, F.: Simulating human impacts on global water resources using VIC-5,
  Geosci. Model Dev., 13, 5029-5052, 2020.
  <https://doi.org/10.5194/gmd-13-5029-2020>
* Dang, T. D., Chowdhury, A. F. M. K., and Galelli, S.: On the representation
  of water reservoir storage and operations in large-scale hydrological
  models: implications on model parameterization and climate change impact
  assessments, Hydrol. Earth Syst. Sci., 24, 397-416, 2020.
  <https://doi.org/10.5194/hess-24-397-2020>
* Haddeland, I., Skaugen, T., and Lettenmaier, D. P.: Anthropogenic impacts on
  continental surface water fluxes, Geophys. Res. Lett., 33, L08406, 2006.
  <https://doi.org/10.1029/2006gl026047>
* de Wit, A., Boogaard, H., Fumagalli, D., Janssen, S., Knapen, R., van
  Kraalingen, D., Supit, I., van der Wijngaart, R., and van Diepen, K.: 25
  years of the WOFOST cropping systems model, Agric. Syst., 168, 154-167,
  2019. <https://doi.org/10.1016/j.agsy.2018.06.018>
* van Diepen, C. A., Wolf, J., van Keulen, H., and Rappoldt, C.: WOFOST: a
  simulation model of crop production, Soil Use Manage., 5, 16-24, 1989.
* Stöckle, C. O., Donatelli, M., and Nelson, R.: CropSyst, a cropping systems
  simulation model, Eur. J. Agron., 18, 289-307, 2003.
  <https://doi.org/10.1016/S1161-0301(02)00109-0>
* Malek, K., Stöckle, C., Chinnayakanahalli, K., Nelson, R., Liu, M.,
  Rajagopalan, K., Barik, M., and Adam, J. C.: VIC-CropSyst-v2: a
  regional-scale modeling platform to simulate the nexus of climate,
  hydrology, cropping systems, and human decisions, Geosci. Model Dev., 10,
  3059-3084, 2017. <https://doi.org/10.5194/gmd-10-3059-2017>
* Edinger, J. E., Duttweiler, D. W., and Geyer, J. C.: The response of water
  temperatures to meteorological conditions, Water Resour. Res., 4, 1137-1143,
  1968.
* Hunt, B. R., Kostelich, E. J., and Szunyogh, I.: Efficient data assimilation
  for spatiotemporal chaos: a local ensemble transform Kalman filter, Physica
  D, 230, 112-126, 2007. <https://doi.org/10.1016/j.physd.2006.11.008>

The models and codes that the coupled system re-implements or is checked
against are credited on the [How to cite](citing.md#credits) page. Data sets
used by individual connectors (dams, irrigated area, aquifer properties) are
cited on the connector pages.

## Evaluation

* Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper Danube
  basin under an ensemble of climate change scenarios, J. Hydrol., 424-425,
  264-277, 2012. <https://doi.org/10.1016/j.jhydrol.2012.01.011>
* Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition of
  the mean squared error and NSE performance criteria: implications for
  improving hydrological modelling, J. Hydrol., 377, 80-91, 2009.
  <https://doi.org/10.1016/j.jhydrol.2009.08.003>
