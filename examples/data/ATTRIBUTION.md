# Example data sources

Third-party data bundled here for the documentation notebooks. Every file below was downloaded with
[earthlens](https://github.com/serapeum-org/earthlens), which names the source, the licence and the
citation each provider requires.

## `germany_adm1.geojson` — Germany's 16 federal states

`earthlens` backend `admin`, dataset `geoboundaries:adm1`, country `DEU`. Simplified to a 0.005-degree
tolerance to keep the file small; shape fidelity is far finer than any national-scale figure needs.

- Source: **geoBoundaries (gbOpen)**, <https://www.geoboundaries.org/>
- Licence: **CC-BY-4.0**
- Citation: Runfola, D. et al. (2020) *geoBoundaries: A global database of political administrative
  boundaries.* PLoS ONE 15(4): e0231866.

## `germany_pop_2010_1km.tif`, `germany_pop_2020_1km.tif` — population grids

`earthlens` backend `worldpop`, product `pop`, `aoi="DEU"`, 1 km resolution, unconstrained and
UN-unadjusted. People per grid cell.

- Source: **WorldPop**, <https://www.worldpop.org/>
- Licence: **CC-BY-4.0**
- Citation: WorldPop (www.worldpop.org), School of Geography and Environmental Science, University of
  Southampton. DOI 10.5258/SOTON/WP00645.

## `heidelberg_roads.geojson` — road centrelines

`earthlens` backend `osm`, named query `live:roads`, bbox 49.39-49.43 N, 8.65-8.72 E. Filtered to the
road hierarchy (`motorway` through `living_street`, dropping footways, paths and steps) and to the
columns the notebooks teach from; `lanes` and `maxspeed` coerced to numbers.

- Source: **OpenStreetMap contributors**, via the Overpass API
- Licence: **ODbL 1.0** — <https://www.openstreetmap.org/copyright>. Every figure drawn from these
  roads carries a “© OpenStreetMap contributors” credit, and any redistributed map must keep it.

## `heidelberg_birds_2023.geojson` — bird occurrence records

`earthlens` backend `gbif`, taxon `birds`, 2023, bbox 49.30-49.52 N, 8.55-8.85 E, capped at 3,000
records. Each record is one dated observation; `uncertainty_m` is the recorded coordinate uncertainty.

- Source: **GBIF** (Global Biodiversity Information Facility), <https://www.gbif.org/>
- Licence: per-record upstream. This extract was filtered to openly-licensed records
  (CC0 / CC-BY / CC-BY-NC) at download time; the per-record `license` field is not
  carried in the committed file, so re-download from GBIF if you need it.
- Citation: GBIF.org (2023) *GBIF Occurrence Download*. Cite GBIF and the contributing datasets
  when redistributing.

## `global/ncep_tavg_01..12.tif` — monthly mean air temperature

NCEP/NCAR Reanalysis 1 surface-air-temperature long-term monthly means (1991-2020 climatology, read from the file's own
climatology_bounds), read
from NOAA PSL as a single 0.6 MB NetCDF and written as twelve GeoTIFFs (0.48 MB in total) so notebooks
11, 12 and 14 read them from the checkout instead of fetching anything. Native resolution is 2.5
degrees (73 x 144); the committed grids are aligned onto a clean -180..180 / -90..90 global grid
(72 x 144), which both settles the 0-360 longitude frame the source uses and clips the source's
out-of-range +/-91.25 latitude rows. Values are degrees Celsius, nodata -9999, and coverage is global
— ocean as well as land.

- Source: **NOAA PSL, NCEP/NCAR Reanalysis 1**,
  <https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.derived.surface.html>
- Licence: US Government work, public domain.
- Citation: Kalnay, E. et al. (1996) *The NCEP/NCAR 40-Year Reanalysis Project.* Bulletin of the
  American Meteorological Society 77(3): 437-471.

## `global/etopo_relief.tif` — global relief

ETOPO180 altitude, read from NOAA's ERDDAP griddap service as an ESRI-ASCII grid at stride 10, then
resampled to 0.5 degrees and written as GeoTIFF (12.2 MB of ASCII becomes 0.58 MB). Covers both ocean
bathymetry and land elevation, so notebook 11 drapes it whole and also derives a land-only
elevation grid from it by masking everything at or below sea level.

- Source: **NOAA NGDC ETOPO1 / ETOPO180**, via <https://coastwatch.pfeg.noaa.gov/erddap/>
- Licence: US Government work, public domain.
- Citation: Amante, C. and Eakins, B.W. (2009) *ETOPO1 1 Arc-Minute Global Relief Model.* NOAA
  Technical Memorandum NESDIS NGDC-24.

## Pre-existing fixtures

`acc4000.tif`, `LisbonElevation.tif`, `DEM5km_Rhine_burned_acc.tif`, `points.csv`, `rhine_*.geojson`,
`MetricsHM_Q_Obs.geojson` and `global/ersstv5.nc` predate this file and are used by the other notebooks.
