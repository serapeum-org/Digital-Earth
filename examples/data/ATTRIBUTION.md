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
- Licence: **ODbL 1.0** — <https://www.openstreetmap.org/copyright>. Any derived map must credit
  "© OpenStreetMap contributors".

## `heidelberg_birds_2023.geojson` — bird occurrence records

`earthlens` backend `gbif`, taxon `birds`, 2023, bbox 49.30-49.52 N, 8.55-8.85 E, capped at 3,000
records. Each record is one dated observation; `uncertainty_m` is the recorded coordinate uncertainty.

- Source: **GBIF** (Global Biodiversity Information Facility), <https://www.gbif.org/>
- Licence: per-record; this extract contains only openly-licensed records (CC0 / CC-BY).
- Citation: GBIF.org (2023) *GBIF Occurrence Download*. Honour the per-record `license` field when
  redistributing.

## Pre-existing fixtures

`acc4000.tif`, `LisbonElevation.tif`, `DEM5km_Rhine_burned_acc.tif`, `points.csv`, `rhine_*.geojson`,
`MetricsHM_Q_Obs.geojson` and `global/ersstv5.nc` predate this file and are used by the other notebooks.
