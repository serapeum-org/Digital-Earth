[![Python Versions](https://img.shields.io/pypi/pyversions/digitalearth.png)](https://img.shields.io/pypi/pyversions/digitalearth)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/serapeum-org/Digital-Earth.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/serapeum-org/Digital-Earth/context:python)
[![Documentation Status](https://readthedocs.org/projects/digitalearth/badge/?version=latest)](https://cleopatra.readthedocs.io/en/latest/?badge=latest)


[![codecov](https://codecov.io/gh/serapeum-org/Digital-Earth/branch/main/graph/badge.svg?token=nDBDBjsyij)](https://codecov.io/gh/serapeum-org/Digital-Earth)
![GitHub last commit](https://img.shields.io/github/last-commit/serapeum-org/Digital-Earth)
![GitHub forks](https://img.shields.io/github/forks/serapeum-org/Digital-Earth?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/serapeum-org/Digital-Earth?style=social)


Current release info
====================

| Name | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Version | Platforms |
| --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-digitalearth-green.svg)](https://anaconda.org/conda-forge/digitalearth) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/digitalearth.svg)](https://anaconda.org/conda-forge/digitalearth) [![Downloads](https://pepy.tech/badge/digitalearth)](https://pepy.tech/project/digitalearth) [![Downloads](https://pepy.tech/badge/digitalearth/month)](https://pepy.tech/project/digitalearth)  [![Downloads](https://pepy.tech/badge/digitalearth/week)](https://pepy.tech/project/digitalearth)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/digitalearth?color=blue&style=flat-square) ![GitHub all releases](https://img.shields.io/github/downloads/serapeum-org/Digital-Earth/total) ![GitHub release (latest by date)](https://img.shields.io/github/downloads/serapeum-org/Digital-Earth/0.1.0/total) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/digitalearth.svg)](https://anaconda.org/conda-forge/digitalearth) [![PyPI version](https://badge.fury.io/py/digitalearth.svg)](https://badge.fury.io/py/digitalearth) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/digitalearth/badges/version.svg)](https://anaconda.org/conda-forge/digitalearth) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/digitalearth.svg)](https://anaconda.org/conda-forge/digitalearth) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

digitalearth - Remote Sensing package
=====================================================================
**digitalearth** is a Remote Sensing package

digitalearth

Main Features
-------------
  - plot static maps


Future work
-------------
  - dynamic/interactive maps



Installing digitalearth
===============

Installing `digitalearth` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge digitalearth
```

It is possible to list all of the versions of `digitalearth` available on your platform with:

```
conda search digitalearth --channel conda-forge
```

## Install from Github
to install the last development to time you can install the library from github
```
pip install git+https://github.com/serapeum-org/Digital-Earth
```

## pip
to install the last release you can easly use pip
```
pip install digitalearth
```

## Optional rendering tiers
The static matplotlib tier works out of the box. Richer renderers are optional extras (installed only when
you ask for them):

```
pip install 'digitalearth[interactive]'   # interactive 2-D web maps (HoloViz: GeoViews/Bokeh)
pip install 'digitalearth[3d]'            # true-3D scenes (PyVista)
pip install 'digitalearth[web]'           # MapLibre + deck.gl web maps, shareable HTML
```

The `web` tier renders pyramids rasters/vectors as MapLibre GL JS / deck.gl layers and exports a
self-contained HTML page — `WebMap().choropleth(gdf, column="pop").basemap().save("map.html")`, or
`quickplot(data, backend="web")`. See `docs/examples/web/` for a runnable gallery.

Quick start
===========

The entry points are `quickmap` (one call) and the `Map` scene. Build a finished map from a raster and save it:

```python
from pyramids.dataset import Dataset
from digitalearth import quickmap

src = Dataset.read_file("examples/data/acc4000.tif")
m = quickmap(src, crs=src.epsg)   # finished Map with a colorbar
m.set_title("Flow Accumulation")
m.save("flow_accumulation.png")
```
![Flowaccumulation](examples/images/flow_accumulation.png)

For more control, compose layers on a `Map` directly — e.g. overlay points on a raster:

```python
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from digitalearth import Map

src = Dataset.read_file("examples/data/acc4000.tif")
points = FeatureCollection.read_file("tests/data/points.geojson")

m = Map(crs=src.epsg)
m.imshow(src)
m.scatter(points)
m.colorbar(layer=0)
m.set_title("Flow Accumulation")
m.save("flow_accumulation_with_labels.png")
```
![Flowaccumulation](examples/images/flow_accumulation_with_labels.png)

[other code samples](https://digitalearth.readthedocs.io/en/latest/?badge=latest)

Legacy API (deprecated)
-----------------------

`StaticGlyph` is the original entry point and is **deprecated** — it emits a `DeprecationWarning` and will be
removed in a future release. Prefer `quickmap` / `Map` above. It still works for now:

```python
from pyramids.dataset import Dataset
from digitalearth.static import StaticGlyph

src = Dataset.read_file("examples/data/acc4000.tif")
fig, ax = StaticGlyph.plot(src, title="Flow Accumulation", cbar_label="Flow Accumulation")
```
