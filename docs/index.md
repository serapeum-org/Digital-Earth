[![Documentation](https://img.shields.io/badge/Documentation-blue?logo=github&logoColor=white)](https://serapeum-org.github.io/Digital-Earth/)
[![Python Versions](https://img.shields.io/pypi/pyversions/digitalearth.png)](https://img.shields.io/pypi/pyversions/digitalearth)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyPI version](https://badge.fury.io/py/digitalearth.svg)](https://badge.fury.io/py/digitalearth)

# Digital-Earth

**digitalearth** is a Python package providing different plots for rasters and
vector data, built on top of [cleopatra](https://github.com/serapeum-org/cleopatra),
[pyramids](https://github.com/serapeum-org/pyramids), matplotlib, and geopandas.

## Main Features

- Static raster plotting via `quickmap` / `Map`, with rich, customizable
  styling (color scales, color bars, cell-value annotations).
- Overlay vector points on raster maps with configurable size, color, and labels.

## Installation

- pip (PyPI):

```bash
pip install digitalearth
```

- conda (conda-forge):

```bash
conda install -c conda-forge digitalearth
```

- pixi (development):

```bash
pixi add digitalearth
```

See [Installation](installation.md) for the full guide.

## Quick start

```python
from pyramids.dataset import Dataset
from digitalearth import quickmap

dataset = Dataset.read_file("examples/data/acc4000.tif")
m = quickmap(dataset, crs=dataset.epsg)
m.set_title("Flow Accumulation")
```

## Next steps

- Read the [Plot raster/array](plotarray.md) guide for the full plotting API.
- Work through the [Maps & raster fields](examples/02_maps_and_fields.ipynb) example.
