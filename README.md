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
pip install digitalearth==0.1.11
```

Quick start
===========

```
from pyramids.dataset import Dataset
from digitalearth.static import StaticGlyph
src = Dataset.read_file("examples/data/acc4000.tif")
fig, ax = StaticGlyph.plot(src, title="Flow Accumulation", cbar_label="Flow Accumulation")
```
![Flowaccumulation](examples/images/flow_accumulation.png)
```
points = pd.read_csv("examples/data/points.csv")
point_color = "blue"
point_size = 100
id_color = "yellow"
id_size = 20

display_cell_value = True
num_size = 8
background_color_threshold = None
ticks_spacing = 500

fig, ax = StaticGlyph.plot(
            src,
            point_color=point_color,
            point_size=point_size,
            pid_color=id_color,
            pid_size=id_size,
            points=points,
            display_cell_value=display_cell_value,
            num_size=num_size,
            background_color_threshold=background_color_threshold,
            ticks_spacing=ticks_spacing,
            title="Flow Accumulation",
            cbar_label="Flow Accumulation"
        )
```
![Flowaccumulation](examples/images/flow_accumulation_with_labels.png)
[other code samples](https://digitalearth.readthedocs.io/en/latest/?badge=latest)
