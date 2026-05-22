# Installation

**Package name:** `digitalearth`
**Supported Python versions:** 3.11, 3.12, 3.13 (requires `>=3.11,<4`)

Please install `digitalearth` in a virtual environment so its requirements
don't tamper with your system Python.

## With conda (recommended)

`digitalearth` and its native dependency (GDAL) are easiest to install via
the [conda-forge](https://conda-forge.org/) channel:

```console
conda install -c conda-forge digitalearth
```

This installs `digitalearth` together with all dependencies, including Python
and GDAL.

## With pip (PyPI)

```console
pip install digitalearth
```

GDAL is not pulled in automatically by pip (no Windows wheel exists on PyPI).
Install native GDAL via conda or your system package manager first, then:

```console
pip install digitalearth
```

## With pixi (development)

This repository is configured with [pixi](https://pixi.sh). To set up a
development environment that manages GDAL via conda-forge:

```console
git clone https://github.com/serapeum-org/Digital-Earth.git
cd Digital-Earth
pixi install -e dev
pixi run -e dev main      # runs the test suite
```

Pixi environments available:

| Environment | Purpose |
|-------------|---------|
| `dev` | Default development env (test tooling + geoplot extra) |
| `docs` | Documentation toolchain (mkdocs + plugins) |
| `notebook` | Jupyter notebook environment |
| `py311`, `py312`, `py313` | Single-Python-version test envs |

## Install directly from GitHub

Latest `main`:

```console
pip install "git+https://github.com/serapeum-org/Digital-Earth.git"
```

A specific tagged release:

```console
pip install "git+https://github.com/serapeum-org/Digital-Earth.git@<version>"
```

## Verify the install

```python
import digitalearth
print(digitalearth.__version__)
```
