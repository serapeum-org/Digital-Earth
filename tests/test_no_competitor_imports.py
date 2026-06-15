"""DX.3 — the pyramids-only enforcer.

Fails the build if any module in the interactive/3D/web tiers (``digitalearth.three_d`` /
``digitalearth.interactive`` / ``digitalearth.web``) imports a forbidden GIS competitor. Per ``CLAUDE.md``,
pyramids is the ONLY GIS dependency; these tiers must ingest data exclusively through pyramids
(numpy + geotransform + GeoDataFrame) and never reach around it to ``xarray``/``rasterio``/``rioxarray``/… —
see ``planning/interactive-3d/00-architecture-and-ingestion.md``.

``geopandas``/``shapely`` are forbidden as **imports** too: we *read* coordinates off the GeoDataFrame pyramids
returns (no import needed). A genuine need to import them is a signal to push the capability into pyramids, not to
relax this test.
"""
import ast
import pathlib

#: Top-level packages the interactive/3D/web tiers must never import (use pyramids instead).
FORBIDDEN = {
    "xarray",
    "rasterio",
    "rioxarray",
    "fiona",
    "gdal",
    "osgeo",
    "netCDF4",
    "cfgrib",
    "xee",
    "pyproj",
    "cartopy",
    "shapely",
    "geopandas",
    "pyvista_xarray",
    "pvxarray",
}

#: Package roots the rule applies to (relative to the repo root). Only ``src/`` is guarded — test
#: files are deliberately exempt: fixtures legitimately ``import geopandas`` to construct the
#: GeoDataFrames the tiers consume, which is not the same as a tier module reaching around pyramids.
_GUARDED_ROOTS = (
    "src/digitalearth/three_d",
    "src/digitalearth/interactive",
    "src/digitalearth/web",
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _top_level_imports(path: pathlib.Path):
    """Yield the top-level package name of every absolute import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def _guarded_modules():
    """Return every ``.py`` module under the guarded tier roots that currently exists."""
    modules = []
    for root in _GUARDED_ROOTS:
        modules.extend((_REPO_ROOT / root).rglob("*.py"))
    return modules


def test_tiers_import_no_gis_competitor():
    """No module in the interactive/3D tiers imports a forbidden GIS competitor."""
    offenders = {}
    for module in _guarded_modules():
        bad = FORBIDDEN & set(_top_level_imports(module))
        if bad:
            offenders[str(module.relative_to(_REPO_ROOT))] = sorted(bad)
    assert not offenders, (
        "interactive/3D tier modules must use pyramids, not a GIS competitor:\n"
        + "\n".join(f"  {mod}: {pkgs}" for mod, pkgs in offenders.items())
    )


def test_guard_detects_a_forbidden_import(tmp_path):
    """The guard's import scanner flags a forbidden import (so the rule has real teeth)."""
    sample = tmp_path / "bad_module.py"
    sample.write_text("import xarray as xr\nfrom rasterio import open\n", encoding="utf-8")
    assert FORBIDDEN & set(_top_level_imports(sample)) == {"xarray", "rasterio"}
