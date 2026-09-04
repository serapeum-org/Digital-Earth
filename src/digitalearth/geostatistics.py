"""Visualize **geostatista** results in the Digital-Earth map tiers.

geostatista (the geostatistics engine on top of pyramids) owns the computation — kriging, variography and
spatial statistics; this module owns only the *map composition* of its outputs, exactly as the rest of
Digital-Earth composes pyramids/cleopatra. It reimplements no geostatistics.

Its inputs are the objects geostatista already returns, which are pyramids types Digital-Earth's tiers
already know how to draw:

* :func:`lisa_map` / :func:`hotspot_map` — a ``FeatureCollection`` annotated by
  ``geostatista.local_morans`` (a ``cluster`` column of LISA classes ``HH``/``LL``/``HL``/``LH``/``ns``) or by
  ``geostatista.getis_ord_gi`` / ``geostatista.hotspots`` (a ``hotspot`` column of ``hot``/``cold``/``ns``).
  Both are drawn as a **categorical choropleth** with the conventional GeoDa/PySAL colour scheme (hot/HH red,
  cold/LL blue, non-significant grey), keyed by a swatch legend.
* :func:`kriging_map` — a ``geostatista.KrigedSurface`` (a pyramids ``Dataset`` of the kriged estimate, with a
  ``.variance`` companion), drawn as a raster field with an optional overlay of the source sample points.

The categorical colouring rides Digital-Earth's shared ``scheme="categorical"`` path, so the same call renders
identically in the static, interactive and web tiers.

.. note::
   ``kriging_map`` is wired but **not yet exercisable** against pyramids 0.59: geostatista 0.2.0's
   ``KrigedSurface.from_arrays`` calls ``Dataset.create_from_array``, which pyramids 0.59 renamed to
   ``from_array`` / ``create`` — so ``Samples.krige`` / ``OrdinaryKriging.predict_grid`` raise
   ``AttributeError`` before a surface is ever produced. The visualization here is correct; it will work once
   geostatista is fixed and released. Tracked as a geostatista upstream bug.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from digitalearth._symbology import MISSING_COLOR, _categories
from digitalearth.scene import Map

#: Conventional LISA (local Moran) cluster colours — GeoDa/PySAL scheme. Keyed by the ``cluster`` label
#: ``geostatista.local_morans`` writes. High-High and Low-Low are the strong (same-sign) clusters, High-Low and
#: Low-High the spatial outliers, ``ns`` the non-significant remainder.
LISA_COLORS: Dict[str, str] = {
    "HH": "#d7191c",  # high value in a high neighbourhood (hot cluster)
    "LL": "#2c7bb6",  # low value in a low neighbourhood (cold cluster)
    "HL": "#fdae61",  # high outlier among low neighbours
    "LH": "#abd9e9",  # low outlier among high neighbours
    "ns": MISSING_COLOR,  # not significant
}

#: Conventional Getis-Ord Gi* hot/cold-spot colours. Keyed by the ``hotspot`` label
#: ``geostatista.getis_ord_gi`` / ``geostatista.hotspots`` writes.
HOTSPOT_COLORS: Dict[str, str] = {
    "hot": "#d7191c",   # statistically significant high cluster
    "cold": "#2c7bb6",  # statistically significant low cluster
    "ns": MISSING_COLOR,  # not significant
}


def _register_palette(categories: List[Any], color_map: Dict[str, str], name: str) -> str:
    """Register a matplotlib ``ListedColormap`` aligned to ``categories`` and return its registry name.

    Digital-Earth's categorical machinery colours the *sorted* distinct categories from a **named** colormap,
    position by position (see :func:`digitalearth._symbology.categorical_colors`). To pin each semantic class to
    its conventional colour regardless of which classes happen to be present, we build the colormap from the
    same sorted category list, look each label up in ``color_map`` (unknown labels fall back to a qualitative
    ``tab10`` cycle), and register it under ``name`` (overwriting any prior registration, so the colours always
    match the current data).

    Args:
        categories: The distinct categories to colour, already in the sorted order the tiers use.
        color_map: Semantic ``label -> #rrggbb`` mapping (e.g. :data:`LISA_COLORS`).
        name: The registry name to (re)register the colormap under.

    Returns:
        ``name`` — pass it as the ``cmap`` to a categorical ``choropleth``.
    """
    import warnings

    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap

    fallback = list(colormaps["tab10"].colors)
    colors = [color_map.get(str(cat), fallback[i % len(fallback)]) for i, cat in enumerate(categories)]
    with warnings.catch_warnings():
        # Re-registering the same name (colours track the current data) is intentional; matplotlib warns on it
        # even with force=True.
        warnings.filterwarnings("ignore", message="Overwriting the cmap", category=UserWarning)
        colormaps.register(ListedColormap(colors, name=name), name=name, force=True)
    return name


def _categorical_map(
    features: Any, column: str, color_map: Dict[str, str], palette_name: str, **kwargs: Any
) -> Map:
    """Draw ``features`` as a categorical choropleth of ``column`` with a conventional palette.

    Args:
        features: A polygon ``FeatureCollection`` carrying ``column`` (areal units).
        column: The categorical class column to colour by.
        color_map: Semantic ``label -> colour`` mapping for the classes.
        palette_name: Registry name for this map's colormap.
        **kwargs: Forwarded to :meth:`Map.choropleth`; ``crs`` sets the display CRS (defaults to the layer's).
            Pass an explicit ``cmap`` to override the conventional palette.

    Returns:
        The :class:`Map` with the categorical layer drawn.
    """
    if column not in features.columns:
        raise KeyError(
            f"column {column!r} not found — run the matching geostatista op first "
            f"(local_morans → 'cluster', getis_ord_gi/hotspots → 'hotspot'). Columns: {list(features.columns)}"
        )
    categories = _categories(features[column])
    if not categories:
        raise ValueError(f"column {column!r} has no non-null classes to draw")
    crs = kwargs.pop("crs", getattr(features, "epsg", None))
    cmap = kwargs.pop("cmap", None) or _register_palette(categories, color_map, palette_name)
    scene = Map(crs=crs) if crs is not None else Map()
    scene.choropleth(features, column=column, scheme="categorical", cmap=cmap, **kwargs)
    return scene


def lisa_map(features: Any, *, column: str = "cluster", **kwargs: Any) -> Map:
    """Map LISA (local Moran) clusters from ``geostatista.local_morans`` as a categorical choropleth.

    Args:
        features: A **polygon** ``FeatureCollection`` annotated by ``geostatista.local_morans`` — i.e. carrying a
            ``cluster`` column of ``HH``/``LL``/``HL``/``LH``/``ns`` labels.
        column: The cluster-label column (default ``"cluster"``).
        **kwargs: Forwarded to :meth:`Map.choropleth` (``crs``, ``title``, ``cmap`` to override the conventional
            LISA palette, …).

    Returns:
        A :class:`Map` with the LISA clusters drawn (HH red, LL blue, outliers light, ``ns`` grey), keyed by a
        swatch legend.

    Examples:
        - Colour LISA clusters on areal units (network call-free):
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import geopandas as gpd
            >>> from shapely.geometry import box
            >>> from pyramids.feature import FeatureCollection
            >>> from geostatista import local_morans, Weights
            >>> from digitalearth.geostatistics import lisa_map
            >>> polys = [box(i, j, i + 1, j + 1) for j in range(4) for i in range(4)]
            >>> vals = [0, 0, 1, 1] * 4
            >>> fc = FeatureCollection(gpd.GeoDataFrame({"v": vals}, geometry=polys, crs="EPSG:32631"))
            >>> lm = local_morans(fc, "v", Weights.queen(fc))
            >>> m = lisa_map(lm)
            >>> len(m.layers)
            1

            ```
    """
    return _categorical_map(features, column, LISA_COLORS, "_de_geostat_lisa", **kwargs)


def hotspot_map(features: Any, *, column: str = "hotspot", **kwargs: Any) -> Map:
    """Map Getis-Ord Gi* hot/cold spots from ``geostatista.getis_ord_gi`` / ``hotspots`` as a categorical choropleth.

    Args:
        features: A **polygon** ``FeatureCollection`` annotated by ``geostatista.getis_ord_gi`` or
            ``geostatista.hotspots`` — i.e. carrying a ``hotspot`` column of ``hot``/``cold``/``ns`` labels.
        column: The hotspot-label column (default ``"hotspot"``).
        **kwargs: Forwarded to :meth:`Map.choropleth` (``crs``, ``title``, ``cmap`` to override the conventional
            hot/cold palette, …).

    Returns:
        A :class:`Map` with the hot spots red, cold spots blue and the non-significant remainder grey, keyed by a
        swatch legend.
    """
    return _categorical_map(features, column, HOTSPOT_COLORS, "_de_geostat_hotspot", **kwargs)


def kriging_map(
    surface: Any, *, samples: Optional[Any] = None, variance: bool = False, field: str = "imshow", **kwargs: Any
) -> Map:
    """Drape a ``geostatista.KrigedSurface`` as a raster field, optionally overlaying the source samples.

    Args:
        surface: A ``geostatista.KrigedSurface`` (a pyramids ``Dataset`` of the kriged estimate, with a
            ``.variance`` companion).
        samples: Optional point ``FeatureCollection`` (e.g. the ``geostatista.Samples``) overlaid as a scatter.
        variance: When ``True`` draw the kriging *variance* surface (``surface.variance``) instead of the
            estimate — the standard uncertainty map.
        field: The raster field method to use (``"imshow"`` default, or ``"pcolormesh"`` / ``"contourf"``).
        **kwargs: Forwarded to the raster field method (``crs``, ``cmap``, ``title``, ``cbar_label``, …).

    Returns:
        A :class:`Map` with the surface drawn and, when given, the samples overlaid.

    .. note::
        Blocked upstream on pyramids 0.59: geostatista 0.2.0 cannot *produce* a ``KrigedSurface`` (its
        ``from_arrays`` calls the removed ``Dataset.create_from_array``), so this cannot yet be exercised
        end-to-end. The composition itself is correct.
    """
    dataset = surface.variance if variance else surface
    crs = kwargs.pop("crs", getattr(dataset, "epsg", None))
    scene = Map(crs=crs) if crs is not None else Map()
    getattr(scene, field)(dataset, **kwargs)
    if samples is not None:
        scene.scatter(samples)
    return scene
