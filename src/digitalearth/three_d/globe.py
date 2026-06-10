"""GlobeMixin — render a global field on a true textured 3-D sphere (via geovista).

Wraps a lon/lat field around an ellipsoidal Earth using **geovista** (the cartographic layer over PyVista), with
optional Natural-Earth coastlines. geovista is **optional and isolated**: it is imported *lazily inside*
:meth:`globe` (never at module import), so the rest of the 3-D tier works without it, and a missing install yields
a clear, actionable error rather than an import-time crash. CRS/reprojection stays in pyramids; geovista is used
only to drape the already-prepared lon/lat numpy field onto a sphere.

geovista pulls cartopy transitively — that is *its* dependency, never imported here (the HARD RULE /
``test_no_competitor_imports`` guard); this module imports only ``geovista`` itself, lazily.
"""
from typing import Any

import numpy as np

from digitalearth.sources import Source, get_source

#: Scalar-array name ``geovista.Transform.from_1d`` assigns to the draped field.
_GEOVISTA_DATA = "point_data"


def _require_geovista():
    """Import and return geovista, or raise a clear, actionable error when it is not installed.

    Returns:
        module: the imported ``geovista`` package.

    Raises:
        ImportError: when geovista is not installed, with the exact install command.
    """
    try:
        import geovista as gv
    except ImportError as exc:  # pragma: no cover - exercised via test monkeypatch
        raise ImportError(
            "Scene3D.globe() needs the optional textured-globe dependency 'geovista'. "
            "Install it with:  pip install 'digitalearth[3d]'"
        ) from exc
    return gv


class GlobeMixin:
    """Adds :meth:`globe` — render a global lon/lat field on a textured sphere — to a :class:`Scene3D`."""

    def globe(
        self,
        data: Any,
        *,
        band: int = 1,
        cmap: str = "viridis",
        coastlines: bool = True,
        coastline_resolution: str = "110m",
        coastline_color: str = "black",
        **kwargs: Any,
    ) -> Any:
        """Drape a global lon/lat field onto a 3-D sphere and register it as a layer.

        Args:
            data: A pyramids ``Dataset`` (or anything :func:`~digitalearth.sources.get_source` accepts), or an
                already-built :class:`~digitalearth.sources.Source`, whose x/y are longitude/latitude and whose
                z is the field to drape.
            band: 1-based band index to read.
            cmap: Colormap for the draped field.
            coastlines: Overlay Natural-Earth coastlines on the globe.
            coastline_resolution: Coastline detail (``"110m"``, ``"50m"``, ``"10m"``).
            coastline_color: Colour of the coastline lines.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh` for the field sphere.

        Returns:
            The registered :class:`pyvista.Actor` for the field sphere.

        Raises:
            ImportError: if the optional ``geovista`` dependency is not installed.

        Examples:
            - Drape a cosine-of-latitude field on a globe with coastlines (needs the ``3d`` extra / geovista):
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.sources import get_source
                >>> lon = np.linspace(-180, 180, 37)
                >>> lat = np.linspace(-90, 90, 19)
                >>> field = np.add.outer(np.cos(np.deg2rad(lat)), np.zeros(len(lon)))
                >>> scene = Scene3D(off_screen=True)
                >>> actor = scene.globe(get_source(field, x=lon, y=lat), coastlines=False)
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```
        """
        gv = _require_geovista()
        src = data if isinstance(data, Source) else get_source(data, band=band)
        lon = np.asarray(src.x.values, dtype="float64")
        lat = np.asarray(src.y.values, dtype="float64")
        field = np.asarray(src.z.values, dtype="float64")

        mesh = gv.Transform.from_1d(lon, lat, data=field)
        actor = self.add_mesh(mesh, scalars=_GEOVISTA_DATA, cmap=cmap, **kwargs)

        if coastlines:
            from geovista.geometry import coastlines as _load_coastlines

            self.add_mesh(_load_coastlines(coastline_resolution), color=coastline_color)
        return actor
