"""GlobeMixin — render a global field on a true textured 3-D sphere (via geovista).

Wraps a lon/lat field around an ellipsoidal Earth using **geovista** (the cartographic layer over PyVista), with
optional Natural-Earth coastlines. geovista is **optional and isolated**: it is imported *lazily inside*
:meth:`globe` (never at module import), so the rest of the 3-D tier works without it, and a missing install yields
a clear, actionable error rather than an import-time crash. CRS/reprojection stays in pyramids — projected input is
warped to EPSG:4326 via ``Dataset.to_crs`` before extraction, the same way every other tier reaches its display
CRS — and geovista is used only to drape the already-prepared lon/lat numpy field onto a sphere.

geovista pulls cartopy transitively — that is *its* dependency, never imported here (the HARD RULE /
``test_no_competitor_imports`` guard); this module imports only ``geovista`` itself, lazily.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from digitalearth.base.crs import declared_crs, is_geographic, reproject
from digitalearth.base.sources import Source, get_source
from digitalearth.base.spec.bounds import same_crs

#: Fallback scalar-array name ``geovista.Transform.from_1d`` assigns to the draped field (used only if the mesh
#: exposes no active scalars). Prefer ``mesh.active_scalars_name`` so the binding tracks geovista's own choice.
_GEOVISTA_DATA = "point_data"

#: The CRS a globe is drawn in. geovista drapes ``(lon, lat)`` onto a WGS84 sphere, so projected input is
#: warped to this through pyramids before it is extracted.
GEOGRAPHIC_EPSG = 4326


def _check_geographic(lon: np.ndarray, lat: np.ndarray) -> None:
    """Last-resort guard for input carrying **no CRS**: do its coordinates at least look like lon/lat?

    A pyramids ``Dataset`` never reaches here unreprojected (:meth:`GlobeMixin._to_geographic_source` warps it
    first), and a :class:`~digitalearth.base.sources.Source` that carries a projected CRS is rejected by that
    CRS, named outright. What is left is input with no CRS at all — a bare ``Source``, or a raw numpy array —
    for which the coordinate range is the only evidence available. geovista drapes ``(lon, lat)`` onto a WGS84
    sphere, so metre-scale coordinates would silently place the field in the wrong spot.

    Args:
        lon: Longitude coordinate vector.
        lat: Latitude coordinate vector.

    Raises:
        ValueError: if any coordinate falls outside geographic bounds (``|lon| > 360`` or ``|lat| > 90``).
    """
    if np.nanmax(np.abs(lon)) > 360.0 or np.nanmax(np.abs(lat)) > 90.0:
        raise ValueError(
            "globe() got data with no CRS whose coordinates look projected rather than lon/lat. Without a CRS "
            "there is nothing to reproject from, so reproject in pyramids first, e.g. dataset.to_crs(4326), "
            "then call globe()."
        )


def _name_crs(src: Source) -> str:
    """Name a source's CRS the way its owner spelled it, so the refusal quotes something recognisable.

    ``Source.crs`` may be an EPSG ``int``, an ``"EPSG:<code>"`` string or a proj4/WKT definition, so the code
    cannot simply be interpolated behind a hardcoded ``EPSG:`` prefix — that printed ``EPSG:EPSG:3857`` for
    the string spelling and ``EPSG:+proj=...`` for a definition.

    Args:
        src: The source whose CRS is being reported.

    Returns:
        ``"EPSG:<code>"`` when the CRS has an authority code, else the definition as given.
    """
    return f"EPSG:{src.epsg}" if src.epsg is not None else f"{src.crs!r}"


def _require_geographic(src: Source) -> None:
    """Raise unless ``src`` is a geographic lon/lat field, ready to drape on a sphere.

    Splits the two failure modes that used to be one coordinate-magnitude test: a source whose CRS pyramids
    can read is judged by that CRS (and a projected one is named by its code or definition), while a source
    with no CRS falls through to :func:`_check_geographic`, the only evidence left. The CRS is read by
    :func:`~digitalearth.base.crs.is_geographic`, which understands **every** spelling ``Source.crs`` may
    carry — a reader that understood only the ``int`` spelling called ``"EPSG:3857"`` "unknown" and let a
    Web-Mercator source through to the coordinate guard, which accepts small metre coordinates as lon/lat.

    Args:
        src: The extracted source about to be draped.

    Raises:
        ValueError: if the source carries a projected CRS, or carries none and its coordinates look projected.
    """
    geographic = is_geographic(src.crs)
    if geographic is True:
        return
    if geographic is False:
        raise ValueError(
            f"globe() expects geographic lon/lat data, but this source is in {_name_crs(src)}, a projected "
            "CRS. A pyramids Dataset is reprojected for you; a bare Source cannot be, so reproject it in "
            "pyramids first, e.g. dataset.to_crs(4326), then call globe()."
        )
    _check_geographic(
        np.asarray(src.x.values, dtype="float64"),
        np.asarray(src.y.values, dtype="float64"),
    )


def _require_geovista():
    """Import and return geovista, or raise a clear, actionable error when it is not installed.

    Returns:
        module: the imported ``geovista`` package.

    Raises:
        ImportError: when geovista is not installed, with the exact install command.
    """
    try:
        import geovista as gv
    except ImportError as exc:
        raise ImportError(
            "Scene3D.globe() needs the optional textured-globe dependency 'geovista'. "
            "Install it with:  pip install 'digitalearth[3d]'"
        ) from exc
    return gv


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


from digitalearth.base.spec import LayerSpec, Selection
from digitalearth.three_d.layer import drawing_props


class GlobeMixin(_MixinBase):
    """Adds :meth:`globe` — render a global lon/lat field on a textured sphere — to a :class:`Scene3D`.

    A capability mixin of :class:`~digitalearth.three_d.scene3d.Scene3D`: it is only ever composed into that scene
    class, never instantiated or subclassed on its own. Its methods reach the wrapped ``pyvista.Plotter``, the layer
    registry and the render/export lifecycle — and the sibling mixins' methods — through ``self``, and only the
    composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.three_d.base.Scene3DBase`, the state ``Scene3D``
    inherits. At runtime that base is plain ``object``, so composing this mixin leaves the ``Scene3D`` MRO exactly
    what it was before the annotation.

    See Also:
        digitalearth.three_d.scene3d.Scene3D: the composition that supplies the state these methods use.
        digitalearth.three_d.base.Scene3DBase: the typing-only base declared above the class.
    """

    def _to_geographic_source(self, data: Any, *, band: int = 1) -> Source:
        """Reproject ``data`` to EPSG:4326 through pyramids and extract it as a lon/lat :class:`Source`.

        The globe's display-CRS choke point, mirroring the interactive tier's ``_to_display_source``: anything
        that exposes pyramids' ``to_crs`` and whose declared CRS
        (:func:`~digitalearth.base.crs.declared_crs`) is not already geographic is warped by
        :func:`digitalearth.base.crs.reproject` (i.e. ``Dataset.to_crs(4326)``, plus the shared
        :class:`~digitalearth.base.crs.OffLimbError` translation). **No reprojection is implemented here** — a
        CRS pyramids cannot read is left alone rather than guessed at, and validated by
        :func:`_require_geographic` instead.

        When the warp does run, the CRS it warped to is passed to ``get_source(..., crs=...)`` rather than
        re-derived from the result: the extracted source is then the globe's CRS by construction, which is
        exactly the contract :attr:`digitalearth.base.sources.Source.crs` states.

        Args:
            data: A pyramids ``Dataset`` (or anything :func:`~digitalearth.base.sources.get_source` accepts),
                or an already-built :class:`~digitalearth.base.sources.Source` (which cannot be reprojected,
                and so must already be geographic).
            band: 1-based band index to read.

        Returns:
            Source: the lon/lat view of ``data``.

        Raises:
            ValueError: if the result is not geographic — see :func:`_require_geographic`.
        """
        warped = False
        if (
            not isinstance(data, Source)
            and hasattr(data, "to_crs")
            and is_geographic(declared_crs(data)) is False
        ):
            data = reproject(data, GEOGRAPHIC_EPSG)
            warped = True
        if isinstance(data, Source):
            src = data
        else:
            # Name the CRS we warped to rather than re-deriving it from the warped dataset: this is the
            # caller the ``crs=`` contract is written for (a warp's own CRS is the one the coordinates are
            # in, and pyramids cannot always report a code for it).
            src = get_source(data, band=band, crs=GEOGRAPHIC_EPSG if warped else None)
        _require_geographic(src)
        return src

    def globe(
        self,
        data: Any,
        *,
        name: Any = None,
        band: int = 1,
        cmap: str | None = None,
        coastlines: bool = True,
        coastline_resolution: str = "110m",
        coastline_color: str = "black",
        **kwargs: Any,
    ) -> Any:
        """Drape a global field onto a 3-D sphere and register it as a layer.

        Projected input is **reprojected, not refused**: a pyramids ``Dataset`` in any CRS is warped to
        EPSG:4326 through pyramids (``Dataset.to_crs``) on the way in, exactly as the static, interactive and
        web tiers do. Only input that cannot be reprojected — a bare
        :class:`~digitalearth.base.sources.Source` or raw array carrying a projected CRS or none at all — is
        rejected, with an error naming what is wrong.

        A globe draws in EPSG:4326, and declares that as the scene's display CRS when the scene has none.

        **The sphere spans the cell centres, not the cell edges**, as `terrain` does: geovista places one
        vertex per lon/lat pair, so a global field is drawn half a cell inside its own extent at each pole and
        meridian (#301).

        Args:
            data: A pyramids ``Dataset`` (or anything :func:`~digitalearth.base.sources.get_source` accepts), or an
                already-built :class:`~digitalearth.base.sources.Source` whose x/y are longitude/latitude and whose
                z is the field to drape.
            band: 1-based band index to read.
            cmap: Colormap for the draped field. ``None`` (the default) resolves it from the variable through
                :func:`~digitalearth.base.autostyle.auto_style` — the same lookup the other three tiers use, so
                a field is drawn in the same colours whichever tier renders it.
            coastlines: Overlay Natural-Earth coastlines on the globe.
            coastline_resolution: Coastline detail (``"110m"``, ``"50m"``, ``"10m"``).
            coastline_color: Colour of the coastline lines.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh` for the field sphere.

        Returns:
            The registered :class:`pyvista.Actor` for the field sphere.

        Raises:
            ImportError: if the optional ``geovista`` dependency is not installed.
            ValueError: if ``data`` cannot be reprojected and is not geographic — a ``Source`` in a projected
                CRS, or one with no CRS whose coordinates are not lon/lat — or if the scene is already drawn in
                a CRS other than EPSG:4326.

        Examples:
            - Drape a cosine-of-latitude field on a globe with coastlines (needs the ``3d`` extra / geovista):
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> from digitalearth.base.sources import get_source
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
        actor = self._add_described_layer(
            kind="raster",
            data=data,
            name=name,
            selection=Selection(band=(band,)),
            cmap=cmap,
            **kwargs,
        )
        if coastlines:
            self._add_described_layer(
                kind="coastlines",
                resolution=coastline_resolution,
                color=coastline_color,
            )
        return actor


def draw_globe(scene: Any, data: Any, layer: LayerSpec) -> Any:
    """Drape the field a globe layer describes onto a sphere.

    Args:
        scene: The scene being drawn into. A globe draws in EPSG:4326 and declares it as the scene's CRS.
        data: The layer's source object: a global raster.
        layer: The layer's description, whose selection names the band and whose props carry the colormap.

    Returns:
        The `(mesh, actor)` pair.

    Raises:
        ImportError: when geovista is not installed.
        ValueError: when the scene is already drawn in another CRS, or the data cannot be made geographic.
    """
    gv = _require_geovista()
    if scene.display_crs is None:
        scene.display_crs = GEOGRAPHIC_EPSG
    elif not same_crs(scene.display_crs, GEOGRAPHIC_EPSG):
        # geovista wraps lon/lat onto a sphere; the scene's other layers would be flat in another CRS.
        raise ValueError(
            f"globe() draws in EPSG:{GEOGRAPHIC_EPSG}, but this scene is drawn in {scene.display_crs!r}; "
            "draw the globe in its own Scene3D"
        )
    props = drawing_props(layer.symbology.props)
    cmap = props.pop("cmap", None)
    band = layer.selection.band[0] if layer.selection.band else 1
    src = scene._to_geographic_source(data, band=band)
    lon = np.asarray(src.x.values, dtype="float64")
    lat = np.asarray(src.y.values, dtype="float64")
    field = np.asarray(src.z.values, dtype="float64")

    mesh = gv.Transform.from_1d(lon, lat, data=field)
    # Colour by whatever scalar geovista set active, so the binding tracks geovista rather than a hardcoded
    # array name; fall back to the documented constant only if no active scalar is present.
    scalars = mesh.active_scalars_name or _GEOVISTA_DATA
    return mesh, scene.plotter.add_mesh(
        mesh, scalars=scalars, cmap=scene._auto_cmap(src, cmap), **props
    )


def draw_coastlines(scene: Any, data: Any, layer: LayerSpec) -> Any:
    """Draw the coastlines a globe asked for.

    Args:
        scene: The scene being drawn into.
        data: Unused — coastlines are geometry geovista holds, not a layer's source.
        layer: The layer's description, whose props carry the resolution and the colour.

    Returns:
        The `(mesh, actor)` pair.

    Raises:
        ImportError: when geovista is not installed.
    """
    _require_geovista()
    from geovista.geometry import coastlines as _load_coastlines

    props = drawing_props(layer.symbology.props)
    mesh = _load_coastlines(props.pop("resolution", "110m"))
    return mesh, scene.plotter.add_mesh(mesh, **props)
