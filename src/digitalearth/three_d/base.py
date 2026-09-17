"""Scene3DBase — the core PyVista-plotter plumbing the 3D capability mixins build on.

``Scene3DBase`` owns the :class:`pyvista.Plotter`, the layer registry, and the render/export/context-manager
lifecycle. Capability mixins (terrain, point clouds, volumes, vectors, globe) live in sibling modules and add
``terrain()`` / ``point_cloud()`` / … methods that call ``self.add_mesh(...)``; the public
:class:`digitalearth.three_d.scene3d.Scene3D` composes the base with those mixins — exactly mirroring the 2-D
``Map(GeoLayerBase, RasterMixin, …)`` pattern.

PyVista is a renderer, not a GIS engine: meshes are built from pyramids-sourced numpy (never xarray/rasterio — see
the tier's HARD RULE); all CRS/reproject work stays in pyramids. The default ``off_screen`` follows
:data:`pyvista.OFF_SCREEN`, so the same code renders interactively on a desktop and headless in CI.
"""

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Self, Union

import numpy as np

from digitalearth.base.crs import OffLimbError, declared_crs, reproject
from digitalearth.base.display import auto_cmap, needs_reproject
from digitalearth.base.sources import Source
from digitalearth.base.spec import Scale

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    import pyvista as pv

logger = logging.getLogger(__name__)

#: Anything acceptable as an output destination.
Destination = Union[str, "os.PathLike[str]"]

#: Suffixes :meth:`Scene3DBase.save` renders as a raster frame. PyVista writes the format the suffix names —
#: ``.tif`` really is a TIFF — so this is "a rendered frame", not "a PNG".
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: The one scene-export suffix :meth:`Scene3DBase.save` does not look up in :data:`SCENE_EXPORTERS`:
#: it routes through :meth:`Scene3DBase.export_html`, which normalises the suffix and guards the
#: VTK build. Named once so the dispatch, the normalisation and the help text cannot drift apart.
HTML_SUFFIX: str = ".html"

#: Suffixes :meth:`Scene3DBase.save` exports the *scene* to, mapped to the :class:`pyvista.Plotter` method that
#: writes each. ``.html`` is dispatched separately because it routes through :meth:`Scene3DBase.export_html`,
#: which normalises the suffix and guards the VTK build. ``.glb`` is deliberately absent: VTK's glTF exporter
#: writes the JSON flavour, so a ``.glb`` name would promise a binary container it does not produce.
SCENE_EXPORTERS: dict[str, str] = {
    ".gltf": "export_gltf",
    ".obj": "export_obj",
    ".vrml": "export_vrml",
    ".wrl": "export_vrml",
    ".vtksz": "export_vtksz",
}


def classified_scalars(
    values: Any, *, scheme: Any | None, k: int, cmap: Any
) -> dict[str, Any]:
    """Turn a value column into the ``scalars``/``cmap`` keywords that colour a PyVista layer.

    The 3-D counterpart of the classification the other tiers apply to a choropleth, and deliberately the
    same computation: a graduated ``scheme`` gets its class edges from
    :meth:`~digitalearth.base.spec.scale.Scale.breaks_of` (which reaches the registered classifier, so every
    tier cuts identical edges) and ``scheme="categorical"`` its colours from
    :func:`digitalearth.base.symbology.categorical_colors`, so one ``scheme``/``k``/``cmap`` triple paints the
    same classes here as on a static, interactive or web map.

    What differs is only how the result is handed to the engine. VTK has no notion of class breaks: it maps a
    scalar range through a lookup table. So a classified layer is rendered by replacing the values with their
    **class index** and handing PyVista a table of exactly that many colours — ``clim`` set to
    ``(-0.5, n - 0.5)`` so index *i* lands in the middle of the *i*-th colour. The picture is a flat-filled
    classified layer, which is what the scheme asked for.

    **A missing value stays missing.** A feature whose value is ``NaN``/``None``/``pd.NA`` gets no class
    index at all: it reaches the engine as ``NaN``, and the style carries ``nan_color`` —
    :data:`~digitalearth.base.symbology.MISSING_COLOR`, the same neutral grey the static, interactive and web
    choropleths paint missing data with. A lookup table's NaN colour is VTK's own mechanism for "no value",
    so nothing is squeezed into the ramp to express it. It has to be done deliberately because both
    classifiers would otherwise give missing data a real class: ``np.digitize`` sorts ``NaN`` above every
    edge, so a graduated scheme would clip it into the **top** class and draw it as the column's maximum — a
    hotspot that is not in the data — and a categorical scheme would fall back to the first category.

    Args:
        values: The per-feature values to colour by; coerced to float for the graduated schemes, taken as-is
            (labels or codes) for ``"categorical"``.
        scheme: ``None`` for a continuous ramp over the raw values (the default everywhere);
            ``"categorical"`` for one colour per distinct value; any other
            ``cleopatra.styling.styles.classify`` scheme name (``"quantiles"``, ``"equal_interval"``,
            ``"fisher_jenks"``, …) — or an explicit sequence of class edges — for graduated colouring.
        k: Number of classes for the graduated schemes; ignored when ``scheme`` is ``None`` or
            ``"categorical"``.
        cmap: Colormap name for the ramp / classes. For ``"categorical"`` it is resolved through
            :func:`~digitalearth.base.symbology.resolve_categorical_cmap`, which swaps a continuous default
            for a qualitative one. An explicit **sequence** of colours is taken as given rather than
            sampled, so on a graduated scheme it must carry exactly one colour per class; a shorter list is
            refused, not recycled, because the lookup table would otherwise clamp every class past its end
            onto the last colour and they would all render identically.

    Returns:
        dict: keyword arguments to splat into :meth:`pyvista.Plotter.add_mesh` /
        :meth:`pyvista.Plotter.add_points` — always ``scalars``, ``cmap`` and ``nan_color``, plus ``clim``
        and ``n_colors`` when the layer was classified.

    Raises:
        ValueError: when the column cannot be classified — an unknown scheme, a constant column with no
            spread to split, ``k`` below 1, or (for ``"categorical"``) no non-null values — and when an
            explicit ``cmap`` sequence carries a different number of colours than the scheme produced
            classes. The message names the scheme and ``k`` so the caller can see which of those it was.

    Examples:
        - No scheme is a continuous ramp: the values reach the engine unchanged:
            ```python
            >>> from digitalearth.three_d.base import classified_scalars
            >>> style = classified_scalars([1.0, 5.0, 9.0], scheme=None, k=5, cmap="viridis")
            >>> sorted(style), [float(v) for v in style["scalars"]]
            (['cmap', 'nan_color', 'scalars'], [1.0, 5.0, 9.0])

            ```
        - A graduated scheme replaces the values with class indices and supplies one colour per class:
            ```python
            >>> from digitalearth.three_d.base import classified_scalars
            >>> style = classified_scalars(
            ...     [1.0, 2.0, 3.0, 40.0], scheme="quantiles", k=2, cmap="viridis"
            ... )
            >>> style["n_colors"], len(style["cmap"]), style["clim"]
            (2, 2, (-0.5, 1.5))
            >>> sorted(set(int(v) for v in style["scalars"]))
            [0, 1]

            ```
        - A categorical scheme gives every distinct value its own colour:
            ```python
            >>> from digitalearth.three_d.base import classified_scalars
            >>> style = classified_scalars(
            ...     ["a", "b", "a", "c"], scheme="categorical", k=5, cmap="tab10"
            ... )
            >>> style["n_colors"], [int(v) for v in style["scalars"]]
            (3, [0, 1, 0, 2])

            ```
        - A feature with no value is drawn as missing, never as the top class:
            ```python
            >>> import numpy as np
            >>> from digitalearth.three_d.base import classified_scalars
            >>> style = classified_scalars(
            ...     [1.0, 2.0, 3.0, 40.0, np.nan], scheme="quantiles", k=4, cmap="viridis"
            ... )
            >>> [float(v) for v in style["scalars"]]
            [0.0, 1.0, 2.0, 3.0, nan]
            >>> style["nan_color"]
            '#cccccc'

            ```
    """
    from digitalearth.base.symbology import (
        MISSING_COLOR,
        categorical_colors,
        resolve_categorical_cmap,
        sample_cmap,
    )

    if scheme is None:
        return {
            "scalars": np.asarray(values, dtype="float64"),
            "cmap": cmap,
            "nan_color": MISSING_COLOR,
        }

    if isinstance(scheme, str) and scheme.lower() == "categorical":
        categories, colours = categorical_colors(
            values, cmap=resolve_categorical_cmap(cmap)
        )
        index = {category: position for position, category in enumerate(categories)}
        # `categorical_colors` builds its categories from the non-null values alone, so a value the index
        # has no entry for is a missing one — code it NaN rather than defaulting it into category 0, which
        # would paint a feature with no value in the first category's colour.
        codes = np.array(
            [
                index.get(value, np.nan)
                for value in np.asarray(values, dtype=object).ravel()
            ],
            dtype="float64",
        )
        return _discrete_style(codes, colours)

    numbers = np.asarray(values, dtype="float64")
    try:
        edges = Scale.breaks_of(numbers, scheme, k)
    # ValueError, not Exception: get_classifier raises RuntimeError when nothing filled the seam, and
    # swallowing that would present a wiring failure as bad data. The other two tiers already let it
    # through, so catching it here made the three disagree on exactly that case.
    except ValueError as error:  # unknown scheme, constant column, k < 1 …
        # Scale's own message already names the scheme and `k`, and there is no column here to add, so
        # this only normalises the exception type the tiers raise.
        raise ValueError(f"cannot classify values: {error}") from error
    # `edges` bounds the classes, so it holds one more entry than there are classes; digitize against the
    # interior edges to land every value in 0 .. n_classes - 1 (clip catches the closed upper bound).
    n_classes = max(len(edges) - 1, 1)
    codes = np.clip(
        np.digitize(numbers, np.asarray(edges)[1:-1]), 0, n_classes - 1
    ).astype("float64")
    # `np.digitize` sorts NaN above every edge, so a missing value would be clipped into the top class and
    # render as the column's maximum. `classify` already cut the edges from the finite values alone, so the
    # classes themselves are unaffected — only the missing values need lifting back out of the ramp.
    codes[np.isnan(numbers)] = np.nan
    colours = sample_cmap(cmap, n_classes)
    if len(colours) != n_classes:
        # A colormap *name* is sampled to fit; an explicit sequence is taken as given. `_discrete_style`
        # then derives `clim`/`n_colors` from the sequence's length, so a short one clamped every class past
        # its end onto the last colour — three of five classes vanished with no warning (review M5).
        raise ValueError(
            f"cannot classify values (scheme={scheme!r}, k={k}): cmap has {len(colours)} colours for "
            f"{n_classes} classes; pass one colour per class, or a colormap name to sample"
        )
    return _discrete_style(codes, colours)


def _discrete_style(codes: np.ndarray, colours: list[str]) -> dict[str, Any]:
    """Build the add_mesh keywords that paint integer ``codes`` with one flat colour each.

    Args:
        codes: Per-feature class index, ``0 .. len(colours) - 1``, with ``NaN`` where the feature had no
            value to classify.
        colours: One colour per class, in class order.

    Returns:
        dict: ``scalars``/``cmap``/``clim``/``n_colors``/``nan_color`` for :meth:`pyvista.Plotter.add_mesh`.
        ``clim`` is half-open around the integers so each index sits at the centre of its colour band rather
        than on the boundary between two, and ``nan_color`` is what the lookup table paints the ``NaN``
        codes with — so missing data reads as missing instead of borrowing a class's colour.
    """
    from digitalearth.base.symbology import MISSING_COLOR

    return {
        "scalars": codes,
        "cmap": colours,
        "clim": (-0.5, len(colours) - 0.5),
        "n_colors": len(colours),
        "nan_color": MISSING_COLOR,
    }


def supported_destinations() -> str:
    """Return the "use one of these suffixes" sentence :meth:`Scene3DBase.save` raises with.

    Kept next to :data:`IMAGE_SUFFIXES` / :data:`SCENE_EXPORTERS` so the error text and the dispatch table
    cannot drift apart.

    Returns:
        The supported suffixes, split into the two families ``save`` dispatches.

    Examples:
        ```python
        >>> from digitalearth.three_d.base import supported_destinations
        >>> supported_destinations().startswith("use a raster-frame suffix")
        True

        ```
    """
    frames = " ".join(IMAGE_SUFFIXES)
    exports = " ".join([HTML_SUFFIX, *sorted(SCENE_EXPORTERS)])
    return f"use a raster-frame suffix ({frames}) or a scene-export suffix ({exports})"


def house_theme() -> "pv.themes.Theme":
    """Return Digital-Earth's default PyVista theme (document-style, anti-aliased).

    Returns:
        pyvista.themes.Theme: a tuned :class:`pyvista.themes.DocumentTheme` — white background, ``viridis``
        default colormap, SSAA anti-aliasing — for clean publication-grade frames.

    Examples:
        - Read back the settings a scene renders with by default:
            ```python
            >>> from digitalearth.three_d.base import house_theme
            >>> theme = house_theme()
            >>> theme.cmap
            'viridis'
            >>> theme.background.hex_rgb
            '#ffffff'
            >>> theme.anti_aliasing
            'ssaa'

            ```
        - Every call hands back a fresh theme, so tweaking one scene's colours leaves the next untouched:
            ```python
            >>> from digitalearth.three_d.base import house_theme
            >>> mine = house_theme()
            >>> mine.cmap = "magma"
            >>> house_theme().cmap
            'viridis'

            ```

    See Also:
        Scene3DBase: applies this theme whenever its ``theme`` argument is left as ``None``.
    """
    import pyvista as pv

    theme = pv.themes.DocumentTheme()
    theme.background = "white"
    theme.cmap = "viridis"
    theme.anti_aliasing = "ssaa"
    theme.font.color = "black"
    return theme


def _pyvista_vtk_root() -> str:
    """Return the top-level VTK package pyvista is bound to (``vtkmodules`` for a stock build).

    pyvista resolves this itself and caches it as ``pyvista._vtk._VTK_ROOT``, which is the authoritative
    answer: the backend can be any distribution name, not only a ``vtk``-prefixed one — ``PYVISTA_VTK_BACKEND``
    names it outright, and ``cvista`` is selected merely by being importable. That attribute arrived in pyvista
    0.49, exactly the versions where the trame component branch is reachable. On 0.48, where it is absent, the
    MRO of a pyvista type gives the same answer for every stock build.

    Returns:
        The package name, e.g. ``"vtkmodules"``.
    """
    import pyvista as pv

    resolved = getattr(getattr(pv, "_vtk", None), "_VTK_ROOT", None)
    if resolved:
        return str(resolved)
    for klass in pv.PolyData.__mro__:
        root: str = klass.__module__.split(".")[0]
        if root.startswith("vtk"):
            return root
    return "vtkmodules"


def _trame_vtk_root() -> str:
    """Return the top-level VTK package trame will use.

    trame resolves its VTK binding through ``VTK_MODULE_NAME``, caching the import as ``vtk_module``.

    Returns:
        The package name, e.g. ``"vtkmodules"``.
    """
    resolved = sys.modules.get("vtk_module")
    if resolved is not None:
        return resolved.__name__
    return os.environ.get("VTK_MODULE_NAME", "vtkmodules")


def _require_one_vtk_build() -> None:
    """Raise if trame and pyvista are bound to different VTK builds.

    A process must use one VTK build: objects cannot be shared between two, and handing a mesh from one to a
    renderer built against the other fails deep inside trame on a wrapped-type mismatch. pyvista 0.49 makes
    this check inside its (deprecated) ``Plotter.export_html`` and raises a ``RuntimeError`` rather than an
    ``ImportError``, so a misconfiguration cannot be mistaken for a missing package; going straight to the
    plotter component skips it. pyvista 0.48 makes no such check anywhere. Running it here therefore restores
    it on the component branch and adds it on the fallback, so every export is guarded on both versions.

    Raises:
        RuntimeError: If the two roots differ, naming the variable to set.
    """
    trame_root, pyvista_root = _trame_vtk_root(), _pyvista_vtk_root()
    if trame_root != pyvista_root:
        raise RuntimeError(
            f"trame is using the {trame_root!r} VTK build but PyVista is using {pyvista_root!r}. Objects "
            f"cannot be shared between two VTK builds — set VTK_MODULE_NAME={pyvista_root} before importing "
            "trame, or install a single VTK."
        )


class Scene3DBase:
    """Core single-:class:`pyvista.Plotter` host: layer registry + render/export lifecycle.

    **The scene has one display CRS, and every layer is placed in it.** Given as `crs=`, or — when omitted —
    taken from the first layer that declares a CRS, so a scene whose layers share one CRS reprojects nothing
    and renders exactly as its data is. A later layer in another CRS is reprojected through pyramids
    (`Dataset.to_crs` / `FeatureCollection.to_crs`) before it becomes geometry; data that declares no CRS — a
    bare array, a hand-built mesh — is placed as given. `globe()` draws in EPSG:4326 and declares it. Unlike a
    2-D map, the camera, not a projection, then decides what the viewer sees.

    Args:
        off_screen: Render without opening a window. ``None`` (default) follows :data:`pyvista.OFF_SCREEN`.
        window_size: Render window size in pixels (``(width, height)``).
        theme: A PyVista theme to apply. ``None`` uses :func:`house_theme`.
        strict: How a layer with nothing to draw is handled. ``False`` (default) skips it with a warning
            naming the layer and why, so one empty layer never kills a composed scene; ``True`` raises
            :class:`~digitalearth.base.crs.OffLimbError` instead.
        crs: The display CRS every layer is placed in. `None` (default) takes the CRS of the first layer that
            declares one.
        **plotter_kwargs: Forwarded to :class:`pyvista.Plotter`.

    Attributes:
        plotter: The wrapped :class:`pyvista.Plotter`.
        layers: Registered ``(mesh, actor)`` pairs, in add order.
        strict: Whether an empty layer raises rather than being skipped (see ``strict`` above).
        display_crs: The CRS the scene draws in, or `None` while no `crs=` was given and no layer has declared
            one (see the paragraph above).

    Examples:
        - Build a scene, stack two meshes on its single plotter, and read the layer registry back:
            ```python
            >>> import pyvista as pv
            >>> from digitalearth.three_d.base import Scene3DBase
            >>> scene = Scene3DBase(off_screen=True)
            >>> _ = scene.add_mesh(pv.Sphere())
            >>> _ = scene.add_mesh(pv.Cube(center=(3, 0, 0)))
            >>> len(scene.layers)
            2
            >>> scene.close()

            ```
        - Size the render window, then render a frame from it:
            ```python
            >>> from digitalearth.three_d.base import Scene3DBase
            >>> scene = Scene3DBase(off_screen=True, window_size=(320, 240))
            >>> list(scene.plotter.window_size)
            [320, 240]
            >>> scene.screenshot().ndim
            3
            >>> scene.close()

            ```
        - Used as a context manager the plotter is closed on the way out, even if the body raises:
            ```python
            >>> import pyvista as pv
            >>> from digitalearth.three_d.base import Scene3DBase
            >>> with Scene3DBase(off_screen=True) as scene:
            ...     _ = scene.add_mesh(pv.Sphere())
            ...     len(scene.layers)
            1

            ```

    See Also:
        digitalearth.three_d.scene3d.Scene3D: composes this base with the terrain/point-cloud/volume/vector
            capability mixins, and is the class to use directly.
    """

    def __init__(
        self,
        off_screen: bool | None = None,
        window_size: tuple[int, int] = (1024, 768),
        theme: "pv.themes.Theme | None" = None,
        strict: bool = False,
        crs: Any = None,
        **plotter_kwargs: Any,
    ):
        """Build the 3-D scene; the PyVista plotter behind it is built when something first needs it.

        Args:
            off_screen: Render without opening a window. `None` follows PyVista's own setting, which is what
                makes the tier usable in a notebook and in CI without changing the call.
            window_size: Render size in pixels, used for both the window and `screenshot`.
            theme: PyVista theme; defaults to the package's document-style theme.
            strict: Raise `OffLimbError` for a layer with nothing to draw — an empty point table, a DEM with
                no finite elevation — instead of skipping it with a warning.
            crs: The display CRS every layer is placed in. `None` takes the CRS of the first layer that
                declares one.
            **plotter_kwargs: Forwarded to `pyvista.Plotter` when it is built.
        """
        self._plotter_settings: dict[str, Any] = {
            "off_screen": off_screen,
            "window_size": list(window_size),
            "theme": theme,
            **plotter_kwargs,
        }
        self._plotter: Optional["pv.Plotter"] = None
        self.layers: list[tuple[Any, Any]] = []
        #: Whether a layer with nothing to draw raises instead of being skipped with a warning.
        self.strict: bool = strict
        #: The CRS every layer is placed in; `None` until given or declared by the first layer carrying one.
        self.display_crs: Any = crs

    @property
    def plotter(self) -> "pv.Plotter":
        """The PyVista plotter the scene draws on, built the first time it is asked for.

        Building a scene opens no render window: describing layers needs none, and a scene that is only
        described never pays for one. The settings given to the constructor are applied when the plotter is
        built, with :func:`house_theme` when no theme was given.

        Returns:
            The scene's `pyvista.Plotter`.

        Examples:
            - The plotter carries the size the scene was built with:
                ```python
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True, window_size=(320, 240))
                >>> list(scene.plotter.window_size)
                [320, 240]
                >>> scene.close()

                ```
            - Asking twice returns the same plotter:
                ```python
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> scene.plotter is scene.plotter
                True
                >>> scene.close()

                ```
        """
        if self._plotter is None:
            import pyvista as pv

            settings = dict(self._plotter_settings)
            settings["theme"] = settings["theme"] or house_theme()
            self._plotter = pv.Plotter(**settings)
        return self._plotter

    @plotter.setter
    def plotter(self, plotter: Any) -> None:
        """Replace the plotter the scene draws on.

        Args:
            plotter: The plotter to use from now on; a stand-in with the methods the scene calls will do.
        """
        self._plotter = plotter

    def _place(self, data: Any, *, layer: str) -> Any:
        """Return `data` in the scene's display CRS, adopting the data's CRS when the scene has none yet.

        Args:
            data: A layer's input — a pyramids `Dataset` or `FeatureCollection`, a `GeoDataFrame`, a `Source`,
                a `PointArrays`, or a bare array.
            layer: The builder's name, for the message.

        Returns:
            `data` itself when it declares no CRS, when it sets the scene's CRS, or when it is already in it;
            otherwise the reprojected data.

        Raises:
            ValueError: when `data` is in another CRS and cannot be reprojected — a `Source` or `PointArrays`,
                which hold coordinates rather than a warpable dataset.
            OffLimbError: when the reprojection places none of the data.

        Examples:
            - The first layer carrying a CRS sets the scene's; data with none is placed as given:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> placed = scene._place(np.zeros((2, 3)), layer="demo")
                >>> scene.display_crs is None
                True
                >>> _ = scene._place(Dataset.read_file("examples/data/acc4000.tif"), layer="demo")
                >>> scene.display_crs
                32618

                ```
        """
        own = declared_crs(data)
        if own is None:
            return data
        if self.display_crs is None:
            self.display_crs = own
            return data
        if not needs_reproject(data, self.display_crs):
            return data
        if not hasattr(data, "to_crs"):
            raise ValueError(
                f"{layer}() got data in {own!r}, but this scene is drawn in {self.display_crs!r}, and a "
                f"{type(data).__name__} cannot be reprojected; reproject it in pyramids first "
                "(Dataset.to_crs / FeatureCollection.to_crs)"
            )
        return reproject(data, self.display_crs)

    def _skip_empty(self, layer: str, reason: str) -> None:
        """Report that ``layer`` drew nothing, by warning (default) or raising (``strict=True``).

        A 3-D scene is usually composed from several layers, and one of them having nothing to place — an
        all-nodata DEM tile, a feature collection whose geometries carry no rings, an empty point table — is
        not a reason to lose the other layers. So the default is to skip it and say so at ``WARNING``, which
        is visible without configuring logging; otherwise the only symptom is a missing layer.

        ``strict=True`` is for pipelines where a silently absent layer is worse than a failure: it raises
        :class:`~digitalearth.base.crs.OffLimbError`, the same type the 2-D tiers use for "none of the data
        could be placed".

        Args:
            layer: The public method that drew nothing, named in the warning / error.
            reason: Why there was nothing to draw, phrased to complete "``layer``: ``reason``".

        Raises:
            OffLimbError: when the scene was constructed with ``strict=True``.
        """
        if self.strict:
            raise OffLimbError(f"{layer}: {reason}")
        logger.warning("%s: %s — nothing drawn", layer, reason)

    @property
    def vertical_exaggeration(self) -> float:
        """The vertical (z) scale the whole scene is rendered at (``1.0`` = true scale).

        Exaggeration is a property of the **view**, never of the geometry. PyVista applies it as a renderer
        transform (:meth:`pyvista.Plotter.set_scale`), which scales every actor — including ones added after it
        is set — so one scene has exactly one exaggeration, the value can be read back, and changing it costs a
        camera reset rather than a mesh rebuild. :meth:`digitalearth.three_d.Scene3D.terrain`'s
        ``z_exaggeration`` argument sets this.

        Returns:
            float: the current z scale.

        Examples:
            - A fresh scene renders at true scale; the value set reads straight back:
                ```python
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> scene.vertical_exaggeration
                1.0
                >>> scene.vertical_exaggeration = 4.0
                >>> scene.vertical_exaggeration
                4.0
                >>> scene.close()

                ```
            - It is the view that stretches, not the mesh — and a layer added afterwards is stretched too:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> scene.vertical_exaggeration = 4.0
                >>> sphere = pv.Sphere()
                >>> actor = scene.add_mesh(sphere)
                >>> round(sphere.bounds[5] - sphere.bounds[4], 3)
                1.0
                >>> round(actor.scale[2], 3)
                4.0
                >>> scene.close()

                ```
        """
        return float(self.plotter.renderer.scale[2])

    @vertical_exaggeration.setter
    def vertical_exaggeration(self, factor: float) -> None:
        """Set the scene's vertical (z) view scale.

        Args:
            factor: The z scale to render at (``1.0`` = true scale, ``>1`` accentuates relief).
        """
        self.plotter.set_scale(zscale=float(factor), render=False)

    def _auto_cmap(
        self, source: Source, cmap: str | None, fallback: str = "viridis"
    ) -> str:
        """Resolve a colormap: the caller's ``cmap`` if given, else the autostyle default.

        Mirrors the ``interactive`` and ``web`` tiers' ``_auto_cmap`` (and the static ``Map``'s ``auto_style``
        call), so a variable is drawn in the same colours whichever tier renders it — the same
        :func:`digitalearth.base.autostyle.auto_style` variable→style lookup, including the ECMWF-Magics match.
        ``fallback`` is the tier's own last-resort literal, and sits *behind* the lookup: it is reached only
        when ``auto_style`` yields no colormap at all.

        Args:
            source: The :class:`~digitalearth.base.sources.Source` whose variable drives the lookup.
            cmap: The caller-supplied colormap, or ``None`` to auto-resolve.
            fallback: Colormap to fall back on when the lookup yields none.

        Returns:
            The colormap name to use.
        """
        return auto_cmap(source, cmap, fallback)

    def _add_actor(self, mesh: Any, actor: Any) -> Any:
        """Register a rendered ``mesh`` and its ``actor``, returning the actor.

        Args:
            mesh: The PyVista mesh that was added to the plotter.
            actor: The :class:`pyvista.Actor` the plotter produced.

        Returns:
            The ``actor`` (so callers can chain or tweak its properties).
        """
        self.layers.append((mesh, actor))
        return actor

    def add_mesh(self, mesh: Any, **kwargs: Any) -> Any:
        """Add a PyVista ``mesh`` to the scene and register it as a layer.

        The low-level entry point the capability mixins build on. ``kwargs`` pass straight to
        :meth:`pyvista.Plotter.add_mesh` (``scalars``, ``cmap``, ``opacity``, ``show_edges``, ``pbr`` …).

        Args:
            mesh: Any PyVista dataset (``ImageData``/``StructuredGrid``/``PolyData``/``UnstructuredGrid``).
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered :class:`pyvista.Actor`.

        Examples:
            - Add one mesh and find it, paired with its actor, in the layer registry:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> actor = scene.add_mesh(pv.Sphere())
                >>> mesh, registered = scene.layers[0]
                >>> registered is actor
                True
                >>> scene.close()

                ```
            - Style the mesh through ``kwargs``, then tune the returned actor further:
                ```python
                >>> import numpy as np, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> grid = pv.ImageData(dimensions=(8, 8, 1))
                >>> grid.point_data["z"] = np.linspace(0.0, 1.0, 64)
                >>> scene = Scene3DBase(off_screen=True)
                >>> actor = scene.add_mesh(grid, scalars="z", cmap="terrain", show_edges=True)
                >>> actor.prop.opacity = 0.5
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```

        See Also:
            add_volume: the ray-cast counterpart, for scalar fields rather than surfaces.
        """
        actor = self.plotter.add_mesh(mesh, **kwargs)
        return self._add_actor(mesh, actor)

    def add_volume(self, volume: Any, **kwargs: Any) -> Any:
        """Add a volumetric ``volume`` (ray-cast rendering) and register it as a layer.

        Args:
            volume: An ``ImageData``/``UnstructuredGrid`` carrying a scalar field to ray-cast.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_volume`.

        Returns:
            The registered volume actor.

        Examples:
            - Ray-cast a 3-D scalar field and see it registered as one layer:
                ```python
                >>> import numpy as np, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> grid = pv.ImageData(dimensions=(6, 6, 6))
                >>> grid.cell_data["v"] = np.linspace(0.0, 1.0, 125)
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_volume(grid, cmap="viridis")
                >>> len(scene.layers)
                1
                >>> scene.close()

                ```
            - A volume and a surface share the one plotter, stacking in add order:
                ```python
                >>> import numpy as np, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> grid = pv.ImageData(dimensions=(6, 6, 6))
                >>> grid.cell_data["v"] = np.linspace(0.0, 1.0, 125)
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_volume(grid)
                >>> _ = scene.add_mesh(pv.Sphere(radius=1.0, center=(2, 2, 2)))
                >>> len(scene.layers)
                2
                >>> scene.close()

                ```

        See Also:
            add_mesh: the surface counterpart, and the method the capability mixins call.
        """
        actor = self.plotter.add_volume(volume, **kwargs)
        return self._add_actor(volume, actor)

    def screenshot(self, path: Destination | None = None, **kwargs: Any) -> np.ndarray:
        """Render the scene off-screen and return the RGB image (optionally writing it to ``path``).

        Args:
            path: Optional destination for the PNG, as a string or ``os.PathLike``. When ``None`` the
                image is only returned.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.screenshot`.

        Returns:
            numpy.ndarray: the ``(height, width, 3)`` RGB frame.

        Examples:
            - Render to memory and work with the frame as an array:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True, window_size=(200, 150))
                >>> _ = scene.add_mesh(pv.Sphere())
                >>> frame = scene.screenshot()
                >>> frame.ndim, frame.shape[-1]
                (3, 3)
                >>> bool(frame.any())
                True
                >>> scene.close()

                ```
            - Write a PNG to disk; the frame is still returned:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3DBase(off_screen=True)
                ...     _ = scene.add_mesh(pv.Cube())
                ...     out = os.path.join(folder, "frame.png")
                ...     frame = scene.screenshot(path=out)
                ...     scene.close()
                ...     os.path.getsize(out) > 0
                True

                ```

        See Also:
            save: dispatches here for the raster-frame suffixes (:data:`IMAGE_SUFFIXES`).
        """
        return self.plotter.screenshot(filename=path, return_img=True, **kwargs)

    def export_html(self, path: Destination) -> str:
        """Export the scene to a self-contained interactive HTML page (via trame/vtk.js).

        The page embeds the whole mesh, so it grows with the geometry rather than with the rendered image: about
        a 1 MB vtk.js floor plus ~19 bytes per point (a 12x12 grid gives 1.1 MB, 512x512 gives 6 MB, and a
        4700x4700 DEM gives ~430 MB). Past a modest tile the result is technically interactive but too heavy to
        sit beside a notebook or in docs — prefer :meth:`digitalearth.three_d.Scene3D.orbit`, which writes a
        compact GIF/MP4 fly-through of the same scene.

        The destination is normalised to a ``.html`` suffix before writing, and that normalised path is what
        comes back. pyvista's own behaviour here differs by version — the ``trame-pyvista`` component used from
        0.49 rewrites a non-``.html`` suffix, while 0.48's native export honours the name it was given — so
        normalising up front is what makes the two agree and keeps the returned path the file that exists.

        Args:
            path: Destination file. A suffix other than ``.html`` (including ``.HTML``) is replaced with
                ``.html``; a name with no suffix gains one. Replacement is :meth:`pathlib.Path.with_suffix`,
                matching what ``trame-pyvista`` does, so a dotted stem loses its last segment
                (``report.v2`` becomes ``report.html``). Note :meth:`save` dispatches only on a literal
                ``.html`` suffix, so ``save(\"x.htm\")`` raises on the unknown suffix while
                ``export_html(\"x.htm\")`` writes ``x.html``.

        Returns:
            The ``.html`` path written, as a string.

        Raises:
            ImportError: If the trame/vtk.js export stack is missing. pyvista raises this itself and its message
                names the package to install; the ``3d`` extra pulls the stack via ``pyvista[jupyter]``.
                Note pyvista's registry turns a *failing* plugin import into a ``UserWarning`` and drops the
                entry, so a broken-but-installed ``trame-pyvista`` reports that same message — when the
                package is present, read the warning for the real cause.
            RuntimeError: If trame and pyvista are bound to different VTK builds — the message names
                ``VTK_MODULE_NAME``, the variable that reconciles them.

        Examples:
            - Export a small scene. Asking for ``scene.HTML`` writes ``scene.html``, and the returned path is
              the normalised one — the page that exists, on either pyvista — so it can be passed straight on:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3DBase(off_screen=True)
                ...     _ = scene.add_mesh(pv.Sphere())
                ...     out = scene.export_html(os.path.join(folder, "scene.HTML"))
                ...     scene.close()
                ...     os.path.basename(out), os.path.getsize(out) > 0, sorted(os.listdir(folder))
                ('scene.html', True, ['scene.html'])

                ```

        See Also:
            digitalearth.three_d.Scene3D.orbit: a compact GIF/MP4 fly-through, the better choice for a heavy
                scene.
            save: routes here automatically for a ``*.html`` destination.
        """
        # pyvista >=0.49 moved trame support out into the separate `trame-pyvista` package: the export now lives
        # on a registered `trame` plotter component and `Plotter.export_html` is deprecated. This is a capability
        # switch, not a version one — 0.48 ships the same component registry, so a 0.48 user who installs
        # trame-pyvista takes the component branch too. Falling back (rather than raising here) hands the
        # not-installed case to pyvista's own actionable ImportError. The VTK-build check runs before either
        # branch: 0.49 makes it inside the deprecated Plotter.export_html we no longer call, and 0.48 makes
        # it nowhere at all, so doing it here is what guards both versions rather than neither.
        destination = str(Path(path).with_suffix(HTML_SUFFIX))
        _require_one_vtk_build()
        component = getattr(self.plotter, "trame", None)
        if component is None:
            self.plotter.export_html(destination)
        else:
            component.export_html(destination)
        return destination

    def save(self, path: Destination, **kwargs: Any) -> Path:
        """Save the scene, dispatching on ``path``'s suffix — a rendered frame, or a scene export.

        Two families share the one entry point, and the suffix decides which:

        - ``.html`` → :meth:`export_html`, a self-contained interactive page (see there for why a heavy scene
          is better served by :meth:`digitalearth.three_d.Scene3D.orbit`, and for the suffix normalisation it
          applies).
        - :data:`IMAGE_SUFFIXES` (``.png`` ``.jpg`` ``.jpeg`` ``.bmp`` ``.tif`` ``.tiff``) →
          :meth:`screenshot`, a rendered raster frame **in the format the suffix names** — ``.tif`` writes a
          TIFF, not a PNG.
        - :data:`SCENE_EXPORTERS` (``.gltf`` ``.obj`` ``.vrml`` ``.wrl`` ``.vtksz``) → the matching
          :class:`pyvista.Plotter` scene exporter, which writes the geometry itself rather than a picture of
          it.
        - anything else — **including a path with no suffix at all** — raises. A format is never invented on
          the caller's behalf, so ``save("scene")`` no longer quietly writes ``scene.png``.

        The suffix match is case-insensitive throughout (``SCENE.HTML`` exports a page).

        **It returns the path it wrote**, like every other tier's ``save()``, so a caller can chain on the
        result without rebuilding the filename — and ``.html`` returns the *normalised* name, which is not
        always the one passed in.

        **Changed, with no transition: it used to return the RGB frame** for a raster suffix, and ``None``
        for an export — a return type that depended on the suffix, and gave the one branch producing a
        useful value no counterpart on the others. Ask for the frame by name instead:
        :meth:`screenshot` writes the same file *and* hands back the array, so ``frame = scene.save(out)``
        becomes ``frame = scene.screenshot(out)`` — one call, the same two effects. Unlike the renamed
        keywords in this batch, which keep their old spelling for a release, a return *type* has no shim to
        offer: a value contrived to read as both a ``Path`` and an array would be worse than the break, so
        this one is announced rather than eased.

        Args:
            path: Output file, as a string or ``os.PathLike``. Its suffix selects the branch.
            **kwargs: Forwarded to the branch that runs — :meth:`screenshot` for a raster frame, the
                ``pyvista.Plotter`` exporter for a scene export (e.g. ``inline_data=False`` for ``.gltf``);
                ignored for HTML.

        Returns:
            pathlib.Path: the file that was written — ``path`` itself for every branch but ``.html``, which
            returns the suffix-normalised name :meth:`export_html` actually wrote.

        Raises:
            ValueError: when ``path`` has no suffix, or one outside the dispatch table; the message names both
                families' supported suffixes. Also when the installed pyvista lacks the exporter a supported
                scene suffix needs.

        Examples:
            - A raster suffix screenshots, and hands back the path it wrote — not the frame:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3DBase(off_screen=True, window_size=(200, 150))
                ...     _ = scene.add_mesh(pv.Sphere())
                ...     out = os.path.join(folder, "scene.png")
                ...     written = scene.save(out)
                ...     scene.close()
                ...     written.name, written.is_file()
                ('scene.png', True)

                ```
            - A scene-export suffix writes the geometry through PyVista's own exporter, and returns its path
              the same way:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3DBase(off_screen=True)
                ...     _ = scene.add_mesh(pv.Sphere())
                ...     written = scene.save(os.path.join(folder, "scene.gltf"))
                ...     scene.close()
                ...     written.suffix, written.stat().st_size > 0
                ('.gltf', True)

                ```
            - An ``.html`` suffix exports an interactive page instead. The match is case-insensitive, and
              :meth:`export_html` normalises the suffix it writes — so ``SCENE.HTML`` lands as ``SCENE.html``
              and the returned path is the normalised one, not the name that was passed:
                ```python
                >>> import os, tempfile, pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> with tempfile.TemporaryDirectory() as folder:
                ...     scene = Scene3DBase(off_screen=True)
                ...     _ = scene.add_mesh(pv.Sphere())
                ...     written = scene.save(os.path.join(folder, "SCENE.HTML"))
                ...     scene.close()
                ...     written.name, sorted(os.listdir(folder))
                ('SCENE.html', ['SCENE.html'])

                ```
            - A suffix-less destination is an error, not a silent ``.png``:
                ```python
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> try:
                ...     scene.save("scene")
                ... except ValueError as error:
                ...     print(str(error).split(";")[0])
                ... finally:
                ...     scene.close()
                save() cannot infer a format from 'scene': it has no suffix

                ```

        See Also:
            screenshot: the raster-frame branch, where its ``**kwargs`` end up, and the only method that
                returns the rendered array.
            export_html: the interactive-page branch.
        """
        suffix = Path(str(path)).suffix.lower()
        if suffix == HTML_SUFFIX:
            # export_html normalises the suffix it writes, so report the name it actually produced rather
            # than the one that was asked for.
            return Path(self.export_html(path))
        if suffix in IMAGE_SUFFIXES:
            self.screenshot(path=path, **kwargs)
            return Path(str(path))
        exporter_name = SCENE_EXPORTERS.get(suffix)
        if exporter_name is not None:
            # Guarded with getattr rather than assumed: the exporter set is a pyvista capability, and a
            # version that lacks one must say so plainly instead of failing on a missing attribute.
            exporter = getattr(self.plotter, exporter_name, None)
            if exporter is None:
                import pyvista as pv

                raise ValueError(
                    f"the installed pyvista ({pv.__version__}) has no Plotter.{exporter_name}, so {suffix!r} "
                    f"cannot be exported; {supported_destinations()}"
                )
            exporter(str(path), **kwargs)
            return Path(str(path))
        reason = (
            "it has no suffix"
            if not suffix
            else f"{suffix!r} is not a format it can write"
        )
        raise ValueError(
            f"save() cannot infer a format from {str(path)!r}: {reason}; {supported_destinations()}"
        )

    def show(self, **kwargs: Any) -> Any:
        """Display the scene interactively (or render a frame off-screen).

        Args:
            **kwargs: Forwarded to :meth:`pyvista.Plotter.show`.

        Returns:
            Whatever :meth:`pyvista.Plotter.show` returns.

        Examples:
            - Off-screen (as in CI, or under :data:`pyvista.OFF_SCREEN`) it renders a frame and returns without
              opening a window. What comes back is pyvista's own return value, which its ``return_cpos``
              theme setting decides — do not rely on it:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_mesh(pv.Sphere())
                >>> _ = scene.show()
                >>> scene.close()

                ```
            - Keyword arguments reach :meth:`pyvista.Plotter.show`, so the camera can be framed on the way in:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_mesh(pv.Cube())
                >>> _ = scene.show(cpos="xy")
                >>> scene.close()

                ```

        See Also:
            screenshot: returns the rendered frame as an array instead of displaying it.
        """
        return self.plotter.show(**kwargs)

    def close(self) -> None:
        """Close the wrapped plotter and free its render window.

        Closing twice is harmless, so a scene can be closed explicitly inside a ``with`` block that will close
        it again on exit.

        Examples:
            - Free the render window when the scene is finished with:
                ```python
                >>> import pyvista as pv
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> _ = scene.add_mesh(pv.Sphere())
                >>> scene.close()
                >>> len(scene.layers)
                1

                ```
            - A second close is a no-op, not an error:
                ```python
                >>> from digitalearth.three_d.base import Scene3DBase
                >>> scene = Scene3DBase(off_screen=True)
                >>> scene.close()
                >>> scene.close()

                ```

        See Also:
            __exit__: calls this on the way out of a ``with`` block.
        """
        # A scene that never drew has no render window to free, and must not open one in order to close it.
        if self._plotter is not None:
            self._plotter.close()

    def __enter__(self) -> Self:
        """Enter the runtime context, returning the scene.

        Returns:
            The same scene instance, so ``with Scene3D(...) as scene:`` binds this object (and
            :meth:`__exit__` closes its plotter).
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close the plotter on exit (whether or not the body raised); exceptions propagate.

        Args:
            exc_type: Exception class raised in the block, or ``None``.
            exc: The exception instance, or ``None``.
            tb: The traceback, or ``None``.

        Returns:
            ``False`` — exceptions are not suppressed.
        """
        self.close()
        return False
