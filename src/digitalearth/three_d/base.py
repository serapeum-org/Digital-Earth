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
from dataclasses import replace as with_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Self, Union

import numpy as np

from digitalearth.base.crs import OffLimbError, declared_crs, reproject
from digitalearth.base.custom import custom_kind
from digitalearth.base.display import auto_cmap, needs_reproject
from digitalearth.base.sources import Source
from digitalearth.base.spec import (
    Camera,
    DataRef,
    FigureSpec,
    LayerSpec,
    PanelSpec,
    Scale,
    Selection,
    Symbology,
)
from digitalearth.three_d.layer import next_layer_id, stored_props
from digitalearth.three_d.renderer import Renderer3D

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


#: The id of the one panel a 3-D scene draws into. A scene is a single view of a single set of layers, so the
#: figure it writes has exactly one panel; the name is fixed so a stored figure reads the same way every time.
PANEL_ID: str = "scene"

#: Where a scene looks from before anything has been drawn or asked for — the isometric view a reader expects
#: of a 3-D figure. PyVista fits its own camera to the first mesh, so this is what the description says until
#: either the scene has rendered or a caller sets a camera.
DEFAULT_CAMERA: Camera = Camera((1.0, -1.0, 1.0))


def _vector3(values: Any) -> tuple[float, float, float]:
    """Return VTK's three coordinates as the triple a `Camera` field holds.

    Args:
        values: A position, focal point or up vector, as PyVista hands it out.

    Returns:
        The three values as floats.
    """
    x, y, z = (float(value) for value in values)
    return x, y, z


def _with_panel_layers(figure: FigureSpec) -> FigureSpec:
    """Return `figure` with its panel listing exactly the layers in its tree.

    Args:
        figure: The figure to correct.

    Returns:
        The figure, with the panel's `layers` in tree order. A scene has one panel showing everything, so the
        panel follows the tree rather than being maintained beside it.
    """
    panel = with_fields(figure.panels[0], layers=tuple(figure.layers.ids))
    return with_fields(figure, panels=(panel,))


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
        - A scene given a display CRS reports it before any layer is added, and builds no plotter to do so:
            ```python
            >>> from digitalearth.three_d.base import Scene3DBase
            >>> scene = Scene3DBase(off_screen=True, crs=4326)
            >>> scene.display_crs, scene.layers
            (4326, [])
            >>> scene.close()

            ```
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
        #: What the scene draws, as data. Every builder adds a layer to it; `figure_spec` hands it out with the
        #: live camera on its panel.
        self._figure: FigureSpec = FigureSpec(
            panels=(PanelSpec(PANEL_ID, DEFAULT_CAMERA),)
        )
        #: Draws the figure onto the plotter, and holds the `(mesh, actor)` pair of every drawn layer.
        self._renderer: Renderer3D = Renderer3D(self)
        #: The objects a caller handed to `add_mesh`/`add_volume`, keyed by layer id (#293).
        self._custom: dict[str, Any] = {}
        #: The camera a caller set, and whether they set one. Applied to the plotter before each render, since
        #: PyVista resets its own camera to fit the first mesh added.
        self._camera: Camera = DEFAULT_CAMERA
        self._camera_set: bool = False
        #: The vertical view scale, kept here so it survives a plotter that has not been built yet and so it
        #: can be written onto the figure's camera.
        self._vertical: float = 1.0
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
            # A view scale or a camera set before anything was drawn belongs to the scene, not to whichever
            # plotter happens to exist: both are applied to the new window rather than quietly lost.
            if self._vertical != 1.0:
                self._plotter.set_scale(zscale=self._vertical, render=False)
            self._apply_camera()
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
        if self._plotter is None:
            # Readable before anything is drawn: the value belongs to the view (`Camera.vertical_exaggeration`),
            # and asking for it should not build a render window (#290).
            return self._vertical
        return float(self._plotter.renderer.scale[2])

    @vertical_exaggeration.setter
    def vertical_exaggeration(self, factor: float) -> None:
        """Set the scene's vertical (z) view scale.

        Args:
            factor: The z scale to render at (``1.0`` = true scale, ``>1`` accentuates relief).
        """
        self._vertical = float(factor)
        if self._plotter is not None:
            self._plotter.set_scale(zscale=float(factor), render=False)

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

    def _add_custom(
        self, obj: Any, *, name: Any, band: Any, volume: bool, **kwargs: Any
    ) -> Any:
        """Register an object the caller built, and draw it.

        Args:
            obj: The PyVista mesh or volume.
            name: The id to address it by; `None` numbers it.
            band: Where it is drawn (#292); `None` draws it among the data.
            volume: Whether it is ray-cast (`add_volume`) rather than surfaced (`add_mesh`).
            **kwargs: Forwarded to PyVista.

        Returns:
            The actor PyVista produced.
        """
        layer_id = next_layer_id(self._figure.layers, custom_kind("pyvista"), name)
        self._custom[layer_id] = obj
        actor = self._add_described_layer(
            kind=custom_kind("pyvista"),
            name=layer_id,
            band=band,
            volume=volume,
            **kwargs,
        )
        if (
            actor is None
        ):  # pragma: no cover - the object was stored a line above, so it is there
            self._custom.pop(layer_id, None)
        return actor

    def add_mesh(
        self, mesh: Any, *, name: Any = None, band: Any = None, **kwargs: Any
    ) -> Any:
        """Add a PyVista ``mesh`` to the scene and register it as a layer.

        The low-level entry point the capability mixins build on. ``kwargs`` pass straight to
        :meth:`pyvista.Plotter.add_mesh` (``scalars``, ``cmap``, ``opacity``, ``show_edges``, ``pbr`` …).

        An object you build yourself is a **custom layer**: what a figure keeps is its description — an id, the
        kind `custom:pyvista`, a label, a band and whether it is visible — and never the object, which has no
        description to write. The object stays with the scene that was handed it, so a figure saved and loaded
        again names the layer but cannot rebuild it, and another backend cannot draw it at all
        (:mod:`digitalearth.base.custom` says which case a reader is in). The tier records custom layers in its
        layer tree as its seam lands (#295); until then the object is drawn and nothing else is kept.

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
        return self._add_custom(mesh, name=name, band=band, volume=False, **kwargs)

    def add_volume(
        self, volume: Any, *, name: Any = None, band: Any = None, **kwargs: Any
    ) -> Any:
        """Add a volumetric ``volume`` (ray-cast rendering) and register it as a layer.

        An object you build yourself is a **custom layer**: what a figure keeps is its description — an id, the
        kind `custom:pyvista`, a label, a band and whether it is visible — and never the object, which has no
        description to write. The object stays with the scene that was handed it, so a figure saved and loaded
        again names the layer but cannot rebuild it, and another backend cannot draw it at all
        (:mod:`digitalearth.base.custom` says which case a reader is in). The tier records custom layers in its
        layer tree as its seam lands (#295); until then the object is drawn and nothing else is kept.

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
        return self._add_custom(volume, name=name, band=band, volume=True, **kwargs)

    @property
    def figure_spec(self) -> FigureSpec:
        """What the scene draws, as data: its layers, their sources, and the camera it is seen from.

        Returns:
            A :class:`~digitalearth.base.spec.FigureSpec` with one panel, `"scene"`, whose view is the scene's
            current :attr:`camera` and whose layers are the ids in draw order. Every builder writes into it, so
            it is the scene's description rather than a report about it: written with `to_dict()`, read back
            with `from_dict()` and drawn again with :meth:`from_figure`, it renders the same picture.

            A scene built from data already in memory can be handed straight to :meth:`from_figure`, but not
            written: `to_dict()` refuses an `object:` source, because a reference into this process's memory
            would be unreadable everywhere else. Give a builder a path to get a figure that can be stored.

        Examples:
            - A built scene describes itself, and the description names what was drawn:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
                >>> layer = scene.figure_spec.layers.get("terrain-1")
                >>> layer.kind, layer.symbology.props["scalars"]
                ('terrain', 'elevation')
                >>> scene.close()

                ```
        """
        panel = with_fields(
            self._figure.panels[0],
            view=self.camera,
            layers=tuple(self._figure.layers.ids),
        )
        return with_fields(self._figure, panels=(panel,))

    @property
    def renderer(self) -> Renderer3D:
        """The renderer that draws this scene's figure, and holds what it drew.

        Returns:
            The :class:`~digitalearth.three_d.renderer.Renderer3D`. Its `drawn` mapping is how a layer id
            reaches the mesh and actor behind it.
        """
        return self._renderer

    @property
    def held_objects(self) -> dict[str, Any]:
        """The engine objects a caller handed to :meth:`add_mesh`/:meth:`add_volume`, keyed by layer id.

        Returns:
            The table the renderer looks a `custom:pyvista` layer up in. A figure describes such a layer but
            never carries the object (:mod:`digitalearth.base.custom`), so this is where it lives.
        """
        return self._custom

    @property
    def layers(self) -> list[tuple[Any, Any]]:
        """The drawn `(mesh, actor)` pairs, in draw order — the view this tier has always exposed.

        Returns:
            One pair per layer that was actually drawn, ordered as :attr:`layer_ids` is. A described layer that
            was skipped (an empty raster, a missing custom object) has no pair.

        Note:
            This is a **compatibility view** of what the renderer holds, not the scene's state: the scene is
            described by :attr:`figure_spec`, and layers are addressed by id through :attr:`layer_ids`,
            :meth:`mesh_of`, :meth:`actor_of`, :meth:`remove_layer` and :meth:`set_visible`. It is kept —
            without a deprecation warning — because it is what `animate`'s callback and a good deal of user
            code reach for; the rename that retires it belongs with the tier method names (#299), so callers
            get one migration rather than two.
        """
        drawn = self._renderer.drawn
        return [
            drawn[layer_id] for layer_id in self._figure.layers.ids if layer_id in drawn
        ]

    @property
    def layer_ids(self) -> list[str]:
        """The ids of the scene's layers, in draw order.

        Returns:
            The ids, bottom first. A builder that was given `name=` used it; the others were numbered by kind.

        Examples:
            - Two layers, each addressable by the name the scene gave it:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> dem = get_source(np.add.outer(np.arange(4.0), np.arange(5.0)))
                >>> _ = scene.terrain(dem)
                >>> _ = scene.point_cloud(np.array([[0.0, 0.0, 9.0]]), name="peak")
                >>> scene.layer_ids
                ['terrain-1', 'peak']
                >>> scene.close()

                ```
        """
        return list(self._figure.layers.ids)

    def mesh_of(self, layer_id: str) -> Any:
        """Return the mesh drawn for a layer.

        Args:
            layer_id: The layer to look up.

        Returns:
            The PyVista mesh, or `None` when the layer was described but not drawn.

        Raises:
            KeyError: if no layer has that id, naming the ids that do.

        Examples:
            - The mesh a builder produced, reached by the layer's id rather than by position:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
                >>> scene.mesh_of("terrain-1").n_points
                20
                >>> scene.close()

                ```
        """
        return self._drawn_pair(layer_id, "mesh")

    def actor_of(self, layer_id: str) -> Any:
        """Return the actor drawn for a layer.

        Args:
            layer_id: The layer to look up.

        Returns:
            The PyVista actor, or `None` when the layer was described but not drawn.

        Raises:
            KeyError: if no layer has that id, naming the ids that do.
        """
        return self._drawn_pair(layer_id, "actor")

    def _drawn_pair(self, layer_id: str, part: str) -> Any:
        """Return one half of what was drawn for a layer.

        Args:
            layer_id: The layer to look up.
            part: `"mesh"` or `"actor"`.

        Returns:
            The mesh or the actor, or `None` when nothing was drawn for the layer.

        Raises:
            KeyError: if no layer has that id.
        """
        if layer_id not in self._figure.layers:
            raise KeyError(
                f"no layer {layer_id!r} in this scene; its layers are {self.layer_ids}"
            )
        drawn = self._renderer.drawn.get(layer_id)
        if drawn is None:
            return None
        return drawn[0] if part == "mesh" else drawn[1]

    def get_layer(self, layer_id: str) -> LayerSpec:
        """Return the description of one layer, by id.

        Args:
            layer_id: The layer to look up.

        Returns:
            Its :class:`~digitalearth.base.spec.LayerSpec` — kind, source, symbology, band and visibility.

        Raises:
            KeyError: if no layer has that id, naming the ids that do.

        Examples:
            - What a builder recorded, read back by id:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
                >>> scene.get_layer("terrain-1").kind
                'terrain'
                >>> scene.close()

                ```
        """
        if layer_id not in self._figure.layers:
            raise KeyError(
                f"no layer {layer_id!r} in this scene; its layers are {self.layer_ids}"
            )
        return self._figure.layers.get(layer_id)

    def replace_layer(self, layer: LayerSpec) -> Self:
        """Swap a layer's description for another, keeping its id and its place in draw order.

        This is how a layer is restyled or re-pointed after it has been added: the renderer draws it again
        from the new description, since VTK bakes a colormap into the mesh it built.

        Args:
            layer: The new description. Its id names the layer it replaces.

        Returns:
            This scene (chainable).

        Raises:
            KeyError: if no layer has that id.
            ValueError: if `layer` is not a `LayerSpec`.

        Examples:
            - A layer redrawn in another colormap keeps its id:
                ```python
                >>> import numpy as np
                >>> from dataclasses import replace
                >>> from digitalearth.base.spec import Symbology
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
                >>> held = scene.get_layer("terrain-1")
                >>> _ = scene.replace_layer(replace(held, symbology=Symbology(props={"cmap": "magma"})))
                >>> scene.get_layer("terrain-1").symbology.props["cmap"]
                'magma'
                >>> scene.close()

                ```
        """
        if layer.id not in self._figure.layers:
            raise KeyError(
                f"no layer {layer.id!r} in this scene; its layers are {self.layer_ids}"
            )
        self._change(self._figure_with(layers=self._figure.layers.replace(layer)))
        return self

    def render(self) -> Any:
        """Build and return the renderer's own object — this tier's plotter.

        Every tier answers to `render()` with the thing its engine draws on, so a caller who wants to reach
        past the facade has one name to learn. Here that is the `pyvista.Plotter`, built on first use (#290).

        Returns:
            The scene's plotter, with every layer drawn on it.

        Examples:
            - The plotter is what a 3-D scene renders to:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> scene.render() is scene.plotter
                True
                >>> scene.close()

                ```
        """
        self._apply_camera()
        return self.plotter

    def remove_layer(self, layer_id: str) -> Self:
        """Take a layer off the scene, by id.

        Args:
            layer_id: The layer to remove.

        Returns:
            This scene (chainable).

        Raises:
            KeyError: if no layer has that id, naming the ids that do.

        Examples:
            - A layer added and then removed leaves nothing behind:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
                >>> scene.remove_layer("terrain-1").layer_ids
                []
                >>> len(scene.layers)
                0
                >>> scene.close()

                ```
        """
        if layer_id not in self._figure.layers:
            raise KeyError(
                f"no layer {layer_id!r} in this scene; its layers are {self.layer_ids}"
            )
        sources = {
            key: ref
            for key, ref in self._figure.sources.items()
            if key != self._figure.layers.get(layer_id).source_id
        }
        self._change(
            self._figure_with(
                layers=self._figure.layers.remove(layer_id), sources=sources
            )
        )
        self._custom.pop(layer_id, None)
        return self

    def set_visible(self, layer_id: str, visible: bool = True) -> Self:
        """Show or hide a layer, by id.

        The layer stays in the scene and in its description — which is what lets a viewer switch it back on,
        and what tells a saved figure that the layer is there but hidden.

        Args:
            layer_id: The layer to toggle.
            visible: Whether it is drawn.

        Returns:
            This scene (chainable).

        Raises:
            KeyError: if no layer has that id, naming the ids that do.

        Examples:
            - A hidden layer is still described, and comes back on:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
                >>> hidden = scene.set_visible("terrain-1", False)
                >>> hidden.figure_spec.layers.is_visible("terrain-1")
                False
                >>> scene.set_visible("terrain-1").figure_spec.layers.is_visible("terrain-1")
                True
                >>> scene.close()

                ```
        """
        if layer_id not in self._figure.layers:
            raise KeyError(
                f"no layer {layer_id!r} in this scene; its layers are {self.layer_ids}"
            )
        self._change(
            self._figure_with(layers=self._figure.layers.set_visible(layer_id, visible))
        )
        return self

    def move_layer(self, layer_id: str, index: int) -> Self:
        """Move a layer in draw order, by id.

        Args:
            layer_id: The layer to move.
            index: Its position afterwards, counted as a list index is — `0` is the bottom, `-1` the top.

        Returns:
            This scene (chainable).

        Raises:
            KeyError: if no layer has that id.
            IndexError: if the position is outside the scene, or outside the layer's band (#292).

        Note:
            VTK composites its actors by depth, so this changes the description rather than the picture. It is
            recorded because the figure is drawn by the other tiers too, where order decides what is on top.

        Examples:
            - Two layers, reordered by id:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> dem = get_source(np.add.outer(np.arange(4.0), np.arange(5.0)))
                >>> _ = scene.terrain(dem, name="under")
                >>> _ = scene.terrain(dem, name="over")
                >>> scene.move_layer("over", 0).layer_ids
                ['over', 'under']
                >>> scene.close()

                ```
        """
        self._change(
            self._figure_with(layers=self._figure.layers.move(layer_id, index))
        )
        return self

    def draw_figure(self, figure: FigureSpec) -> Self:
        """Draw a figure into this scene, bringing it from whatever it showed before.

        Args:
            figure: The figure to draw. Its one panel's camera becomes the scene's.

        Returns:
            This scene (chainable).

        Raises:
            KeyError: when a layer names a kind this tier does not draw.
            OffLimbError: when a layer cannot be drawn and the scene is `strict`.
        """
        view = figure.panels[0].view
        if isinstance(view, Camera):
            self.camera = view
        self._change(figure)
        return self

    @classmethod
    def from_figure(cls, figure: FigureSpec, **scene_kwargs: Any) -> Self:
        """Build a scene and draw a figure into it.

        This is the round trip the seam exists for: a scene describes itself with :attr:`figure_spec`, the
        description survives `to_dict()`/`from_dict()`, and this draws it again.

        Args:
            figure: The figure to draw.
            **scene_kwargs: Passed to the constructor — `off_screen`, `window_size`, `strict`, …. The display
                CRS comes from the figure's camera unless `crs=` is given here.

        Returns:
            The scene, with every layer drawn.

        Examples:
            - A figure written by one scene is drawn by another:
                ```python
                >>> from digitalearth.base.spec import FigureSpec
                >>> from digitalearth.three_d import Scene3D
                >>> first = Scene3D(off_screen=True)
                >>> _ = first.terrain("examples/data/acc4000.tif")
                >>> stored = first.figure_spec.to_dict()
                >>> first.close()
                >>> second = Scene3D.from_figure(FigureSpec.from_dict(stored), off_screen=True)
                >>> second.layer_ids
                ['terrain-1']
                >>> second.close()

                ```
        """
        view = figure.panels[0].view
        crs = getattr(view, "crs", None)
        scene = cls(**{"crs": crs, **scene_kwargs})
        scene.draw_figure(figure)
        return scene

    def _figure_with(self, *, layers: Any = None, sources: Any = None) -> FigureSpec:
        """Return the scene's figure with other layers or sources, and its panel kept in step.

        The panel lists the layers it shows, so it has to change in the same construction: a figure built with
        new layers and an old panel names a layer that is not there, and `FigureSpec` refuses it — correctly.

        Args:
            layers: The new `LayerTree`, or `None` to keep the current one.
            sources: The new sources, or `None` to keep the current ones.

        Returns:
            The new figure.
        """
        tree = self._figure.layers if layers is None else layers
        panel = with_fields(self._figure.panels[0], layers=tuple(tree.ids))
        return with_fields(
            self._figure,
            panels=(panel,),
            layers=tree,
            sources=self._figure.sources if sources is None else sources,
        )

    def _change(self, figure: FigureSpec) -> None:
        """Move the scene to another figure, applying the difference to the plotter.

        Args:
            figure: The figure the scene should show.

        Raises:
            KeyError: when a layer names a kind this tier does not draw.
            OffLimbError: when a layer cannot be drawn and the scene is `strict`.
        """
        before = self._figure
        self._figure = _with_panel_layers(figure)
        self._renderer.apply(before, self._figure)

    def _add_described_layer(
        self,
        *,
        kind: str,
        data: Any = None,
        name: Any = None,
        band: Any = None,
        selection: Any = None,
        label: Any = None,
        **props: Any,
    ) -> Any:
        """Describe a layer, add it to the scene's figure, and draw it.

        Every builder ends here. It is where a layer gets its id, its source and the keywords the renderer
        redraws it with — the description that used to be lost the moment PyVista had the mesh.

        Args:
            kind: The registered layer kind (#288) — `"terrain"`, `"point_cloud"`, `"extrusion"`, ….
            data: What the layer draws: a path or URL, a `DataRef`, or an object already in memory. `None` for
                a layer drawn from no data at all, such as the globe's coastlines. A path is stored as a
                reference to that path, so the figure can be written to disk and drawn again elsewhere; an
                object is stored as a reference that resolves in this process only.
            name: The caller's name for the layer; `None` numbers it by kind.
            band: Where it is drawn relative to the data (#292); `None` takes the kind's own band.
            selection: Which slice of the source it draws.
            label: What a layer switcher would call it.
            **props: The engine keywords, stored on the layer's symbology. An array among them is stored as a
                reference rather than copied (:mod:`digitalearth.three_d.layer`).

        Returns:
            The actor PyVista produced, or `None` when the layer had nothing to draw. A skipped layer is not
            added to the figure, so the description lists what is actually there.

        Raises:
            OffLimbError: when the layer has nothing to draw and the scene is `strict`.
        """
        layer_id = next_layer_id(self._figure.layers, kind, name)
        sources = dict(self._figure.sources)
        source_id = None
        if data is not None:
            source_id = layer_id
            sources[source_id] = DataRef.of(data, name=kind)
        spec = LayerSpec(
            layer_id,
            kind,
            source_id=source_id,
            band=band,
            label=label,
            selection=selection if selection is not None else Selection(),
            symbology=Symbology(props=stored_props(**props)),
        )
        candidate = self._figure_with(
            layers=self._figure.layers.add(spec), sources=sources
        )
        actor = self._renderer.draw_layer(candidate, layer_id)
        if actor is None:
            # A skipped layer is not kept: `layer_ids` lists what the scene actually draws, and a figure that
            # named a layer nothing was drawn for would draw nothing on the way back either.
            return None
        self._figure = candidate
        return actor

    @property
    def camera(self) -> Camera:
        """Where the scene is looked at from, as a :class:`~digitalearth.base.spec.Camera`.

        Reading it once the scene has rendered gives the live view — pan, zoom and orbit included — so a camera
        found interactively can be written down, stored with `to_dict()` and set again later. Before the
        plotter exists it is the camera that was set, or the scene's default isometric view.

        Returns:
            The camera, carrying the scene's vertical exaggeration and display CRS.

        Examples:
            - A camera set on a scene is the camera it reports:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> scene.camera = Camera.look_at((0.0, 0.0, 0.0), azimuth=225.0, elevation=30.0, distance=10.0)
                >>> [round(value, 3) for value in scene.camera.position]
                [-6.124, -6.124, 5.0]
                >>> scene.close()

                ```
        """
        if self._plotter is None:
            return with_fields(
                self._camera,
                vertical_exaggeration=self.vertical_exaggeration,
                crs=self.display_crs,
            )
        live = self._plotter.camera
        parallel = bool(live.parallel_projection)
        return Camera(
            position=_vector3(live.position),
            focal_point=_vector3(live.focal_point),
            view_up=_vector3(live.up),
            view_angle=float(live.view_angle),
            parallel=parallel,
            parallel_scale=float(live.parallel_scale) if parallel else None,
            vertical_exaggeration=self.vertical_exaggeration,
            crs=self.display_crs,
        )

    @camera.setter
    def camera(self, camera: Camera) -> None:
        """Look at the scene from `camera`.

        The camera is applied to the plotter immediately when there is one, and again just before each render:
        PyVista resets its camera to fit the first mesh added, which would otherwise undo a view set before the
        layers were.

        Args:
            camera: Where to look from.

        Raises:
            ValueError: if `camera` is not a `Camera` — a tuple of three points is PyVista's `camera_position`,
                which this deliberately does not accept, since it says nothing about projection or exaggeration.
        """
        if not isinstance(camera, Camera):
            raise ValueError(
                f"Scene3D.camera must be a Camera; got {type(camera).__name__}. Build one with "
                "Camera(position=...) or Camera.look_at(focal_point, azimuth=..., elevation=..., distance=...)"
            )
        self._camera = camera
        self._camera_set = True
        self.vertical_exaggeration = camera.vertical_exaggeration
        if self._plotter is not None:
            self._apply_camera()

    def _apply_camera(self) -> None:
        """Put the stored camera on the plotter, if one was ever set."""
        if not self._camera_set or self._plotter is None:
            return
        live = self._plotter.camera
        live.position = tuple(self._camera.position)
        live.focal_point = tuple(self._camera.focal_point)
        live.up = tuple(self._camera.view_up)
        live.view_angle = float(self._camera.view_angle)
        live.parallel_projection = bool(self._camera.parallel)
        if self._camera.parallel and self._camera.parallel_scale is not None:
            live.parallel_scale = float(self._camera.parallel_scale)

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
        self._apply_camera()
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
        self._apply_camera()
        return self.plotter.show(**kwargs)

    def close(self) -> None:
        """Close the wrapped plotter and free its render window.

        Closing twice is harmless, so a scene can be closed explicitly inside a ``with`` block that will close
        it again on exit. A scene that never drew has no render window, and closing it opens none.

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
