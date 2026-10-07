"""batch — render a whole series of inputs to image files in one call (RP.11).

``Batch`` is a thin driver around the one-call :func:`~digitalearth.api.quickmap`: give it an iterable of
inputs (raster/vector paths or already-loaded pyramids objects), a set of shared plot options, and an output
directory, and it renders + saves one scene per input, closing each scene so a long run stays
memory-bounded. It is the operational counterpart of earthkit-plots' ``Batch``/``workflows`` — pure
orchestration over the existing visualization API (no new GIS or matplotlib machinery).
"""

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from pyramids.dataset import Dataset

from digitalearth.api import quickmap
from digitalearth.static import Map

# The interactive/web/3-D extras need not be installed to type-check this driver.
if TYPE_CHECKING:
    from digitalearth.interactive import InteractiveMap
    from digitalearth.three_d import Scene3D
    from digitalearth.web import WebMap

#: What ``quickmap`` returns, and therefore what a :class:`Batch`'s plotter produces: the static
#: :class:`~digitalearth.static.map.Map` by default, or one of the other three tiers' scenes
#: (:class:`~digitalearth.interactive.map.InteractiveMap`, :class:`~digitalearth.three_d.scene3d.Scene3D`,
#: :class:`~digitalearth.web.map.WebMap`) when a ``backend=`` is passed through the shared defaults (TD-22).
#: Named here so the driver's signatures state the whole set it can drive, not just the one backend it began
#: with. Every member exposes the ``save(path)`` and ``close()`` :class:`Batch` relies on.
BackendScene = Union[Map, "InteractiveMap", "Scene3D", "WebMap"]

__all__ = ["Batch", "load_input"]

logger = logging.getLogger(__name__)


def load_input(path: Any) -> Any:
    """Load a raster path as a pyramids ``Dataset``, falling back to a vector ``FeatureCollection``.

    The single loader behind both CLI subcommands: ``digitalearth plot`` and ``digitalearth batch`` open a
    path the same way, so a file one accepts is never rejected by the other. It lives here (rather than in
    :mod:`digitalearth.ops.cli`) because :meth:`Batch.render_one` is the lower layer — the CLI re-exports it.

    Args:
        path: A filesystem path (``str``/``Path``) to a raster or vector file.

    Returns:
        The pyramids ``Dataset`` (raster) or ``FeatureCollection`` (vector) the path opened as.

    Raises:
        Exception: the vector read's error, **chained** from the raster read's error (``raise ... from``),
            when the path is neither — so neither cause is hidden behind the other.

    Examples:
        - A vector path opens as a ``FeatureCollection``, not a raster:
            ```python
            >>> from digitalearth.ops.batch import load_input
            >>> type(load_input("tests/data/points.geojson")).__name__
            'FeatureCollection'

            ```
    """
    try:
        return Dataset.read_file(str(path))
    except (
        Exception
    ) as raster_error:  # not a raster pyramids can open — try it as vector
        from pyramids.feature import FeatureCollection

        try:
            return FeatureCollection.read_file(str(path))
        except Exception as vector_error:
            raise vector_error from raster_error


def _default_namer(item: Any, index: int) -> str:
    """Name an output by the input's file stem, or ``figure_<index>`` when the input is in-memory."""
    if isinstance(item, (str, Path)):
        return Path(item).stem
    return f"figure_{index:03d}"


class Batch:
    """Render many inputs to image files with one shared configuration.

    Args:
        plotter: The callable that turns one input into a backend scene — a
            :class:`~digitalearth.static.map.Map`, or one of the other three tiers' scenes when a ``backend=``
            default is set (see :data:`BackendScene`). Defaults to :func:`~digitalearth.api.quickmap`; any
            callable with the same ``(data, **kwargs) -> BackendScene`` contract works.
        ext: Image extension/format for saved figures (e.g. ``"png"``, ``"pdf"``).
        **defaults: Plot options applied to every input (e.g. ``crs``, ``kind``, ``cmap``, ``colorbar``);
            per-run ``overrides`` passed to :meth:`run` take precedence.

    Attributes:
        plotter: The configured plotting callable.
        ext: The output image extension.
        defaults: The shared plot options.

    Examples:
        - Configure a batch and read back its shared options:
            ```python
            >>> from digitalearth.ops.batch import Batch
            >>> b = Batch(crs=3857, kind="contourf", ext="png")
            >>> b.ext
            'png'
            >>> b.defaults["kind"]
            'contourf'

            ```
    """

    def __init__(
        self,
        plotter: Callable[..., "BackendScene"] = quickmap,
        *,
        ext: str = "png",
        **defaults: Any,
    ) -> None:
        """Store the plotting callable, output format, and shared plot options."""
        self.plotter = plotter
        self.ext = ext.lstrip(".")
        self.defaults = defaults

    def render_one(self, item: Any, **overrides: Any) -> "BackendScene":
        """Render a single input to a backend scene (without saving).

        Args:
            item: A raster/vector path (``str``/``Path``, opened by :func:`load_input`, which reads a
                raster via ``Dataset.read_file`` and falls back to ``FeatureCollection.read_file``) or an
                already-loaded pyramids object (``Dataset``/``FeatureCollection``), passed through as-is.
            **overrides: Plot options for this item, merged over (and overriding) the batch ``defaults``.

        Returns:
            The finished backend scene (see :data:`BackendScene`) — a
            :class:`~digitalearth.static.map.Map` by default, or the tier's own scene when the plotter was
            given a ``backend=`` default.

        Examples:
            - Render the bundled sample raster in its own CRS:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.ops.batch import Batch
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> m = Batch(colorbar=False).render_one(ds, crs=ds.epsg)
                >>> len(m.layers)
                1

                ```
        """
        data = load_input(item) if isinstance(item, (str, Path)) else item
        return self.plotter(data, **{**self.defaults, **overrides})

    def run(
        self,
        items: Iterable[Any],
        outdir: Any,
        *,
        namer: Callable[[Any, int], str] | None = None,
        **overrides: Any,
    ) -> list[Path]:
        """Render every input and save one image per input into ``outdir``.

        The output directory is created if needed. Each scene is closed immediately after saving (through its
        own ``close()``, whatever backend it is) so a long batch does not accumulate open figures or scenes.

        Args:
            items: Iterable of inputs (paths or pyramids objects).
            outdir: Directory to write images into (created if missing).
            namer: ``(item, index) -> stem`` naming each output file (no extension). Defaults to the input's
                file stem, or ``figure_<index>`` for in-memory inputs. Colliding stems (e.g. same file name
                from different directories) are disambiguated with the input index so no earlier image is
                silently overwritten.
            **overrides: Plot options merged over the batch ``defaults`` for this whole run.

        Returns:
            The list of written image paths, in input order.

        Examples:
            - Render a one-item batch and confirm the file was written:
                ```python
                >>> import matplotlib, tempfile, os
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.ops.batch import Batch
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> out = tempfile.mkdtemp()
                >>> paths = Batch(crs=ds.epsg, colorbar=False).run([ds], out, namer=lambda item, i: "acc")
                >>> [p.name for p in paths]
                ['acc.png']
                >>> os.path.getsize(paths[0]) > 0
                True

                ```
        """
        namer = namer or _default_namer
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        used: set[str] = set()
        for index, item in enumerate(items):
            scene = self.render_one(item, **overrides)
            stem = namer(item, index)
            if (
                stem in used
            ):  # disambiguate a colliding name so an earlier image is not overwritten
                logger.warning(
                    "batch output name %r already used; disambiguating with index %d",
                    stem,
                    index,
                )
                stem = f"{stem}_{index}"
            used.add(stem)
            path = out / f"{stem}.{self.ext}"
            scene.save(str(path))
            # Free the scene through its own ``close()`` — every backend has one, only the matplotlib tier
            # has a ``.fig`` — so a web/interactive/3-D batch is memory-bounded the same way without being
            # crashed on a figure it does not carry.
            scene.close()
            written.append(path)
        return written
