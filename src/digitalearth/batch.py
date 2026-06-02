"""batch — render a whole series of inputs to image files in one call (RP.11).

``Batch`` is a thin driver around the one-call :func:`~digitalearth.api.quickmap`: give it an iterable of
inputs (raster/vector paths or already-loaded pyramids objects), a set of shared plot options, and an output
directory, and it renders + saves one figure per input, closing each figure so a long run stays
memory-bounded. It is the operational counterpart of earthkit-plots' ``Batch``/``workflows`` — pure
orchestration over the existing visualization API (no new GIS or matplotlib machinery).
"""
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import matplotlib.pyplot as plt
from pyramids.dataset import Dataset

from digitalearth.api import quickmap
from digitalearth.scene import Map

__all__ = ["Batch"]


def _default_namer(item: Any, index: int) -> str:
    """Name an output by the input's file stem, or ``figure_<index>`` when the input is in-memory."""
    if isinstance(item, (str, Path)):
        return Path(item).stem
    return f"figure_{index:03d}"


class Batch:
    """Render many inputs to image files with one shared configuration.

    Args:
        plotter: The callable that turns one input into a :class:`~digitalearth.scene.map.Map`. Defaults to
            :func:`~digitalearth.api.quickmap`; any callable with the same ``(data, **kwargs) -> Map``
            contract works.
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
            >>> from digitalearth.batch import Batch
            >>> b = Batch(crs=3857, kind="contourf", ext="png")
            >>> b.ext
            'png'
            >>> b.defaults["kind"]
            'contourf'

            ```
    """

    def __init__(self, plotter: Callable[..., Map] = quickmap, *, ext: str = "png", **defaults: Any) -> None:
        """Store the plotting callable, output format, and shared plot options."""
        self.plotter = plotter
        self.ext = ext.lstrip(".")
        self.defaults = defaults

    def render_one(self, item: Any, **overrides: Any) -> Map:
        """Render a single input to a :class:`Map` (without saving).

        Args:
            item: A raster/vector path (``str``/``Path``, read via pyramids ``Dataset.read_file``) or an
                already-loaded pyramids object (``Dataset``/``FeatureCollection``), passed through as-is.
            **overrides: Plot options for this item, merged over (and overriding) the batch ``defaults``.

        Returns:
            The finished :class:`~digitalearth.scene.map.Map`.

        Examples:
            - Render the bundled sample raster in its own CRS:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.batch import Batch
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> m = Batch(colorbar=False).render_one(ds, crs=ds.epsg)
                >>> len(m.layers)
                1

                ```
        """
        data = Dataset.read_file(str(item)) if isinstance(item, (str, Path)) else item
        return self.plotter(data, **{**self.defaults, **overrides})

    def run(
        self,
        items: Iterable[Any],
        outdir: Any,
        *,
        namer: Optional[Callable[[Any, int], str]] = None,
        **overrides: Any,
    ) -> List[Path]:
        """Render every input and save one image per input into ``outdir``.

        The output directory is created if needed. Each figure is closed immediately after saving so a long
        batch does not accumulate open figures.

        Args:
            items: Iterable of inputs (paths or pyramids objects).
            outdir: Directory to write images into (created if missing).
            namer: ``(item, index) -> stem`` naming each output file (no extension). Defaults to the input's
                file stem, or ``figure_<index>`` for in-memory inputs.
            **overrides: Plot options merged over the batch ``defaults`` for this whole run.

        Returns:
            The list of written image paths, in input order.

        Examples:
            - Render a one-item batch and confirm the file was written:
                ```python
                >>> import matplotlib, tempfile, os
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.batch import Batch
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
        written: List[Path] = []
        for index, item in enumerate(items):
            scene = self.render_one(item, **overrides)
            path = out / f"{namer(item, index)}.{self.ext}"
            scene.save(str(path))
            plt.close(scene.fig)
            written.append(path)
        return written
