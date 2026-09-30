"""cli — the ``digitalearth`` command line: render rasters/vectors to images without writing Python (RP.11).

Two subcommands wrap the existing API so plots can be produced from a shell or a Makefile:

* ``digitalearth plot INPUT [-o OUT] [options]`` — render one input via :func:`~digitalearth.api.quickmap`.
* ``digitalearth batch INPUTS... -o OUTDIR [--html PAGE] [options]`` — render many via
  :class:`~digitalearth.ops.batch.Batch`, optionally collecting them into a static HTML gallery
  (:func:`~digitalearth.ops.browser.gallery`).

The CLI renders headless — the matplotlib backend to ``Agg``, and ``--backend`` picks any of the four
``quickmap`` dispatches (``matplotlib``/``interactive``/``3d``/``web``), whose scenes save to their own native
file: a PNG for matplotlib, an HTML page for the other three (TD-22). Human-facing progress goes to ``stderr``
so any piped ``stdout`` stays clean. This is earthkit-plots' ``cli/`` entry point, scoped to Digital-Earth's
API.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

import matplotlib

from digitalearth.api import quickmap
from digitalearth.ops.batch import Batch, load_input
from digitalearth.ops.browser import gallery

__all__ = ["build_parser", "main"]

#: The one raster-then-vector loader, owned by :mod:`digitalearth.ops.batch` so ``plot`` and ``batch`` cannot
#: drift apart: ``batch`` opens its inputs through :meth:`Batch.render_one`, which calls the same function.
#: Kept under the old private name because that is what this module's callers and tests import.
_load = load_input

#: The file format each backend writes natively, chosen when the caller names no extension (TD-22). The
#: matplotlib scene is a raster figure (``savefig`` writes a PNG); the other three are self-contained pages
#: their own ``save`` serialises as HTML. Every value is a suffix the matching tier's ``save`` accepts, so a
#: caller who overrides it (``-o out.svg``, ``--ext pdf``) is handled by that tier — which writes the format
#: or raises its own clear error for one it cannot.
_NATIVE_EXT: dict = {
    "matplotlib": "png",
    "interactive": "html",
    "3d": "html",
    "web": "html",
}


def _parse_crs(value: str) -> Any:
    """Parse a ``--crs`` argument as an EPSG int when all-digits, else a proj4/WKT string."""
    return int(value) if value.lstrip("-").isdigit() else value


def _add_plot_options(parser: argparse.ArgumentParser) -> None:
    """Attach the shared plotting flags (forwarded to ``quickmap``) to a subcommand parser."""
    parser.add_argument(
        "--backend",
        choices=tuple(_NATIVE_EXT),
        default="matplotlib",
        help="rendering backend quickmap dispatches to (default: matplotlib)",
    )
    parser.add_argument(
        "--crs",
        type=_parse_crs,
        default=3857,
        help="display CRS (EPSG int or proj4 string)",
    )
    parser.add_argument(
        "--kind",
        default="auto",
        help="raster renderer: auto|imshow|contourf|contour|pcolormesh",
    )
    parser.add_argument(
        "--cmap", default=None, help="colormap name (default: auto-styled)"
    )
    parser.add_argument(
        "--levels", type=int, default=None, help="number of contour levels"
    )
    parser.add_argument(
        "--domain", default=None, help="named region / domain to set the extent"
    )
    parser.add_argument(
        "--basemap",
        action="store_true",
        help="overlay an XYZ tile basemap (needs network)",
    )
    parser.add_argument(
        "--coastlines", action="store_true", help="overlay coastlines (needs network)"
    )
    parser.add_argument(
        "--no-colorbar", dest="colorbar", action="store_false", help="omit the colorbar"
    )
    parser.set_defaults(colorbar=True)


def _plot_kwargs(args: argparse.Namespace) -> dict:
    """Collect the ``quickmap`` keyword arguments set on ``args`` (omitting unset optional styling)."""
    kwargs: dict = {
        "backend": args.backend,
        "crs": args.crs,
        "kind": args.kind,
        "basemap": args.basemap,
        "coastlines": args.coastlines,
        "colorbar": args.colorbar,
    }
    if args.cmap is not None:
        kwargs["cmap"] = args.cmap
    if args.levels is not None:
        kwargs["levels"] = args.levels
    if args.domain is not None:
        kwargs["domain"] = args.domain
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    """Build the ``digitalearth`` argument parser (``plot`` and ``batch`` subcommands).

    Returns:
        The configured :class:`argparse.ArgumentParser`; each subcommand sets a ``func`` default used by
        :func:`main` to dispatch.

    Examples:
        - Parse a ``plot`` invocation and read back the options:
            ```python
            >>> from digitalearth.ops.cli import build_parser
            >>> args = build_parser().parse_args(["plot", "in.tif", "-o", "out.png", "--kind", "contourf"])
            >>> args.input, args.output, args.kind, args.crs
            ('in.tif', 'out.png', 'contourf', 3857)
            >>> args.backend  # the default backend, preserving matplotlib behaviour
            'matplotlib'

            ```
        - Parse a ``batch`` invocation with a gallery page, a non-EPSG CRS and a chosen backend:
            ```python
            >>> from digitalearth.ops.cli import build_parser
            >>> args = build_parser().parse_args(
            ...     ["batch", "a.tif", "b.tif", "-o", "out", "--html", "g.html", "--no-colorbar",
            ...      "--backend", "web"])
            >>> args.inputs, args.outdir, args.html, args.colorbar
            (['a.tif', 'b.tif'], 'out', 'g.html', False)
            >>> args.backend  # forwarded to quickmap, which dispatches to the web tier
            'web'

            ```
    """
    parser = argparse.ArgumentParser(
        prog="digitalearth", description="Render geospatial data to images."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plot = sub.add_parser("plot", help="render a single input to an image")
    plot.add_argument("input", help="raster or vector file to plot")
    plot.add_argument(
        "-o", "--output", default=None, help="output image path (default: <input>.png)"
    )
    _add_plot_options(plot)
    plot.set_defaults(func=_cmd_plot)

    batch = sub.add_parser(
        "batch", help="render many inputs, optionally into an HTML gallery"
    )
    batch.add_argument("inputs", nargs="+", help="raster/vector files to plot")
    batch.add_argument(
        "-o", "--outdir", required=True, help="directory to write images into"
    )
    batch.add_argument(
        "--html",
        default=None,
        help="also write a self-contained HTML gallery to this path",
    )
    batch.add_argument(
        "--ext",
        default=None,
        help="output file format (default: the backend's native format — png for matplotlib, html otherwise)",
    )
    _add_plot_options(batch)
    batch.set_defaults(func=_cmd_batch)
    return parser


def _cmd_plot(args: argparse.Namespace) -> int:
    """Render one input and save it; return a process exit code.

    Without ``-o`` the output defaults to ``<input-stem>.<native ext>``, where the native extension follows
    the chosen ``--backend`` (:data:`_NATIVE_EXT`) — a PNG for matplotlib, an HTML page for the others. An
    explicit ``-o`` is honoured as given, and the tier's own ``save`` writes that format or raises for one it
    cannot.
    """
    output = args.output or f"{Path(args.input).stem}.{_NATIVE_EXT[args.backend]}"
    scene = quickmap(_load(args.input), **_plot_kwargs(args))
    scene.save(output)
    print(f"wrote {output}", file=sys.stderr)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    """Render many inputs (and an optional gallery); return a process exit code.

    The output format is ``--ext`` when given, else the chosen ``--backend``'s native extension
    (:data:`_NATIVE_EXT`) — so ``--backend web`` writes HTML pages without the caller restating the format,
    while an explicit ``--ext`` still wins.
    """
    ext = args.ext or _NATIVE_EXT[args.backend]
    batch = Batch(ext=ext, **_plot_kwargs(args))
    paths = batch.run(args.inputs, args.outdir)
    print(f"wrote {len(paths)} image(s) to {args.outdir}", file=sys.stderr)
    if args.html:
        page = gallery(paths, args.html)
        print(f"wrote gallery {page}", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``digitalearth`` CLI.

    Args:
        argv: Argument list (excluding the program name); defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on success).

    Examples:
        - Render the bundled sample raster to a temporary PNG:
            ```python
            >>> import tempfile, os
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.ops.cli import main
            >>> epsg = Dataset.read_file("examples/data/acc4000.tif").epsg
            >>> out = os.path.join(tempfile.mkdtemp(), "acc.png")
            >>> main(["plot", "examples/data/acc4000.tif", "-o", out, "--crs", str(epsg), "--no-colorbar"])
            0
            >>> os.path.getsize(out) > 0
            True

            ```
    """
    matplotlib.use(
        "Agg", force=True
    )  # render headless to a file — set on invocation, never on import
    args = build_parser().parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code
