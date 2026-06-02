"""cli — the ``digitalearth`` command line: render rasters/vectors to images without writing Python (RP.11).

Two subcommands wrap the existing API so plots can be produced from a shell or a Makefile:

* ``digitalearth plot INPUT [-o OUT] [options]`` — render one input via :func:`~digitalearth.api.quickmap`.
* ``digitalearth batch INPUTS... -o OUTDIR [--html PAGE] [options]`` — render many via
  :class:`~digitalearth.batch.Batch`, optionally collecting them into a static HTML gallery
  (:func:`~digitalearth.browser.gallery`).

The CLI always renders headless (matplotlib ``Agg``); human-facing progress goes to ``stderr`` so any piped
``stdout`` stays clean. This is earthkit-plots' ``cli/`` entry point, scoped to Digital-Earth's API.
"""
import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # the CLI only ever saves to a file — force a headless backend before pyplot loads

from pyramids.dataset import Dataset  # noqa: E402

from digitalearth.api import quickmap  # noqa: E402
from digitalearth.batch import Batch  # noqa: E402
from digitalearth.browser import gallery  # noqa: E402

__all__ = ["build_parser", "main"]


def _parse_crs(value: str) -> Any:
    """Parse a ``--crs`` argument as an EPSG int when all-digits, else a proj4/WKT string."""
    return int(value) if value.lstrip("-").isdigit() else value


def _load(path: Any) -> Any:
    """Load a raster as a pyramids ``Dataset``, falling back to a vector ``FeatureCollection``."""
    try:
        return Dataset.read_file(str(path))
    except Exception:  # not a raster pyramids can open — try it as vector
        from pyramids.feature import FeatureCollection

        return FeatureCollection.read_file(str(path))


def _add_plot_options(parser: argparse.ArgumentParser) -> None:
    """Attach the shared plotting flags (forwarded to ``quickmap``) to a subcommand parser."""
    parser.add_argument("--crs", type=_parse_crs, default=3857, help="display CRS (EPSG int or proj4 string)")
    parser.add_argument("--kind", default="auto", help="raster renderer: auto|imshow|contourf|contour|pcolormesh")
    parser.add_argument("--cmap", default=None, help="colormap name (default: auto-styled)")
    parser.add_argument("--levels", type=int, default=None, help="number of contour levels")
    parser.add_argument("--domain", default=None, help="named region / domain to set the extent")
    parser.add_argument("--basemap", action="store_true", help="overlay an XYZ tile basemap (needs network)")
    parser.add_argument("--coastlines", action="store_true", help="overlay coastlines (needs network)")
    parser.add_argument("--no-colorbar", dest="colorbar", action="store_false", help="omit the colorbar")
    parser.set_defaults(colorbar=True)


def _plot_kwargs(args: argparse.Namespace) -> dict:
    """Collect the ``quickmap`` keyword arguments set on ``args`` (omitting unset optional styling)."""
    kwargs: dict = {
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
            >>> from digitalearth.cli import build_parser
            >>> args = build_parser().parse_args(["plot", "in.tif", "-o", "out.png", "--kind", "contourf"])
            >>> args.input, args.output, args.kind, args.crs
            ('in.tif', 'out.png', 'contourf', 3857)

            ```
        - Parse a ``batch`` invocation with a gallery page and a non-EPSG CRS:
            ```python
            >>> from digitalearth.cli import build_parser
            >>> args = build_parser().parse_args(
            ...     ["batch", "a.tif", "b.tif", "-o", "out", "--html", "g.html", "--no-colorbar"])
            >>> args.inputs, args.outdir, args.html, args.colorbar
            (['a.tif', 'b.tif'], 'out', 'g.html', False)

            ```
    """
    parser = argparse.ArgumentParser(prog="digitalearth", description="Render geospatial data to images.")
    sub = parser.add_subparsers(dest="command", required=True)

    plot = sub.add_parser("plot", help="render a single input to an image")
    plot.add_argument("input", help="raster or vector file to plot")
    plot.add_argument("-o", "--output", default=None, help="output image path (default: <input>.png)")
    _add_plot_options(plot)
    plot.set_defaults(func=_cmd_plot)

    batch = sub.add_parser("batch", help="render many inputs, optionally into an HTML gallery")
    batch.add_argument("inputs", nargs="+", help="raster/vector files to plot")
    batch.add_argument("-o", "--outdir", required=True, help="directory to write images into")
    batch.add_argument("--html", default=None, help="also write a self-contained HTML gallery to this path")
    batch.add_argument("--ext", default="png", help="output image format (default: png)")
    _add_plot_options(batch)
    batch.set_defaults(func=_cmd_batch)
    return parser


def _cmd_plot(args: argparse.Namespace) -> int:
    """Render one input and save it; return a process exit code."""
    output = args.output or f"{Path(args.input).stem}.png"
    scene = quickmap(_load(args.input), **_plot_kwargs(args))
    scene.save(output)
    print(f"wrote {output}", file=sys.stderr)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    """Render many inputs (and an optional gallery); return a process exit code."""
    batch = Batch(ext=args.ext, **_plot_kwargs(args))
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
            >>> from digitalearth.cli import main
            >>> epsg = Dataset.read_file("examples/data/acc4000.tif").epsg
            >>> out = os.path.join(tempfile.mkdtemp(), "acc.png")
            >>> main(["plot", "examples/data/acc4000.tif", "-o", out, "--crs", str(epsg), "--no-colorbar"])
            0
            >>> os.path.getsize(out) > 0
            True

            ```
    """
    args = build_parser().parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code
