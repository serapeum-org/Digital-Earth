"""Tests for RP.11 — the digitalearth command line (digitalearth.cli)."""

from pathlib import Path

import pytest
from pyramids.feature import FeatureCollection

from digitalearth.cli import _load, _parse_crs, _plot_kwargs, build_parser, main


class TestParseCrs:
    """Tests for _parse_crs."""

    @pytest.mark.parametrize(
        "value, expected",
        [("3857", 3857), ("-1", -1), ("+proj=ortho +lon_0=0", "+proj=ortho +lon_0=0")],
    )
    def test_int_vs_string(self, value, expected):
        """An all-digit (optionally signed) value parses to int; anything else stays a string.

        Args:
            value: The raw --crs argument.
            expected: The parsed CRS (int or str).
        """
        assert _parse_crs(value) == expected, f"{value!r} -> {_parse_crs(value)!r}"


class TestLoad:
    """Tests for the _load raster-then-vector loader."""

    def test_loads_raster(self, tmp_path, dataset):
        """A raster path loads as a pyramids Dataset (the primary path)."""
        src = tmp_path / "r.tif"
        dataset.to_file(str(src))
        loaded = _load(str(src))
        assert hasattr(loaded, "read_array"), "a raster should load as a Dataset-like object"

    def test_falls_back_to_vector(self):
        """A vector file that is not a raster falls back to a FeatureCollection."""
        loaded = _load("tests/data/points.geojson")
        assert isinstance(loaded, FeatureCollection), f"expected FeatureCollection, got {type(loaded)}"


class TestPlotKwargs:
    """Tests for _plot_kwargs."""

    def test_omits_unset_styling(self):
        """Unset cmap/levels/domain are omitted; the always-present flags remain."""
        args = build_parser().parse_args(["plot", "in.tif"])
        kwargs = _plot_kwargs(args)
        assert "cmap" not in kwargs and "levels" not in kwargs and "domain" not in kwargs
        assert kwargs["crs"] == 3857 and kwargs["colorbar"] is True

    def test_includes_set_styling(self):
        """Provided cmap/levels/domain are forwarded."""
        args = build_parser().parse_args(
            ["plot", "in.tif", "--cmap", "terrain", "--levels", "8", "--domain", "europe"]
        )
        kwargs = _plot_kwargs(args)
        assert kwargs["cmap"] == "terrain" and kwargs["levels"] == 8 and kwargs["domain"] == "europe"


class TestBuildParser:
    """Tests for build_parser."""

    def test_missing_subcommand_errors(self):
        """Invoking with no subcommand exits with an error (required subparser)."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestMain:
    """Tests for main (end-to-end, headless)."""

    def test_plot_with_explicit_output(self, tmp_path, dataset):
        """`plot -o OUT` renders a single raster to the given path and returns 0."""
        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        out = tmp_path / "map.png"
        rc = main(["plot", str(src), "-o", str(out), "--crs", str(dataset.epsg), "--no-colorbar"])
        assert rc == 0 and out.stat().st_size > 0, "plot should write a non-empty image and exit 0"

    def test_plot_default_output_name(self, tmp_path, dataset, monkeypatch):
        """Without -o, the output defaults to <input-stem>.png in the working directory."""
        src = tmp_path / "scene.tif"
        dataset.to_file(str(src))
        monkeypatch.chdir(tmp_path)
        rc = main(["plot", str(src), "--crs", str(dataset.epsg), "--no-colorbar"])
        assert rc == 0 and (tmp_path / "scene.png").exists(), "default output should be scene.png"

    def test_batch_with_gallery(self, tmp_path, dataset):
        """`batch ... --html` renders every input and writes a self-contained gallery page."""
        src = tmp_path / "a.tif"
        dataset.to_file(str(src))
        outdir = tmp_path / "imgs"
        page = tmp_path / "gallery.html"
        rc = main(
            ["batch", str(src), "-o", str(outdir), "--html", str(page),
             "--crs", str(dataset.epsg), "--no-colorbar"]
        )
        assert rc == 0, "batch should exit 0"
        assert list(outdir.glob("*.png")), "batch should write at least one image"
        assert "data:image/png;base64," in page.read_text(encoding="utf-8"), "gallery should embed images"

    def test_batch_without_gallery(self, tmp_path, dataset):
        """`batch` with no --html renders images but writes no gallery page."""
        src = tmp_path / "a.tif"
        dataset.to_file(str(src))
        outdir = tmp_path / "imgs"
        rc = main(["batch", str(src), "-o", str(outdir), "--crs", str(dataset.epsg), "--no-colorbar"])
        assert rc == 0 and list(outdir.glob("*.png")), "batch should still write images without a gallery"

    def test_python_m_entrypoint(self, tmp_path, dataset, monkeypatch):
        """`python -m digitalearth` runs __main__.py, exiting with main()'s return code."""
        import runpy
        import sys

        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        out = tmp_path / "m.png"
        monkeypatch.setattr(
            sys, "argv",
            ["digitalearth", "plot", str(src), "-o", str(out), "--crs", str(dataset.epsg), "--no-colorbar"],
        )
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("digitalearth", run_name="__main__")
        assert exc.value.code == 0 and out.exists(), "the -m entry point should render and exit 0"

    def test_dunder_main_import_is_inert(self):
        """Importing digitalearth.__main__ (not as a script) exposes main without running it."""
        import importlib

        mod = importlib.import_module("digitalearth.__main__")
        assert hasattr(mod, "main"), "the module should expose main without executing the CLI on import"
