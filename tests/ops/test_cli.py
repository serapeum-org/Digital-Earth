"""Tests for RP.11 — the digitalearth command line (digitalearth.ops.cli)."""

from pathlib import Path

import pytest
from pyramids.feature import FeatureCollection

from digitalearth.ops.batch import load_input
from digitalearth.ops.cli import _load, _parse_crs, _plot_kwargs, build_parser, main


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

    def test_plot_and_batch_share_one_loader(self):
        """``cli._load`` is ``batch.load_input`` itself, not a second copy of it (#229).

        Test scenario:
            The two subcommands drifted apart because each opened its inputs its own way; identity here is
            what stops that happening again.
        """
        assert _load is load_input, (
            "cli._load must be the batch loader, not a private duplicate"
        )

    def test_batch_subcommand_renders_a_vector_file(self, tmp_path):
        """``digitalearth batch <vector>`` writes an image, as ``digitalearth plot`` already did (#229).

        Test scenario:
            The reported repro: batch on tests/data/points.geojson used to raise a bare GDAL RuntimeError.
        """
        code = main(
            [
                "batch",
                "tests/data/points.geojson",
                "-o",
                str(tmp_path),
                "--crs",
                "4326",
                "--no-colorbar",
            ]
        )
        written = sorted(tmp_path.glob("*.png"))
        assert code == 0, (
            f"batch should succeed on a vector input, got exit code {code}"
        )
        assert len(written) == 1, f"expected one image, got {written}"
        assert written[0].stat().st_size > 0, "the written image must not be empty"

    def test_loads_raster(self, tmp_path, dataset):
        """A raster path loads as a pyramids Dataset (the primary path)."""
        src = tmp_path / "r.tif"
        dataset.to_file(str(src))
        loaded = _load(str(src))
        assert hasattr(loaded, "read_array"), (
            "a raster should load as a Dataset-like object"
        )

    def test_falls_back_to_vector(self):
        """A vector file that is not a raster falls back to a FeatureCollection."""
        loaded = _load("tests/data/points.geojson")
        assert isinstance(loaded, FeatureCollection), (
            f"expected FeatureCollection, got {type(loaded)}"
        )

    def test_both_loaders_fail_chains_errors(self, tmp_path):
        """A path that is neither raster nor vector raises the vector error chained from the raster one (L4)."""
        bogus = tmp_path / "not_geo.tif"
        bogus.write_text("this is plain text, not a geospatial file", encoding="utf-8")
        as_text = str(bogus)
        with pytest.raises(Exception) as exc:
            _load(as_text)
        assert exc.value.__cause__ is not None, (
            "the raster cause should be chained onto the vector error"
        )


class TestPlotKwargs:
    """Tests for _plot_kwargs."""

    def test_omits_unset_styling(self):
        """Unset cmap/levels/domain are omitted; the always-present flags remain."""
        args = build_parser().parse_args(["plot", "in.tif"])
        kwargs = _plot_kwargs(args)
        assert "cmap" not in kwargs
        assert "levels" not in kwargs
        assert "domain" not in kwargs
        assert kwargs["crs"] == 3857
        assert kwargs["colorbar"] is True

    def test_includes_set_styling(self):
        """Provided cmap/levels/domain are forwarded."""
        args = build_parser().parse_args(
            [
                "plot",
                "in.tif",
                "--cmap",
                "terrain",
                "--levels",
                "8",
                "--domain",
                "europe",
            ]
        )
        kwargs = _plot_kwargs(args)
        assert kwargs["cmap"] == "terrain"
        assert kwargs["levels"] == 8
        assert kwargs["domain"] == "europe"

    def test_backend_defaults_to_matplotlib(self):
        """With no --backend, quickmap is asked for the matplotlib backend — today's behaviour (TD-22)."""
        args = build_parser().parse_args(["plot", "in.tif"])
        assert _plot_kwargs(args)["backend"] == "matplotlib", (
            f"default backend should be matplotlib, got {_plot_kwargs(args).get('backend')!r}"
        )

    def test_backend_is_forwarded_when_set(self):
        """A chosen --backend reaches quickmap through the forwarded kwargs, not just the parser (TD-22)."""
        args = build_parser().parse_args(["plot", "in.tif", "--backend", "interactive"])
        assert _plot_kwargs(args)["backend"] == "interactive", (
            f"--backend interactive should be forwarded, got {_plot_kwargs(args).get('backend')!r}"
        )


class TestBackendCrsDefault:
    """The ``--crs`` default is backend-aware so ``--backend web`` works out of the box (M2).

    The web tier renders in EPSG:4326 only (``digitalearth.web.base.DISPLAY_CRS``), so the shared 3857
    default made a bare ``--backend web`` always raise. When the caller passes no ``--crs``, the web backend
    now defaults to 4326 while every other tier keeps 3857; an explicit ``--crs`` is never overridden.
    """

    def test_web_backend_defaults_crs_to_4326(self):
        """With no --crs, --backend web forwards crs=4326 (the only CRS the web tier accepts)."""
        args = build_parser().parse_args(["plot", "in.tif", "--backend", "web"])
        assert _plot_kwargs(args)["crs"] == 4326, (
            f"a web plot with no --crs should default to 4326, got {_plot_kwargs(args).get('crs')!r}"
        )

    def test_matplotlib_backend_keeps_crs_3857(self):
        """With no --crs, the matplotlib backend keeps the 3857 default — unchanged behaviour."""
        args = build_parser().parse_args(["plot", "in.tif"])
        assert _plot_kwargs(args)["crs"] == 3857, (
            f"a matplotlib plot with no --crs should stay 3857, got {_plot_kwargs(args).get('crs')!r}"
        )

    def test_interactive_and_3d_keep_crs_3857(self):
        """interactive/3d accept 3857, so they keep the shared default — only web is special-cased."""
        for backend in ("interactive", "3d"):
            args = build_parser().parse_args(["plot", "in.tif", "--backend", backend])
            assert _plot_kwargs(args)["crs"] == 3857, (
                f"--backend {backend} with no --crs should stay 3857, got {_plot_kwargs(args).get('crs')!r}"
            )

    def test_explicit_crs_is_not_overridden_for_web(self):
        """An explicit --crs wins even for web, so the tier's own error still fires for a bad explicit choice."""
        args = build_parser().parse_args(
            ["plot", "in.tif", "--backend", "web", "--crs", "3857"]
        )
        assert _plot_kwargs(args)["crs"] == 3857, (
            "an explicit --crs 3857 must be preserved for web, not silently rewritten to 4326"
        )

    def test_plot_web_default_runs_and_forwards_4326(self, tmp_path, dataset, mocker):
        """`plot --backend web` with no --crs no longer raises and reaches quickmap with crs=4326 (spied)."""
        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        spy = mocker.patch("digitalearth.ops.cli.quickmap")
        rc = main(
            ["plot", str(src), "-o", str(tmp_path / "m.html"), "--backend", "web"]
        )
        assert rc == 0, "a bare `plot --backend web` should succeed, not raise"
        assert spy.call_args.kwargs.get("crs") == 4326, (
            f"quickmap should be told crs=4326 for a default web plot, got {spy.call_args}"
        )

    def test_batch_web_default_forwards_4326(self, tmp_path, dataset, mocker):
        """`batch --backend web` with no --crs builds a Batch with crs=4326, not the failing 3857 default."""
        src = tmp_path / "a.tif"
        dataset.to_file(str(src))
        fake = mocker.patch("digitalearth.ops.cli.Batch")
        fake.return_value.run.return_value = []
        rc = main(["batch", str(src), "-o", str(tmp_path / "out"), "--backend", "web"])
        assert rc == 0, "a bare `batch --backend web` should succeed, not raise"
        assert fake.call_args.kwargs.get("crs") == 4326, (
            f"Batch should be told crs=4326 for a default web batch, got {fake.call_args}"
        )


class TestBuildParser:
    """Tests for build_parser."""

    def test_missing_subcommand_errors(self):
        """Invoking with no subcommand exits with an error (required subparser)."""
        build_parser2 = build_parser()
        with pytest.raises(SystemExit):
            build_parser2.parse_args([])

    def test_backend_rejects_unknown_choice(self):
        """--backend is constrained to the four quickmap dispatches; an unknown one is a usage error."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["plot", "in.tif", "--backend", "bogus"])


class TestMain:
    """Tests for main (end-to-end, headless)."""

    def test_plot_with_explicit_output(self, tmp_path, dataset):
        """`plot -o OUT` renders a single raster to the given path and returns 0."""
        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        out = tmp_path / "map.png"
        rc = main(
            [
                "plot",
                str(src),
                "-o",
                str(out),
                "--crs",
                str(dataset.epsg),
                "--no-colorbar",
            ]
        )
        assert rc == 0 and out.stat().st_size > 0, (
            "plot should write a non-empty image and exit 0"
        )

    def test_plot_default_output_name(self, tmp_path, dataset, monkeypatch):
        """Without -o, the output defaults to <input-stem>.png in the working directory."""
        src = tmp_path / "scene.tif"
        dataset.to_file(str(src))
        monkeypatch.chdir(tmp_path)
        rc = main(["plot", str(src), "--crs", str(dataset.epsg), "--no-colorbar"])
        assert rc == 0 and (tmp_path / "scene.png").exists(), (
            "default output should be scene.png"
        )

    def test_batch_with_gallery(self, tmp_path, dataset):
        """`batch ... --html` renders every input and writes a self-contained gallery page."""
        src = tmp_path / "a.tif"
        dataset.to_file(str(src))
        outdir = tmp_path / "imgs"
        page = tmp_path / "gallery.html"
        rc = main(
            [
                "batch",
                str(src),
                "-o",
                str(outdir),
                "--html",
                str(page),
                "--crs",
                str(dataset.epsg),
                "--no-colorbar",
            ]
        )
        assert rc == 0, "batch should exit 0"
        assert list(outdir.glob("*.png")), "batch should write at least one image"
        assert "data:image/png;base64," in page.read_text(encoding="utf-8"), (
            "gallery should embed images"
        )

    def test_batch_without_gallery(self, tmp_path, dataset):
        """`batch` with no --html renders images but writes no gallery page."""
        src = tmp_path / "a.tif"
        dataset.to_file(str(src))
        outdir = tmp_path / "imgs"
        rc = main(
            [
                "batch",
                str(src),
                "-o",
                str(outdir),
                "--crs",
                str(dataset.epsg),
                "--no-colorbar",
            ]
        )
        assert rc == 0 and list(outdir.glob("*.png")), (
            "batch should still write images without a gallery"
        )

    def test_python_m_entrypoint(self, tmp_path, dataset, monkeypatch):
        """`python -m digitalearth` runs __main__.py, exiting with main()'s return code."""
        import runpy
        import sys

        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        out = tmp_path / "m.png"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "digitalearth",
                "plot",
                str(src),
                "-o",
                str(out),
                "--crs",
                str(dataset.epsg),
                "--no-colorbar",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("digitalearth", run_name="__main__")
        assert exc.value.code == 0 and out.exists(), (
            "the -m entry point should render and exit 0"
        )

    def test_dunder_main_import_is_inert(self):
        """Importing digitalearth.__main__ (not as a script) exposes main without running it."""
        import importlib

        mod = importlib.import_module("digitalearth.__main__")
        assert hasattr(mod, "main"), (
            "the module should expose main without executing the CLI on import"
        )


class TestBackend:
    """Tests for headless backend handling (M2 — no global side effect on import)."""

    def test_backend_switch_is_not_at_import_scope(self):
        """matplotlib.use must run inside a function, never at module import scope."""
        import inspect

        import digitalearth.ops.cli as climod

        for line in inspect.getsource(climod).splitlines():
            if "matplotlib.use(" in line:
                assert line.startswith(" "), (
                    f"matplotlib.use must not run at import scope: {line!r}"
                )

    def test_main_forces_agg_on_invocation(self, tmp_path, dataset, mocker):
        """main() selects the Agg backend when invoked (force=True), not merely on import."""
        spy = mocker.patch("digitalearth.ops.cli.matplotlib.use")
        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        rc = main(
            [
                "plot",
                str(src),
                "-o",
                str(tmp_path / "m.png"),
                "--crs",
                str(dataset.epsg),
                "--no-colorbar",
            ]
        )
        assert rc == 0, "the command should still succeed"
        spy.assert_any_call("Agg", force=True)


class TestBackendDispatch:
    """Tests for --backend (TD-22, #208) — the CLI drives all four quickmap backends, not just matplotlib.

    The forwarding is checked by spying on ``quickmap``/``Batch`` rather than rendering, so these run in the
    ``dev`` env without the interactive/web/viz3d engines: a real ``--backend web`` render needs the ``web``
    extra, but whether the CLI *forwards* the choice does not.
    """

    def test_plot_forwards_backend_to_quickmap(self, tmp_path, dataset, mocker):
        """``plot --backend web`` reaches ``quickmap(..., backend="web")`` (spied, not rendered)."""
        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        spy = mocker.patch("digitalearth.ops.cli.quickmap")
        rc = main(
            [
                "plot",
                str(src),
                "-o",
                str(tmp_path / "m.png"),
                "--backend",
                "web",
                "--crs",
                "4326",
            ]
        )
        assert rc == 0, "the plot command should still exit 0"
        assert spy.call_args.kwargs.get("backend") == "web", (
            f"quickmap should be told backend='web', got {spy.call_args}"
        )

    def test_plot_default_output_extension_follows_backend(
        self, tmp_path, dataset, mocker
    ):
        """Without -o, a web scene defaults to a .html file — a page, not a PNG (native output)."""
        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        spy = mocker.patch("digitalearth.ops.cli.quickmap")
        rc = main(["plot", str(src), "--backend", "web", "--crs", "4326"])
        saved = str(spy.return_value.save.call_args.args[0])
        assert rc == 0, "the plot command should still exit 0"
        assert saved.endswith(".html"), (
            f"a web scene's default output should be .html, got {saved!r}"
        )

    def test_plot_matplotlib_default_output_stays_png(self, tmp_path, dataset, mocker):
        """The matplotlib default output stays <stem>.png, exactly as before --backend existed."""
        src = tmp_path / "scene.tif"
        dataset.to_file(str(src))
        spy = mocker.patch("digitalearth.ops.cli.quickmap")
        rc = main(["plot", str(src), "--crs", str(dataset.epsg)])
        saved = str(spy.return_value.save.call_args.args[0])
        assert rc == 0, "the plot command should still exit 0"
        assert saved.endswith(".png"), (
            f"matplotlib default output should stay .png, got {saved!r}"
        )

    def test_plot_explicit_output_wins_over_native_ext(self, tmp_path, dataset, mocker):
        """An explicit -o path is written as given; the backend's native extension does not override it."""
        src = tmp_path / "in.tif"
        dataset.to_file(str(src))
        out = tmp_path / "custom.png"
        spy = mocker.patch("digitalearth.ops.cli.quickmap")
        rc = main(
            ["plot", str(src), "-o", str(out), "--backend", "web", "--crs", "4326"]
        )
        saved = str(spy.return_value.save.call_args.args[0])
        assert rc == 0, "the plot command should still exit 0"
        assert saved.endswith("custom.png"), (
            f"the explicit -o path should win, got {saved!r}"
        )

    def test_batch_passes_backend_and_native_ext(self, tmp_path, dataset, mocker):
        """``batch --backend web`` builds a Batch with backend='web' and its native ext ('html')."""
        src = tmp_path / "a.tif"
        dataset.to_file(str(src))
        fake = mocker.patch("digitalearth.ops.cli.Batch")
        fake.return_value.run.return_value = []
        rc = main(
            [
                "batch",
                str(src),
                "-o",
                str(tmp_path / "out"),
                "--backend",
                "web",
                "--crs",
                "4326",
            ]
        )
        assert rc == 0, "the batch command should still exit 0"
        assert fake.call_args.kwargs.get("backend") == "web", (
            f"Batch should be told backend='web', got {fake.call_args}"
        )
        assert fake.call_args.kwargs.get("ext") == "html", (
            f"a web batch should default to ext='html', got {fake.call_args}"
        )

    def test_batch_ext_default_stays_png_for_matplotlib(
        self, tmp_path, dataset, mocker
    ):
        """A matplotlib batch with no --ext keeps ext='png', exactly as before."""
        src = tmp_path / "a.tif"
        dataset.to_file(str(src))
        fake = mocker.patch("digitalearth.ops.cli.Batch")
        fake.return_value.run.return_value = []
        rc = main(
            ["batch", str(src), "-o", str(tmp_path / "out"), "--crs", str(dataset.epsg)]
        )
        assert rc == 0, "the batch command should still exit 0"
        assert fake.call_args.kwargs.get("ext") == "png", (
            f"a matplotlib batch should default to ext='png', got {fake.call_args}"
        )

    def test_batch_explicit_ext_beats_native(self, tmp_path, dataset, mocker):
        """An explicit --ext overrides the backend's native extension (the caller asked for that format)."""
        src = tmp_path / "a.tif"
        dataset.to_file(str(src))
        fake = mocker.patch("digitalearth.ops.cli.Batch")
        fake.return_value.run.return_value = []
        rc = main(
            [
                "batch",
                str(src),
                "-o",
                str(tmp_path / "out"),
                "--backend",
                "web",
                "--ext",
                "png",
                "--crs",
                "4326",
            ]
        )
        assert rc == 0, "the batch command should still exit 0"
        assert fake.call_args.kwargs.get("ext") == "png", (
            f"an explicit --ext should win over the native ext, got {fake.call_args}"
        )
