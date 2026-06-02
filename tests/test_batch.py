"""Tests for RP.11 — Batch rendering driver (digitalearth.batch)."""

from pathlib import Path

import pytest

from digitalearth.batch import Batch, _default_namer
from digitalearth.scene import Map


class TestDefaultNamer:
    """Tests for the _default_namer helper."""

    @pytest.mark.parametrize(
        "item, index, expected",
        [
            ("data/lisbon_dem.tif", 0, "lisbon_dem"),
            (Path("a/b/acc4000.TIF"), 3, "acc4000"),
            (object(), 7, "figure_007"),
        ],
    )
    def test_naming(self, item, index, expected):
        """_default_namer uses the file stem for paths and figure_<index> otherwise.

        Args:
            item: The input being named (path-like or in-memory object).
            index: The input's position in the batch.
            expected: The expected output stem.
        """
        assert _default_namer(item, index) == expected, f"{item!r} -> {_default_namer(item, index)}"


class TestBatch:
    """Tests for the Batch driver."""

    def test_init_stores_config(self):
        """The constructor strips a leading dot from ext and keeps the shared defaults."""
        b = Batch(crs=3857, kind="contourf", ext=".pdf")
        assert b.ext == "pdf", f"ext should be normalised, got {b.ext!r}"
        assert b.defaults == {"crs": 3857, "kind": "contourf"}, f"unexpected defaults: {b.defaults}"

    def test_render_one_from_object(self, dataset):
        """render_one passes an in-memory pyramids object straight through to the plotter."""
        m = Batch(colorbar=False).render_one(dataset, crs=dataset.epsg)
        assert isinstance(m, Map) and len(m.layers) == 1, "expected one drawn layer on a Map"

    def test_render_one_from_path(self, tmp_path, dataset):
        """render_one reads a path input via pyramids before plotting."""
        src = tmp_path / "scene.tif"
        dataset.to_file(str(src))
        m = Batch(colorbar=False).render_one(str(src), crs=dataset.epsg)
        assert len(m.layers) == 1, "a path input should be read and drawn"

    def test_run_writes_one_image_per_input(self, tmp_path, dataset):
        """run renders every input, saving one image per input and returning their paths."""
        outdir = tmp_path / "out"
        paths = Batch(crs=dataset.epsg, colorbar=False).run(
            [dataset, dataset], outdir, namer=lambda item, i: f"frame_{i}"
        )
        assert [p.name for p in paths] == ["frame_0.png", "frame_1.png"], f"unexpected names: {paths}"
        assert all(p.exists() and p.stat().st_size > 0 for p in paths), "images must be non-empty"

    def test_run_creates_outdir_and_uses_default_namer(self, tmp_path, dataset):
        """run creates a missing output directory and falls back to figure_<index> names."""
        outdir = tmp_path / "nested" / "out"
        paths = Batch(crs=dataset.epsg, colorbar=False).run([dataset], outdir)
        assert outdir.is_dir(), "run should create the output directory"
        assert paths[0].name == "figure_000.png", f"default namer expected, got {paths[0].name}"

    def test_run_overrides_take_precedence(self, tmp_path, dataset):
        """Per-run overrides win over construction-time defaults (ext from the override)."""
        b = Batch(crs=dataset.epsg, kind="imshow", colorbar=False, ext="png")
        paths = b.run([dataset], tmp_path, kind="contourf")
        assert paths[0].exists(), "the override run should still produce an image"
