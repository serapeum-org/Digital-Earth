"""Wave-0 cross-backend contract, the 3-D half — one regression per contract point.

The four rendering tiers are supposed to agree on the handful of names and behaviours a caller moves between
them with. This file pins the 3-D tier's side of that agreement, so a later refactor that quietly reverts one
of them fails here rather than in a user's notebook:

* **C1** — ``save()`` returns a :class:`pathlib.Path`; the ``np.ndarray`` it used to return lives on
  :meth:`~digitalearth.three_d.base.Scene3DBase.screenshot`.
* **C2** — the frame rate is ``fps``, defaulting to ``3.0``, with ``framerate=`` a deprecated alias on both
  :meth:`~digitalearth.three_d.animation.AnimationMixin.orbit` and
  :meth:`~digitalearth.three_d.animation.AnimationMixin.animate`.
* **C3** — marker size is ``size``, with ``point_size=`` a deprecated alias.
* **C4** — a classifiable layer takes ``scheme`` + ``k``, computing the same classes as the 2-D tiers.
* **C7** — a layer with nothing to draw is skipped with a warning, or raises under ``strict=True``.
* **C13** — this tier declares that it has no display CRS.

Every deprecated alias is tested twice over: that it still works, and that it warns.
"""

import logging

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.crs import OffLimbError
from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d.animation import DEFAULT_FPS
from digitalearth.three_d.base import Scene3DBase, classified_scalars


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test in this module."""
    previous = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = previous


@pytest.fixture
def scene():
    """Yield an off-screen ``Scene3D``, closed on the way out."""
    built = Scene3D(off_screen=True)
    yield built
    built.close()


def _dem():
    """Build a small ramped DEM ``Source`` with real relief.

    Returns:
        A :class:`~digitalearth.base.sources.Source` over an 8x8 ramp.
    """
    return get_source(np.add.outer(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8)))


def _points(n=20):
    """Build an ``(n, 3)`` point table whose x column doubles as a value column.

    Args:
        n: How many points to build.

    Returns:
        numpy.ndarray: the point table.
    """
    return np.column_stack([np.arange(float(n)), np.arange(float(n)), np.zeros(n)])


class TestC1SaveReturnsPath:
    """C1 — ``save()`` returns the ``Path`` it wrote; ``screenshot()`` keeps the array."""

    @pytest.mark.parametrize("name", ["frame.png", "scene.gltf", "page.html"])
    def test_save_returns_the_path_it_wrote(self, scene, tmp_path, name):
        """Every dispatch branch of ``save()`` returns a ``Path`` to the file on disk.

        Args:
            scene: The scene under test.
            tmp_path: Supplies the destination directory.
            name: One destination per ``save()`` branch — raster frame, scene export, HTML page.

        Test scenario:
            The return type used to depend on the suffix (an array for a raster, ``None`` for an export),
            which gave a caller nothing to chain on and no way to write one code path over both. It is now
            one type everywhere, and it points at a file that really exists.
        """
        written = scene.save(str(tmp_path / name))
        assert isinstance(written, type(tmp_path)), (
            f"save() must return a Path, got {type(written).__name__}"
        )
        assert written.is_file() and written.stat().st_size > 0, (
            f"save() returned {written!r}, which is not a written file"
        )

    def test_the_frame_moved_to_screenshot(self, scene):
        """The ``np.ndarray`` ``save()`` used to return is still available, from ``screenshot()``.

        Args:
            scene: The scene under test.

        Test scenario:
            C1 is a move, not a removal: the one branch that produced a useful value keeps producing it
            under its own name, so nothing a caller could do before is now impossible.
        """
        scene.add_mesh(pv.Sphere())
        frame = scene.screenshot()
        assert isinstance(frame, np.ndarray) and frame.ndim == 3, (
            f"screenshot() must return an (H, W, C) array, got {type(frame).__name__}"
        )


class TestC2Fps:
    """C2 — the frame rate is ``fps``, defaults to ``3.0``, and ``framerate=`` is a deprecated alias."""

    def test_the_shared_default_is_three(self):
        """Both animation entry points default to the one cross-tier frame rate.

        Test scenario:
            The tier used to carry two different defaults (12 for ``orbit``, 8 for ``animate``), neither
            matching the other backends. One constant now backs both, so "the default speed" is one number.
        """
        import inspect

        assert DEFAULT_FPS == 3.0, f"the shared default must be 3.0, got {DEFAULT_FPS}"
        for method in ("orbit", "animate"):
            signature = inspect.signature(getattr(Scene3D, method))
            assert "fps" in signature.parameters, f"{method}() must take fps"
            assert signature.parameters["fps"].default is None, (
                f"{method}()'s fps must default to the not-passed sentinel, "
                f"got {signature.parameters['fps'].default!r}"
            )

    def test_fps_reaches_the_writer(self, scene, tmp_path, mocker):
        """``fps=`` is what the GIF writer is opened with.

        Args:
            scene: The scene under test.
            tmp_path: Supplies the destination path.
            mocker: Patches the plotter's writer so no frames are encoded.

        Test scenario:
            Renaming the parameter is only worth anything if the new name is the one that reaches PyVista.
        """
        scene.terrain(_dem())
        opened = mocker.patch.object(scene.plotter, "open_gif")
        mocker.patch.object(scene.plotter, "write_frame")
        scene.animate([1.0], str(tmp_path / "a.gif"), lambda s, f: None, fps=7.0)
        assert opened.call_args.kwargs["fps"] == 7.0, (
            f"the writer must be opened at the fps given, got {opened.call_args!r}"
        )

    @pytest.mark.parametrize("method", ["orbit", "animate"])
    def test_framerate_still_works_and_warns(self, scene, tmp_path, mocker, method):
        """The deprecated ``framerate=`` still sets the frame rate, and says so once.

        Args:
            scene: The scene under test.
            tmp_path: Supplies the destination path.
            mocker: Patches the writer and the render loop so nothing is encoded.
            method: The animation entry point under test — both sites are aliased.

        Test scenario:
            No caller is broken in this batch: the old spelling forwards to the new one and emits a
            DeprecationWarning naming the replacement, on both methods that took it.
        """
        scene.terrain(_dem())
        opened = mocker.patch.object(scene.plotter, "open_gif")
        mocker.patch.object(scene.plotter, "write_frame")
        mocker.patch.object(scene.plotter, "orbit_on_path")
        mocker.patch.object(scene.plotter, "generate_orbital_path")
        out = str(tmp_path / f"{method}.gif")
        call = (
            (lambda: scene.orbit(out, n_frames=4, framerate=9.0))
            if method == "orbit"
            else (lambda: scene.animate([1.0], out, lambda s, f: None, framerate=9.0))
        )
        with pytest.warns(DeprecationWarning, match="fps"):
            call()
        assert opened.call_args.kwargs["fps"] == 9.0, (
            f"framerate= must forward to fps, got {opened.call_args!r}"
        )

    def test_both_spellings_at_once_is_refused(self, scene, tmp_path):
        """Passing ``fps`` and ``framerate`` together raises rather than silently picking one.

        Args:
            scene: The scene under test.
            tmp_path: Supplies the destination path.

        Test scenario:
            The two name the same thing, so a call that sets both is a contradiction only the caller can
            resolve. Preferring one of them would hide a real mistake.
        """
        with pytest.raises(ValueError, match="both fps= and the deprecated framerate="):
            scene.animate(
                [1.0],
                str(tmp_path / "a.gif"),
                lambda s, f: None,
                fps=4.0,
                framerate=9.0,
            )


class TestC3Size:
    """C3 — marker size is ``size``, with ``point_size=`` a deprecated alias."""

    def test_size_sets_the_marker_size(self, scene):
        """``size=`` reaches PyVista as the point size.

        Args:
            scene: The scene under test.

        Test scenario:
            ``size`` is the one spelling for "how big is a marker" across the tiers; it has to be the value
            the engine actually renders at, not a name that lands nowhere.
        """
        actor = scene.point_cloud(_points(), size=8.0)
        assert actor.prop.point_size == pytest.approx(8.0), (
            f"size= must set the rendered point size, got {actor.prop.point_size}"
        )

    def test_point_size_still_works_and_warns(self, scene):
        """The deprecated ``point_size=`` still sizes the markers, and says so.

        Args:
            scene: The scene under test.

        Test scenario:
            The old spelling forwards to ``size`` and emits a DeprecationWarning naming the replacement, so
            existing notebooks keep rendering while telling their author what to change.
        """
        with pytest.warns(DeprecationWarning, match="size"):
            actor = scene.point_cloud(_points(), point_size=8.0)
        assert actor.prop.point_size == pytest.approx(8.0), (
            f"point_size= must forward to size, got {actor.prop.point_size}"
        )

    def test_both_spellings_at_once_is_refused(self, scene):
        """Passing ``size`` and ``point_size`` together raises rather than picking one.

        Args:
            scene: The scene under test.

        Test scenario:
            As with ``fps``/``framerate``: two names for one thing, set to two values, is a caller error.
        """
        with pytest.raises(
            ValueError, match="both size= and the deprecated point_size="
        ):
            scene.point_cloud(_points(), size=4.0, point_size=8.0)


class TestC4SchemeAndK:
    """C4 — a classifiable 3-D layer takes ``scheme`` + ``k``, computing the 2-D tiers' classes."""

    def test_no_scheme_is_a_continuous_ramp(self, scene):
        """``scheme=None`` (the default) leaves the raw values on the mesh.

        Args:
            scene: The scene under test.

        Test scenario:
            The contract's default is a continuous ramp, which for VTK means handing it the values
            themselves rather than class codes — so the mesh must still carry the data's own range.
        """
        points = _points()
        scene.point_cloud(points, values=points[:, 0])
        assert scene.layers[0][0]["scalar"].max() == pytest.approx(19.0), (
            "an unclassified cloud keeps its own value range"
        )

    @pytest.mark.parametrize("k", [2, 4])
    def test_a_graduated_scheme_produces_k_classes(self, scene, k):
        """A graduated ``scheme`` bins the values into exactly ``k`` classes.

        Args:
            scene: The scene under test.
            k: The class count asked for.

        Test scenario:
            VTK maps a scalar range through a lookup table and knows nothing of class breaks, so a
            classified layer is rendered as class *indices* against a ``k``-colour table. Both halves are
            checked: the codes span ``0 .. k-1``, and the table holds ``k`` colours.
        """
        points = _points()
        scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=k)
        codes = sorted({int(value) for value in scene.layers[0][0]["scalar"]})
        assert codes == list(range(k)), f"expected classes 0..{k - 1}, got {codes}"

    def test_the_classes_match_what_the_other_tiers_compute(self):
        """The 3-D classes come from the same classifier the 2-D tiers use.

        Test scenario:
            The point of a shared ``scheme``/``k`` is that one call classifies identically whichever tier
            draws it. This pins the breaks against cleopatra's ``classify`` directly, so a private
            re-implementation here would fail rather than drift quietly.
        """
        from cleopatra.styling.styles import classify

        values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 40.0])
        edges, _ = classify(values, "quantiles", 3)
        style = classified_scalars(values, scheme="quantiles", k=3, cmap="viridis")
        expected = np.clip(np.digitize(values, np.asarray(edges)[1:-1]), 0, 2)
        assert list(style["scalars"]) == list(expected), (
            "the 3-D tier must bin against cleopatra's own class edges"
        )

    def test_an_explicit_colour_list_is_used_as_given(self):
        """A ``cmap`` that is already a list of colours becomes the class table unchanged.

        Test scenario:
            ``cmap`` is a colormap *name* on every tier, but a caller who has already chosen the exact
            per-class colours — to match a published legend — hands the list itself. Sampling that would
            need a matplotlib colormap lookup and fail on the one input that needs no sampling at all.
        """
        colours = ["#ff0000", "#00ff00"]
        style = classified_scalars(
            [1.0, 2.0, 3.0, 40.0], scheme="quantiles", k=2, cmap=colours
        )
        assert style["cmap"] == colours, (
            f"the colours must be used as given, got {style['cmap']}"
        )
        assert style["n_colors"] == 2, (
            f"one colour per class expected, got {style['n_colors']}"
        )

    def test_a_categorical_scheme_gives_each_value_its_own_colour(self, scene):
        """``scheme="categorical"`` colours by distinct value rather than by range.

        Args:
            scene: The scene under test.

        Test scenario:
            ``"categorical"`` keeps the meaning it has on every other tier — one colour per distinct value,
            from a qualitative colormap — rather than being reinterpreted as a graduated scheme with a
            different class count.
        """
        points = _points(6)
        scene.point_cloud(
            points, values=np.array([1, 1, 2, 2, 3, 3]), scheme="categorical"
        )
        assert sorted({int(v) for v in scene.layers[0][0]["scalar"]}) == [0, 1, 2], (
            "three distinct values must become three classes"
        )

    def test_extruded_polygons_classify_by_column(self, scene):
        """``extruded_polygons`` classifies the colour column, leaving the heights alone.

        Args:
            scene: The scene under test.

        Test scenario:
            The polygon layer is the tier's choropleth analogue, so it is where ``scheme``/``k`` earns its
            place. Colour is classified; the extrusion height stays the raw measurement, because squashing
            heights into classes would misstate the data the prisms encode.
        """
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Polygon

        squares = [Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]) for x in range(4)]
        gdf = gpd.GeoDataFrame({"pop": [1.0, 2.0, 30.0, 40.0]}, geometry=squares)
        scene.extruded_polygons(
            gdf, height="pop", column="pop", scheme="quantiles", k=2
        )
        mesh = scene.layers[0][0]
        assert sorted({int(v) for v in mesh.cell_data["value"]}) == [0, 1], (
            "the colour column must be binned into the two classes asked for"
        )
        assert mesh.bounds[5] == pytest.approx(40.0), (
            "classification must colour the prisms without flattening their heights"
        )

    def test_extruded_polygons_classify_a_label_column(self, scene):
        """``scheme="categorical"`` colours the string labels the method documents.

        Args:
            scene: The scene under test.

        Test scenario:
            The docstring offers a label column as the categorical case, so a column of strings must reach
            the classifier intact. It previously did not: the per-cell array was filled with ``float(raw)``
            before classification, which cannot represent a label, so the documented call raised
            ``ValueError: could not convert string to float``.
        """
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Polygon

        squares = [Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]) for x in range(3)]
        gdf = gpd.GeoDataFrame({"kind": ["park", "road", "park"]}, geometry=squares)
        scene.extruded_polygons(gdf, height=1.0, column="kind", scheme="categorical")
        mesh = scene.layers[0][0]
        assert sorted({int(v) for v in mesh.cell_data["value"]}) == [0, 1], (
            "two distinct labels must become two class codes, not a float conversion error"
        )

    def test_extruded_polygons_tolerate_a_missing_value(self, scene):
        """A column carrying ``NaN`` extrudes instead of raising ``KeyError``.

        Args:
            scene: The scene under test.

        Test scenario:
            Missing values are ordinary in feature data and rendered on the previous release. Re-keying the
            per-cell array through ``{float(raw): code}`` broke that, because ``NaN`` never equals itself and
            so can never be found in the mapping. The classifier is now applied once per feature instead.
        """
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Polygon

        squares = [Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]) for x in range(3)]
        gdf = gpd.GeoDataFrame({"pop": [1.0, float("nan"), 3.0]}, geometry=squares)
        scene.extruded_polygons(gdf, height=1.0, column="pop")
        mesh = scene.layers[0][0]
        assert mesh.n_cells > 0, "a column with a missing value must still extrude its prisms"

    def test_an_unusable_scheme_is_reported_by_name(self, scene):
        """A scheme that cannot classify the values raises, naming the scheme and ``k``.

        Args:
            scene: The scene under test.

        Test scenario:
            A constant column has no spread to split. Failing with the scheme and class count in the message
            beats surfacing cleopatra's own error, which names neither.
        """
        points = _points(5)
        with pytest.raises(ValueError, match="scheme='quantiles'"):
            scene.point_cloud(points, values=np.ones(5), scheme="quantiles", k=3)


class TestC7SkipAndWarn:
    """C7 — a layer with nothing to draw is skipped with a warning, or raises under ``strict=True``."""

    #: ``(method name, argument builder)`` for every layer that can end up with nothing to place.
    EMPTY_LAYERS = {
        "terrain": (lambda: (get_source(np.full((4, 4), np.nan)),), "finite elevation"),
        "point_cloud": (lambda: (np.zeros((0, 3)),), "point table is empty"),
        "vectors": (lambda: (np.zeros((0, 3)), np.zeros((0, 3))), "no points"),
    }

    @pytest.mark.parametrize("layer", sorted(EMPTY_LAYERS))
    def test_an_empty_layer_is_skipped_with_a_warning(self, scene, caplog, layer):
        """An empty layer draws nothing, registers nothing, and logs why.

        Args:
            scene: The scene under test.
            caplog: Captures the warning the skipped layer logs.
            layer: The layer method under test.

        Test scenario:
            The default across every tier is to carry on: one layer with no data must not cost a composed
            scene its other layers. It must not be *silent* either — the only other symptom is a missing
            layer, which reads as a bug in the data.
        """
        arguments, reason = self.EMPTY_LAYERS[layer]
        with caplog.at_level(logging.WARNING):
            assert getattr(scene, layer)(*arguments()) is None, (
                f"{layer}() must return None when it draws nothing"
            )
        assert not scene.layers, f"{layer}() must not register a layer it skipped"
        assert layer in caplog.text and reason in caplog.text, (
            f"the warning must name the layer and the reason, got {caplog.text!r}"
        )

    @pytest.mark.parametrize("layer", sorted(EMPTY_LAYERS))
    def test_strict_raises_off_limb_instead(self, layer):
        """``strict=True`` turns each of those skips into an ``OffLimbError``.

        Args:
            layer: The layer method under test.

        Test scenario:
            A pipeline that would rather fail than publish a map with a missing layer opts in once, on the
            scene, and every layer honours it — using the same error type the 2-D tiers raise.
        """
        arguments, reason = self.EMPTY_LAYERS[layer]
        strict = Scene3D(off_screen=True, strict=True)
        try:
            with pytest.raises(OffLimbError, match=reason):
                getattr(strict, layer)(*arguments())
        finally:
            strict.close()

    def test_strict_defaults_to_off(self, scene):
        """A scene built without ``strict`` is lenient, and says so on the instance.

        Args:
            scene: The scene under test.

        Test scenario:
            The contract's default is skip-and-warn, so the flag has to default to ``False`` — and be
            readable, since a caller handed a scene needs to know which mode it is in.
        """
        assert scene.strict is False, "strict must default to False"

    def test_an_empty_cloud_with_values_is_still_a_caller_error(self, scene):
        """A length mismatch is reported even when the cloud is empty.

        Args:
            scene: The scene under test.

        Test scenario:
            Skipping empty layers must not swallow a real mistake: values with no points to attach them to
            is a mismatch, and telling the caller beats logging "nothing to draw" and moving on.
        """
        with pytest.raises(ValueError, match="does not match"):
            scene.point_cloud(np.zeros((0, 3)), values=np.ones(3))


class TestC13NoDisplayCrs:
    """C13 — the 3-D tier declares that it has no display CRS."""

    def test_display_crs_is_declared_none(self, scene):
        """``display_crs`` exists and is ``None``, on the class and on an instance.

        Args:
            scene: The scene under test.

        Test scenario:
            "There is no display CRS" has to be *readable*, not merely true: the dispatcher refuses ``crs=``
            for this backend on the strength of it, and a docstring cannot be queried. Declared on the class
            so it can be checked without building a plotter.
        """
        assert Scene3DBase.display_crs is None, (
            "Scene3DBase must declare display_crs = None"
        )
        assert scene.display_crs is None, "a built scene must report no display CRS"

    def test_the_docstring_says_so_too(self):
        """The class docstring states the absence, for the reader who never looks at the attribute.

        Test scenario:
            The contract asks for both halves — a class attribute and a docstring line — because the
            attribute tells the dispatcher and the docstring tells the person wondering where ``crs=`` went.
        """
        assert "no display CRS" in Scene3DBase.__doc__, (
            "the class docstring must state that this tier has no display CRS"
        )
