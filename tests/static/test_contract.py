"""The static tier's half of the cross-backend contract (Wave 0, batch B).

Every backend answers the same questions the same way — what ``save`` gives back, what a frame rate is
called and defaults to, what names a marker size and a classification, where a colormap comes from when
none was asked for, and what happens to a layer with nothing to draw. This module pins the static
(matplotlib) tier's answers, one class per contract point, plus the deprecated spellings each rename keeps
alive for a release.

The clauses themselves are stated once, in :data:`digitalearth.base.contract_clauses.CLAUSES`. Each class
below names the clause it pins and then says only what *this* tier brings to it — which builders, which
literal, which dial — so amending a clause is one edit in ``base/`` rather than one per tier (#326).

Every rename here is a *rename*: the old keyword still works, still does the same thing, and warns once
naming its replacement. A test per alias proves both halves of that, so the release that drops the alias
has a failing test to delete rather than a silent behaviour change to discover.
"""

import inspect
import logging
import re
import warnings

import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection

import digitalearth.static.maps.animation as animation_module
import digitalearth.static.maps.raster as raster_module
from digitalearth.static import Map, OffLimbError, Scene, TexturedGlobe, projections
from digitalearth.static.maps.decoration import STOCK_IMG_CMAP
from digitalearth.static.maps.raster import DEFAULT_FIELD_CMAP

#: The Natural-Earth reference layers the static tier names as methods (C12), and the ``natural_earth``
#: layer each one draws.
NATURAL_EARTH_METHODS = {
    "coastlines": "coastline",
    "borders": "borders",
    "land": "land",
    "ocean": "ocean",
    "lakes": "lakes",
    "rivers": "rivers",
}


@pytest.fixture
def points_fc():
    """The committed point fixture (EPSG:32618, numeric ``fid`` column).

    Returns:
        FeatureCollection: the ``tests/data/points.geojson`` points.
    """
    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygons_fc(points_fc):
    """Buffered polygons carrying the numeric ``fid`` column, for the choropleth tests.

    Args:
        points_fc: The committed point fixture.

    Returns:
        FeatureCollection: polygons in the points' CRS.
    """
    points_fc["geometry"] = points_fc.geometry.buffer(500.0)
    return points_fc


@pytest.fixture
def regional():
    """A small EPSG:4326 raster over the North Sea, used to exercise the off-limb path.

    Returns:
        Dataset: a 20x20 field around 4 E / 53 N.
    """
    return Dataset.from_array(
        np.ones((20, 20), "float32"),
        geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
    )


@pytest.fixture
def global_field():
    """A coarse whole-world raster, so an animation/rotation has something on every frame.

    Returns:
        Dataset: a single-band EPSG:4326 raster spanning the full lon/lat domain.
    """
    return Dataset.from_array(
        arr=np.full((60, 120), 1.0, "float32"),
        geo_ref=GeoReference(geo=(-180.0, 3.0, 0.0, 90.0, 0.0, -3.0), epsg=4326),
    )


def _styled(monkeypatch, **style):
    """Pin what ``auto_style`` resolves for every source the raster path reads.

    The committed fixtures carry no variable name (GDAL reports ``Band_1``), so the library lookup they
    really go through answers with its own default and nothing else — which cannot tell "consumed the
    lookup" from "ignored it". Replacing the lookup with a fixed answer makes the consumption observable.

    Args:
        monkeypatch: The pytest fixture doing the patching.
        **style: The style dict ``auto_style`` should return.
    """
    monkeypatch.setattr(raster_module, "auto_style", lambda source: dict(style))


class TestSaveReturnsPath:
    """C1 on the static tier — the ``Scene``, ``Map`` and globe saves, which take three different paths."""

    def test_scene_save_returns_the_written_path(self, tmp_path):
        """Scene.save returns the path it wrote, and the file is really there.

        Args:
            tmp_path: Destination directory for the written figure.

        Test scenario:
            A bare Scene is saved; the return value equals the requested path and names an existing file.
        """
        out = tmp_path / "scene.png"
        written = Scene().save(out)
        assert written == out, f"save returned {written!r}, not the path it was given"
        assert written.exists(), f"save reported {written!r} but no file is there"
        assert written.stat().st_size > 0, f"{written.name} was written empty"

    def test_map_save_returns_the_written_path(self, tmp_path, dataset):
        """Map.save — which applies the projection frame first — returns the path too.

        Args:
            tmp_path: Destination directory for the written figure.
            dataset: The committed ``acc4000`` raster.

        Test scenario:
            The ProjectionMixin override must forward the base class's return value, not swallow it.
        """
        m = Map(crs=dataset.epsg)
        m.imshow(dataset)
        written = m.save(tmp_path / "map.png")
        assert written == tmp_path / "map.png", (
            "the projection override dropped the return value"
        )
        assert written.exists(), "nothing was written"

    def test_globe_save_returns_the_written_path(self, tmp_path):
        """TexturedGlobe.save follows the same contract as the flat scenes.

        Args:
            tmp_path: Destination directory for the written figure.

        Test scenario:
            A tiny globe is drawn and saved; the return value is the path that now exists.
        """
        globe = TexturedGlobe(np.zeros((8, 16, 3), dtype=np.uint8), n_lon=8, n_lat=4)
        globe.draw()
        written = globe.save(tmp_path / "globe.png")
        assert written == tmp_path / "globe.png", (
            "the globe's save must return its path"
        )
        assert written.exists(), "nothing was written"


class TestFrameRateDefault:
    """C2 on the static tier — both animation entry points, ``animate`` and ``rotate``."""

    @pytest.mark.parametrize("method", ["animate", "rotate"])
    def test_fps_defaults_to_the_shared_rate(self, method):
        """animate and rotate declare the same ``fps`` default.

        Args:
            method: The animation entry point under test.

        Test scenario:
            rotate used to default to 8 fps against animate's 3, so the same stack played at two speeds
            depending on which entry point drew it.
        """
        default = inspect.signature(getattr(Map, method)).parameters["fps"].default
        assert default == animation_module.DEFAULT_FPS == 3.0, (
            f"{method} defaults to {default} fps, not the shared {animation_module.DEFAULT_FPS}"
        )

    def test_rotate_builds_at_the_shared_rate(self, global_field):
        """A rotation built with defaults really runs at the shared rate.

        Args:
            global_field: A whole-world raster to spin.

        Test scenario:
            The declared default reaches the FuncAnimation's interval and the saved-clip rate, rather than
            only the signature.
        """
        m = Map(crs=4326)
        anim = m.rotate(global_field, n_frames=3)
        assert m._animation_fps == 3.0, (
            "rotate recorded a different rate than it declared"
        )
        assert anim._interval == pytest.approx(1000.0 / 3.0), (
            "the frame interval does not match fps"
        )


class TestMarkerSizeAndColumn:
    """C3 on the static tier — the scatter builders, and the rename each of them carried."""

    def test_size_column_scales_the_markers(self, points_fc):
        """scatter(size_column=...) maps the column's values across ``size_limits``.

        Args:
            points_fc: The committed point fixture.

        Test scenario:
            The parameter renamed from ``scale`` still drives per-point marker area.
        """
        pc = Map(crs=points_fc.epsg).scatter(
            points_fc, size_column="fid", size_limits=(20, 200)
        )
        sizes = np.asarray(pc.get_sizes())
        assert sizes.min() == pytest.approx(20), (
            f"column= did not reach the lower size limit, got {sizes.min()}"
        )
        assert sizes.max() == pytest.approx(200), (
            f"column= did not reach the upper size limit, got {sizes.max()}"
        )

    def test_size_sets_a_uniform_marker_size(self, points_fc):
        """scatter(size=...) sets one marker size for every point.

        Args:
            points_fc: The committed point fixture.

        Test scenario:
            ``size`` is what the other backends call a marker's visual size, so the static tier takes it
            too and folds it onto cleopatra's ``point_size``.
        """
        pc = Map(crs=points_fc.epsg).scatter(points_fc, size=64)
        assert set(np.asarray(pc.get_sizes()).tolist()) == {64.0}, (
            "size= did not reach the markers"
        )

    def test_scale_alias_warns_and_still_works(self, points_fc):
        """The deprecated ``scale=`` keeps working and names its replacement.

        Args:
            points_fc: The committed point fixture.

        Test scenario:
            ``scale="fid"`` produces exactly what ``size_column="fid"`` produces, and warns once.
        """
        deprecated_map = Map(crs=points_fc.epsg)
        with pytest.warns(
            DeprecationWarning, match=r"scale= is deprecated.*use size_column="
        ):
            deprecated = deprecated_map.scatter(
                points_fc, scale="fid", size_limits=(20, 200)
            )
        renamed = Map(crs=points_fc.epsg).scatter(
            points_fc, size_column="fid", size_limits=(20, 200)
        )
        assert np.allclose(deprecated.get_sizes(), renamed.get_sizes()), (
            "the deprecated spelling no longer does what the new one does"
        )

    def test_point_size_alias_warns_and_still_works(self, points_fc):
        """The deprecated ``point_size=`` keeps working and names ``size``.

        Args:
            points_fc: The committed point fixture.

        Test scenario:
            cleopatra's own spelling reaches the markers, with a warning pointing at ``size``.
        """
        m = Map(crs=points_fc.epsg)
        with pytest.warns(
            DeprecationWarning, match=r"point_size= is deprecated.*use size="
        ):
            pc = m.scatter(points_fc, point_size=77)
        assert set(np.asarray(pc.get_sizes()).tolist()) == {77.0}, (
            "point_size= no longer sets the size"
        )

    @pytest.mark.parametrize(
        "kwargs, new_name, old_name",
        [
            ({"size_column": "fid", "scale": "fid"}, "size_column", "scale"),
            ({"size": 20, "point_size": 77}, "size", "point_size"),
        ],
    )
    def test_both_spellings_at_once_is_refused(
        self, points_fc, kwargs, new_name, old_name
    ):
        """Passing the old and new spelling together is a ``TypeError``, not a silent preference.

        Args:
            points_fc: The committed point fixture.
            kwargs: The contradictory call, one renamed parameter per case.
            new_name: The spelling the caller should keep.
            old_name: The deprecated spelling they should drop.

        Test scenario:
            They are one parameter, so two values for it cannot both be honoured — and this is the one
            answer all four tiers give, from the shared
            :func:`~digitalearth.base.deprecation.renamed_parameter`. The message has to name both
            spellings and the one to keep, or the caller cannot tell which of their two keywords to delete.
        """
        m = Map(crs=points_fc.epsg)
        with pytest.raises(TypeError) as excinfo:
            m.scatter(points_fc, **kwargs)
        message = str(excinfo.value)
        assert f"{new_name}=" in message, (
            f"the error must name the spelling to keep ({new_name}=), got {message!r}"
        )
        assert f"{old_name}=" in message, (
            f"the error must name the spelling to drop ({old_name}=), got {message!r}"
        )
        assert f"pass only {new_name}=" in message, message


#: Every static builder that resolves a deprecated spelling, called the way a user writes it. The list is the
#: whole class, not a sample: `scatter` is wrapped by `_skips_off_limb`, `grid_points` is not, and
#: `point_cloud` delegates to `grid_points`, so each sits a different number of frames from the caller.
DEPRECATED_SPELLINGS = {
    # Called under the Core spelling `points`, not the deprecated `scatter`: this table is about a
    # deprecated **parameter**'s stacklevel, and going through the method alias as well would raise a second
    # warning that has nothing to do with what is being measured. The alias is asked the same question, with
    # both warnings live, by `TestADeprecatedParameterOnADeprecatedMethod` below.
    "points(point_size=)": lambda fc, ds: Map(crs=fc.epsg).points(fc, point_size=5),
    "points(scale=)": lambda fc, ds: Map(crs=fc.epsg).points(fc, scale="fid"),
    "grid_points(point_size=)": lambda fc, ds: Map(crs=ds.epsg).grid_points(
        ds, point_size=5
    ),
    "point_cloud(point_size=)": lambda fc, ds: Map(crs=ds.epsg).point_cloud(
        ds, point_size=5
    ),
}


class TestADeprecationWarningPointsAtTheCaller:
    """A deprecation warning is only seen when it lands on the caller's line.

    Python's default filters show a ``DeprecationWarning`` only when it is attributed to ``__main__``, so a
    warning attributed to a line inside this package is one a user never sees at all.
    """

    @pytest.mark.parametrize("spelling", sorted(DEPRECATED_SPELLINGS))
    def test_the_warning_names_this_file(self, spelling, points_fc, dataset):
        """The ``stacklevel`` counted for each builder must reach the frame that called it.

        Args:
            spelling: The entry in :data:`DEPRECATED_SPELLINGS` under test.
            points_fc: The committed point fixture.
            dataset: The committed raster fixture.

        Test scenario:
            The call is made from this file, so the warning's ``filename`` must be this file. A decorator
            between the caller and the builder adds a frame; a count that forgets it blames the wrapper.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DEPRECATED_SPELLINGS[spelling](points_fc, dataset)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1, [str(w.message) for w in deprecations]
        warned = deprecations[0]
        assert warned.filename == __file__, f"{warned.filename}:{warned.lineno}"


class TestADeprecatedParameterOnADeprecatedMethod:
    """Both warnings have to land on the caller's line, and one of them is counted through the other.

    A caller with an old script writes `Map.scatter(fc, point_size=5)`, and order 27a made *both* halves of
    that deprecated: the method is an alias for `points` now, and `point_size=` was already an alias for
    `size=`. The alias forwards, which puts a frame between the caller and the builder that resolves the
    parameter — so a `stacklevel` counted for a direct call would blame `base/deprecation.py` instead of the
    user. `_ALIAS_DEPTH` is what closes that, and until this rename no static builder exercised it.
    """

    def test_both_warnings_name_the_callers_file(self, points_fc):
        """Two deprecations, one call, and both must point at the line that made it.

        Args:
            points_fc: The committed point fixture.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Map(crs=points_fc.epsg).scatter(points_fc, point_size=5)
        blamed = sorted(
            {w.filename for w in caught if issubclass(w.category, DeprecationWarning)}
        )
        assert blamed == [__file__], blamed

    def test_the_two_warnings_name_the_two_spellings_to_change(self, points_fc):
        """And a reader has to be told about both, not just the outer one.

        Args:
            points_fc: The committed point fixture.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Map(crs=points_fc.epsg).scatter(points_fc, point_size=5)
        said = sorted(
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        )
        assert [text.split(" is deprecated")[0] for text in said] == [
            "Map.points(): point_size=",
            "Map.scatter()",
        ], said


class TestClassification:
    """C4 on the static tier — the builders here that classify, and the legend they key."""

    def test_signature_declares_scheme_and_k(self):
        """choropleth declares the contract's parameters and defaults.

        Test scenario:
            ``scheme`` and ``k`` were only reachable through ``**opts`` before, so neither the default nor
            the meaning of ``None`` was stated anywhere a caller could read.
        """
        params = inspect.signature(Map.choropleth).parameters
        assert params["scheme"].default is None, (
            "scheme must default to a continuous ramp"
        )
        assert params["k"].default == 5, "k must default to 5 classes"

    def test_scheme_none_is_continuous(self, polygons_fc):
        """Without a scheme the fill stays a continuous ramp.

        Args:
            polygons_fc: Buffered polygons with a numeric ``fid``.

        Test scenario:
            A continuous norm, not a class-boundary one, backs the default render.
        """
        pc = Map(crs=polygons_fc.epsg).choropleth(polygons_fc, column="fid")
        assert not isinstance(pc.norm, BoundaryNorm), (
            "the default render must not classify"
        )

    @pytest.mark.parametrize("k", [3, 5])
    def test_scheme_with_k_cuts_that_many_classes(self, polygons_fc, k):
        """A named scheme is graduated into exactly ``k`` classes.

        Args:
            polygons_fc: Buffered polygons with a numeric ``fid``.
            k: Requested number of classes.

        Test scenario:
            ``k`` reaches the classifier, so the boundary norm carries ``k + 1`` edges.
        """
        pc = Map(crs=polygons_fc.epsg).choropleth(
            polygons_fc, column="fid", scheme="quantiles", k=k
        )
        assert isinstance(pc.norm, BoundaryNorm), "a named scheme must classify"
        assert len(pc.norm.boundaries) == k + 1, (
            f"k={k} produced {len(pc.norm.boundaries) - 1} classes"
        )

    def test_categorical_keeps_its_meaning(self, polygons_fc):
        """``scheme="categorical"`` still gives every distinct value its own colour.

        Args:
            polygons_fc: Buffered polygons; a nominal ``zone`` column is added here.

        Test scenario:
            The contract changes the signature, not what the schemes compute.
        """
        polygons_fc["zone"] = ["urban", "rural"] * (len(polygons_fc) // 2) + [
            "urban"
        ] * (len(polygons_fc) % 2)
        m = Map(crs=polygons_fc.epsg)
        pc = m.choropleth(polygons_fc, column="zone", scheme="categorical")
        assert isinstance(pc.norm, BoundaryNorm), (
            "a categorical fill maps discrete class codes"
        )
        labels = [t.get_text() for t in m.layers[-1][0].category_legend.get_texts()]
        assert labels == ["rural", "urban"], (
            f"the swatch legend lost its labels: {labels}"
        )


class TestColormapResolution:
    """C5 on the static tier — every raster builder, with the tier's own literal behind the lookup."""

    @pytest.mark.parametrize(
        "method", ["imshow", "contourf", "contour", "pcolormesh", "block"]
    )
    def test_no_raster_builder_defaults_to_a_literal(self, method):
        """No raster builder names a colormap in its signature.

        Args:
            method: The raster field method under test.

        Test scenario:
            A bare string default would decide the colour before the data's own metadata could.
        """
        params = inspect.signature(getattr(Map, method)).parameters
        assert "cmap" not in params or params["cmap"].default is None, (
            f"{method} hard-codes a colormap default"
        )

    def test_cmap_none_takes_the_resolved_colormap(self, dataset, monkeypatch):
        """With ``cmap=None`` the variable's own colormap is what renders.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            The lookup's answer reaches the glyph rather than matplotlib's default.
        """
        _styled(monkeypatch, cmap="magma")
        im = Map(crs=dataset.epsg).imshow(dataset)
        assert im.get_cmap().name == "magma", (
            "the resolved colormap did not reach the render"
        )

    def test_caller_cmap_wins_over_the_lookup(self, dataset, monkeypatch):
        """An explicit ``cmap`` is never overridden by the lookup.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            The lookup offers ``magma``; the caller asked for ``plasma`` and gets it.
        """
        _styled(monkeypatch, cmap="magma")
        im = Map(crs=dataset.epsg).imshow(dataset, cmap="plasma")
        assert im.get_cmap().name == "plasma", "the lookup overrode an explicit cmap"

    def test_literal_sits_behind_the_lookup(self, dataset, monkeypatch):
        """When the lookup has no opinion the tier's literal answers.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Makes the style lookup answer with nothing.

        Test scenario:
            The default is still deterministic — it just lives behind the lookup instead of in a signature.
        """
        _styled(monkeypatch)
        im = Map(crs=dataset.epsg).imshow(dataset)
        assert im.get_cmap().name == DEFAULT_FIELD_CMAP, (
            "the fallback colormap was not applied"
        )

    def test_stock_img_resolves_its_backdrop_colormap(self, dataset, monkeypatch):
        """The backdrop resolves through the same lookup, with its own literal behind it.

        Args:
            dataset: The committed ``acc4000`` raster, used as a backdrop.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            A recognised backdrop takes the variable's colormap; an unrecognised one falls back to the
            hypsometric literal ``stock_img`` used to hard-code.
        """
        assert inspect.signature(Map.stock_img).parameters["cmap"].default is None, (
            "stock_img still hard-codes a colormap default"
        )
        _styled(monkeypatch, cmap="terrain")
        assert Map(crs=dataset.epsg).stock_img(dataset).get_cmap().name == "terrain", (
            "the backdrop ignored the resolved colormap"
        )
        _styled(monkeypatch)
        assert (
            Map(crs=dataset.epsg).stock_img(dataset).get_cmap().name == STOCK_IMG_CMAP
        ), "the backdrop's own literal is no longer behind the lookup"


class TestAutoStyleLevelsAndUnits:
    """C6 on the static tier — where the lookup's ``levels`` and ``units`` land once they are read."""

    def test_levels_fill_in_for_a_contour_render(self, dataset, monkeypatch):
        """A contour render takes the variable's canonical levels when the caller passed none.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            The resolved edges are the ones the contour set is built on.
        """
        _styled(monkeypatch, cmap="viridis", levels=[0.0, 5.0, 10.0, 20.0])
        cs = Map(crs=dataset.epsg).contourf(dataset)
        assert np.allclose(cs.levels, [0.0, 5.0, 10.0, 20.0]), (
            f"the resolved levels were ignored: {cs.levels}"
        )

    def test_caller_levels_win(self, dataset, monkeypatch):
        """Explicit ``levels`` are never replaced by the lookup's.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            The caller asks for four even bands over a range the lookup does not describe.
        """
        _styled(monkeypatch, cmap="viridis", levels=[0.0, 5.0, 10.0, 20.0])
        cs = Map(crs=dataset.epsg).contourf(dataset, levels=[0.0, 1.0, 2.0])
        assert np.allclose(cs.levels, [0.0, 1.0, 2.0]), (
            "the lookup overrode explicit levels"
        )

    def test_levels_do_not_band_a_continuous_render(self, dataset, monkeypatch):
        """An ``imshow`` stays continuous even when the variable has canonical contour levels.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            Contour levels describe a contour render; silently banding a pixel grid would change what the
            caller asked to see.
        """
        _styled(monkeypatch, cmap="viridis", levels=[0.0, 5.0, 10.0, 20.0])
        im = Map(crs=dataset.epsg).imshow(dataset)
        assert not isinstance(im.norm, BoundaryNorm), (
            "imshow was banded by the resolved levels"
        )

    def test_units_label_the_colorbar(self, dataset, monkeypatch):
        """The colorbar labels itself with the variable's units when the caller passes none.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            ``Map.colorbar()`` with no label picks up the layer's resolved units.
        """
        _styled(monkeypatch, cmap="viridis", units="m3/s")
        m = Map(crs=dataset.epsg)
        m.imshow(dataset)
        assert m.colorbar().ax.get_ylabel() == "m3/s", (
            "the resolved units did not label the bar"
        )

    def test_caller_label_wins(self, dataset, monkeypatch):
        """An explicit colorbar label is never replaced by the units.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            The caller's wording survives a variable that also has units.
        """
        _styled(monkeypatch, cmap="viridis", units="m3/s")
        m = Map(crs=dataset.epsg)
        m.imshow(dataset)
        assert m.colorbar(label="discharge").ax.get_ylabel() == "discharge", (
            "the units overrode an explicit label"
        )

    def test_unitless_layer_is_labelled_as_before(self, dataset, monkeypatch):
        """A variable with no units leaves the bar unlabelled, rather than guessing one.

        Args:
            dataset: The committed ``acc4000`` raster.
            monkeypatch: Makes the style lookup answer without units.

        Test scenario:
            "Do not guess a unit" — absent units must change nothing.
        """
        _styled(monkeypatch, cmap="viridis")
        m = Map(crs=dataset.epsg)
        m.imshow(dataset)
        assert m.colorbar().ax.get_ylabel() == "", "an unlabelled bar was given a label"

    def test_animation_colorbar_follows_the_same_style(self, global_field, monkeypatch):
        """The animation's one persistent colorbar is keyed to the frames' colormap and units.

        Args:
            global_field: A whole-world raster animated twice over.
            monkeypatch: Pins what the style lookup answers.

        Test scenario:
            The bar is built once, before any frame is drawn, so it has to resolve the same style the
            frames will — otherwise a coolwarm animation gets a viridis key.
        """
        _styled(monkeypatch, cmap="magma", units="m3/s")
        monkeypatch.setattr(
            animation_module,
            "auto_style",
            lambda source: {"cmap": "magma", "units": "m3/s"},
        )
        m = Map(crs=4326)
        m.animate([global_field, global_field], colorbar=True)
        bar = m.fig.axes[-1]
        assert bar.get_ylabel() == "m3/s", (
            "the animation colorbar ignored the resolved units"
        )


class TestStrictOffLimb:
    """C7 on the static tier — off-limb data, an empty geometry set, and the ``strict`` dial over both.

    The clause's other half, a kind this tier does not draw at all, is ``terrain`` here; it is pinned for
    every tier in ``tests/base/test_custom_layers.py::TestWhatC7DoesNotCover`` rather than again per tier.
    """

    def test_strict_defaults_to_false(self):
        """Scenes are lenient unless asked otherwise.

        Test scenario:
            The flag exists on the scene constructor and starts ``False`` on both scene classes.
        """
        assert Scene().strict is False, "a bare Scene must default to lenient"
        assert Map(crs=4326).strict is False, "a Map must default to lenient"

    def test_lenient_skips_and_warns(self, regional, caplog):
        """By default an off-limb layer draws nothing, logs a warning, and the render carries on.

        Args:
            regional: A raster that lands outside the display CRS.
            caplog: Captures the warning the skip logs.

        Test scenario:
            On an unclipped projection the skip is worth a warning, since the only other symptom is a
            blank figure.
        """
        m = Map(crs=projections.orthographic(lon=-175, lat=15))
        with caplog.at_level(logging.WARNING):
            assert m.imshow(regional) is None, "an off-limb layer must draw nothing"
        assert "imshow" in caplog.text, f"the skip was not reported: {caplog.text!r}"

    def test_strict_raises_naming_the_layer(self, regional):
        """With ``strict=True`` the same layer raises instead of being skipped.

        Args:
            regional: A raster that lands outside the display CRS.

        Test scenario:
            The error names the layer that would have been dropped, so a pipeline fails where the problem
            is rather than at the empty figure.
        """
        m = Map(crs=projections.orthographic(lon=-175, lat=15), globe=True, strict=True)
        with pytest.raises(OffLimbError, match="imshow"):
            m.imshow(regional)

    def test_strict_reaches_vector_layers_too(self, regional):
        """The strict flag is honoured by every layer, not just the raster fields.

        Args:
            regional: A raster that lands outside the display CRS.

        Test scenario:
            ``grid_points`` funnels through the same skip, so it raises under strict as well.
        """
        m = Map(crs=projections.orthographic(lon=-175, lat=15), globe=True, strict=True)
        with pytest.raises(OffLimbError, match="grid_points"):
            m.grid_points(regional)


class TestNaturalEarthSurface:
    """C12 on the static tier — the six named methods here, which are the surface the others align to."""

    @pytest.mark.parametrize("method, layer", sorted(NATURAL_EARTH_METHODS.items()))
    def test_method_draws_its_own_layer(self, method, layer, mocker):
        """Each named method delegates to the shared reference-layer helper with its own layer.

        Args:
            method: The public method under test.
            layer: The Natural-Earth layer it must request.
            mocker: Stubs the shared helper so no geometry is drawn.

        Test scenario:
            The six names other tiers are being aligned to exist here and each maps to one layer.
        """
        m = Map(crs=4326)
        stub = mocker.patch.object(Map, "_natural_earth", return_value=None)
        getattr(m, method)()
        assert stub.call_args.args[0] == layer, (
            f"{method}() drew {stub.call_args.args[0]!r} instead of {layer!r}"
        )


class TestAnimationState:
    """C14 on the static tier — the animation rate ``Map`` carries between a build and a save."""

    def test_declared_on_a_fresh_map(self):
        """A Map has the attribute before anything is animated.

        Test scenario:
            ``save_animation`` reads it directly, so it must exist from construction.
        """
        m = Map(crs=4326)
        assert "_animation_fps" in vars(m), (
            "the animation rate is not declared in __init__"
        )
        assert m._animation_fps is None, "an un-animated map has no rate yet"

    def test_recorded_when_an_animation_is_built(self, global_field):
        """Building an animation records the rate it was built at.

        Args:
            global_field: A whole-world raster to animate.

        Test scenario:
            ``save_animation`` defaults to this rate, so the file matches what was previewed.
        """
        m = Map(crs=4326)
        m.animate([global_field, global_field], fps=6)
        assert m._animation_fps == 6.0, "the built rate was not recorded"


def test_no_deprecation_warning_on_the_modern_spellings(points_fc, recwarn):
    """The renamed parameters are silent — only the old spellings warn.

    Args:
        points_fc: The committed point fixture.
        recwarn: Records every warning the calls emit.

    Test scenario:
        A deprecation that fires for callers who already migrated is noise, and trains people to filter the
        warning that matters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        Map(crs=points_fc.epsg).points(points_fc, size_column="fid", size=30)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)], (
        "the modern spellings must not warn"
    )


#: Every public data builder on `Map` and the argument that names what it draws. A path or URL is accepted
#: wherever a pyramids object is — and a path-backed layer is the only kind a figure can be written from —
#: so the argument's own documentation has to say so (round 2, L8). `spaghetti` is deliberately absent: it
#: takes a `DatasetCollection`, which is a set of rasters rather than one file, and refuses a path.
#:
#: The three Core spellings adopted at order 27a are listed under those names — `field`, `points`,
#: `polygons` — because that is where the documentation lives. Their deprecated aliases (`imshow`,
#: `scatter`, `shapes`) carry the one-line docstring `renamed_method` generates, whose whole job is to name
#: the replacement; documenting the data argument twice is what the rename exists to stop.
PATH_TAKING_BUILDERS = {
    "field": "dataset",
    "contour": "dataset",
    "contourf": "dataset",
    "pcolormesh": "dataset",
    "block": "dataset",
    "rgb_composite": "dataset",
    "hsv_composite": "dataset",
    "grid_points": "dataset",
    "point_cloud": "dataset",
    "grid_cells": "dataset",
    "stock_img": "dataset",
    "quiver": "u_dataset",
    "barbs": "u_dataset",
    "streamplot": "u_dataset",
    "tricontour": "data",
    "tricontourf": "data",
    "tripcolor": "data",
    "points": "features",
    "polygons": "features",
    "choropleth": "features",
    "voronoi": "features",
    "cartogram": "features",
    "quadtree": "features",
    "kde": "features",
    "sankey": "features",
}


def _argument_paragraph(builder: str, argument: str) -> str:
    """Return the ``Args:`` entry one builder writes for one argument.

    Args:
        builder: The method name on `Map`.
        argument: The argument whose entry is wanted.

    Returns:
        The entry's text as one line, from its name up to the next entry, or ``""`` when the builder
        documents no such argument. Continuation lines are joined, so a phrase the wrapping broke across
        two lines still reads as one.
    """
    doc = inspect.getdoc(getattr(Map, builder)) or ""
    block = re.split(r"\n(?=\S)", doc.split("Args:", 1)[-1])[0]
    for entry in re.split(r"\n(?=    \S)", block):
        if entry.strip().startswith(f"{argument}:"):
            return " ".join(entry.split())
    return ""


class TestEveryDataBuilderDocumentsThePathItTakes:
    """A caller following the per-builder documentation could never produce a figure that writes.

    Every data builder takes a path or a URL as readily as a pyramids object, and only a path-backed layer
    can be written down — an in-memory one is an ``object:`` reference `FigureSpec.to_dict` refuses. None of
    the builders said so, and the module docstrings that did are not what a caller reads (round 2, L8).
    """

    @pytest.mark.parametrize("builder", sorted(PATH_TAKING_BUILDERS))
    def test_the_data_argument_says_a_path_is_accepted(self, builder):
        """Args:
        builder: The entry in :data:`PATH_TAKING_BUILDERS` under test.
        """
        entry = _argument_paragraph(builder, PATH_TAKING_BUILDERS[builder])
        assert "path or URL" in entry, entry

    @pytest.mark.parametrize("builder", sorted(PATH_TAKING_BUILDERS))
    def test_the_data_argument_says_why_that_matters(self, builder):
        """Args:
        builder: The entry in :data:`PATH_TAKING_BUILDERS` under test.
        """
        entry = _argument_paragraph(builder, PATH_TAKING_BUILDERS[builder])
        assert "written down" in entry, entry
