"""The interactive tier's three Core spellings, and the three they were adopted from (order 27a).

`image`, `path` and `add_element` sat in ``PLANNED_RENAMES`` — agreed and unadopted — from the day the Core
contract was frozen. Order 27a adopts them: the tier answers to ``field``, ``lines`` and ``add_layer``, and
each old spelling is a live alias that warns and names its replacement.

`tests/test_contract_names.py` holds the *shape* of that off the class, without calling anything. What it
cannot ask is whether the alias still **builds** — an alias forwarding to the wrong builder, or to one whose
recipe key moved with the rename, would satisfy every check there and put a different element on the map. So
each pair below is built twice, once under each spelling, and the two are compared on the element HoloViews
received and on what the figure records.

`add_element` is the interesting one of the three: every builder on this tier ends at it, so a rename that
missed one internal call site would leave that builder emitting a deprecation warning the caller never earned.
`TestNoBuilderReachesTheDeprecatedFunnel` is what asks that, by drawing through unrelated builders and
insisting nothing warns.
"""

import warnings

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection
from shapely.geometry import LineString, Point

from digitalearth.interactive import InteractiveMap

gv = pytest.importorskip("geoviews")

#: The three renames, as `(old spelling, Core spelling)`.
RENAMED = (("image", "field"), ("path", "lines"), ("add_element", "add_layer"))

#: The recipe key each renamed builder records under `symbology.props["via"]`. **Unchanged by the rename**:
#: the drawer table is keyed by these, so a figure written before the rename has to keep reading back into the
#: drawer that made it. Measured — the raster builder still records `"image"`, and the line builder records
#: the tier's shared vector recipe `"geometry"` rather than anything named after the method.
VIA = {"field": "image", "lines": "geometry"}


def _raster():
    """Return a small single-band raster in EPSG:4326.

    Returns:
        A pyramids ``Dataset``.
    """
    return Dataset.from_array(
        np.arange(400, dtype="float32").reshape(20, 20),
        geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
    )


def _line_features():
    """Return one line feature in EPSG:4326.

    Returns:
        A pyramids ``FeatureCollection``.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"pop": [1.0]},
            geometry=[LineString([(4.0, 52.0), (4.2, 52.2)])],
            crs=4326,
        )
    )


def _point_features():
    """Return two point features in EPSG:4326.

    Returns:
        A pyramids ``FeatureCollection``.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"pop": [1.0, 9.0]},
            geometry=[Point(4.1, 52.9), Point(4.2, 52.8)],
            crs=4326,
        )
    )


#: What each Core name is called with, so one parametrised body can build both drawn kinds.
DATA = {"field": _raster, "lines": _line_features}


def _built_by(spelling: str, core: str):
    """Build one layer under one spelling and report the element and the description.

    Args:
        spelling: The method to call — the Core name or the deprecated one.
        core: The Core name, which chooses the data.

    Returns:
        `(element type name, layer kind, recipe key, layer count)`. The element's *type* rather than the
        element, because two maps build two different objects and what has to match is what kind of thing
        HoloViews was handed.
    """
    scene = InteractiveMap()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        getattr(scene, spelling)(DATA[core]())
    figure = scene.figure_spec
    layer = figure.layers.get(figure.layers.ids[-1])
    return (
        type(scene.layers[-1]).__name__,
        layer.kind,
        layer.symbology.props.get("via"),
        len(scene.layers),
    )


class TestTheTierAnswersToItsCoreSpelling:
    """The new names build, and record what the drawer table is keyed by."""

    @pytest.mark.parametrize("core", sorted(VIA))
    def test_the_core_name_builds_and_records_its_recipe(self, core):
        """Each Core name adds one layer and files it under the recipe that built it.

        Args:
            core: The Core name under test.
        """
        _, _, via, count = _built_by(core, core)
        assert (via, count) == (VIA[core], 1), (
            f"InteractiveMap.{core}() recorded via={via!r} and left {count} layers"
        )

    def test_the_core_funnel_registers_a_callers_own_element(self):
        """`add_layer` is the low-level entry point, and it has to take an object a caller built.

        Test scenario:
            The other two renames are builders; this one is the funnel every builder ends at, and its own
            public job is registering a custom layer. Asked with a plain string, which is what the funnel's
            own doctests use — the point is the registration, not the element.
        """
        scene = InteractiveMap()
        scene.add_layer("an-element", name="mine")
        assert list(scene.layer_ids) == ["mine"], scene.layer_ids


class TestTheOldSpellingStillBuildsTheSameLayer:
    """The half a name check cannot reach: the alias has to forward to the builder that draws."""

    @pytest.mark.parametrize("core", sorted(DATA))
    def test_the_alias_builds_what_the_core_name_builds(self, core):
        """Both spellings must hand HoloViews the same kind of element and record the same layer.

        Args:
            core: The Core name under test; its deprecated spelling is looked up from :data:`RENAMED`.
        """
        old = next(spelling for spelling, new in RENAMED if new == core)
        assert _built_by(old, core) == _built_by(core, core), (
            f"InteractiveMap.{old}() built {_built_by(old, core)} and "
            f"InteractiveMap.{core}() built {_built_by(core, core)}"
        )

    def test_the_deprecated_funnel_still_registers_a_layer(self):
        """`add_element` is not a drawn builder, so it is asked its own question.

        Test scenario:
            The alias has to register the layer, under the id the caller asked for, exactly as the Core name
            does — which is what makes it safe for a caller's existing script.
        """
        scene = InteractiveMap()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            scene.add_element("an-element", name="mine")
        assert list(scene.layer_ids) == ["mine"], scene.layer_ids

    @pytest.mark.parametrize(("old", "core"), RENAMED)
    def test_the_alias_warns_and_names_the_core_spelling(self, old, core):
        """The warning has to tell the caller what to write instead.

        Args:
            old: The deprecated spelling.
            core: The Core name it forwards to.
        """
        scene = InteractiveMap()
        alias = getattr(scene, old)
        argument = DATA[core]() if core in DATA else "an-element"
        with pytest.warns(DeprecationWarning) as caught:
            alias(argument)
        said = str(caught[0].message)
        assert f"InteractiveMap.{core}()" in said, (
            f"InteractiveMap.{old}() warned {said!r}"
        )

    @pytest.mark.parametrize(("old", "core"), RENAMED)
    def test_the_warning_points_at_the_callers_own_line(self, old, core):
        """A warning that blames a line inside the package is one nobody can act on.

        Args:
            old: The deprecated spelling.
            core: The Core name it forwards to.
        """
        scene = InteractiveMap()
        alias = getattr(scene, old)
        argument = DATA[core]() if core in DATA else "an-element"
        with pytest.warns(DeprecationWarning) as caught:
            alias(argument)
        assert caught[0].filename == __file__, caught[0].filename


class TestNoBuilderReachesTheDeprecatedFunnel:
    """Every builder on this tier ends at `add_layer`, and ~50 call sites had to move with the name.

    A single missed `self.add_element(...)` would make an unrelated builder warn about a spelling its caller
    never wrote — the "fixed the instance, described the class as fixed" failure. Drawing through builders
    that have nothing to do with the rename and insisting on silence is what catches that, and it catches it
    for every funnel call the drawn path reaches rather than for the three renamed methods alone.
    """

    @pytest.mark.parametrize("draw", ("points", "graticule", "field"))
    def test_an_untouched_builder_emits_no_deprecation(self, draw):
        """A builder nobody renamed must be silent.

        Args:
            draw: The builder to exercise.
        """
        scene = InteractiveMap()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            if draw == "graticule":
                scene.graticule(lon_step=30.0, lat_step=30.0)
            elif draw == "points":
                scene.points(_point_features())
            else:
                scene.field(_raster())
        blamed = [
            str(record.message)
            for record in caught
            if issubclass(record.category, DeprecationWarning)
            and "add_element" in str(record.message)
        ]
        assert blamed == [], (
            f"InteractiveMap.{draw}() still reaches the deprecated funnel: {blamed}"
        )
