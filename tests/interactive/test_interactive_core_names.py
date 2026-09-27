"""The interactive tier's three Core spellings (order 27a).

``field``, ``lines`` and ``add_layer`` were agreed and unadopted from the day the
Core contract was frozen. Order 27a adopts them, and the spellings they replaced are **deleted** rather than
aliased: nothing here is released, so there is no caller to keep a promise to.

`tests/test_contract_names.py` holds the *shape* of that off the class, without calling anything. What it
cannot ask is whether the Core name still **builds**, and whether it records the recipe key the drawer table
is keyed by — a builder whose recipe moved with its name would satisfy every check there and put a figure on
the page that cannot be read back. So each is built here for real and compared against the recipe it must
record.

``add_layer`` is the interesting one of the three: every builder on this tier ends at it, so a rename that
missed one internal call site would leave that builder reaching a name nothing answers to.
`TestEveryBuilderReachesTheCoreFunnel` is what asks that, by drawing through unrelated builders.
"""

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection
from shapely.geometry import LineString, Point

from digitalearth.interactive import InteractiveMap

gv = pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)

#: The spelling each Core name replaced, so the check that it is gone reads from one place.
REPLACED = {"field": "image", "lines": "path", "add_layer": "add_element"}

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
    """Build one layer and report the element and the description.

    Args:
        spelling: The method to call.
        core: The Core name, which chooses the data.

    Returns:
        `(element type name, layer kind, recipe key, layer count)`. The element's *type* rather than the
        element, because what has to be right is what kind of thing HoloViews was handed.
    """
    scene = InteractiveMap()
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
    """The Core names build, and record what the drawer table is keyed by."""

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

    @pytest.mark.parametrize("core", sorted(REPLACED))
    def test_the_spelling_the_core_name_replaced_is_gone(self, core):
        """The old spelling is deleted, not kept as a second name.

        Args:
            core: The Core name whose predecessor must be absent.

        Test scenario:
            A method left behind under both names would pass every other check here and go on teaching the
            vocabulary the contract exists to end.
        """
        assert not hasattr(InteractiveMap, REPLACED[core]), (
            f"InteractiveMap still answers to {REPLACED[core]!r}, the spelling {core}() replaced"
        )


class TestEveryBuilderReachesTheCoreFunnel:
    """Every builder on this tier ends at `add_layer`, and ~50 call sites had to move with the name.

    A single missed `self.add_layer(...)` would reach a name nothing answers to, and the only symptom is an
    ``AttributeError`` raised from inside the package on a builder that has nothing to do with the rename.
    Drawing through those builders is what catches that, and it catches it for every funnel call the drawn
    path reaches rather than for the three renamed methods alone.
    """

    @pytest.mark.parametrize("draw", ("points", "graticule", "field"))
    def test_an_untouched_builder_still_registers_its_layer(self, draw):
        """A builder nobody renamed must still reach the funnel and come back with a layer.

        Args:
            draw: The builder to exercise.
        """
        scene = InteractiveMap()
        if draw == "graticule":
            scene.graticule(lon_step=30.0, lat_step=30.0)
        elif draw == "points":
            scene.points(_point_features())
        else:
            scene.field(_raster())
        assert list(scene.layer_ids), (
            f"InteractiveMap.{draw}() registered no layer, so it did not reach add_layer()"
        )
