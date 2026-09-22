"""What the 3-D tier declares, held against what it actually draws (U-1, #294).

`tests/three_d/test_seam3d.py::TestTheDeclaration` already covers the row's identity: whose it is, that it is
a :class:`~digitalearth.base.capabilities.Capabilities`, that it loads without PyVista, that `api.py` derives
its row from it, and that a `quickmap` refusal carries its reason. What it does not cover is whether the row
is *true* — and a declaration that claims something nothing implements is worse than no declaration
(`planning/refactor/backends/api-unification.md` §2, "`hasattr` capability lies").

These are the three claims, each read off what the tier does rather than off a second table beside it:

* **every declared kind has a builder that records it** — the builder is called, and the kind it writes into
  the figure is what is compared. `test_seam3d.py` checks that every declared kind resolves to a *drawer*;
  a drawer with no builder to feed it is still a kind no caller can ask for by name.
* **every declared channel reaches the layer** — this tier folds no `Encoding`: a channel arrives as a
  keyword and is stored in `Symbology.props` for the drawer to hand PyVista, so that is where it is read
  back from. The `data_driven` pair are read back as a column or an array reference instead of a constant.
* **the schemes are the shared classifier's**, which is contract C4.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

import geopandas as gpd  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from digitalearth.base.spec import Scale  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402
from digitalearth.three_d.capabilities import CAPABILITIES  # noqa: E402
from digitalearth.three_d.point_cloud import SCALAR  # noqa: E402
from digitalearth.three_d.renderer import DRAWN_KINDS, drawer_for  # noqa: E402

#: The committed raster the surface builders are given. Read from the repo root, as the rest of the suite does.
DEM_PATH = "examples/data/acc4000.tif"


def _cloud() -> np.ndarray:
    """Return three xyz points.

    Returns:
        The array every point-cloud call below is given.
    """
    return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.5, 0.2]])


def _four_points() -> np.ndarray:
    """Return four xyz points, one per value the classification check colours.

    Returns:
        The array the classified point-cloud call is given.
    """
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.5, 0.2], [3.0, 1.5, 0.4]]
    )


def _cube() -> np.ndarray:
    """Return a small 3-D scalar field.

    Returns:
        The array `volume` and `isosurface` are given.
    """
    return np.random.default_rng(3).random((6, 6, 6))


def _polygons() -> gpd.GeoDataFrame:
    """Return two polygons with a numeric column and a height column.

    Returns:
        The collection `extruded_polygons` is given.
    """
    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0], "storeys": [10.0, 20.0]},
        geometry=[
            Polygon([(4.0, 52.0), (5.0, 52.0), (5.0, 53.0), (4.0, 53.0)]),
            Polygon([(5.0, 52.0), (6.0, 52.0), (6.0, 53.0), (5.0, 53.0)]),
        ],
        crs=4326,
    )


def _dem():
    """Return the committed raster.

    Returns:
        A pyramids `Dataset`.
    """
    from pyramids.dataset import Dataset

    return Dataset.read_file(DEM_PATH)


#: Declared kind -> the builder call that records it. Every kind the row claims is here, so a kind added to
#: the declaration without a builder has no entry and is caught by the coverage check below — and a builder
#: that stops recording its kind fails on the call rather than on a list.
#:
#: `globe` is the only one that records two: it draws the raster onto the sphere and puts the shoreline on it
#: in the same call, which is why the tier declares a `coastlines` kind while having no standalone coastline
#: builder (review M6 on #295).
BUILDERS = {
    "terrain": lambda scene: scene.terrain(_dem()),
    "point_cloud": lambda scene: scene.point_cloud(_cloud()),
    "volume": lambda scene: scene.volume(_cube()),
    "isosurface": lambda scene: scene.isosurface(_cube(), isosurfaces=[0.5]),
    "vectors": lambda scene: scene.vectors(np.zeros((2, 3)), np.ones((2, 3))),
    "extrusion": lambda scene: scene.extruded_polygons(_polygons(), height=10.0),
    "raster": lambda scene: scene.globe(_dem()),
    "coastlines": lambda scene: scene.globe(_dem()),
    "custom:pyvista": lambda scene: scene.add_mesh(pv.Sphere()),
}


@pytest.fixture
def scene():
    """Yield an off-screen scene, closed on the way out.

    Yields:
        The scene under test.
    """
    built = Scene3D(off_screen=True)
    yield built
    built.close()


def _props(scene, build) -> dict:
    """Return the properties the last layer a builder call records carries.

    Args:
        scene: The scene to build on.
        build: A callable taking the scene and calling one builder on it.

    Returns:
        The last layer's `Symbology.props`, which is where this tier keeps the engine keywords a channel
        arrived as.
    """
    build(scene)
    return dict(scene.figure_spec.layers.get(scene.layer_ids[-1]).symbology.props)


class TestEveryDeclaredKindIsBuilt:
    """A drawer is half the answer; a builder that records the kind is the other half."""

    def test_every_declared_kind_has_a_builder_here(self):
        """The table above has to name every declared kind, or a claim goes unexercised."""
        difference = sorted(set(BUILDERS).symmetric_difference(CAPABILITIES.kinds))
        assert difference == [], (
            f"{difference} are declared with no builder call, or called without being declared"
        )

    def test_the_declaration_and_the_drawer_table_are_the_same_list(self):
        """This tier draws everything it declares from a description, so the two lists are one.

        Test scenario:
            The interactive tier's `custom:holoviews` is the counter-example — declared and deliberately not
            in `DRAWN_KINDS`, because a caller's own element has no description to rebuild it from. Here a
            caller's own mesh *is* redrawn from its description, so nothing is outside the drawer table, and
            asserting the stronger form is what makes a kind that quietly drops out of one list fail.
        """
        difference = sorted(CAPABILITIES.kinds.symmetric_difference(DRAWN_KINDS))
        assert difference == [], f"{difference} is in one list and not the other"

    @pytest.mark.parametrize("kind", sorted(BUILDERS))
    def test_a_builder_records_the_declared_kind(self, scene, kind):
        """Calling the builder leaves a layer of that kind in the figure.

        Args:
            scene: The scene under test.
            kind: The declared kind being built.
        """
        BUILDERS[kind](scene)
        recorded = [scene.figure_spec.layers.get(name).kind for name in scene.layer_ids]
        assert kind in recorded, f"{kind} was declared but {recorded} was recorded"


class TestEveryDeclaredChannelReachesTheLayer:
    """This tier folds no encoding, so a channel has to arrive as a property the drawer can read."""

    #: Channel -> (the builder call that sets it, the property it lands under, the value asked for).
    CARRIED = {
        "color": (
            lambda scene: scene.point_cloud(_cloud(), cmap="magma"),
            "cmap",
            "magma",
        ),
        "opacity": (
            lambda scene: scene.point_cloud(_cloud(), opacity=0.4),
            "opacity",
            0.4,
        ),
        "size": (lambda scene: scene.point_cloud(_cloud(), size=9.0), "size", 9.0),
        "height": (
            lambda scene: scene.extruded_polygons(_polygons(), height=42.0),
            "height",
            42.0,
        ),
    }

    def test_every_declared_channel_is_covered_here(self):
        """The table has to name every channel the row claims."""
        difference = sorted(
            set(self.CARRIED).symmetric_difference(CAPABILITIES.channels)
        )
        assert difference == [], (
            f"{difference} are declared and unchecked, or checked and undeclared"
        )

    @pytest.mark.parametrize("channel", sorted(CARRIED))
    def test_a_channel_set_from_a_keyword_reaches_the_recorded_properties(
        self, scene, channel
    ):
        """What the caller asked for is what the layer records for the drawer to hand PyVista.

        Args:
            scene: The scene under test.
            channel: The declared channel under test.
        """
        build, key, asked = self.CARRIED[channel]
        assert _props(scene, build)[key] == asked, (channel, key)


class TestOnlyTheDataDrivenChannelsTakeAField:
    """`data_driven` is the narrower claim: which channels a *column* can drive, not just a constant."""

    def test_a_data_driven_colour_takes_a_column(self, scene):
        """`column=` colours each prism by its own value rather than by one chosen colour."""
        recorded = _props(
            scene,
            lambda built: built.extruded_polygons(
                _polygons(), column="value", scheme="quantiles", k=2
            ),
        )
        assert recorded["column"] == "value", recorded

    def test_a_data_driven_colour_takes_an_array_of_values(self, scene):
        """The other spelling: a point cloud is coloured by an array as long as the cloud.

        Test scenario:
            The array is held by reference rather than written into the description — a cloud's values are
            hundreds of thousands of numbers — so what is recorded is a reference, and the check is that the
            slot was filled at all rather than left at `None`.
        """
        recorded = _props(
            scene,
            lambda built: built.point_cloud(_cloud(), values=np.array([1.0, 2.0, 3.0])),
        )
        assert recorded["values"] is not None, recorded

    def test_a_data_driven_height_takes_a_column(self, scene):
        """`height="storeys"` extrudes each polygon by its own value."""
        recorded = _props(
            scene, lambda built: built.extruded_polygons(_polygons(), height="storeys")
        )
        assert recorded["height"] == "storeys", recorded

    @pytest.mark.parametrize(
        ("channel", "key", "asked"),
        [("opacity", "opacity", 0.4), ("size", "size", 9.0)],
    )
    def test_a_channel_outside_data_driven_records_a_constant(
        self, scene, channel, key, asked
    ):
        """The narrower claim has to be narrower: these two take a number and nothing else.

        Args:
            scene: The scene under test.
            channel: The declared channel under test.
            key: The property it lands under.
            asked: The constant the builder was given.

        Test scenario:
            A channel a column could drive would record the column's name here. Declaring one of these as
            `data_driven` without giving its builder a `column=` is the lie this catches.
        """
        assert channel not in CAPABILITIES.data_driven, sorted(CAPABILITIES.data_driven)
        recorded = _props(
            scene, lambda built: built.point_cloud(_cloud(), **{channel: asked})
        )
        assert recorded[key] == asked, (channel, recorded)


class TestTheSchemesAreTheSharedClassifiers:
    """Contract C4 — one `scheme`/`k` pair cuts the same classes here as on every other tier."""

    @pytest.mark.parametrize("scheme", sorted(CAPABILITIES.schemes - {"categorical"}))
    def test_every_declared_scheme_cuts_classes(self, scheme):
        """A scheme the shared classifier does not know would be a name nothing accepts.

        Args:
            scheme: The declared scheme under test.
        """
        assert len(Scale.breaks_of(list(range(100)), scheme, 3)) >= 2, scheme

    def test_the_nominal_scheme_is_declared(self):
        """`categorical` is not a classifier scheme; it is the nominal path, and C4 pins its meaning."""
        assert "categorical" in CAPABILITIES.schemes, sorted(CAPABILITIES.schemes)

    def test_a_classified_layer_cuts_the_number_of_classes_it_was_asked_for(
        self, scene
    ):
        """C4's own shape: `scheme` + `k` reach the classifier rather than being swallowed.

        Args:
            scene: The scene under test.

        Test scenario:
            Reading `scheme` and `k` back off the description asked nothing: they are the keywords the call
            had just passed, so the check passed with the classifier itself stubbed out (review L7). What is
            read now is what was **drawn** — VTK has no class breaks, so a classified layer reaches the
            engine as per-point class *indices* — against the classes the shared classifier's own edges put
            those values in. The four values are chosen so that the schemes disagree: quantiles split them
            at the median, an equal-interval cut of the same `k` would not.
        """
        values = np.array([1.0, 2.0, 3.0, 40.0])
        scene.point_cloud(_four_points(), values=values, scheme="quantiles", k=2)
        drawn = [int(value) for value in scene.mesh_of(scene.layer_ids[-1])[SCALAR]]
        # The class each value falls in according to the shared classifier's edges, computed here rather
        # than taken from the tier: `breaks_of` is what C4 says every tier cuts with.
        interior = list(Scale.breaks_of(values.tolist(), "quantiles", 2))[1:-1]
        expected = [
            int(np.searchsorted(interior, value, side="right")) for value in values
        ]
        assert drawn == expected, (drawn, expected, interior)


class TestARefusalCarriesTheDeclaredReason:
    """The kind the tier cannot draw is refused in the declaration's words, not in new ones."""

    def test_a_kind_the_tier_declared_against_says_why(self):
        """`basemap` is declared absent with a reason, and the refusal carries it.

        Test scenario:
            Before #294 closed, the refusal said only that the kind was not drawn. A caller holding a
            figure written for the web tier learned nothing from that; "there are no map tiles to drape
            under a scene drawn in three dimensions" tells them the figure is for another backend.
        """
        with pytest.raises(KeyError, match="no map tiles to drape"):
            drawer_for("basemap")

    def test_a_kind_the_tier_never_spoke_about_gets_no_invented_reason(self):
        """The other half: no declared reason, no sentence written at the call site.

        Test scenario:
            The tier says nothing about `choropleth` — it is simply another tier's kind — so the refusal
            names it and lists what this tier draws, and stops there. If the clause were unconditional it
            would be the hand-written sentence #294 exists to remove, wearing the declaration's clothes.
        """
        assert CAPABILITIES.reason("choropleth") is None, CAPABILITIES.absent
        with pytest.raises(KeyError) as refused:
            drawer_for("choropleth")
        assert "—" not in str(refused.value), str(refused.value)
