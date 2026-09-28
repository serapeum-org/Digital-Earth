"""The 3-D tier's colour key: `colorbar()` and `legend()`, as guides on the layer they explain (order 24).

These were the last two Core names this tier was short of. `contract.PENDING["3d"]` listed both against order
24 — "the scalar bar is PyVista's, and becomes a guide on the encoding", "a keyed list beside a scene" — and a
colour key on this tier was until now whatever PyVista drew of its own accord: titled after the scalars array,
untitleable, and impossible to switch off except by reaching past the tier with `show_scalar_bar=False`.

What the tests below hold is the ordering that changes it. A key is **recorded on the layer's colour encoding
first** and drawn from that record, so four things come for free and are each pinned here: the key follows the
layer through `remove_layer`, `set_visible` and `move_layer`; it survives `Symbology.to_dict()` →
`from_dict()`; the *form* is read off the description (a ramp is a scalar bar, classes are a keyed list); and
the legend's rows are derived from the layer's own `Scale` with the colours read back off the lookup table it
is being drawn through, so a swatch cannot disagree with the picture.

They also pin what PyVista's own slots force, measured on 0.48.4 rather than assumed: **one scalar bar per
title** (a second under the same title binds its mapper to the first bar and shows the first layer's range,
which is refused here by name) and **one legend actor per window** (so the keyed list is built from every
layer that asks for one, rather than the last caller replacing the previous).

Gated on the optional ``3d`` extra (pyvista), as every module in this directory is.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.base.spec import Symbology  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402
from digitalearth.three_d.capabilities import CAPABILITIES  # noqa: E402
from digitalearth.three_d.guides import (  # noqa: E402
    color_scale,
    guide_field,
)


@pytest.fixture
def scene():
    """Yield an off-screen scene, closed on the way out.

    Yields:
        The scene under test.
    """
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


def _raster():
    """Build the georeferenced raster ``quickmap`` dispatches on, from the same ramp as :func:`_dem`.

    ``quickmap`` takes a pyramids ``Dataset`` or ``FeatureCollection`` and refuses anything else by name, so
    the dispatcher tests cannot reuse the bare ``Source`` the scene-level tests build. Built in memory rather
    than read from ``examples/data`` so the test does not depend on the process's working directory.

    Returns:
        A pyramids ``Dataset`` over an 8x8 ramp in EPSG:4326.
    """
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset

    return Dataset.from_array(
        np.add.outer(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8)),
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 8.0, 0.0, -1.0), epsg=4326),
    )


class TestTheKeyIsRecordedBeforeItIsDrawn:
    """The ordering order 24 is about: a `Guide` on the layer's encoding, and the drawing derived from it."""

    def test_the_guide_is_on_the_layer_that_carries_the_colour(self, scene):
        """`colorbar()` writes a `Guide` onto the layer's colour encoding, not beside it.

        Args:
            scene: The scene under test.

        Test scenario:
            The tier had no record of a colour key at all — the bar existed only inside PyVista, titled after
            the scalars array. Recording it on the encoding is what lets every other behaviour below follow
            without a second mechanism to keep in step.
        """
        scene.terrain(_dem())
        scene.colorbar(label="Elevation (m)")
        guide = scene.get_layer("terrain-1").symbology.guide()
        assert (guide.show, guide.title) == (True, "Elevation (m)"), guide

    def test_the_recorded_title_is_what_reaches_the_plotter(self, scene):
        """The bar is drawn *from* the record, so its title is the guide's rather than the array's.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem())
        assert "elevation" in scene.plotter.scalar_bars, (
            "PyVista's own bar, titled by the array"
        )
        scene.colorbar(label="Elevation (m)")
        assert list(scene.plotter.scalar_bars.keys()) == ["Elevation (m)"], (
            f"the engine's bar must be replaced by the guide's, got {list(scene.plotter.scalar_bars.keys())}"
        )

    def test_no_label_leaves_the_bar_the_engine_already_drew(self, scene):
        """Asking for the key a caller already has changes nothing about the picture.

        Args:
            scene: The scene under test.

        Test scenario:
            The guide's default title is the field the colour is driven by, which is the name PyVista titles
            its own bar with — so `colorbar()` on a freshly drawn terrain is a no-op on the window while
            still recording the ask. That is what makes `quickmap(colorbar=True)` honest.
        """
        scene.terrain(_dem())
        scene.colorbar()
        assert list(scene.plotter.scalar_bars.keys()) == ["elevation"], list(
            scene.plotter.scalar_bars.keys()
        )

    def test_the_guide_survives_the_round_trip_the_seam_exists_for(self, scene):
        """`Symbology.to_dict()` → `from_dict()` returns the same guide, because it travels on the layer.

        Args:
            scene: The scene under test.

        Test scenario:
            A key drawn when asked and then forgotten could not be written into a figure. This is the
            property that the record — rather than the drawing — is the source of truth.
        """
        scene.terrain(_dem())
        scene.colorbar(label="Elevation (m)")
        symbology = scene.get_layer("terrain-1").symbology
        restored = Symbology.from_dict(symbology.to_dict())
        assert restored == symbology, (restored.to_dict(), symbology.to_dict())
        assert restored.guide().title == "Elevation (m)", restored.guide()

    def test_a_described_figure_draws_its_key_again(self, scene):
        """A figure drawn into a fresh scene brings its colour key with it.

        Args:
            scene: The scene that describes the figure.

        Test scenario:
            The key is drawn at the end of the renderer's reconcile, which `from_figure` goes through — so
            the round trip above is not only storable but drawable.
        """
        scene.terrain(_dem())
        scene.colorbar(label="Elevation (m)")
        second = Scene3D.from_figure(scene.figure_spec, off_screen=True)
        try:
            drawn = list(second.plotter.scalar_bars.keys())
        finally:
            second.close()
        assert drawn == ["Elevation (m)"], drawn

    def test_both_calls_chain(self, scene):
        """Every decoration method on this tier returns the scene; these are no exception.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem())
        assert scene.colorbar() is scene, "colorbar() must return the scene"
        scene.point_cloud(_points(), values=_points()[:, 0], scheme="quantiles", k=3)
        assert scene.legend() is scene, "legend() must return the scene"


class TestTheKeyFollowsItsLayer:
    """What recording the key on the layer buys: it moves, hides and goes away with it."""

    def test_removing_the_layer_takes_the_bar_off(self, scene):
        """The bar is not left in the corner of a window whose data is gone.

        Args:
            scene: The scene under test.

        Note:
            The engine does this half: measured on PyVista 0.48.4, `Plotter.remove_actor` takes the scalar
            bar bound to that actor with it, and `Renderer3D.remove` calls it. The half this tier owns is
            the *record*, which is what
            :meth:`test_a_removed_layers_key_is_not_inherited_by_the_next_layer_of_that_name` holds.
        """
        scene.terrain(_dem())
        scene.colorbar(label="Elevation (m)")
        scene.remove_layer("terrain-1")
        assert list(scene.plotter.scalar_bars.keys()) == [], list(
            scene.plotter.scalar_bars.keys()
        )

    def test_a_removed_layers_key_is_not_inherited_by_the_next_layer_of_that_name(
        self, scene
    ):
        """An id goes back to the pool when its layer goes, and the key recorded against it must too.

        Args:
            scene: The scene under test.

        Test scenario:
            This tier hands a removed layer's name back unsuffixed (`free_layer_id`, review R2-M9). The
            drawn state a key leaves behind is keyed by that id — the bar's title and the caller's row
            labels — so a record that outlived its layer would be picked up by the next layer called the
            same thing: a fresh terrain titled after a cloud that is no longer on the scene, or drawn with
            its labels.
        """
        points = _points(6)
        labels = np.array(["a", "b", "a", "c", "b", "c"], dtype=object)
        scene.point_cloud(points, values=labels, scheme="categorical", name="shared")
        scene.legend(title="Cover", labels=["one", "two", "three"])
        scene.remove_layer("shared")
        assert "shared" not in scene._guides, dict(scene._guides)
        scene.terrain(_dem(), name="shared")
        scene.colorbar("shared")
        assert list(scene.plotter.scalar_bars.keys()) == ["elevation"], (
            "the new layer must be titled after its own array, not the removed layer's key"
        )

    def test_hiding_the_layer_hides_the_bar_and_showing_it_brings_it_back(self, scene):
        """A hidden layer explains nothing, so its key is not drawn — and returns with it.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem())
        scene.colorbar(label="Elevation (m)")
        scene.set_visible("terrain-1", False)
        hidden = list(scene.plotter.scalar_bars.keys())
        scene.set_visible("terrain-1", True)
        assert (hidden, list(scene.plotter.scalar_bars.keys())) == (
            [],
            ["Elevation (m)"],
        ), (
            hidden,
            list(scene.plotter.scalar_bars.keys()),
        )

    def test_moving_the_layer_does_not_orphan_its_bar(self, scene):
        """Draw order is recorded rather than drawn here, and a key must not be lost to a reorder.

        Args:
            scene: The scene under test.
        """
        points = _points()
        scene.terrain(_dem(), name="dem")
        scene.point_cloud(points, values=points[:, 0], name="cloud")
        scene.colorbar("dem", label="Elevation (m)")
        scene.move_layer("dem", -1)
        assert scene.layer_ids == ["cloud", "dem"], scene.layer_ids
        assert "Elevation (m)" in scene.plotter.scalar_bars, list(
            scene.plotter.scalar_bars.keys()
        )

    def test_removing_a_keyed_layer_takes_the_legend_off(self, scene):
        """The one legend box goes when the last layer contributing to it does.

        Args:
            scene: The scene under test.

        Test scenario:
            The box is rebuilt from the figure on every change, and the record of having drawn one is read
            *before* the removed layer's entry is pruned — read after, nothing was left to say the box had to
            come off and it stayed on the window for good.
        """
        points = _points()
        scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=3)
        scene.legend(title="Distance")
        assert scene.plotter.legend is not None, "the box must be drawn first"
        scene.remove_layer("point_cloud-1")
        assert scene.plotter.legend is None, "the box must go with its last keyed layer"

    def test_hiding_a_keyed_layer_hides_the_legend(self, scene):
        """A hidden layer contributes no rows, and a box of no rows is not drawn.

        Args:
            scene: The scene under test.
        """
        points = _points()
        scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=3)
        scene.legend(title="Distance")
        scene.set_visible("point_cloud-1", False)
        hidden = scene.plotter.legend
        scene.set_visible("point_cloud-1", True)
        assert (hidden, scene.plotter.legend is None) == (None, False), hidden


class TestTheFormIsReadOffTheDescription:
    """Two widgets on one channel, and the layer's `Scale` — not a stored flag — says which."""

    def test_a_ramp_is_a_scalar_bar_and_classes_are_a_keyed_list(self, scene):
        """The two names draw the two PyVista widgets, each over the colouring it can describe.

        Args:
            scene: The scene under test.
        """
        points = _points()
        scene.terrain(_dem(), name="dem")
        scene.point_cloud(
            points, values=points[:, 0], scheme="quantiles", k=3, name="classes"
        )
        scene.colorbar("dem", label="Elevation (m)")
        scene.legend("classes", title="Distance")
        assert "Elevation (m)" in scene.plotter.scalar_bars, list(
            scene.plotter.scalar_bars.keys()
        )
        assert scene.plotter.legend.GetNumberOfEntries() == 3, (
            scene.plotter.legend.GetNumberOfEntries()
        )

    def test_a_keyed_list_over_a_ramp_is_refused_and_points_at_the_bar(self, scene):
        """A continuous ramp has no classes to list, so `legend()` says which name draws it.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem())
        with pytest.raises(ValueError, match="continuous ramp.*colorbar"):
            scene.legend()

    def test_a_bar_over_classes_is_refused_and_points_at_the_list(self, scene):
        """A scalar bar over class indices reads `0, 1, 2 …`, so `colorbar()` says which name draws them.

        Args:
            scene: The scene under test.

        Test scenario:
            A classified layer's mappable carries opaque integer class codes, so a continuous key over it
            labels the wrong thing — which is why `api._has_a_key_to_draw` asks each layer's description
            whether it publishes a colour a *bar* can describe rather than keying whatever was drawn last.
        """
        points = _points()
        scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=3)
        with pytest.raises(ValueError, match="coloured by classes.*legend"):
            scene.colorbar()

    def test_the_keyed_list_replaces_the_class_index_bar(self, scene):
        """PyVista's own bar over a classified layer is taken off, not left beside the key.

        Args:
            scene: The scene under test.
        """
        points = _points()
        scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=3)
        assert list(scene.plotter.scalar_bars.keys()) == ["scalar"], (
            "the engine's index bar"
        )
        scene.legend(title="Distance")
        assert list(scene.plotter.scalar_bars.keys()) == [], list(
            scene.plotter.scalar_bars.keys()
        )


class TestTheRowsComeFromTheScaleThatWasDrawn:
    """DE-19's point, on this tier: a swatch is the colour the picture uses, not a second computation."""

    def test_a_graduated_row_per_class_labelled_by_its_range(self, scene):
        """The rows are `LegendSpec.from_scale`'s, so the class count and the labels come from the `Scale`.

        Args:
            scene: The scene under test.
        """
        points = _points()
        scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=4)
        scene.legend()
        box = scene.plotter.legend
        labels = [box.GetEntryString(i) for i in range(box.GetNumberOfEntries())]
        ranges = (
            scene.get_layer("point_cloud-1")
            .symbology.encoding("color")
            .scale.class_ranges()
        )
        assert len(labels) == len(ranges) == 4, (labels, ranges)
        assert labels[0].startswith(str(ranges[0][0])), (labels[0], ranges[0])

    def test_every_swatch_is_the_colour_the_layer_is_drawn_in(self, scene):
        """The colours are read back off the lookup table the layer renders through.

        Args:
            scene: The scene under test.

        Test scenario:
            This is the disagreement `LegendSpec` exists to remove, and the reason the colours are not
            recomputed from the colormap for the key: a swatch and a class are then the same value, not two
            values that happen to agree.
        """
        from matplotlib.colors import to_hex

        points = _points()
        scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=3)
        scene.legend()
        table = np.asarray(
            scene.renderer.drawn["point_cloud-1"][1].mapper.lookup_table.values
        )
        drawn = [to_hex(row[:3] / 255.0) for row in table]
        box = scene.plotter.legend
        swatches = [
            to_hex(np.asarray(box.GetEntryColor(i))[:3])
            for i in range(box.GetNumberOfEntries())
        ]
        assert swatches == drawn, (swatches, drawn)

    def test_a_categorical_layer_gets_one_row_per_category(self, scene):
        """A categorical scale carries its own colours, so the rows are its categories.

        Args:
            scene: The scene under test.
        """
        points = _points(6)
        labels = np.array(["a", "b", "a", "c", "b", "c"], dtype=object)
        scene.point_cloud(points, values=labels, scheme="categorical", cmap="tab10")
        scene.legend()
        box = scene.plotter.legend
        rows = [box.GetEntryString(i) for i in range(box.GetNumberOfEntries())]
        assert rows == ["a", "b", "c"], rows

    def test_the_callers_own_labels_replace_the_derived_ones(self, scene):
        """`labels=` is the override the Core declares for `legend`, and it reaches the drawn box.

        Args:
            scene: The scene under test.
        """
        points = _points()
        scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=2)
        scene.legend(labels=["low", "high"])
        box = scene.plotter.legend
        rows = [box.GetEntryString(i) for i in range(box.GetNumberOfEntries())]
        assert rows == ["low", "high"], rows


class TestWhatPyvistasOwnSlotsForce:
    """One bar per title, one legend per window — measured on PyVista 0.48.4, not assumed."""

    def test_two_layers_asking_for_one_title_is_refused_by_name(self, scene):
        """A second bar under a title already there shows the first layer's range; that is refused.

        Args:
            scene: The scene under test.

        Test scenario:
            Measured: `add_scalar_bar(title="T")` twice leaves **one** bar with two mappers bound to it, and
            the bar keeps the first mapper's range. Nothing raises, so a caller believes they titled their
            own key and is shown someone else's numbers.
        """
        scene.terrain(_dem(), name="a")
        scene.terrain(_dem(), name="b")
        scene.colorbar("a", label="DEM")
        with pytest.raises(ValueError, match="one bar per title"):
            scene.colorbar("b", label="DEM")
        # The refusal comes from the reconcile, so what it rolls back is the proof the refusal cost nothing:
        # the first layer's key is untouched and the second carries no guide at all.
        assert "DEM" in scene.plotter.scalar_bars, list(
            scene.plotter.scalar_bars.keys()
        )
        assert scene.get_layer("b").symbology.guide() is None, scene.get_layer(
            "b"
        ).symbology.to_dict()

    def test_two_layers_with_their_own_titles_each_get_a_bar(self, scene):
        """Distinct titles are two bars, and the engine's shared one goes once nothing reads it.

        Args:
            scene: The scene under test.

        Test scenario:
            Two terrains both bind `elevation`, so PyVista drew **one** bar for both and neither layer may
            take it away on its own. Once each has a key of its own that bar explains nothing, and leaving it
            would park a stray key in the corner for good.
        """
        scene.terrain(_dem(), name="a")
        scene.terrain(_dem(), name="b")
        scene.colorbar("a", label="DEM A")
        shared = list(scene.plotter.scalar_bars.keys())
        scene.colorbar("b", label="DEM B")
        assert shared == ["elevation", "DEM A"], shared
        assert sorted(scene.plotter.scalar_bars.keys()) == ["DEM A", "DEM B"], list(
            scene.plotter.scalar_bars.keys()
        )

    def test_a_shared_engine_bar_is_not_taken_from_the_layer_still_reading_it(
        self, scene
    ):
        """Retitling one of two layers that share the engine's bar must not un-key the other.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem(), name="a")
        scene.terrain(_dem(), name="b")
        scene.colorbar("a", label="DEM A")
        assert "elevation" in scene.plotter.scalar_bars, (
            "b still reads the engine's bar, so it must survive a's retitle"
        )

    def test_two_keyed_layers_share_one_box_prefixed_by_their_titles(self, scene):
        """A window holds one legend actor, so the box is built from every keyed layer rather than replaced.

        Args:
            scene: The scene under test.

        Test scenario:
            Measured: a second `add_legend` overwrites the plotter's one legend. Letting the last caller win
            would silently drop the first layer's classes, which is the shape of both high-severity
            regressions this branch has already had — a shared slot serving whichever user reached it last.
        """
        points = _points()
        scene.point_cloud(
            points, values=points[:, 0], scheme="quantiles", k=2, name="one"
        )
        scene.point_cloud(
            points, values=points[:, 1], scheme="quantiles", k=3, name="two"
        )
        scene.legend("one", title="First")
        scene.legend("two", title="Second")
        box = scene.plotter.legend
        rows = [box.GetEntryString(i) for i in range(box.GetNumberOfEntries())]
        assert len(rows) == 5, rows
        assert [row.split(":")[0] for row in rows] == ["First"] * 2 + ["Second"] * 3, (
            rows
        )


class TestEverythingIsCheckedBeforeTheFlagIsRead:
    """Review L7's rule, applied to the whole call: one spelling is not valid only half the time."""

    @pytest.mark.parametrize("visible", [True, False])
    def test_a_scene_with_nothing_coloured_by_data_is_refused_either_way(
        self, scene, visible
    ):
        """`colorbar(visible=False)` on a scene with no colour key raises exactly as `colorbar()` does.

        Args:
            scene: The scene under test.
            visible: Both readings of the flag.
        """
        with pytest.raises(ValueError, match="nothing to describe"):
            scene.colorbar(visible=visible)

    @pytest.mark.parametrize("visible", [True, False])
    def test_a_flat_coloured_layer_is_refused_either_way(self, scene, visible):
        """A layer whose colour is flat has no values to label, whichever way the flag reads.

        Args:
            scene: The scene under test.
            visible: Both readings of the flag.
        """
        scene.terrain(_dem(), scalars=None, color="red")
        with pytest.raises(ValueError, match="nothing to describe"):
            scene.colorbar(visible=visible)

    @pytest.mark.parametrize("visible", [True, False])
    def test_an_id_nobody_drew_is_refused_either_way(self, scene, visible):
        """An unknown layer id is a caller error under both readings of the flag.

        Args:
            scene: The scene under test.
            visible: Both readings of the flag.
        """
        scene.terrain(_dem())
        with pytest.raises(KeyError, match="nope"):
            scene.colorbar("nope", visible=visible)

    def test_visible_false_records_what_the_key_would_have_said(self, scene):
        """Switching a key off keeps its title, which is what `Guide(show=False)` is for.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem())
        scene.colorbar(label="Elevation (m)", visible=False)
        guide = scene.get_layer("terrain-1").symbology.guide()
        assert (guide.show, guide.title) == (False, "Elevation (m)"), guide
        assert list(scene.plotter.scalar_bars.keys()) == [], list(
            scene.plotter.scalar_bars.keys()
        )


class TestTheDefaultLayerIsTheColourDrivenOne:
    """`layer_id=None` is "the most recent layer coloured by data", not "the last layer added"."""

    def test_a_label_added_after_the_data_is_not_what_gets_keyed(self, scene):
        """A `text` layer is the last layer and has nothing to key, so the terrain is chosen.

        Args:
            scene: The scene under test.

        Test scenario:
            Taking the last layer outright would refuse a call whose answer is obvious — and every other
            tier's "most recent" accessor has had exactly this defect at some point.
        """
        scene.terrain(_dem(), name="dem")
        scene.text(0.0, 0.0, "mark", crs=None)
        scene.colorbar(label="Elevation (m)")
        assert scene.get_layer("dem").symbology.guide() is not None, "the terrain"
        assert scene.get_layer("text-1").symbology.guide() is None, "not the label"


class TestTheBuildersPublishWhatTheyColourBy:
    """A key needs a colour encoding to hang on, and this is where each builder declares one."""

    def test_a_terrain_names_the_array_it_binds(self, scene):
        """The default surface is coloured by its elevation array, and says so.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem())
        assert guide_field(scene.get_layer("terrain-1")) == "elevation", (
            scene.get_layer("terrain-1").symbology.to_dict()
        )

    def test_scalars_none_is_still_coloured_by_the_elevation(self, scene):
        """`scalars=None` is not a flat surface: `add_mesh` falls back to the mesh's active scalars.

        Args:
            scene: The scene under test.

        Test scenario:
            Measured on PyVista 0.48.4: `terrain(dem, scalars=None)` renders coloured and PyVista draws its
            own bar titled `elevation` for it. Reading `scalars=None` as "flat" left `quickmap(colorbar=False)`
            unable to take that bar off, because there was no encoding for the guide to hang on.
        """
        scene.terrain(_dem(), scalars=None)
        assert guide_field(scene.get_layer("terrain-1")) == "elevation", (
            scene.get_layer("terrain-1").symbology.to_dict()
        )

    def test_a_colour_keyword_with_no_scalars_is_a_flat_surface(self, scene):
        """The one combination that really is flat publishes nothing.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem(), scalars=None, color="red")
        assert guide_field(scene.get_layer("terrain-1")) is None, scene.get_layer(
            "terrain-1"
        ).symbology.to_dict()

    def test_a_cloud_names_its_value_column(self, scene):
        """A cloud coloured by an attribute names that column, which is what a key is titled after.

        Args:
            scene: The scene under test.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        features = gpd.GeoDataFrame(
            {"depth": [1.0, 5.0, 9.0]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs=4326,
        )
        scene.point_cloud(features, value_column="depth")
        assert guide_field(scene.get_layer("point_cloud-1")) == "depth", (
            scene.get_layer("point_cloud-1").symbology.to_dict()
        )

    def test_an_uncoloured_cloud_publishes_nothing(self, scene):
        """No values and no column is a cloud in one colour, which a key cannot describe.

        Args:
            scene: The scene under test.
        """
        scene.point_cloud(_points())
        assert guide_field(scene.get_layer("point_cloud-1")) is None, scene.get_layer(
            "point_cloud-1"
        ).symbology.to_dict()

    def test_extruded_polygons_name_their_column_and_carry_its_classes(self, scene):
        """The prisms' colour column, and the `Scale` the drawer cuts it with.

        Args:
            scene: The scene under test.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon

        squares = gpd.GeoDataFrame(
            {"pop": [1.0, 5.0, 9.0]},
            geometry=[
                Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]) for x in (0, 2, 4)
            ],
            crs=4326,
        )
        scene.extruded_polygons(
            squares, height="pop", column="pop", scheme="quantiles", k=3
        )
        encoding = scene.get_layer("extrusion-1").symbology.encoding("color")
        assert encoding.field == "pop", encoding
        assert len(encoding.scale.class_ranges()) == 3, encoding.scale

    def test_a_volume_and_an_isosurface_name_their_field(self, scene):
        """Both bind the same `field` array, so both publish it.

        Args:
            scene: The scene under test.
        """
        cube = np.random.default_rng(0).random((6, 6, 6))
        scene.volume(cube)
        scene.isosurface(cube)
        assert [guide_field(scene.get_layer(held)) for held in scene.layer_ids] == [
            "field",
            "field",
        ], scene.layer_ids

    def test_a_column_the_data_does_not_carry_leaves_the_refusal_to_the_drawer(
        self, scene
    ):
        """Publishing an encoding must not move a caller's error to a different call.

        Args:
            scene: The scene under test.

        Test scenario:
            The builder reads the colour column to measure its scale. A column that is not there, or one the
            scheme cannot cut, must not raise *here* — the drawer reads the same column a moment later and
            raises the tier's own message for it.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon

        squares = gpd.GeoDataFrame(
            {"pop": [1.0, 9.0]},
            geometry=[
                Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]) for x in (0, 2)
            ],
            crs=4326,
        )
        with pytest.raises(KeyError, match="missing"):
            scene.extruded_polygons(squares, column="missing")

    def test_a_column_the_scheme_cannot_cut_publishes_no_scale(self, scene):
        """An unclassifiable column is described without a scale rather than refused early.

        Args:
            scene: The scene under test.
        """
        assert (
            color_scale([1.0, 1.0, 1.0], scheme="quantiles", k=4, cmap="viridis")
            is None
        ), "a constant column has no spread to cut"


class TestTheDeclarationSaysBothAreBuilt:
    """The two `contract.PENDING["3d"]` rows came off, so the tier has to answer to both names."""

    def test_neither_name_is_pending_any_more(self):
        """Order 24 built them, so the rows are gone and the facade answers instead."""
        from digitalearth.base.contract import pending_for

        pending = pending_for("3d")
        assert "colorbar" not in pending and "legend" not in pending, sorted(pending)

    def test_both_are_declared_features_of_the_tier(self):
        """`colorbar` was declared on the strength of PyVista's bar; `legend` joins it now the box is drawn."""
        assert {"colorbar", "legend"} <= CAPABILITIES.features, sorted(
            CAPABILITIES.features
        )

    def test_the_scene_answers_to_the_declared_keywords(self):
        """The Core declares the keyword set for each; a tier may take more but not spell these differently."""
        import inspect

        from digitalearth.base.contract import core_method

        for name in ("colorbar", "legend"):
            taken = set(inspect.signature(getattr(Scene3D, name)).parameters)
            missing = sorted(core_method(name).keywords - taken)
            assert missing == [], f"Scene3D.{name} does not take {missing}"


class TestTheDispatcherRoutesThroughTheMethod:
    """`quickmap(backend="3d", colorbar=...)` reaches the tier's own names, not a PyVista keyword."""

    def test_the_default_leaves_the_bar_the_engine_drew(self):
        """`colorbar=True` keys the layer without changing the picture."""
        from digitalearth.api import quickmap

        built = quickmap(_raster(), backend="3d")
        try:
            drawn = list(built.plotter.scalar_bars.keys())
            guide = built.get_layer(built.layer_ids[0]).symbology.guide()
        finally:
            built.close()
        assert drawn == ["elevation"], drawn
        assert guide is not None and guide.show, guide

    def test_false_takes_the_bar_off(self):
        """`colorbar=False` used to reach past the tier with `show_scalar_bar=False`; now it records a guide."""
        from digitalearth.api import quickmap

        built = quickmap(_raster(), backend="3d", colorbar=False)
        try:
            drawn = list(built.plotter.scalar_bars.keys())
            guide = built.get_layer(built.layer_ids[0]).symbology.guide()
        finally:
            built.close()
        assert drawn == [], drawn
        assert guide is not None and not guide.show, guide

    def test_a_scalars_none_terrain_can_still_be_switched_off(self):
        """The case a "flat means no encoding" reading broke: the bar is really there, so it comes off."""
        from digitalearth.api import quickmap

        built = quickmap(_raster(), backend="3d", scalars=None, colorbar=False)
        try:
            drawn = list(built.plotter.scalar_bars.keys())
        finally:
            built.close()
        assert drawn == [], drawn

    @pytest.mark.parametrize("colorbar", [True, False])
    def test_a_layer_with_no_key_is_neither_keyed_nor_an_error(self, colorbar):
        """A flat-coloured layer gets no key and no error, as on the matplotlib and web paths.

        Args:
            colorbar: Both readings of the flag.
        """
        from digitalearth.api import quickmap

        built = quickmap(
            _raster(), backend="3d", scalars=None, color="red", colorbar=colorbar
        )
        try:
            drawn = list(built.plotter.scalar_bars.keys())
            guide = built.get_layer(built.layer_ids[0]).symbology.guide()
        finally:
            built.close()
        assert (drawn, guide) == ([], None), (drawn, guide)


class TestAGuideIsNotAMeshChange:
    """Asking for a key must not rebuild the layer: the mesh and the mapper are what a drawer builds."""

    def test_the_mesh_is_not_rebuilt_when_a_key_is_asked_for(self, scene):
        """`colorbar()` changes the layer's symbology, and only the guide in it.

        Args:
            scene: The scene under test.

        Test scenario:
            `diff` groups a guide with a colormap, because both are "the symbology changed". Answering both
            with a rebuild meant a `colorbar(label=...)` on a large DEM re-opened the raster, re-ran the
            reprojection and rebuilt the surface to put a text label on the window — the same defect review
            M8 fixed for a change of `label`.
        """
        scene.terrain(_dem())
        before = scene.mesh_of("terrain-1")
        scene.colorbar(label="Elevation (m)")
        assert scene.mesh_of("terrain-1") is before, (
            "a guide is not something PyVista draws the mesh from"
        )

    def test_a_colormap_change_still_rebuilds(self, scene):
        """The guard is narrowed to guides, not switched off: a restyle that reaches VTK still redraws.

        Args:
            scene: The scene under test.
        """
        from dataclasses import replace as with_fields

        scene.terrain(_dem())
        before = scene.mesh_of("terrain-1")
        held = scene.get_layer("terrain-1")
        scene.replace_layer(
            with_fields(held, symbology=held.symbology.with_props(cmap="magma"))
        )
        assert scene.mesh_of("terrain-1") is not before, (
            "a colormap is baked into the mesh's mapper, so it is a rebuild"
        )
