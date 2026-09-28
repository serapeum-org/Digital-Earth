"""Order 24 on the interactive tier — a colour key that follows its layer (#261).

`colorbar` and `legend` were ``show=`` toggles that wrote a HoloViews option onto the element of whichever
layer was added last, and then forgot. Nothing tied the key to the layer: it could not move with it, could not
go away with it, and appeared in no figure — which is the same shape of defect the static tier had keyed by
position and the web tier by "the most recent classification".

What replaces it is a :class:`~digitalearth.base.spec.encoding.Guide` on the layer's own ``color`` encoding,
attached through :meth:`~digitalearth.base.spec.style.Symbology.with_guide`. Two consequences are what this
module measures. **A layer whose colour varies with nothing cannot carry a key** — a basemap, a coastline, a
flat fill — so the call is refused rather than drawn as an empty box. And **the key is the layer's**: removing
the layer takes it, hiding the layer leaves it on a layer that comes back, moving the layer keeps it on the
same one, and `to_dict()` → `from_dict()` carries it.

The drawn half is checked too, because a record nobody draws from is not a key. The engine's own limits are
part of that and are stated rather than assumed: Bokeh draws **one** colorbar per plot and builds **one**
legend from an overlay's labelled members, both measured here.
"""

import warnings

import pytest

from digitalearth.base.spec import Guide, Symbology
from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")

#: The raster every probe draws. Relative, as every other test in this suite is.
DEM = "examples/data/acc4000.tif"


@pytest.fixture
def m():
    """Yield a fresh Web-Mercator map for each test, closing it on the way out.

    The object registry is process-global and holds strong references, so a map that is never closed keeps
    the data it drew for the rest of the session.

    Yields:
        The map.
    """
    interactive_map = InteractiveMap()
    yield interactive_map
    interactive_map.close()


def _dem():
    """Return the raster the raster probes draw.

    Returns:
        A pyramids `Dataset`.
    """
    from pyramids.dataset import Dataset

    return Dataset.read_file(DEM)


def _polygons():
    """Return a two-polygon frame with one numeric column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {"pop": [1.0, 9.0]},
        geometry=[
            Polygon([(4, 52), (5, 52), (5, 53), (4, 53)]),
            Polygon([(5, 52), (6, 52), (6, 53), (5, 53)]),
        ],
        crs="EPSG:4326",
    )


def _guide(interactive_map, layer_id):
    """Return the guide one layer carries, read off the figure the map reports.

    Args:
        interactive_map: The map.
        layer_id: The layer to read.

    Returns:
        Its colour :class:`~digitalearth.base.spec.encoding.Guide`, or `None`.
    """
    figure = interactive_map.figure_spec
    return figure.layers.get(layer_id).symbology.guide()


def _plot_options(element):
    """Return the Bokeh plot options HoloViews has applied to one element.

    Args:
        element: The element to ask.

    Returns:
        The applied `plot` keywords.
    """
    return hv.Store.lookup_options("bokeh", element, "plot").kwargs


def _element_of(interactive_map, element_type):
    """Return the one element of a type the map composes.

    Args:
        interactive_map: The map.
        element_type: The HoloViews element type wanted.

    Returns:
        The element — found by type rather than by position, because position is band order.
    """
    return next(
        element
        for element in interactive_map.layers
        if isinstance(element, element_type)
    )


class TestTheGuideIsRecordedOnTheLayer:
    """The record is the point of the order: a key that is only an applied option is not the layer's."""

    def test_a_colorbar_is_recorded_as_a_guide_on_the_colour_encoding(self, m, dataset):
        """The call writes the guide where the layer's style already travels.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            Before this, `colorbar(False)` wrote `colorbar=False` into HoloViews' option store against the
            element and nothing else — `get_layer(id).symbology` said nothing about a key at all, so a figure
            written and read back drew one.
        """
        m.field(dataset).colorbar(label="Flow")
        guide = _guide(m, m.layer_ids[0])
        assert guide == Guide(show=True, title="Flow"), guide

    def test_a_legend_records_the_same_guide_from_its_own_keyword(self, m, dataset):
        """`title=` is the legend's spelling of the one thing a guide is called.

        Args:
            m: The map.
            dataset: A small raster to contour.
        """
        m.contours(dataset, levels=4).legend(title="Contours")
        guide = _guide(m, m.layer_ids[0])
        assert guide == Guide(show=True, title="Contours"), guide

    def test_visible_false_records_the_guide_switched_off(self, m, dataset):
        """A key the caller turned off is a recorded decision, not an absent record.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            `Guide(show=False, title=…)` is how the vocabulary spells "there is a key here and it is not
            drawn", which is what lets a switcher offer it back. Recording nothing would say the layer never
            asked for one.
        """
        m.field(dataset).colorbar(label="Flow", visible=False)
        guide = _guide(m, m.layer_ids[0])
        assert guide == Guide(show=False, title="Flow"), guide

    def test_the_recorded_guide_survives_a_round_trip_through_the_dict_form(
        self, m, dataset
    ):
        """A key in the figure is a key that comes back with it.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            The one thing a `show=` toggle could never do. `Symbology.to_dict()`/`from_dict()` is the form a
            saved figure holds, so this is the check that a saved map still knows its key.
        """
        m.field(dataset).colorbar(label="Flow (m³/s)")
        symbology = m.figure_spec.layers.get(m.layer_ids[0]).symbology
        restored = Symbology.from_dict(symbology.to_dict())
        assert restored.guide() == Guide(show=True, title="Flow (m³/s)"), restored

    def test_a_second_call_replaces_the_guide_rather_than_stacking_one(
        self, m, dataset
    ):
        """One layer, one colour channel, one guide — so the last call is what the layer says.

        Args:
            m: The map.
            dataset: A small raster.
        """
        m.field(dataset).colorbar(label="First").colorbar(label="Second")
        assert _guide(m, m.layer_ids[0]) == Guide(show=True, title="Second")


class TestWhichLayerTheKeyBelongsTo:
    """`layer_id=None` is the most recent layer whose colour varies with its data."""

    def test_the_layer_added_last_takes_the_key_even_when_it_is_drawn_beneath(
        self, m, dataset
    ):
        """The care the old toggle took, kept: a raster added after a label is still the caller's subject.

        Args:
            m: The map.
            dataset: A small raster.
        """
        m.text(4.0, 52.0, "label").field(dataset).colorbar(label="Flow")
        raster = next(
            layer_id for layer_id in m.layer_ids if layer_id.startswith("raster")
        )
        assert _guide(m, raster) == Guide(show=True, title="Flow")

    def test_a_layer_that_colours_by_nothing_does_not_take_the_key_over(
        self, m, dataset
    ):
        """The tightening: a coastline added after the raster cannot be what a colour key explains.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            The old toggle followed `_last_layer_id` unconditionally, so `field(dem).coastlines().colorbar()`
            wrote a `colorbar` option onto the coastline — a call that did nothing the caller could see, and
            the quieter half of review H5. A guide cannot even be attached there, because the layer publishes
            no colour encoding for one to explain.
        """
        m.field(dataset).coastlines().colorbar(label="Flow")
        raster = next(
            layer_id for layer_id in m.layer_ids if layer_id.startswith("raster")
        )
        coastline = next(
            layer_id for layer_id in m.layer_ids if layer_id.startswith("coastlines")
        )
        assert _guide(m, raster) == Guide(show=True, title="Flow")
        assert _guide(m, coastline) is None, "the coastline took the key"

    def test_naming_a_layer_describes_that_layer_and_not_the_recent_one(self, m):
        """`layer_id=` is the whole reason the key is addressable at all.

        Args:
            m: The map.
        """
        m.choropleth(_polygons(), "pop", name="pop").field(_dem(), name="flow")
        m.colorbar("pop", label="People")
        assert _guide(m, "pop") == Guide(show=True, title="People")
        assert _guide(m, "flow") is None, "the raster took a key meant for the polygons"

    def test_a_flat_layer_named_explicitly_is_refused_by_name(self, m, dataset):
        """An empty box would be worse than an error, which is the web tier's position too.

        Args:
            m: The map.
            dataset: A small raster.
        """
        m.field(dataset, name="flow").coastlines(name="coast")
        with pytest.raises(ValueError, match="draws no colour that varies"):
            m.colorbar("coast")

    def test_an_id_no_layer_has_is_refused_as_every_other_call_refuses_it(self, m):
        """The tier's one refusal for an unknown id, so a caller's `except` clause is the same everywhere.

        Args:
            m: The map.
        """
        m.field(_dem(), name="flow")
        with pytest.raises(KeyError, match="no layer 'nope' on this map"):
            m.legend("nope")

    def test_a_flat_fill_publishes_no_colour_and_so_takes_no_key(self, m):
        """`polygons()` with no column draws outlines, and outlines have no colour to explain.

        Args:
            m: The map.

        Test scenario:
            A caller's own `color="#f00"` would be *lifted* onto the colour channel as a constant by
            `portable_encodings`, so "the layer publishes a colour encoding" is not the question — whether
            anything *varies* it is.
        """
        m.polygons(_polygons(), color="#ff0000")
        with pytest.raises(ValueError, match="needs a layer whose colour varies"):
            m.legend()


class TestTheKeyFollowsItsLayer:
    """Move it, hide it, remove it — the three things a figure-level key could not do."""

    def test_removing_the_layer_takes_the_key_with_it(self, m, dataset):
        """A key on a layer that is gone is a key nothing draws, and nothing describes.

        Args:
            m: The map.
            dataset: A small raster.
        """
        m.field(dataset, name="flow").colorbar(label="Flow")
        m.remove_layer("flow")
        assert m.layer_ids == [], m.layer_ids
        with pytest.raises(ValueError, match="needs a layer whose colour varies"):
            m.colorbar()

    def test_hiding_the_layer_keeps_its_key_so_the_key_comes_back(self, m, dataset):
        """A hidden layer is still on the map, so its key is still its own.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            The distinction `set_visible` draws everywhere else on this tier: a hidden layer stays described
            so it can be switched back on. Its key does too — which is only true because the key is recorded
            on the layer rather than beside the figure.
        """
        m.field(dataset, name="flow").colorbar(label="Flow")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m.set_visible("flow", False)
        assert m.figure_spec.layers.is_visible("flow") is False
        assert _guide(m, "flow") == Guide(show=True, title="Flow")

    def test_moving_the_layer_does_not_orphan_its_key(self, m, dataset):
        """The key travels with the layer in draw order, and stays applied to its element.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            `move_layer` goes through `_change`, which re-arranges `InteractiveMap.layers`. A key applied to
            an element by position would end up on whatever element moved into that slot; recorded on the
            layer and applied to its own element, it moves with both.
        """
        m.field(dataset, name="flow").colorbar(label="Flow", visible=False)
        m.choropleth(_polygons(), "pop", name="pop")
        m.move_layer("flow", -1)
        assert m.layer_ids[-1] == "flow", m.layer_ids
        assert _guide(m, "flow") == Guide(show=False, title="Flow")
        moved = m.layers[m.layer_ids.index("flow")]
        assert _plot_options(moved)["colorbar"] is False, _plot_options(moved)


class TestTheKeyIsAlsoDrawn:
    """A record nobody draws from is not a key, so the options are checked as well."""

    def test_the_label_reaches_the_drawn_colorbar(self, m, dataset):
        """`label=` is HoloViews' `clabel` on the element, applied from the record.

        Args:
            m: The map.
            dataset: A small raster.
        """
        m.field(dataset).colorbar(label="Flow (m³/s)")
        options = _plot_options(_element_of(m, hv.Image))
        assert (options["colorbar"], options["clabel"]) == (True, "Flow (m³/s)"), (
            options
        )

    def test_the_title_reaches_the_drawn_legend(self, m, dataset):
        """`title=` is Bokeh's own `Legend.title`, reached through the option HoloViews forwards.

        Args:
            m: The map.
            dataset: A small raster to contour.
        """
        m.contours(dataset, levels=4).legend(title="Contours")
        options = _plot_options(_element_of(m, hv.Contours))
        assert options["show_legend"] is True, options
        assert options["legend_opts"] == {"title": "Contours"}, options

    def test_labels_rename_the_row_this_layer_contributes(self, m):
        """A Bokeh legend's rows are the overlay's labelled members, so a layer contributes one.

        Args:
            m: The map.

        Test scenario:
            Measured rather than assumed: a single element with `show_legend=True` renders no legend at all,
            because there is no overlay to build one from. What `labels=` can honestly do on this engine is
            rename the row the layer contributes, which is captioned by its own `name=`.
        """
        m.choropleth(_polygons(), "pop", name="pop", scheme="quantiles", k=2)
        m.legend(labels=["People per km²"])
        options = _plot_options(_element_of(m, hv.Polygons))
        assert options["legend_labels"] == {"pop": "People per km²"}, options

    def test_a_label_count_that_does_not_match_is_refused(self, m):
        """Zipping would drop the difference and leave rows out of the key — the web tier's rule.

        Args:
            m: The map.
        """
        m.choropleth(_polygons(), "pop", name="pop")
        with pytest.raises(ValueError, match="zip would drop the difference"):
            m.legend(labels=["one", "two"])

    def test_a_rejected_label_count_leaves_the_description_untouched(self, m):
        """A refused call must not half-apply: the guide is recorded after the count is checked.

        Args:
            m: The map.
        """
        m.choropleth(_polygons(), "pop", name="pop")
        with pytest.raises(ValueError):
            m.legend(title="People", labels=["one", "two"])
        assert _guide(m, "pop") is None, "a refused legend() still recorded a guide"

    def test_an_option_the_element_cannot_draw_is_reported_not_dropped(
        self, m, dataset
    ):
        """An `hv.Image` takes no `legend_opts`, and a caller who asked for one hears about it.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            Measured against the registered Bokeh options: `Image` and `QuadMesh` carry `colorbar`, `clabel`
            and `show_legend` but not `legend_opts` or `legend_labels`, while `Contours`, `Points`, `Polygons`
            and `Path` carry all of them. Saying so is the choice `renderer._show` already made for the same
            shape of question — the alternative is a title that silently goes nowhere.
        """
        m.field(dataset, name="flow")
        with pytest.warns(UserWarning, match="no legend_opts to draw"):
            m.legend(title="Flow")
        # The rest of the key is still applied, and the guide is still recorded.
        options = _plot_options(_element_of(m, hv.Image))
        assert options["show_legend"] is True, options
        assert "legend_opts" not in options, options
        assert _guide(m, "flow") == Guide(show=True, title="Flow")

    def test_visible_is_honoured_only_after_the_layer_is_validated(self, m, dataset):
        """One spelling of a call must not be valid only half the time (the web tier's review L7).

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            The trap is an early `if not visible: return self`, which made `legend(position="middle",
            visible=False)` pass on the web tier while `legend(position="middle")` raised. Here the same trap
            would make `colorbar("coast", visible=False)` a silent no-op on a layer that can carry no key.
        """
        m.field(dataset, name="flow").coastlines(name="coast")
        with pytest.raises(ValueError, match="draws no colour that varies"):
            m.colorbar("coast", visible=False)


class TestWhatTheEngineDrawsWhenSeveralLayersCarryAGuide:
    """Bokeh's own limits, stated because the opts suggest otherwise (one guide per layer, one key per plot)."""

    def test_two_guided_rasters_still_render_one_colorbar(self, m):
        """Each layer records its own guide; the plot draws one bar, the lowest layer's.

        Args:
            m: The map.

        Test scenario:
            The opts are per-layer, so it would be easy to document "one key per guided layer". Bokeh does
            not: rendering an overlay of two `colorbar=True` images produces a single `ColorBar`, the one
            belonging to the layer lowest in draw order. That is the same one-key-per-panel limit the web
            tier states from the other direction, and it is the engine's rather than this tier's.
        """
        from bokeh.models import ColorBar

        m.field(_dem(), name="lower").colorbar(label="Lower")
        m.field(_dem(), name="upper").colorbar(label="Upper")
        assert _guide(m, "lower").title == "Lower"
        assert _guide(m, "upper").title == "Upper"
        figure = hv.render(m.render(), backend="bokeh")
        bars = [
            model
            for panel in (figure.right, figure.left, figure.above, figure.below)
            for model in panel
            if isinstance(model, ColorBar)
        ]
        assert len(bars) == 1, f"Bokeh drew {len(bars)} colorbars"
        assert str(bars[0].title) == "Lower", bars[0].title
