"""`FigureSpec.diff()` — what changed between two figures, in the terms a renderer updates (DE-46, #289).

Wave 3 left the method unbuilt, deferred to "a renderer that reconciles". Every Wave 4 seam reconciles against it. Each
figure pair below is built by two separate constructions, so no comparison is an object against itself.
"""

import pytest

from digitalearth.base.spec import (
    DataRef,
    FigureDiff,
    FigureSpec,
    LayerSpec,
    LayerTree,
    PanelSpec,
    Selection,
    Symbology,
    Viewport,
)


def _figure(
    layers=None,
    *,
    sources=None,
    panel_layers=None,
    view=None,
    hidden_groups=(),
    size=None,
    title=None,
):
    """Build a figure with one panel showing every layer, from plain arguments.

    Args:
        layers: The layers, bottom first. Defaults to a raster over `srtm` and points over `obs`.
        sources: The sources by id. Defaults to the two the default layers read.
        panel_layers: The ids the panel shows. Defaults to every layer.
        view: The panel's view. Defaults to `Viewport(4326)`.
        hidden_groups: The groups switched off.
        size: The figure size.
        title: The figure title.

    Returns:
        A new `FigureSpec`, built from scratch on every call.
    """
    if layers is None:
        layers = (
            LayerSpec("dem", "raster", source_id="srtm"),
            LayerSpec("stations", "points", source_id="obs", group="obs"),
        )
    if sources is None:
        sources = {
            "srtm": DataRef("data/dem.tif"),
            "obs": DataRef("data/stations.geojson"),
        }
    ids = tuple(layer.id for layer in layers) if panel_layers is None else panel_layers
    return FigureSpec(
        panels=(PanelSpec("main", view or Viewport(4326), layers=ids),),
        sources=sources,
        layers=LayerTree(tuple(layers), hidden_groups=frozenset(hidden_groups)),
        size=size,
        title=title,
    )


class TestNoChange:
    """Two figures built alike."""

    def test_identical_figures_built_separately_have_an_empty_diff(self):
        """Nothing changed, so every field is empty and the diff is falsy."""
        before, after = _figure(), _figure()
        diff = before.diff(after)
        assert diff == FigureDiff(), diff
        assert not diff, "an empty diff should be falsy"

    def test_a_diff_with_one_change_is_truthy(self):
        """A single change is enough to make the diff truthy."""
        diff = _figure().diff(_figure(title="Relief"))
        assert diff, diff


class TestLayersAddedAndRemoved:
    """Layers that exist on one side only."""

    def test_an_added_layer_is_reported_by_id(self):
        """A layer only the new figure holds is `added`."""
        after_layers = (
            LayerSpec("dem", "raster", source_id="srtm"),
            LayerSpec("stations", "points", source_id="obs", group="obs"),
            LayerSpec("grid", "graticule"),
        )
        diff = _figure().diff(_figure(after_layers))
        assert diff.added == ("grid",), diff

    def test_added_layers_come_in_the_new_figure_draw_order(self):
        """Several added layers are listed bottom first, as the new tree orders them."""
        after_layers = (
            LayerSpec("tiles", "basemap"),
            LayerSpec("dem", "raster", source_id="srtm"),
            LayerSpec("stations", "points", source_id="obs", group="obs"),
            LayerSpec("grid", "graticule"),
        )
        diff = _figure().diff(_figure(after_layers))
        assert diff.added == ("tiles", "grid"), diff

    def test_a_removed_layer_is_reported_by_id(self):
        """A layer only the old figure holds is `removed`, and its source going with it is reported too."""
        after = _figure(
            (LayerSpec("dem", "raster", source_id="srtm"),),
            sources={"srtm": DataRef("data/dem.tif")},
        )
        diff = _figure().diff(after)
        assert (diff.removed, diff.sources) == (("stations",), ("obs",)), diff


class TestLayersRebuiltOrRestyled:
    """A layer on both sides whose description changed."""

    @pytest.mark.parametrize(
        "changed",
        [
            LayerSpec("dem", "mesh", source_id="srtm"),
            LayerSpec("dem", "raster", source_id="obs"),
            LayerSpec(
                "dem", "raster", source_id="srtm", selection=Selection(band=(2,))
            ),
        ],
        ids=["kind", "source-id", "selection"],
    )
    def test_a_new_kind_source_or_slice_is_a_rebuild(self, changed):
        """What is drawn changed, so the layer is drawn again from scratch — never restyled.

        Args:
            changed: The dem layer with one of its kind, source or selection changed.

        Test scenario:
            A renderer cannot restyle its way to a different kind of layer, other data, or another band.
        """
        after = _figure(
            (changed, LayerSpec("stations", "points", source_id="obs", group="obs"))
        )
        diff = _figure().diff(after)
        assert (diff.rebuilt, diff.restyled) == (("dem",), ()), diff

    def test_a_changed_source_rebuilds_every_layer_reading_it(self):
        """A source id pointing at different data rebuilds its layers, and the source is reported."""
        after = _figure(
            sources={
                "srtm": DataRef("data/dem_v2.tif"),
                "obs": DataRef("data/stations.geojson"),
            }
        )
        diff = _figure().diff(after)
        assert (diff.sources, diff.rebuilt) == (("srtm",), ("dem",)), diff

    @pytest.mark.parametrize(
        "changed",
        [
            LayerSpec(
                "stations",
                "points",
                source_id="obs",
                group="obs",
                symbology=Symbology.of(color="#f00"),
            ),
            LayerSpec(
                "stations",
                "points",
                source_id="obs",
                group="obs",
                filter="elevation > 100",
            ),
            LayerSpec(
                "stations",
                "points",
                source_id="obs",
                group="obs",
                label="Weather stations",
            ),
        ],
        ids=["symbology", "filter", "label"],
    )
    def test_a_new_style_filter_or_label_is_a_restyle(self, changed):
        """The data drawn is the same, so the layer is restyled in place.

        Args:
            changed: The stations layer with its style, filter or label changed.
        """
        after = _figure((LayerSpec("dem", "raster", source_id="srtm"), changed))
        diff = _figure().diff(after)
        assert (diff.restyled, diff.rebuilt) == (("stations",), ()), diff


class TestVisibility:
    """Effective visibility: a layer's own switch and its group's."""

    def test_switching_a_layer_off_reports_it_hidden_not_restyled(self):
        """`visible=False` is a visibility change only."""
        after = _figure(
            (
                LayerSpec("dem", "raster", source_id="srtm", visible=False),
                LayerSpec("stations", "points", source_id="obs", group="obs"),
            )
        )
        diff = _figure().diff(after)
        assert (diff.hidden, diff.shown, diff.restyled) == (("dem",), (), ()), diff

    def test_hiding_a_group_hides_every_layer_in_it(self):
        """A group switched off reports each of its layers, though none changed its own switch."""
        diff = _figure().diff(_figure(hidden_groups=("obs",)))
        assert diff.hidden == ("stations",), diff

    def test_showing_a_group_again_reports_its_layers_shown(self):
        """The reverse direction lists the same layers under `shown`."""
        diff = _figure(hidden_groups=("obs",)).diff(_figure())
        assert (diff.shown, diff.hidden) == (("stations",), ()), diff


class TestOrder:
    """Draw order."""

    def test_a_reorder_reports_the_full_new_order(self):
        """Surviving layers in a different relative order give the whole new order."""
        after = _figure(
            (
                LayerSpec("stations", "points", source_id="obs", group="obs"),
                LayerSpec("dem", "raster", source_id="srtm"),
            )
        )
        diff = _figure().diff(after)
        assert diff.order == ("stations", "dem"), diff

    def test_adding_a_layer_on_top_is_not_a_reorder(self):
        """Survivors kept their relative order, so no order is reported."""
        after_layers = (
            LayerSpec("dem", "raster", source_id="srtm"),
            LayerSpec("stations", "points", source_id="obs", group="obs"),
            LayerSpec("grid", "graticule"),
        )
        diff = _figure().diff(_figure(after_layers))
        assert diff.order is None, diff

    def test_a_reorder_with_an_add_and_a_remove_reports_the_full_new_order(self):
        """The new order includes the added layer and leaves out the removed one."""
        before_layers = (
            LayerSpec("tiles", "basemap"),
            LayerSpec("dem", "raster", source_id="srtm"),
            LayerSpec("stations", "points", source_id="obs", group="obs"),
        )
        after_layers = (
            LayerSpec("stations", "points", source_id="obs", group="obs"),
            LayerSpec("grid", "graticule"),
            LayerSpec("dem", "raster", source_id="srtm"),
        )
        diff = _figure(before_layers).diff(_figure(after_layers))
        summary = (diff.added, diff.removed, diff.order)
        assert summary == (("grid",), ("tiles",), ("stations", "grid", "dem")), diff


class TestPanelsAndFigure:
    """What is not a layer."""

    def test_a_panel_with_a_new_view_is_reported(self):
        """A different view on the same panel reports the panel id."""
        diff = _figure().diff(_figure(view=Viewport(3857)))
        assert diff.panels == ("main",), diff

    def test_a_panel_showing_other_layers_is_reported(self):
        """The layers a panel shows are part of the panel."""
        diff = _figure().diff(_figure(panel_layers=("dem",)))
        assert diff.panels == ("main",), diff

    def test_an_added_and_a_removed_panel_are_both_reported(self):
        """A panel present on one side only is reported, new figure's panels first."""
        before = FigureSpec(panels=(PanelSpec("left"),))
        after = FigureSpec(panels=(PanelSpec("right"),))
        diff = before.diff(after)
        assert diff.panels == ("right", "left"), diff

    @pytest.mark.parametrize(
        "after_kwargs",
        [{"title": "Relief"}, {"size": (8.0, 4.0)}],
        ids=["title", "size"],
    )
    def test_a_new_title_or_size_is_a_figure_change(self, after_kwargs):
        """The figure's own fields report through `figure`.

        Args:
            after_kwargs: The figure field that changed.
        """
        diff = _figure().diff(_figure(**after_kwargs))
        assert diff.figure is True, diff


class TestArguments:
    """What `diff` accepts."""

    @pytest.mark.parametrize(
        "other", [None, {"panels": []}, "figure"], ids=["none", "dict", "str"]
    )
    def test_diff_needs_a_figure(self, other):
        """Only a `FigureSpec` can be compared, and the message names what was given.

        Args:
            other: A value that is not a figure.
        """
        figure = _figure()
        with pytest.raises(TypeError, match="FigureSpec.diff needs a FigureSpec"):
            figure.diff(other)
