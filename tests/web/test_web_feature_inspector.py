"""WB-11 — a feature inspector that binds a popup/tooltip across more than one layer.

``popup`` and ``tooltip`` each bound exactly one layer: a bare call took the most recent data layer, and
``layer=`` took one named id. A map with several queryable layers therefore needed one call per layer, and a
list handed to ``layer=`` was refused. This widens both so ``layer=`` takes a list of ids — binding each —
while the single-id and default-to-last behaviour is unchanged, and the per-layer closure the renderer reads
still carries its own layer id so ``remove_layer`` takes the right popup off.

The whole file ``importorskip``s maplibre, so it SKIPs under ``-e dev`` and RUNs under ``-e web``.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap  # noqa: E402


def _points():
    """Return two points in EPSG:4326 with a numeric column.

    Returns:
        The collection each layer below is built from.
    """
    return gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
        crs=4326,
    )


def _two_layer_map():
    """Return a map carrying two named point layers, ``a`` and ``b``.

    Returns:
        The map every multi-layer test below binds against.
    """
    return WebMap().points(_points(), name="a").points(_points(), name="b")


def _interactions(web_map, layer_id):
    """Return the recorded interaction channels for one layer.

    Args:
        web_map: The map to read.
        layer_id: The layer whose popup/tooltip binding is read.

    Returns:
        The ``interactions`` props mapping, or ``{}`` when the layer carries none.
    """
    props = web_map.figure_spec.layers.get(layer_id).symbology.props
    return props.get("interactions", {})


def _closure_ids(web_map):
    """Return the layer ids the queued popup/tooltip closures carry.

    Args:
        web_map: The map to read.

    Returns:
        The set of ``_digitalearth_layer_id`` tags on the queued callables.
    """
    return {
        getattr(item, "_digitalearth_layer_id", None)
        for item in web_map._queued
        if callable(item)
    }


class TestAListBindsEveryNamedLayer:
    """The widening itself: ``layer=[...]`` binds the popup/tooltip to each id in the list."""

    def test_a_popup_list_records_the_click_fields_on_the_first_layer(self):
        """Each named layer gets the click interaction recorded, starting with the first."""
        bound = _interactions(_two_layer_map().popup(["v"], layer=["a", "b"]), "a")
        assert tuple(bound["click"]) == ("v",), bound

    def test_a_popup_list_records_the_click_fields_on_the_second_layer(self):
        """The second named layer is bound too, which is the thing one-layer binding could not do."""
        bound = _interactions(_two_layer_map().popup(["v"], layer=["a", "b"]), "b")
        assert tuple(bound["click"]) == ("v",), bound

    def test_a_tooltip_list_records_the_hover_fields_on_each_layer(self):
        """Hover binds across the list the same way click does."""
        m = _two_layer_map().tooltip(["v"], layer=["a", "b"])
        assert tuple(_interactions(m, "a")["hover"]) == ("v",), _interactions(m, "a")

    def test_each_bound_layer_gets_its_own_closure(self):
        """Two layers, two closures, each tagged with its own id so removal takes the right one off."""
        assert _closure_ids(_two_layer_map().popup(["v"], layer=["a", "b"])) == {
            "a",
            "b",
        }

    def test_binding_every_layer_is_the_map_s_own_ids(self):
        """A feature inspector over the whole map is ``layer=m.layer_ids`` — the common case."""
        m = _two_layer_map()
        m.tooltip(["v"], layer=m.layer_ids)
        assert "hover" in _interactions(m, "a") and "hover" in _interactions(m, "b")


class TestTheSingleLayerBehaviourIsUnchanged:
    """Widening ``layer=`` must not change what a bare call or a single id already did."""

    def test_a_bare_popup_still_binds_only_the_last_layer(self):
        """``layer=None`` takes the most recent layer, as it always has — not every layer."""
        m = _two_layer_map().popup(["v"])
        assert "click" not in _interactions(m, "a"), _interactions(m, "a")

    def test_a_bare_popup_binds_the_last_layer(self):
        """The other half: the last layer is the one that is bound."""
        m = _two_layer_map().popup(["v"])
        assert tuple(_interactions(m, "b")["click"]) == ("v",), _interactions(m, "b")

    def test_a_single_named_id_still_binds_that_one_layer(self):
        """A plain string id is one layer, exactly as before the list was accepted."""
        assert _closure_ids(_two_layer_map().popup(["v"], layer="a")) == {"a"}


class TestMalformedBindingsAreRefused:
    """A typo in the list, and an empty list, are refused at the call rather than wired silently."""

    def test_an_unknown_id_in_the_list_is_refused(self):
        """A named id the map has not drawn is a mistake, named the way a single bad id already is."""
        with pytest.raises(KeyError, match="no layer 'typo'"):
            _two_layer_map().popup(["v"], layer=["a", "typo"])

    def test_an_empty_list_is_refused(self):
        """``layer=[]`` names nothing to inspect, so it is refused rather than binding nothing."""
        with pytest.raises(ValueError, match="names no layer"):
            _two_layer_map().popup(["v"], layer=[])
