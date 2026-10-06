"""WB-15 — a MapTiler-backed geocoder (place search) control on the web tier.

py-maplibregl ships ``MapTilerGeocodingControl`` but the tier exposed no method for it, and the control is
keyed: it needs a MapTiler API key the caller supplies. :meth:`DecorationMixin.geocoder` wires it the way
every other control is wired — a ``geocoder`` furniture item for the figure's description, plus an
``apply(widget)`` closure that adds the live control — and keeps the secret key out of the serialisable
figure, the way ST-24 keeps a basemap credential out of a saved map.

The whole file ``importorskip``s maplibre, so it SKIPs under ``-e dev`` (no engine) and RUNs under ``-e web``.
"""

import json

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap  # noqa: E402


def _geocoder_item(web_map):
    """Return the single ``geocoder`` furniture item a map recorded.

    Args:
        web_map: The map to read.

    Returns:
        The :class:`~digitalearth.base.spec.Furniture` whose kind is ``"geocoder"``.
    """
    return next(item for item in web_map._furniture if item.kind == "geocoder")


class TestGeocoderRecordsFurniture:
    """The control is described as panel furniture, so it travels with the figure."""

    def test_a_geocoder_is_recorded_on_the_panel(self):
        """A geocoder is furniture, not a layer, so it is read off the panel's furniture."""
        kinds = [
            item.kind
            for item in WebMap().geocoder("FAKE-KEY").figure_spec.panels[0].furniture
        ]
        assert "geocoder" in kinds, kinds

    def test_the_position_is_recorded_as_the_anchor(self):
        """``position=`` places the box, and the recorded anchor is where a replay would redraw it."""
        item = _geocoder_item(WebMap().geocoder("FAKE-KEY", position="bottom-left"))
        assert item.anchor == "bottom-left", item.anchor

    def test_a_non_secret_option_reaches_the_recorded_furniture(self):
        """The layout options a caller sets are part of the description and are recorded."""
        item = _geocoder_item(WebMap().geocoder("FAKE-KEY", placeholder="Find a place"))
        assert item.options["placeholder"] == "Find a place", item.options

    def test_the_builder_is_chainable(self):
        """Every builder on this tier returns the map, so calls chain."""
        web_map = WebMap()
        assert web_map.geocoder("FAKE-KEY") is web_map

    def test_one_queue_entry_wires_the_live_control(self):
        """The control is added through a single queued ``apply`` closure, like every other control."""
        assert len(WebMap().geocoder("FAKE-KEY")._queued) == 1


class TestTheKeyIsNotLeakedIntoTheFigure:
    """A saved page or a serialised figure must not carry the caller's MapTiler key (ST-24's rule)."""

    def test_the_api_key_is_absent_from_the_recorded_options(self):
        """The key is a secret, so it is held only on the live control, never in the description."""
        item = _geocoder_item(WebMap().geocoder("SECRET-KEY-XYZ"))
        assert "api_key" not in item.options, item.options

    def test_the_api_key_never_reaches_the_serialised_figure(self):
        """Serialising the figure to JSON must not embed the key anyone could read off the page."""
        serialised = json.dumps(
            WebMap().geocoder("SECRET-KEY-XYZ").figure_spec.to_dict()
        )
        assert "SECRET-KEY-XYZ" not in serialised, serialised


class TestTheLiveControlAttaches:
    """The queued closure carries the key and adds the real control when the widget is built."""

    def test_the_control_attaches_to_the_rendered_widget(self):
        """Rendering runs the ``apply`` closure, so a crash in it would surface here.

        Test scenario:
            The key lives only on the live control, so the only way to prove the wiring is to build the
            widget and see the closure add the control without raising.
        """
        from maplibre.ipywidget import MapWidget

        widget = WebMap().geocoder("FAKE-KEY").render()
        assert isinstance(widget, MapWidget), type(widget)


class TestGeocoderRefuses:
    """Three malformed calls, each named at the call rather than deep in py-maplibregl."""

    def test_an_empty_api_key_is_refused(self):
        """The service is keyed and the tier ships no key, so an empty one is a mistake named here."""
        with pytest.raises(ValueError, match="api_key"):
            WebMap().geocoder("")

    def test_a_non_string_api_key_is_refused(self):
        """A key that is not even a string could not authenticate, so it is refused the same way."""
        with pytest.raises(ValueError, match="api_key"):
            WebMap().geocoder(None)

    def test_a_bad_position_is_refused(self):
        """A corner MapLibre does not know is silently dropped, so it is refused at the call."""
        with pytest.raises(ValueError, match="control position"):
            WebMap().geocoder("FAKE-KEY", position="middle")
