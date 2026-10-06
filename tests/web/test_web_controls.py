"""WB-14 — the four remaining unused py-maplibregl controls on the web tier.

``AttributionControl``, ``GeolocateControl``, ``GlobeControl`` and ``TerrainControl`` ship in py-maplibregl
0.3.6 but the tier exposed none of them. Each is wired the way ``navigation``/``scale_bar``/``fullscreen``
already are — a furniture item for the figure's description plus an ``apply(widget)`` closure that adds the
live control (the geocoder, the fifth control, is WB-15's ``geocoder``).

The whole file ``importorskip``s maplibre, so it SKIPs under ``-e dev`` and RUNs under ``-e web``.
"""

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap  # noqa: E402

#: Control furniture kind -> a builder call that adds it to a fresh map with its default placement.
CONTROLS = {
    "attribution": lambda m: m.attribution(),
    "geolocate": lambda m: m.geolocate(),
    "globe_control": lambda m: m.globe_control(),
    "terrain_control": lambda m: m.terrain_control("dem"),
}

#: The same controls, each placed in a corner MapLibre does not know, so the refusal is under test.
BAD_POSITION = {
    "attribution": lambda m: m.attribution(position="middle"),
    "geolocate": lambda m: m.geolocate(position="middle"),
    "globe_control": lambda m: m.globe_control(position="middle"),
    "terrain_control": lambda m: m.terrain_control("dem", position="middle"),
}


def _item(web_map, kind):
    """Return the single furniture item of ``kind`` a map recorded.

    Args:
        web_map: The map to read.
        kind: The furniture kind to find.

    Returns:
        The matching :class:`~digitalearth.base.spec.Furniture`.
    """
    return next(item for item in web_map._furniture if item.kind == kind)


class TestEachControlIsWired:
    """The shared control shape: described as furniture, wired through one queued closure, chainable."""

    @pytest.mark.parametrize("kind", sorted(CONTROLS))
    def test_the_control_is_recorded_as_panel_furniture(self, kind):
        """A control is furniture, not a layer, so it is read off the panel's furniture.

        Args:
            kind: The control furniture kind under test.
        """
        kinds = [
            item.kind
            for item in CONTROLS[kind](WebMap()).figure_spec.panels[0].furniture
        ]
        assert kind in kinds, kinds

    @pytest.mark.parametrize("kind", sorted(CONTROLS))
    def test_the_builder_is_chainable(self, kind):
        """Every builder returns the map, so calls chain.

        Args:
            kind: The control furniture kind under test.
        """
        web_map = WebMap()
        assert CONTROLS[kind](web_map) is web_map

    @pytest.mark.parametrize("kind", sorted(CONTROLS))
    def test_one_queue_entry_wires_the_live_control(self, kind):
        """The control is added through a single queued closure, like every other control.

        Args:
            kind: The control furniture kind under test.
        """
        assert len(CONTROLS[kind](WebMap())._queued) == 1

    @pytest.mark.parametrize("kind", sorted(CONTROLS))
    def test_the_control_attaches_to_the_rendered_widget(self, kind):
        """Rendering runs the ``apply`` closure, so a crash wiring the control would surface here.

        Args:
            kind: The control furniture kind under test.
        """
        from maplibre.ipywidget import MapWidget

        assert isinstance(CONTROLS[kind](WebMap()).render(), MapWidget)

    @pytest.mark.parametrize("kind", sorted(BAD_POSITION))
    def test_a_bad_position_is_refused(self, kind):
        """A corner MapLibre does not know is silently dropped, so it is refused at the call.

        Args:
            kind: The control furniture kind under test.
        """
        bad_control = BAD_POSITION[kind]
        mapped = WebMap()
        with pytest.raises(ValueError, match="control position"):
            bad_control(mapped)


class TestControlOptionsReachTheDescription:
    """Each control's own options are part of the figure's description and are recorded."""

    def test_attribution_records_compact(self):
        """A compact attribution is a layout choice, so it travels with the figure."""
        assert (
            _item(WebMap().attribution(compact=True), "attribution").options["compact"]
            is True
        )

    def test_attribution_records_custom_credit_text(self):
        """Extra credit text a caller passes is part of the description and is recorded as ``custom``."""
        item = _item(WebMap().attribution(custom="My data source"), "attribution")
        assert item.options["custom"] == "My data source", item.options

    def test_attribution_defaults_to_the_bottom_right(self):
        """Attribution sits bottom-right unless the caller moves it, as it does in every MapLibre map."""
        assert _item(WebMap().attribution(), "attribution").anchor == "bottom-right"

    def test_geolocate_records_track(self):
        """Tracking the user is a behaviour the description has to carry."""
        assert (
            _item(WebMap().geolocate(track=True), "geolocate").options["track"] is True
        )

    def test_terrain_control_records_the_source_and_exaggeration(self):
        """The control drives a named DEM source at a vertical exaggeration; both are recorded."""
        item = _item(
            WebMap().terrain_control("dem", exaggeration=1.8), "terrain_control"
        )
        assert item.options["exaggeration"] == 1.8, item.options

    def test_terrain_control_requires_a_source(self):
        """Without a DEM source id the control has nothing to drive, so it is refused at the call."""
        mapped = WebMap()
        with pytest.raises(ValueError, match="source"):
            mapped.terrain_control("")

    def test_terrain_control_refuses_a_non_finite_exaggeration(self):
        """A NaN exaggeration is refused at the call, not late in the figure serializer like every sibling.

        Test scenario:
            ``text``/``coastlines``/``borders``/``vector_tiles`` run numeric kwargs through ``as_finite`` and
            refuse NaN/inf at the call; ``terrain_control`` must do the same rather than record the NaN and
            fail only when the figure is written down.
        """
        mapped = WebMap()
        not_a_number = float("nan")
        with pytest.raises(ValueError, match="exaggeration= as a finite number"):
            mapped.terrain_control("dem", exaggeration=not_a_number)
