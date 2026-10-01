"""``hydrate_foreign_props`` — the web tier fills a foreign figure's missing style props (U-6 cross-tier).

The cross-tier matrix exercises this end-to-end; these pin the function's own branches directly: a raster and a
vector from another tier get this tier's defaults, a same-tier figure is returned untouched, a present ``None``
with a real default (``cmap``) is treated as missing while one with a ``None`` default (``vmin``) is not (so the
pass is idempotent), and a kind with no web defaults is left alone.
"""

from pyramids.feature import FeatureCollection

from digitalearth.static import Map
from digitalearth.web import WebMap, raster, vector
from digitalearth.web.renderer import (
    _FOREIGN_VECTOR_DEFAULTS,
    DRAWN_KINDS,
    drawer_for,
    hydrate_foreign_props,
)

#: Web vector kinds deliberately NOT given cross-tier defaults yet — they need more than ``maplibre_type``/
#: ``paint`` (a label's text field, a contour's levels) and are uncommon cross-tier, so a foreign figure of one
#: raises inside the drawer, as :func:`digitalearth.api.to_backend` documents. Listed so a *new* ``draw_vector``
#: kind forces a choice rather than silently losing cross-tier support.
_NOT_HYDRATED = frozenset({"labels", "contours", "filled_contours"})


def test_the_foreign_default_tables_match_the_web_drawers():
    """The default tables point at the right drawers, and every ``draw_vector`` kind is accounted for.

    The raster defaults target the kind ``draw_field`` serves; every vector-default kind is really drawn by
    ``draw_vector``; and every kind ``draw_vector`` serves is either hydrated or explicitly listed as
    not-yet-hydrated — so a new vector kind cannot silently lose cross-tier support (it fails this guard until
    added to one set or the other).
    """
    assert drawer_for("raster") is raster.draw_field, (
        "raster defaults target the wrong drawer"
    )
    for kind in _FOREIGN_VECTOR_DEFAULTS:
        assert drawer_for(kind) is vector.draw_vector, (
            f"{kind!r} default is not a draw_vector kind"
        )
    serves = {kind for kind in DRAWN_KINDS if drawer_for(kind) is vector.draw_vector}
    unaccounted = serves - set(_FOREIGN_VECTOR_DEFAULTS) - _NOT_HYDRATED
    assert not unaccounted, (
        f"draw_vector kind(s) neither hydrated nor listed not-hydrated: {sorted(unaccounted)}"
    )


class TestHydrateForeignProps:
    """Filling the per-layer style a foreign figure does not carry, without touching a same-tier one."""

    def test_a_foreign_raster_gets_this_tiers_colour_defaults(self, dataset):
        """A raster another tier described (no web cmap/limits/opacity) is filled with the tier defaults.

        Args:
            dataset: The raster fixture, drawn as a field by the static tier (foreign to web).
        """
        source = Map(crs=4326)
        source.field(dataset)
        layer_id = source.figure_spec.layers.ids[0]
        out = hydrate_foreign_props(source.figure_spec)
        props = dict(out.layers.get(layer_id).symbology.props)
        assert props["cmap"] == "viridis", f"cmap not filled: {props.get('cmap')}"
        assert props["opacity"] == 1.0, f"opacity not filled: {props.get('opacity')}"
        assert "vmin" in props, "lower colour limit not filled"
        assert "vmax" in props, "upper colour limit not filled"

    def test_a_foreign_vector_gets_a_maplibre_type_and_paint(self, points):
        """A points layer another tier described is filled with a MapLibre type and a default paint.

        Args:
            points: The point-geometry fixture, drawn by the static tier (foreign to web).
        """
        source = Map(crs=4326)
        source.points(FeatureCollection(points))
        layer_id = source.figure_spec.layers.ids[0]
        out = hydrate_foreign_props(source.figure_spec)
        props = dict(out.layers.get(layer_id).symbology.props)
        assert props["maplibre_type"] == "circle", (
            f"maplibre_type not filled: {props.get('maplibre_type')}"
        )
        assert "circle-color" in props["paint"], (
            f"paint not filled: {props.get('paint')}"
        )

    def test_a_same_tier_figure_is_returned_unchanged(self, dataset):
        """A web figure already records every prop, so hydration is a no-op and returns the same object.

        Args:
            dataset: The raster fixture, drawn by the web tier itself.
        """
        source = WebMap(crs=4326)
        source.field(dataset)
        figure = source.figure_spec
        assert hydrate_foreign_props(figure) is figure, (
            "a same-tier figure must pass through unchanged"
        )

    def test_hydration_is_idempotent(self, dataset):
        """Hydrating an already-hydrated foreign figure returns it unchanged — a present ``None`` ``vmin``
        is not re-filled, so the pass does not keep rebuilding the figure.

        Args:
            dataset: The raster fixture, drawn by the static tier (foreign to web).
        """
        source = Map(crs=4326)
        source.field(dataset)
        once = hydrate_foreign_props(source.figure_spec)
        assert hydrate_foreign_props(once) is once, "a second hydration must be a no-op"

    def test_a_kind_with_no_web_defaults_is_left_alone(self, dataset):
        """A graticule has no fill-in defaults, so its layer passes through unchanged.

        Args:
            dataset: Unused beyond giving the map a reason to exist; the graticule carries no source.
        """
        source = WebMap(crs=4326)
        source.graticule()
        figure = source.figure_spec
        assert hydrate_foreign_props(figure) is figure, (
            "a graticule-only figure needs no hydration"
        )
