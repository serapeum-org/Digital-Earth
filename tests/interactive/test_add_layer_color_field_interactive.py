"""Tests for ``InteractiveMap.add_layer(color_field=)`` (#349, interactive tier).

A hand-built HoloViews element publishes no colour encoding of its own, so ``colorbar``/``legend`` refuse it.
``color_field`` names what its colour varies by, recorded as ``Encoding.by_field("color", color_field)`` under
the ``via="custom"`` marker — the same shape the static and web tiers record. These tests verify the encoding
is published (via ``_colour_binding``, the accessor the guides use) and that ``color_field`` cannot be combined
with the low-level ``symbology`` parameter. The engine is skipped where absent.
"""

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip(
    "holoviews"
)  # the interactive tier engines; skip in envs without them
pytest.importorskip("geoviews")
pytest.importorskip("panel")


def _element():
    """Return a minimal HoloViews element to register as a custom layer.

    Returns:
        A two-point ``hv.Curve`` — enough to stand in for a caller's own element.
    """
    return hv.Curve([(0.0, 0.0), (1.0, 1.0)])


class TestInteractiveAddLayerColorField:
    """Tests for the ``color_field`` keyword on :meth:`InteractiveMap.add_layer`."""

    def test_color_field_publishes_the_colour_encoding_naming_the_field(self):
        """``color_field`` records a colour encoding bound to the named field, which ``_colour_binding`` reads.

        Test scenario:
            A custom layer added with ``color_field="elevation"`` answers that field from ``_colour_binding``
            (the accessor ``colorbar``/``legend`` resolve the key through), so the guides recognise it.
        """
        m = InteractiveMap().add_layer(
            _element(), name="keyed", color_field="elevation"
        )
        encoding = m._colour_binding("keyed")
        assert encoding is not None, "color_field should publish a colour encoding"
        assert encoding.field == "elevation", (
            f"expected field 'elevation', got {encoding.field!r}"
        )

    def test_without_color_field_the_custom_layer_is_keyless(self):
        """The default (``color_field=None``) keeps the custom layer keyless.

        Test scenario:
            A custom layer added with no ``color_field`` has no colour binding, so the guides refuse it.
        """
        m = InteractiveMap().add_layer(_element(), name="plain")
        assert m._colour_binding("plain") is None, (
            "a keyless custom layer should have no colour binding"
        )

    def test_color_field_cannot_be_combined_with_symbology(self):
        """Passing both ``color_field`` and the low-level ``symbology`` is refused rather than dropped.

        Test scenario:
            ``color_field`` is the custom-layer producer and ``symbology`` is what a drawn kind carries, so
            giving both is a contradiction and raises ``ValueError`` instead of silently ignoring one.
        """
        m = InteractiveMap()
        element = _element()
        sentinel = object()
        with pytest.raises(ValueError, match="cannot be combined with symbology"):
            m.add_layer(
                element, name="conflict", color_field="elevation", symbology=sentinel
            )
