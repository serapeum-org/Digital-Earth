"""Tests for ``WebMap.add_layer(color_field=)`` (#349, web tier).

A hand-built web layer publishes no colour classification of its own, so ``colorbar``/``legend`` refuse it.
``color_field`` is the producer that names what its colour varies by, recorded as
``Encoding.by_field("color", color_field)`` on the layer — the same shape the static and interactive tiers
record. These tests verify the encoding is published (the producer's job); the engine is skipped where absent.
"""

import pytest

pytest.importorskip("maplibre")  # the web tier engine; skip in envs without it

from digitalearth.web import WebMap


class TestWebAddLayerColorField:
    """Tests for the ``color_field`` keyword on :meth:`WebMap.add_layer`."""

    def test_color_field_publishes_the_colour_encoding_naming_the_field(self):
        """``color_field`` records a colour encoding bound to the named field on the custom layer.

        Test scenario:
            ``add_layer("a-layer", name="keyed", color_field="elevation")`` makes the layer's colour encoding
            carry the field ``"elevation"`` under the ``via="custom"`` marker — what a colour key is titled on.
        """
        m = WebMap().add_layer("a-layer", name="keyed", color_field="elevation")
        encoding = m._layer_tree.get("keyed").symbology.encoding("color")
        assert encoding is not None, "color_field should publish a colour encoding"
        assert encoding.field == "elevation", (
            f"expected field 'elevation', got {encoding.field!r}"
        )

    def test_without_color_field_the_custom_layer_is_keyless(self):
        """The default (``color_field=None``) keeps the pre-#349 keyless custom layer.

        Test scenario:
            ``add_layer("a-layer", name="plain")`` records no colour encoding, so the layer carries no key.
        """
        m = WebMap().add_layer("a-layer", name="plain")
        encoding = m._layer_tree.get("plain").symbology.encoding("color")
        assert encoding is None, (
            f"a keyless custom layer should carry no colour encoding, got {encoding!r}"
        )
