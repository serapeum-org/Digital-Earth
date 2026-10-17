"""Tests for ``Map.add_layer(color_field=)`` on the static tier (#349).

A custom layer — a caller's own matplotlib artist — publishes no colour encoding of its own, so ``colorbar()``
and ``legend()`` refuse it. ``color_field=`` is the one producer that was missing: it names what the artist's
colour varies by, publishing ``Encoding.by_field("color", color_field)`` so the guides work on the escape hatch
the builders do not cover.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from digitalearth.static import Map


class TestAddLayerColorField:
    """Tests for the ``color_field`` keyword on :meth:`Map.add_layer`."""

    @staticmethod
    def _image(m):
        """Draw a throwaway mappable on the map's axes and return it.

        Args:
            m: The :class:`Map` whose axes the image is drawn on.

        Returns:
            The ``AxesImage`` a custom layer can be registered from.
        """
        return m.ax.imshow(np.arange(25.0).reshape(5, 5))

    def test_a_custom_layer_without_color_field_has_no_colour_key(self):
        """Without ``color_field`` the custom layer publishes no colour encoding, so ``colorbar`` refuses it.

        Test scenario:
            ``add_layer(im, name="mine")`` then ``colorbar("mine")`` raises ``ValueError`` whose message names
            the layer and says it was not drawn with a colour that varies — the pre-#349 behaviour, preserved.
        """
        m = Map()
        try:
            m.add_layer(self._image(m), name="mine")
            with pytest.raises(ValueError, match="not drawn with a colour that varies"):
                m.colorbar("mine")
        finally:
            m.close()

    def test_color_field_publishes_a_colour_encoding_naming_the_field(self):
        """``color_field`` records ``Encoding.by_field("color", color_field)`` on the layer.

        Test scenario:
            ``add_layer(im, name="keyed", color_field="elevation")`` makes the layer's colour encoding carry
            the field ``"elevation"`` — the name a key is titled after.
        """
        m = Map()
        try:
            m.add_layer(self._image(m), name="keyed", color_field="elevation")
            encoding = m._layer_tree.get("keyed").symbology.encoding("color")
            assert encoding is not None, "color_field should publish a colour encoding"
            assert encoding.field == "elevation", (
                f"expected field 'elevation', got {encoding.field!r}"
            )
        finally:
            m.close()

    def test_colorbar_works_on_a_custom_layer_given_color_field(self):
        """With ``color_field`` set, ``colorbar`` draws for the custom layer instead of refusing it.

        Test scenario:
            The guide the escape hatch used to lose: ``colorbar("keyed")`` returns the map (chainable) and
            leaves the layer addressable, with no ``ValueError``.
        """
        m = Map()
        try:
            m.add_layer(self._image(m), name="keyed", color_field="elevation")
            assert m.colorbar("keyed") is m, (
                "colorbar should return the map for chaining"
            )
            assert "keyed" in m.layer_ids, (
                f"the custom layer should still be listed, got {m.layer_ids}"
            )
        finally:
            m.close()

    def test_an_empty_color_field_is_refused(self):
        """An empty ``color_field`` is refused, as the docstring's Raises promises.

        Test scenario:
            ``add_layer(im, color_field="")`` raises ``ValueError`` from ``Encoding.by_field`` — a colour
            encoding needs a non-empty field name, so an empty one is rejected rather than published.
        """
        m = Map()
        try:
            image = self._image(m)
            with pytest.raises(ValueError, match="non-empty"):
                m.add_layer(image, name="bad", color_field="")
        finally:
            m.close()

    def test_color_field_none_leaves_the_layer_keyless(self):
        """The default (``color_field=None``) keeps the pre-#349 keyless custom layer.

        Test scenario:
            ``add_layer(im, name="plain")`` with no ``color_field`` records no colour encoding, so the layer
            carries no key.
        """
        m = Map()
        try:
            m.add_layer(self._image(m), name="plain")
            encoding = m._layer_tree.get("plain").symbology.encoding("color")
            assert encoding is None, (
                f"a keyless custom layer should carry no colour encoding, got {encoding!r}"
            )
        finally:
            m.close()
