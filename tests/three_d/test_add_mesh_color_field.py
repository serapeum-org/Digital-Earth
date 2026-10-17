"""Tests for ``add_mesh(color_field=)`` on the 3-D tier (#349).

A custom layer — a caller's own PyVista mesh — publishes no colour encoding of its own, so ``colorbar()`` and
``legend()`` have nothing to describe. ``color_field=`` is the one producer that was missing: it names what the
mesh's colour varies by, publishing ``Encoding.by_field("color", color_field)`` so the guides work on the escape
hatch the builders do not cover. This mirrors the static tier's ``add_layer(color_field=)`` (see
``tests/static/test_add_layer_color_field.py``).

Gated on the optional ``3d`` extra (pyvista), as every module in this directory is.
"""

import pytest

from digitalearth.three_d.base import Scene3DBase

pv = pytest.importorskip("pyvista")


class TestAddMeshColorField:
    """The ``color_field`` keyword on :meth:`Scene3DBase.add_mesh`."""

    def test_color_field_publishes_a_colour_encoding_naming_the_field(self):
        """``color_field`` records ``Encoding.by_field("color", color_field)`` on the custom layer.

        Test scenario:
            ``add_mesh(pv.Sphere(), name="keyed", color_field="elevation")`` makes the layer's colour encoding
            carry the field ``"elevation"`` — the name a key is titled after — read back via ``get_layer``.
        """
        scene = Scene3DBase(off_screen=True)
        try:
            scene.add_mesh(pv.Sphere(), name="keyed", color_field="elevation")
            encoding = scene.get_layer("keyed").symbology.encoding("color")
            assert encoding is not None, "color_field should publish a colour encoding"
            assert encoding.field == "elevation", (
                f"expected field 'elevation', got {encoding.field!r}"
            )
        finally:
            scene.close()

    def test_a_custom_mesh_without_color_field_has_no_colour_encoding(self):
        """The default (``color_field=None``) keeps the custom layer keyless.

        Test scenario:
            ``add_mesh(pv.Sphere(), name="plain")`` records no colour encoding, so ``encoding("color")`` is
            ``None`` — a key over it would have nothing to describe.
        """
        scene = Scene3DBase(off_screen=True)
        try:
            scene.add_mesh(pv.Sphere(), name="plain")
            encoding = scene.get_layer("plain").symbology.encoding("color")
            assert encoding is None, (
                f"a keyless custom mesh should carry no colour encoding, got {encoding!r}"
            )
        finally:
            scene.close()

    def test_an_empty_color_field_is_refused(self):
        """An empty ``color_field`` is refused by ``Encoding.by_field``.

        Test scenario:
            ``add_mesh(mesh, color_field="")`` raises ``ValueError`` — a colour encoding needs a non-empty
            field name, so an empty one is rejected rather than published. The mesh is built outside the
            ``pytest.raises`` block so only the one refused call is inside it.
        """
        scene = Scene3DBase(off_screen=True)
        try:
            mesh = pv.Sphere()
            with pytest.raises(ValueError, match="non-empty"):
                scene.add_mesh(mesh, name="bad", color_field="")
        finally:
            scene.close()
