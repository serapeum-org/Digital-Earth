"""The layer-kind registry and its engine-neutral vocabulary (DE-25a, #288).

`LayerSpec.kind` was a checked spelling with nothing behind it, and the one tier that wrote kinds wrote MapLibre
layer types — `fill` for both plain polygons and a choropleth.
"""

import pytest

from digitalearth.base.registry import (
    KIND_TAKES,
    KindInfo,
    is_kind_name,
    kind_info,
    kinds,
    register_kind,
    temporary_kind,
)
from digitalearth.base.spec import LayerSpec


class TestTheBuiltInVocabulary:
    """What every tier can write into a `LayerSpec` without registering anything."""

    @pytest.mark.parametrize(
        "name",
        [
            "raster",
            "mesh",
            "rgb",
            "contours",
            "filled_contours",
            "vectors",
            "streamlines",
            "terrain",
            "volume",
            "isosurface",
            "points",
            "point_cloud",
            "heatmap",
            "clusters",
            "labels",
            "lines",
            "flow",
            "polygons",
            "choropleth",
            "extrusion",
            "unstructured",
            "text",
            "graticule",
            "basemap",
            "coastlines",
            "borders",
            "land",
            "ocean",
            "lakes",
            "rivers",
            "model",
        ],
    )
    def test_each_built_in_kind_is_registered(self, name):
        """Every name in the vocabulary resolves to its own entry.

        Args:
            name: A built-in kind.
        """
        assert kind_info(name).name == name, kind_info(name)

    def test_every_built_in_kind_names_what_it_takes_and_documents_itself(self):
        """A kind says which data it draws and what it is, so a registry listing is readable."""
        undocumented = [
            name
            for name in kinds()
            if kind_info(name).takes not in KIND_TAKES or not kind_info(name).doc
        ]
        assert undocumented == [], undocumented

    def test_polygons_and_a_choropleth_are_different_kinds(self):
        """The two layers the web tree used to record identically as `fill` are distinct kinds."""
        polygons, choropleth = kind_info("polygons"), kind_info("choropleth")
        assert polygons.name != choropleth.name, (polygons, choropleth)

    def test_the_listing_is_sorted_and_a_copy(self):
        """`kinds()` returns names in order, as a tuple nobody can edit the registry through."""
        listed = kinds()
        assert isinstance(listed, tuple), type(listed)
        assert list(listed) == sorted(listed), listed


class TestLookingUpAKind:
    """Reaching an entry, and the message for one that does not exist."""

    def test_an_unknown_kind_raises_key_error_naming_what_is_registered(self):
        """The message lists the registered kinds, so a typo is fixed without opening the source."""
        with pytest.raises(
            KeyError, match=r"no layer kind 'polygon' is registered.*'polygons'"
        ):
            kind_info("polygon")

    def test_an_unhashable_name_is_reported_as_unknown(self):
        """A list is not a kind; the lookup answers with the same `KeyError` rather than a `TypeError`.

        Test scenario:
            A dict lookup with an unhashable key raises `TypeError`, which would name neither the kind nor the
            registered ones.
        """
        name = ["points"]
        with pytest.raises(KeyError, match="no layer kind"):
            kind_info(name)


class TestRegisteringAKind:
    """How a plugin or an engine-specific layer adds a kind."""

    def test_a_temporary_kind_resolves_inside_the_block_and_is_gone_after(self):
        """The scoped form leaves the process registry as it found it."""
        info = KindInfo("demo:hexbin", "points", "binned point density from a plugin")
        with temporary_kind(info):
            inside = kind_info("demo:hexbin")
        assert inside == info, inside
        assert "demo:hexbin" not in kinds(), kinds()

    def test_a_temporary_kind_restores_the_entry_it_replaced(self):
        """Replacing a built-in for a block puts the built-in back afterwards."""
        original = kind_info("points")
        replacement = KindInfo(
            "points", "points", "a replacement used only in this test"
        )
        with temporary_kind(replacement):
            swapped = kind_info("points")
        assert swapped == replacement, swapped
        assert kind_info("points") == original, kind_info("points")

    def test_registering_the_same_entry_twice_is_harmless(self):
        """Re-registering an identical entry — a module imported twice — changes nothing."""
        info = kind_info("raster")
        register_kind(info)
        assert kind_info("raster") == info, kind_info("raster")

    def test_a_different_entry_under_a_taken_name_is_refused(self):
        """Two plugins claiming one name would silently overwrite each other; the second is refused."""
        clash = KindInfo("raster", "points", "a different meaning for a taken name")
        with pytest.raises(ValueError, match="'raster' is already registered"):
            register_kind(clash)

    @pytest.mark.parametrize("entry", [("points", "points", "doc"), "points", None])
    def test_register_kind_needs_a_kind_info(self, entry):
        """Only a built `KindInfo` is registered, so its checks cannot be skipped by passing a tuple or a name.

        Args:
            entry: A value that is not a `KindInfo`.
        """
        with pytest.raises(TypeError, match="register_kind needs a KindInfo"):
            register_kind(entry)

    def test_temporary_kind_needs_a_kind_info(self):
        """The scoped form refuses a non-entry too, before touching the registry."""
        before = kinds()
        with pytest.raises(TypeError, match="temporary_kind needs a KindInfo"):
            with temporary_kind("points"):
                pass
        assert kinds() == before, (
            "a refused temporary registration changed the registry"
        )

    @pytest.mark.parametrize(
        "name, takes, doc, message",
        [
            ("Hexbin", "points", "doc", "not a layer-kind name"),
            ("a:b:c", "points", "doc", "not a layer-kind name"),
            ("hexbin", "cells", "doc", "takes must be one of"),
            ("hexbin", "points", "", "needs a description"),
        ],
        ids=["upper-case", "two-namespaces", "unknown-takes", "no-doc"],
    )
    def test_an_entry_that_could_not_be_used_is_refused_when_built(
        self, name, takes, doc, message
    ):
        """A malformed entry fails where it is written, not when a renderer looks it up.

        Args:
            name: The kind name.
            takes: The data class it draws.
            doc: Its description.
            message: The fragment the error names.
        """
        with pytest.raises(ValueError, match=message):
            KindInfo(name, takes, doc)


class TestTheKindGrammar:
    """One spelling rule, shared by the registry and `LayerSpec`."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("points", True),
            ("filled_contours", True),
            ("fill-extrusion", True),
            ("custom:pyvista", True),
            ("mypkg:hex-bin", True),
            (":pyvista", False),
            ("custom:", False),
            ("a:b:c", False),
            ("Custom:pyvista", False),
            ("custom:PyVista", False),
            ("1points", False),
            ("", False),
            (None, False),
        ],
    )
    def test_is_kind_name(self, name, expected):
        """A kind is a lowercase identifier, optionally after one lowercase namespace and a colon.

        Args:
            name: The spelling under test.
            expected: Whether it is a valid kind name.
        """
        assert is_kind_name(name) is expected, name

    def test_a_layer_accepts_a_namespaced_kind_and_round_trips_it(self):
        """`LayerSpec` uses the same rule, so a custom layer's kind is valid and survives JSON."""
        layer = LayerSpec("wells", "custom:pyvista")
        stored = layer.to_dict()
        assert LayerSpec.from_dict(stored).kind == "custom:pyvista", stored

    def test_a_layer_refuses_a_kind_with_two_namespaces(self):
        """Only one namespace is allowed, so `a:b:c` is refused by the layer too."""
        with pytest.raises(ValueError, match="kind must be"):
            LayerSpec("a", "a:b:c")
