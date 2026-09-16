"""`LayerSpec` and `LayerTree` — a layer that describes itself, addressed by id (DE-20, #281).

Every tier kept a list called `layers`, three of them addressed it by position, and none kept what a layer is.
These cover the description that replaces that, and the tree that orders it.
"""

import json

import pytest

from digitalearth.base.spec import (
    LAYER_REFERENCE,
    LayerSpec,
    LayerTree,
    Scale,
    Selection,
    Symbology,
)


def _tree(*ids):
    """Return a tree of plain point layers with these ids, bottom first."""
    return LayerTree(tuple(LayerSpec(layer_id, "points") for layer_id in ids))


class TestLayerSpec:
    """What one layer is, without drawing it."""

    def test_a_layer_defaults_to_visible_with_no_source_slice_or_style(self):
        """A minimal layer is an id and a kind; everything else has a stated default."""
        layer = LayerSpec("grid", "graticule")
        assert layer.source_id is None, "a graticule draws from no data source"
        assert layer.selection == Selection(), "the default slice is the default band"
        assert layer.symbology == Symbology(), "no styling means an empty symbology"
        assert layer.visible is True, "a layer is drawn unless switched off"

    @pytest.mark.parametrize("bad_id", ["", "   ", " dem", "dem ", 7, None])
    def test_an_id_that_cannot_address_a_layer_is_refused(self, bad_id):
        """An empty, padded or non-string id is refused.

        Args:
            bad_id: The id under test.

        Test scenario:
            The id is the only way a layer is addressed, so `" dem"` and `"dem"` naming two layers would be a lookup
            that fails for a reason nobody can see.
        """
        with pytest.raises(ValueError, match="LayerSpec needs an id"):
            LayerSpec(bad_id, "raster")

    @pytest.mark.parametrize(
        "bad_kind", ["", "Raster", "1raster", "raster layer", None]
    )
    def test_a_kind_that_is_not_a_lowercase_identifier_is_refused(self, bad_kind):
        """Kinds become registry keys, so a spelling that could not be one is refused now.

        Args:
            bad_kind: The kind under test.
        """
        with pytest.raises(ValueError, match="kind must be a lowercase identifier"):
            LayerSpec("a", bad_kind)

    @pytest.mark.parametrize(
        "kind", ["raster", "fill-extrusion", "rgb_composite", "h3"]
    )
    def test_a_lowercase_identifier_kind_is_accepted(self, kind):
        """Hyphens, underscores and digits after the first letter are all fine.

        Args:
            kind: The kind under test.
        """
        assert LayerSpec("a", kind).kind == kind

    @pytest.mark.parametrize("visible", ["False", 0, 1, None])
    def test_visible_must_be_a_real_boolean(self, visible):
        """`"False"` is truthy, so only a real boolean says what it means.

        Args:
            visible: The non-boolean under test.
        """
        with pytest.raises(ValueError, match="visible must be True or False"):
            LayerSpec("a", "points", visible=visible)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("source_id", ""),
            ("z_source", "  "),
            ("label", ""),
            ("group", 3),
            ("filter", ""),
        ],
    )
    def test_an_optional_string_that_is_set_must_not_be_empty(self, field, value):
        """An optional string is `None` or says something.

        Args:
            field: The field under test.
            value: An empty or non-string value for it.
        """
        with pytest.raises(ValueError, match=f"needs {field} as a non-empty string"):
            LayerSpec("a", "points", **{field: value})

    def test_a_selection_of_the_wrong_type_is_refused(self):
        """A bare band number is not a `Selection`, and a later `.band` would fail far from here."""
        with pytest.raises(ValueError, match="selection must be a Selection"):
            LayerSpec("a", "raster", selection=2)

    def test_a_symbology_of_the_wrong_type_is_refused(self):
        """A flat style dict is not a `Symbology`."""
        with pytest.raises(ValueError, match="symbology must be a Symbology"):
            LayerSpec("a", "raster", symbology={"color": "#f00"})

    def test_a_layer_cannot_drape_over_itself(self):
        """`z_source="layer:<own id>"` is refused where it is written."""
        with pytest.raises(ValueError, match="cannot take its elevation from itself"):
            LayerSpec("dem", "raster", z_source="layer:dem")

    def test_a_layer_reference_must_name_a_layer(self):
        """`"layer:"` with nothing after it names no layer."""
        with pytest.raises(ValueError, match="names no layer"):
            LayerSpec("imagery", "rgb", z_source=LAYER_REFERENCE)

    @pytest.mark.parametrize(
        "z_source, expected", [(None, None), ("srtm", None), ("layer:dem", "dem")]
    )
    def test_z_layer_names_only_a_layer_reference(self, z_source, expected):
        """A flat layer or a source-backed elevation names no layer; a drape names the layer beneath.

        Args:
            z_source: The elevation source under test.
            expected: The layer id `z_layer` should report.
        """
        assert LayerSpec("a", "rgb", z_source=z_source).z_layer == expected

    def test_display_label_falls_back_to_the_id(self):
        """A switcher needs something to show, and the id is the honest fallback."""
        assert LayerSpec("roads", "lines").display_label == "roads"
        assert LayerSpec("roads", "lines", label="Roads").display_label == "Roads"

    def test_to_dict_writes_only_what_differs_from_the_default(self):
        """A plain layer is two keys, so a stored figure does not fill with defaults."""
        assert LayerSpec("a", "points").to_dict() == {"id": "a", "kind": "points"}

    def test_a_fully_specified_layer_survives_a_json_round_trip(self):
        """Every field comes back through real JSON text.

        Test scenario:
            The rebuilt side is made by `from_dict` on parsed JSON and the other by the constructor, so this is a
            statement about serialisation rather than about `==` on one object.
        """
        layer = LayerSpec(
            "imagery",
            "rgb",
            source_id="s2",
            selection=Selection.of((4, 3, 2), time="2024-06"),
            symbology=Symbology.of(opacity=0.8).with_props(stretch=[2, 98]),
            z_source="layer:dem",
            visible=False,
            label="Sentinel-2",
            group="imagery",
            filter="cloud < 20",
        )
        rebuilt = LayerSpec.from_dict(json.loads(json.dumps(layer.to_dict())))
        assert rebuilt == layer, f"the layer changed in a round trip: {rebuilt!r}"

    def test_from_dict_refuses_an_unknown_key(self):
        """A key a newer writer added is refused, not dropped."""
        stored = {"id": "a", "kind": "points", "order": 3}
        with pytest.raises(ValueError, match=r"unknown keys \['order'\]"):
            LayerSpec.from_dict(stored)

    def test_from_dict_names_a_missing_kind(self):
        """A stored layer without a kind cannot be drawn, and the message says which key is missing."""
        stored = {"id": "a"}
        with pytest.raises(ValueError, match="needs 'kind'"):
            LayerSpec.from_dict(stored)


class TestLayerTreeReading:
    """Looking layers up by id rather than by position."""

    def test_ids_are_in_draw_order_bottom_first(self):
        """The first layer is drawn first and so sits beneath the rest."""
        assert _tree("tiles", "dem", "roads").ids == ("tiles", "dem", "roads")

    def test_len_iteration_and_membership(self):
        """A tree reads like the ordered collection it is."""
        tree = _tree("a", "b")
        assert len(tree) == 2
        assert [layer.id for layer in tree] == ["a", "b"]
        assert "a" in tree
        assert "z" not in tree

    def test_get_an_unknown_id_names_the_ids_that_exist(self):
        """A miss lists what is there, so a typo is visible."""
        tree = _tree("dem", "roads")
        with pytest.raises(KeyError, match=r"no layer 'rods'.*\['dem', 'roads'\]"):
            tree.get("rods")

    def test_groups_are_listed_once_in_the_order_they_first_appear(self):
        """Two layers in one group make one group."""
        tree = LayerTree(
            (
                LayerSpec("a", "points", group="obs"),
                LayerSpec("b", "points"),
                LayerSpec("c", "lines", group="rivers"),
                LayerSpec("d", "points", group="obs"),
            )
        )
        assert tree.groups == ("obs", "rivers")


class TestLayerTreeValidation:
    """A tree whose ids and references do not hold together is refused."""

    def test_two_layers_sharing_an_id_are_refused(self):
        """Identity is the whole point; a duplicate id would make `get` answer with one of two layers."""
        layers = (LayerSpec("a", "points"), LayerSpec("a", "lines"))
        with pytest.raises(ValueError, match=r"\['a'\] appear more than once"):
            LayerTree(layers)

    def test_a_drape_over_a_layer_that_is_not_there_is_refused(self):
        """A `layer:` reference must resolve inside the tree."""
        layers = (LayerSpec("imagery", "rgb", z_source="layer:dem"),)
        with pytest.raises(ValueError, match="which is not in the tree"):
            LayerTree(layers)

    def test_a_loop_of_drapes_is_refused(self):
        """A chain of `layer:` references must end on a surface, not come back round.

        Test scenario:
            `a` drapes over `b` and `b` over `a`: no layer in the chain has an elevation of its own, so a renderer
            following the chain would never stop.
        """
        layers = (
            LayerSpec("a", "rgb", z_source="layer:b"),
            LayerSpec("b", "rgb", z_source="layer:a"),
        )
        with pytest.raises(ValueError, match="references loop"):
            LayerTree(layers)

    def test_a_hidden_group_nobody_belongs_to_is_refused(self):
        """Hiding a group that does not exist is a typo, not a no-op."""
        layers = (LayerSpec("a", "points", group="obs"),)
        hidden = frozenset({"ob"})
        with pytest.raises(ValueError, match=r"hides groups \['ob'\]"):
            LayerTree(layers, hidden)

    def test_something_that_is_not_a_layer_is_refused(self):
        """A tree holds `LayerSpec` values, not engine handles."""
        with pytest.raises(ValueError, match="holds LayerSpec values"):
            LayerTree(("raster-layer",))

    def test_sequences_are_frozen(self):
        """A list passed in is stored as a tuple, so the tree cannot be re-ordered from outside."""
        tree = LayerTree([LayerSpec("a", "points")], set())
        assert isinstance(tree.layers, tuple), type(tree.layers)
        assert isinstance(tree.hidden_groups, frozenset), type(tree.hidden_groups)


class TestLayerTreeChanges:
    """Each change returns a new tree and leaves the old one as it was."""

    def test_add_puts_a_layer_on_top_by_default(self):
        """New layers draw over what is there."""
        assert _tree("a").add(LayerSpec("b", "lines")).ids == ("a", "b")

    def test_add_at_index_zero_puts_a_layer_beneath_everything(self):
        """Where a basemap goes — without shifting anything's identity, which a position would."""
        assert _tree("a", "b").add(LayerSpec("tiles", "tiles"), index=0).ids == (
            "tiles",
            "a",
            "b",
        )

    @pytest.mark.parametrize("index", [-1, 3, True])
    def test_add_refuses_a_position_outside_the_tree(self, index):
        """`list.insert` would clamp, putting the layer somewhere the caller did not ask for.

        Args:
            index: The out-of-range position under test.
        """
        tree = _tree("a", "b")
        layer = LayerSpec("c", "points")
        with pytest.raises(IndexError, match="positions run from 0"):
            tree.add(layer, index=index)

    def test_add_refuses_a_duplicate_id(self):
        """Adding an id already in the tree is refused rather than shadowing the first layer."""
        tree = _tree("a")
        layer = LayerSpec("a", "lines")
        with pytest.raises(ValueError, match="already in the tree"):
            tree.add(layer)

    def test_a_change_leaves_the_original_tree_unchanged(self):
        """A renderer compares the tree it drew with the tree it is given, so the old one must not move."""
        before = _tree("a", "b")
        after = (
            before.move("a", -1).set_visible("b", False).add(LayerSpec("c", "points"))
        )
        assert before.ids == ("a", "b"), before.ids
        assert before.is_visible("b"), "the original tree's layer is still visible"
        assert after.ids == ("b", "a", "c"), after.ids

    def test_remove_drops_the_layer(self):
        """The layer is gone and the rest keep their order."""
        assert _tree("a", "b", "c").remove("b").ids == ("a", "c")

    def test_remove_an_unknown_id_is_a_key_error(self):
        """Removing something that is not there is a mistake worth hearing about."""
        tree = _tree("a")
        with pytest.raises(KeyError, match="no layer 'z'"):
            tree.remove("z")

    def test_remove_refuses_a_surface_something_is_draped_over(self):
        """Removing the terrain would leave the imagery pointing at nothing."""
        tree = LayerTree(
            (
                LayerSpec("dem", "raster"),
                LayerSpec("imagery", "rgb", z_source="layer:dem"),
            )
        )
        with pytest.raises(
            ValueError, match=r"while \['imagery'\] take their elevation"
        ):
            tree.remove("dem")

    def test_removing_the_last_layer_of_a_hidden_group_forgets_the_group(self):
        """A hidden group with nothing left in it would otherwise make the tree invalid."""
        tree = LayerTree((LayerSpec("a", "points", group="obs"),), frozenset({"obs"}))
        assert tree.remove("a").hidden_groups == frozenset()

    @pytest.mark.parametrize(
        "index, expected",
        [
            (0, ("c", "a", "b")),
            (-1, ("a", "b", "c")),
            (1, ("a", "c", "b")),
            (-3, ("c", "a", "b")),
        ],
    )
    def test_move_places_a_layer_at_a_list_style_position(self, index, expected):
        """`0` is the bottom and `-1` the top, as with a list; the others keep their relative order.

        Args:
            index: The target position.
            expected: The ids afterwards.
        """
        assert _tree("a", "b", "c").move("c", index).ids == expected

    @pytest.mark.parametrize("index", [3, -4])
    def test_move_refuses_a_position_outside_the_tree(self, index):
        """A position past either end is refused rather than wrapped.

        Args:
            index: The out-of-range position.
        """
        tree = _tree("a", "b", "c")
        with pytest.raises(IndexError, match="holds 3 layers"):
            tree.move("a", index)

    def test_replace_restyles_a_layer_in_place(self):
        """The id and the position stay; the description changes — the restyle-after-draw #188 needs."""
        tree = _tree("a", "b", "c")
        styled = LayerSpec("b", "points", symbology=Symbology.of(color="#f00"))
        after = tree.replace(styled)
        assert after.ids == ("a", "b", "c"), after.ids
        assert after.get("b") == styled, after.get("b")

    def test_replace_an_unknown_id_is_a_key_error(self):
        """A replacement names the layer it replaces, which has to exist."""
        tree = _tree("a")
        layer = LayerSpec("z", "points")
        with pytest.raises(KeyError, match="no layer 'z'"):
            tree.replace(layer)

    def test_replace_cannot_introduce_a_dangling_drape(self):
        """The tree's own validation applies to the result."""
        tree = _tree("a")
        layer = LayerSpec("a", "rgb", z_source="layer:missing")
        with pytest.raises(ValueError, match="which is not in the tree"):
            tree.replace(layer)

    def test_set_visible_switches_one_layer(self):
        """A switched-off layer stays in the tree, so it can come back."""
        hidden = _tree("a", "b").set_visible("a", False)
        assert hidden.is_visible("a") is False
        assert hidden.is_visible("b") is True
        assert hidden.ids == ("a", "b"), "hiding a layer does not remove it"

    def test_set_visible_refuses_a_non_boolean(self):
        """The layer's own validation refuses `"no"`."""
        tree = _tree("a")
        with pytest.raises(ValueError, match="visible must be True or False"):
            tree.set_visible("a", "no")


class TestGroupVisibility:
    """A group hides its layers without forgetting their own visibility."""

    @pytest.fixture
    def tree(self):
        """Two layers in `obs`, one of them already switched off, and one layer in no group."""
        return LayerTree(
            (
                LayerSpec("a", "points", group="obs"),
                LayerSpec("b", "points", group="obs", visible=False),
                LayerSpec("c", "lines"),
            )
        )

    def test_hiding_a_group_hides_every_layer_in_it(self, tree):
        """Neither `obs` layer is drawn; the ungrouped one is."""
        hidden = tree.set_group_visible("obs", False)
        assert [hidden.is_visible(layer_id) for layer_id in ("a", "b", "c")] == [
            False,
            False,
            True,
        ]

    def test_showing_the_group_again_restores_what_each_layer_said(self, tree):
        """`b` was off before the group was hidden, so it stays off afterwards.

        Test scenario:
            Implemented by flipping each layer's own flag, hiding and re-showing the group would switch `b` on — a
            layer the caller had turned off coming back by itself.
        """
        restored = tree.set_group_visible("obs", False).set_group_visible("obs", True)
        assert restored.is_visible("a") is True
        assert restored.is_visible("b") is False

    def test_an_unknown_group_is_a_key_error(self, tree):
        """A typo in a group name is reported with the groups that exist."""
        with pytest.raises(
            KeyError, match=r"no layer belongs to group 'ob'.*\['obs'\]"
        ):
            tree.set_group_visible("ob", False)

    def test_group_visibility_must_be_a_boolean(self, tree):
        """The same rule as a layer's own flag."""
        with pytest.raises(ValueError, match="visible as True or False"):
            tree.set_group_visible("obs", 0)


class TestLayerTreeSerialisation:
    """A tree round-trips, order and hidden groups included."""

    def test_a_tree_survives_a_json_round_trip(self):
        """Order, drapes, groups, hidden groups and styles all come back.

        Test scenario:
            Built by chaining changes on one side and by `from_dict` on parsed JSON on the other.
        """
        tree = (
            LayerTree()
            .add(LayerSpec("dem", "raster", source_id="srtm"))
            .add(LayerSpec("imagery", "rgb", source_id="s2", z_source="layer:dem"))
            .add(
                LayerSpec(
                    "stations",
                    "points",
                    source_id="obs",
                    group="obs",
                    symbology=Symbology.of(color="#f00").with_props(radius=4),
                )
            )
            .set_group_visible("obs", False)
            .move("imagery", -1)
        )
        rebuilt = LayerTree.from_dict(json.loads(json.dumps(tree.to_dict())))
        assert rebuilt == tree, f"the tree changed in a round trip: {rebuilt.to_dict()}"

    def test_a_scale_inside_a_layer_survives_too(self):
        """The deepest nesting a layer carries — a field-driven encoding's scale — round-trips."""
        from digitalearth.base.spec import Encoding

        encoding = Encoding.by_field(
            "color", "t2m", scale=Scale.from_limits(250.0, 310.0)
        )
        tree = LayerTree(
            (LayerSpec("t", "raster", symbology=Symbology({"color": encoding})),)
        )
        rebuilt = LayerTree.from_dict(json.loads(json.dumps(tree.to_dict())))
        assert rebuilt.get("t").symbology.encoding("color").scale == Scale(250.0, 310.0)

    def test_an_empty_tree_is_one_key(self):
        """No layers, nothing hidden."""
        assert LayerTree().to_dict() == {"layers": []}

    def test_from_dict_refuses_an_unknown_key(self):
        """A key a newer writer added is refused, not dropped."""
        stored = {"layers": [], "order": ["a"]}
        with pytest.raises(ValueError, match=r"unknown keys \['order'\]"):
            LayerTree.from_dict(stored)
