"""`to_dict` / `from_dict` on the Wave 1 value types (DE-20, #281).

A `LayerSpec` can only serialise if everything it holds does, and before Wave 3 only `DataRef` could. These cover
the pairs `Bounds`, `Selection`, `Scale`, `Encoding` and `Symbology` gained, and the shared rules in
`base/spec/_serial.py` they are built on.
"""

import copy
import dataclasses
import datetime
import enum
import json
import pickle
import re

import numpy as np
import pytest

from digitalearth.base.spec import (
    Bounds,
    Camera,
    DataRef,
    Encoding,
    FigureSpec,
    LayerSpec,
    LayerTree,
    PanelSpec,
    RenderTarget,
    Scale,
    Selection,
    StyleKey,
    StyleSchema,
    Symbology,
    Viewport,
    ViewRequest,
)
from digitalearth.base.spec._serial import (
    MAX_TRAVELLING_ELEMENTS,
    FrozenDict,
    crs_to_json,
    finite_number,
    frozen_value,
    hashable_value,
    read_entry,
    thawed_value,
    to_json_value,
    travels_in_a_figure,
)

#: A minimal valid panel list, for the figure reads that fail on another field.
_PANELS = [{"id": "p", "viewport": {"crs": 3857}}]


class _SubSymbology(Symbology):
    """A caller's subclass of `Symbology`, to check copies keep it."""


class _SubStyleSchema(StyleSchema):
    """A caller's subclass of `StyleSchema`, to check copies keep it."""


class _SubFigure(FigureSpec):
    """A caller's subclass of `FigureSpec`, to check copies keep it."""


class _SubTree(LayerTree):
    """A caller's subclass of `LayerTree`, to check its changes keep it."""


def _foreign_types(written, where="to_dict()"):
    """List every value in a written dict that is not a plain Python JSON type, with where it sits.

    Args:
        written: A `to_dict` result, or any part of one.
        where: The path to `written`, for the report.

    Returns:
        ``(path, type name)`` pairs — a `numpy.str_` key or value among them — empty when the dict is plain.
    """
    if isinstance(written, dict):
        found = [
            (f"{where} key {key!r}", type(key).__name__)
            for key in written
            if type(key) is not str
        ]
        for key, item in written.items():
            found.extend(_foreign_types(item, f"{where}[{key!r}]"))
        return found
    if isinstance(written, list):
        return [
            pair
            for index, item in enumerate(written)
            for pair in _foreign_types(item, f"{where}[{index}]")
        ]
    if written is None or type(written) in (bool, int, float, str):
        return []
    return [(where, type(written).__name__)]


def _through_json(value):
    """Round-trip `value` through its dict form and real JSON text."""
    return type(value).from_dict(json.loads(json.dumps(value.to_dict())))


def _raise_a_decode_error(value):
    """Refuse `value` with a `ValueError` subclass whose constructor needs more than a message.

    Args:
        value: Ignored.

    Raises:
        json.JSONDecodeError: always.
    """
    raise json.JSONDecodeError("Expecting value", "doc", 0)


class TestEachTypeRoundTrips:
    """What goes out through JSON comes back equal."""

    @pytest.mark.parametrize(
        "value",
        [
            Bounds(0.0, 1.0, 2.0, 3.0, crs=4326),
            Bounds(-10.5, -5.0, 10.5, 5.0, crs="EPSG:3857"),
            Selection(),
            Selection.of(
                (3, 2, 1), time="2024-01", level=850, member=0, overview=2, budget=1000
            ),
            Scale.from_limits(0.0, 10.0),
            Scale(0.0, 9.0, scheme="equal_interval", breaks=(0.0, 3.0, 6.0, 9.0)),
            Scale(0.0, 9.0, scheme=(0.0, 4.5, 9.0), breaks=(0.0, 4.5, 9.0)),
            Scale.categorical(["a", 1], ["#f00", "#00f"], missing="#ccc"),
            Encoding.constant("opacity", 0.5),
            Encoding.by_field(
                "size", "pop", scale=Scale.from_limits(0, 100), output_range=(1.0, 9.0)
            ),
            Symbology(),
            Symbology.of(color="#f00", opacity=0.5).with_props(
                levels=[1, 2, 3], hillshade=True
            ),
        ],
        ids=lambda value: type(value).__name__,
    )
    def test_a_value_survives_a_json_round_trip(self, value):
        """The value rebuilt from JSON text equals the one written.

        Args:
            value: The spec value under test, built through its normal constructor or builder.

        Test scenario:
            The two sides are built by different routes — one by the constructor or a builder, the other by
            `from_dict` on parsed JSON — so the assertion is about serialisation, not about `==` on one object.
            JSON has no tuple, so every value here holds lists or scalars where the round trip must match.
        """
        rebuilt = _through_json(value)
        assert rebuilt == value, (
            f"{type(value).__name__} changed in a round trip: {value!r} -> {rebuilt!r}"
        )

    @pytest.mark.parametrize(
        "value",
        [
            Symbology.of(color=(1, 0, 0)),
            Symbology().with_props(levels=(1, 2, 3)),
            Selection.of(1, time=("2020", "2021")),
            Scale.categorical([(1, 2), (3, 4)], ["#f00", "#00f"]),
            Encoding.constant("color", (0.2, 0.4, 0.6)),
        ],
        ids=[
            "tuple-constant",
            "tuple-props",
            "tuple-time",
            "tuple-categories",
            "tuple-encoding",
        ],
    )
    def test_a_tuple_value_round_trips_equal_and_stays_hashable(self, value):
        """A tuple written as a JSON list reads back as the same value, which still hashes.

        Args:
            value: A spec value holding tuples in a free-form field.

        Test scenario:
            JSON has no tuple, so `to_dict` writes a list and `from_dict` read the list back as a list: the rebuilt
            value compared unequal to the one written, and a `Symbology` or `Selection` that had hashed raised
            `unhashable type: 'list'`. A reconciling renderer would read that as a change that never happened.
        """
        rebuilt = _through_json(value)
        assert rebuilt == value, (
            f"{type(value).__name__} changed in a round trip: {value!r} -> {rebuilt!r}"
        )
        assert hash(rebuilt) == hash(value), (
            "the rebuilt value must hash as the original did"
        )

    def test_a_dict_property_with_nested_tuples_round_trips_equal(self):
        """Sequences inside a dict property are canonical too, so it reads back equal; a dict does not hash."""
        value = Symbology().with_props(
            legend={"stops": (0.0, 1.0), "labels": ["lo", "hi"]}
        )
        rebuilt = _through_json(value)
        assert rebuilt == value, f"{rebuilt.props!r} != {value.props!r}"

    def test_a_list_and_a_tuple_spelling_are_one_value(self):
        """`[1, 0, 0]` and `(1, 0, 0)` build the same symbology, so either spelling hashes and compares equal."""
        assert Symbology.of(color=[1, 0, 0]) == Symbology.of(color=(1, 0, 0))
        assert hash(Symbology.of(color=[1, 0, 0])) == hash(
            Symbology.of(color=(1, 0, 0))
        )

    def test_a_categorical_scale_looks_up_a_list_spelled_category(self):
        """A category stored from a list still matches a query spelled as a list."""
        scale = Scale.categorical([[1, 2], [3, 4]], ["#f00", "#00f"])
        assert scale.color_for([3, 4]) == "#00f", scale.categories

    def test_a_classified_scale_reads_back_without_classifying_again(self, monkeypatch):
        """Stored breaks are the breaks; no classifier is consulted on read.

        Test scenario:
            A frozen scale exists so a sequence of frames keeps one set of colours. Re-classifying on read would
            let the colours move with whatever classifier happens to be registered.
        """
        import digitalearth.base.spec.scale as scale_module

        stored = {
            "vmin": 0.0,
            "vmax": 10.0,
            "scheme": "quantiles",
            "breaks": [0.0, 2.0, 10.0],
        }

        def refuse(*_args, **_kwargs):
            raise AssertionError("from_dict must not classify")

        monkeypatch.setattr(scale_module, "get_classifier", refuse)
        rebuilt = Scale.from_dict(stored)
        assert rebuilt.class_ranges() == [(0.0, 2.0), (2.0, 10.0)], (
            rebuilt.class_ranges()
        )


class TestWhatReadingRefuses:
    """`from_dict` refuses rather than guesses."""

    @pytest.mark.parametrize(
        "owner, stored",
        [
            (
                Bounds,
                {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1, "crs": 4326, "units": "m"},
            ),
            (Selection, {"band": [1], "stride": 2}),
            (Scale, {"vmin": 0, "vmax": 1, "ramp": "viridis"}),
            (Encoding, {"channel": "color", "value": "#f00", "legend": True}),
            (Symbology, {"encodings": {}, "theme": "dark"}),
        ],
        ids=lambda item: getattr(item, "__name__", ""),
    )
    def test_an_unknown_key_is_refused_not_dropped(self, owner, stored):
        """A key the type does not write is an error that names it.

        Args:
            owner: The type reading.
            stored: A dict carrying one key that type does not know.

        Test scenario:
            Dropping it would lose a newer writer's field silently, which is the failure `DataRef.from_dict`
            already refused; the other types follow the same rule.
        """
        with pytest.raises(ValueError, match="unknown keys"):
            owner.from_dict(stored)

    @pytest.mark.parametrize("owner", [Bounds, Selection, Scale, Encoding, Symbology])
    def test_something_that_is_not_a_mapping_is_refused_by_type(self, owner):
        """A list is not a stored spec, and says so.

        Args:
            owner: The type reading.
        """
        with pytest.raises(TypeError, match="needs a mapping"):
            owner.from_dict([1, 2, 3])

    @pytest.mark.parametrize(
        "owner, stored, missing",
        [
            (Bounds, {"xmin": 0, "ymin": 0, "xmax": 1}, "ymax"),
            (Scale, {"vmin": 0}, "vmax"),
            (Encoding, {"value": "#f00"}, "channel"),
        ],
    )
    def test_a_required_key_is_named_when_absent(self, owner, stored, missing):
        """A missing required field is reported by name, not as a bare `KeyError`.

        Args:
            owner: The type reading.
            stored: A dict lacking one required key.
            missing: The key it lacks.
        """
        with pytest.raises(ValueError, match=repr(missing)):
            owner.from_dict(stored)

    def test_what_the_constructor_refuses_from_dict_refuses_too(self):
        """`from_dict` goes through the constructor, so its validation applies.

        Test scenario:
            An inverted rectangle stored by hand must not come back as a value the constructor would refuse.
        """
        stored = {"xmin": 5, "ymin": 0, "xmax": 1, "ymax": 1, "crs": 4326}
        with pytest.raises(ValueError, match="xmin <= xmax"):
            Bounds.from_dict(stored)

    @pytest.mark.parametrize(
        "read, message",
        [
            (
                lambda: Camera.from_dict({"position": 5}),
                "Camera.from_dict needs 'position' as a list; got int 5",
            ),
            (
                lambda: PanelSpec.from_dict(
                    {"id": "p", "viewport": {"crs": 3857}, "layers": 5}
                ),
                "PanelSpec.from_dict needs 'layers' as a list",
            ),
            (
                lambda: FigureSpec.from_dict(
                    {"schema_version": 1, "panels": _PANELS, "size": 5}
                ),
                "FigureSpec.from_dict needs 'size' as a list",
            ),
            (
                lambda: FigureSpec.from_dict({"schema_version": 1, "panels": 5}),
                "FigureSpec.from_dict needs 'panels' as a list",
            ),
            (
                lambda: FigureSpec.from_dict(
                    {"schema_version": 1, "panels": _PANELS, "sources": 5}
                ),
                "FigureSpec.from_dict needs 'sources' as a mapping",
            ),
            (
                lambda: Selection.from_dict({"band": 2}),
                "Selection.from_dict needs 'band' as a list",
            ),
            (
                lambda: LayerTree.from_dict({"layers": 5}),
                "LayerTree.from_dict needs 'layers' as a list",
            ),
            (
                lambda: LayerTree.from_dict({"layers": [], "hidden_groups": "obs"}),
                "LayerTree.from_dict needs 'hidden_groups' as a list",
            ),
            (
                lambda: Scale.from_dict({"vmin": 0, "vmax": 1, "breaks": 5}),
                "Scale.from_dict needs 'breaks' as a list",
            ),
            (
                lambda: Encoding.from_dict(
                    {"channel": "size", "field": "p", "output_range": 5}
                ),
                "Encoding.from_dict needs 'output_range' as a list",
            ),
            (
                lambda: Symbology.from_dict({"encodings": 5}),
                "Symbology.from_dict needs 'encodings' as a mapping",
            ),
        ],
        ids=[
            "camera-position",
            "panel-layers",
            "figure-size",
            "figure-panels",
            "figure-sources",
            "selection-band",
            "tree-layers",
            "tree-hidden-groups",
            "scale-breaks",
            "encoding-output-range",
            "symbology-encodings",
        ],
    )
    def test_a_field_of_the_wrong_shape_is_named(self, read, message):
        """A stored value of the wrong shape is refused naming the type and the field.

        Args:
            read: A `from_dict` call over a dict with one wrongly shaped field.
            message: The start of the message that must name it.

        Test scenario:
            Each of these raised `TypeError: 'int' object is not iterable` from inside `tuple()` or `dict()`, naming
            neither the type being read nor the field. `"obs"` for `hidden_groups` is a string, which is iterable and
            would have become a set of letters, so it is refused as not a list too.
        """
        with pytest.raises(TypeError, match=re.escape(message)):
            read()

    @pytest.mark.parametrize(
        "entry, kind",
        [(None, "NoneType"), ("x.tif", "str"), (["x.tif"], "list")],
        ids=["none", "bare-path", "list"],
    )
    def test_a_source_entry_that_is_not_a_mapping_is_named(self, entry, kind):
        """A stored source that is not a mapping is refused as such, whatever shape it has.

        Args:
            entry: The stored value for one source.
            kind: The type name the message must report.

        Test scenario:
            Each entry went straight to `DataRef.from_dict`, which assumed a mapping: `None` raised
            `'NoneType' object is not iterable`, and a bare path was read as a mapping of its letters, reporting
            "unknown keys ['.', 'f', 'i', 't', 'x']" — a message that points at the wrong problem entirely.
        """
        stored = {"schema_version": 1, "panels": _PANELS, "sources": {"a": entry}}
        with pytest.raises(
            TypeError, match=f"DataRef.from_dict needs a mapping; got {kind}"
        ):
            FigureSpec.from_dict(stored)

    def test_a_source_entry_with_no_uri_names_the_missing_key(self):
        """A stored source without a `uri` says the key is missing, not that an empty string was given.

        Test scenario:
            A missing `uri` was read as ``""`` and refused as an empty uri, which reads as if the file had stored
            one.
        """
        stored = {"driver": "COG"}
        with pytest.raises(
            ValueError,
            match=re.escape("DataRef.from_dict needs 'uri'; got keys ['driver']"),
        ):
            DataRef.from_dict(stored)

    @pytest.mark.parametrize(
        "read, where",
        [
            (
                lambda: FigureSpec.from_dict(
                    {
                        "schema_version": 1,
                        "panels": _PANELS,
                        "sources": {"a": {"uri": "a.tif"}, "b": None},
                    }
                ),
                "FigureSpec.from_dict sources['b']: DataRef.from_dict needs a mapping",
            ),
            (
                lambda: FigureSpec.from_dict(
                    {"schema_version": 1, "panels": [*_PANELS, None]}
                ),
                "FigureSpec.from_dict panels[1]: PanelSpec.from_dict needs a mapping",
            ),
            (
                lambda: LayerTree.from_dict(
                    {"layers": [{"id": "a", "kind": "points"}, None]}
                ),
                "LayerTree.from_dict layers[1]: LayerSpec.from_dict needs a mapping",
            ),
            (
                lambda: Symbology.from_dict({"encodings": {"color": None}}),
                "Symbology.from_dict encodings['color']: Encoding.from_dict needs a mapping",
            ),
            (
                lambda: FigureSpec.from_dict(
                    {
                        "schema_version": 1,
                        "panels": [
                            {
                                "id": "p",
                                "viewport": {"crs": 3857, "bounds": {"xmin": 0}},
                            }
                        ],
                    }
                ),
                "FigureSpec.from_dict panels[0]: PanelSpec.from_dict viewport: Viewport.from_dict bounds: "
                "Bounds.from_dict needs 'ymin'",
            ),
        ],
        ids=[
            "figure-source",
            "figure-panel",
            "tree-layer",
            "symbology-encoding",
            "nested-path",
        ],
    )
    def test_a_refused_nested_entry_names_where_it_sits(self, read, where):
        """A stored part that is refused says which entry it was, down through every level it is nested in.

        Args:
            read: A `from_dict` call over a dict with one broken nested entry.
            where: The start of the message, naming the path to that entry.

        Test scenario:
            Each entry was handed straight to the reader for its type, whose message names only that type:
            "DataRef.from_dict needs a mapping; got NoneType" does not say which of a figure's sources is broken,
            and an error three levels down named only the innermost type.
        """
        with pytest.raises((TypeError, ValueError), match=re.escape(where)):
            read()

    def test_a_reader_error_of_a_subclass_is_raised_as_it_was(self):
        """A reader's `ValueError` subclass passes through `read_entry` with its own type and message.

        Test scenario:
            `read_entry` rebuilds a plain `TypeError` or `ValueError` from its message with the path in front.
            `json.JSONDecodeError` also needs `doc` and `pos`, so rebuilding it from a message alone raises a
            `TypeError` about its constructor in place of the reader's error.
        """
        with pytest.raises(json.JSONDecodeError) as caught:
            read_entry("FigureSpec", "sources['a']", _raise_a_decode_error, {})
        assert str(caught.value) == "Expecting value: line 1 column 1 (char 0)", str(
            caught.value
        )

    @pytest.mark.parametrize("edge", [None, "0"])
    def test_a_bounds_edge_that_is_not_a_number_is_named(self, edge):
        """`None` or a numeric string for an edge is refused naming the edge, not from inside `float()`.

        Args:
            edge: The stored value for `xmin`.
        """
        stored = {"xmin": edge, "ymin": 0, "xmax": 1, "ymax": 1, "crs": 4326}
        with pytest.raises(ValueError, match="Bounds needs xmin as a number"):
            Bounds.from_dict(stored)


class TestCopyingAndPickling:
    """The value types copy and pickle, including those holding read-only mappings."""

    @pytest.mark.parametrize("how", ["pickle", "deepcopy", "copy"])
    @pytest.mark.parametrize(
        "build",
        [
            lambda: Symbology.of(color="#f00").with_props(levels=(1, 2)),
            lambda: StyleSchema.of(StyleKey("cmap", "Colormap name.")),
            lambda: LayerTree(
                (LayerSpec("a", "raster", symbology=Symbology.of(opacity=0.5)),)
            ),
            lambda: FigureSpec(
                panels=(PanelSpec("p", layers=("a",)),),
                sources={"s": DataRef("a.tif")},
                layers=LayerTree((LayerSpec("a", "raster", source_id="s"),)),
            ),
        ],
        ids=["symbology", "style-schema", "layer-tree", "figure"],
    )
    def test_a_value_survives_pickling_and_copying(self, build, how):
        """`pickle`, `copy.deepcopy` and `copy.copy` give back an equal value.

        Args:
            build: Builds the value under test.
            how: Which of the three to use.

        Test scenario:
            `Symbology`, `StyleSchema` and `FigureSpec` store read-only mapping views, which cannot be pickled, so
            each of these raised `cannot pickle 'mappingproxy' object` — and so did everything holding one: a
            `LayerTree`, and a web map with a layer, which a notebook user deep-copies as a matter of course.
        """
        value = build()
        copied = {
            "pickle": lambda: pickle.loads(pickle.dumps(value)),
            "deepcopy": lambda: copy.deepcopy(value),
            "copy": lambda: copy.copy(value),
        }[how]()
        assert copied == value, f"{how} changed the value: {copied!r}"

    @pytest.mark.parametrize("how", ["pickle", "deepcopy", "copy"])
    def test_a_frozen_mapping_copies_as_a_frozen_mapping(self, how):
        """A bare `FrozenDict` pickles and copies to an equal `FrozenDict` that still refuses a change.

        Args:
            how: `pickle`, `copy.deepcopy` or `copy.copy`.

        Test scenario:
            The types holding one copy it as a plain dict through their own `__reduce__`, which never calls this
            one. The default reduction for a dict subclass refills the copy through `__setitem__`, which a
            `FrozenDict` refuses, so all three raised "read-only" without it.
        """
        original = FrozenDict({"levels": (1, 2)})
        copied = {
            "pickle": lambda: pickle.loads(pickle.dumps(original)),
            "deepcopy": lambda: copy.deepcopy(original),
            "copy": lambda: copy.copy(original),
        }[how]()
        assert type(copied) is FrozenDict, type(copied).__name__
        assert copied == {"levels": (1, 2)}, copied
        with pytest.raises(TypeError, match="read-only"):
            copied["levels"] = (3,)


class TestAsdictAndSubclasses:
    """`dataclasses.asdict` works on every type, and copying or changing a value keeps a caller's subclass."""

    @pytest.mark.parametrize(
        "build, reach",
        [
            (lambda: LayerSpec("a", "points"), lambda d: d["symbology"]["props"]),
            (
                lambda: LayerSpec("a", "points", symbology=Symbology.of(color="#f00")),
                lambda d: d["symbology"]["encodings"]["color"],
            ),
            (
                lambda: FigureSpec(
                    panels=(PanelSpec("p"),), sources={"s": DataRef("a.tif")}
                ),
                lambda d: d["sources"]["s"],
            ),
            (
                lambda: StyleSchema.of(StyleKey("cmap", "Colormap name.")),
                lambda d: d["keys"]["cmap"],
            ),
        ],
        ids=["plain-layer", "styled-layer", "figure-sources", "style-schema"],
    )
    def test_asdict_turns_the_frozen_mappings_into_dicts(self, build, reach):
        """`dataclasses.asdict` gives dicts all the way down, through the mappings the types freeze.

        A frozen mapping comes back as its own `dict` subclass, as `asdict` returns any dict subclass; the values in
        it are converted like any others.

        Args:
            build: Builds a value holding a frozen mapping.
            reach: Picks a part inside such a mapping out of the `asdict` result.

        Test scenario:
            The mappings were read-only `mappingproxy` views, which cannot be deep-copied, so `asdict` raised
            `cannot pickle 'mappingproxy' object` — for every `LayerSpec`, a plain one included, since an empty
            `Symbology` still holds two views. The `__reduce__` fix covered `pickle` and `copy` only.
        """
        part = reach(dataclasses.asdict(build()))
        assert isinstance(part, dict), type(part).__name__

    @pytest.mark.parametrize("how", ["pickle", "deepcopy"])
    @pytest.mark.parametrize(
        "build",
        [
            lambda: _SubSymbology.of(color="#f00"),
            lambda: _SubStyleSchema.of(StyleKey("cmap", "Colormap name.")),
            lambda: _SubFigure(panels=(PanelSpec("p"),)),
        ],
        ids=["symbology", "style-schema", "figure"],
    )
    def test_a_copy_keeps_a_callers_subclass(self, build, how):
        """Copying a subclass of a type that holds a frozen mapping gives back the subclass.

        Args:
            build: Builds an instance of a subclass.
            how: `pickle` or `copy.deepcopy`.

        Test scenario:
            Each `__reduce__` rebuilt the base class by name, so a caller's subclass copied back as the base.
        """
        value = build()
        copied = (
            pickle.loads(pickle.dumps(value))
            if how == "pickle"
            else copy.deepcopy(value)
        )
        assert type(copied) is type(value), type(copied).__name__

    @pytest.mark.parametrize(
        "change",
        [
            lambda tree: tree.remove("b"),
            lambda tree: tree.replace(LayerSpec("a", "lines")),
            lambda tree: tree.add(LayerSpec("c", "points")),
            lambda tree: tree.move("a", -1),
        ],
        ids=["remove", "replace", "add", "move"],
    )
    def test_a_tree_change_keeps_a_callers_subclass(self, change):
        """Every change to a `LayerTree` subclass returns the subclass.

        Args:
            change: One tree change.

        Test scenario:
            `add` and `move` rebuilt the tree with `dataclasses.replace` and kept the subclass; `remove` and
            `replace` called `LayerTree(...)` by name and dropped it.
        """
        tree = _SubTree((LayerSpec("a", "points"), LayerSpec("b", "points")))
        assert type(change(tree)) is _SubTree

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda props: props.__setitem__("x", 1),
            lambda props: props.__delitem__("levels"),
            lambda props: props.update(x=1),
            lambda props: props.setdefault("x", 1),
            lambda props: props.pop("levels"),
            lambda props: props.popitem(),
            lambda props: props.clear(),
            lambda props: props.__ior__({"x": 1}),
        ],
        ids=[
            "setitem",
            "delitem",
            "update",
            "setdefault",
            "pop",
            "popitem",
            "clear",
            "ior",
        ],
    )
    def test_a_frozen_mapping_refuses_every_change(self, mutate):
        """Every way of changing a dict is refused on the mappings the types freeze.

        Args:
            mutate: One dict mutator, applied to a symbology's properties.
        """
        symbology = Symbology().with_props(levels=(1, 2))
        with pytest.raises(TypeError, match="read-only"):
            mutate(symbology.props)
        assert dict(symbology.props) == {"levels": (1, 2)}, symbology.props


class TestWhatWritingRefuses:
    """A value with no JSON form is refused where it is written."""

    def test_a_datetime_on_a_selection_is_refused_naming_the_field(self):
        """`Selection.time` holding a `datetime` names the field, not a `json.dumps` frame.

        Test scenario:
            A live object in a figure description is what the seam forbids; finding it inside `json.dumps` would
            name neither the type nor the field.
        """
        selection = Selection.of(1, time=datetime.date(2024, 1, 1))
        with pytest.raises(TypeError, match=r"Selection\.time holds a date"):
            selection.to_dict()

    def test_a_set_in_a_property_is_refused_because_it_has_no_order(self):
        """A set has no order to write down, so it is refused rather than listed.

        Test scenario:
            `list(set)` would pick an order by hash, and a round trip would then compare unequal at random.
        """
        symbology = Symbology().with_props(classes={1, 2})
        with pytest.raises(TypeError, match=r"props\['classes'\] holds a set"):
            symbology.to_dict()

    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            np.float64("nan"),
            np.float32("inf"),
        ],
        ids=["nan", "inf", "-inf", "np-nan", "np32-inf"],
    )
    def test_nan_and_infinity_are_refused_because_strict_json_cannot_spell_them(
        self, value
    ):
        """A non-finite number is refused where it is written, even though Python's `json` would write it.

        Args:
            value: The non-finite number under test.

        Test scenario:
            `json.dumps` writes `NaN` and `Infinity` by default, and Python reads them back — but
            `json.dumps(allow_nan=False)` refuses them and so does JavaScript's `JSON.parse`, which reads an exported
            page. A description that only Python can read back is not the portable form the seam promises.
        """
        symbology = Symbology().with_props(levels=[1.0, value])
        with pytest.raises(
            TypeError, match=r"props\['levels'\]\[1\] is .*no JSON form"
        ):
            symbology.to_dict()

    @pytest.mark.parametrize(
        "build, field",
        [
            (
                lambda: Scale(0.0, 1.0, breaks=(0.0, float("nan"), 1.0)),
                r"Scale\.breaks\[1\] is nan",
            ),
            (
                lambda: Scale(0.0, 1.0, missing=object()),
                r"Scale\.missing holds a object",
            ),
            (
                lambda: Encoding.by_field(
                    "size", "p", output_range=(0.0, float("inf"))
                ),
                r"output_range\[1\] is inf",
            ),
        ],
        ids=["scale-nan-edge", "scale-live-missing", "encoding-inf-range"],
    )
    def test_typed_fields_are_checked_like_free_form_ones(self, build, field):
        """A field with a declared type still goes through the JSON rules when it is written.

        Args:
            build: Builds a value whose typed field holds something JSON cannot write.
            field: A pattern naming that field in the message.

        Test scenario:
            The constructors check a domain is finite but not the class edges, `missing` or `output_range`, and
            `to_dict` wrote those three straight into the dict — a NaN edge, an infinite range and a live object all
            came out of `to_dict` and failed later inside `json.dumps`, naming neither the type nor the field.
        """
        value = build()
        with pytest.raises(TypeError, match=field):
            value.to_dict()

    @pytest.mark.parametrize(
        "value",
        [
            Scale(np.float32(0.0), np.float32(1.0)),
            Bounds(np.float32(0.0), 0.0, np.float32(1.0), 1.0, crs=4326),
        ],
        ids=["scale-np-float32", "bounds-np-float32"],
    )
    def test_numpy_floats_in_typed_fields_are_written_as_python_floats(self, value):
        """A domain or edge held as `np.float32` is written in a form strict JSON accepts.

        Args:
            value: A value built with numpy floats, which the constructor accepts.

        Test scenario:
            `np.float32` is not a `float` subclass, so `json.dumps` raised `TypeError` on the dict `to_dict` returned.
        """
        stored = value.to_dict()
        assert json.loads(json.dumps(stored, allow_nan=False)) == stored, stored

    def test_a_written_figure_is_strict_json(self):
        """What `to_dict` produces passes `json.dumps(allow_nan=False)`, the strict writer."""
        stored = Symbology.of(opacity=0.5).with_props(levels=[1.5, 2.5]).to_dict()
        assert json.loads(json.dumps(stored, allow_nan=False)) == stored

    @pytest.mark.parametrize(
        "value",
        [
            np.datetime64("2020-01-01T00:00:00.000000000"),
            np.timedelta64(5, "ns"),
            np.array(["2020-01-01T00:00"], dtype="datetime64[ns]"),
            np.array([5], dtype="timedelta64[ns]"),
        ],
        ids=[
            "datetime64-ns",
            "timedelta64-ns",
            "datetime64-ns-array",
            "timedelta64-ns-array",
        ],
    )
    def test_a_nanosecond_time_is_refused_not_written_as_an_integer(self, value):
        """A `datetime64[ns]` is refused like a coarser one, instead of being written as an int.

        Args:
            value: A nanosecond-precision time or duration, scalar or array.

        Test scenario:
            `np.datetime64(...).item()` returns a `date` at day precision, which the writer already refused — but at
            nanosecond precision it returns an int, because `datetime` cannot hold nanoseconds, and the int was
            written as a plain number. Nanosecond is the default for pandas and netCDF time coordinates, which is
            where `Selection.time` values come from; the stored figure would read back an int and select the wrong
            step.
        """
        with pytest.raises(TypeError, match="store a time as an ISO 8601 string"):
            to_json_value(value, "Selection.time")

    def test_a_non_string_mapping_key_is_refused(self):
        """A JSON object's keys are strings, so an int key is refused rather than stringified."""
        with pytest.raises(TypeError, match="non-string key 1"):
            to_json_value({1: "a"}, "props")


class TestCrsObjectsHashLikeTheirRoundTrip:
    """A spec built with a CRS object is equal to its JSON round trip, and hashes the same."""

    @pytest.mark.parametrize(
        "definition",
        ["EPSG:3857", "+proj=utm +zone=31 +ellps=intl +units=m +no_defs"],
        ids=["with-a-code", "code-less"],
    )
    def test_bounds_viewport_and_figure_hash_alike_after_a_round_trip(self, definition):
        """`Bounds`, `Viewport` and a `FigureSpec` over them rebuild equal and hash equal, so one keys a cache.

        Args:
            definition: What the CRS object is parsed from.

        Test scenario:
            The object stayed in the `crs` field. pyproj's `__eq__` parses the other side, so a rectangle equalled
            its round trip — which holds the written string — while `hash` used the object's WKT: equal values
            hashed apart, a rebuilt figure missed its own cache entry, and `Viewport(obj)` equalled both
            `Viewport(3857)` and `Viewport("EPSG:3857")`, which are not equal to each other.
        """
        from pyramids.base.crs import crs_from_user_input

        crs = crs_from_user_input(definition)
        box = Bounds(0.0, 0.0, 1.0, 1.0, crs=crs)
        view = Viewport(crs, bounds=box)
        figure = FigureSpec(panels=(PanelSpec("p", view),))
        cache = {figure: "drawn"}
        for value in (box, view, figure):
            rebuilt = _through_json(value)
            assert rebuilt == value, f"{type(value).__name__} changed in a round trip"
            assert hash(rebuilt) == hash(value), (
                f"{type(value).__name__} hashes apart from its round trip"
            )
        assert cache.get(_through_json(figure)) == "drawn", (
            "a rebuilt figure must find its cache entry"
        )

    def test_a_crs_object_view_equals_only_views_written_the_same_way(self):
        """A view in a CRS object is the view its written spelling names, and equality stays transitive."""
        from pyramids.base.crs import crs_from_user_input

        in_object = Viewport(crs_from_user_input(3857))
        assert in_object == Viewport("EPSG:3857"), in_object
        assert in_object != Viewport(3857), (
            "an int and a string spelling are different values, as they always were"
        )


class TestNumpyInputs:
    """A value computed with numpy is accepted wherever its Python form is, and stored as the Python form."""

    @pytest.mark.parametrize(
        "build, read, stored",
        [
            (
                lambda: FigureSpec(
                    panels=(PanelSpec("p"),), size=(np.int64(8), np.float32(4.5))
                ),
                lambda spec: spec.size,
                (8.0, 4.5),
            ),
            (
                lambda: RenderTarget("image", pixel_ratio=np.float32(2.0)),
                lambda target: target.pixel_ratio,
                2.0,
            ),
            (
                lambda: ViewRequest(pixel_ratio=np.float32(2.0)),
                lambda request: request.pixel_ratio,
                2.0,
            ),
            (
                lambda: Camera((0.0, -10.0, 5.0), parallel=np.bool_(True)),
                lambda camera: camera.parallel,
                True,
            ),
            (
                lambda: Viewport(4326, globe=np.bool_(True)),
                lambda view: view.globe,
                True,
            ),
            (
                lambda: LayerSpec("a", "points", visible=np.bool_(False)),
                lambda layer: layer.visible,
                False,
            ),
            (
                lambda: LayerTree((LayerSpec("a", "points"),)).set_visible(
                    "a", np.bool_(False)
                ),
                lambda tree: tree.get("a").visible,
                False,
            ),
            (
                lambda: LayerTree(
                    (LayerSpec("a", "points", group="g"),)
                ).set_group_visible("g", np.bool_(False)),
                lambda tree: tuple(tree.hidden_groups),
                ("g",),
            ),
        ],
        ids=[
            "figure-size",
            "target-pixel-ratio",
            "request-pixel-ratio",
            "camera-parallel",
            "viewport-globe",
            "layer-visible",
            "tree-set-visible",
            "tree-set-group-visible",
        ],
    )
    def test_a_numpy_value_is_accepted_and_stored_as_the_python_one(
        self, build, read, stored
    ):
        """Sizes, ratios and flags computed with numpy build, and hold plain Python values afterwards.

        Args:
            build: Builds a value from numpy inputs.
            read: Reads the field back.
            stored: The field as it must be held — Python floats and booleans.

        Test scenario:
            `finite_number` accepted numpy numbers for bounds, scales and cameras, but `FigureSpec.size` and
            `RenderTarget.pixel_ratio` refused `np.int64`/`np.float32`, and every boolean field refused `np.bool_`
            — one vocabulary disagreeing with itself. `ViewRequest` accepted a numpy ratio but kept it as numpy.
            The repr comparison checks the type as well as the value: a stored `np.True_` or `np.float32` would not
            survive `json.dumps`.
        """
        assert repr(read(build())) == repr(stored)


class TestNumpyArrayValues:
    """A numpy array held as a free-form value writes, compares and hashes as the list of its elements would."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda: LayerSpec(
                "dem", "raster", selection=Selection.of(1, time=np.array([1, 2]))
            ),
            lambda: LayerSpec(
                "dem",
                "raster",
                symbology=Symbology().with_props(
                    levels=np.array([[0.0, 1.0], [2.0, 3.0]])
                ),
            ),
            lambda: Encoding.constant("color", np.array([1.0, 0.0, 0.0])),
        ],
        ids=["selection-time", "symbology-2d-levels", "encoding-constant"],
    )
    def test_an_array_value_round_trips_equal_and_hashes_alike(self, build):
        """A spec holding an array writes, reads back equal, and hashes as its round trip does.

        Args:
            build: Builds a spec with an array in a free-form value.

        Test scenario:
            `frozen_value` kept an array as given and `to_json_value` wrote it as a list, so the writer took a value
            the comparison beside it could not: `LayerSpec.to_dict` compares its selection with the default and
            raised "The truth value of an array with more than one element is ambiguous", as did `==` on the spec,
            and the spec did not hash.
        """
        value = build()
        rebuilt = _through_json(value)
        assert rebuilt == value, rebuilt
        assert hash(rebuilt) == hash(value), (
            "an array value must hash as its round trip does"
        )


class TestNumpyStrings:
    """A string computed with numpy — `np.unique` over a column gives `numpy.str_` — is written as a Python string."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda text: LayerSpec(
                text("dem"),
                text("raster"),
                source_id=text("srtm"),
                label=text("Elevation"),
                group=text("terrain"),
                filter=text("v > 0"),
            ),
            lambda text: LayerTree(
                (LayerSpec("dem", "raster", group=text("terrain")),),
                frozenset({text("terrain")}),
            ),
            lambda text: PanelSpec(
                text("p"), layers=(text("dem"),), title=text("Left")
            ),
            lambda text: FigureSpec(
                panels=(PanelSpec("p"),),
                sources={text("srtm"): DataRef("dem.tif")},
                title=text("Figure"),
            ),
            lambda text: DataRef(
                text("dem.tif"), driver=text("GTiff"), version=text("v1")
            ),
            lambda text: Encoding(text("color"), field=text("elevation")),
            lambda text: Symbology.of(color=text("#f00")),
            lambda text: Scale.categorical([text("a"), text("b")], ["#f00", "#0f0"]),
            lambda text: RenderTarget(text("html")),
            lambda text: Viewport(text("EPSG:4326"), domain=text("europe")),
            lambda text: Bounds(0.0, 0.0, 1.0, 1.0, crs=text("EPSG:4326")),
            lambda text: Selection.of(1, time=text("2024-01")),
        ],
        ids=[
            "layer",
            "tree-hidden-groups",
            "panel",
            "figure",
            "dataref",
            "encoding",
            "symbology",
            "scale-categories",
            "target-kind",
            "viewport",
            "bounds-crs",
            "selection-time",
        ],
    )
    def test_a_numpy_string_is_written_as_a_python_string(self, build):
        """Every string field and key a spec writes comes out as `str`, whatever string type it was built from.

        Args:
            build: Builds a spec whose string fields are made by the function it is given.

        Test scenario:
            `to_json_value` returned any `str` subclass unchanged, and the typed string fields — ids, kinds, labels,
            titles, source ids, a target's kind — were written as held. `json` copes with `numpy.str_`, but YAML,
            TOML and msgpack writers refuse it: the same failure `np.float64` had before it was written as a float.
        """
        written = build(np.str_).to_dict()
        assert _foreign_types(written) == [], written


class TestTheSharedRules:
    """`base/spec/_serial.py` on its own."""

    def test_a_code_less_crs_object_is_written_as_its_own_definition(self):
        """A CRS object with no authority code is written as its WKT, which reads back to the same definition.

        Test scenario:
            `crs_to_json` wrote a CRS object as the EPSG code PROJ identified at 70% confidence. For this UTM zone on
            the International ellipsoid that was "EPSG:23031" — ED50, a different datum — so a stored figure read
            back in another CRS, and a corner warped from it landed about 148 m from where the view put it.
        """
        from pyramids.base.crs import crs_from_user_input

        obj = crs_from_user_input("+proj=utm +zone=31 +ellps=intl +units=m +no_defs")
        written = crs_to_json(obj, "crs")
        assert not written.startswith("EPSG:"), written
        assert crs_from_user_input(written).to_wkt() == obj.to_wkt(), written

    def test_a_crs_object_carrying_its_code_is_written_by_that_code(self):
        """An object built from an EPSG code keeps that code as its written form."""
        from pyramids.base.crs import crs_from_user_input

        assert crs_to_json(crs_from_user_input("EPSG:3857"), "crs") == "EPSG:3857"

    def test_numpy_values_are_written_as_python_values(self):
        """A numpy scalar or array is written in the form JSON reads back."""
        written = to_json_value(
            {"n": np.int64(3), "x": np.float32(0.5), "a": np.array([1, 2])}, "props"
        )
        assert written == {"n": 3, "x": 0.5, "a": [1, 2]}, written
        assert type(written["n"]) is int, type(written["n"])

    @pytest.mark.parametrize(
        "value, expected",
        [
            (np.float64(0.5), 0.5),
            (np.float32(0.5), 0.5),
            (np.int64(3), 3),
            (np.bool_(True), True),
        ],
        ids=["float64", "float32", "int64", "bool_"],
    )
    def test_every_numpy_scalar_is_written_as_the_python_type(self, value, expected):
        """`np.float64` is written as a Python `float`, like every other numpy scalar.

        Args:
            value: The numpy scalar.
            expected: The Python value it must become.

        Test scenario:
            `np.float64` subclasses `float`, so it took the plain-number branch and came back as numpy — contrary to
            the docstring, and unreadable by a YAML safe dumper. The type is asserted, not only the value, because
            `np.float64(0.5) == 0.5` holds either way.
        """
        written = to_json_value(value, "props")
        assert type(written) is type(expected), (
            f"got {type(written).__name__} for {type(value).__name__}"
        )
        assert written == expected, written

    def test_a_tuple_is_written_as_a_list(self):
        """JSON has no tuple; writing one as a list is what `json.dumps` would do anyway, stated here."""
        assert to_json_value((1, (2, 3)), "value") == [1, [2, 3]]

    @pytest.mark.parametrize(
        "value",
        [np.datetime64("2024-01-01"), np.complex128(1 + 2j), np.bytes_(b"x")],
        ids=["datetime64", "complex128", "bytes_"],
    )
    def test_a_numpy_scalar_with_no_json_form_is_refused_naming_its_type(self, value):
        """A numpy scalar whose Python value is not a JSON scalar is refused, not passed through.

        Args:
            value: A numpy scalar whose `.item()` is a `date`, a `complex` or `bytes`.

        Test scenario:
            Unwrapping with `.item()` is only safe for bool, int, float and str. The others fall through to the
            refusal, which names the field and the numpy type rather than the unwrapped Python one.
        """
        expected = (
            rf"Selection\.time holds a {type(value).__name__}, which has no JSON form"
        )
        with pytest.raises(TypeError, match=expected):
            to_json_value(value, "Selection.time")

    @pytest.mark.parametrize(
        "crs", [None, 4326, "EPSG:3857", "+proj=longlat +datum=WGS84 +no_defs"]
    )
    def test_a_crs_already_in_json_form_passes_through(self, crs):
        """An EPSG integer or a string is written as given.

        Args:
            crs: The CRS spelling under test.
        """
        assert crs_to_json(crs, "Bounds.crs") == crs

    def test_a_numpy_integer_crs_is_written_as_an_int(self):
        """An EPSG code that arrived as `np.int64` is written as a Python int."""
        written = crs_to_json(np.int64(4326), "Bounds.crs")
        assert (written, type(written)) == (4326, int), (written, type(written))

    def test_a_crs_object_is_written_as_its_authority_code(self):
        """A pyproj CRS object becomes `"EPSG:<code>"`, which reads back to the same system.

        Test scenario:
            The object has no JSON form of its own, but it names an authority; the string reads back through the
            same pyramids parser, so a round trip keeps the system even though the Python type changes.
        """
        from pyramids.base.crs import crs_from_user_input

        crs = crs_from_user_input(32618)
        written = crs_to_json(crs, "Bounds.crs")
        assert written == "EPSG:32618", written
        assert crs_from_user_input(written) == crs, (
            f"{written!r} must read back to the same system as the object"
        )

    def test_a_crs_object_with_no_authority_code_is_written_as_wkt(self):
        """A CRS object that names no EPSG code is written as its WKT, which reads back to the same system.

        Test scenario:
            An orthographic projection built from a PROJ string carries no authority, so `"EPSG:<code>"` is not
            available. The WKT is the fallback; it must be a string and must parse back to an equal CRS.
        """
        from pyramids.base.crs import crs_from_user_input

        crs = crs_from_user_input("+proj=ortho +lat_0=53 +lon_0=4")
        written = crs_to_json(crs, "Viewport.crs")
        assert isinstance(written, str), (
            f"expected WKT text, got {type(written).__name__}"
        )
        assert not written.startswith("EPSG:"), (
            f"a CRS with no code was written as {written!r}"
        )
        assert crs_from_user_input(written) == crs, (
            f"the WKT must read back to the same system as the object; got {written[:80]!r}"
        )

    def test_a_boolean_crs_is_refused(self):
        """`True` is an int in Python and would be written as EPSG code 1."""
        with pytest.raises(TypeError, match="names no coordinate reference system"):
            crs_to_json(True, "Bounds.crs")

    def test_an_object_that_is_not_a_crs_is_refused(self):
        """Something pyramids cannot read as a CRS is refused, naming the field."""
        not_a_crs = object()
        with pytest.raises(TypeError, match="not a readable CRS"):
            crs_to_json(not_a_crs, "Bounds.crs")

    @pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), None])
    def test_finite_number_refuses_what_is_not_a_finite_number(self, value):
        """A bool, a string, `nan`, an infinity and `None` are each refused.

        Args:
            value: The candidate under test.
        """
        with pytest.raises(ValueError, match="Camera"):
            finite_number("Camera", "view_angle", value)

    def test_finite_number_accepts_numpy_numbers(self):
        """A numpy number is returned as a Python float."""
        number = finite_number("Camera", "view_angle", np.float32(30.0))
        assert (number, type(number)) == (30.0, float), (number, type(number))


class TestHashingAMappingThatDoesNotSort:
    """`hashable_value` orders a mapping's items, and a mapping's keys need not compare (review L6)."""

    @pytest.mark.parametrize(
        "mapping",
        [
            {1: "a", "b": 2},
            {(1, 2): "x", "a": 1},
            {None: 1, "a": 2},
            {"outer": {1: "a", "b": 2}},
        ],
    )
    def test_a_mapping_whose_keys_do_not_compare_still_hashes(self, mapping):
        """Sorting the items compares the keys, and two key types need not be ordered against each other.

        Args:
            mapping: The mapping under test.

        Test scenario:
            The documented `Raises:` named an unhashable *value* as the one cause. A mixed-key mapping is
            perfectly hashable — it is the sort that fails — so `{1: 'a', 'b': 2}` raised
            ``'<' not supported between instances of 'str' and 'int'`` from a helper whose whole job is to
            make a value hashable.
        """
        assert hash(hashable_value(mapping)) is not None, (
            f"{mapping!r} hashes; ordering its items must not be what refuses it"
        )

    def test_two_orderings_of_such_a_mapping_still_hash_alike(self):
        """The fallback order has to be a function of the keys, not of the insertion order.

        Test scenario:
            Equal objects must hash equal. Two dicts built key-by-key in opposite orders are equal, so
            whatever order the items are put in has to be derived from the keys themselves.
        """
        one = hashable_value({1: "a", "b": 2, None: 3})
        other = hashable_value({None: 3, "b": 2, 1: "a"})
        assert hash(one) == hash(other), (
            f"two spellings of one mapping hashed apart: {one!r} vs {other!r}"
        )

    def test_a_symbology_holding_one_hashes_too(self):
        """The helper exists for `Symbology.__hash__`, so the defect is reachable from a real style."""
        mixed = Symbology(props={"lookup": {1: "a", "b": 2}})
        assert hash(mixed) is not None, "a style holding a mixed-key mapping must hash"


class TestAMappingDoesNotHashAsTheItemsItBecomes:
    """A mapping is hashed as a tuple of pairs, and a tuple of pairs is a value in its own right (review N1).

    The `__eq__`/`__hash__` contract was never broken — equal objects hash equal — but two properties that
    are *not* equal landed on the same hash, so a cache or a set keyed on a style paid for a collision it
    could avoid.
    """

    def test_a_mapping_and_the_pairs_it_becomes_are_different_keys(self):
        """The surrogate for a mapping has to say that a mapping is what it came from."""
        as_mapping = hashable_value({"x": 1})
        as_pairs = hashable_value((("x", 1),))
        assert as_mapping != as_pairs, (
            f"a mapping and a tuple of its pairs both hashed as {as_pairs!r}"
        )

    def test_two_styles_that_differ_only_in_that_do_not_collide(self):
        """The reachable form: one property spelled as a mapping, and the same property spelled as pairs."""
        mapping = Symbology(props={"p": {"x": 1}})
        pairs = Symbology(props={"p": (("x", 1),)})
        assert hash(mapping) != hash(pairs), (
            f"two unequal styles ({mapping.props!r} and {pairs.props!r}) hash alike"
        )

    def test_a_tagged_surrogate_still_hashes_two_spellings_of_one_mapping_alike(self):
        """Tagging must not cost what the helper is for: equal mappings still collapse to one entry."""
        written = Symbology(props={"paint": {"a": 1, "b": 2}})
        rewritten = Symbology(props={"paint": {"b": 2, "a": 1}})
        assert len({written, rewritten}) == 1, "two equal styles must hold one slot"


class TestAMappingHashesByItsItemsAndNothingElse:
    """Equal mappings must hash alike however they were spelled (round 2 `/docstring`)."""

    class _SameRepr:
        """A key whose `repr` tells it nothing apart from its siblings.

        Attributes:
            tag: What actually distinguishes one key from another.
        """

        def __init__(self, tag):
            """Store the tag.

            Args:
                tag: What distinguishes this key.
            """
            self.tag = tag

        def __repr__(self):
            """Return a repr every instance shares.

            Returns:
                The same string for every key, which is the point.
            """
            return "<key>"

        def __eq__(self, other):
            """Compare by tag.

            Args:
                other: The other key.

            Returns:
                Whether both are keys with the same tag.
            """
            return isinstance(
                other, TestAMappingHashesByItsItemsAndNothingElse._SameRepr
            ) and (other.tag == self.tag)

        def __hash__(self):
            """Hash by tag.

            Returns:
                The tag's hash.
            """
            return hash(self.tag)

    def test_two_spellings_of_one_mapping_hash_alike_when_the_keys_share_a_repr(self):
        """The order the keys were written in must not reach the hash.

        Test scenario:
            Unorderable keys fell back to `sorted(..., key=repr)`, and a stable sort keeps insertion order
            for keys whose `repr` matches — so two mappings that compare equal produced different hashes,
            breaking the invariant the hashing work exists to hold. The keys here share one `repr` and
            differ by tag, which is exactly the case the fallback could not order.
        """
        first, second = self._SameRepr("one"), self._SameRepr("two")
        written_one_way = hashable_value({first: "a", second: "b"})
        written_the_other = hashable_value({second: "b", first: "a"})
        assert written_one_way == written_the_other, (
            "two spellings of one mapping must reduce to the same value"
        )
        assert hash(written_one_way) == hash(written_the_other), (
            "and equal values must hash alike, or a set holds one object twice"
        )

    def test_the_same_holds_through_a_symbology(self):
        """The invariant has to survive the type that made it necessary."""
        first, second = self._SameRepr("one"), self._SameRepr("two")
        one_way = Symbology(props={"paint": {first: "a", second: "b"}})
        other_way = Symbology(props={"paint": {second: "b", first: "a"}})
        assert one_way == other_way, "the two symbologies are equal by value"
        assert hash(one_way) == hash(other_way), (
            "so they must hash alike; a set keyed on style would otherwise hold both"
        )
        assert len({one_way, other_way}) == 1, "and a set must hold them once"


#: A matplotlib dash pattern in its ``(offset, (on, off))`` form — the container the shared rule is argued from,
#: and the one whose round trip the rule's docstring cites.
_DASH_PATTERN = (0, (5, 5))

#: A graduated choropleth's class edges and a categorical palette, as a builder resolves them. Both are
#: lists, both are far below the size bound, and losing them is the regression #330 reports: a figure
#: reloaded on another map redrew in the tier's default colours while its classification survived.
_COLOR_LEVELS = [0.0, 0.25, 0.5, 1.0]
_PALETTE = ["#440154", "#21918c", "#fde725"]


#: The types the oracle holds to their exact class. A container is the half of the rule where the type is
#: load-bearing: a tuple flattened to a list is a value matplotlib refuses outright, and a `dict` subclass
#: flattened to a `dict` has had its contents — an `xyzservices.TileProvider`'s API key — copied into the
#: figure. Neither is true of a scalar, which carries nothing but itself.
_CONTAINERS = (list, tuple, dict, set, frozenset, np.ndarray)


def _comes_back_the_same(left, right):
    """Whether the trip returned `left` as `right` under the rule the code actually makes.

    Two comparisons, not one, because the rule is two-sided (`R-M5`). A **container** must come back as the
    same class as well as equal — plain `==` would call a tuple equal to nothing, but a `list` holding a
    `tuple` one level down compares equal to the list holding a list it becomes, and that is the loss the
    rule exists to catch. A **scalar** is compared by value alone, because the trip flattens every scalar
    subclass to its plain counterpart and that counterpart is what the drawer is handed.

    Args:
        left: The value a builder recorded.
        right: What came back from the trip.

    Returns:
        `True` when the two are equal, and — wherever either side is a container — of the same class, at
        every depth.
    """
    if isinstance(left, _CONTAINERS) or isinstance(right, _CONTAINERS):
        if type(left) is not type(right):
            return False
        if isinstance(left, np.ndarray):
            return bool(np.array_equal(left, right))
        if isinstance(left, dict):
            if set(left) != set(right):
                return False
            return all(_comes_back_the_same(left[key], right[key]) for key in left)
        if len(left) != len(right):
            return False
        return all(_comes_back_the_same(one, other) for one, other in zip(left, right))
    return bool(left == right)


def _survives_the_trip(value):
    """Whether the record-to-draw path gives `value` back equal, and a container back as its own class.

    The oracle the rule is measured against, rather than a second spelling of the rule: it runs the real
    path a described value takes — `frozen_value` into the spec, `to_json_value` out, `json` there and
    back, `frozen_value` on read, `thawed_value` at the drawer — and compares what comes out.

    Args:
        value: The value a builder would record.

    Returns:
        `True` when both the in-process freeze and the stored JSON trip return a value that satisfies
        :func:`_comes_back_the_same`; `False` when either loses it or the writer refuses it outright.
    """
    try:
        after_freeze = thawed_value(frozen_value(value))
        written = to_json_value(frozen_value(value), "Symbology.props")
        after_json = thawed_value(frozen_value(json.loads(json.dumps(written))))
    except (TypeError, ValueError):
        return False
    if not _comes_back_the_same(value, after_freeze):
        return False
    return _comes_back_the_same(value, after_json)


class _KeyedProvider(dict):
    """A `dict` subclass carrying a credential, standing in for an `xyzservices.TileProvider`.

    Declared here rather than imported so the rule is checked without the tile stack installed; the real
    provider is checked by the static tier's own seam suite.
    """


class _TaggedList(list):
    """A `list` subclass, standing for any sequence a library hands back with behaviour attached."""


class _TaggedStr(str):
    """A `str` subclass with nothing added, standing for any string a library hands back as its own type."""


class _TaggedInt(int):
    """An `int` subclass with nothing added, the counterpart of `_TaggedStr` on the number side."""


class _Linestyle(str, enum.Enum):
    """A caller's `str`-valued enumeration of a style keyword's allowed values.

    The one scalar whose flattening changes the **value** rather than only the type: `str` on a mixin
    enumeration is `Enum.__str__`, so the writer stores the member's name where its value belongs.
    """

    SOLID = "solid"


class TestWhatTravelsInAFigure:
    """One rule, in one place, for what a figure's description carries (#322).

    Two tiers used to answer this separately: static asked `travels_in_a_figure` and kept plain scalars only,
    while interactive asked its own `is_json_value` and kept everything the writer accepts. The rule is now
    shared, and it is drawn where the **round trip** puts it rather than where either tier had guessed: a
    value travels when the path back gives it back equal, and — for a container, where the class is what
    carries the loss — as the same class too.

    Holding every container was the over-correction in between (#330). It is true of a tuple and was
    generalised to the containers beside it without being re-measured, so a graduated choropleth's
    `color_levels` and a categorical palette — both plain lists, both lossless — were held beside the layer
    and lost the moment the figure was read anywhere else.
    """

    @pytest.mark.parametrize(
        "value",
        [None, True, False, "solid", 0.25, 7, np.float64(2.5), np.str_("solid")],
        ids=["none", "true", "false", "text", "float", "int", "np-float", "np-str"],
    )
    def test_a_plain_scalar_travels(self, value):
        """A string, a boolean, a finite number or `None` is what a reader on another machine can act on.

        Args:
            value: The scalar under test.
        """
        assert travels_in_a_figure(value), (
            f"{value!r} is a plain scalar, so a figure's description carries it"
        )

    @pytest.mark.parametrize(
        "value",
        [np.bool_(True), np.float64(2.5), np.int64(5), np.float32(1.5), np.uint8(3)],
        ids=["np-bool", "np-float64", "np-int64", "np-float32", "np-uint8"],
    )
    def test_every_numpy_scalar_the_writer_takes_travels(self, value):
        """The numpy family answers as one, rather than per type.

        Args:
            value: The numpy scalar under test.

        Test scenario:
            `np.bool_` was held while `np.float64` beside it travelled (#329) — not by decision, but because
            the gate named types and a numpy bool is neither `bool` nor `numbers.Real`, so it fell off before
            the writer was asked. A flag from any numpy comparison is an ordinary thing to hand a builder.
            Parametrised across the family so a type added later is not a fresh special case.
        """
        assert travels_in_a_figure(value), (
            f"{value!r} is a numpy scalar the writer accepts, so a figure carries it"
        )

    @pytest.mark.parametrize(
        "value",
        [np.datetime64("2024-01-01"), np.complex128(1 + 2j), np.timedelta64(5, "D")],
        ids=["np-datetime64", "np-complex128", "np-timedelta64"],
    )
    def test_a_numpy_scalar_the_writer_refuses_still_does_not_travel(self, value):
        """Admitting the family did not admit what it cannot write down.

        Args:
            value: The numpy scalar under test.

        Test scenario:
            The gate lets every `np.generic` reach the writer, so the writer is what still refuses these. The
            counterpart to the check above: widening membership must not widen the answer.
        """
        assert not travels_in_a_figure(value), (
            f"the writer has no JSON form for {value!r}, so the tier holds it beside the layer"
        )

    @pytest.mark.parametrize(
        "value",
        [
            _PALETTE,
            _COLOR_LEVELS,
            [4, 4],
            ["fid"],
            {"a": 1},
            [[1, 2], [3, 4]],
            {"a": ["x", "y"]},
            [],
            {},
        ],
        ids=[
            "palette",
            "color-levels",
            "dash-list",
            "columns",
            "mapping",
            "nested-lists",
            "mapping-of-lists",
            "empty-list",
            "empty-mapping",
        ],
    )
    def test_a_list_or_a_mapping_of_plain_values_travels(self, value):
        """A list and a dict come back as themselves, so a figure carries them.

        Args:
            value: The container under test.

        Test scenario:
            This is the half #330 lost. Holding a palette meant the map that drew it merged the held copy
            back and rendered correctly, while a figure reloaded anywhere else redrew in the tier's default
            colours — the classification surviving, the colours not.
        """
        assert travels_in_a_figure(value), (
            f"{value!r} comes back from the trip as itself, so a figure's description carries it"
        )

    @pytest.mark.parametrize(
        "value",
        [_DASH_PATTERN, (1, 2), [1, (2, 3)], {"dash": (5, 5)}, np.array([1.0, 2.0])],
        ids=["dash-pattern", "pair", "tuple-in-a-list", "tuple-in-a-mapping", "array"],
    )
    def test_a_tuple_or_an_array_does_not(self, value):
        """JSON has no tuple, so anything holding one is held beside the layer instead.

        Args:
            value: The container under test.

        Test scenario:
            The loss is the tuple's, not the container's, so it has to be found wherever the tuple sits —
            alone, inside a list, or inside a mapping. A rule that only looked at the outermost type would
            describe `[1, (2, 3)]` and hand the engine `[1, [2, 3]]`, which is the defect holding
            containers existed to prevent.
        """
        assert not travels_in_a_figure(value), (
            f"{value!r} is re-typed by the trip, so the tier holds it beside the layer"
        )

    def test_a_dict_subclass_does_not_travel_either(self):
        """A tile provider *is* a dict, and a figure is not a place to write an API key.

        Test scenario:
            The one case where the same-type half of the rule is not a technicality. An
            `xyzservices.TileProvider` is a `dict` of plain strings, one of which is the caller's
            credential, so an `isinstance` gate wrote the key into the figure — measured, as a real
            failure of `tests/static/test_static_seam.py`, while this rule was being widened. The trip
            returns a plain `dict`, a different type, so the same check that refuses a tuple refuses this.
        """
        provider = _KeyedProvider(
            {"name": "Thunderforest", "apikey": "FAKE-KEY-NOT-REAL"}
        )
        assert not travels_in_a_figure(provider), (
            "a dict subclass is an engine object wearing a dict, so the tier holds it beside the layer"
        )

    @pytest.mark.parametrize(
        "value,plain",
        [
            (np.float64(2.5), 2.5),
            (np.bool_(True), True),
            (np.int64(7), 7),
            (_TaggedStr("solid"), "solid"),
            (_TaggedInt(3), 3),
        ],
        ids=["np-float64", "np-bool", "np-int64", "str-subclass", "int-subclass"],
    )
    def test_a_scalar_subclass_travels_as_its_plain_counterpart(self, value, plain):
        """The loss the rule accepts, written down rather than left to be discovered (`R-M5`).

        Args:
            value: The scalar subclass a builder recorded.
            plain: The plain value the drawer is handed in its place.

        Test scenario:
            The rule says "same class" for a container and only "equal" for a scalar, and the difference
            was stated as one rule, so a reader could not tell the flattening was intended. It is: the
            value survives, nothing rides along with it — a scalar has no contents to copy, which is the
            whole of the `TileProvider` argument on the container side — and the plain value is what every
            engine would have received anyway.
        """
        written = to_json_value(frozen_value(value), "Symbology.props")
        drawn = thawed_value(frozen_value(json.loads(json.dumps(written))))
        assert drawn == value, (
            f"{value!r} must come back equal, and came back {drawn!r}"
        )
        assert type(drawn) is type(plain), (
            f"and as a plain {type(plain).__name__}, not a {type(drawn).__name__}"
        )

    def test_a_scalar_whose_value_moves_in_the_writing_does_not_travel(self):
        """Equal is the whole of the scalar rule, so a flattening that changes the value is refused.

        Test scenario:
            `str` on a mixin enumeration is `Enum.__str__`, so the writer stores ``'_Linestyle.SOLID'``
            where the member's value ``'solid'`` belongs — a figure that reloads with a linestyle no
            engine has heard of. Asking only whether the writer *accepts* a scalar admitted it; asking
            whether the writer gives back an equal value refuses it, and refuses by measurement anything
            else whose flattening moves the value.
        """
        assert (
            to_json_value(_Linestyle.SOLID, "Symbology.props") == "_Linestyle.SOLID"
        ), "the writer accepts the member and stores its name, not its value"
        assert not travels_in_a_figure(_Linestyle.SOLID), (
            "so the tier holds it beside the layer and hands the drawer the member itself"
        )

    def test_a_list_subclass_does_not_travel_either(self):
        """The same holds on the sequence side, so the rule needs no list of blessed subclasses.

        Test scenario:
            A subclass carries behaviour the trip cannot restore — it comes back a plain `list` — so it is
            re-typed exactly as a tuple is, and answering by type rather than by name means a subclass
            nobody has heard of yet is already handled.
        """
        assert not travels_in_a_figure(_TaggedList(_PALETTE)), (
            "a list subclass comes back as a plain list, so the tier holds it beside the layer"
        )

    @pytest.mark.parametrize(
        "value",
        [
            _PALETTE,
            _COLOR_LEVELS,
            ["fid"],
            {"a": 1},
            _DASH_PATTERN,
            [1, (2, 3)],
            np.array([1.0, 2.0]),
            [1.0, float("nan")],
            {1: "a"},
            _KeyedProvider({"apikey": "FAKE-KEY-NOT-REAL"}),
            _TaggedList([1, 2]),
            0.25,
            "solid",
            None,
            float("nan"),
            np.float64(2.5),
            np.bool_(True),
            np.int64(7),
            np.str_("solid"),
            [np.float64(2.5)],
            {"width": np.int64(1)},
            _TaggedStr("solid"),
            _TaggedInt(3),
            _Linestyle.SOLID,
        ],
        ids=[
            "palette",
            "color-levels",
            "columns",
            "mapping",
            "dash-pattern",
            "tuple-in-a-list",
            "array",
            "nan-in-a-list",
            "non-string-key",
            "keyed-provider",
            "list-subclass",
            "float",
            "text",
            "none",
            "nan",
            "np-float64",
            "np-bool",
            "np-int64",
            "np-str",
            "np-float-in-a-list",
            "np-int-in-a-mapping",
            "str-subclass",
            "int-subclass",
            "str-enum",
        ],
    )
    def test_the_rule_answers_what_the_round_trip_measures(self, value):
        """The rule is not a taste: for anything inside the size bound it *is* the measured trip.

        Args:
            value: The value under test.

        Test scenario:
            Holding every container was justified by one measurement — a dash pattern — generalised to
            every container beside it without re-measuring. This compares the rule against the trip
            itself, case by case, so the next generalisation has to survive the same comparison.

            The parametrisation carried no numpy scalar and no scalar subclass, which is exactly where the
            rule and the oracle disagreed, so the comparison could not fail (`R-H1`). They are here now, and
            with them the `str`-valued enumeration whose trip changes the value rather than only the type.
        """
        assert travels_in_a_figure(value) is _survives_the_trip(value), (
            f"the rule and the measured round trip disagree about {value!r}"
        )

    def test_a_described_list_reaches_the_drawer_as_the_list_it_was(self):
        """The whole point of describing it: what the drawer is handed is what the builder recorded.

        Test scenario:
            Describing without thawing hands the engine a tuple — HoloViews' `color_levels` is a
            `ClassSelector` of `(int, list, range)` and refuses one outright — so the rule and the read
            boundary have to land together. This pins the value half; the tier suites pin the engine half.
        """
        written = to_json_value(frozen_value(_PALETTE), "Symbology.props")
        drawn = thawed_value(frozen_value(json.loads(json.dumps(written))))
        assert drawn == _PALETTE, f"the palette came back as {drawn!r}"
        assert isinstance(drawn, list), f"and as a {type(drawn).__name__}, not a list"

    def test_a_container_of_the_bounded_size_travels(self):
        """The bound is inclusive, so the biggest describable container really is describable.

        Test scenario:
            Stated as a rule rather than left implicit. Refusing every container bounded the cost as a
            side effect; describing lists has to keep that bound on purpose, and a bound nobody can reach
            from either side is not a bound that has been checked.
        """
        assert travels_in_a_figure(list(range(MAX_TRAVELLING_ELEMENTS - 1))), (
            "a container holding exactly the bounded number of values must travel"
        )

    def test_a_container_over_the_bound_does_not(self):
        """A per-pixel `alpha` is the layer's data, and a description is not where data is copied.

        Test scenario:
            Measured on the full record path: 1,000 values cost ~2 ms and ~7 KB, 1,000,000 cost ~3 s and
            ~9 MB. The second is a 1000 x 1000 raster's worth of per-pixel alpha, and it is many times the
            render it belonged to.
        """
        assert not travels_in_a_figure(list(range(MAX_TRAVELLING_ELEMENTS))), (
            "a container one value over the bound must be held beside the layer"
        )

    def test_the_bound_counts_every_value_at_every_depth(self):
        """Nesting cannot be used to smuggle a big container past a bound that counted only the top level.

        Test scenario:
            `[[0, 0], [1, 1], ...]` has few entries and many values. A bound on `len()` would wave through
            an arbitrarily large payload one level down, which is the cost the bound exists to refuse.
        """
        pairs = [[index, index] for index in range(MAX_TRAVELLING_ELEMENTS // 2)]
        assert len(pairs) < MAX_TRAVELLING_ELEMENTS, (
            "the outer list is well inside the bound, so only its depth can refuse it"
        )
        assert not travels_in_a_figure(pairs), (
            "and it holds half as many values again as the bound allows, so it must not travel"
        )

    def test_a_leaf_the_writer_refuses_sinks_the_container_around_it(self):
        """The writer is still the oracle for every leaf, reached however deep it sits.

        Test scenario:
            A `nan` written into a figure is read back by `JSON.parse` as a `SyntaxError`, so a list
            holding one is no more writable than the bare `nan` is.
        """
        assert not travels_in_a_figure({"levels": [1.0, float("nan")]}), (
            "a mapping holding a value JSON cannot spell must be held beside the layer"
        )

    def test_the_rule_is_narrower_than_the_writer_on_purpose(self):
        """The writer takes the dash pattern; the rule refuses it anyway.

        Test scenario:
            This is the one divergence between the two tiers' old oracles, so it is the assertion that would
            have caught them disagreeing. Reading only the rule, refusing a value the writer accepts looks
            like a bug — the round trip below is why it is not.
        """
        assert to_json_value(_DASH_PATTERN, "Symbology.props") == [0, [5, 5]], (
            "the writer takes the dash pattern, so the rule is not simply asking the writer"
        )

    def test_the_container_it_refuses_would_not_read_back_as_itself(self):
        """JSON has no tuple, so a described dash pattern comes back as nested lists.

        Test scenario:
            `(0, (5, 5))` read back is `[0, [5, 5]]`, which matplotlib refuses with
            `ValueError: Unrecognized linestyle`. Describing the container would trade a layer that redraws
            with the engine's defaults for one that cannot redraw at all, which is why the rule stops at
            scalars rather than at whatever `json.dumps` will write.
        """
        read_back = json.loads(
            json.dumps(to_json_value(_DASH_PATTERN, "Symbology.props"))
        )
        assert read_back != _DASH_PATTERN, (
            "the round trip must lose the tuple, or holding the container would buy nothing"
        )

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_a_number_json_cannot_spell_does_not_travel(self, value):
        """A scalar still has to survive the writer, and JSON has no spelling for these three.

        Args:
            value: The non-finite number under test.
        """
        assert not travels_in_a_figure(value), (
            f"{value!r} has no JSON form, so a figure holding it could not be written down"
        )

    @pytest.mark.parametrize(
        "value",
        [datetime.datetime(2024, 1, 1), {"a", "b"}, object()],
        ids=["datetime", "set", "object"],
    )
    def test_a_live_object_does_not_travel(self, value):
        """An engine value has no JSON form at all, so it is held rather than described.

        Args:
            value: The object under test.
        """
        assert not travels_in_a_figure(value), (
            f"{value!r} has no JSON form, so the tier holds it beside the layer"
        )
