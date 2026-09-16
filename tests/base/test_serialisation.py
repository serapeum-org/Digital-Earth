"""`to_dict` / `from_dict` on the Wave 1 value types (DE-20, #281).

A `LayerSpec` can only serialise if everything it holds does, and before Wave 3 only `DataRef` could. These cover
the pairs `Bounds`, `Selection`, `Scale`, `Encoding` and `Symbology` gained, and the shared rules in
`base/spec/_serial.py` they are built on.
"""

import datetime
import json
import re

import numpy as np
import pytest

from digitalearth.base.spec import (
    Bounds,
    Camera,
    Encoding,
    FigureSpec,
    LayerTree,
    PanelSpec,
    Scale,
    Selection,
    Symbology,
)
from digitalearth.base.spec._serial import crs_to_json, finite_number, to_json_value

#: A minimal valid panel list, for the figure reads that fail on another field.
_PANELS = [{"id": "p", "viewport": {"crs": 3857}}]


def _through_json(value):
    """Round-trip `value` through its dict form and real JSON text."""
    return type(value).from_dict(json.loads(json.dumps(value.to_dict())))


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

    @pytest.mark.parametrize("edge", [None, "0"])
    def test_a_bounds_edge_that_is_not_a_number_is_named(self, edge):
        """`None` or a numeric string for an edge is refused naming the edge, not from inside `float()`.

        Args:
            edge: The stored value for `xmin`.
        """
        stored = {"xmin": edge, "ymin": 0, "xmax": 1, "ymax": 1, "crs": 4326}
        with pytest.raises(ValueError, match="Bounds needs xmin as a number"):
            Bounds.from_dict(stored)


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

    def test_a_non_string_mapping_key_is_refused(self):
        """A JSON object's keys are strings, so an int key is refused rather than stringified."""
        with pytest.raises(TypeError, match="non-string key 1"):
            to_json_value({1: "a"}, "props")


class TestTheSharedRules:
    """`base/spec/_serial.py` on its own."""

    def test_numpy_values_are_written_as_python_values(self):
        """A numpy scalar or array is written in the form JSON reads back."""
        written = to_json_value(
            {"n": np.int64(3), "x": np.float32(0.5), "a": np.array([1, 2])}, "props"
        )
        assert written == {"n": 3, "x": 0.5, "a": [1, 2]}, written
        assert type(written["n"]) is int, type(written["n"])

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
