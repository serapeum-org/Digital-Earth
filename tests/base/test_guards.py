"""Direct tests for the engine-neutral vector-input guards in ``digitalearth.base.guards``.

The two shared 2-D-tier guards — :func:`~digitalearth.base.guards.require_vector` and
:func:`~digitalearth.base.guards.reject_raster` — are exercised end-to-end through the web and interactive
tiers, but they are pure functions that inspect only ``geometry`` and ``columns`` and import no renderer, so
they are tested here directly with tiny duck-typed stand-ins (no pyramids, no engine, no I/O). This pins each
branch: a vector passes silently, a raster is turned away naming the builder and both raster entry points, a
geometry-less table is sent to ``set_geometry`` (with the wide-frame truncation), and a bare object gets the
generic refusal.
"""

import pytest

from digitalearth.base.guards import MAX_COLUMNS_IN_ERROR, reject_raster, require_vector


def _vector():
    """Return a minimal vector-like object (exposes ``geometry``, as a GeoDataFrame does)."""
    return type("V", (), {"geometry": ()})()


def _raster(width=5):
    """Return a raster-like duck whose ``columns`` is an ``int`` (the grid width in cells)."""
    return type("R", (), {"columns": width})()


def _table(names):
    """Return a geometry-less table-like duck whose ``columns`` is a list of names (a non-int Index)."""
    return type("Frame", (), {"columns": list(names)})()


def _bare():
    """Return a bare object with neither ``geometry`` nor ``columns`` (a plain sequence stand-in)."""
    return type("Bare", (), {})()


class TestModuleConstant:
    """``MAX_COLUMNS_IN_ERROR`` — the shared cap on how many column names a table error lists."""

    def test_cap_is_twelve(self):
        """The cap is 12, the one both 2-D tiers agree on before truncating a wide frame.

        Test scenario:
            The constant is load-bearing for the truncation branch below, so it is pinned directly — a
            change to it fails here as well as flipping the ``... (+N more)`` arithmetic.
        """
        assert MAX_COLUMNS_IN_ERROR == 12, (
            f"expected the shared cap of 12, got {MAX_COLUMNS_IN_ERROR}"
        )


class TestRequireVector:
    """``require_vector`` — raise ``TypeError`` unless the input is a vector layer."""

    def test_vector_like_input_passes_silently(self):
        """An object exposing ``geometry`` returns ``None`` without raising.

        Test scenario:
            The one accepted shape (a ``FeatureCollection`` / GeoDataFrame has ``geometry``) must pass the
            guard untouched, so a legal call is never turned away.
        """
        result = require_vector(_vector(), "points", single="field", stack="timecube")
        assert result is None, (
            f"a vector layer must pass the guard silently, got {result!r}"
        )

    def test_a_raster_is_rejected_naming_the_builder_and_both_entry_points(self):
        """A raster (``columns`` is an ``int``) is refused, quoting the builder and both raster routes.

        Test scenario:
            A pyramids ``Dataset`` reports ``columns`` as an ``int``; the message must start with the
            builder name, say it does not take a raster, and route the caller to the tier's single-raster
            and raster-stack entry points rather than leaking an incidental failure.
        """
        raster = _raster()
        with pytest.raises(TypeError) as excinfo:
            require_vector(raster, "points", single="field", stack="timecube")
        message = str(excinfo.value)
        assert message.startswith("points()"), (
            f"the builder must be named first: {message}"
        )
        assert "does not take a raster" in message, message
        assert "field()" in message, (
            f"the single-raster entry point must be named: {message}"
        )
        assert "timecube()" in message, (
            f"the raster-stack entry point must be named: {message}"
        )

    def test_a_geometry_less_table_is_told_to_set_geometry(self):
        """A table whose active geometry was never set gets the ``set_geometry`` message, not raster advice.

        Test scenario:
            A geometry-less frame reports ``columns`` as a list of names (a non-int Index), which the guard
            must separate from a raster, naming the columns and telling the caller to call ``set_geometry``.
        """
        table = _table(["value", "name"])
        with pytest.raises(TypeError) as excinfo:
            require_vector(table, "polygons", single="field", stack="timecube")
        message = str(excinfo.value)
        assert "set_geometry" in message, (
            f"a geometry-less table must be told to set_geometry: {message}"
        )
        assert "active geometry column" in message, message
        assert "'value'" in message, f"the present columns must be listed: {message}"
        assert "'name'" in message, f"the present columns must be listed: {message}"
        assert "more)" not in message, (
            f"a narrow frame must not be truncated: {message}"
        )

    def test_a_wide_table_truncates_its_column_list(self):
        """A frame past ``MAX_COLUMNS_IN_ERROR`` columns elides the rest with a ``... (+N more)`` marker.

        Test scenario:
            Fifteen columns exceed the cap of 12, so the message lists the first 12 and appends
            ``... (+3 more)`` rather than printing every name into a traceback.
        """
        names = [f"c{index}" for index in range(15)]
        table = _table(names)
        with pytest.raises(TypeError) as excinfo:
            require_vector(table, "points", single="field", stack="timecube")
        message = str(excinfo.value)
        assert "... (+3 more)" in message, (
            f"a wide frame must truncate to a (+N more) marker: {message}"
        )
        assert "'c11'" in message, f"the first 12 names must still show: {message}"
        assert "'c12'" not in message, f"names past the cap must be elided: {message}"

    def test_exactly_the_cap_is_not_truncated(self):
        """A frame with exactly ``MAX_COLUMNS_IN_ERROR`` columns lists them all, with no marker.

        Test scenario:
            Twelve columns is the boundary — ``len(names) > MAX_COLUMNS_IN_ERROR`` is false — so every name
            shows and no ``... (+N more)`` marker is appended.
        """
        names = [f"c{index}" for index in range(MAX_COLUMNS_IN_ERROR)]
        table = _table(names)
        with pytest.raises(TypeError) as excinfo:
            require_vector(table, "points", single="field", stack="timecube")
        message = str(excinfo.value)
        assert "more)" not in message, (
            f"a frame at the cap must not be truncated: {message}"
        )
        assert "'c11'" in message, f"the last in-cap name must show: {message}"

    def test_a_bare_object_gets_the_generic_vector_message(self):
        """An input with neither ``geometry`` nor ``columns`` gets the generic ``needs a vector layer`` refusal.

        Test scenario:
            A bare sequence has no ``columns`` attribute at all, so it is neither a raster nor a table; the
            guard falls through to the generic message, still routing to the raster entry points.
        """
        bare = _bare()
        with pytest.raises(TypeError) as excinfo:
            require_vector(bare, "lines", single="field", stack="timecube")
        message = str(excinfo.value)
        assert "needs a vector layer" in message, (
            f"a bare object must get the generic refusal: {message}"
        )
        assert "set_geometry" not in message, f"a bare object is not a table: {message}"
        assert "field()" in message, message
        assert "timecube()" in message, message

    def test_the_entry_point_names_are_passed_through_verbatim(self):
        """The ``single``/``stack`` names reach the message unchanged, so each tier routes to its own methods.

        Test scenario:
            The web tier passes ``timeslider`` where the interactive tier passes ``timecube``; the guard must
            quote whatever it is handed, not a hard-coded pair.
        """
        raster = _raster()
        with pytest.raises(TypeError) as excinfo:
            require_vector(raster, "heatmap", single="field", stack="timeslider")
        message = str(excinfo.value)
        assert "timeslider()" in message, (
            f"the stack entry point must pass through verbatim: {message}"
        )


class TestRejectRaster:
    """``reject_raster`` — raise only for a raster, leaving every other input alone."""

    def test_a_raster_is_rejected(self):
        """An object whose ``columns`` is an ``int`` is refused, naming the builder and the raster routes.

        Test scenario:
            This is the narrower guard a builder like ``trimesh`` uses; a raster must still be turned away
            with the builder name and both entry points.
        """
        raster = _raster(7)
        with pytest.raises(TypeError) as excinfo:
            reject_raster(raster, "trimesh", single="field", stack="timecube")
        message = str(excinfo.value)
        assert message.startswith("trimesh()"), f"the builder must be named: {message}"
        assert "does not take a raster" in message, message
        assert "field()" in message, message
        assert "timecube()" in message, message

    @pytest.mark.parametrize(
        "data, label",
        [
            (
                type("Mesh", (), {"node_x": (0.0,), "node_y": (0.0,)})(),
                "a geometry-less mesh",
            ),
            (
                type("Frame", (), {"columns": ["a", "b"]})(),
                "a table with a named-Index columns",
            ),
            (type("Bare", (), {})(), "a bare object with no columns"),
            (type("V", (), {"geometry": ()})(), "a vector layer"),
        ],
    )
    def test_a_non_raster_passes_silently(self, data, label):
        """Anything whose ``columns`` is not an ``int`` returns ``None`` — the carve-out for non-vector inputs.

        Args:
            data: The non-raster duck under test.
            label: A human label for the input, so a failure names it.

        Test scenario:
            ``reject_raster`` is deliberately permissive: a mesh, a real table, a bare sequence and a vector
            all pass, because only a raster (``columns`` is an ``int``) is its concern.
        """
        assert (
            reject_raster(data, "trimesh", single="field", stack="timecube") is None
        ), f"{label} must pass reject_raster silently"
