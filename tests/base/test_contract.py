"""The contract as data, and the helper that keeps a renamed method working (U-3, #299).

`base/contract.py` is what a tier is held against, and `renamed_method` is how a spelling is retired without
breaking the caller who already wrote it. These cover both in the environment that owns them — no engine, no
facade, just the vocabulary and the shim.
"""

import warnings

import pytest

from digitalearth.base.contract import (
    CORE,
    Method,
    alias_table,
    core_method,
    pending_for,
)
from digitalearth.base.deprecation import renamed_method, renamed_parameter


class TestReadingTheContract:
    """Looking a name up, and being told when there is none."""

    def test_a_core_name_is_found(self):
        """The declaration says what a name means before any tier is consulted."""
        assert core_method("save").returns == "path", core_method("save")

    def test_a_name_outside_the_core_is_refused_with_the_vocabulary(self):
        """A misspelling is answered with the names there are, not with `None`."""
        with pytest.raises(KeyError, match="is not a Core method"):
            core_method("add_raster")

    def test_the_last_name_is_reachable(self):
        """The lookup walks the whole table, not just its head."""
        assert core_method(CORE[-1].name).name == CORE[-1].name, CORE[-1]

    def test_a_tier_that_renamed_nothing_has_an_empty_table(self):
        """`alias_table` answers for every backend, including one nobody renamed."""
        assert alias_table("nobody") == {}, alias_table("nobody")

    def test_a_tier_with_nothing_pending_answers_the_same_way(self):
        """`pending_for` is a question about a tier, not about the table's keys."""
        assert pending_for("nobody") == {}, pending_for("nobody")

    def test_a_method_declares_its_keywords(self):
        """The canonical keywords are part of the promise, not documentation beside it."""
        assert "cmap" in core_method("field").keywords, core_method("field")

    def test_a_method_can_be_built_with_defaults(self):
        """A declaration is a value: the optional halves have defaults a reader can rely on."""
        built = Method("demo", "a demonstration method")
        assert (built.returns, built.keywords, built.builds_in) == (
            "self",
            frozenset(),
            None,
        ), built


class TestRenamingAMethod:
    """The alias keeps the old call working, and says what to write instead."""

    class _Map:
        """A stand-in facade with one renamed method and one renamed parameter."""

        def set_bounds(self, bounds, *, padding=0):
            """Frame on a region.

            Args:
                bounds: The region.
                padding: Pixels around it.

            Returns:
                A description of what was framed.
            """
            return f"{bounds} with {padding}"

        def save_animation(self, path, *, fps=None, framerate=None):
            """Write frames to a file, resolving the deprecated frame-rate spelling.

            Args:
                path: Where to write.
                fps: Frames per second.
                framerate: The deprecated spelling.

            Returns:
                A description of what was written.
            """
            rate = renamed_parameter(
                new="fps",
                value=fps,
                old="framerate",
                alias=framerate,
                caller="Map.save_animation()",
                default=3.0,
            )
            return f"{path} at {rate}"

        fit_bounds = renamed_method(new="set_bounds", old="fit_bounds", owner="Map")
        animate = renamed_method(new="save_animation", old="animate", owner="Map")

    def test_the_old_name_still_works(self):
        """A promise to the caller who already wrote it."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert self._Map().fit_bounds([0, 0, 1, 1]) == "[0, 0, 1, 1] with 0", (
                "forwarded"
            )

    def test_keywords_travel_through_the_alias(self):
        """A whole call keeps working, not only its positional half."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert self._Map().fit_bounds([0, 0, 1, 1], padding=8) == (
                "[0, 0, 1, 1] with 8"
            ), "keywords must travel"

    def test_the_warning_names_both_spellings(self):
        """The other half of the promise: the old name must not linger silently."""
        with pytest.warns(DeprecationWarning) as caught:
            self._Map().fit_bounds([0, 0, 1, 1])
        message = str(caught[0].message)
        assert "Map.fit_bounds()" in message and "Map.set_bounds()" in message, message

    def test_the_warning_lands_on_the_caller_s_line(self):
        """A warning pointing at the helper is one nobody can act on."""
        with pytest.warns(DeprecationWarning) as caught:
            self._Map().fit_bounds([0, 0, 1, 1])
        assert caught[0].filename == __file__, caught[0].filename

    def test_a_parameter_rename_inside_the_new_method_also_points_at_the_caller(self):
        """The alias counts the frame it added, so both warnings name the caller's line.

        Test scenario:
            The case the one hand-written alias handled by hand: calling the old method *and* the old keyword
            raised two warnings, and without the frame count the second landed in `deprecation.py`.
        """
        with pytest.warns(DeprecationWarning) as caught:
            self._Map().animate("out.gif", framerate=9.0)
        attributed = {
            str(record.message).split("(")[0]: record.filename for record in caught
        }
        assert len(attributed) == 2, [str(record.message) for record in caught]
        for label, filename in attributed.items():
            assert filename == __file__, f"{label} landed in {filename}"

    def test_the_frame_count_is_put_back_afterwards(self):
        """A later call through the new name is not credited with the alias's frame."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._Map().animate("out.gif", framerate=9.0)
        with pytest.warns(DeprecationWarning) as caught:
            self._Map().save_animation("out.gif", framerate=9.0)
        assert caught[0].filename == __file__, caught[0].filename

    def test_the_alias_carries_the_names_it_was_built_with(self):
        """A traceback and a `help()` should show the method a caller actually wrote."""
        held = self._Map.__dict__["fit_bounds"]
        assert (held.__name__, held.__qualname__) == ("fit_bounds", "Map.fit_bounds"), (
            held
        )

    def test_the_alias_documents_where_it_forwards(self):
        """`help(Map.fit_bounds)` must name its replacement."""
        assert "set_bounds" in (self._Map.__dict__["fit_bounds"].__doc__ or ""), "doc"
