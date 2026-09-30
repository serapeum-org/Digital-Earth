"""Tests for T6.2 — auto-style: Source metadata -> cleopatra style params."""

from copy import deepcopy

import numpy as np
import pytest
from pyramids.dataset import GeoReference

from digitalearth.base import autostyle
from digitalearth.base.autostyle import (
    auto_style,
    load_library,
    register_style_library,
    temporary_style_library,
)
from digitalearth.base.sources import DimensionInfo, Source
from digitalearth.static import Map


def _source(variable):
    """Build a minimal raster Source carrying a variable name."""
    return Source(
        DimensionInfo(np.zeros((2, 2)), "z"),
        DimensionInfo(np.array([0.0, 1.0]), "x"),
        DimensionInfo(np.array([0.0, 1.0]), "y"),
        metadata={"variable": variable},
    )


class TestLoadLibrary:
    """Tests for load_library."""

    def test_has_default(self):
        """The library always provides a default group with a cmap."""
        assert load_library()["default"]["cmap"] == "viridis"

    def test_has_variable_groups(self):
        """The shipped library defines variable groups beyond default."""
        lib = load_library()
        assert {"temperature", "precipitation", "elevation"} <= set(lib)


class _RaisesOnDeepCopy:
    """A style value whose deep copy raises — stands in for a plugin's live object (a Colormap, a callable).

    ``register_style_library`` accepts any mapping without checking its values are plain data, so a plugin may
    contribute a live object as a style parameter. This reproduces the M1 regression: an unconditional
    ``deepcopy`` of the whole merged library raised on such a value and disabled *all* styling, where the prior
    shallow merge passed it through by reference untouched.
    """

    def __deepcopy__(self, memo):
        """Raise, as a not-deep-copyable live object (e.g. some Colormap / ColorScale wrappers) would."""
        raise RuntimeError("cannot deepcopy this style value")


class TestLoadLibraryReturnsAFreshTopLevelMapping:
    """load_library() returns a fresh top-level dict; the group dicts inside it are shared by reference (M1).

    The N1 deepcopy was reverted: deep-copying the merged library made every load_library()/auto_style() call
    raise the moment a ``digitalearth.styles`` plugin contributed a not-deep-copyable style value — a case the
    prior shallow merge handled. The contract is now the honest, narrower one N1 also offered: only the
    top-level mapping is fresh, so a caller may add/drop/replace whole groups but must not mutate a returned
    group in place. That is safe because the sole in-tree caller, auto_style, only ever reads the library.
    """

    @pytest.fixture(autouse=True)
    def _clear_bundled_cache(self):
        """Drop the lru_cache after each test.

        The returned group dicts are now the lru-cached bundled objects, so a mutation that reached one (a
        regression, or a neutralised fix during mutation-proofing) would leak into every later test in the
        session; clearing forces the next call to reload the shipped values from YAML.
        """
        yield
        autostyle._bundled_library.cache_clear()

    def test_a_non_copyable_plugin_style_value_does_not_break_load_library(self):
        """A plugin group carrying a not-deep-copyable value no longer breaks load_library (M1).

        Test scenario:
            The reverted N1 deepcopy raised on any value whose ``__deepcopy__`` raises (a live Colormap, a
            ColorScale, a callable), so one off-contract plugin disabled *all* styling. A shallow top-level
            copy passes the value through by reference, so the group loads and the object survives intact.
        """
        live = _RaisesOnDeepCopy()
        with temporary_style_library({"live_group": {"cmap": live}}):
            resolved = load_library()
        assert resolved["live_group"]["cmap"] is live, (
            "the plugin's live style value must pass through by reference, not be (deep-)copied"
        )

    def test_adding_a_group_to_a_returned_library_does_not_leak_into_a_later_load(self):
        """The top-level mapping is fresh: adding a group to one result is invisible to a later load.

        Test scenario:
            A shallow copy of the top level is enough to isolate membership changes — add, drop or replace a
            group key — which is the guarantee the reverted fix keeps. (Mutating an *inner* group is explicitly
            no longer isolated; the docstring warns callers to treat a returned group as read-only.)
        """
        returned = load_library()
        returned["__scratch_group__"] = {"cmap": "magma"}
        assert "__scratch_group__" not in load_library(), (
            "adding a top-level group to one result must not reach a later load_library()"
        )

    def test_auto_style_does_not_mutate_the_shared_library(self):
        """auto_style only reads the library it loads, so sharing the group dicts is safe (the M1 rationale).

        Test scenario:
            Because the copy is now shallow, the group dicts load_library returns are the lru-cached bundled
            ones; were auto_style to mutate a group in place it would corrupt them for every later caller.
            Resolving a matched and an unmatched variable must leave the shipped groups unchanged. The snapshot
            is an independent deep copy taken before, so the comparison is against a different object, not
            itself.
        """
        before = deepcopy(load_library())
        auto_style(_source("t2m"))  # matched: exercises the style.update path
        auto_style(_source("mystery_variable"))  # unmatched: default-only path
        assert load_library() == before, (
            "auto_style must not mutate the shared group dicts it reads"
        )


class TestAutoStyle:
    """Tests for auto_style."""

    @pytest.mark.parametrize(
        "variable, expected_cmap",
        [
            ("t2m", "coolwarm"),
            ("2t_daily_mean", "coolwarm"),
            ("total_precipitation", "Blues"),
            ("DEM_orog", "terrain"),
            ("mystery_variable", "viridis"),
            ("", "viridis"),
        ],
    )
    def test_cmap_selection(self, variable, expected_cmap):
        """auto_style selects the colormap for a variable (default for unknown).

        Args:
            variable: The variable name carried by the Source.
            expected_cmap: The colormap the library should resolve.
        """
        result = auto_style(_source(variable))
        assert result["cmap"] == expected_cmap, f"{variable!r} -> {result['cmap']}"

    def test_units_hint_present_for_temperature(self):
        """Temperature styling carries a celsius units hint."""
        assert auto_style(_source("t2m")).get("units") == "celsius"

    def test_match_key_is_stripped(self):
        """The internal 'match' key is never returned in the resolved style."""
        assert "match" not in auto_style(_source("t2m"))

    def test_string_match_pattern(self, mocker):
        """A group whose 'match' is a bare string (not a list) is handled."""
        mocker.patch(
            "digitalearth.base.autostyle.load_library",
            return_value={
                "default": {"cmap": "viridis"},
                "ice": {"match": "siconc", "cmap": "Blues_r"},
            },
        )
        assert auto_style(_source("siconc"))["cmap"] == "Blues_r"


def test_field_uses_auto_style_cmap(dataset):
    """A field method with no explicit cmap picks up the auto-style default for the variable."""
    t2m = dataset.__class__.from_array(
        arr=np.nan_to_num(dataset.read_array(band=0)),
        geo_ref=GeoReference(geo=dataset.geotransform, epsg=dataset.epsg),
    )
    t2m.band_names = ["t2m"]
    m = Map(crs=t2m.epsg)
    m.contours(t2m, filled=True)
    assert m.layers[0][0].default_options["cmap"] == "coolwarm"


def test_explicit_cmap_overrides_auto_style(dataset):
    """An explicit cmap wins over the auto-style default."""
    m = Map(crs=dataset.epsg)
    m.field(dataset, cmap="magma")
    assert m.layers[0][0].default_options["cmap"] == "magma"


class TestPluginLibraryRegistration:
    """register_style_library validates plugin groups, and temporary_style_library scopes them (RP.11)."""

    @pytest.fixture(autouse=True)
    def _restore(self):
        """Snapshot and restore the process-global plugin style table around each test.

        The table is module-global and register_style_library has no un-register, so a leaked group would
        change what every later auto_style sees.
        """
        before = dict(autostyle._PLUGIN_LIBRARY)
        yield
        autostyle._PLUGIN_LIBRARY.clear()
        autostyle._PLUGIN_LIBRARY.update(before)

    def test_a_non_mapping_is_refused_by_type(self):
        """A styles plugin that loads to a non-mapping is refused, so the library is never corrupted.

        Test scenario:
            The contract is a mapping of style-group name to a parameter dict; a list is not one. The guard
            raises TypeError before the merge, which is what lets the wiring skip a malformed plugin.
        """
        not_a_mapping = ["ocean_heat", "temperature"]
        with pytest.raises(TypeError, match="must load to a mapping"):
            register_style_library(not_a_mapping)

    def test_temporary_library_merges_inside_the_block_and_restores_after(self):
        """A group registered temporarily resolves inside the block and is gone once it exits.

        Test scenario:
            The scoped form exists so a demonstration does not leave a group behind; the two membership
            answers (present, then absent) prove the merge and the restore are each real.
        """
        with temporary_style_library({"scratch_var": {"cmap": "magma"}}):
            inside = load_library()["scratch_var"]["cmap"]
        assert inside == "magma", (
            f"the scoped group should resolve inside the block, got {inside!r}"
        )
        assert "scratch_var" not in load_library(), (
            "the scoped group must not outlive the block that registered it"
        )
