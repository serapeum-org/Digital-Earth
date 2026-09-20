"""Tests for RP.10 — ECMWF Magics-style identity matching (autostyle/magics.py + auto_style wiring)."""

import numpy as np
import pytest

from digitalearth.base.autostyle import auto_style
from digitalearth.base.autostyle.magics import (
    _alias_in,
    _as_list,
    _style_of,
    load_magics_library,
    magics_style,
)
from digitalearth.base.sources import DimensionInfo, Source


def _source(variable="", standard_name=None, units=None):
    """Build a minimal raster Source carrying an identity (variable / standard_name / units)."""
    meta = {"variable": variable}
    if standard_name is not None:
        meta["standard_name"] = standard_name
    return Source(
        DimensionInfo(np.zeros((2, 2)), "z"),
        DimensionInfo(np.array([0.0, 1.0]), "x"),
        DimensionInfo(np.array([0.0, 1.0]), "y"),
        metadata=meta,
        units=units,
    )


class TestHelpers:
    """Tests for the small matching helpers."""

    @pytest.mark.parametrize(
        "value, expected",
        [(None, []), ("x", ["x"]), (["a", "b"], ["a", "b"]), (("a",), ["a"])],
    )
    def test_as_list(self, value, expected):
        """_as_list coerces None / scalar / sequence into a list.

        Args:
            value: The input to coerce.
            expected: The expected list form.
        """
        assert _as_list(value) == expected

    def test_style_of_strips_match_keys(self):
        """_style_of drops the match-only keys, keeping just the renderable style."""
        params = {
            "match": ["x"],
            "standard_name": ["y"],
            "match_units": ["K"],
            "cmap": "viridis",
            "units": "m",
        }
        assert _style_of(params) == {"cmap": "viridis", "units": "m"}

    @pytest.mark.parametrize(
        "alias, name, expected",
        [
            ("precip", "precipitation", True),  # prefix within a token
            ("2t", "2t_daily_mean", True),  # at a token start after the boundary
            ("tp", "output", False),  # mid-token coincidence is rejected
            ("rain", "terrain", False),  # mid-token coincidence is rejected
            ("", "anything", False),  # an empty alias never matches (N1 guard)
        ],
    )
    def test_alias_in(self, alias, name, expected):
        """_alias_in matches only at a token start, and an empty alias never matches.

        Args:
            alias: The candidate alias.
            name: The (lower-cased) variable name to test against.
            expected: Whether the alias should be considered present.
        """
        assert _alias_in(alias, name) is expected, f"_alias_in({alias!r}, {name!r})"


class TestLoadMagicsLibrary:
    """Tests for load_magics_library."""

    def test_covers_common_fields(self):
        """The shipped library defines the common operational fields."""
        lib = load_magics_library()
        assert {
            "temperature_2m",
            "mean_sea_level_pressure",
            "total_precipitation",
        } <= set(lib)

    def test_entries_carry_canonical_style(self):
        """Every entry carries a colormap and a magics_name (its canonical style identity)."""
        for name, params in load_magics_library().items():
            assert "cmap" in params, name
            assert "magics_name" in params, name

    def test_is_cached(self):
        """load_magics_library is lru_cached, so repeated calls return the same object."""
        assert load_magics_library() is load_magics_library()


class TestMagicsStyle:
    """Tests for magics_style identity matching."""

    def test_match_by_name_exact_alias(self):
        """An exact short name matches its alias and yields the canonical style."""
        style = magics_style("t2m")
        assert style["cmap"] == "coolwarm"
        assert style["magics_name"] == "t2m"
        assert style["levels"][0] == -40
        assert "match" not in style

    def test_match_by_name_substring(self):
        """A decorated name (e.g. a daily-mean suffix) still matches by substring."""
        assert magics_style("2t_daily_mean")["magics_name"] == "t2m"

    def test_match_by_standard_name(self):
        """When the short name is unknown, the CF standard_name resolves the field."""
        style = magics_style("unknown", standard_name="air_pressure_at_mean_sea_level")
        assert style["magics_name"] == "msl"
        assert style["units"] == "hPa"

    def test_match_by_units_fallback(self):
        """Units are a last-resort match (here, gpm -> geopotential height)."""
        assert magics_style(units="gpm")["magics_name"] == "z"

    def test_name_takes_precedence_over_standard_name(self):
        """Name matching wins even when a (conflicting) standard_name is also supplied."""
        style = magics_style("tp", standard_name="air_temperature")
        assert style["magics_name"] == "tp"

    def test_standard_name_takes_precedence_over_units(self):
        """With no name, standard_name is tried before units (and a conflicting unit is ignored)."""
        style = magics_style(standard_name="air_temperature", units="gpm")
        assert style["magics_name"] == "t2m", (
            "standard_name (t2m) must win over the gpm unit (z)"
        )

    @pytest.mark.parametrize(
        "name, standard_name, units, expected",
        [
            ("T2M", None, None, "t2m"),  # name, upper-case
            ("Total_Precipitation", None, None, "tp"),  # name, mixed-case substring
            (
                None,
                "AIR_PRESSURE_AT_MEAN_SEA_LEVEL",
                None,
                "msl",
            ),  # standard_name, upper-case
            (None, None, "GPM", "z"),  # units, upper-case
        ],
    )
    def test_matching_is_case_insensitive(self, name, standard_name, units, expected):
        """Every identity key (name, standard_name, units) matches case-insensitively.

        Args:
            name: The variable/short name identity (or None).
            standard_name: The CF standard_name identity (or None).
            units: The units identity (or None).
            expected: The magics_name the field should resolve to.
        """
        style = magics_style(name, standard_name=standard_name, units=units)
        assert style is not None, f"{name!r}/{standard_name!r}/{units!r} -> {style}"
        assert style["magics_name"] == expected, (
            f"{name!r}/{standard_name!r}/{units!r} -> {style}"
        )

    def test_match_by_real_units(self):
        """A genuine field unit (K, in temperature's match_units) resolves via the units pass."""
        assert magics_style(units="K")["magics_name"] == "t2m"

    @pytest.mark.parametrize("name", ["output", "footprint", "OUTPUT", "FootPrint"])
    def test_no_mid_token_false_match(self, name):
        """A short alias must not match a mid-token coincidence (e.g. 'tp' inside 'output') (L2).

        Args:
            name: A variable name that merely contains an alias as a mid-token substring.
        """
        assert magics_style(name) is None, f"{name!r} should not resolve to any field"

    @pytest.mark.parametrize(
        "name, expected",
        [("precipitation", "tp"), ("temperature", "t2m"), ("geopotential_height", "z")],
    )
    def test_token_prefix_still_matches(self, name, expected):
        """An alias that is a prefix of a token still matches (anchored, not whole-token equality).

        Args:
            name: A field name whose leading token starts with an alias.
            expected: The magics_name it should resolve to.
        """
        assert magics_style(name)["magics_name"] == expected, (
            f"{name!r} -> {magics_style(name)}"
        )

    def test_no_match_returns_none(self):
        """An unrecognised field returns None so the caller can fall back."""
        assert magics_style("mystery", standard_name="nope", units="parsecs") is None

    def test_all_none_returns_none(self):
        """With no identity at all, nothing matches (every strategy is skipped)."""
        assert magics_style() is None

    def test_library_override(self):
        """A caller-supplied library overrides the shipped one."""
        lib = {"x": {"match": ["foo"], "cmap": "magma", "magics_name": "x"}}
        assert magics_style("foo_bar", library=lib)["cmap"] == "magma"
        assert magics_style("other", library=lib) is None


class TestAutoStyleIntegration:
    """Tests for auto_style consulting the Magics layer before the lighter library."""

    def test_magics_supplies_levels(self):
        """A recognised field carries canonical contour levels, not just a colormap."""
        style = auto_style(_source("msl"))
        assert style["units"] == "hPa"
        assert style["levels"][0] == 960
        assert style["magics_name"] == "msl"

    def test_resolves_by_standard_name(self):
        """auto_style matches on the Source's standard_name when the variable name is opaque."""
        style = auto_style(_source("var123", standard_name="air_temperature"))
        assert style["cmap"] == "coolwarm"
        assert style["magics_name"] == "t2m"

    def test_resolves_by_units(self):
        """auto_style falls through to units matching for an otherwise-unknown field."""
        style = auto_style(_source("opaque", units="gpm"))
        assert style["magics_name"] == "z"

    def test_falls_back_to_variables_library(self):
        """A field only the lighter library knows (discharge) still resolves there."""
        style = auto_style(_source("discharge_acc"))
        assert style["cmap"] == "Blues"
        assert "magics_name" not in style

    def test_unknown_field_uses_default(self):
        """A wholly unknown field falls back to the default colormap."""
        assert auto_style(_source("mystery_variable"))["cmap"] == "viridis"


def _netcdf_with_identity(standard_name, units, variable="var123"):
    """Build an in-memory NetCDF whose one variable declares a CF identity.

    Args:
        standard_name: The CF ``standard_name`` attribute to write on the variable.
        units: The CF ``units`` attribute to write on the variable.
        variable: The variable's (deliberately opaque) name.

    Returns:
        pyramids.netcdf.NetCDF: the container, with the attributes readable through ``meta_data``.
    """
    from pyramids.dataset import Dataset, GeoReference
    from pyramids.netcdf import NetCDF

    geo = GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326)
    nc = NetCDF.from_array(
        np.ones((1, 4, 5), "float32"), geo_ref=geo, variable_name=variable
    )
    band = Dataset.from_array(
        np.ones((4, 5), "float32"), geo_ref=geo, no_data_value=-9999
    )
    attrs = {}
    if standard_name is not None:
        attrs["standard_name"] = standard_name
    if units is not None:
        attrs["units"] = units
    nc.set_variable(variable, band, attrs=attrs)
    return nc


class TestIdentityFromRealInputs:
    """Regression tests for #231 — the matcher's standard_name / units steps were unreachable in practice.

    The two integration tests above hand ``auto_style`` a ``Source`` built by hand with the metadata already
    in it, so they passed over a path no real file reached: no extractor wrote ``standard_name``, and
    ``_from_netcdf`` wrote no ``units`` either — on the one input type where CF metadata actually lives.
    These build the ``Source`` with ``get_source``, from a NetCDF that declares the identity itself.
    """

    def test_netcdf_standard_name_resolves_the_colormap(self):
        """An opaque variable name styled by its CF standard_name, straight off the file.

        Test scenario:
            The reported repro: a variable called ``var123`` carrying
            ``standard_name: air_temperature`` fell through to viridis.
        """
        from digitalearth.base.sources import get_source

        src = get_source(_netcdf_with_identity("air_temperature", "K"))
        style = auto_style(src)
        assert src.metadata("standard_name") == "air_temperature", (
            f"the extractor must carry the CF standard_name, got {src.metadata('standard_name')!r}"
        )
        assert style["cmap"] == "coolwarm", (
            f"expected the temperature colormap, got {style.get('cmap')!r}"
        )
        assert style["magics_name"] == "t2m", (
            f"expected the temperature Magics preset, got {style.get('magics_name')!r}"
        )

    def test_netcdf_units_reach_the_source(self):
        """``Source.units`` is populated from the variable's CF ``units``.

        Test scenario:
            ``_from_raster`` set units from ``band_units`` but ``_from_netcdf`` set none, so the matcher's
            units fallback was dead exactly where CF units live.
        """
        from digitalearth.base.sources import get_source

        src = get_source(_netcdf_with_identity(None, "gpm"))
        assert src.units == "gpm", (
            f"expected the CF units on the Source, got {src.units!r}"
        )

    def test_netcdf_units_alone_resolve_a_style(self):
        """With no name and no standard_name to go on, the units step now has something to match.

        Test scenario:
            ``gpm`` is geopotential height's unit; matching it is the matcher's last-resort step.
        """
        from digitalearth.base.sources import get_source

        style = auto_style(get_source(_netcdf_with_identity(None, "gpm")))
        assert style["magics_name"] == "z", (
            f"the units fallback should have resolved geopotential, got {style.get('magics_name')!r}"
        )
