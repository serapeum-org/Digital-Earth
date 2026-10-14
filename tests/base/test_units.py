"""Tests for base.units — display-time unit conversion (gap D)."""

import numpy as np
import pytest

from digitalearth.base.units import can_convert, convert, convert_array, normalize_units


class TestNormalize:
    """normalize_units folds spellings to one canonical key."""

    def test_common_spellings(self):
        """Symbols, CF strings and words all fold to the same key."""
        assert normalize_units("K") == "kelvin"
        assert normalize_units("deg C") == "celsius"
        assert normalize_units("°C") == "celsius"
        assert normalize_units("m s-1") == "m/s"
        assert normalize_units("mbar") == "hpa"

    def test_space_separated_long_forms(self):
        """Spaced CF long forms fold like their underscore spellings."""
        assert normalize_units("degrees celsius") == "celsius"
        assert normalize_units("degree Celsius") == "celsius"
        assert normalize_units("degrees fahrenheit") == "fahrenheit"

    def test_unknown_is_returned_lowercased(self):
        """An unknown unit comes back as-given (lower-cased), not mangled."""
        assert normalize_units("Widgets") == "widgets"


class TestConvert:
    """convert() on scalars."""

    def test_temperature_kelvin_to_celsius(self):
        assert convert(273.15, "K", "celsius") == 0.0
        assert round(convert(300.0, "K", "celsius"), 2) == 26.85

    def test_temperature_celsius_to_fahrenheit(self):
        assert round(convert(100.0, "celsius", "fahrenheit"), 6) == 212.0
        assert round(convert(0.0, "celsius", "fahrenheit"), 6) == 32.0

    def test_pressure_pa_to_hpa(self):
        assert convert(101325.0, "Pa", "hPa") == 1013.25

    def test_precip_metre_to_mm(self):
        assert convert(0.005, "m", "mm") == 5.0

    def test_speed_ms_to_kmh(self):
        assert round(convert(10.0, "m/s", "km/h"), 1) == 36.0

    def test_identity_is_exact(self):
        assert convert(42.0, "hPa", "hPa") == 42.0

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="unknown unit"):
            convert(1.0, "widgets", "K")

    def test_cross_family_raises(self):
        with pytest.raises(ValueError, match="different"):
            convert(1.0, "celsius", "Pa")


class TestConvertArray:
    """convert_array() on fields, preserving NaN."""

    def test_converts_and_returns_target_units(self):
        out, units = convert_array(np.array([273.15, 373.15]), "K", "celsius")
        assert units == "celsius"
        assert [round(float(v), 6) for v in out] == [0.0, 100.0]

    def test_nan_is_preserved(self):
        """A masked nodata cell stays NaN rather than becoming a number."""
        out, _ = convert_array(np.array([273.15, np.nan, 373.15]), "K", "celsius")
        assert (
            round(float(out[0]), 6) == 0.0
            and np.isnan(out[1])
            and round(float(out[2]), 6) == 100.0
        )

    def test_cross_family_raises(self):
        values = np.array([1.0])
        with pytest.raises(ValueError, match="different"):
            convert_array(values, "m/s", "Pa")

    def test_offset_family_identity_is_exact(self):
        """celsius->celsius returns the values unchanged, not an affine round-trip that drifts.

        Test scenario:
            x + 273.15 - 273.15 != x in float64, so a full round-trip would drift; the docstring promises
            the values come back unchanged, which must hold for an offset family too, not just offset-0.
        """
        values = np.array([0.1, 20.0, -5.3])
        out, units = convert_array(values, "celsius", "celsius")
        assert np.array_equal(out, values), f"identity drifted: {out} != {values}"
        assert units == "celsius"


class TestCanConvert:
    """can_convert() tests a pair without converting."""

    def test_same_family_true(self):
        assert can_convert("K", "celsius") is True

    def test_cross_family_false(self):
        assert can_convert("K", "hPa") is False

    def test_unknown_unit_false(self):
        assert can_convert("K", "widgets") is False
