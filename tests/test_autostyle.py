"""Tests for T6.2 — auto-style: Source metadata -> cleopatra style params."""

import numpy as np
import pytest
from pyramids.dataset import GeoReference

from digitalearth.autostyle import auto_style, load_library
from digitalearth.scene import Map
from digitalearth.sources import DimensionInfo, Source


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
            "digitalearth.autostyle.load_library",
            return_value={"default": {"cmap": "viridis"}, "ice": {"match": "siconc", "cmap": "Blues_r"}},
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
    m.contourf(t2m)
    assert m.layers[0][0].default_options["cmap"] == "coolwarm"


def test_explicit_cmap_overrides_auto_style(dataset):
    """An explicit cmap wins over the auto-style default."""
    m = Map(crs=dataset.epsg)
    m.imshow(dataset, cmap="magma")
    assert m.layers[0][0].default_options["cmap"] == "magma"
