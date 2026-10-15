"""Tests for base.captions — titles and value labels from a Source's metadata (gap C)."""

import numpy as np

from digitalearth.base.captions import Caption, caption, format_metadata
from digitalearth.base.sources import DimensionInfo, Source


def _source(metadata, units=None):
    """A minimal Source carrying the given metadata and units."""
    return Source(
        DimensionInfo(np.zeros((2, 2)), "z"),
        DimensionInfo(np.array([0.0]), "x"),
        DimensionInfo(np.array([0.0]), "y"),
        metadata=metadata,
        units=units,
    )


class TestCaption:
    """caption(source) -> Caption(title, label)."""

    def test_name_precedence_prefers_long_name(self):
        """long_name beats standard_name beats variable, with underscores spaced."""
        src = _source(
            {
                "long_name": "2 metre temperature",
                "standard_name": "air_temperature",
                "variable": "t2m",
            }
        )
        assert caption(src).title == "2 metre temperature"

    def test_falls_back_to_standard_then_variable(self):
        """With no long_name, standard_name is used; with neither, the variable."""
        assert (
            caption(_source({"standard_name": "air_temperature"})).title
            == "air temperature"
        )
        assert caption(_source({"variable": "t2m"})).title == "t2m"

    def test_no_metadata_is_value(self):
        """A source naming nothing is captioned 'value', never blank."""
        assert caption(_source({})) == Caption(title="value", label="value")

    def test_label_appends_units_in_brackets(self):
        """The value label carries the units when the source names them."""
        assert caption(_source({"variable": "q"}, units="kg/kg")).label == "q [kg/kg]"

    def test_units_are_prettified(self):
        """Common unit spellings are tidied for display."""
        assert caption(_source({"variable": "t"}, units="celsius")).label == "t [°C]"

    def test_style_units_fill_when_source_has_none(self):
        """A style's units are used only when the source declares none."""
        assert caption(_source({"variable": "t"}), style_units="K").label == "t [K]"
        # the source's own units win over the style's
        assert (
            caption(_source({"variable": "t"}, units="°C"), style_units="K").label
            == "t [°C]"
        )

    def test_time_is_appended_to_the_title(self):
        """A time or valid_time stamp is appended to the title, not the label."""
        cap = caption(
            _source({"long_name": "sea surface temperature", "time": "2021-07"})
        )
        assert cap.title == "sea surface temperature — 2021-07"
        assert "2021-07" not in cap.label


class TestFormatMetadata:
    """format_metadata(source, template)."""

    def test_fills_named_keys(self):
        """Known metadata keys, plus {name} and {units}, fill the template."""
        src = _source({"variable": "t2m", "level": "2m"}, units="K")
        assert (
            format_metadata(src, "{variable} at {level} [{units}]") == "t2m at 2m [K]"
        )

    def test_missing_key_is_empty_not_error(self):
        """A key the source lacks becomes empty, and surrounding whitespace is trimmed."""
        src = _source({"variable": "t2m"})
        assert format_metadata(src, "{variable} {step}") == "t2m"

    def test_name_token_uses_the_human_name(self):
        """{name} resolves to the same human name caption() uses."""
        src = _source({"long_name": "total precipitation"})
        assert format_metadata(src, "{name}") == "total precipitation"
