"""Tests for :mod:`digitalearth.base.sources.bandname` — the one shared band-name lookup (#351).

The lookup lived twice (``web._band_name``, ``interactive.coloured_by``) and agreed only by maintenance; it now
lives here and both tiers delegate to it. These tests pin the contract that kept the two in step — band name
first, first variable next, the shared placeholder last — so the cross-tier promise rests on a measured rule
rather than on two copies happening to match.
"""

import pytest

from digitalearth.base.sources import UNNAMED_BAND, band_field_name
from digitalearth.base.spec import DEFAULT_BAND


class _Named:
    """A stand-in raster that declares ``band_names`` and/or ``variable_names`` like a pyramids object."""

    def __init__(self, band_names=None, variable_names=None):
        """Store whichever metadata lists the scenario sets, leaving the others unset.

        Args:
            band_names: The raster's band names, or ``None`` to declare none.
            variable_names: The raster's variable names, or ``None`` to declare none.
        """
        if band_names is not None:
            self.band_names = band_names
        if variable_names is not None:
            self.variable_names = variable_names


class TestUnnamedBand:
    """The shared placeholder a nameless band falls back to."""

    def test_the_placeholder_is_the_empty_axis_name(self):
        """The constant is ``"z"`` — the value axis a bare-array source carries, so a nameless raster reads the
        same on every tier.

        Test scenario:
            A fixed value both tiers used independently (web ``_UNNAMED_BAND``, interactive ``UNNAMED_VALUE``);
            pinned so a change to one shared constant is a deliberate, cross-tier decision.
        """
        assert UNNAMED_BAND == "z", (
            f"the shared placeholder changed to {UNNAMED_BAND!r}"
        )


class TestBandFieldName:
    """Tests for :func:`band_field_name`."""

    def test_a_named_band_returns_its_own_name(self):
        """A raster whose band carries a name answers with that name.

        Test scenario:
            ``band_names=["elevation", "slope"]`` with ``band=1`` returns ``"elevation"`` — the honest name
            for the field the colour is bound to.
        """
        result = band_field_name(_Named(band_names=["elevation", "slope"]), 1)
        assert result == "elevation", f"expected 'elevation', got {result!r}"

    def test_the_band_index_is_one_based(self):
        """``band`` is 1-based, so ``band=2`` selects the second name.

        Test scenario:
            ``band_names=["elevation", "slope"]`` with ``band=2`` returns ``"slope"``, not ``"elevation"``,
            confirming the ``band - 1`` offset.
        """
        result = band_field_name(_Named(band_names=["elevation", "slope"]), 2)
        assert result == "slope", (
            f"expected 'slope' for the 1-based second band, got {result!r}"
        )

    def test_the_default_band_is_the_first(self):
        """Called with no band, the default (:data:`DEFAULT_BAND` = 1) picks the first name.

        Test scenario:
            ``band_names=["t2m", "sp"]`` with the default band returns ``"t2m"``.
        """
        assert DEFAULT_BAND == 1, (
            f"this test assumes DEFAULT_BAND == 1, got {DEFAULT_BAND}"
        )
        result = band_field_name(_Named(band_names=["t2m", "sp"]))
        assert result == "t2m", (
            f"expected the first band 't2m' by default, got {result!r}"
        )

    @pytest.mark.parametrize("band", [0, 3, -1, 99])
    def test_a_band_out_of_range_falls_through_to_the_variable(self, band):
        """A band index outside ``band_names`` falls through to the first variable name.

        Args:
            band: A 1-based index with no entry in a two-name list (too low, too high, or negative).

        Test scenario:
            ``band_names=["a", "b"]`` plus ``variable_names=["v0"]``; any out-of-range band returns ``"v0"``
            rather than raising an ``IndexError`` or inventing a name.
        """
        data = _Named(band_names=["a", "b"], variable_names=["v0"])
        result = band_field_name(data, band)
        assert result == "v0", (
            f"band {band} should fall through to the first variable, got {result!r}"
        )

    def test_an_empty_band_name_falls_through(self):
        """A band present but named with the empty string is treated as unnamed.

        Test scenario:
            ``band_names=[""]`` with ``variable_names=["flow"]`` returns ``"flow"`` — an empty name is no name,
            and an :class:`~digitalearth.base.spec.encoding.Encoding` refuses the empty string as a field name.
        """
        result = band_field_name(_Named(band_names=[""], variable_names=["flow"]), 1)
        assert result == "flow", (
            f"an empty band name should fall through to the variable, got {result!r}"
        )

    def test_a_netcdf_with_no_bands_uses_its_first_variable(self):
        """With no ``band_names`` at all, the first variable name stands in.

        Test scenario:
            ``variable_names=["t2m", "sp"]`` and no ``band_names`` returns ``"t2m"`` — the first variable is the
            one the display extractor takes, so it is the one a colour key would explain.
        """
        result = band_field_name(_Named(variable_names=["t2m", "sp"]), 1)
        assert result == "t2m", f"expected the first variable 't2m', got {result!r}"

    def test_band_names_win_over_variable_names(self):
        """When both are present, the band's own name is preferred.

        Test scenario:
            ``band_names=["dem"]`` and ``variable_names=["ignored"]`` returns ``"dem"``.
        """
        result = band_field_name(
            _Named(band_names=["dem"], variable_names=["ignored"]), 1
        )
        assert result == "dem", (
            f"the band name should win over the variable, got {result!r}"
        )

    def test_neither_list_yields_the_shared_placeholder(self):
        """A raster declaring neither list falls back to :data:`UNNAMED_BAND`.

        Test scenario:
            A bare object has no ``band_names`` and no ``variable_names`` (read through ``getattr``), so the
            result is the shared placeholder, never an empty string.
        """
        result = band_field_name(object(), 1)
        assert result == UNNAMED_BAND, (
            f"expected the placeholder {UNNAMED_BAND!r}, got {result!r}"
        )

    @pytest.mark.parametrize("empty", [None, []])
    def test_none_or_empty_metadata_falls_back(self, empty):
        """``band_names``/``variable_names`` that are ``None`` or empty are treated as absent.

        Args:
            empty: The falsey metadata value under test — ``None`` or an empty list.

        Test scenario:
            Both lists set to the same falsey value return the placeholder; the ``or []`` guard means neither a
            missing attribute nor an empty one raises.
        """
        result = band_field_name(_Named(band_names=empty, variable_names=empty), 1)
        assert result == UNNAMED_BAND, (
            f"falsey metadata {empty!r} should fall back, got {result!r}"
        )

    def test_the_name_is_returned_as_a_plain_string(self):
        """A non-string band name is coerced to ``str``, since a field name must be one.

        Test scenario:
            ``band_names=[1]`` (a numeric label) returns ``"1"`` — a plain string an encoding can carry.
        """
        result = band_field_name(_Named(band_names=[1]), 1)
        assert result == "1", f"expected the coerced string '1', got {result!r}"
        assert isinstance(result, str), (
            f"the result must be a str, got {type(result).__name__}"
        )
