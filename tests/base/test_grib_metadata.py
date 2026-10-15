"""Tests for surfacing GRIB NWP metadata onto a Source, and into captions / autostyle (gap #458).

The GRIB *driver* is not installed in CI, so real ``.grib2`` files cannot be opened here (pyramids' own GRIB
doctests are ``+SKIP`` for the same reason). The ``grib_dataset`` / ``grib_stack`` fixtures instead carry the
``GRIB_*`` band-metadata keys on an in-memory GDAL dataset, which pyramids' ``grib_band_metadata`` decodes
driver-free — so these exercise the real extraction path, through pyramids' own decoder, with no GRIB file.
"""

import numpy as np

from digitalearth.base.captions import caption, format_metadata
from digitalearth.base.sources import get_source
from digitalearth.base.sources.extractors import _grib_identity, _unbracket


class TestGribMetadataOnSource:
    """get_source folds a GRIB band's decoded message metadata onto the Source."""

    def test_identity_keys_are_surfaced(self, grib_dataset):
        """element, short_name, discipline and comment reach the Source as metadata."""
        src = get_source(grib_dataset)
        assert src.metadata("element") == "TMP"
        assert src.metadata("short_name") == "2-HTGL"
        assert src.metadata("discipline") == "0"
        assert src.metadata("comment") == "Temperature"

    def test_valid_time_and_step_are_decoded(self, grib_dataset):
        """valid_time is a decoded datetime and the forecast horizon is surfaced as ``step`` seconds."""
        src = get_source(grib_dataset)
        valid_time = src.metadata("valid_time")
        assert valid_time is not None
        assert valid_time.year == 2021
        assert src.metadata("step") == 21600

    def test_units_come_from_the_grib_unit_unbracketed(self, grib_dataset):
        """A GRIB ``[C]`` unit becomes the Source's units without the brackets GDAL wraps it in."""
        assert get_source(grib_dataset).units == "C"

    def test_unit_only_grib_band_still_gets_units(self, grib_dataset_unitonly):
        """A GRIB band declaring only a unit (no identity/time) still resolves its units.

        Test scenario:
            The GRIB_UNIT fallback was gated on the decoded identity dict being non-empty, so a band whose
            message decodes to no identity/time keys lost its unit. Gating on GRIB *detection* instead, the
            unit is still read and unbracketed.
        """
        assert get_source(grib_dataset_unitonly).units == "K"

    def test_human_name_falls_back_to_the_grib_comment(self, grib_dataset):
        """With no CF long/standard name, the GRIB comment becomes long_name, so the caption reads it."""
        src = get_source(grib_dataset)
        assert src.metadata("long_name") == "Temperature"
        assert caption(src).title.startswith("Temperature")

    def test_caption_stamps_the_valid_time(self, grib_dataset):
        """caption() appends the decoded valid time to the title, as it does for any ``valid_time``."""
        title = caption(get_source(grib_dataset)).title
        assert "2021-01-01" in title

    def test_format_metadata_reads_the_grib_keys(self, grib_dataset):
        """A format template can pull the surfaced GRIB keys."""
        src = get_source(grib_dataset)
        assert format_metadata(src, "{element} +{step}s") == "TMP +21600s"

    def test_each_band_carries_its_own_step(self, grib_stack):
        """A multi-step stack surfaces a distinct step/valid_time per band."""
        steps = [get_source(grib_stack, band=b).metadata("step") for b in (1, 2, 3)]
        assert steps == [21600, 43200, 64800]

    def test_no_long_name_when_the_message_names_nothing(self, grib_dataset_timeonly):
        """A GRIB band with a time but no comment/short_name/element gets no long_name, yet still surfaces step.

        Test scenario:
            _from_raster derives long_name from the GRIB comment/short_name/element. When all three are absent
            (only a valid_time + forecast horizon are present), that derivation must be skipped — no long_name
            is invented — while the time/step keys still reach the Source.
        """
        src = get_source(grib_dataset_timeonly)
        assert src.metadata("long_name") is None
        assert src.metadata("step") == 21600


class TestGribMetadataThroughCollection:
    """A DatasetCollection of GRIB members keeps each member's decoded NWP metadata."""

    def test_collection_member_keeps_grib_keys(self, grib_collection):
        """A collection member's Source carries the GRIB keys + derived long_name, plus its position.

        Test scenario:
            _from_collection rebuilt the Source and copied forward only a subset of keys, dropping the GRIB
            metadata _from_raster now surfaces. A collection of GRIB members must keep element / step /
            long_name so caption/autostyle/quickpanels(titles_from=) work over a per-step forecast stack.
        """
        src = get_source(grib_collection)
        assert src.metadata("element") == "TMP"
        assert src.metadata("step") == 21600
        assert src.metadata("long_name") == "Temperature"
        assert src.metadata("member") == 0
        assert src.metadata("n_members") == 2


class TestGribDecodeCached:
    """The GRIB decode is cached per dataset, so a stack is not re-decoded once per band."""

    def test_decoder_runs_once_per_dataset_across_bands(self, grib_stack, mocker):
        """Reading every band of a GRIB stack calls pyramids' decoder once, not once per band.

        Test scenario:
            grib_band_metadata decodes all bands, so reading an n-band stack band-by-band without a cache
            calls it n times (quadratic). The per-dataset cache must collapse that to a single decode.
        """
        import pyramids.grib as grib_module

        spy = mocker.spy(grib_module, "grib_band_metadata")
        for band in (1, 2, 3):
            get_source(grib_stack, band=band)
        assert spy.call_count == 1, (
            f"expected one decode for the dataset, got {spy.call_count}"
        )


class TestGribCacheInvalidation:
    """Reassigning a dataset's band metadata in place is not served a stale cached decode."""

    def test_in_place_mutation_is_not_served_stale(self, grib_dataset):
        """After band_meta_data is reassigned on the same Dataset, the identity refreshes with the units.

        Test scenario:
            The per-dataset decode cache is keyed by Dataset identity with no invalidation. Reassigning
            band_meta_data in place and re-reading returned the stale cached identity (element stayed 'TMP')
            while units re-read fresh from GRIB_UNIT — an internally inconsistent Source. The identity must
            refresh too.
        """
        first = get_source(grib_dataset)
        assert first.metadata("element") == "TMP"
        assert first.units == "C"
        grib_dataset.band_meta_data = [
            {
                "GRIB_ELEMENT": "PRES",
                "GRIB_SHORT_NAME": "0-SFC",
                "GRIB_UNIT": "[Pa]",
                "GRIB_VALID_TIME": "1609480800 sec UTC",
                "GRIB_FORECAST_SECONDS": "21600 sec",
            }
        ]
        second = get_source(grib_dataset)
        assert second.metadata("element") == "PRES", "stale cached identity served after mutation"
        assert second.units == "Pa", "units should refresh with the identity"


class TestNonGribRasterUnaffected:
    """A plain raster gains none of the GRIB keys and pays no decode cost."""

    def test_no_grib_keys_leak_onto_a_plain_raster(self, dataset):
        """A GeoTIFF Source has no element/valid_time/step metadata."""
        src = get_source(dataset)
        assert src.metadata("element") is None
        assert src.metadata("valid_time") is None
        assert src.metadata("step") is None

    def test_grib_identity_is_empty_without_grib_keys(self, dataset):
        """_grib_identity short-circuits to {} when the band metadata carries no GRIB_* key."""
        assert _grib_identity(dataset, {"units": "mm"}, 0) == {}


class TestUnbracket:
    """_unbracket strips the [...] GDAL wraps a GRIB unit in, leaving other spellings alone."""

    def test_brackets_are_stripped(self):
        """A bracketed unit loses its brackets."""
        assert _unbracket("[C]") == "C"

    def test_plain_unit_passes_through(self):
        """An unbracketed unit is returned unchanged."""
        assert _unbracket("mm") == "mm"

    def test_none_and_empty_pass_through(self):
        """None stays None, and an empty/whitespace-only string collapses to None."""
        assert _unbracket(None) is None
        assert _unbracket("[]") is None


class TestAutostyleUsesGribIdentity:
    """auto_style tries the GRIB short_name / element when the band name does not match."""

    def test_short_name_drives_the_match_when_variable_is_a_bare_band_name(self):
        """A GRIB band named 'Band_1' but carrying a recognisable short_name styles by that short_name."""
        from digitalearth.base.autostyle import auto_style
        from digitalearth.base.sources import DimensionInfo, Source

        def src(metadata):
            return Source(
                DimensionInfo(np.zeros((2, 2)), "z"),
                DimensionInfo(np.array([0.0]), "x"),
                DimensionInfo(np.array([0.0]), "y"),
                metadata=metadata,
            )

        # "t2m" is recognised (temperature); as a short_name it must style the same as when it is the variable,
        # even though the variable itself is the unhelpful auto band name.
        by_variable = auto_style(src({"variable": "t2m"}))
        by_short_name = auto_style(src({"variable": "Band_1", "short_name": "t2m"}))
        assert by_short_name.get("cmap") == by_variable.get("cmap")
        assert by_variable.get("cmap") is not None
