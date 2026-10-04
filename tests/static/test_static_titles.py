"""The static tier's titles, and the colour key's fallback to the data's own units (ST-18).

Two halves of one row, and the first thing settled about it was **what already worked**. Measured on this
branch before anything was written:

```text
band_names ['msl']        -> Map._layer_labels ['hPa'], colorbar guide title 'hPa'
band_units ['widgets/s']  -> Map._layer_labels [None],  colorbar guide title None
set_title("Discharge, 2020") -> ax.get_title() 'Discharge, 2020'
                             -> figure_spec.title None, figure_spec.panels[0].title None
```

So the units :func:`~digitalearth.base.autostyle.auto_style` *recognises* already reach the bar's label — the
row's first sentence was already true — while a band carrying units nothing in the style library matches
labelled nothing at all, and a title reached matplotlib and no description. Those two are what this file
pins: the title is recorded on the figure's panel the way the 3-D tier records its own, and a colour key
falls back to the source's own units when the library resolves none.
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.base.autostyle import auto_style
from digitalearth.base.sources import DimensionInfo, Source
from digitalearth.static import Map, Scene

#: A 2x2 lat/lon geo-reference, so a field can be drawn on a `Map(crs=4326)` without reading a file.
GEO = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)


def _raster(units):
    """Return a 2x2 in-memory raster whose single band carries `units`.

    Args:
        units: The band's unit string, or `None` to leave it unset.

    Returns:
        A pyramids ``Dataset``.
    """
    data = Dataset.from_array(
        arr=np.array([[1.0, 2.0], [3.0, 4.0]]), geo_ref=GEO, no_data_value=-9999.0
    )
    if units is not None:
        data.band_units = [units]
    return data


def _source(variable, units):
    """Return a minimal :class:`~digitalearth.base.sources.Source` naming `variable` and `units`.

    Args:
        variable: The variable name the style library is matched on.
        units: The source's own unit string, or `None`.

    Returns:
        A ``Source`` over a 2x2 array.
    """
    return Source(
        DimensionInfo(np.zeros((2, 2)), "z"),
        DimensionInfo(np.array([0.0, 1.0]), "x"),
        DimensionInfo(np.array([0.0, 1.0]), "y"),
        metadata={"variable": variable},
        units=units,
    )


class TestTheTitleIsRecorded:
    """`set_title` writes the figure's description as well as the axes."""

    def test_the_title_reaches_the_panel_description(self):
        """A title set on the scene is carried by the figure's one panel.

        Test scenario:
            It was drawn and not described — `figure_spec.panels[0].title` was `None` for a titled map — so
            a figure written out lost its heading and `Map.from_figure` had nothing to restore. The 3-D tier
            has always recorded it on its panel; this is the same place.
        """
        with Scene() as scene:
            scene.set_title("Discharge, 2020")
            assert scene.figure_spec.panels[0].title == "Discharge, 2020"

    def test_the_recorded_title_reads_back_off_the_scene(self):
        """:attr:`Scene.title` answers what was set.

        Test scenario:
            The axes title is matplotlib's to report per position (`get_title(loc=...)`); the scene's own
            heading is one value, and this is where a caller reads it.
        """
        with Scene() as scene:
            scene.set_title("Depth, m")
            assert scene.title == "Depth, m"

    def test_an_untitled_scene_records_no_title(self):
        """A scene nobody titled describes no title at all.

        Test scenario:
            `PanelSpec` spells "no title" as `None`, and a figure that invented `""` for it would compare
            unequal to one that had never been titled.
        """
        with Scene() as scene:
            assert scene.figure_spec.panels[0].title is None

    def test_a_blank_title_clears_the_record(self):
        """`set_title("")` is "no title", in the record as well as on the axes.

        Test scenario:
            `""` and `"   "` are the one way to ask matplotlib for no title, and `PanelSpec` refuses a blank
            one — so a recorded blank would make a titled-then-cleared figure impossible to build.
        """
        with Scene() as scene:
            scene.set_title("Discharge")
            scene.set_title("   ")
            assert scene.title is None

    def test_a_blank_title_still_reaches_the_axes(self):
        """Clearing the record does not stop matplotlib being told.

        Test scenario:
            The record is written *beside* the `set_title` call, never instead of it, so the drawn title is
            cleared too.
        """
        with Scene() as scene:
            scene.set_title("Discharge")
            scene.set_title("")
            assert scene.ax.get_title() == ""

    def test_a_second_title_replaces_the_first(self):
        """The scene has one heading, so titling it again supersedes the record.

        Test scenario:
            Two titles are not two panels; the later call is the current one.
        """
        with Scene() as scene:
            scene.set_title("first")
            scene.set_title("second")
            assert scene.title == "second"

    def test_a_title_drawn_off_centre_is_still_recorded(self):
        """`loc="left"` moves where matplotlib draws it, not whether the figure carries it.

        Test scenario:
            The record is the figure's heading, which the caller wrote; where the axes paints it is this
            tier's own styling and travels with nothing.
        """
        with Scene() as scene:
            scene.set_title("Left-aligned", loc="left")
            assert scene.figure_spec.panels[0].title == "Left-aligned"

    def test_the_title_survives_a_round_trip_through_a_figure(self):
        """A map rebuilt from its own description comes back titled.

        Test scenario:
            This is what recording it buys. `Map.draw_figure` already restored
            `figure.title or figure.panels[0].title`, and this tier emitted neither — so a same-tier round
            trip silently dropped the heading. The redrawn title is read off the **axes**, which is the half
            a reader sees.
        """
        first = Map(crs=4326)
        first.set_title("Depth, m")
        second = Map.from_figure(first.figure_spec)
        try:
            assert second.ax.get_title() == "Depth, m"
        finally:
            first.close()
            second.close()

    def test_a_removed_layer_leaves_the_recorded_title_alone(self):
        """The title is figure decoration, so it outlives every layer.

        Test scenario:
            It is recorded on the panel rather than on a layer, which is why taking the last layer off does
            not take the heading with it.
        """
        with Map(crs=4326) as m:
            m.field(_raster(None), name="depth")
            m.set_title("Depth, m")
            m.remove_layer("depth")
            assert m.title == "Depth, m"


class TestTheColourKeyFallsBackToTheSourcesUnits:
    """A bar labels itself with the data's own units when the style library names none."""

    def test_auto_style_answers_the_sources_own_units(self):
        """`auto_style` carries `Source.units` through for a variable it does not recognise.

        Test scenario:
            Measured before the fallback: `auto_style(_source("widget_flux", "widgets/s"))` answered
            `['cmap']` and nothing else, while the source said `widgets/s`. The library has no opinion about
            this variable, and "no opinion" is not the same as "no units".
        """
        assert auto_style(_source("widget_flux", "widgets/s"))["units"] == "widgets/s"

    def test_a_recognised_variables_units_still_win(self):
        """The library's canonical units beat the source's own.

        Test scenario:
            `msl` is matched as mean sea-level pressure, whose canonical unit is hPa, and the file saying
            `Pa` is exactly the case the canonical answer exists for — the levels the same match supplies
            are in hPa too, so taking the file's word would label them wrongly.
        """
        assert auto_style(_source("msl", "Pa"))["units"] == "hPa"

    def test_a_source_with_no_units_adds_none(self):
        """A source that names no units leaves the style saying nothing about them.

        Test scenario:
            The fallback fills a gap; it must not invent a `None`-valued key for an unlabelled band, which
            a caller reading `"units" in style` would take for a label.
        """
        assert "units" not in auto_style(_source("widget_flux", None))

    def test_the_bar_is_labelled_with_the_bands_own_units(self):
        """A field whose band units nothing matches still titles its colour key.

        Test scenario:
            The end of the row, measured end to end: `band_units = ['widgets/s']` on an unrecognised band
            labelled nothing before, because the label came from `auto_style` alone.
        """
        with Map(crs=4326) as m:
            m.field(_raster("widgets/s"), name="w")
            m.colorbar("w")
            assert m.get_layer("w").symbology.guide().title == "widgets/s"

    def test_the_callers_own_label_still_wins(self):
        """`label=` beats both the library and the band.

        Test scenario:
            The fallback is only ever reached for a `label=None`, which is what `colorbar`'s contract
            already says; a caller's text is the last word.
        """
        with Map(crs=4326) as m:
            m.field(_raster("widgets/s"), name="w")
            m.colorbar("w", label="Flux")
            assert m.get_layer("w").symbology.guide().title == "Flux"

    def test_an_unlabelled_band_leaves_the_bar_unlabelled(self):
        """A band naming no units is still keyed with no title.

        Test scenario:
            The fallback must not put a stray string on a bar for data that named nothing — `Guide` refuses
            a blank title, so "nothing" has to stay `None`.
        """
        with Map(crs=4326) as m:
            m.field(_raster(None), name="plain")
            m.colorbar("plain")
            assert m.get_layer("plain").symbology.guide().title is None


@pytest.mark.parametrize("asked, recorded", [(123, "123"), (4.5, "4.5"), (None, None)])
def test_the_record_is_the_text_the_axes_draws(asked, recorded):
    """A title matplotlib accepts but a description cannot hold is recorded as the text it drew.

    Args:
        asked: What the caller passed as the title.
        recorded: What the figure should carry for it.

    Test scenario:
        Measured on a bare axes: `set_title(123)` draws `'123'`, `set_title(4.5)` draws `'4.5'` and
        `set_title(None)` draws `''`. `PanelSpec` holds a non-empty string or `None`, so the record follows
        matplotlib rather than refusing a call that has always worked — and `None` is "no title" on both
        sides rather than the string `'None'`.
    """
    with Scene() as scene:
        scene.set_title(asked)
        assert scene.title == recorded
