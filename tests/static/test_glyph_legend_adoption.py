"""A categorical glyph's own swatch legend is adopted as the layer's guide artist (#350).

A categorical fill's cleopatra glyph draws its **own** swatch legend on its way in. Before #350 that artist
was never recorded as the layer's guide: ``DrawnLayer.guides`` stayed empty while a colour key the layer's
record knew nothing about sat on the axes, so ``set_visible`` did not hide it, ``remove`` did not take it off,
and ``quickmap(colorbar=False)`` had to draw the key through the tier's own method first only to take it off
again. The renderer now adopts that legend
(:meth:`~digitalearth.static.render.renderer.Renderer._adopt_glyph_legend`), so the whole guide lifecycle owns it.

These tests pin the adoption and answer the issue's three open questions: ``set_visible(False)`` hides it,
``colorbars()`` leaves a categorical layer alone (its key is a swatch list, not a bar), and the adoption is
the renderer's alone — the layer's description records no guide, so what a figure carries is unchanged.
"""

import geopandas as gpd
import matplotlib
import pytest
from shapely.geometry import Polygon

matplotlib.use("Agg")

from digitalearth import Map  # noqa: E402 - after the Agg backend is set


@pytest.fixture
def zones():
    """Return a four-feature categorical ``FeatureCollection`` of unit squares in three zones.

    Returns:
        A :class:`pyramids.feature.FeatureCollection` whose ``zone`` column takes three distinct values, so a
        categorical choropleth over it draws a three-swatch legend.
    """
    from pyramids.feature import FeatureCollection

    gdf = gpd.GeoDataFrame(
        {"zone": ["a", "b", "c", "a"]},
        geometry=[Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(4)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


def _glyph_legend(scene, layer_id):
    """Return the swatch legend a layer's categorical glyph drew for itself.

    Args:
        scene: The map holding the drawn layer.
        layer_id: The categorical fill's layer id.

    Returns:
        The ``matplotlib.legend.Legend`` the glyph exposes as ``category_legend``.
    """
    return scene._renderer.drawn[layer_id].glyph.category_legend


class TestTheGlyphLegendIsAdopted:
    """The legend the categorical glyph drew becomes the layer's one recorded guide."""

    def test_it_is_the_layers_recorded_guide(self, zones):
        """``DrawnLayer.guides`` holds the glyph's own legend, where it held nothing before #350.

        Args:
            zones: A three-zone categorical ``FeatureCollection``.

        Test scenario:
            Straight after the fill and before any ``legend()`` call, the record's one guide is the very
            legend on the axes — the gap the issue measured as ``guides == ()`` with the legend unowned.
        """
        with Map(crs=4326) as m:
            m.choropleth(zones, column="zone", scheme="categorical", name="z")
            drawn = m._renderer.drawn["z"]
            assert drawn.guides == (_glyph_legend(m, "z"),), (
                f"the glyph's legend should be the layer's one guide; got {drawn.guides}"
            )
            assert m.ax.get_legend() is drawn.guides[0], (
                "the adopted guide should be the legend the axes is showing"
            )

    def test_set_visible_false_hides_it(self, zones):
        """``set_visible(False)`` hides the adopted legend, which it could not reach before #350.

        Args:
            zones: A three-zone categorical ``FeatureCollection``.

        Test scenario:
            Measured before the change: the glyph's legend stayed ``visible=True`` through
            ``set_visible('z', False)`` because the record did not own it. Now the flag reaches it, and
            ``set_visible('z', True)`` brings it back.
        """
        with Map(crs=4326) as m:
            m.choropleth(zones, column="zone", scheme="categorical", name="z")
            m.set_visible("z", False)
            assert m.ax.get_legend().get_visible() is False, (
                "set_visible(False) did not hide the adopted swatch legend"
            )
            m.set_visible("z", True)
            assert m.ax.get_legend().get_visible() is True, (
                "set_visible(True) did not bring the adopted swatch legend back"
            )

    def test_remove_layer_takes_it_off(self, zones):
        """Removing the layer takes its adopted legend off the axes with it.

        Args:
            zones: A three-zone categorical ``FeatureCollection``.

        Test scenario:
            The key follows the layer it explains: ``remove_layer`` detaches the adopted legend, leaving the
            axes with no legend at all.
        """
        with Map(crs=4326) as m:
            m.choropleth(zones, column="zone", scheme="categorical", name="z")
            m.remove_layer("z")
            assert m.ax.get_legend() is None, (
                "remove_layer left the categorical fill's swatch legend on the axes"
            )


class TestAdoptionDoesNotChangeTheDescription:
    """The adoption is the renderer's: the layer's description records no guide (open question 3)."""

    def test_the_layer_records_no_guide(self, zones):
        """A bare categorical fill carries no ``Guide`` in its symbology, as it did before #350.

        Args:
            zones: A three-zone categorical ``FeatureCollection``.

        Test scenario:
            Adoption records which artist comes off with the layer and nothing in the description, so a
            figure's ``to_dict`` carries exactly what it did before — the record the renderer holds is the
            only thing that changed.
        """
        with Map(crs=4326) as m:
            m.choropleth(zones, column="zone", scheme="categorical", name="z")
            assert m.get_layer("z").symbology.guide() is None, (
                "adoption should not record a guide in the layer's description"
            )


class TestColorbarsLeavesACategoricalLayerAlone:
    """``colorbars()`` keys every value-coloured layer with a bar, and skips categorical ones."""

    def test_the_swatch_legend_survives_colorbars(self, zones):
        """``colorbars()`` adds no bar for a categorical fill and leaves its swatch legend in place.

        Args:
            zones: A three-zone categorical ``FeatureCollection``.

        Test scenario:
            A categorical scale cannot be read as a bar (``bar_refusal``), so ``colorbars()`` skips the
            layer — the figure keeps its single axes and the adopted swatch legend stays on it, rather than
            the call choking on a key the guide path never drew (open question 2).
        """
        with Map(crs=4326) as m:
            m.choropleth(zones, column="zone", scheme="categorical", name="z")
            m.colorbars()
            assert len(m.fig.axes) == 1, (
                f"colorbars() should add no colorbar axes for a categorical fill; got {len(m.fig.axes)}"
            )
            assert m.ax.get_legend() is not None, (
                "colorbars() removed the categorical fill's swatch legend"
            )


class TestQuickmapColorbarFalseTakesOneCall:
    """``quickmap(colorbar=False)`` now leaves no key, through a single take-off call (#350)."""

    def test_no_legend_is_left_on_the_axes(self, zones):
        """The map comes back with no swatch legend and a guide recorded switched off.

        Args:
            zones: A three-zone categorical ``FeatureCollection``.

        Test scenario:
            The glyph's own legend is the layer's guide, so one ``legend(visible=False)`` records the
            decision and takes the legend off — the take-over draw that used to be needed first is gone. The
            decision survives as ``Guide(show=False)`` so a switcher can offer the key back.
        """
        from digitalearth import quickmap

        m = quickmap(
            zones, crs=zones.epsg, column="zone", scheme="categorical", colorbar=False
        )
        try:
            assert m.ax.get_legend() is None, "colorbar=False left a legend on the axes"
            shown = [m.get_layer(i).symbology.guide() for i in m.layer_ids]
            assert any(g is not None and g.show is False for g in shown), (
                f"colorbar=False should record a guide switched off; got {shown}"
            )
        finally:
            m.close()


class TestQuickmapColorbarFalseOnAContinuousField:
    """The one-call collapse is safe on a continuous field too — it has no autonomous colorbar to orphan."""

    def test_no_colorbar_axis_is_left(self):
        """`quickmap(colorbar=False)` on a continuous raster leaves the figure one axes and records show=False.

        Test scenario:
            #350 collapsed the key-off to one `colorbar(visible=False)`. That is safe only if a continuous
            glyph does not draw its own colorbar the way a categorical glyph draws its own legend; the guide
            system owns the continuous colorbar, so `colorbar=False` records `Guide(show=False)` with nothing
            on the axes to take off. A continuous field's extra axis would be a colorbar axis, so one axes
            means none was drawn.
        """
        import numpy as np
        from pyramids.dataset import Dataset, GeoReference

        from digitalearth import quickmap

        field = Dataset.from_array(
            arr=np.arange(16.0).reshape(4, 4),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
            no_data_value=-9999.0,
        )
        m = quickmap(field, crs=field.epsg, colorbar=False)
        try:
            assert len(m.fig.axes) == 1, (
                f"colorbar=False should leave the figure one axes; got {len(m.fig.axes)}"
            )
            shown = [m.get_layer(i).symbology.guide() for i in m.layer_ids]
            assert any(g is not None and g.show is False for g in shown), (
                f"colorbar=False should record a guide switched off; got {shown}"
            )
        finally:
            m.close()
