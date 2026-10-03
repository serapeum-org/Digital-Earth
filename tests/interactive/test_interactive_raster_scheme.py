"""``InteractiveMap.field(..., scheme=, k=)`` — a raster coloured class by class on the HoloViz tier.

Expected values are computed independently of the classifier; cross-tier agreement is pinned in
``tests/test_cross_tier_raster_classes.py``.
"""

import numpy as np
import pytest

hv = pytest.importorskip("holoviews")
pytest.importorskip("geoviews")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.base.symbology import (  # noqa: E402
    categorical_colors,
    resolve_categorical_cmap,
    sample_cmap,
)
from digitalearth.interactive import InteractiveMap  # noqa: E402

CODES = np.array([[1.0, 2.0, 5.0], [5.0, 1.0, 2.0]])


def _band(values: np.ndarray):
    """Wrap ``values`` as a lon/lat source."""
    rows, columns = values.shape
    return get_source(
        values,
        x=np.linspace(10.0, 10.0 + 0.1 * (columns - 1), columns),
        y=np.linspace(50.0, 50.0 - 0.1 * (rows - 1), rows),
        crs=4326,
    )


def _options(element, group: str) -> dict:
    """Return the Bokeh options of ``group`` HoloViews resolved for ``element``."""
    return hv.Store.lookup_options("bokeh", element, group).kwargs


class TestClassifiedInteractiveField:
    """A ``scheme`` steps the image's colours at its class edges and records the classes."""

    def test_codes_step_at_half_steps_in_the_shared_palette(self):
        """Each code is a class of its own, the bar steps between codes and is ticked at them.

        Test scenario:
            Codes 1, 2, 5: colours are the categorical palette, ``color_levels`` the half-steps
            ``0.5, 1.5, 3.5, 5.5``, ``clim`` their span, and the colour bar's ticks the codes.
        """
        m = InteractiveMap(crs=4326).field(
            _band(CODES), scheme="categorical", name="cover"
        )
        element = m.layers[0]
        style, plot = _options(element, "style"), _options(element, "plot")
        palette = categorical_colors([1, 2, 5], resolve_categorical_cmap(None))[1]
        assert list(style["cmap"]) == list(palette), f"cmap {style['cmap']}"
        assert list(plot["color_levels"]) == [0.5, 1.5, 3.5, 5.5], (
            f"levels {plot['color_levels']}"
        )
        assert tuple(plot["clim"]) == (0.5, 5.5), f"clim {plot['clim']}"
        assert list(plot["colorbar_opts"]["ticker"].ticks) == [1, 2, 5], (
            "the bar should be ticked at the codes"
        )
        scale = m.figure_spec.layers.get("cover").symbology.encoding("color").scale
        assert list(scale.categories) == [1, 2, 5], f"categories {scale.categories}"

    def test_graduated_classes_step_at_their_edges(self):
        """Quantile classes take ``cmap``'s samples and step at the quantiles.

        Test scenario:
            ``0..11`` in four quantile classes of Set2; the levels are ``numpy``'s quantiles.
        """
        values = np.arange(12.0).reshape(3, 4)
        m = InteractiveMap(crs=4326).field(
            _band(values), scheme="quantiles", k=4, cmap="Set2", name="ramp"
        )
        element = m.layers[0]
        expected = list(np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0]))
        assert list(_options(element, "style")["cmap"]) == sample_cmap("Set2", 4), (
            "cmap should be Set2's samples"
        )
        assert list(_options(element, "plot")["color_levels"]) == pytest.approx(
            expected
        ), "levels are quantiles"
        recorded = m.figure_spec.layers.get("ramp").symbology.props
        assert (recorded["scheme"], recorded["k"]) == ("quantiles", 4), (
            f"recorded {recorded['scheme']}"
        )

    def test_without_a_scheme_the_ramp_is_continuous(self):
        """No ``scheme`` leaves the image a continuous ramp with no colour levels.

        Test scenario:
            The default must not step: ``color_levels`` stays unset.
        """
        m = InteractiveMap(crs=4326).field(_band(CODES), name="plain")
        assert "color_levels" not in _options(m.layers[0], "plot"), (
            "an unclassified field should not step"
        )

    def test_a_fractional_band_is_refused_as_categorical(self):
        """``scheme="categorical"`` on a magnitude is refused before anything is added.

        Test scenario:
            ``0.5`` is no code; the map stays empty.
        """
        m = InteractiveMap(crs=4326)
        band = _band(np.array([[0.5, 1.0], [2.0, 3.0]]))
        with pytest.raises(ValueError, match="integer class codes"):
            m.field(band, scheme="categorical")
        assert m.layer_ids == [], f"a refused layer was added: {m.layer_ids}"

    def test_a_path_is_opened_to_cut_the_classes(self, tmp_path):
        """A band handed as a path is opened at build time, so its classes are cut from its cells.

        Args:
            tmp_path: Where the GeoTIFF is written.

        Test scenario:
            A GeoTIFF of codes 1, 2, 5 passed as a ``str`` path: the recorded scale's categories are the
            codes, and the figure records the path itself as the layer's data.
        """
        from pyramids.dataset import Dataset, GeoReference

        path = str(tmp_path / "codes.tif")
        geo = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.1, epsg=4326)
        Dataset.from_array(CODES.astype(np.int32), geo_ref=geo).to_file(path)
        m = InteractiveMap(crs=4326).field(path, scheme="categorical", name="cover")
        scale = m.figure_spec.layers.get("cover").symbology.encoding("color").scale
        assert list(scale.categories) == [1, 2, 5], f"categories {scale.categories}"


class TestClassOptions:
    """``_class_options`` — the HoloViews options that step an image at its classes."""

    def test_graduated_classes_have_no_ticker(self):
        """Graduated classes step at their edges, over their span, with no code ticks.

        Test scenario:
            ``0..9`` in three equal intervals: levels ``0, 3, 6, 9``, ``clim`` ``(0, 9)``, three viridis
            samples, and no ``colorbar_opts`` — the edges are the bar's own ticks.
        """
        from digitalearth.base.raster_classes import classify_band

        from digitalearth.interactive.raster import _class_options

        options = _class_options(
            classify_band(np.arange(10.0), "equal_interval", 3, "viridis")
        )
        assert options["color_levels"] == pytest.approx([0.0, 3.0, 6.0, 9.0]), (
            f"levels {options['color_levels']}"
        )
        assert tuple(options["clim"]) == pytest.approx((0.0, 9.0)), (
            f"clim {options['clim']}"
        )
        assert options["cmap"] == sample_cmap("viridis", 3), f"cmap {options['cmap']}"
        assert "colorbar_opts" not in options, f"unexpected ticker in {options}"

    def test_categorical_classes_tick_at_the_codes(self):
        """Categorical classes step at the half-steps between codes and tick the bar at the codes.

        Test scenario:
            Codes 1, 2, 5: levels ``0.5, 1.5, 3.5, 5.5`` and ticks ``1, 2, 5``.
        """
        from digitalearth.base.raster_classes import classify_band

        from digitalearth.interactive.raster import _class_options

        options = _class_options(classify_band(CODES, "categorical", None, None))
        assert options["color_levels"] == [0.5, 1.5, 3.5, 5.5], (
            f"levels {options['color_levels']}"
        )
        assert list(options["colorbar_opts"]["ticker"].ticks) == [1, 2, 5], (
            "the bar should be ticked at the codes"
        )


class TestStaticFigureReplay:
    """``draw_image`` draws a figure the static tier classified, dropping its cleopatra-only keywords."""

    def test_static_scheme_keywords_are_dropped_and_classes_drawn(self, tmp_path):
        """A static-tier figure recording ``scheme``/``k`` draws here, stepped at its recorded edges.

        Args:
            tmp_path: Where the GeoTIFF is written.

        Test scenario:
            The static tier records ``scheme="quantiles"``, ``k=4`` among its options; HoloViews has no
            option of either name, so they are stripped, and the image steps at ``numpy``'s quantiles in
            Set2's samples.
        """
        from pyramids.dataset import Dataset, GeoReference

        from digitalearth import to_backend
        from digitalearth.base.spec import FigureSpec
        from digitalearth.static import Map

        values = np.arange(12.0, dtype=np.float32).reshape(3, 4)
        path = str(tmp_path / "ramp.tif")
        geo = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.1, epsg=4326)
        Dataset.from_array(values, geo_ref=geo).to_file(path)
        with Map(crs=4326) as scene:
            scene.field(path, scheme="quantiles", k=4, cmap="Set2", name="ramp")
            spec = FigureSpec.from_dict(scene.figure_spec.to_dict())
        drawn = to_backend(spec, backend="interactive")
        element = drawn.layers[0]
        style, plot = _options(element, "style"), _options(element, "plot")
        assert "scheme" not in {**style, **plot} and "k" not in {**style, **plot}, (
            f"cleopatra keywords leaked into HoloViews options: {sorted({**style, **plot})}"
        )
        expected = list(np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0]))
        assert list(plot["color_levels"]) == pytest.approx(expected), (
            f"levels {plot['color_levels']}, expected {expected}"
        )
        assert list(style["cmap"]) == sample_cmap("Set2", 4), f"cmap {style['cmap']}"
