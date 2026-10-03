"""``scheme=``/``k=`` on a raster field: the classes are cut, drawn, published and keyed (ST-7, #222).

A classified raster used to be impossible — cleopatra's ``ArrayGlyph.plot`` took no ``classify`` group, so
``render_compat`` refused ``scheme``/``k`` on every raster layer. Since cleopatra 0.38 it does, and the static
tier needed no new code: ``field`` forwards the flat keys, ``render_compat`` folds them into the ``classify``
group, and the scene reads the class edges back off the drawn norm. Nothing pinned that path, though, so a
renamed upstream keyword would have turned every classified raster back into a refusal with no failing test.

These tests hold each link of the chain to edges computed **independently from the data** — never read back
from the figure under test — so they cannot agree with the drawing by construction.
"""

import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm

from digitalearth.base.symbology import categorical_colors, resolve_categorical_cmap
from digitalearth.static import Map


@pytest.fixture
def values(dataset) -> np.ndarray:
    """Return the fixture raster's valid cell values (nodata masked out), as a flat float array.

    Args:
        dataset: The ``acc4000.tif`` raster fixture.

    Returns:
        The cell values the classifier sees.
    """
    band = dataset.read_array(masked=True)
    return np.ma.compressed(band).astype(float)


def _equal_interval_edges(values: np.ndarray, k: int) -> list:
    """Return the ``k`` equal-interval class edges over ``values``, computed straight from the data."""
    return list(np.linspace(values.min(), values.max(), k + 1))


def _drawn_norm(m: Map):
    """Return the norm of the last image drawn on ``m``'s axes."""
    return m.ax.images[-1].norm


def _painted(image, value: float) -> str:
    """Return the hex colour ``image`` paints a cell holding ``value``."""
    from matplotlib.colors import to_hex

    return to_hex(image.cmap(image.norm(value)))


class TestClassifiedRasterField:
    """``Map.field`` with ``scheme=``/``k=`` draws, describes and keys discrete classes."""

    def test_a_named_scheme_cuts_the_raster_into_classes(self, dataset, values):
        """``scheme="equal_interval", k=4`` draws the band with a ``BoundaryNorm`` on the data's own edges.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            The mappable must carry discrete classes — a ``BoundaryNorm`` — whose boundaries are the four
            equal-width intervals over the data's range, computed here from the values themselves.
        """
        expected = _equal_interval_edges(values, 4)
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            norm = _drawn_norm(m)
            assert isinstance(norm, BoundaryNorm), (
                f"a classified raster should draw a BoundaryNorm, got {norm!r}"
            )
            assert np.allclose(norm.boundaries, expected), (
                f"class edges {list(norm.boundaries)} should be the equal intervals {expected}"
            )

    def test_explicit_edges_are_used_as_given(self, dataset):
        """A list passed as ``scheme`` is taken as the class edges themselves.

        Args:
            dataset: The raster fixture.

        Test scenario:
            ``scheme=[0, 10, 50, 88]`` must draw exactly those boundaries — no re-cutting.
        """
        edges = [0.0, 10.0, 50.0, 88.0]
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme=edges)
            boundaries = list(_drawn_norm(m).boundaries)
        assert boundaries == edges, (
            f"explicit edges {edges} were redrawn as {boundaries}"
        )

    def test_no_scheme_stays_a_continuous_ramp(self, dataset):
        """Without ``scheme`` the field is a continuous ramp, not discrete classes.

        Args:
            dataset: The raster fixture.

        Test scenario:
            The default must be unchanged by this feature: no ``BoundaryNorm``.
        """
        with Map(crs=dataset.epsg) as m:
            m.field(dataset)
            norm = _drawn_norm(m)
        assert not isinstance(norm, BoundaryNorm), (
            f"an unclassified raster should be continuous, got {norm!r}"
        )

    def test_the_classes_are_published_as_breaks(self, dataset, values):
        """The cut edges reach ``last_breaks`` and the layer's colour encoding.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            Other tiers and the figure description read the classes from these two places, so both must
            carry the edges that were drawn.
        """
        expected = _equal_interval_edges(values, 4)
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            breaks = list(m.last_breaks)
            spec = m.figure_spec.layers.get(m.layer_ids[-1])
            encoded = list(spec.symbology.encoding("color").scale.breaks)
        assert np.allclose(breaks, expected), (
            f"last_breaks {breaks} should be {expected}"
        )
        assert np.allclose(encoded, expected), (
            f"the colour encoding's breaks {encoded} should be {expected}"
        )

    def test_the_colorbar_steps_at_the_class_edges(self, dataset, values):
        """``colorbar()`` on a classified raster puts its ticks on the class edges.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            A stepped key, not a continuous ramp: one tick per class edge.
        """
        expected = _equal_interval_edges(values, 4)
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4, name="flow")
            m.colorbar()
            # The colorbar the scene drew for this layer, through the renderer's own record of it (the idiom
            # `test_static_guides.py` uses) rather than matplotlib's private `Axes._colorbar` back-reference.
            ticks = list(m._renderer.drawn["flow"].guides[0].get_ticks())
        assert np.allclose(ticks, expected), (
            f"colorbar ticks {ticks} should sit on the class edges {expected}"
        )

    def test_the_legend_is_a_class_key_in_the_drawn_colours(self, dataset, values):
        """``legend()`` on a classified raster lists each class's range in the colour that class was drawn.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            ``k=4`` equal intervals give four classes. The key must label each with its own range — built here
            from the independently computed edges — and paint each swatch the colour the image paints a value
            inside that class, so the key cannot disagree with the picture.
        """
        from matplotlib.colors import to_hex

        edges = _equal_interval_edges(values, 4)
        ranges = list(zip(edges, edges[1:]))
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            m.legend()
            legend = m.ax.get_legend()
            labels = [text.get_text() for text in legend.get_texts()]
            swatches = [
                to_hex(handle.get_facecolor()) for handle in legend.legend_handles
            ]
            image = m.ax.images[-1]
            drawn = [
                to_hex(image.cmap(image.norm((low + high) / 2))) for low, high in ranges
            ]
        assert labels == [f"{low} – {high}" for low, high in ranges], (
            f"legend labels {labels} should be the class ranges {ranges}"
        )
        assert swatches == drawn, (
            f"swatches {swatches} should be the drawn class colours {drawn}"
        )

    def test_cells_outside_explicit_edges_take_the_end_class_colours(self, dataset):
        """Explicit edges narrower than the data paint the cells outside them as the first and last classes.

        Args:
            dataset: The raster fixture (values ``0``–``88``).

        Test scenario:
            ``scheme=[10, 50, 70]`` leaves ``0`` below the first edge and ``88`` above the last. matplotlib paints
            them the colormap's end colours, which for two classes are the first and last class colours — so
            they read as in a class — while the key lists only the two ranges. Pinned so the documented
            behaviour cannot drift unnoticed.
        """
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme=[10.0, 50.0, 70.0])
            image = m.ax.images[-1]
            first, last = _painted(image, 30.0), _painted(image, 60.0)
            below, above = _painted(image, 0.0), _painted(image, 88.0)
            m.legend()
            labels = [text.get_text() for text in m.ax.get_legend().get_texts()]
        assert below == first, (
            f"a cell below the first edge is painted {below}, the first class is {first}"
        )
        assert above == last, (
            f"a cell above the last edge is painted {above}, the last class is {last}"
        )
        assert labels == ["10.0 – 50.0", "50.0 – 70.0"], (
            f"the key should list only the two ranges, got {labels}"
        )

    def test_quantiles_on_tied_values_collapse_duplicate_edges(self, dataset, values):
        """Quantile cuts that land on the same value merge, so a skewed raster gets fewer than ``k`` classes.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values (heavily tied at zero).

        Test scenario:
            ``scheme="quantiles", k=5`` cuts at the 0/20/40/60/80/100th percentiles; on this raster the first
            three coincide, so the drawn edges are the *distinct* cut points — not five classes with empty ones.
        """
        expected = list(np.unique(np.quantile(values, np.linspace(0.0, 1.0, 6))))
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="quantiles", k=5)
            boundaries = list(_drawn_norm(m).boundaries)
        assert len(expected) < 6, (
            f"the fixture should have tied quantiles, got edges {expected}"
        )
        assert np.allclose(boundaries, expected), (
            f"quantile edges {boundaries} should be the distinct cut points {expected}"
        )

    def test_a_single_class_spans_the_data_range(self, dataset, values):
        """``k=1`` gives one class from the data minimum to its maximum.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            The smallest classification: two edges, the data's min and max.
        """
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=1)
            boundaries = list(_drawn_norm(m).boundaries)
        assert np.allclose(boundaries, [values.min(), values.max()]), (
            f"one class should span [{values.min()}, {values.max()}], got {boundaries}"
        )

    def test_nodata_cells_stay_masked_when_classified(self, dataset):
        """Classifying a raster leaves its nodata cells masked — they fall in no class.

        Args:
            dataset: The raster fixture, which carries nodata cells.

        Test scenario:
            The drawn array must mask exactly the cells the source marks as nodata, so they are left blank
            rather than painted with the lowest class colour.
        """
        expected = int(np.ma.count_masked(dataset.read_array(masked=True)))
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            drawn = int(np.ma.count_masked(m.ax.images[-1].get_array()))
        assert expected > 0, (
            "the fixture should carry nodata cells for this test to mean anything"
        )
        assert drawn == expected, (
            f"{expected} nodata cells should stay masked, {drawn} were"
        )

    def test_an_unknown_scheme_is_refused_by_name(self, dataset):
        """An unrecognised ``scheme`` raises ``ValueError`` naming it.

        Args:
            dataset: The raster fixture.

        Test scenario:
            A typo must not silently fall back to a continuous ramp.
        """
        with Map(crs=dataset.epsg) as m:
            with pytest.raises(ValueError, match="nope"):
                m.field(dataset, scheme="nope", k=4)


#: A nominal raster: integer class codes 1, 2, 3 and 5 (no 4, so the edges cannot be a plain ``0.5`` grid).
CODES = np.array([[1.0, 1.0, 2.0], [3.0, 3.0, 2.0], [5.0, 5.0, 1.0]])


def _palette(codes, cmap=None) -> list:
    """Return the shared categorical palette for ``codes`` — the colours the vector layers would use."""
    return categorical_colors(codes, resolve_categorical_cmap(cmap))[1]


def _cell_colors(image) -> np.ndarray:
    """Return the hex colour each cell of a drawn image was painted, as an array shaped like the image."""
    from matplotlib.colors import to_hex

    rgba = image.to_rgba(image.get_array())
    return np.array([[to_hex(px, keep_alpha=False) for px in row] for row in rgba])


class TestCategoricalRasterField:
    """``Map.field(..., scheme="categorical")`` draws one class per integer code, with a per-code legend."""

    def test_each_code_is_painted_its_palette_colour(self):
        """Every cell takes the shared categorical colour of its own code.

        Test scenario:
            Codes 1, 2, 3, 5 get the palette's first four colours in sorted order — the same assignment a
            categorical vector column would get — and each cell is painted the colour of its code.
        """
        palette = dict(zip([1, 2, 3, 5], _palette([1, 2, 3, 5])))
        with Map() as m:
            m.field(CODES, scheme="categorical")
            painted = _cell_colors(m.ax.images[-1])
        expected = np.vectorize(lambda code: palette[int(code)])(CODES)
        assert (painted == expected).all(), (
            f"cells painted {painted.tolist()}, expected {expected.tolist()}"
        )

    def test_a_caller_cmap_is_the_palette(self):
        """A ``cmap`` passed with the categorical scheme supplies the class colours.

        Test scenario:
            ``cmap="Set2"`` must colour the codes from Set2, exactly as ``categorical_colors`` samples it.
        """
        with Map() as m:
            m.field(CODES, scheme="categorical", cmap="Set2")
            painted = set(_cell_colors(m.ax.images[-1]).ravel())
        expected = set(_palette([1, 2, 3, 5], "Set2"))
        assert painted == expected, (
            f"painted {sorted(painted)}, expected Set2's {sorted(expected)}"
        )

    def test_the_legend_lists_each_code_with_its_colour(self):
        """``legend()`` draws one swatch per code, labelled with the code, in the code's colour.

        Test scenario:
            The key must name the categories themselves (``1``, ``2``, ``3``, ``5``), not value ranges, and
            each swatch must be the colour its code was painted.
        """
        with Map() as m:
            m.field(CODES, scheme="categorical")
            m.legend()
            legend = m.ax.get_legend()
            labels = [text.get_text() for text in legend.get_texts()]
            from matplotlib.colors import to_hex

            colors = [
                to_hex(handle.get_facecolor()) for handle in legend.legend_handles
            ]
        assert labels == ["1", "2", "3", "5"], (
            f"legend labels {labels} should be the codes"
        )
        assert colors == _palette([1, 2, 3, 5]), (
            f"legend colours {colors} should be the palette"
        )

    def test_categories_are_published_not_class_edges(self):
        """The layer's colour encoding is categorical, and ``last_breaks`` stays empty.

        Test scenario:
            A categorical layer has categories, not graduated class edges — the same answer a categorical
            choropleth gives — so other tiers reading the figure see the codes.
        """
        with Map() as m:
            m.field(CODES, scheme="categorical", name="codes")
            scale = m.figure_spec.layers.get("codes").symbology.encoding("color").scale
            breaks = m.last_breaks
        assert scale.is_categorical, (
            f"the colour scale should be categorical, got {scale!r}"
        )
        assert list(scale.categories) == [1, 2, 3, 5], (
            f"categories {list(scale.categories)} should be the codes"
        )
        assert breaks is None, (
            f"a categorical raster has no class edges to publish, got {breaks}"
        )

    def test_a_colorbar_is_refused_in_favour_of_the_legend(self):
        """A colorbar over categorical codes is refused, naming ``legend()``.

        Test scenario:
            A bar would read the codes as a magnitude; the categorical choropleth refuses it the same way.
        """
        with Map() as m:
            m.field(CODES, scheme="categorical")
            with pytest.raises(ValueError, match="legend"):
                m.colorbar()

    def test_nodata_cells_stay_blank(self):
        """A NaN cell falls in no category and is left transparent.

        Test scenario:
            The missing cell must not become a category of its own nor take a class colour.
        """
        codes = CODES.copy()
        codes[0, 0] = np.nan
        with Map() as m:
            m.field(codes, scheme="categorical", name="codes")
            image = m.ax.images[-1]
            alpha = image.to_rgba(image.get_array())[0, 0, 3]
            categories = list(
                m.figure_spec.layers.get("codes")
                .symbology.encoding("color")
                .scale.categories
            )
        assert alpha == 0, f"the nodata cell should be transparent, alpha={alpha}"
        assert categories == [1, 2, 3, 5], (
            f"nodata must not become a category, got {categories}"
        )

    def test_non_integer_values_are_refused(self, dataset):
        """A band of non-integer values is refused: it is a magnitude, not class codes.

        Args:
            dataset: Unused; kept so the fixture set matches the class.

        Test scenario:
            ``0.5`` is not a class code, so the categorical scheme must refuse and point at a graduated one.
        """
        with Map() as m:
            with pytest.raises(ValueError, match="integer"):
                m.field(np.array([[0.5, 1.0], [2.0, 3.0]]), scheme="categorical")

    def test_too_many_codes_are_refused(self):
        """A band with more distinct codes than a legend can carry is refused, naming the count.

        Test scenario:
            100 distinct integer codes is a continuous field in disguise; the refusal must say how many.
        """
        with Map() as m:
            with pytest.raises(ValueError, match="100"):
                m.field(np.arange(100.0).reshape(10, 10), scheme="categorical")
