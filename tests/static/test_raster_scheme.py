"""``scheme=``/``k=`` on a raster field: the classes are cut, drawn, published and keyed (ST-7, #222).

A classified raster used to be impossible — cleopatra's ``ArrayGlyph.plot`` took no ``classify`` group, so
``render_compat`` refused ``scheme``/``k`` on every raster layer. On the pinned cleopatra (``>=0.40``) it
takes one: ``field`` forwards the flat keys, ``render_compat`` folds them into the ``classify`` group, and the
scene reads the class edges back off the drawn norm. Nothing pinned that path, though, so a renamed upstream
keyword would have turned every classified raster back into a refusal with no failing test. A categorical
raster — integer class codes — is the one case cleopatra still refuses; the static tier draws it as explicit
edges, one class per code, which :class:`TestCategoricalRasterField` holds.

These tests hold each link of the chain to edges computed **independently from the data** — never read back
from the figure under test — so they cannot agree with the drawing by construction.
"""

import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm

from digitalearth.base.symbology import categorical_colors, resolve_categorical_cmap
from digitalearth.static import Map
from digitalearth.static.maps.raster import _raster_categories


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

    @pytest.mark.parametrize(
        "given",
        [[0.0, 10.0, 50.0, 88.0], [50.0, 0.0, 10.0, 88.0, 10.0]],
        ids=["sorted", "unsorted-with-a-duplicate"],
    )
    def test_explicit_edges_are_the_classes_sorted_and_deduplicated(
        self, dataset, given
    ):
        """A list passed as ``scheme`` is the class edges — sorted and de-duplicated, never re-cut.

        Args:
            dataset: The raster fixture.
            given: The edges as the caller wrote them.

        Test scenario:
            ``[0, 10, 50, 88]`` draws exactly those boundaries; ``[50, 0, 10, 88, 10]`` draws the same, because
            the edges are normalised (sorted, duplicates dropped) rather than taken in the order written.
        """
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme=given)
            boundaries = list(_drawn_norm(m).boundaries)
        assert boundaries == sorted(set(given)), (
            f"explicit edges {given} were drawn as {boundaries}"
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

    @pytest.mark.parametrize("builder", ["field", "pcolormesh", "block"])
    def test_every_raster_builder_classifies(self, dataset, values, builder):
        """``field``, ``pcolormesh`` and ``block`` all cut ``scheme``/``k`` into the same classes.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.
            builder: The raster builder under test.

        Test scenario:
            The three share the classify path, so a renamed upstream keyword would break all three; each must
            hand back a mappable with a ``BoundaryNorm`` on the equal-interval edges.
        """
        expected = _equal_interval_edges(values, 4)
        with Map(crs=dataset.epsg) as m:
            mappable = getattr(m, builder)(dataset, scheme="equal_interval", k=4)
            norm = mappable.norm
        assert isinstance(norm, BoundaryNorm), (
            f"{builder} should classify, got {norm!r}"
        )
        assert np.allclose(norm.boundaries, expected), (
            f"{builder} edges {list(norm.boundaries)} != {expected}"
        )

    def test_fisher_jenks_cuts_at_data_values_within_the_range(self, dataset, values):
        """``scheme="fisher_jenks"`` cuts the raster into at most ``k`` classes at values the data holds.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            Natural breaks have no closed form to compute independently, so the test holds the properties that
            define them: the outer edges are the data's minimum and maximum, the inner ones are values present
            in the data, the edges ascend, and there are at most ``k + 1`` of them.
        """
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="fisher_jenks", k=3)
            edges = [float(edge) for edge in _drawn_norm(m).boundaries]
        present = set(values.tolist())
        assert edges[0] == values.min(), (
            f"the first edge of {edges} should be the data minimum {values.min()}"
        )
        assert edges[-1] == values.max(), (
            f"the last edge of {edges} should be the data maximum {values.max()}"
        )
        assert all(edge in present for edge in edges[1:-1]), (
            f"inner edges {edges[1:-1]} should be data values"
        )
        assert edges == sorted(set(edges)), f"edges {edges} should strictly ascend"
        assert len(edges) <= 4, f"k=3 gives at most four edges, got {edges}"

    def test_k_without_a_scheme_is_ignored(self, dataset):
        """``k=`` alone leaves the field a continuous ramp: it counts a named scheme's classes, so needs one.

        Args:
            dataset: The raster fixture.

        Test scenario:
            ``k=4`` with no ``scheme`` must not classify — no ``BoundaryNorm``, no published breaks — matching
            the documented contract rather than quietly defaulting to some scheme.
        """
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, k=4)
            norm = _drawn_norm(m)
            breaks = m.last_breaks
        assert not isinstance(norm, BoundaryNorm), (
            f"k alone should not classify, got {norm!r}"
        )
        assert breaks is None, f"k alone should publish no class edges, got {breaks}"

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
            The drawn array must mask exactly the cells the source marks as nodata, *on a classified field*
            (a ``BoundaryNorm`` is asserted, so the test cannot pass with classification dropped), and the
            colour a masked cell is painted must be fully transparent — which is what "left blank" means.
        """
        expected = int(np.ma.count_masked(dataset.read_array(masked=True)))
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            image = m.ax.images[-1]
            drawn = int(np.ma.count_masked(image.get_array()))
            norm, bad_alpha = image.norm, image.cmap.get_bad()[3]
        assert expected > 0, (
            "the fixture should carry nodata cells for this test to mean anything"
        )
        assert isinstance(norm, BoundaryNorm), (
            f"the field should be classified, got {norm!r}"
        )
        assert drawn == expected, (
            f"{expected} nodata cells should stay masked, {drawn} were"
        )
        assert bad_alpha == 0, (
            f"a masked cell should be transparent, its alpha is {bad_alpha}"
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

    def test_non_integer_values_are_refused(self):
        """A band of non-integer values is refused: it is a magnitude, not class codes.

        Test scenario:
            ``0.5`` is not a class code, so the categorical scheme must refuse and point at a graduated one.
        """
        fractional = np.array([[0.5, 1.0], [2.0, 3.0]])
        with Map() as m:
            with pytest.raises(ValueError, match="integer"):
                m.field(fractional, scheme="categorical")

    def test_too_many_codes_are_refused(self):
        """A band with more distinct codes than a legend can carry is refused, naming the count.

        Test scenario:
            100 distinct integer codes is a continuous field in disguise; the refusal must say how many.
        """
        hundred_codes = np.arange(100.0).reshape(10, 10)
        with Map() as m:
            with pytest.raises(ValueError, match="100"):
                m.field(hundred_codes, scheme="categorical")

    def test_an_all_nodata_band_is_refused(self):
        """A band with no valid cell has no codes to classify, and the refusal says so.

        Test scenario:
            Every cell NaN leaves nothing to make a category of; drawing an empty key silently would hide
            that the band carries no data.
        """
        empty = np.full((2, 3), np.nan)
        with Map() as m:
            with pytest.raises(ValueError, match="nodata"):
                m.field(empty, scheme="categorical")

    def test_a_raster_far_past_the_canvas_keeps_its_true_codes(self):
        """A categorical raster big enough to be read at a reduced resolution still classifies its real codes.

        Test scenario:
            A 1024x1024 raster of codes ``{1, 2, 7, 9}`` on a 2-inch figure is far past the canvas, so a graduated
            field would be read decimated — which combines neighbouring cells. Codes are nominal, so combining
            them invents codes (``3``, ``4``, … appeared before) and loses real ones; the categories must be the
            band's own codes.
        """
        from pyramids.dataset import Dataset, GeoReference

        rng = np.random.default_rng(7)
        codes = rng.choice(np.array([1, 2, 7, 9], dtype=np.int32), size=(1024, 1024))
        geo = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.001, epsg=4326)
        large = Dataset.from_array(codes, geo_ref=geo)
        with Map(crs=4326, figsize=(2, 2)) as m:
            m.field(large, scheme="categorical", name="cover")
            categories = list(
                m.figure_spec.layers.get("cover")
                .symbology.encoding("color")
                .scale.categories
            )
        assert categories == [1, 2, 7, 9], (
            f"the categories {categories} should be the band's codes [1, 2, 7, 9]"
        )

    def test_a_replayed_figure_paints_the_colours_its_key_shows(self, tmp_path):
        """A categorical raster drawn with an unregistered colormap replays in the same colours it keys.

        Args:
            tmp_path: Where the path-backed raster is written, so the figure can be stored.

        Test scenario:
            A ``Colormap`` object cannot be written into a figure description, so a replay through
            ``to_dict``/``from_dict``/``to_backend`` used to fall back to the default palette while the stored
            colour encoding still carried the original colours — the key disagreed with the picture. The replay
            must paint, and key, the colours the figure was drawn with.
        """
        from matplotlib.colors import ListedColormap, to_hex
        from pyramids.dataset import Dataset, GeoReference

        from digitalearth import to_backend
        from digitalearth.base.spec import FigureSpec

        path = str(tmp_path / "codes.tif")
        geo = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.1, epsg=4326)
        Dataset.from_array(CODES.astype(np.int32), geo_ref=geo).to_file(path)
        custom = ["#111111", "#222222", "#333333", "#444444"]
        with Map(crs=4326) as m:
            m.field(
                path, scheme="categorical", cmap=ListedColormap(custom), name="cover"
            )
            stored = FigureSpec.from_dict(m.figure_spec.to_dict())
        replay = to_backend(stored, backend="matplotlib")
        try:
            painted = sorted(set(_cell_colors(replay.ax.images[-1]).ravel()))
            replay.legend()
            keyed = [
                to_hex(h.get_facecolor()) for h in replay.ax.get_legend().legend_handles
            ]
        finally:
            replay.close()
        assert painted == custom, (
            f"the replay painted {painted}, the figure was drawn in {custom}"
        )
        assert keyed == custom, (
            f"the replay keyed {keyed}, the figure was drawn in {custom}"
        )

    @pytest.mark.parametrize("filled", [True, False], ids=["filled", "lines"])
    def test_contours_refuse_the_categorical_scheme(self, filled):
        """``contours(..., scheme="categorical")`` is refused: a contour interpolates between codes.

        Args:
            filled: Whether the contours are filled bands or lines.

        Test scenario:
            Codes are labels, not a surface — filling between ``1`` and ``5`` paints bands of ``2`` and ``3``
            where no cell holds them, and lines sit on half-codes. The refusal must name the cell-based renders.
        """
        from pyramids.dataset import Dataset, GeoReference

        geo = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.1, epsg=4326)
        codes = Dataset.from_array(CODES.astype(np.int32), geo_ref=geo)
        with Map(crs=4326) as m:
            with pytest.raises(ValueError, match="pcolormesh"):
                m.contours(codes, filled=filled, scheme="categorical")

    @pytest.mark.parametrize(
        "base", [2**52, 2**60, -(2**60)], ids=["2^52", "2^60", "-2^60"]
    )
    def test_codes_too_large_for_exact_edges_are_refused(self, base):
        """Codes a float cannot hold half a step apart are refused rather than merged.

        Args:
            base: The first of two adjacent codes.

        Test scenario:
            Classes are bounded half a step either side of each code, in floats. From ``2**52`` on a float no
            longer resolves ``0.5``, and from ``2**53`` it no longer resolves ``1`` — ``2**60`` and ``2**60 + 1``
            came back as one code, and two codes at ``2**52`` shared an edge. The band must be refused instead.
        """
        values = np.array([base, base + 1], dtype=np.int64)
        with pytest.raises(ValueError, match=r"2\*\*52"):
            _raster_categories(values)
