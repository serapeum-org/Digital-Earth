"""``WebMap.field(..., scheme=, k=)`` — a raster coloured class by class on the web tier.

Expected values are computed independently of the classifier: codes by hand, edges with ``numpy``, colours
from the shared ``categorical_colors``/``sample_cmap``. Cross-tier agreement is pinned in
``tests/test_cross_tier_raster_classes.py``.
"""

import base64
import io

import numpy as np
import pytest

pytest.importorskip("maplibre", reason="web tier engine")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.base.symbology import (  # noqa: E402
    categorical_colors,
    resolve_categorical_cmap,
    sample_cmap,
)
from digitalearth.web import TileRoute, WebMap  # noqa: E402

CODES = np.array([[1.0, 2.0, 5.0], [5.0, 1.0, 2.0]])


def _band(values: np.ndarray):
    """Wrap ``values`` as a lon/lat source the web tier draws without warping."""
    rows, columns = values.shape
    return get_source(
        values,
        x=np.linspace(10.0, 10.0 + 0.1 * (columns - 1), columns),
        y=np.linspace(50.0, 50.0 - 0.1 * (rows - 1), rows),
        crs=4326,
    )


def _cells(web_map: WebMap, layer_id: str) -> np.ndarray:
    """Return the ``#rrggbb`` colour of each pixel the layer's inline PNG holds, and ``None`` for clear ones."""
    from PIL import Image

    url = web_map._renderer.drawn[layer_id].source_spec["url"]
    png = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA")
    )
    return np.array(
        [
            ["#{:02x}{:02x}{:02x}".format(*px[:3]) if px[3] else None for px in row]
            for row in png.tolist()
        ],
        dtype=object,
    )


class TestClassifiedWebField:
    """A ``scheme`` colours the band class by class, records the classes, and keys them."""

    def test_codes_are_painted_and_keyed_in_the_shared_palette(self):
        """Each code is its own class, in the categorical palette, and ``last_breaks`` names the codes.

        Test scenario:
            Codes 1, 2, 5: the key has three swatches in the palette's order, the band's pixels take their
            code's colour, and ``last_breaks`` is the codes, as a categorical choropleth publishes them.
        """
        palette = categorical_colors([1, 2, 5], resolve_categorical_cmap(None))[1]
        lookup = dict(zip([1, 2, 5], palette))
        web_map = WebMap().field(_band(CODES), scheme="categorical", name="cover")
        cells = _cells(web_map, "cover")
        expected = np.vectorize(lambda code: lookup[int(code)])(CODES)
        assert (cells == expected).all(), (
            f"pixels {cells.tolist()}, expected {expected.tolist()}"
        )
        assert web_map.last_legend["colors"] == list(palette), (
            f"key {web_map.last_legend['colors']}"
        )
        assert web_map.last_breaks == [1, 2, 5], f"last_breaks {web_map.last_breaks}"

    def test_graduated_classes_publish_their_edges(self):
        """Equal-interval classes are keyed by their edges and coloured by ``cmap``'s samples.

        Test scenario:
            0..9 cut into three equal intervals: edges at 0, 3, 6, 9, three viridis samples.
        """
        values = np.arange(9.0).reshape(3, 3) * 9 / 8
        web_map = WebMap().field(
            _band(values), scheme="equal_interval", k=3, cmap="viridis", name="ramp"
        )
        expected_edges = list(np.linspace(values.min(), values.max(), 4))
        assert web_map.last_breaks == pytest.approx(expected_edges), (
            f"edges {web_map.last_breaks}"
        )
        assert web_map.last_legend["colors"] == sample_cmap("viridis", 3), (
            f"key {web_map.last_legend}"
        )
        scale = web_map.get_layer("ramp").symbology.encoding("color").scale
        assert list(scale.breaks) == pytest.approx(expected_edges), (
            f"recorded breaks {scale.breaks}"
        )

    def test_nodata_stays_transparent(self):
        """A NaN cell is left clear, not painted a class colour.

        Test scenario:
            One NaN among the codes; its pixel's alpha is zero and it is no category.
        """
        codes = CODES.copy()
        codes[0, 0] = np.nan
        web_map = WebMap().field(_band(codes), scheme="categorical", name="cover")
        assert _cells(web_map, "cover")[0, 0] is None, (
            "the nodata pixel should be transparent"
        )
        scale = web_map.get_layer("cover").symbology.encoding("color").scale
        assert list(scale.categories) == [1, 2, 5], f"categories {scale.categories}"

    @pytest.mark.parametrize("tiles", ["xyz", "cog"])
    def test_a_tiled_route_refuses_a_scheme(self, tmp_path, tiles):
        """``scheme`` with ``tiles`` is refused by name rather than drawn from averaged overviews.

        Args:
            tmp_path: Where a tiled route would write.
            tiles: The route.
        """
        band = _band(CODES)
        destination = tmp_path / "tiles"
        web_map = WebMap()
        route = TileRoute(tiles, destination)
        with pytest.raises(ValueError, match="inline route"):
            web_map.field(band, scheme="categorical", tiles=route)
        assert web_map.layer_ids == [], (
            f"a refused layer was added: {web_map.layer_ids}"
        )
        assert not destination.exists(), "a refused tiled field should write nothing"

    @pytest.mark.parametrize("scheme", ["categorical", "equal_interval"])
    def test_units_reach_a_classified_key(self, scheme):
        """``units=`` lands on the key of a classified band, as it does on a ramp's.

        Args:
            scheme: A categorical or a graduated scheme.

        Test scenario:
            ``units="class"`` is recorded as ``last_units`` and copied onto the key the builder filed.
        """
        web_map = WebMap().field(
            _band(CODES), scheme=scheme, units="class", name="cover"
        )
        assert web_map.last_legend.get("units") == "class", (
            f"key {web_map.last_legend} should carry the band's units"
        )

    def test_a_ramp_key_is_sampled_at_evenly_spaced_stops(self):
        """Without a scheme the key is a ramp: stops evenly spaced over the limits, coloured from ``cmap``.

        Test scenario:
            ``limits=(0, 8)`` with viridis: five stops ``0, 2, 4, 6, 8`` and five viridis samples, the
            stops published as ``last_breaks``.
        """
        web_map = WebMap().field(
            _band(CODES), cmap="viridis", limits=(0.0, 8.0), name="ramp"
        )
        expected_stops = list(np.linspace(0.0, 8.0, 5))
        assert web_map.last_breaks == pytest.approx(expected_stops), (
            f"stops {web_map.last_breaks}, expected {expected_stops}"
        )
        assert web_map.last_legend["colors"] == sample_cmap("viridis", 5), (
            f"ramp colours {web_map.last_legend['colors']}"
        )


class TestBandColour:
    """``RasterMixin._band_colour`` — the colour encoding a band publishes, and the key it files."""

    def test_neither_classes_nor_limits_publishes_nothing(self):
        """A band with no classes and no limits has no scale to publish, so the mapping is empty.

        Test scenario:
            ``limits=None``, ``classes=None``: ``{}`` comes back and no key is filed.
        """
        web_map = WebMap()
        encodings = web_map._band_colour(
            _band(CODES), band=1, cmap="viridis", limits=None
        )
        assert encodings == {}, f"expected no encoding, got {encodings}"
        assert web_map.last_legend is None, (
            f"no key should be filed, got {web_map.last_legend}"
        )

    def test_classes_take_precedence_over_limits(self):
        """When a band is classified its classes' scale is published, whatever limits are passed.

        Test scenario:
            Categorical classes of codes 1, 2, 5 alongside limits ``(0, 100)``: the encoding's scale is
            categorical over the codes, not a ``0..100`` span.
        """
        from digitalearth.base.raster_classes import classify_band

        classes = classify_band(CODES, "categorical", None, None)
        encodings = WebMap()._band_colour(
            _band(CODES), band=1, cmap=None, limits=(0.0, 100.0), classes=classes
        )
        scale = encodings["color"].scale
        assert list(scale.categories) == [1, 2, 5], (
            f"categories {scale.categories} should be the codes"
        )


class TestBandKey:
    """``RasterMixin._band_key`` — one key body for a ramp, graduated classes and categorical codes."""

    def test_categorical_key_values_are_the_codes(self):
        """A categorical band is keyed one swatch per code, valued at the codes.

        Test scenario:
            Codes 1, 2, 5: ``kind`` categorical, ``values`` the codes, colours the shared palette.
        """
        from digitalearth.base.raster_classes import classify_band

        classes = classify_band(CODES, "categorical", None, None)
        key = WebMap()._band_key("cover", classes.scale, None, classes)
        palette = list(categorical_colors([1, 2, 5], resolve_categorical_cmap(None))[1])
        assert (key["kind"], list(key["values"])) == ("categorical", [1, 2, 5]), (
            f"key {key}"
        )
        assert list(key["colors"]) == palette, (
            f"colours {key['colors']}, expected {palette}"
        )

    def test_graduated_key_values_are_the_edges(self):
        """A graduated band is keyed one swatch per class, valued at its edges.

        Test scenario:
            ``0..12`` in four equal intervals of magma: edges every 3, four magma samples.
        """
        from digitalearth.base.raster_classes import classify_band

        values = np.arange(13.0)
        classes = classify_band(values, "equal_interval", 4, "magma")
        key = WebMap()._band_key("depth", classes.scale, "magma", classes)
        assert list(key["values"]) == pytest.approx([0.0, 3.0, 6.0, 9.0, 12.0]), (
            f"values {key['values']}"
        )
        assert list(key["colors"]) == sample_cmap("magma", 4), (
            f"colours {key['colors']}"
        )
        assert key["kind"] == "graduated", f"kind {key['kind']}"

    def test_ramp_key_without_classes(self):
        """With no classes the key is continuous, sampled at five evenly spaced stops.

        Test scenario:
            A ``-2..2`` span: stops ``-2, -1, 0, 1, 2`` coloured by five RdBu samples.
        """
        from digitalearth.base.spec import Scale

        key = WebMap()._band_key("anomaly", Scale.from_limits(-2.0, 2.0), "RdBu", None)
        assert key["kind"] == "continuous", f"kind {key['kind']}"
        assert list(key["values"]) == pytest.approx([-2.0, -1.0, 0.0, 1.0, 2.0]), (
            f"values {key['values']}"
        )
        assert list(key["colors"]) == sample_cmap("RdBu", 5), f"colours {key['colors']}"


class TestClassedPng:
    """``_classed_png`` and ``WebMap._rgba_png_datauri(classes=)`` — class-by-class encoding."""

    def test_all_nodata_band_gives_none(self):
        """A band with no valid cell has nothing to colour, so no PNG is produced.

        Test scenario:
            Every cell NaN: ``_classed_png`` answers ``None`` rather than an all-clear image.
        """
        from digitalearth.base.raster_classes import classify_band
        from digitalearth.web.raster import _classed_png

        classes = classify_band(CODES, "categorical", None, None)
        assert _classed_png(np.full((2, 2), np.nan), classes) is None, (
            "an all-nodata band should produce no PNG"
        )

    def test_datauri_refuses_an_all_nodata_classified_band(self):
        """The data-URI wrapper refuses the ``None`` the classed encoder gives for an empty band.

        Test scenario:
            Categorical classes, an all-NaN band: ``ValueError`` naming the empty band.
        """
        from digitalearth.base.raster_classes import classify_band

        classes = classify_band(CODES, "categorical", None, None)
        empty = np.full((1, 2), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            WebMap._rgba_png_datauri(empty, "viridis", classes=classes)

    def test_datauri_with_classes_ignores_cmap(self):
        """With classes the colours are the classes' own; an unknown ``cmap`` name is never looked up.

        Test scenario:
            ``cmap="no-such-cmap"`` with categorical classes still encodes; each pixel is its code's colour.
        """
        from PIL import Image

        from digitalearth.base.raster_classes import classify_band

        classes = classify_band(CODES, "categorical", None, None)
        url = WebMap._rgba_png_datauri(CODES, "no-such-cmap", classes=classes)
        png = np.asarray(
            Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert(
                "RGBA"
            )
        )
        lookup = dict(
            zip(
                [1, 2, 5],
                categorical_colors([1, 2, 5], resolve_categorical_cmap(None))[1],
            )
        )
        expected = np.vectorize(lambda code: lookup[int(code)])(CODES)
        painted = np.array(
            [
                ["#{:02x}{:02x}{:02x}".format(*px[:3]) for px in row]
                for row in png.tolist()
            ]
        )
        assert (painted == expected).all(), (
            f"pixels {painted.tolist()}, expected {expected.tolist()}"
        )


class TestClassifiedReplay:
    """``draw_field`` files a key for a classified layer drawn from a figure rather than built here."""

    @pytest.mark.parametrize("scheme", ["categorical", "equal_interval"])
    def test_replayed_figure_files_the_builders_key(self, tmp_path, scheme):
        """A web figure read back and drawn again keys the band exactly as the builder did.

        Args:
            tmp_path: Where the raster is written, so the figure records a path it can reopen.
            scheme: A categorical or a graduated scheme.

        Test scenario:
            ``draw_field`` finds no key filed for the replayed layer and files one from its colour
            encoding; its kind, values and colours equal those the builder filed.
        """
        from pyramids.dataset import Dataset, GeoReference

        from digitalearth import to_backend
        from digitalearth.base.spec import FigureSpec

        path = str(tmp_path / "codes.tif")
        geo = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.1, epsg=4326)
        Dataset.from_array(CODES.astype(np.int32), geo_ref=geo).to_file(path)
        built = WebMap().field(path, scheme=scheme, k=3, cmap="viridis", name="cover")
        original = built._legends["cover"]
        spec = FigureSpec.from_dict(built.figure_spec.to_dict())
        replayed = to_backend(spec, backend="web")._legends["cover"]
        assert replayed["kind"] == original["kind"], (
            f"kind: replayed {replayed['kind']}, built {original['kind']}"
        )
        assert list(replayed["values"]) == pytest.approx(list(original["values"])), (
            f"values: replayed {replayed['values']}, built {original['values']}"
        )
        assert list(replayed["colors"]) == list(original["colors"]), (
            f"colours: replayed {replayed['colors']}, built {original['colors']}"
        )
