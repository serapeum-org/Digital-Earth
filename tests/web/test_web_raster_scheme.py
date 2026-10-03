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

from digitalearth.base.symbology import (  # noqa: E402
    categorical_colors,
    resolve_categorical_cmap,
    sample_cmap,
)
from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.web import WebMap  # noqa: E402

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
            [("#%02x%02x%02x" % tuple(px[:3])) if px[3] else None for px in row]
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
        with pytest.raises(ValueError, match="inline route"):
            web_map.field(
                band, scheme="categorical", tiles=tiles, tiles_path=destination
            )
        assert web_map.layer_ids == [], (
            f"a refused layer was added: {web_map.layer_ids}"
        )
        assert not destination.exists(), "a refused tiled field should write nothing"
