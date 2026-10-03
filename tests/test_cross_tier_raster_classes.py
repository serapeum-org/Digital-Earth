"""A classified raster drawn on one 2-D tier draws, and keys, the same classes on the other two.

A figure records a classified band's classes in the layer's colour encoding. Before, only the static tier read
them: the interactive tier handed the static tier's ``scheme``/``k`` to HoloViews, which refused the element
(``Unexpected option 'scheme' for Image``), and the web tier drew a continuous ramp and filed no key, so
``legend()`` found nothing to describe — for a continuous band too.

The expected colours are computed here, independently: each cell's class from ``numpy.digitize`` on the
recorded edges, each class's colour from ``categorical_colors``/``sample_cmap`` — so a tier cannot pass by
agreeing with the shared classifier it is built on.
"""

import base64
import io

import numpy as np
import pytest

pytest.importorskip("maplibre", reason="web tier engine")
pytest.importorskip("geoviews", reason="interactive tier engine")

from matplotlib.colors import to_hex  # noqa: E402
from pyramids.dataset import Dataset, GeoReference  # noqa: E402

from digitalearth import to_backend  # noqa: E402
from digitalearth.base.spec import FigureSpec  # noqa: E402
from digitalearth.base.symbology import (  # noqa: E402
    categorical_colors,
    resolve_categorical_cmap,
    sample_cmap,
)

#: A nominal band: codes 1, 2, 3 and 5 (a gap, so the classes are not a plain unit grid).
CODES = np.array([[1, 2, 3], [5, 1, 2], [3, 3, 5]], dtype=np.int32)

#: A magnitude, cut into four quantile classes coloured from a short qualitative map.
RAMP = np.arange(25, dtype=np.float32).reshape(5, 5)

CASES = {
    "categorical": (CODES, {}),
    "quantiles": (RAMP, {"k": 4, "cmap": "Set2"}),
}

TIERS = ("matplotlib", "interactive", "web")


def _build(tier: str, path: str, scheme: str, options: dict):
    """Draw ``path`` classified on ``tier``, in EPSG:4326 so every tier classifies the same, unwarped cells."""
    if tier == "matplotlib":
        from digitalearth.static import Map

        scene = Map(crs=4326)
    elif tier == "interactive":
        from digitalearth.interactive import InteractiveMap

        scene = InteractiveMap(crs=4326)
    else:
        from digitalearth.web import WebMap

        scene = WebMap()
    scene.field(path, scheme=scheme, name="cover", **options)
    return scene


def _expected(band: np.ndarray, scheme: str, scale) -> tuple:
    """Return each cell's expected colour and each class's, from the recorded scale and the shared palettes."""
    if scheme == "categorical":
        codes = sorted(int(code) for code in np.unique(band))
        colors = categorical_colors(codes, resolve_categorical_cmap(None))[1]
        lookup = dict(zip(codes, colors))
        return np.vectorize(lambda code: lookup[int(code)])(band), list(colors)
    edges = np.asarray(scale.breaks, dtype=float)
    colors = sample_cmap("Set2", len(edges) - 1)
    index = np.clip(np.digitize(band, edges[1:-1], right=False), 0, len(colors) - 1)
    return np.asarray(colors)[index], list(colors)


def _painted(tier: str, scene, band: np.ndarray) -> tuple:
    """Return the colour ``tier`` painted each cell, and the colours its key shows."""
    if tier == "matplotlib":
        image = scene.ax.images[-1]
        cells = np.array(
            [
                [to_hex(px, keep_alpha=False) for px in row]
                for row in image.to_rgba(image.get_array())
            ]
        )
        scene.legend()
        key = [
            to_hex(handle.get_facecolor())
            for handle in scene.ax.get_legend().legend_handles
        ]
        return cells, key
    if tier == "web":
        from PIL import Image

        url = scene._renderer.drawn["cover"].source_spec["url"]
        png = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert(
            "RGBA"
        )
        cells = np.array(
            [
                ["#{:02x}{:02x}{:02x}".format(*px[:3]) for px in row]
                for row in np.asarray(png).tolist()
            ]
        )
        scene.legend()
        return cells, list(scene._legends["cover"]["colors"])
    import holoviews as hv

    figure = hv.render(scene.layers[0], backend="bokeh")
    mapper = next(
        renderer.glyph.color_mapper
        for renderer in figure.renderers
        if type(getattr(renderer, "glyph", None)).__name__ == "Image"
    )
    palette = list(mapper.palette)
    span = mapper.high - mapper.low

    def colour(value: float) -> str:
        """The palette entry Bokeh's linear mapper paints ``value`` with."""
        slot = int((value - mapper.low) / span * len(palette))
        return palette[min(max(slot, 0), len(palette) - 1)]

    cells = np.vectorize(colour)(band.astype(float))
    # Bokeh draws an image's key as its colour bar, which is the same mapper; one swatch per class is read off
    # it at a cell value inside each class.
    return cells, None


@pytest.mark.parametrize("scheme", sorted(CASES))
@pytest.mark.parametrize(
    "origin, target",
    [(origin, target) for origin in TIERS for target in TIERS if origin != target],
)
def test_a_classified_raster_replays_in_its_classes(tmp_path, scheme, origin, target):
    """A figure classified on ``origin`` paints and keys the same classes when drawn on ``target``.

    Args:
        tmp_path: Where the raster is written, so the figure records a path every tier can open.
        scheme: ``"categorical"`` codes, or quantile classes of a ramp.
        origin: The tier the figure was described on.
        target: The tier it is drawn on.

    Test scenario:
        Every cell takes the colour of its class, and the key (where the tier has swatches) shows each class's
        colour in order — the shared categorical palette for codes, Set2's samples for the quantiles.
    """
    band, options = CASES[scheme]
    path = str(tmp_path / f"{scheme}.tif")
    geo = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.1, epsg=4326)
    Dataset.from_array(band, geo_ref=geo).to_file(path)
    built = _build(origin, path, scheme, options)
    spec = FigureSpec.from_dict(built.figure_spec.to_dict())
    scale = spec.layers.get("cover").symbology.encoding("color").scale
    expected_cells, expected_key = _expected(band, scheme, scale)
    drawn = to_backend(spec, backend=target)
    try:
        cells, key = _painted(target, drawn, band)
    finally:
        if hasattr(drawn, "close"):
            drawn.close()
    assert (cells == expected_cells).all(), (
        f"{origin} -> {target}: cells painted {cells.tolist()}, expected {expected_cells.tolist()}"
    )
    if key is not None:
        assert key == expected_key, (
            f"{origin} -> {target}: key {key}, expected {expected_key}"
        )


@pytest.mark.parametrize("target", ["interactive", "web"])
def test_a_continuous_raster_is_keyed_after_replay(tmp_path, target):
    """A plain ramp drawn on the static tier can be keyed once drawn on the other two.

    Args:
        tmp_path: Where the raster is written.
        target: The tier the figure is drawn on.

    Test scenario:
        The web tier filed a key only from its own builder, so a replayed band had none to show; ``legend()``
        raised "nothing to describe" for a map with a coloured band on it.
    """
    from digitalearth.static import Map

    path = str(tmp_path / "ramp.tif")
    geo = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.1, epsg=4326)
    Dataset.from_array(RAMP, geo_ref=geo).to_file(path)
    with Map(crs=4326) as scene:
        scene.field(path, name="ramp")
        spec = FigureSpec.from_dict(scene.figure_spec.to_dict())
    drawn = to_backend(spec, backend=target)
    keyed = drawn.legend() if target == "web" else drawn.colorbar()
    guide = keyed.get_layer("ramp").symbology.guide()
    assert guide is not None, f"the replayed ramp on {target} should carry a key"
    assert guide.show, f"the replayed ramp's key on {target} should be shown"
