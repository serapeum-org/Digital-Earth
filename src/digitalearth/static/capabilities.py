"""What the matplotlib tier can draw, declared as data (U-1, #294; the static seam's part of #303).

The last tier to say so in its own words. Until now its answer lived in `api.py`, as one hand-written row of
six keyword names — the only row left after the other three tiers declared theirs — and everything beyond
those six was a refusal raised wherever the builder happened to reach it.

This is the tier's own declaration, readable without importing matplotlib. `api.py` derives the
`"matplotlib"` row of :data:`~digitalearth.api.BACKEND_CAPABILITIES` from it, and it answers the same six
keywords it answered before: `crs`, `kind`, `domain`, `basemap`, `coastlines`, `colorbar`.

Two fields are worth reading twice, because they are where this tier differs most from the other three:

- **`channels` is two names.** `opacity` and `size` are the two
  :data:`~digitalearth.static.render_compat.STATIC_STYLE_SCHEMA` declares, and `data_driven` is empty: the
  flat keywords cleopatra's glyphs take carry a *constant*, so a channel driven by a field has to be resolved
  to values by the builder — which :func:`~digitalearth.static.render_compat.fold_symbology` reports as
  unsupported rather than pretends to do.
- **`export_vector` is this tier's alone.** A matplotlib figure is written as PDF, SVG or EPS; the other
  three draw onto a canvas and say so in their own `absent`.
- **`custom:matplotlib` is declared**, as each other tier declares its own engine's custom kind: an
  artist a caller hands to :meth:`~digitalearth.static.scene.Scene._add_layer` is described, held, and
  drawn back from that description. Drawing straight onto ``Map.ax`` is still the escape hatch, and is
  still outside the description by design (D-3) — the figure never hears about it.
**Unlike the other three tiers, this module does not promise an engine-free import.** `three_d`,
`web` and `interactive` keep their declaration readable without pyvista, maplibre or holoviews,
because each of those is an optional extra and `api.py` has to answer for a backend nobody installed.
matplotlib is a core dependency, and `digitalearth.static` imports `Map` eagerly, so importing this
module loads it. There is nothing to defer: a caller who can import `digitalearth` at all already has
matplotlib (#294).

"""

from digitalearth.base.capabilities import Capabilities

__all__ = ["CAPABILITIES"]

#: The static tier's declaration. The kinds are the builders `Map` composes — the registry's own
#: descriptions name them (`raster` is "static imshow", `flow` is "static sankey") — and the schemes are the
#: shared classifier's, so one `scheme`/`k` pair paints the same classes here as on every other tier
#: (contract C4). `categorical` is among them for the vector glyphs; cleopatra's `ArrayGlyph` refuses it,
#: since a raster's cells are a continuous field rather than nominal labels.
CAPABILITIES = Capabilities(
    backend="matplotlib",
    kinds=frozenset(
        {
            "raster",
            "mesh",
            "rgb",
            "contours",
            "filled_contours",
            "vectors",
            "streamlines",
            "points",
            "polygons",
            "choropleth",
            "unstructured",
            "heatmap",
            "flow",
            "text",
            "graticule",
            "basemap",
            "coastlines",
            "borders",
            "land",
            "ocean",
            "lakes",
            "rivers",
            "custom:matplotlib",
        }
    ),
    channels=frozenset({"opacity", "size"}),
    data_driven=frozenset(),
    schemes=frozenset(
        {
            "categorical",
            "quantiles",
            "equal_interval",
            "natural_breaks",
            "fisher_jenks",
            "percentiles",
            "std_mean",
        }
    ),
    features=frozenset(
        {
            "display_crs",
            "domain",
            "coastline_overlay",
            "raster_renderer",
            "colorbar",
            "legend",
            "animation",
            "export_image",
            "export_vector",
        }
    ),
    absent={
        "tooltip": (
            "a matplotlib figure is a picture: there is no pointer over it to hover, so a value is read from "
            "the colorbar or printed into the cell"
        ),
        "height": (
            "an axes is flat; a layer raised by a column is the 3-D tier's extrusion or the web tier's"
        ),
        "export_html": (
            "the figure is written as an image, not as a page; an HTML export is the interactive or web tier's"
        ),
        "layer_switcher": (
            "the figure is drawn once and does not respond to a pointer, so there is nothing for a switch to "
            "toggle; layers are chosen before the figure is drawn"
        ),
        "time_slider": (
            "a sequence over time is written out as an animation here rather than scrubbed, which is what "
            "`animate` is"
        ),
        "navigation": "there is no viewport to pan: the extent is set by `set_extent` before drawing",
        "fullscreen": "a saved image has no screen to fill; its size is the figure's",
        "measure": "there is no pointer to measure with; a distance is drawn as a layer of its own",
        "attribution": (
            "a credit is text placed on the figure — `text` or `stamp` — rather than a control the tier draws"
        ),
    },
)
