"""What the web tier can draw, declared as data (U-1, #294).

The tier answered this in as many places as there were questions: `api.py` held the `quickmap` keywords it
honours, `WebMap(crs=)` refused anything but EPSG:4326 in its own words, and whether a builder existed was
something a caller found out by calling it. This module is the tier's own answer, readable without importing
MapLibre, so a dispatcher can consult it before it decides which backend to build.

The `absent` half is where the web tier differs most from the others. It pans and zooms, so there is no extent
to set; its colour key is a builder that takes content rather than a toggle; and the field and mesh renderings
the matplotlib tier offers have no MapLibre equivalent, which is a decision about what this tier is for rather
than a gap somebody has yet to fill.
"""

from digitalearth.base.capabilities import Capabilities

__all__ = ["CAPABILITIES"]

#: The web tier's declaration. The kinds are the builders `WebMap` composes, plus `custom:maplibre` for a layer
#: a caller builds and hands to `add_layer` (#293). The furniture is the tier's controls and its time slider,
#: declared under the names #292 registers them with.
CAPABILITIES = Capabilities(
    backend="web",
    kinds=frozenset(
        {
            "raster",
            "rgb",
            "contours",
            "filled_contours",
            "points",
            "lines",
            "polygons",
            "choropleth",
            "labels",
            "heatmap",
            "clusters",
            "point_cloud",
            "extrusion",
            "terrain",
            "model",
            "text",
            "graticule",
            "basemap",
            "custom:maplibre",
        }
    ),
    channels=frozenset(
        {"color", "opacity", "size", "width", "height", "text", "tooltip"}
    ),
    data_driven=frozenset({"color", "height", "text"}),
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
            "legend",
            "animation",
            "export_image",
            "export_html",
            "navigation",
            "scale_bar",
            "fullscreen",
            "layer_switcher",
            "time_slider",
            "measure",
            "attribution",
        }
    ),
    absent={
        "domain": "a web map pans and zooms, so it is framed by a centre and a zoom rather than by a region",
        "raster_renderer": (
            "a raster is drawn as an image and contours are their own builder, so there is no choice of "
            "renderer to make on top of one"
        ),
        "coastlines": (
            "coastlines come with the basemap style a caller picks, rather than as a Natural Earth layer the "
            "tier draws"
        ),
        "colorbar": (
            "the tier's colour key is legend(), a builder that takes content, rather than a toggle over a bar "
            "matplotlib drew"
        ),
        "vectors": "MapLibre has no arrow glyph; a u/v field is drawn on the static or interactive tier",
        "streamlines": "there is no streamline primitive to trace a field with in a browser",
        "mesh": "a raster is drawn as an image rather than as cells, which is what a tile pipeline expects",
        "unstructured": "a UGRID mesh has no MapLibre layer type; it is drawn on the static tier",
        "north_arrow": (
            "the navigation control's compass already shows the bearing, and turns the map back to north when "
            "it is clicked"
        ),
        "export_vector": "a page is a raster canvas; a PDF of it would be a screenshot in a wrapper",
    },
)
