"""What the interactive tier can draw, declared as data (U-1, #294).

The tier's refusals were scattered across the modules that raise them: `_require_web_mercator` guarded six call
sites, `tiles()` refused a cartopy projection in its own words, `layer_control(reorder=True)` raised
`NotImplementedError`, and `api.py` held one table of the `quickmap` keywords the tier honours. None of it could
be read before something was drawn.

This module is the tier's own answer, readable without importing HoloViews. Its `channels` and `data_driven`
come from #298's fold, measured against the installed engine rather than guessed: `height` has no HoloViews
option at all, so it is absent here with that reason.

The `text` *channel* has no option either — HoloViews draws a label as a `Labels` value dimension rather than
styling it — but it is not listed as absent, because the tier draws a `text` *layer* and one name cannot be
both claimed and disclaimed. Which channels fold is
:data:`~digitalearth.interactive.style_fold.UNEXPRESSIBLE`'s answer; this declaration answers what a caller can
ask the tier to draw.

**Three capabilities here are conditional, and stay tier checks.** `basemap`, `coastlines` and the Natural
Earth `features` are declared — the tier draws all three — but only on the default Web-Mercator display CRS,
because Bokeh renders tiles and GeoViews' Bokeh features in EPSG:3857 only and on any other CRS they would
misalign with the pre-reprojected data silently. A `Capabilities` answers by name, not by the value of
another parameter, so it cannot say "yes, when `crs=3857`";
:meth:`~digitalearth.interactive.base.InteractiveMapBase._require_web_mercator` says it instead, at the six
call sites that need it, and is named here so the two are read together.
"""

from digitalearth.base.capabilities import Capabilities

__all__ = ["CAPABILITIES"]

#: The interactive tier's declaration. The kinds are the builders `InteractiveMap` composes — HoloViews and
#: GeoViews elements — plus `custom:holoviews` for an element a caller builds and hands to `add_layer`
#: (#293). The schemes are the shared classifier's, so one `scheme`/`k` pair paints the same classes here as on
#: every other tier (contract C4).
CAPABILITIES = Capabilities(
    backend="interactive",
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
            "lines",
            "polygons",
            "choropleth",
            "labels",
            "heatmap",
            "flow",
            "unstructured",
            "text",
            "graticule",
            "basemap",
            "coastlines",
            "borders",
            "land",
            "ocean",
            "lakes",
            "rivers",
            "custom:holoviews",
        }
    ),
    channels=frozenset({"color", "opacity", "size", "width", "rotation", "tooltip"}),
    data_driven=frozenset({"color"}),
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
            "raster_renderer",
            "coastline_overlay",
            "colorbar",
            "legend",
            "animation",
            "export_image",
            "export_html",
            "layer_switcher",
            "time_slider",
        }
    ),
    absent={
        "domain": (
            "no domain= on the constructor: a region is asked for with set_bounds(), which frames the map "
            "on a rectangle or on its own data, and the viewer pans on from there"
        ),
        "height": (
            "HoloViews has no z-height option: an extrusion is a 3-D or web layer (measured in #298)"
        ),
        "scale_bar": "Bokeh has no scale-bar tool; the axes carry the coordinates instead",
        "north_arrow": "a Bokeh frame is drawn north-up, so an arrow would say only what the axes say",
        "attribution": "a tile source's credit is drawn by the tile layer itself, not by a separate control",
        "navigation": "Bokeh's own toolbar pans and zooms, so a second set of buttons would duplicate it",
        "fullscreen": "a notebook cell or a served page is sized by its host, not by a control in the figure",
        "measure": "there is no measurement tool in Bokeh's toolbar; a distance is drawn as a layer",
        "export_vector": "the figure is a Bokeh canvas; a vector export is the static tier's",
    },
)
