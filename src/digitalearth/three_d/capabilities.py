"""What the 3-D tier can draw, declared as data (U-1, #294).

`api.py` knew one thing about this tier — which `quickmap` keywords it honours — and everything else was
answered by trying: a builder either existed or it did not, a keyword either reached PyVista or raised. This
module is the tier's own answer, readable without importing PyVista, so a dispatcher can consult it before it
decides which backend to build.

Read it as a list of decisions rather than a list of methods. `absent` is the interesting half: a 3-D scene has
no extent to frame and no tile basemap, and that is a property of drawing in a camera's view rather than a gap
somebody has yet to fill. What is simply not built yet is in neither half, because "we decided not to" and "not
yet" are different answers and only the tier can tell them apart. Text at a coordinate was in neither half and
is now a declared kind (#203); the scalar bar and the keyed list were in neither either, and order 24 has built
both — so `colorbar` and `legend` are declared **features** here and are no longer in `contract.PENDING`
(:mod:`digitalearth.three_d.legends.legend`).
"""

from digitalearth.base.capabilities import Capabilities

__all__ = ["CAPABILITIES"]

#: The 3-D tier's declaration. The kinds are the builders `Scene3D` composes, plus `custom:pyvista` for a mesh
#: or volume a caller builds and hands to `add_mesh`/`add_volume` (#293). The schemes are the shared
#: classifier's, which every tier cuts its classes with, so one `scheme`/`k` pair paints the same classes here
#: as on a static map (contract C4).
CAPABILITIES = Capabilities(
    backend="3d",
    kinds=frozenset(
        {
            "terrain",
            "point_cloud",
            "points",
            "polygons",
            "choropleth",
            "volume",
            "isosurface",
            "vectors",
            "extrusion",
            "lines",
            "streamlines",
            "raster",
            # A globe draws its own shoreline onto the sphere — `globe(data)` records one by default — so the
            # tier draws this kind even though it has no standalone coastline builder (review M6).
            "coastlines",
            # Natural-Earth coastlines/borders drawn as reference lines in a non-globe scene (TD-5b, #205),
            # through `coastlines()`/`borders()` on the decoration mixin.
            "reference_lines",
            # Natural-Earth land/ocean/lakes drawn as hole-aware polygon fills in a non-globe scene — the
            # fill half of TD-5b/#205, unblocked by cleopatra 0.42.0's `natural_earth_polygons` (#384),
            # through `land()`/`ocean()`/`lakes()` on the decoration mixin.
            "reference_fill",
            # A string placed at a coordinate (#203). It is a layer rather than furniture for the same reason
            # it is one on the interactive and web tiers: it is drawn where the data is, not in a corner.
            "text",
            # Web-map tiles draped on the ground plane as an underlay (#294 follow-up). The tiles are read
            # through cleopatra (image I/O, never GIS) and draped as a textured z=0 plane, the same way
            # `terrain(texture=)` drapes imagery onto relief. It used to be declared absent; the tier draws it
            # now, so it is a kind here and gone from `absent`.
            "basemap",
            # A latitude/longitude grid drawn across the framed extent and labelled along its edges
            # (forge3d parity, #490). A layer rather than furniture for the same reason `text` is: it is
            # drawn where the data is, reprojected into the display CRS through pyramids, not pinned to a
            # corner of the window.
            "graticule",
            # A set of placed labels, each with a filled halo and occlusion handling (forge3d parity, #491).
            # The plural, styled companion to `text`: a layer, drawn where the data is and reprojected into
            # the display CRS, not corner furniture.
            "labels",
            "custom:pyvista",
        }
    ),
    channels=frozenset({"color", "opacity", "size", "height"}),
    data_driven=frozenset({"color", "height"}),
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
            "colorbar",
            # "a keyed list of classes", which order 24 built: a classified layer's classes are drawn as
            # PyVista's own legend box, derived from the `Scale` the layer was painted with. `colorbar` was
            # already here on the strength of the scalar bar PyVista draws for a scalar layer; the two are
            # now both real methods as well as declared features (:mod:`digitalearth.three_d.legends.legend`).
            "legend",
            "animation",
            "export_image",
            "export_html",
            # Vector export (`export_vector()` / `save("scene.svg")`) through VTK's GL2PS writer — forge3d
            # parity (#494). SVG/PDF/EPS/PS/TEX; a high-resolution raster master is `screenshot(scale=N)`.
            # Textured surfaces embed as raster insets, and EXR is not offered (VTK writes 8-bit rasters).
            "export_vector",
            # The map can be framed to a named region or a bbox: `domain()` sets the camera to frame that
            # extent rather than leaving the view to whatever the data happened to fit (#294 follow-up). It
            # used to be declared absent ("framed by its camera, not an extent"); framing the camera *to* an
            # extent is how that reconciles, which is what forge3d's region framing does too.
            "domain",
            # A ground-distance scale bar (`scale_bar()`), drawn as VTK's legend-scale ruler. Honest only
            # under a parallel (orthographic) projection — where one screen distance is one ground distance
            # everywhere — so the method turns parallel projection on, and says so.
            "scale_bar",
            # A camera-aware north arrow (`north_arrow()`): an orientation-marker arrow pointing along the
            # display CRS's north, placed in a corner and turning with the camera, so it reads correctly from
            # any viewpoint. The companion to `orientation_axes`, for the one direction a map most wants named.
            "north_arrow",
            # A data/tile credit line (`attribution()`), drawn in a corner like the title's subtitle. The old
            # reason ("a caller writes one beside the image") was under-building; the tier draws it now.
            "attribution",
        }
    ),
    absent={
        "raster_renderer": (
            "a raster becomes a surface, a volume or a globe here — which one is the builder that was called, "
            "not a rendering choice on top of one"
        ),
        # `coastline_overlay` used to be declared absent ("a flat scene has none to trace"); TD-5b built it.
        # `coastlines()`/`borders()` now draw Natural-Earth reference lines in a non-globe scene (the
        # `reference_lines` kind above), so the overlay is a feature of the tier rather than a gap.
        # `domain`, `basemap`, `scale_bar`, `north_arrow` and `attribution` were all declared absent until the
        # #294 follow-up built them; they are declared above now (the first four as features, `basemap` as a
        # kind) and so are gone from here — "we decided against it" became "the tier does it".
        "navigation": "the render window pans, zooms and rotates with the mouse rather than with buttons",
        "fullscreen": "the render window is resized by the window manager, not by a control in the scene",
        "layer_switcher": "layers are switched by id through the scene's own API rather than from a panel",
        "time_slider": "frames over time are rendered by record() rather than scrubbed in the window",
        "measure": "PyVista's own measurement widget is the tier's answer, and is not part of a figure",
    },
)
