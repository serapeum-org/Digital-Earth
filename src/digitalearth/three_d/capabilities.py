"""What the 3-D tier can draw, declared as data (U-1, #294).

`api.py` knew one thing about this tier — which `quickmap` keywords it honours — and everything else was
answered by trying: a builder either existed or it did not, a keyword either reached PyVista or raised. This
module is the tier's own answer, readable without importing PyVista, so a dispatcher can consult it before it
decides which backend to build.

Read it as a list of decisions rather than a list of methods. `absent` is the interesting half: a 3-D scene has
no extent to frame and no tile basemap, and that is a property of drawing in a camera's view rather than a gap
somebody has yet to fill. What is simply not built yet — a title, text at a coordinate (#203) — is in neither
half, because "we decided not to" and "not yet" are different answers and only the tier can tell them apart.
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
            "volume",
            "isosurface",
            "vectors",
            "extrusion",
            "raster",
            # A globe draws its own shoreline onto the sphere — `globe(data)` records one by default — so the
            # tier draws this kind even though it has no standalone coastline builder (review M6).
            "coastlines",
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
            "animation",
            "export_image",
            "export_html",
        }
    ),
    absent={
        "domain": "a scene is framed by its camera, not by an extent, so there is no region to set",
        "raster_renderer": (
            "a raster becomes a surface, a volume or a globe here — which one is the builder that was called, "
            "not a rendering choice on top of one"
        ),
        "basemap": "there are no map tiles to drape under a scene drawn in three dimensions",
        "coastline_overlay": (
            "a globe draws its own shoreline onto the sphere, and a flat scene has none to trace, so there "
            "is no builder to add one to a scene"
        ),
        # `legend` and `colorbar` are not listed: they are Core *methods* this tier has not built, which is
        # `contract.PENDING`'s answer, not this one. "We decided against it" and "nobody has written it yet"
        # are different, and a name cannot be both (review M19).
        "scale_bar": "a screen distance means nothing when the camera decides the scale of what is in front",
        "north_arrow": "the scene can be looked at from any direction, so there is no fixed north on screen",
        "attribution": "a render window has no credit line; a caller writes one beside the image it saves",
        "navigation": "the render window pans, zooms and rotates with the mouse rather than with buttons",
        "fullscreen": "the render window is resized by the window manager, not by a control in the scene",
        "layer_switcher": "layers are switched by id through the scene's own API rather than from a panel",
        "time_slider": "frames over time are rendered by animate() rather than scrubbed in the window",
        "measure": "PyVista's own measurement widget is the tier's answer, and is not part of a figure",
    },
)
