# TexturedGlobe

Drape a pyramids raster — or a whole-world tile basemap — over a 3-D sphere on a matplotlib `Axes3D`.

`TexturedGlobe` wraps cleopatra's `TexturedGlobeGlyph`, which is deliberately geometry-only: it paints an
equirectangular array onto a tilted, spinnable sphere and knows nothing about rasters, CRSes or nodata. This
class owns the geospatial half of that seam — building the texture from geodata, and mapping lon/lat back
onto the drawn surface so vector overlays land where they belong at any spin.

::: digitalearth.scene.textured_globe.TexturedGlobe
