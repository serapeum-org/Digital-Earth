"""VectorMixin — vector builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Will own: points / path / polygons / choropleth (DI.1b); vectorfield / streamlines / barbs (DI.5); trimesh / hexbin / kde (DI.6); graph / flow (DI.15).

Currently an **empty stub** (DI.0): methods land task by task; each builds a HoloViews/GeoViews element from
pyramids-sourced data and registers it via ``self.add_element(...)`` on the composed ``InteractiveMap``.
"""


class VectorMixin:
    """Vector builders mixin (stub — methods land in later DI tasks)."""
