"""Deprecated alias for :mod:`digitalearth.static` — the matplotlib backend.

``scene`` was renamed to ``static`` so every backend is named after what it renders with: ``static``
(matplotlib), ``interactive`` (HoloViz), ``three_d`` (PyVista) and ``web`` (MapLibre + deck.gl). That also
matches the ``backend=`` values ``quickmap``/``quickplot`` already accept.

This shim keeps ``from digitalearth.scene import Map`` working for one release. Import from
:mod:`digitalearth.static` instead — or, better, from the package root (``from digitalearth import Map``),
which never changed.

The names ``scene`` exported are forwarded (``Scene``, ``Map``, ``TexturedGlobe``, ``grid``,
``shared_colorbar``), plus ``projections`` — it is in the package root's ``__all__`` and resolved off the
old ``scene`` package as a submodule, so following the root API to its source must not dead-end here. Submodule paths such as ``digitalearth.scene.maps.vector`` are **not** aliased; use
``digitalearth.static.maps.vector``.
"""

import warnings

from digitalearth.static import (
    Map,
    Scene,
    TexturedGlobe,
    grid,
    projections,
    shared_colorbar,
)

__all__ = ["Scene", "Map", "TexturedGlobe", "grid", "projections", "shared_colorbar"]

warnings.warn(
    "digitalearth.scene is deprecated and will be removed in a future release; "
    "use digitalearth.static (the matplotlib backend) or import from digitalearth directly.",
    DeprecationWarning,
    stacklevel=2,
)
