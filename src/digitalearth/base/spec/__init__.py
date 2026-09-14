"""spec — the engine-neutral vocabulary a figure is described in.

These are **value objects**: frozen, comparable, renderer-free. They say *what* a map shows — which rectangle,
which slice, which colour rule — and never *how* a backend draws it. Nothing here may import matplotlib,
pyvista, holoviews or maplibre; :mod:`tests.test_base_is_engine_neutral` enforces that.

The point of a shared vocabulary is that a value crosses a tier boundary without a convention having to travel
beside it in a docstring. A bare ``[float, float, float, float]`` cannot say whether it is
``[xmin, xmax, ymin, ymax]`` or ``[xmin, ymin, xmax, ymax]``, and this package existed because both spellings
were live in one tier at once.

What lives here:

* :mod:`~digitalearth.base.spec.bounds` — :class:`~digitalearth.base.spec.bounds.Bounds`, a rectangle that
  carries its own CRS and names each ordering as a method rather than leaving it to position.
* :mod:`~digitalearth.base.spec.scale` — :class:`~digitalearth.base.spec.scale.Scale`, how a value becomes
  a colour: the domain, the widening rule five tiers had each written out, the classifier three tiers each
  called, and the categorical mapping — resolved once and freezable.
"""

from digitalearth.base.spec.bounds import Bounds
from digitalearth.base.spec.scale import DEFAULT_CLASS_COUNT, Scale

__all__ = ["Bounds", "DEFAULT_CLASS_COUNT", "Scale"]
