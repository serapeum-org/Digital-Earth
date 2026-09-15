"""spec — the engine-neutral vocabulary a figure is described in.

These are **value objects**: frozen, comparable, renderer-free. They say *what* a map shows — which
rectangle, which slice, which colour rule — and never *how* a backend draws it. Nothing here may import
matplotlib, pyvista, holoviews or maplibre; :mod:`tests.test_base_is_engine_neutral` enforces that.

"Frozen" covers the structure, not everything a caller puts in it. Every sequence field is copied to a
tuple and every mapping to a read-only view, so a type cannot be re-shaped from outside — but a field
holding an arbitrary value (`Selection.time`, `Encoding.value`, a `Symbology` property) keeps whatever
object it was given. So these hash when their contents do: `Selection.of(1, time="2024-01")` hashes and
`Selection.of(1, time=[1, 2])` does not.

The point of a shared vocabulary is that a value crosses a tier boundary without a convention having to travel
beside it in a docstring. A bare ``[float, float, float, float]`` cannot say whether it is
``[xmin, xmax, ymin, ymax]`` or ``[xmin, ymin, xmax, ymax]``, and this package existed because both spellings
were live in one tier at once.

What lives here:

* :mod:`~digitalearth.base.spec.bounds` — :class:`~digitalearth.base.spec.bounds.Bounds`, a rectangle that
  carries its own CRS and names each ordering as a method rather than leaving it to position.
* :mod:`~digitalearth.base.spec.dataref` — :class:`~digitalearth.base.spec.dataref.DataRef`, where a
  layer's data comes from, so a layer refers to data instead of holding it. Resolved through
  :mod:`digitalearth.base.registry`.
* :mod:`~digitalearth.base.spec.selection` — :class:`~digitalearth.base.spec.selection.Selection`, which
  slice of a dataset a layer draws: bands as a **tuple**, plus time, level, member, overview and budget —
  the axes each tier used to carry in a parameter of its own beside the scalar ``band``.
* :mod:`~digitalearth.base.spec.scale` — :class:`~digitalearth.base.spec.scale.Scale`, how a value becomes
  a colour: the domain, the widening rule five tiers had each written out, the classifier three tiers each
  called, and the categorical mapping — resolved once and freezable.
* :mod:`~digitalearth.base.spec.encoding` — :class:`~digitalearth.base.spec.encoding.Encoding` and the
  :data:`~digitalearth.base.spec.encoding.CHANNELS` table: what drives one visual channel. Adding a channel
  is a row there rather than a keyword on every builder on every tier.
* :mod:`~digitalearth.base.spec.style` — :class:`~digitalearth.base.spec.style.Symbology`, the look of one
  layer, and :class:`~digitalearth.base.spec.style.StyleSchema`, which **declares** the style keywords a
  builder surface accepts so a typo can stop being a silently ignored keyword.

Note the neighbour: :mod:`digitalearth.base.symbology` is the colour *arithmetic* (resolving a categorical
cmap, sampling it, the missing colour). :class:`~digitalearth.base.spec.style.Symbology` here is the
*vocabulary* — what a layer declares. They are different jobs, which is why they are different modules.
"""

from digitalearth.base.spec.bounds import Bounds
from digitalearth.base.spec.dataref import DataRef
from digitalearth.base.spec.encoding import CHANNELS, Channel, Encoding
from digitalearth.base.spec.scale import DEFAULT_CLASS_COUNT, Scale
from digitalearth.base.spec.selection import DEFAULT_BAND, Selection
from digitalearth.base.spec.style import StyleKey, StyleSchema, Symbology
from digitalearth.base.spec.viewrequest import ViewRequest

__all__ = [
    "Bounds",
    "CHANNELS",
    "DEFAULT_BAND",
    "DEFAULT_CLASS_COUNT",
    "Channel",
    "DataRef",
    "Encoding",
    "Scale",
    "Selection",
    "StyleKey",
    "StyleSchema",
    "Symbology",
    "ViewRequest",
]
