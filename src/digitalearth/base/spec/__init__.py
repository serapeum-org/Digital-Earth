"""spec — the engine-neutral vocabulary a figure is described in.

These are **value objects**: frozen, comparable, renderer-free. They say *what* a map shows — which
rectangle, which slice, which colour rule — and never *how* a backend draws it. Nothing here may import
matplotlib, pyvista, holoviews or maplibre; :mod:`tests.test_base_is_engine_neutral` enforces that.

"Frozen" covers the structure, and the sequences inside it. Every sequence field is copied to a tuple and
every mapping to a read-only `FrozenDict`, and the free-form values (`Selection.time`, `Encoding.value`, a `Symbology`
property, a `Scale` category) store their lists as tuples, however nested. So `Selection.of(1, time=[1, 2])` and
`Selection.of(1, time=(1, 2))` are one value that hashes, and a round trip through JSON — which has no tuple —
reads back equal. A numpy array held as such a value is stored the same way, as nested tuples of its elements.
A dict is copied with its values frozen, and does not hash.

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
* :mod:`~digitalearth.base.spec.layer` — :class:`~digitalearth.base.spec.layer.LayerSpec`, what one layer draws,
  and :class:`~digitalearth.base.spec.layer.LayerTree`, the layers in draw order, addressed by id.
* :mod:`~digitalearth.base.spec.viewport` — :class:`~digitalearth.base.spec.viewport.Viewport` and
  :class:`~digitalearth.base.spec.viewport.Camera`: the view of a flat map and of a 3-D scene, as values.
* :mod:`~digitalearth.base.spec.target` — :class:`~digitalearth.base.spec.target.RenderTarget`, the output a
  figure is rendered to, which owns the read budget.
* :mod:`~digitalearth.base.spec.figure` — :class:`~digitalearth.base.spec.figure.PanelSpec` and
  :class:`~digitalearth.base.spec.figure.FigureSpec`: the whole figure, which round-trips through a dict with no
  renderer imported.

Every type here that a figure stores has a `to_dict`/`from_dict` pair, and each pair refuses an unknown key rather
than dropping it. All of them, `DataRef`'s included, are built on the shared rules in `_serial.py`, which also
refuse a free-form value — a constant, a style property, a category, a slice axis — with no JSON form where it is
written.

Note the neighbour: :mod:`digitalearth.base.symbology` is the colour *arithmetic* (resolving a categorical
cmap, sampling it, the missing colour). :class:`~digitalearth.base.spec.style.Symbology` here is the
*vocabulary* — what a layer declares. They are different jobs, which is why they are different modules.
"""

from digitalearth.base.spec.bounds import Bounds
from digitalearth.base.spec.dataref import DataRef
from digitalearth.base.spec.encoding import CHANNELS, Channel, Encoding, Guide
from digitalearth.base.spec.figure import (
    SCHEMA_VERSION,
    FigureDiff,
    FigureSpec,
    PanelSpec,
)
from digitalearth.base.spec.furniture import Furniture
from digitalearth.base.spec.layer import (
    LAYER_REFERENCE,
    LayerSpec,
    LayerTree,
    free_layer_id,
)
from digitalearth.base.spec.legend import (
    DEFAULT_RAMP_STOPS,
    LEGEND_KINDS,
    LegendEntry,
    LegendSpec,
)
from digitalearth.base.spec.scale import DEFAULT_CLASS_COUNT, Scale
from digitalearth.base.spec.selection import DEFAULT_BAND, Selection
from digitalearth.base.spec.style import StyleKey, StyleSchema, Symbology
from digitalearth.base.spec.target import DEFAULT_BUDGETS, TARGET_KINDS, RenderTarget
from digitalearth.base.spec.viewport import DEFAULT_VIEW_ANGLE, Camera, Viewport
from digitalearth.base.spec.viewrequest import ViewRequest

__all__ = [
    "Bounds",
    "CHANNELS",
    "Camera",
    "DEFAULT_BAND",
    "DEFAULT_BUDGETS",
    "DEFAULT_CLASS_COUNT",
    "DEFAULT_RAMP_STOPS",
    "DEFAULT_VIEW_ANGLE",
    "Channel",
    "DataRef",
    "Encoding",
    "FigureDiff",
    "FigureSpec",
    "Furniture",
    "Guide",
    "LAYER_REFERENCE",
    "LEGEND_KINDS",
    "LegendEntry",
    "LayerSpec",
    "LayerTree",
    "free_layer_id",
    "LegendSpec",
    "PanelSpec",
    "RenderTarget",
    "SCHEMA_VERSION",
    "Scale",
    "Selection",
    "StyleKey",
    "StyleSchema",
    "Symbology",
    "TARGET_KINDS",
    "ViewRequest",
    "Viewport",
]
