"""A layer that describes itself, and the ordered tree a figure keeps its layers in.

Every tier keeps a list called ``layers`` and no two hold the same thing: ``(glyph, mappable)`` tuples on the static
tier, ``(mesh, actor)`` on the 3-D tier, HoloViews elements on the interactive tier and ``apply(widget)`` closures on
the web tier. Three of them address a layer by its position, and an underlay inserted at index ``0`` moves every
position a caller holds. None of them keeps what a layer *is* — its kind, its data, its slice, its style — so a
layer cannot be exported, re-read, restyled after drawing or rebuilt on another tier.

:class:`LayerSpec` is that description, and holds **no data object, no viewport and no engine handle**: an address
(:class:`~digitalearth.base.spec.dataref.DataRef`, by id), a slice
(:class:`~digitalearth.base.spec.selection.Selection`) and a look
(:class:`~digitalearth.base.spec.style.Symbology`). :class:`LayerTree` orders them and addresses them by id.

Two decisions this module settles:

* **Z-order has one source of truth — the tree.** A layer does not carry an ``order`` field as well; two places for
  the same fact is the drift this refactor removes.
* **A change returns a new tree.** A renderer that reconciles a figure needs the previous and the next description to
  compare, so the tree is a value like everything else in :mod:`digitalearth.base.spec`.
"""

from collections import Counter
from dataclasses import dataclass, field
from dataclasses import replace as with_fields
from numbers import Integral
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

from digitalearth.base.registry import KIND_BANDS, band_of, is_kind_name
from digitalearth.base.spec._serial import (
    as_list,
    plain_text,
    read_entry,
    refuse_unknown,
    require,
    true_or_false,
)
from digitalearth.base.spec.selection import Selection
from digitalearth.base.spec.style import Symbology

__all__ = ["LAYER_REFERENCE", "LayerSpec", "LayerTree"]

#: The prefix a ``z_source`` takes when a layer's elevation comes from **another layer** rather than from a source —
#: imagery draped over terrain (#202). ``"layer:dem"`` names the layer ``dem``; anything else names a source.
LAYER_REFERENCE = "layer:"


def _optional_text(owner: str, name: str, value: Any) -> None:
    """Refuse a field that should be ``None`` or a non-empty string.

    Args:
        owner: The type being built, for the message.
        name: The field.
        value: Its value.

    Raises:
        ValueError: for a non-string, or an empty string. Whitespace is kept, as it is in an id: these strings are
            compared exactly, and the web tier passes a caller's layer name through as the label verbatim.
    """
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{owner} needs {name} as a non-empty string or None; got {value!r}"
        )


@dataclass(frozen=True)
class LayerSpec:
    """What one layer draws, described without drawing it.

    **Where a decoration lives.** Anything drawn in map coordinates is a layer, decoration included: a basemap,
    coastlines, a graticule, a string placed at a coordinate. Its kind's band
    (:data:`~digitalearth.base.registry.KIND_BANDS`) decides where in draw order it lands, so a graticule added
    last is still drawn under the data. The three other homes are for what has no place on the ground: whatever
    explains an encoding is a `Guide` on that :class:`~digitalearth.base.spec.encoding.Encoding` (a legend, a
    colorbar), whatever is fixed to the frame is :class:`~digitalearth.base.spec.furniture.Furniture` on the
    `PanelSpec` (a scale bar, a north arrow, a navigation control), and the panel's heading is its `title`.

    Attributes:
        id: The layer's identity within its figure, unique within a :class:`LayerTree`, and the only way
            a layer is addressed — never by position.
        kind: What sort of layer it is — `"raster"`, `"points"`, `"graticule"` — as a key into the kind
            registry (:func:`~digitalearth.base.registry.kind_info`). A lowercase identifier, optionally after one
            namespace and a colon (`"custom:pyvista"`). The spelling is checked here; whether the kind is
            registered is checked by the renderer that draws it, so a figure naming a plugin's kind still loads
            where the plugin is not installed.
        source_id: The key of the layer's data in the figure's sources, or ``None`` for a layer drawn from no data
            source (a graticule, a tile basemap).
        selection: Which slice of the source the layer draws.
        symbology: How the layer looks.
        z_source: Where the layer's elevation comes from: ``None`` for a flat layer, a source id, or
            ``"layer:<id>"`` for another layer's surface — imagery draped over terrain (#202).
        visible: Whether the layer is drawn. A hidden layer stays in the tree, which is what lets a viewer turn it
            back on.
        label: What a layer switcher calls it; ``None`` means the id.
        group: The group the layer belongs to, so a set of layers can be hidden together.
        band: Where this layer is drawn, overriding the band its kind declares — one of
            :data:`~digitalearth.base.registry.KIND_BANDS`, or `None` to take the kind's. A custom layer needs
            it: a caller's own object is `custom:<engine>` whatever it draws, so only the caller can say whether
            it is ground cover or a label.
        filter: A filter expression, carried for the renderer rather than interpreted here.

    Raises:
        ValueError: for an id that is not a non-empty string, a kind that is not a
            lowercase identifier, a non-boolean `visible`, a `selection` or `symbology` of the wrong type, a
            `source_id`, `z_source`, `label`, `group` or `filter` that is neither `None` nor a non-empty string, a
            `band` outside :data:`~digitalearth.base.registry.KIND_BANDS`, or
            a `z_source` of `"layer:"` that names no layer or names the layer itself.

    Examples:
        - A raster layer over a source, drawn at 80% opacity:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, Symbology
            >>> dem = LayerSpec("dem", "raster", source_id="srtm", symbology=Symbology.of(opacity=0.8))
            >>> dem.kind, dem.source_id, dem.visible, dem.symbology.encoding("opacity").resolve()
            ('raster', 'srtm', True, 0.8)

            ```
        - Imagery draped over that layer names it as its elevation source:
            ```python
            >>> from digitalearth.base.spec import LayerSpec
            >>> LayerSpec("imagery", "rgb", source_id="s2", z_source="layer:dem").z_layer
            'dem'

            ```
        - A layer cannot drape itself over itself:
            ```python
            >>> from digitalearth.base.spec import LayerSpec
            >>> LayerSpec("dem", "raster", z_source="layer:dem")
            Traceback (most recent call last):
                ...
            ValueError: layer 'dem' cannot take its elevation from itself

            ```
    """

    id: str
    kind: str
    source_id: Optional[str] = None
    selection: Selection = field(default_factory=Selection)
    symbology: Symbology = field(default_factory=Symbology)
    z_source: Optional[str] = None
    visible: bool = True
    label: Optional[str] = None
    group: Optional[str] = None
    band: Optional[str] = None
    filter: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse a layer that could not be addressed, looked up or drawn.

        Raises:
            ValueError: as described on the class.
        """
        # Compared exactly and kept exactly: the web tier issues a caller's layer name verbatim as its MapLibre id,
        # padding included, and refusing it turned `name=" amsterdam"` from a working call into a ValueError.
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(
                f"LayerSpec needs an id that is a non-empty string; got {self.id!r}"
            )
        if not is_kind_name(self.kind):
            # The spelling rule lives with the kind registry, so a name a renderer can look up and a name a layer
            # can hold are one rule; whether the kind is *registered* is the renderer's question, not this one's.
            raise ValueError(
                "LayerSpec kind must be a lowercase identifier such as 'raster' or 'points', optionally after one "
                f"namespace such as 'custom:pyvista'; got {self.kind!r}"
            )
        if not isinstance(self.selection, Selection):
            raise ValueError(
                f"LayerSpec selection must be a Selection; got {type(self.selection).__name__}"
            )
        if not isinstance(self.symbology, Symbology):
            raise ValueError(
                f"LayerSpec symbology must be a Symbology; got {type(self.symbology).__name__}"
            )
        visible = true_or_false(self.visible)
        if visible is None:
            # "False" or 0 would read as a caller's intent in some places and not others; only a real boolean
            # says what it means — Python's or numpy's, which is what a comparison on an array gives back.
            raise ValueError(
                f"LayerSpec visible must be True or False; got {self.visible!r}"
            )
        object.__setattr__(self, "visible", visible)
        for name in ("source_id", "z_source", "label", "group", "filter"):
            _optional_text("LayerSpec", name, getattr(self, name))
        if self.band is not None and self.band not in KIND_BANDS:
            raise ValueError(
                f"LayerSpec band must be one of {list(KIND_BANDS)} or None; got {self.band!r}"
            )
        if self.z_source is not None and self.z_source.startswith(LAYER_REFERENCE):
            target = self.z_source[len(LAYER_REFERENCE) :]
            if not target:
                raise ValueError(
                    f"LayerSpec z_source {self.z_source!r} names no layer after {LAYER_REFERENCE!r}"
                )
            if target == self.id:
                raise ValueError(
                    f"layer {self.id!r} cannot take its elevation from itself"
                )

    @property
    def z_layer(self) -> Optional[str]:
        """The id of the layer this one takes its elevation from, if it takes it from a layer.

        Returns:
            The layer id after ``"layer:"``, or ``None`` when `z_source` is unset or names a source.

        Examples:
            - A draped layer names the layer beneath it; a flat one names nothing:
                ```python
                >>> from digitalearth.base.spec import LayerSpec
                >>> LayerSpec("a", "rgb", z_source="layer:dem").z_layer, LayerSpec("b", "raster").z_layer
                ('dem', None)

                ```
        """
        if self.z_source is None or not self.z_source.startswith(LAYER_REFERENCE):
            return None
        return self.z_source[len(LAYER_REFERENCE) :]

    @property
    def display_label(self) -> str:
        """What a layer switcher shows for this layer.

        Returns:
            `label`, or the id when no label was given.

        Examples:
            - The id stands in for a missing label:
                ```python
                >>> from digitalearth.base.spec import LayerSpec
                >>> LayerSpec("roads", "lines").display_label, LayerSpec("r", "lines", label="Roads").display_label
                ('roads', 'Roads')

                ```
        """
        return self.label if self.label is not None else self.id

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            ``id`` and ``kind``, plus each field that differs from its default — so a plain visible layer with no
            source, slice or style stores two keys.

        Raises:
            TypeError: if the selection or the symbology holds a value with no JSON form.

        Examples:
            - Only what was set is written:
                ```python
                >>> from digitalearth.base.spec import LayerSpec
                >>> LayerSpec("dem", "raster", source_id="srtm", visible=False).to_dict()
                {'id': 'dem', 'kind': 'raster', 'source_id': 'srtm', 'visible': False}

                ```
            - A plain visible layer with no source, slice or style stores two keys:
                ```python
                >>> from digitalearth.base.spec import LayerSpec
                >>> LayerSpec("grid", "graticule").to_dict()
                {'id': 'grid', 'kind': 'graticule'}

                ```
        """
        out: Dict[str, Any] = {"id": plain_text(self.id), "kind": plain_text(self.kind)}
        if self.source_id is not None:
            out["source_id"] = plain_text(self.source_id)
        if self.selection != Selection():
            out["selection"] = self.selection.to_dict()
        styled = self.symbology.to_dict()
        if styled:
            out["symbology"] = styled
        for name in ("z_source", "label", "group", "band", "filter"):
            value = getattr(self, name)
            if value is not None:
                out[name] = plain_text(value)
        if not self.visible:
            out["visible"] = False
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerSpec":
        """Rebuild a layer from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The layer, validated as the constructor validates it.

        Raises:
            TypeError: if `data`, or its `selection` or `symbology`, is not a mapping.
            ValueError: for a missing id or kind, an unknown key, a `selection` or `symbology` its own `from_dict`
                refuses, or a field the constructor refuses. An error of either type from the stored selection or
                symbology names where it sits:
                `LayerSpec.from_dict selection: Selection.from_dict needs a mapping; got int`.

        Examples:
            - A stored layer reads back with its slice:
                ```python
                >>> from digitalearth.base.spec import LayerSpec
                >>> layer = LayerSpec.from_dict({"id": "t2m", "kind": "raster", "selection": {"band": [2]}})
                >>> layer.selection.band
                (2,)

                ```
            - Z-order lives in the tree, so an `order` key is unknown and refused rather than dropped:
                ```python
                >>> from digitalearth.base.spec import LayerSpec
                >>> LayerSpec.from_dict({"id": "dem", "kind": "raster", "order": 0})  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: LayerSpec.from_dict got unknown keys ['order']; known keys are ['band', 'filter', ...]

                ```
        """
        refuse_unknown(
            "LayerSpec",
            data,
            (
                "id",
                "kind",
                "source_id",
                "selection",
                "symbology",
                "z_source",
                "visible",
                "label",
                "group",
                "band",
                "filter",
            ),
        )
        selection = data.get("selection")
        symbology = data.get("symbology")
        return cls(
            id=require("LayerSpec", data, "id"),
            kind=require("LayerSpec", data, "kind"),
            source_id=data.get("source_id"),
            selection=Selection()
            if selection is None
            else read_entry("LayerSpec", "selection", Selection.from_dict, selection),
            symbology=Symbology()
            if symbology is None
            else read_entry("LayerSpec", "symbology", Symbology.from_dict, symbology),
            z_source=data.get("z_source"),
            visible=data.get("visible", True),
            label=data.get("label"),
            group=data.get("group"),
            band=data.get("band"),
            filter=data.get("filter"),
        )


def _is_position(index: Any) -> bool:
    """Whether `index` is a whole number a position can be — a numpy integer included, a boolean not.

    Args:
        index: The position a caller passed.

    Returns:
        ``True`` for an `int` or a numpy integer. `True` is an `int` in Python and would read as position 1, and a
        whole float such as `1.0` is refused by `list.insert` with a `TypeError` naming neither the method nor the
        layer, so both answer ``False`` and the caller refuses them as it refuses a position past either end.
    """
    return isinstance(index, Integral) and not isinstance(index, bool)


def _check_layer(method: str, layer: Any) -> None:
    """Refuse something that is not a `LayerSpec` where a tree method needs one.

    Args:
        method: The tree method, for the message.
        layer: What it was given.

    Raises:
        ValueError: naming the method and the type found, as the constructor refuses a non-layer — rather than an
            `AttributeError` from reading an `id` the value does not have.
    """
    if not isinstance(layer, LayerSpec):
        raise ValueError(
            f"LayerTree.{method} needs a LayerSpec; got {type(layer).__name__}"
        )


def _layer_band(layer: LayerSpec) -> str:
    """Return the band a layer is drawn in.

    Args:
        layer: The layer.

    Returns:
        The layer's own `band` when it has one, else the band its kind declares.
    """
    return band_of(layer.kind) if layer.band is None else layer.band


def _band_bounds(layers: Sequence[LayerSpec], band: str) -> Tuple[int, int]:
    """Return the lowest and highest positions a layer of one band may take.

    Args:
        layers: The layers the new one joins, bottom first.
        band: The band it belongs to — one of :data:`~digitalearth.base.registry.KIND_BANDS`.

    Returns:
        ``(low, high)``: `low` is just above the topmost layer of a lower band, `high` just below the lowest
        layer of a higher band, so every position between them keeps the tree in band order. `high` is the top
        of the band, which is where a layer added without a position goes.

        A tree built by hand can hold its layers in any order — nothing stops `LayerTree((label, basemap))`. The
        bounds then meet at the topmost lower-band layer: the new layer is placed above the ground cover it
        belongs over, which is the half of the order that still means something.
    """
    rank = KIND_BANDS.index(band)
    low = 0
    high = len(layers)
    for position, held in enumerate(layers):
        held_rank = KIND_BANDS.index(_layer_band(held))
        if held_rank < rank:
            low = position + 1
        elif held_rank > rank and position < high:
            high = position
    return low, max(low, high)


@dataclass(frozen=True)
class LayerTree:
    """The layers of a figure, in draw order, addressed by id.

    Every change returns a new tree of the same type — a subclass stays a subclass — and leaves this one as it was,
    so a renderer can compare the tree it drew with the tree it is asked to draw.

    A layer is placed in the band its kind belongs to (:data:`~digitalearth.base.registry.KIND_BANDS`), not
    simply where it was added: ground cover stays under the data, a graticule over the ground but under the data,
    labels over everything. That is what every tier already does by hand — the web tier counts underlays onto its
    queue, the interactive tier inserts at position 0, the static tier sets `zorder` — expressed once, so the tree
    order *is* the draw order.

    Attributes:
        layers: The layers, bottom first — the first is drawn first and so sits beneath the rest.
        hidden_groups: The groups switched off. A layer in a hidden group is not drawn whatever its own `visible`
            says, and comes back as it was when the group is switched on again.

    Raises:
        ValueError: for an entry that is not a `LayerSpec`, two layers sharing an id, a `z_source` naming a layer
            that is not in the tree, a chain of `z_source` references that loops back on itself, `hidden_groups`
            given as a bare string or bytes rather than a collection of names, a hidden-group entry that is not a
            string (`LayerTree hidden_groups must be group names (strings); got 1`), or a hidden group no layer
            belongs to.

    Examples:
        - Build a tree and address layers by id rather than by position:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, LayerTree
            >>> tree = LayerTree().add(LayerSpec("dem", "raster")).add(LayerSpec("roads", "lines"))
            >>> tree.ids
            ('dem', 'roads')
            >>> tree.move("roads", 0).ids
            ('roads', 'dem')

            ```
        - Hiding a group hides its layers without forgetting their own visibility:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, LayerTree
            >>> tree = LayerTree().add(LayerSpec("a", "points", group="obs")).add(LayerSpec("b", "points"))
            >>> hidden = tree.set_group_visible("obs", False)
            >>> hidden.is_visible("a"), hidden.is_visible("b"), hidden.get("a").visible
            (False, True, True)

            ```
        - Draped layers that take their elevation from each other in a loop are refused:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, LayerTree
            >>> looped = (LayerSpec("a", "rgb", z_source="layer:b"), LayerSpec("b", "rgb", z_source="layer:a"))
            >>> LayerTree(looped)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: z_source references loop: a -> b -> a. ...

            ```
    """

    layers: Tuple[LayerSpec, ...] = ()
    hidden_groups: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        """Refuse a tree whose ids, references or hidden groups do not hold together.

        Raises:
            ValueError: as described on the class.
        """
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(
            self, "hidden_groups", self._checked_hidden_groups(self.hidden_groups)
        )
        for layer in self.layers:
            if not isinstance(layer, LayerSpec):
                raise ValueError(
                    f"LayerTree holds LayerSpec values; got {type(layer).__name__}"
                )
        # Every check below is linear in the layer count. This runs on every add, move and replace — the web tier
        # calls it once per builder — so a quadratic check here made building a map cubic.
        ids = [layer.id for layer in self.layers]
        duplicated = sorted(
            layer_id for layer_id, count in Counter(ids).items() if count > 1
        )
        if duplicated:
            raise ValueError(
                f"LayerTree ids must be unique; {duplicated} appear more than once"
            )
        self._check_drapes(ids)
        unknown_groups = sorted(
            self.hidden_groups - {layer.group for layer in self.layers}
        )
        if unknown_groups:
            raise ValueError(
                f"LayerTree hides groups {unknown_groups} that no layer belongs to; groups are {list(self.groups)}"
            )

    @staticmethod
    def _checked_hidden_groups(hidden_groups: Any) -> FrozenSet[str]:
        """Return the hidden groups as a frozenset of names, refusing anything that is not one.

        Args:
            hidden_groups: What the constructor was given.

        Returns:
            The names, frozen.

        Raises:
            ValueError: for a bare string or bytes, or an entry that is not a string.
        """
        if isinstance(hidden_groups, (str, bytes)):
            # `frozenset("obs")` is the letters of the name, and a one-letter group would be hidden silently.
            raise ValueError(
                "LayerTree hidden_groups must be a collection of group names; got the string "
                f"{hidden_groups!r}"
            )
        hidden = list(hidden_groups)
        for group in hidden:
            if not isinstance(group, str):
                # Checked before the frozenset and the sorted message in the constructor, which raised bare
                # TypeErrors for an unhashable entry or for names of mixed types.
                raise ValueError(
                    f"LayerTree hidden_groups must be group names (strings); got {group!r}"
                )
        return frozenset(hidden)

    def _check_drapes(self, ids: List[str]) -> None:
        """Refuse a drape over a layer the tree does not hold, or a chain of drapes that loops.

        Args:
            ids: The layer ids, bottom first, for the message.

        Raises:
            ValueError: naming the draped layer and the missing surface, or the loop.
        """
        by_id = {layer.id: layer for layer in self.layers}
        for layer in self.layers:
            if layer.z_layer is not None and layer.z_layer not in by_id:
                raise ValueError(
                    f"layer {layer.id!r} takes its elevation from layer {layer.z_layer!r}, which is not in the tree; "
                    f"layers are {ids}"
                )
        # A layer whose chain of drapes is already known to end on a surface is not walked again.
        grounded: Set[str] = set()
        for layer in self.layers:
            chain = [layer.id]
            on_chain = {layer.id}
            target = layer.z_layer
            while target is not None and target not in grounded:
                if target in on_chain:
                    raise ValueError(
                        f"z_source references loop: {' -> '.join(chain + [target])}. A draped layer must end on a "
                        "surface that takes its elevation from a source, not from another layer in the loop"
                    )
                chain.append(target)
                on_chain.add(target)
                target = by_id[target].z_layer
            grounded.update(chain)

    # ------------------------------------------------------------------ reading

    @property
    def ids(self) -> Tuple[str, ...]:
        """The layer ids, bottom first.

        Returns:
            The ids in draw order.

        Examples:
            - An empty tree has no ids:
                ```python
                >>> from digitalearth.base.spec import LayerTree
                >>> LayerTree().ids
                ()

                ```
        """
        return tuple(layer.id for layer in self.layers)

    @property
    def groups(self) -> Tuple[str, ...]:
        """The groups the layers belong to, in the order each first appears.

        Returns:
            The group names, without repeats.

        Examples:
            - Groups are listed once, in draw order:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree((LayerSpec("a", "points", group="obs"), LayerSpec("b", "lines", group="obs")))
                >>> tree.groups
                ('obs',)

                ```
        """
        # A dict keeps first-seen order and tests membership by hash; a list tested it by comparing against every
        # group seen so far, which made each change quadratic in the number of groups.
        return tuple(
            dict.fromkeys(
                layer.group for layer in self.layers if layer.group is not None
            )
        )

    def __len__(self) -> int:
        """Return how many layers the tree holds.

        Returns:
            The layer count.
        """
        return len(self.layers)

    def __iter__(self) -> Iterator[LayerSpec]:
        """Iterate over the layers, bottom first.

        Returns:
            An iterator over the layers in draw order.
        """
        return iter(self.layers)

    def __contains__(self, layer_id: object) -> bool:
        """Whether a layer with this id is in the tree.

        Args:
            layer_id: The id to look for.

        Returns:
            ``True`` when a layer has that id.
        """
        return layer_id in self.ids

    def get(self, layer_id: str) -> LayerSpec:
        """Return the layer with this id.

        Args:
            layer_id: The id to look up.

        Returns:
            The layer.

        Raises:
            KeyError: if no layer has that id, listing the ids that exist.

        Examples:
            - Look a layer up by id:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> LayerTree().add(LayerSpec("dem", "raster", source_id="srtm")).get("dem").source_id
                'srtm'

                ```
            - An id the tree does not hold is named beside the ids it does:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> LayerTree().add(LayerSpec("dem", "raster")).get("roads")
                Traceback (most recent call last):
                    ...
                KeyError: "no layer 'roads' in this tree; layers are ['dem']"

                ```
        """
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(
            f"no layer {layer_id!r} in this tree; layers are {list(self.ids)}"
        )

    def is_visible(self, layer_id: str) -> bool:
        """Whether a layer is drawn: its own `visible`, and its group not hidden.

        Args:
            layer_id: The layer to ask about.

        Returns:
            ``True`` when the layer is switched on and so is its group.

        Raises:
            KeyError: if no layer has that id.

        Examples:
            - A layer switched off is not drawn:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> LayerTree().add(LayerSpec("a", "points", visible=False)).is_visible("a")
                False

                ```
        """
        layer = self.get(layer_id)
        return layer.visible and layer.group not in self.hidden_groups

    # ------------------------------------------------------------------ changes (each returns a new tree)

    def add(self, layer: LayerSpec, *, index: Optional[int] = None) -> "LayerTree":
        """Return a tree with `layer` added.

        Args:
            layer: The layer to add.
            index: Where it goes **within its band**. ``None`` puts it on top of the band, which is what a
                builder wants: a basemap under the data however late it is added, a label over the data however
                early. A position is counted in the whole tree, so the band's own range is what it must fall in.

        Returns:
            The new tree.

        Raises:
            ValueError: if `layer` is not a `LayerSpec`, a layer with the same id is already in the tree, or the
                layer's `z_source` names a layer that is not.
            IndexError: if `index` is outside the layer's band — `list.insert` would clamp it silently, which puts
                the layer somewhere the caller did not ask for — or is not a whole number: a boolean or a float.

        Examples:
            - A basemap is ground cover, so it goes beneath the data wherever it is added:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("roads", "lines"))
                >>> tree.add(LayerSpec("tiles", "basemap")).ids
                ('tiles', 'roads')

                ```
            - Kinds that draw over the data stay over it, and the order within a band is the order they arrived:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("place", "text")).add(LayerSpec("dem", "raster"))
                >>> tree.add(LayerSpec("grid", "graticule")).ids
                ('grid', 'dem', 'place')

                ```
            - A position that would take the layer out of its band is refused rather than clamped:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> LayerTree().add(LayerSpec("dem", "raster")).add(LayerSpec("tiles", "basemap"), index=1)
                Traceback (most recent call last):
                    ...
                IndexError: cannot add 'tiles' at position 1; the underlay band runs from position 0 to 0

                ```
        """
        _check_layer("add", layer)
        if any(existing.id == layer.id for existing in self.layers):
            raise ValueError(
                f"a layer with id {layer.id!r} is already in the tree; layer ids must be unique"
            )
        layers = list(self.layers)
        band = _layer_band(layer)
        low, high = _band_bounds(layers, band)
        if index is None:
            layers.insert(high, layer)
        elif not _is_position(index) or not low <= index <= high:
            raise IndexError(
                f"cannot add {layer.id!r} at position {index!r}; the {band} band runs from position {low} to {high}"
            )
        else:
            layers.insert(index, layer)
        return with_fields(self, layers=tuple(layers))

    def remove(self, layer_id: str) -> "LayerTree":
        """Return a tree without the layer with this id.

        Args:
            layer_id: The layer to remove.

        Returns:
            The new tree. A hidden group left with no layers stops being hidden, since there is nothing left in it.

        Raises:
            KeyError: if no layer has that id.
            ValueError: if another layer takes its elevation from this one — removing the surface would leave the
                draped layer pointing at nothing.

        Examples:
            - Remove a layer by id:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> LayerTree().add(LayerSpec("a", "points")).add(LayerSpec("b", "lines")).remove("a").ids
                ('b',)

                ```
            - Removing the last layer of a hidden group un-hides the group:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("a", "points", group="obs")).add(LayerSpec("b", "lines"))
                >>> hidden = tree.set_group_visible("obs", False)
                >>> hidden.hidden_groups, hidden.remove("a").hidden_groups
                (frozenset({'obs'}), frozenset())

                ```
            - A surface another layer is draped over cannot be removed first:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> draped = LayerSpec("img", "rgb", z_source="layer:dem")
                >>> tree = LayerTree().add(LayerSpec("dem", "raster")).add(draped)
                >>> tree.remove("dem")  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: layer 'dem' cannot be removed while ['img'] take their elevation from it; ...

                ```
        """
        self.get(layer_id)
        dependents = [layer.id for layer in self.layers if layer.z_layer == layer_id]
        if dependents:
            raise ValueError(
                f"layer {layer_id!r} cannot be removed while {dependents} take their elevation from it; remove or "
                "re-point those first"
            )
        layers = tuple(layer for layer in self.layers if layer.id != layer_id)
        groups = {layer.group for layer in layers}
        # `with_fields`, not `LayerTree(...)`: a caller's subclass stays a subclass, as it does through add and move.
        return with_fields(
            self,
            layers=layers,
            hidden_groups=frozenset(
                group for group in self.hidden_groups if group in groups
            ),
        )

    def move(self, layer_id: str, index: int) -> "LayerTree":
        """Return a tree with a layer moved to another position in draw order.

        Args:
            layer_id: The layer to move.
            index: Its position afterwards, counted as a list index is — ``0`` is the bottom, ``-1`` the top.
                A layer moves **within its band**: reordering the data layers is what a layer switcher does, and
                dragging the basemap over them is not something a caller means to do.

        Returns:
            The new tree. Every other layer keeps its relative order.

        Raises:
            KeyError: if no layer has that id.
            IndexError: if `index` is outside the tree or outside the layer's band, or is not a whole number — a
                boolean or a float — exactly as :meth:`add` refuses one.

        Examples:
            - Bring a layer to the top:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("a", "points")).add(LayerSpec("b", "lines"))
                >>> tree.move("a", -1).ids
                ('b', 'a')

                ```
            - Send a layer to the bottom; the others keep their relative order:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree((LayerSpec("a", "points"), LayerSpec("b", "lines"), LayerSpec("c", "fill")))
                >>> tree.move("c", 0).ids
                ('c', 'a', 'b')

                ```
            - A layer cannot be moved out of its band; the message says where the band runs:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("tiles", "basemap")).add(LayerSpec("dem", "raster"))
                >>> tree.move("tiles", -1)
                Traceback (most recent call last):
                    ...
                IndexError: cannot move 'tiles' to position 1; the underlay band runs from position 0 to 0

                ```
        """
        layer = self.get(layer_id)
        count = len(self.layers)
        if not _is_position(index) or not -count <= index < count:
            raise IndexError(
                f"cannot move {layer_id!r} to position {index}; the tree holds {count} layers"
            )
        position = index % count
        others = [candidate for candidate in self.layers if candidate.id != layer_id]
        band = _layer_band(layer)
        low, high = _band_bounds(others, band)
        if not low <= position <= high:
            raise IndexError(
                f"cannot move {layer_id!r} to position {position}; the {band} band runs from position {low} "
                f"to {high}"
            )
        others.insert(position, layer)
        return with_fields(self, layers=tuple(others))

    def replace(self, layer: LayerSpec) -> "LayerTree":
        """Return a tree with the layer of the same id swapped for `layer`, in the same position.

        This is how a layer is restyled or re-pointed after it has been added: the id and the position stay, the
        description changes.

        Args:
            layer: The new description. Its id names the layer it replaces.

        Returns:
            The new tree. A hidden group the new description leaves with no layers stops being hidden, as
            :meth:`remove` forgets one, since there is nothing left in it.

        Raises:
            KeyError: if no layer has that id.
            ValueError: if `layer` is not a `LayerSpec`, or the new description breaks what the tree holds together:
                its `z_source` names a layer the tree does not have or closes a loop.

        Examples:
            - Restyle a layer without moving it:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree, Symbology
                >>> tree = LayerTree().add(LayerSpec("a", "points")).add(LayerSpec("b", "lines"))
                >>> restyled = tree.replace(LayerSpec("a", "points", symbology=Symbology.of(color="#f00")))
                >>> restyled.ids, restyled.get("a").symbology.encoding("color").resolve()
                (('a', 'b'), '#f00')

                ```
            - Moving the last layer out of a hidden group un-hides the group, as removing it would:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("a", "points", group="obs")).add(LayerSpec("b", "lines"))
                >>> hidden = tree.set_group_visible("obs", False)
                >>> hidden.hidden_groups, hidden.replace(LayerSpec("a", "points")).hidden_groups
                (frozenset({'obs'}), frozenset())

                ```
            - Re-pointing a layer's elevation at a layer the tree lacks is refused:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("dem", "raster"))
                >>> tree.replace(LayerSpec("dem", "raster", z_source="layer:nope"))  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: layer 'dem' takes its elevation from layer 'nope', which is not in the tree; ...

                ```
        """
        _check_layer("replace", layer)
        self.get(layer.id)
        layers = tuple(
            layer if existing.id == layer.id else existing for existing in self.layers
        )
        groups = {existing.group for existing in layers}
        return with_fields(
            self,
            layers=layers,
            hidden_groups=frozenset(
                group for group in self.hidden_groups if group in groups
            ),
        )

    def set_visible(self, layer_id: str, visible: bool) -> "LayerTree":
        """Return a tree with one layer switched on or off.

        Args:
            layer_id: The layer to change.
            visible: Whether it should be drawn.

        Returns:
            The new tree.

        Raises:
            KeyError: if no layer has that id.
            ValueError: if `visible` is not a boolean.

        Examples:
            - Switch a layer off and back on:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("a", "points"))
                >>> tree.set_visible("a", False).is_visible("a"), tree.is_visible("a")
                (False, True)

                ```
        """
        return self.replace(
            cast(LayerSpec, with_fields(self.get(layer_id), visible=visible))
        )

    def set_group_visible(self, group: str, visible: bool) -> "LayerTree":
        """Return a tree with a whole group switched on or off.

        Args:
            group: The group to change.
            visible: Whether its layers should be drawn. Each layer's own `visible` is kept, so switching the group
                back on restores exactly what was drawn before.

        Returns:
            The new tree.

        Raises:
            KeyError: if no layer belongs to that group.
            ValueError: if `visible` is not a boolean.

        Examples:
            - A group switched off hides every layer in it:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("a", "points", group="obs"))
                >>> tree.set_group_visible("obs", False).hidden_groups
                frozenset({'obs'})

                ```
            - Switching the group back on restores a layer that was hidden only by its group:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> hidden = LayerTree().add(LayerSpec("a", "points", group="obs")).set_group_visible("obs", False)
                >>> hidden.is_visible("a"), hidden.set_group_visible("obs", True).is_visible("a")
                (False, True)

                ```
        """
        if group not in self.groups:
            raise KeyError(
                f"no layer belongs to group {group!r}; groups are {list(self.groups)}"
            )
        shown = true_or_false(visible)
        if shown is None:
            raise ValueError(
                f"set_group_visible needs visible as True or False; got {visible!r}"
            )
        hidden = set(self.hidden_groups)
        if shown:
            hidden.discard(group)
        else:
            hidden.add(group)
        return with_fields(self, hidden_groups=frozenset(hidden))

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            ``layers`` as a list in draw order, plus ``hidden_groups`` sorted, when any group is hidden.

        Raises:
            TypeError: if a layer holds a value with no JSON form.

        Examples:
            - The layers are written bottom first:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> LayerTree().add(LayerSpec("a", "points")).to_dict()
                {'layers': [{'id': 'a', 'kind': 'points'}]}

                ```
            - Hidden groups are written as a sorted list:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree((LayerSpec("a", "points", group="obs"), LayerSpec("b", "lines", group="base")))
                >>> tree.set_group_visible("obs", False).set_group_visible("base", False).to_dict()["hidden_groups"]
                ['base', 'obs']

                ```
        """
        out: Dict[str, Any] = {"layers": [layer.to_dict() for layer in self.layers]}
        if self.hidden_groups:
            out["hidden_groups"] = sorted(
                plain_text(group) for group in self.hidden_groups
            )
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerTree":
        """Rebuild a tree from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The tree, validated as the constructor validates it.

        Raises:
            TypeError: if `data` or a stored layer is not a mapping, or `layers` or `hidden_groups` is not a
                list, naming the field.
            ValueError: for an unknown key, a layer `LayerSpec.from_dict` refuses, or a tree the constructor
                refuses. An error of either type from a stored layer names its position, and each nested read adds
                its own step: `LayerTree.from_dict layers[1]: LayerSpec.from_dict needs 'kind'; got keys ['id']`.

        Examples:
            - A stored tree reads back in the same order:
                ```python
                >>> from digitalearth.base.spec import LayerTree
                >>> LayerTree.from_dict({"layers": [{"id": "a", "kind": "points"}, {"id": "b", "kind": "lines"}]}).ids
                ('a', 'b')

                ```
            - A hidden group stays hidden across a round trip:
                ```python
                >>> from digitalearth.base.spec import LayerTree
                >>> stored = {"layers": [{"id": "a", "kind": "points", "group": "obs"}], "hidden_groups": ["obs"]}
                >>> LayerTree.from_dict(stored).is_visible("a")
                False

                ```
        """
        refuse_unknown("LayerTree", data, ("layers", "hidden_groups"))
        return cls(
            tuple(
                read_entry("LayerTree", f"layers[{index}]", LayerSpec.from_dict, layer)
                for index, layer in enumerate(
                    as_list("LayerTree", "layers", data.get("layers", ()))
                )
            ),
            # The constructor checks each entry is a group name and freezes them; a frozenset here would raise a bare
            # "unhashable type" for a nested list before that check could name it.
            cast(
                FrozenSet[str],
                as_list("LayerTree", "hidden_groups", data.get("hidden_groups", ())),
            ),
        )
