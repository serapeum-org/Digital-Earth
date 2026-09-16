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

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Tuple

from digitalearth.base.spec._serial import refuse_unknown, require
from digitalearth.base.spec.selection import Selection
from digitalearth.base.spec.style import Symbology

__all__ = ["LAYER_REFERENCE", "LayerSpec", "LayerTree"]

#: The prefix a ``z_source`` takes when a layer's elevation comes from **another layer** rather than from a source —
#: imagery draped over terrain (#202). ``"layer:dem"`` names the layer ``dem``; anything else names a source.
LAYER_REFERENCE = "layer:"

#: What a layer kind may be spelled as: lowercase, starting with a letter, then letters, digits, ``_`` or ``-``. Kinds
#: become registry keys later in the plan, so a spelling that could not be one is refused now.
_KIND = re.compile(r"[a-z][a-z0-9_-]*")


def _optional_text(owner: str, name: str, value: Any) -> None:
    """Refuse a field that should be ``None`` or a non-empty string.

    Args:
        owner: The type being built, for the message.
        name: The field.
        value: Its value.

    Raises:
        ValueError: for a non-string, or a string that is empty or only whitespace.
    """
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{owner} needs {name} as a non-empty string or None; got {value!r}"
        )


@dataclass(frozen=True)
class LayerSpec:
    """What one layer draws, described without drawing it.

    Attributes:
        id: The layer's identity. Stable for the layer's life, unique within a :class:`LayerTree`, and the only way
            a layer is addressed — never by position.
        kind: What sort of layer it is — ``"raster"``, ``"points"``, ``"graticule"``. Lowercase, starting with a
            letter.
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
        filter: A filter expression, carried for the renderer rather than interpreted here.

    Raises:
        ValueError: for an empty or padded id, a kind that is not a lowercase identifier, a non-boolean `visible`,
            a `selection` or `symbology` of the wrong type, an empty optional string, or a layer whose `z_source`
            names the layer itself.

    Examples:
        - A raster layer over a source, drawn with a colour ramp:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, Symbology
            >>> dem = LayerSpec("dem", "raster", source_id="srtm", symbology=Symbology.of(opacity=0.8))
            >>> dem.kind, dem.source_id, dem.visible
            ('raster', 'srtm', True)

            ```
        - Imagery draped over that layer names it as its elevation source:
            ```python
            >>> from digitalearth.base.spec import LayerSpec
            >>> LayerSpec("imagery", "rgb", source_id="s2", z_source="layer:dem").z_layer
            'dem'

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
    filter: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse a layer that could not be addressed, looked up or drawn.

        Raises:
            ValueError: as described on the class.
        """
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or self.id != self.id.strip()
        ):
            raise ValueError(
                f"LayerSpec needs an id that is a non-empty string with no surrounding whitespace; got {self.id!r}"
            )
        if not isinstance(self.kind, str) or not _KIND.fullmatch(self.kind):
            raise ValueError(
                f"LayerSpec kind must be a lowercase identifier such as 'raster' or 'points'; got {self.kind!r}"
            )
        if not isinstance(self.selection, Selection):
            raise ValueError(
                f"LayerSpec selection must be a Selection; got {type(self.selection).__name__}"
            )
        if not isinstance(self.symbology, Symbology):
            raise ValueError(
                f"LayerSpec symbology must be a Symbology; got {type(self.symbology).__name__}"
            )
        if not isinstance(self.visible, bool):
            # "False" or 0 would read as a caller's intent in some places and not others; only a real boolean
            # says what it means.
            raise ValueError(
                f"LayerSpec visible must be True or False; got {self.visible!r}"
            )
        for name in ("source_id", "z_source", "label", "group", "filter"):
            _optional_text("LayerSpec", name, getattr(self, name))
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
        """
        out: Dict[str, Any] = {"id": self.id, "kind": self.kind}
        if self.source_id is not None:
            out["source_id"] = self.source_id
        if self.selection != Selection():
            out["selection"] = self.selection.to_dict()
        styled = self.symbology.to_dict()
        if styled:
            out["symbology"] = styled
        for name in ("z_source", "label", "group", "filter"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
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
            TypeError: if `data` is not a mapping.
            ValueError: for a missing id or kind, an unknown key, or a field the constructor refuses.

        Examples:
            - A stored layer reads back with its slice:
                ```python
                >>> from digitalearth.base.spec import LayerSpec
                >>> layer = LayerSpec.from_dict({"id": "t2m", "kind": "raster", "selection": {"band": [2]}})
                >>> layer.selection.band
                (2,)

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
            else Selection.from_dict(selection),
            symbology=Symbology()
            if symbology is None
            else Symbology.from_dict(symbology),
            z_source=data.get("z_source"),
            visible=data.get("visible", True),
            label=data.get("label"),
            group=data.get("group"),
            filter=data.get("filter"),
        )


@dataclass(frozen=True)
class LayerTree:
    """The layers of a figure, in draw order, addressed by id.

    Every change returns a new tree and leaves this one as it was, so a renderer can compare the tree it drew with
    the tree it is asked to draw.

    Attributes:
        layers: The layers, bottom first — the first is drawn first and so sits beneath the rest.
        hidden_groups: The groups switched off. A layer in a hidden group is not drawn whatever its own `visible`
            says, and comes back as it was when the group is switched on again.

    Raises:
        ValueError: for two layers sharing an id, a `z_source` naming a layer that is not in the tree, a chain of
            `z_source` references that loops back on itself, or a hidden group no layer belongs to.

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
    """

    layers: Tuple[LayerSpec, ...] = ()
    hidden_groups: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        """Refuse a tree whose ids, references or hidden groups do not hold together.

        Raises:
            ValueError: as described on the class.
        """
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "hidden_groups", frozenset(self.hidden_groups))
        for layer in self.layers:
            if not isinstance(layer, LayerSpec):
                raise ValueError(
                    f"LayerTree holds LayerSpec values; got {type(layer).__name__}"
                )
        ids = [layer.id for layer in self.layers]
        duplicated = sorted({layer_id for layer_id in ids if ids.count(layer_id) > 1})
        if duplicated:
            raise ValueError(
                f"LayerTree ids must be unique; {duplicated} appear more than once"
            )
        by_id = {layer.id: layer for layer in self.layers}
        for layer in self.layers:
            if layer.z_layer is not None and layer.z_layer not in by_id:
                raise ValueError(
                    f"layer {layer.id!r} takes its elevation from layer {layer.z_layer!r}, which is not in the tree; "
                    f"layers are {ids}"
                )
        for layer in self.layers:
            seen = [layer.id]
            target = layer.z_layer
            while target is not None:
                if target in seen:
                    raise ValueError(
                        f"z_source references loop: {' -> '.join(seen + [target])}. A draped layer must end on a "
                        "surface that takes its elevation from a source, not from another layer in the loop"
                    )
                seen.append(target)
                target = by_id[target].z_layer
        unknown_groups = sorted(self.hidden_groups - set(self.groups))
        if unknown_groups:
            raise ValueError(
                f"LayerTree hides groups {unknown_groups} that no layer belongs to; groups are {list(self.groups)}"
            )

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
        seen: List[str] = []
        for layer in self.layers:
            if layer.group is not None and layer.group not in seen:
                seen.append(layer.group)
        return tuple(seen)

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
            index: Where it goes in draw order. ``None`` puts it on top; ``0`` puts it at the bottom, beneath every
                other layer — where a basemap goes.

        Returns:
            The new tree.

        Raises:
            ValueError: if a layer with the same id is already in the tree, or the layer's `z_source` names a layer
                that is not.
            IndexError: if `index` is negative or past the top. `list.insert` would clamp it silently, which puts
                the layer somewhere the caller did not ask for.

        Examples:
            - A basemap goes to the bottom without moving anything else's identity:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("roads", "lines"))
                >>> tree.add(LayerSpec("tiles", "tiles"), index=0).ids
                ('tiles', 'roads')

                ```
        """
        if layer.id in self.ids:
            raise ValueError(
                f"a layer with id {layer.id!r} is already in the tree; layer ids must be unique"
            )
        layers = list(self.layers)
        if index is None:
            layers.append(layer)
        elif isinstance(index, bool) or not 0 <= index <= len(layers):
            raise IndexError(
                f"cannot add {layer.id!r} at position {index!r}; positions run from 0 (bottom) to {len(layers)} (top)"
            )
        else:
            layers.insert(index, layer)
        return replace(self, layers=tuple(layers))

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
        return LayerTree(
            layers, frozenset(group for group in self.hidden_groups if group in groups)
        )

    def move(self, layer_id: str, index: int) -> "LayerTree":
        """Return a tree with a layer moved to another position in draw order.

        Args:
            layer_id: The layer to move.
            index: Its position afterwards, counted as a list index is — ``0`` is the bottom, ``-1`` the top.

        Returns:
            The new tree. Every other layer keeps its relative order.

        Raises:
            KeyError: if no layer has that id.
            IndexError: if `index` is outside the tree.

        Examples:
            - Bring a layer to the top:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree
                >>> tree = LayerTree().add(LayerSpec("a", "points")).add(LayerSpec("b", "lines"))
                >>> tree.move("a", -1).ids
                ('b', 'a')

                ```
        """
        layer = self.get(layer_id)
        count = len(self.layers)
        if not -count <= index < count:
            raise IndexError(
                f"cannot move {layer_id!r} to position {index}; the tree holds {count} layers"
            )
        position = index % count
        others = [candidate for candidate in self.layers if candidate.id != layer_id]
        others.insert(position, layer)
        return replace(self, layers=tuple(others))

    def replace(self, layer: LayerSpec) -> "LayerTree":
        """Return a tree with the layer of the same id swapped for `layer`, in the same position.

        This is how a layer is restyled or re-pointed after it has been added: the id and the position stay, the
        description changes.

        Args:
            layer: The new description. Its id names the layer it replaces.

        Returns:
            The new tree.

        Raises:
            KeyError: if no layer has that id.
            ValueError: if the new description breaks a reference the tree depends on.

        Examples:
            - Restyle a layer without moving it:
                ```python
                >>> from digitalearth.base.spec import LayerSpec, LayerTree, Symbology
                >>> tree = LayerTree().add(LayerSpec("a", "points")).add(LayerSpec("b", "lines"))
                >>> restyled = tree.replace(LayerSpec("a", "points", symbology=Symbology.of(color="#f00")))
                >>> restyled.ids, restyled.get("a").symbology.encoding("color").resolve()
                (('a', 'b'), '#f00')

                ```
        """
        self.get(layer.id)
        return replace(
            self,
            layers=tuple(
                layer if existing.id == layer.id else existing
                for existing in self.layers
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
        return self.replace(replace(self.get(layer_id), visible=visible))

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
        """
        if group not in self.groups:
            raise KeyError(
                f"no layer belongs to group {group!r}; groups are {list(self.groups)}"
            )
        if not isinstance(visible, bool):
            raise ValueError(
                f"set_group_visible needs visible as True or False; got {visible!r}"
            )
        hidden = set(self.hidden_groups)
        if visible:
            hidden.discard(group)
        else:
            hidden.add(group)
        return replace(self, hidden_groups=frozenset(hidden))

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
        """
        out: Dict[str, Any] = {"layers": [layer.to_dict() for layer in self.layers]}
        if self.hidden_groups:
            out["hidden_groups"] = sorted(self.hidden_groups)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerTree":
        """Rebuild a tree from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The tree, validated as the constructor validates it.

        Raises:
            TypeError: if `data` is not a mapping.
            ValueError: for an unknown key, or a tree the constructor refuses.

        Examples:
            - A stored tree reads back in the same order:
                ```python
                >>> from digitalearth.base.spec import LayerTree
                >>> LayerTree.from_dict({"layers": [{"id": "a", "kind": "points"}, {"id": "b", "kind": "lines"}]}).ids
                ('a', 'b')

                ```
        """
        refuse_unknown("LayerTree", data, ("layers", "hidden_groups"))
        return cls(
            tuple(LayerSpec.from_dict(layer) for layer in data.get("layers", ())),
            frozenset(data.get("hidden_groups", ())),
        )
