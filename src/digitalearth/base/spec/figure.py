"""A figure that describes itself: panels, their views, their layers, and where the data comes from.

The design's test for the description tier is one line: ``FigureSpec.from_dict(fig.to_dict())`` reproduces the
figure **with no renderer imported**. Before this module no figure could describe itself at all — the only serialisers
in the package were `DataRef`'s and `LegendSpec`'s — and multi-panel existed on one tier, as a ``grid()`` that gave
every panel the same CRS.

:class:`FigureSpec` is the whole description: a schema version, the sources by id, one
:class:`~digitalearth.base.spec.layer.LayerTree`, and the panels. :class:`PanelSpec` holds a view and the ids of the
layers it shows. Two decisions:

* **A panel names layers; it does not own them.** Linked views and swipe-compare show the same layer in two panels,
  and a tree per panel would copy it.
* **`schema_version` is written from the first version.** It is free to add now and impossible to add later, and a
  reader that meets a version it does not know refuses by name instead of guessing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from digitalearth.base.registry import OBJECT_SCHEME
from digitalearth.base.spec._serial import (
    FrozenDict,
    as_list,
    as_mapping,
    plain_text,
    positive_number,
    read_entry,
    refuse_unknown,
    require,
)
from digitalearth.base.spec.dataref import DataRef
from digitalearth.base.spec.layer import LayerSpec, LayerTree
from digitalearth.base.spec.viewport import Camera, Viewport

__all__ = ["FigureSpec", "PanelSpec", "SCHEMA_VERSION"]

#: The version of the figure description this package writes and reads. Bumped when a stored figure written by an
#: older version would be read differently; a reader refuses any version it does not know.
SCHEMA_VERSION = 1


def _is_known_version(version: Any) -> bool:
    """Whether `version` is exactly the schema version this module reads.

    Args:
        version: The candidate.

    Returns:
        `True` only for :data:`SCHEMA_VERSION` as a Python `int`; a float, a boolean, a string or a numpy integer
        is `False`. `True == 1` and `1.0 == 1` both hold in Python, so an `==` test needs a type check beside it.
        The check this replaced guarded booleans but not floats, and a figure built with `1.0` wrote
        `"schema_version": 1.0` back out, a version spelling no reader should have to expect.
    """
    return type(version) is int and version == SCHEMA_VERSION


def _identifier(owner: str, value: Any) -> None:
    """Refuse an id that could not address anything.

    Args:
        owner: The type being built, for the message.
        value: The candidate id.

    Raises:
        ValueError: for a non-string or an empty string.

    Note:
        An id is compared exactly and nothing else, so whitespace in it is kept rather than refused: `" a"` and
        `"a"` are two ids. Refusing padding broke the web tier, which uses a caller's layer name as the MapLibre id
        verbatim — `WebMap().text(..., name=" amsterdam")` worked on `main` and raised once its index became a
        `LayerTree`.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{owner} needs an id that is a non-empty string; got {value!r}"
        )


def _optional_title(owner: str, title: Any) -> None:
    """Refuse a title that is neither ``None`` nor a non-empty string.

    Args:
        owner: The type being built, for the message.
        title: The candidate title.

    Raises:
        ValueError: for a non-string or an empty string. It is the rule `LayerSpec` applies to its label: ``None``
            is the one spelling of "no title", so ``""`` is not a second one that compares unequal to it.
            Whitespace is kept, as it is in a label.
    """
    if title is not None and (not isinstance(title, str) or not title):
        raise ValueError(
            f"{owner} title must be a non-empty string or None; got {title!r}"
        )


@dataclass(frozen=True)
class PanelSpec:
    """One panel of a figure: a view, and the layers it shows.

    Attributes:
        id: The panel's identity within its figure.
        view: What the panel shows the layers through — a `Viewport` for a flat map, a `Camera` for a 3-D scene. Each
            panel has its own, so two panels can be drawn in two CRSs.
        layers: The ids of the layers the panel shows. The draw order is the figure's tree order, not this tuple's.
        title: The panel's title, or ``None``.

    Raises:
        ValueError: for an id that is not a non-empty string, a view that is neither a
            `Viewport` nor a `Camera`, `layers` given as a bare string, a layer id that is not a non-empty string, a
            layer listed twice, or a title that is not a non-empty string.

    Examples:
        - Two panels over the same layer, in two projections:
            ```python
            >>> from digitalearth.base.spec import PanelSpec, Viewport
            >>> left = PanelSpec("mercator", Viewport(3857), layers=("t2m",))
            >>> right = PanelSpec("polar", Viewport("EPSG:3413"), layers=("t2m",))
            >>> left.view.crs, right.view.crs, left.layers == right.layers
            (3857, 'EPSG:3413', True)

            ```
        - A panel with no view named is a Web Mercator map, and a list of layer ids is stored as a tuple:
            ```python
            >>> from digitalearth.base.spec import PanelSpec
            >>> panel = PanelSpec("main", layers=["dem", "roads"])
            >>> panel.view.crs, panel.layers
            (3857, ('dem', 'roads'))

            ```
        - A bare string is refused rather than read as one layer id per letter:
            ```python
            >>> from digitalearth.base.spec import PanelSpec
            >>> PanelSpec("main", layers="t2m")
            Traceback (most recent call last):
                ...
            ValueError: PanelSpec layers must be a sequence of layer ids; got the string 't2m'

            ```
    """

    id: str
    view: Union[Viewport, Camera] = field(default_factory=Viewport)
    layers: Tuple[str, ...] = ()
    title: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse a panel that could not be drawn or addressed.

        Raises:
            ValueError: as described on the class.
        """
        _identifier("PanelSpec", self.id)
        if not isinstance(self.view, (Viewport, Camera)):
            raise ValueError(
                f"PanelSpec view must be a Viewport or a Camera; got {type(self.view).__name__}"
            )
        if isinstance(self.layers, str):
            # A bare string is iterable, and tuple("t2m") would be three one-letter layer ids.
            raise ValueError(
                f"PanelSpec layers must be a sequence of layer ids; got the string {self.layers!r}"
            )
        layers = tuple(self.layers)
        for layer_id in layers:
            if not isinstance(layer_id, str) or not layer_id:
                raise ValueError(
                    f"PanelSpec layers must be non-empty layer ids; got {layer_id!r}"
                )
        repeated = sorted(
            {layer_id for layer_id in layers if layers.count(layer_id) > 1}
        )
        if repeated:
            raise ValueError(
                f"panel {self.id!r} lists layers {repeated} more than once"
            )
        object.__setattr__(self, "layers", layers)
        _optional_title("PanelSpec", self.title)

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            ``id``, the view under ``viewport`` or ``camera`` — which names the kind of view without a separate type
            field — plus ``layers`` and ``title`` when set.

        Examples:
            - The view is stored under the key that names its kind:
                ```python
                >>> from digitalearth.base.spec import PanelSpec, Viewport
                >>> PanelSpec("main", Viewport(4326), layers=("dem",)).to_dict()
                {'id': 'main', 'viewport': {'crs': 4326}, 'layers': ['dem']}

                ```
            - A 3-D panel stores a camera, and its title:
                ```python
                >>> from digitalearth.base.spec import Camera, PanelSpec
                >>> stored = PanelSpec("3d", Camera((0.0, -10.0, 5.0)), title="Terrain").to_dict()
                >>> sorted(stored), stored["camera"]["position"]
                (['camera', 'id', 'title'], [0.0, -10.0, 5.0])

                ```
        """
        out: Dict[str, Any] = {"id": plain_text(self.id)}
        if isinstance(self.view, Camera):
            out["camera"] = self.view.to_dict()
        else:
            out["viewport"] = self.view.to_dict()
        if self.layers:
            out["layers"] = [plain_text(layer_id) for layer_id in self.layers]
        if self.title is not None:
            out["title"] = plain_text(self.title)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PanelSpec":
        """Rebuild a panel from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The panel, validated as the constructor validates it.

        Raises:
            TypeError: if `data` or its view is not a mapping, or `layers` is not a list, naming the field.
            ValueError: for a missing id, an unknown key, both or neither of `viewport` and `camera`, a view its own
                `from_dict` refuses, or a panel the constructor refuses. An error of either type from the stored view
                names where it sits: `PanelSpec.from_dict viewport: Viewport.from_dict needs a mapping; got int`.

        Examples:
            - A stored 3-D panel reads back with its camera:
                ```python
                >>> from digitalearth.base.spec import PanelSpec
                >>> panel = PanelSpec.from_dict({"id": "3d", "camera": {"position": [0, -10, 5]}})
                >>> panel.view.position
                (0.0, -10.0, 5.0)

                ```
            - A panel must say which kind of view it has:
                ```python
                >>> from digitalearth.base.spec import PanelSpec
                >>> PanelSpec.from_dict({"id": "main", "layers": ["dem"]})  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: PanelSpec.from_dict needs exactly one of 'viewport' (a flat map) or 'camera' ...

                ```
        """
        refuse_unknown(
            "PanelSpec", data, ("id", "viewport", "camera", "layers", "title")
        )
        if ("viewport" in data) == ("camera" in data):
            raise ValueError(
                "PanelSpec.from_dict needs exactly one of 'viewport' (a flat map) or 'camera' (a 3-D scene); "
                f"got keys {sorted(data)}"
            )
        view: Union[Viewport, Camera] = (
            read_entry("PanelSpec", "camera", Camera.from_dict, data["camera"])
            if "camera" in data
            else read_entry(
                "PanelSpec", "viewport", Viewport.from_dict, data["viewport"]
            )
        )
        return cls(
            id=require("PanelSpec", data, "id"),
            view=view,
            layers=as_list("PanelSpec", "layers", data.get("layers", ())),
            title=data.get("title"),
        )


@dataclass(frozen=True, eq=True)
class FigureSpec:
    """A whole figure, described with no renderer present.

    Attributes:
        panels: The panels, in reading order. At least one.
        sources: Source id -> :class:`~digitalearth.base.spec.dataref.DataRef`. Layers name their data by these ids,
            so several layers can draw one source without repeating its address.
        layers: Every layer in the figure, in draw order. A panel shows the ones it names.
        size: ``(width, height)`` of the whole figure, in the renderer's units, or ``None``.
        title: The figure's title, or ``None``.
        schema_version: The description's version. Only :data:`SCHEMA_VERSION`, as an `int`, is accepted.

    Raises:
        ValueError: for a schema version that is not :data:`SCHEMA_VERSION` as an `int` (`1.0` is refused), no
            panels, a panel that is not a `PanelSpec`, two panels sharing an id, a source id that is not a non-empty
            string or maps to something other than a `DataRef`, a layer tree that is
            not a `LayerTree`, a layer whose source or elevation source is not among the sources, a panel naming a
            layer that is not in the tree, a 3-D (`Camera`) panel showing a draped layer without the layer it
            drapes over, a size that is not two positive finite numbers (a boolean counts as
            neither), or a title that is not a non-empty string.

    Examples:
        - One source, one layer, one panel:
            ```python
            >>> from digitalearth.base.spec import DataRef, FigureSpec, LayerSpec, LayerTree, PanelSpec, Viewport
            >>> fig = FigureSpec(
            ...     panels=(PanelSpec("main", Viewport(4326), layers=("dem",)),),
            ...     sources={"srtm": DataRef("data/dem.tif")},
            ...     layers=LayerTree((LayerSpec("dem", "raster", source_id="srtm"),)),
            ... )
            >>> [layer.id for layer in fig.layers_of("main")], fig.schema_version
            (['dem'], 1)

            ```
        - A panel may only show layers the figure holds:
            ```python
            >>> from digitalearth.base.spec import FigureSpec, PanelSpec
            >>> FigureSpec(panels=(PanelSpec("main", layers=("dem",)),))
            Traceback (most recent call last):
                ...
            ValueError: panel 'main' shows layers ['dem'] that are not in the figure; layers are []

            ```
        - The version is the integer, not a float equal to it:
            ```python
            >>> from digitalearth.base.spec import FigureSpec, PanelSpec
            >>> FigureSpec(panels=(PanelSpec("main"),), schema_version=1.0)
            Traceback (most recent call last):
                ...
            ValueError: FigureSpec schema_version 1.0 is not one this version of digitalearth reads; it reads 1

            ```
    """

    panels: Tuple[PanelSpec, ...]
    sources: Mapping[str, DataRef] = field(default_factory=dict)
    layers: LayerTree = field(default_factory=LayerTree)
    size: Optional[Tuple[float, float]] = None
    title: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Refuse a figure whose parts do not refer to each other correctly.

        Raises:
            ValueError: as described on the class.
        """
        if not _is_known_version(self.schema_version):
            raise ValueError(
                f"FigureSpec schema_version {self.schema_version!r} is not one this version of digitalearth reads; "
                f"it reads {SCHEMA_VERSION}"
            )
        object.__setattr__(self, "panels", self._checked_panels(self.panels))
        object.__setattr__(
            self, "sources", FrozenDict(self._checked_sources(self.sources))
        )
        if not isinstance(self.layers, LayerTree):
            raise ValueError(
                f"FigureSpec layers must be a LayerTree; got {type(self.layers).__name__}"
            )
        for layer in self.layers:
            self._check_sources(layer, self.sources)
        self._check_panel_layers()
        object.__setattr__(self, "size", self._checked_size(self.size))
        _optional_title("FigureSpec", self.title)

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...]]:
        """Pickle and copy by rebuilding through the constructor.

        Returns:
            ``(type(self), (panels, dict(sources), layers, size, title, schema_version))`` — the caller's
            subclass, not `FigureSpec` by name. Rebuilding from a plain dict goes through the same validation and
            freezing as any other construction.

        Examples:
            - A copy is equal to the original, and is a separate object:
                ```python
                >>> import copy
                >>> from digitalearth.base.spec import FigureSpec, PanelSpec
                >>> original = FigureSpec(panels=(PanelSpec("p"),))
                >>> clone = copy.deepcopy(original)
                >>> clone == original, clone is original
                (True, False)

                ```
        """
        return type(self), (
            self.panels,
            dict(self.sources),
            self.layers,
            self.size,
            self.title,
            self.schema_version,
        )

    def __hash__(self) -> int:
        """Hash by value, like every other type in the vocabulary.

        Returns:
            A hash over every field. `sources` is held as a read-only `FrozenDict`, which has no hash of its own, so
            it is hashed as its sorted items — hashed as held, it raises `unhashable type: 'FrozenDict'`, while
            `LayerTree`, `PanelSpec` and `Viewport` all hash.

        Raises:
            TypeError: if a part holds an unhashable value — a dict in a layer's `Symbology` properties, say. A
                list or a numpy array is not among them: the vocabulary stores both as tuples.

        Examples:
            - Two figures built alike hash alike, so a figure can key a cache:
                ```python
                >>> from digitalearth.base.spec import DataRef, FigureSpec, PanelSpec
                >>> one = FigureSpec(panels=(PanelSpec("p"),), sources={"a": DataRef("a.tif")})
                >>> two = FigureSpec.from_dict(one.to_dict())
                >>> hash(one) == hash(two), len({one, two})
                (True, 1)

                ```
        """
        return hash(
            (
                self.panels,
                tuple(sorted(self.sources.items())),
                self.layers,
                self.size,
                self.title,
                self.schema_version,
            )
        )

    @staticmethod
    def _checked_panels(panels: Any) -> Tuple[PanelSpec, ...]:
        """Return the panels as a tuple, refusing none, a non-panel, or two panels sharing an id.

        Args:
            panels: What the constructor was given.

        Returns:
            The panels, in order.

        Raises:
            ValueError: for an empty sequence, something that is not a `PanelSpec`, or a repeated id.
        """
        checked = tuple(panels)
        if not checked:
            raise ValueError("FigureSpec needs at least one panel")
        for panel in checked:
            if not isinstance(panel, PanelSpec):
                raise ValueError(
                    f"FigureSpec panels must be PanelSpec values; got {type(panel).__name__}"
                )
        ids = [panel.id for panel in checked]
        shared = sorted({panel_id for panel_id in ids if ids.count(panel_id) > 1})
        if shared:
            raise ValueError(
                f"FigureSpec panel ids must be unique; {shared} appear more than once"
            )
        return checked

    @staticmethod
    def _checked_sources(sources: Mapping[str, Any]) -> Dict[str, DataRef]:
        """Return the sources as a dict, refusing an id that addresses nothing or a value that is not a `DataRef`.

        Args:
            sources: What the constructor was given.

        Returns:
            A fresh dict of the sources.

        Raises:
            ValueError: for a source id that is not a string or is empty, or a value
                that is not a `DataRef`.
        """
        checked = dict(sources)
        for source_id, ref in checked.items():
            _identifier("FigureSpec source", source_id)
            if not isinstance(ref, DataRef):
                raise ValueError(
                    f"source {source_id!r} must be a DataRef; got {type(ref).__name__}"
                )
        return checked

    def _check_panel_layers(self) -> None:
        """Refuse a panel that names a layer the figure does not have, or a 3-D panel missing a drape's surface.

        Raises:
            ValueError: naming the panel, the missing layers and the layers that exist; or, for a `Camera` panel,
                naming the draped layer and the surface it does not show. The tree already refuses a drape over a
                layer it lacks; a 3-D panel is held to the same rule over what it shows, because a renderer has
                nothing to drape the layer onto otherwise. A flat (`Viewport`) panel draws a draped layer flat,
                so it may show one without its surface.
        """
        for panel in self.panels:
            missing = [
                layer_id for layer_id in panel.layers if layer_id not in self.layers
            ]
            if missing:
                raise ValueError(
                    f"panel {panel.id!r} shows layers {missing} that are not in the figure; layers are "
                    f"{list(self.layers.ids)}"
                )
            if not isinstance(panel.view, Camera):
                continue
            for layer_id in panel.layers:
                surface = self.layers.get(layer_id).z_layer
                if surface is not None and surface not in panel.layers:
                    raise ValueError(
                        f"panel {panel.id!r} shows {layer_id!r}, draped over {surface!r}, without {surface!r}; a 3-D "
                        "panel needs the surface a layer takes its elevation from"
                    )

    @staticmethod
    def _checked_size(size: Any) -> Optional[Tuple[float, float]]:
        """Return the figure size as two floats, or ``None``.

        Args:
            size: What the constructor was given.

        Returns:
            ``(width, height)`` as floats, or ``None`` when unset.

        Raises:
            ValueError: for anything but two positive finite numbers — a string, a non-iterable, the wrong count,
                or a boolean, which counts as no number.
        """
        if size is None:
            return None
        if isinstance(size, (str, bytes)) or not hasattr(size, "__iter__"):
            raise ValueError(f"FigureSpec size must be (width, height); got {size!r}")
        # Numpy numbers count, as they do for every other number in the vocabulary: a size is often computed.
        dimensions = [positive_number(value) for value in size]
        if len(dimensions) == 2:
            width, height = dimensions
            if width is not None and height is not None:
                return width, height
        raise ValueError(
            f"FigureSpec size must be two positive finite numbers; got {size!r}"
        )

    @staticmethod
    def _check_sources(layer: LayerSpec, sources: Mapping[str, DataRef]) -> None:
        """Refuse a layer whose data or elevation names a source the figure does not have.

        Args:
            layer: The layer to check.
            sources: The figure's sources.

        Raises:
            ValueError: naming the layer, the missing source and the sources that exist.
        """
        wanted = [layer.source_id]
        if layer.z_source is not None and layer.z_layer is None:
            wanted.append(layer.z_source)
        for source_id in wanted:
            if source_id is not None and source_id not in sources:
                raise ValueError(
                    f"layer {layer.id!r} reads source {source_id!r}, which is not in the figure; sources are "
                    f"{sorted(sources)}"
                )

    def panel(self, panel_id: str) -> PanelSpec:
        """Return the panel with this id.

        Args:
            panel_id: The id to look up.

        Returns:
            The panel.

        Raises:
            KeyError: if no panel has that id, listing the ids that exist.

        Examples:
            - Look a panel up by id:
                ```python
                >>> from digitalearth.base.spec import FigureSpec, PanelSpec
                >>> FigureSpec(panels=(PanelSpec("a", title="Left"),)).panel("a").title
                'Left'

                ```
            - An id no panel has is named beside the ids that exist:
                ```python
                >>> from digitalearth.base.spec import FigureSpec, PanelSpec
                >>> FigureSpec(panels=(PanelSpec("a"),)).panel("b")
                Traceback (most recent call last):
                    ...
                KeyError: "no panel 'b' in this figure; panels are ['a']"

                ```
        """
        for panel in self.panels:
            if panel.id == panel_id:
                return panel
        raise KeyError(
            f"no panel {panel_id!r} in this figure; panels are {[panel.id for panel in self.panels]}"
        )

    def layers_of(self, panel_id: str) -> Tuple[LayerSpec, ...]:
        """Return the layers a panel shows, in draw order.

        Args:
            panel_id: The panel.

        Returns:
            The panel's layers, ordered as the figure's tree orders them — so reordering the tree reorders every
            panel that shows those layers.

        Raises:
            KeyError: if no panel has that id.

        Examples:
            - A panel's layers follow the tree's order, not the order the panel listed them in:
                ```python
                >>> from digitalearth.base.spec import FigureSpec, LayerSpec, LayerTree, PanelSpec
                >>> tree = LayerTree((LayerSpec("base", "tiles"), LayerSpec("roads", "lines")))
                >>> fig = FigureSpec(panels=(PanelSpec("p", layers=("roads", "base")),), layers=tree)
                >>> [layer.id for layer in fig.layers_of("p")]
                ['base', 'roads']

                ```
        """
        shown = set(self.panel(panel_id).layers)
        return tuple(layer for layer in self.layers if layer.id in shown)

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form of the whole figure.

        Returns:
            `schema_version` and `panels`, plus `sources`, `layers`, `size` and `title` when set. The free-form
            values — encoding constants, style properties, scale categories and schemes, selection axes — go
            through the JSON check as they are written, so a figure built from plain, finite Python numbers,
            strings, lists and dicts gives a dict `json.dumps` writes as is.

        Raises:
            TypeError: if a free-form value has no JSON form. A view's CRS is checked when the view is built, so
                it is not a reason this can fail.
            ValueError: if a source is an ``object:`` reference from :meth:`DataRef.to_object
                <digitalearth.base.spec.dataref.DataRef.to_object>`. It names an object in this process's memory,
                so a stored figure holding one would read back and then fail to open anywhere else. Building the
                figure in memory with one is fine; storing it is not.

        Examples:
            - The version is always written:
                ```python
                >>> from digitalearth.base.spec import FigureSpec, PanelSpec
                >>> FigureSpec(panels=(PanelSpec("main"),)).to_dict()
                {'schema_version': 1, 'panels': [{'id': 'main', 'viewport': {'crs': 3857}}]}

                ```
            - A figure survives a JSON round trip, with no renderer involved:
                ```python
                >>> import json
                >>> from digitalearth.base.spec import DataRef, FigureSpec, LayerSpec, LayerTree, PanelSpec, Viewport
                >>> fig = FigureSpec(
                ...     panels=(PanelSpec("main", Viewport(4326), layers=("dem",)),),
                ...     sources={"srtm": DataRef("data/dem.tif")},
                ...     layers=LayerTree((LayerSpec("dem", "raster", source_id="srtm"),)),
                ...     size=(8, 4),
                ... )
                >>> text = json.dumps(fig.to_dict(), allow_nan=False)
                >>> back = FigureSpec.from_dict(json.loads(text))
                >>> back == fig, back.size
                (True, (8.0, 4.0))

                ```
        """
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "panels": [panel.to_dict() for panel in self.panels],
        }
        if self.sources:
            for source_id, ref in self.sources.items():
                if ref.uri.startswith(f"{OBJECT_SCHEME}:"):
                    raise ValueError(
                        f"source {source_id!r} is {ref.uri!r}, which only resolves in the process that registered "
                        "it, so FigureSpec.to_dict cannot store it. Save the data and reference it by path or URL"
                    )
            out["sources"] = {
                plain_text(source_id): ref.to_dict()
                for source_id, ref in self.sources.items()
            }
        if len(self.layers):
            out["layers"] = self.layers.to_dict()
        if self.size is not None:
            out["size"] = list(self.size)
        if self.title is not None:
            out["title"] = plain_text(self.title)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FigureSpec":
        """Rebuild a figure from its dict form.

        The version is read first: a figure written by a newer schema is refused by name before any of its fields is
        interpreted, since a field whose meaning changed would otherwise be read the old way without a word.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The figure, validated as the constructor validates it.

        Raises:
            TypeError: if `data` is not a mapping, or a part is the wrong shape — a panel, the layer tree or
                `sources` that is not a mapping, or `panels` or `size` that is not a list — naming the field.
            ValueError: for a missing or unknown schema version, a missing panel list, an unknown key, a part its own
                `from_dict` refuses, or a figure the constructor refuses. An error of either type from a stored
                panel, source or layer tree names where the part sits — `panels[0]`, `sources['a']`, `layers` — and
                each nested read adds its own step, so the message is the path from the figure down to the broken
                entry.

        Examples:
            - A figure from a newer schema is refused before it is read:
                ```python
                >>> from digitalearth.base.spec import FigureSpec
                >>> FigureSpec.from_dict({"schema_version": 2, "panels": []})
                Traceback (most recent call last):
                    ...
                ValueError: FigureSpec.from_dict got schema_version 2; this version of digitalearth reads 1

                ```
            - A broken entry deep in the figure is named by its path:
                ```python
                >>> from digitalearth.base.spec import FigureSpec
                >>> stored = {"schema_version": 1, "panels": [{"id": "p", "viewport": {"crs": True}}]}
                >>> FigureSpec.from_dict(stored)  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: FigureSpec.from_dict panels[0]: PanelSpec.from_dict viewport: Viewport needs a display ...

                ```
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                f"FigureSpec.from_dict needs a mapping; got {type(data).__name__}"
            )
        version = require("FigureSpec", data, "schema_version")
        if not _is_known_version(version):
            raise ValueError(
                f"FigureSpec.from_dict got schema_version {version!r}; this version of digitalearth reads "
                f"{SCHEMA_VERSION}"
            )
        refuse_unknown(
            "FigureSpec",
            data,
            ("schema_version", "panels", "sources", "layers", "size", "title"),
        )
        size = data.get("size")
        layers = data.get("layers")
        return cls(
            panels=tuple(
                read_entry("FigureSpec", f"panels[{index}]", PanelSpec.from_dict, panel)
                for index, panel in enumerate(
                    as_list(
                        "FigureSpec", "panels", require("FigureSpec", data, "panels")
                    )
                )
            ),
            sources={
                source_id: read_entry(
                    "FigureSpec", f"sources[{source_id!r}]", DataRef.from_dict, ref
                )
                for source_id, ref in as_mapping(
                    "FigureSpec", "sources", data.get("sources", {})
                ).items()
            },
            layers=LayerTree()
            if layers is None
            else read_entry("FigureSpec", "layers", LayerTree.from_dict, layers),
            size=None if size is None else as_list("FigureSpec", "size", size),
            title=data.get("title"),
            schema_version=version,
        )
