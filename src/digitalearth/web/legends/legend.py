"""The web tier's colour key: ``legend()`` (order 24, #261).

Pulled out of ``decoration`` so the key that *explains* a layer's colour encoding lives apart from the
basemap/controls/measure decoration. The low-level guide renderers (``refresh_legend_panel`` and friends)
stay in ``decoration`` as module functions -- other modules import them from there; this mixin calls them.
"""

from dataclasses import replace as _with_fields
from typing import TYPE_CHECKING, Any, Self

from digitalearth.base.spec import Guide, LayerSpec, Symbology
from digitalearth.web.core.base import _require_maplibre
from digitalearth.web.decoration.decoration import (
    DEFAULT_KEY_ANCHOR,
    LEGEND_LABELS,
    _check_position,
    _legend_panel,
    drawn_guide_layer,
    guided_layer_ids,
    keyable_layer_ids,
    refresh_legend_panel,
)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.core.base import WebMapBase as _MixinBase
else:
    _MixinBase = object


class LegendMixin(_MixinBase):
    """The ``legend`` colour-key verb, composed into ``WebMap``."""

    def legend(
        self,
        *,
        layer_id: str | None = None,
        title: str | None = None,
        position: str = DEFAULT_KEY_ANCHOR,
        labels: list | None = None,
        visible: bool = True,
    ) -> Self:
        """Add a key for a classified layer (recipe W2).

        A thematic map is unreadable without one, and until this the tier drew the classes and left the
        caller to build a key out of band from ``last_breaks``. The classification records what it actually
        coloured with (:attr:`~digitalearth.web.core.base.WebMapBase.last_legend`), so the key shows the rendered
        colours rather than a second guess at them.

        The three shapes a classification can take are all handled: a **categorical** one gets a swatch per
        category, a **graduated** one a swatch per class with its range, and a **continuous** ramp a
        gradient bar with its end values.

        **The key belongs to the layer, not to the map** (DE-48, order 24). What this call records is a
        :class:`~digitalearth.base.spec.encoding.Guide` on the layer's ``color``
        :class:`~digitalearth.base.spec.encoding.Encoding` — ``show``, the heading, and the corner — and the
        panel is **derived** from the guides the live layers carry, every time the layers change. So the key
        moves, hides and disappears with the layer it describes: remove the layer and the key goes, hide it
        and the key goes, show it again and the key comes back. It also travels: a guide survives
        ``FigureSpec`` being written and read back, which a panel built once never did.

        Args:
            layer_id: Which layer's classes to describe. `None` takes the topmost **visible** layer that
                published a colour scale — a classified ``choropleth``/``points``, or a raster band's ramp,
                since a band carries a colour encoding of its own — and so the most recent one a viewer can
                see, since a key drawn for nothing on screen is the thing this call exists to avoid. It
                falls back to the topmost such layer of any visibility when none is visible. Under
                ``visible=False`` it instead takes the layer whose key is drawn, so "no key" takes off the
                key there is.
            title: Heading above the key. ``None`` uses the classified column's name, followed by the
                units in parentheses when :func:`~digitalearth.base.autostyle.auto_style` supplied them
                for the raster the classification came from. A title given here always wins, and a unit
                is never guessed: without one the heading is the bare column name, as it always was.
            position: One of the four MapLibre corners. Recorded as the guide's ``anchor``, which takes the
                same four spellings (:data:`~digitalearth.base.registry.FURNITURE_ANCHORS` and
                :data:`~digitalearth.base.controls.CONTROL_POSITIONS` are **equal** tuples — two objects,
                one vocabulary), so nothing is translated between them.
            labels: Explicit row labels, replacing the derived ones — for units, or for renaming
                categories. Ignored for a continuous ramp, which has no rows. Recorded on the layer beside
                its guide, so a key rebuilt after another layer is removed still carries them.
            visible: `False` records the guide with ``show=False`` — the layer says outright that its colour
                is explained by nothing — and draws no key, so a caller passing a flag through does not have
                to branch. **Every other argument is checked before this one is read**, whether or not any
                layer has been keyed yet, so no argument is valid only half the time and none is valid only
                in a particular call order. The one refusal the flag does decide is "this map has nothing to
                describe", which is a fact about the map rather than about the call.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``position`` is not one of the four legal MapLibre corners, when `title` is not
                a non-empty string, when `layer_id` names a layer that carries no classification, when no
                classified layer has been added at all — there is nothing to build a key from, and an empty
                box would be worse than an error — or when `labels` has a different number of entries from
                the classification. Only the "nothing classified" one waits on ``visible``.
            KeyError: when `layer_id` names no layer on this map.
            TypeError: when `labels` is not a sequence.

        Note:
            **This tier draws one key, and the four tiers differ here.** ``_panels`` holds one entry per
            kind, so a second legend replaces the first rather than stacking an identical box in the same
            corner. Layers may each carry a guide all the same, and when several do **the topmost visible
            one wins** — draw order decides, not call order. So keying a lower layer after a higher one
            leaves the higher one's key drawn; take the higher one's off with
            ``legend(layer_id=<higher>, visible=False)`` first. The static tier has no such limit
            (matplotlib takes any number of colorbars) and neither does the interactive tier (its guides are
            per-layer options); the 3-D tier has one scalar-bar slot per title.

            A guide on a **hidden** layer is recorded and draws nothing until the layer is shown, which is
            the same rule stated from the other side: the key is drawn while the layer is.

        Examples:
            - A choropleth and its key:
                ```python
                >>> from digitalearth.web import WebMap                        # doctest: +SKIP
                >>> (                                                          # doctest: +SKIP
                ...     WebMap().basemap().choropleth(gdf, column="pop").legend()
                ... )

                ```

        See Also:
            digitalearth.web.builders.vector.VectorMixin.choropleth: the builder whose classes this describes.
            digitalearth.web.decoration.decoration.refresh_legend_panel: derives the panel from the recorded guides.
        """
        return self._record_key(
            layer_id,
            title=title,
            position=position,
            labels=labels,
            visible=visible,
            caller="legend()",
        )

    def _record_key(
        self,
        layer_id: str | None,
        *,
        title: str | None,
        position: str = DEFAULT_KEY_ANCHOR,
        labels: list | None,
        visible: bool,
        caller: str,
    ) -> Self:
        """Record a colour key on a layer, told which public spelling the caller reached it through.

        :meth:`legend` and :meth:`~digitalearth.web.core.base.WebMapBase.colorbar` are one mechanism under two
        names, so the body lives here and each name passes its own. Without that, the one refusal a
        `colorbar()` caller can actually hit — "nothing to describe" — named `legend()` and told them to add
        a `column=`, about a method they never called; this tier names the caller in every other refusal it
        gives (``WebMap.field()``, ``WebMap.extrusion()``).

        **One mechanism means one validation site.** Both public names being pure forwards onto this body is
        what lets `_check_key_arguments` be the single place a caller argument is refused from, so the two
        spellings cannot drift to two answers for the same malformed call (review H2).

        Args:
            layer_id: The caller's chosen layer, or `None` to resolve one.
            title: The heading, or `None` for the classified column's name.
            position: The corner, one of :data:`~digitalearth.base.controls.CONTROL_POSITIONS`.
            labels: Explicit row labels, or `None`.
            visible: Whether a key is drawn; `False` records that the colour is explained by nothing.
            caller: How to spell the public method in a refusal — ``"legend()"`` or ``"colorbar()"``.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: as :meth:`legend` documents.
            KeyError: as :meth:`legend` documents.
        """
        # EVERY caller argument is checked here, in one call, above the first read of `visible` — and that
        # ordering is pinned structurally, not case by case. It came back three times otherwise: the corner
        # (review L7), then the layer (review M1/M2), then `title` and `labels` (review H2), each fix
        # hoisting the one keyword that had just been caught and leaving the next one under the flag. The
        # *class* of defect is "a caller argument read below the flag", so there is now exactly one place a
        # caller argument may be read from — see `_check_key_arguments` — and
        # `tests/web/test_web_legend.py::TestNoArgumentCanBeAddedBelowTheFlag` reads this body's source to
        # refuse a fourth recurrence: a keyword added to either public spelling and not handed to the check
        # above this line reddens with no case written for it.
        self._check_key_arguments(
            layer_id, title=title, position=position, labels=labels
        )
        target = self._guide_target(layer_id, visible=bool(visible), caller=caller)
        if target is None:
            return self
        _require_maplibre()
        guide = Guide(show=bool(visible), title=title, anchor=position)
        # Checked a second time, now against the layer the record actually lands on: `visible=False`
        # resolves to the layer whose key is *drawn*, which need not be the one the check above could see,
        # and `labels` has to fit the classification it will be drawn beside. Built and thrown away, so a
        # refusal leaves the map exactly as it was rather than leaving behind a guide whose key every later
        # rebuild would fail to draw. Pinned by
        # `tests/web/test_web_legend.py::…::test_a_refusal_leaves_the_map_exactly_as_it_was`, so moving the
        # record above this line reddens rather than passing.
        _legend_panel(self._legend_of(target), guide, labels)
        self._attach_guide(target, guide, labels)
        # Derived, not written: the rule for which of several guides is drawn lives in one place, so this
        # call and a later removal cannot disagree about whose key is on screen.
        refresh_legend_panel(self)
        return self

    def _check_key_arguments(
        self,
        layer_id: str | None,
        *,
        title: str | None,
        position: str,
        labels: list | None,
    ) -> None:
        """Refuse a malformed key description, before the `visible` flag has been read at all.

        **The one place a caller argument to :meth:`legend` /
        :meth:`~digitalearth.web.core.base.WebMapBase.colorbar` may be checked from**, which is what makes the
        rule "no argument is valid only half the time" a property of the method rather than a property of
        the three keywords somebody remembered. The flag is deliberately not a parameter here: it cannot be
        consulted, so it cannot skip anything.

        The split is between two different questions. *Is this a well-formed description?* — the corner, the
        named layer, the heading, the row labels — is a fact about the call and is answered here, whatever
        the flag says. *Does this map have anything to describe?* is a fact about the map, and stays
        :meth:`_guide_target`'s ``visible=True``-only refusal, because the documented point of the flag is
        that ``WebMap().colorbar(visible=False)`` is an answer and not a crash.

        Args:
            layer_id: The caller's chosen layer, or `None`. When given it is looked up and required to
                carry a classification. When `None` the topmost keyable layer stands in, purely so
                `labels` can be counted against a real classification on a map nobody has keyed yet —
                the state the three previous fixes all left open.
            title: The heading, or `None`. Checked by building the
                :class:`~digitalearth.base.spec.encoding.Guide` that will hold it.
            position: The corner, checked against :data:`~digitalearth.base.controls.CONTROL_POSITIONS`.
            labels: Explicit row labels, or `None`. Its *shape* is checked unconditionally — a
                non-sequence is a ``TypeError`` here as it was where the panel used to build it — and its
                *length* whenever there is a classification to count against.

        Raises:
            ValueError: when `position` is not one of the four legal corners, when `layer_id` names a layer
                that carries no classification, when `title` is not a non-empty string, or when `labels`
                has a different number of entries from the classification.
            KeyError: when `layer_id` names no layer on this map.
            TypeError: when `labels` is not a sequence.
        """
        _check_position(position)
        if layer_id is not None:
            # KeyError by name, then ValueError for no classification — the same two refusals
            # `_guide_target` gives a named layer, reached here so neither waits on the flag.
            self._legend_of(layer_id)
            described: str | None = layer_id
        else:
            keyable = keyable_layer_ids(self)
            described = keyable[-1] if keyable else None
        guide = Guide(show=True, title=title, anchor=position)
        rows = None if labels is None else list(labels)
        if described is not None:
            _legend_panel(self._legend_of(described), guide, rows)

    def _guide_target(
        self, layer_id: str | None, *, visible: bool, caller: str = "legend()"
    ) -> str | None:
        """Return the layer a colour key should describe.

        Args:
            layer_id: The caller's choice, or `None` to resolve one from the map.
            visible: Which question a `None` `layer_id` asks. `True` — asking for a key — looks for a
                classified layer to put one on, and a map with none is an error. `False` — asking for no
                key — looks for the key the map already has, and a map with none answers `None` rather than
                raising, which is what keeps ``WebMap().colorbar(visible=False)`` from being a crash on an
                empty map.
            caller: How to spell the public method in the refusal — ``"legend()"`` or ``"colorbar()"``.
                This is the one refusal both spellings can reach, so naming the method that wrote it sent a
                `colorbar()` caller to fix a `legend()` call.

        Returns:
            The layer id, or `None` when nothing qualifies and nothing was required.

        Raises:
            KeyError: when `layer_id` names no layer on this map.
            ValueError: when `layer_id` names a layer that was not drawn with a classification, or when a
                key was asked for and no layer on the map is classified.

        Note:
            **Nothing a caller passed is validated here.** Every caller argument — the corner, the named
            layer, the heading and the row labels — is refused by `_check_key_arguments` before this is
            reached, so no check can be reached through a flag value (review H2). A named layer is looked
            up again below all the same, so this stays answerable on its own; what it must never become is
            the *only* place a caller argument is looked at, which is how `legend(layer_id="nope",
            visible=False)` came to be accepted while `legend(layer_id="nope")` raised. The "nothing on
            this map is classified" question is not a caller argument, so it stays a ``visible=True``-only
            refusal — the documented point of the flag is that it can be passed through without branching.

            **The two flag values resolve differently on purpose, and the flag is not a third refusal
            rule.** Asking for a key and taking one off are different questions of the map: one wants a
            layer to explain, the other wants the key that is already there. One resolution served both,
            `keyable[-1]`, so `colorbar(visible=False)` on a map where a lower layer was keyed wrote
            ``show=False`` onto the topmost *classified* layer — one nobody had keyed — and left the drawn
            key exactly where it was (review M3).

            **Visibility is a preference on both branches, never a refusal.** Asking for a key prefers the
            topmost **visible** keyable layer — one that published a colour scale, which since order 24
            includes a raster band's ramp and not only a classified column (review L3) — because an
            unnamed `colorbar()` means "key what is on the
            map": visibility was consulted nowhere, so a hidden topmost layer took the guide and the call
            drew nothing at all (review L6). Taking one off prefers the key that is drawn for the matching
            reason. Neither *requires* it — a key recorded on a hidden layer draws nothing but is still a
            key the caller asked for, so each branch falls back to the topmost candidate of any visibility:
            otherwise keying a map built hidden would raise, and taking a key off a hidden layer would be a
            no-op the next `set_visible` undid. A hidden layer named outright is keyed as it always was.
        """
        if layer_id is not None:
            self._legend_of(
                layer_id
            )  # KeyError by name, then ValueError for no classification
            return layer_id
        if not visible:
            drawn = drawn_guide_layer(self)
            if drawn is not None:
                return drawn
            asked = guided_layer_ids(self)
            return asked[-1] if asked else None
        keyed = keyable_layer_ids(self)
        shown = [
            candidate for candidate in keyed if self._layer_tree.is_visible(candidate)
        ]
        if shown:
            return shown[-1]
        if keyed:
            return keyed[-1]
        raise ValueError(
            f"{caller} has nothing to describe: no classified layer has been added yet. Add a "
            "choropleth (or any builder given column=...) first."
        )

    def _attach_guide(self, layer_id: str, guide: Guide, labels: list | None) -> None:
        """Record on one layer that its colour is explained, and how.

        Args:
            layer_id: The layer to write on. It carries a field-driven ``color`` encoding — `_guide_target`
                has already established that — which is what a guide can be attached to.
            guide: What to say about the channel.
            labels: Explicit row labels, or `None` to draw the derived ones. `None` **removes** a previous
                override rather than keeping it, so two `legend()` calls do not silently compound.

        Note:
            The tree is written directly rather than through :meth:`~digitalearth.web.core.base.WebMapBase._change`.
            A guide changes no paint, so there is nothing for the renderer to draw again; routing it through
            the reconcile would rebuild the MapLibre layer of every classified layer a caller keys.
            :meth:`~digitalearth.web.core.base.WebMapBase._index_layer` writes the tree the same way, and
            :meth:`~digitalearth.web.core.base.WebMapBase._figure_with` derives the figure from it, so the guide
            reaches `figure_spec` with no further step.
        """
        layer = self._layer_tree.get(layer_id)
        symbology = layer.symbology.with_guide(guide)
        props = {
            key: value for key, value in symbology.props.items() if key != LEGEND_LABELS
        }
        if labels is not None:
            props[LEGEND_LABELS] = tuple(labels)
        self._layer_tree = self._layer_tree.replace(
            _with_fields(
                layer,
                symbology=Symbology(encodings=dict(symbology.encodings), props=props),
            )
        )
