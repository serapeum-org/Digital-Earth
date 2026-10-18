"""The interactive tier's colour key: ``colorbar()`` and ``legend()`` (order 24, #261).

Pulled out of ``decoration`` so the key that *explains* a layer's colour encoding lives apart from the
basemap/features/text decoration that draws furniture. The methods record a ``Guide`` on the layer's own
``Symbology`` and apply the HoloViews options from that record, so the key travels with its layer.
"""

import warnings
from collections.abc import Sequence
from dataclasses import replace as with_fields
from typing import TYPE_CHECKING, Any, Self

from digitalearth.base.spec import Encoding, Guide, LayerSpec
from digitalearth.interactive.core.base import _require_holoviz
from digitalearth.interactive.render.style_fold import (
    GUIDE_KIND_KEY,
    GUIDE_LABELS_KEY,
    fold_guide,
    split_guide_options,
)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.core.base import InteractiveMapBase as _MixinBase
else:
    _MixinBase = object


class LegendMixin(_MixinBase):
    """The ``colorbar``/``legend`` colour-key verbs, composed into ``InteractiveMap``."""

    def _colour_binding(self, layer_id: str) -> Encoding | None:
        """Return what drives one layer's colour, when something varies it.

        Args:
            layer_id: A layer this map describes.

        Returns:
            Its ``color`` :class:`~digitalearth.base.spec.encoding.Encoding`, or `None` both when the layer
            publishes none and when the one it publishes is a **constant**. A flat colour is not something a
            key can explain — there are no values to label and no scale to sample — and the two answer as one
            here because the question a guide asks is whether there is anything to explain at all. The
            distinction matters: a caller's own ``color="#f00"`` is lifted onto the channel as a constant by
            :func:`~digitalearth.interactive.render.style_fold.portable_encodings`, so "publishes a colour encoding"
            on its own would have let a key onto a layer painted one colour.
        """
        encoding = self._layer_tree.get(layer_id).symbology.encoding("color")
        return None if encoding is None or encoding.is_constant else encoding

    def _guided_layer(self, method: str, layer_id: str | None) -> str:
        """Resolve which layer a colour key belongs to, and refuse a layer that can carry none.

        Args:
            method: The public method asking, named in every refusal.
            layer_id: The caller's own ``layer_id``, or `None` for the default.

        Returns:
            The id of the layer the key describes.

            ``None`` means **the most recent layer that carries a colour encoding**. The care
            :meth:`~digitalearth.interactive.core.base.InteractiveMapBase._last_layer_index` took is kept — the
            layer the last builder call added, not the one drawn on top, so a raster added after a coastline
            is still the one this reaches, and an underlay added after data does not take it over — and the
            rule is tightened by exactly one thing: a layer with no colour encoding cannot take it over
            either. So `field(dem).coastlines().colorbar()` titles the raster, where the old `show=` toggle
            wrote a `colorbar` option onto the coastline.

            Where the layer added last carries no colour, the search falls back to the **topmost coloured
            layer in draw order**. Draw order is what the tree keeps and the only order it can be asked
            for; it coincides with the order the layers were added only while none of them has been moved,
            and `move_layer` parts the two. Measured: after `field(dem).field(dem).coastlines()` and
            `move_layer("raster-2", 0)` the draw order is `['raster-2', 'raster-1', 'coastlines-1']` and
            `colorbar()` keys `raster-1` — the topmost coloured layer — not the last-added `raster-2`.

        Raises:
            KeyError: when `layer_id` names no layer on this map, naming the ids that are.
            ValueError: when the named layer's colour is not driven by anything — a flat fill, a basemap, a
                coastline — and, for `None`, when no layer on the map has a colour a key could explain. An
                empty box would be worse than an error, which is the position the web tier's `legend` takes
                for the same question.
        """
        if layer_id is not None:
            self._require_layer(layer_id)
            if self._colour_binding(layer_id) is None:
                raise ValueError(
                    f"{method}(): layer {layer_id!r} draws no colour that varies with its data, so there is "
                    f"nothing for a colour key to explain; the layers that do are {self._guidable()}"
                )
            return layer_id
        held = self._last_layer_id
        # `held in self._layer_tree` before it is read, the guard `_note_last_layer` already applies: the
        # pointer is resolved by looking the id up, and a stale one raised `KeyError` out of the tree where
        # these methods document a refusal naming themselves (review N8).
        if (
            held is not None
            and held in self._layer_tree
            and self._colour_binding(held) is not None
        ):
            return held
        guidable = self._guidable()
        if not guidable:
            raise ValueError(
                f"{method}() needs a layer whose colour varies with its data — add a field(), a contours() "
                "or a builder given column=, or name one with layer_id="
            )
        return guidable[-1]

    def _guidable(self) -> list:
        """Return the ids of the layers a colour key could explain, in draw order.

        Returns:
            One id per layer whose colour is driven by a field, bottom of the draw order first. Read by
            :meth:`_guided_layer` both to pick the default and to name the alternatives in its refusals, so
            the answer and the message cannot describe different sets.
        """
        return [
            layer_id
            for layer_id in self._layer_tree.ids
            if self._colour_binding(layer_id) is not None
        ]

    def _record_guide(
        self,
        layer_id: str,
        guide: Guide,
        *,
        kind: str,
        labels: Sequence[str] | None = None,
    ) -> LayerSpec:
        """Attach `guide` to one layer's colour encoding, and hand back the layer as it now reads.

        **The record is the point of order 24.** A colour key used to be a `show=` option written onto the
        element of whichever layer was added last and then forgotten: it could not move with its layer, could
        not go away with it, and was in no figure. Attached to the layer's own
        :class:`~digitalearth.base.spec.style.Symbology` it does all three for free, because the symbology is
        already what travels — `remove_layer` takes the guide with the layer, `set_visible` leaves it on a
        layer that comes back, and `to_dict()`/`from_dict()` carry it.

        Written **straight into the tree** rather than through
        :meth:`~digitalearth.interactive.core.base.InteractiveMapBase.replace_layer`. A guide is description and
        nothing else — the drawn key is the `.opts()` the caller's method applies next — so there is no
        difference for a renderer to reconcile, and going through `_change` would restyle the layer, redraw
        it (re-reading its source), and hand back a *new* element: the styles
        :attr:`~digitalearth.interactive.core.base.InteractiveMapBase._styles` files by ``id()`` and any stream a
        `hover()` attached are both keyed on the object this would have replaced. The panel `figure_spec`
        reports is derived from the tree on every read, so it follows this write without being told.

        **What is recorded beside the guide is what the redraw needs.** A `Guide` says whether, what the key
        is called and where — never which furniture, because the tiers that draw one kind of key need no such
        field. This tier draws two, so the kind and a caller's own row labels go into the symbology's
        ``props`` (:data:`~digitalearth.interactive.render.style_fold.GUIDE_KIND_KEY`,
        :data:`~digitalearth.interactive.render.style_fold.GUIDE_LABELS_KEY`), which is what lets
        :func:`~digitalearth.interactive.render.style_fold.fold_guide` build the same options again when the layer
        is drawn again — the static tier records the same two under the same names.

        Args:
            layer_id: The layer to explain. Already resolved and checked by :meth:`_guided_layer`.
            guide: What to say about its colour.
            kind: Which key it asks for — ``"colorbar"`` or ``"legend"``
                (:data:`~digitalearth.interactive.render.style_fold.GUIDE_KINDS`).
            labels: The caller's own row labels, or `None` to leave the derived one. `None` also *clears*
                labels an earlier call recorded, so one call's override does not outlive it.

        Returns:
            The layer as the tree now holds it, so the options applied next are folded **from the record**
            and cannot say something the figure does not.
        """
        layer = self._layer_tree.get(layer_id)
        props = dict(layer.symbology.props)
        props[GUIDE_KIND_KEY] = kind
        if labels is None:
            props.pop(GUIDE_LABELS_KEY, None)
        else:
            props[GUIDE_LABELS_KEY] = tuple(str(label) for label in labels)
        symbology = with_fields(layer.symbology, props=props).with_guide(guide)
        described = with_fields(layer, symbology=symbology)
        self._layer_tree = self._layer_tree.replace(described)
        return self._layer_tree.get(layer_id)

    def _draw_guide(self, layer_id: str, method: str, **opts: Any) -> None:
        """Apply one layer's colour-key options to the element it is drawn as.

        Args:
            layer_id: The layer whose element takes the options.
            method: The public method asking, named in the warning below.
            **opts: The HoloViews options the recorded guide resolved to.

        Warns:
            UserWarning: for an option this layer's element does not accept. Both keys are real: an
                ``hv.Image`` takes ``colorbar``, ``clabel`` and ``show_legend`` but **not** ``legend_opts``
                or ``legend_labels`` (measured against the registered Bokeh options), because a Bokeh legend
                is built from an overlay's labelled members and a raster contributes no label to name. Saying
                so is the choice :func:`~digitalearth.interactive.render.renderer._show` already made for the same
                shape of question — a caller who asked for something the engine cannot draw hears about it
                rather than watching it vanish.
        """
        index = self._layer_tree.ids.index(layer_id)
        element = self.layers[index]
        taken, refused = split_guide_options(element, opts)
        if refused:
            warnings.warn(
                f"{method}(): layer {layer_id!r} is drawn as a {type(element).__name__}, which Bokeh gives "
                f"no {', '.join(refused)} to draw; the rest of the key is applied and the guide is recorded",
                UserWarning,
                stacklevel=3,
            )
        if taken:
            self.layers[index] = element.opts(**taken, backend="bokeh")

    def colorbar(
        self,
        layer_id: str | None = None,
        *,
        label: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Show the continuous colour key of a layer (the Core name, #261, order 24).

        A guide on the layer's own colour encoding rather than a ``show=`` toggle on whichever layer was
        added last: it is recorded first and the HoloViews options are applied **from the record**, so the key
        moves with its layer, goes away with it, and is carried by the figure.

        Bokeh draws **one** colorbar per plot, whatever the opts say. Measured: two guided rasters overlaid,
        each with ``colorbar=True`` and its own ``clabel``, render a single ``ColorBar`` — the one belonging
        to the layer lowest in draw order. So guiding a second layer does not stack a second bar; it decides
        which layer's bar is the one drawn. That is the same one-key-per-panel limit the web tier states from
        the other direction ("a second legend replaces the first"), and it is the engine's, not this tier's.

        Args:
            layer_id: Which layer's key to show. `None` (default) takes the most recent layer whose colour
                varies with its data — see :meth:`_guided_layer` for exactly what "most recent" means, which
                is the care the old toggle took plus one tightening.
            label: What to call the key — the variable and its units, usually. It reaches the drawn bar as
                HoloViews' ``clabel``, and is recorded as the guide's title. `None` leaves whatever the
                builder resolved (``field`` takes its own ``clabel=``, defaulting to the variable's units).
            visible: Whether the key is drawn. `False` records the guide switched **off** and applies
                ``colorbar=False``, so a caller passing a flag through does not have to branch — and the
                layer is resolved and checked before the flag is read, so one spelling of a call is not valid
                only half the time (the web tier's review L7).

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            KeyError: when `layer_id` names no layer on this map.
            ValueError: when the named layer's colour varies with nothing, or — for `None` — when no layer's
                does. Also for a ``label`` that is not a non-empty string, which
                :class:`~digitalearth.base.spec.encoding.Guide` refuses.
            ImportError: when the ``interactive`` extra is not installed.

        Examples:
            - Title the raster's colour key:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).colorbar(label="Flow (m³/s)")  # doctest: +SKIP
                >>> m.get_layer(m.layer_ids[0]).symbology.guide().title         # doctest: +SKIP
                'Flow (m³/s)'

                ```
            - Drop it again, by name:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem, name="flow")                # doctest: +SKIP
                >>> m.colorbar("flow", visible=False).save("m.html").name       # doctest: +SKIP
                'm.html'

                ```

        See Also:
            digitalearth.base.spec.style.Symbology.with_guide: where the guide is attached.
        """
        # Called for its actionable ImportError before anything else: every option below is applied for the
        # Bokeh backend, and `_require_holoviz` is what registers it.
        _require_holoviz()
        target = self._guided_layer("colorbar", layer_id)
        described = self._record_guide(
            target, Guide(show=bool(visible), title=label), kind="colorbar"
        )
        self._draw_guide(
            target, "colorbar", **fold_guide(described.symbology, described.label)
        )
        return self

    def legend(
        self,
        layer_id: str | None = None,
        *,
        title: str | None = None,
        labels: Sequence[str] | None = None,
        visible: bool = True,
    ) -> Self:
        """Show the keyed colour list of a layer (the Core name, #261, order 24).

        The same guide :meth:`colorbar` records, drawn as Bokeh's keyed box instead of a continuous bar. Both
        methods explain one layer's ``color`` encoding and so write one guide: calling both on a layer leaves
        the last title and the last ``visible``, and draws both keys.

        Bokeh builds **one** legend per plot, from the labelled members of the overlay, so ``show_legend``
        decides whether this layer *contributes a row* rather than whether it gets a box of its own —
        measured: a single element with ``show_legend=True`` renders no legend at all, because there is no
        overlay to build one from. A layer's row is captioned by its ``name=``, which is what ``labels``
        renames.

        Args:
            layer_id: Which layer's key to show. `None` (default) is the most recent layer whose colour
                varies with its data, exactly as :meth:`colorbar` resolves it.
            title: Heading above the key. It reaches the drawn box through Bokeh's own ``Legend.title``, and
                is recorded as the guide's title.
            labels: Explicit row labels, replacing the derived one. A layer contributes **one** row to a
                Bokeh legend — its own label — so this takes one entry; a count that does not match is
                refused rather than zipped, which would drop the difference and leave rows out of the key
                (the web tier's rule, for the same reason).
            visible: Whether the layer contributes to the key. `False` records the guide switched off and
                applies ``show_legend=False``. Checked after the layer is resolved, never before.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            KeyError: when `layer_id` names no layer on this map.
            ValueError: when the named layer's colour varies with nothing, or — for `None` — when no layer's
                does; when ``labels`` does not hold one entry per row; and for a ``title`` that is not a
                non-empty string.
            ImportError: when the ``interactive`` extra is not installed.

        Examples:
            - Key a contour layer, and title the box:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().contours(dem).legend(title="Contours")  # doctest: +SKIP
                >>> m.get_layer(m.layer_ids[0]).symbology.guide().show          # doctest: +SKIP
                True

                ```
            - Or take it off again:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().contours(dem).legend(visible=False)    # doctest: +SKIP
                >>> m.save("m.html").name                                       # doctest: +SKIP
                'm.html'

                ```

        See Also:
            digitalearth.base.spec.style.Symbology.with_guide: where the guide is attached.
        """
        _require_holoviz()
        target = self._guided_layer("legend", layer_id)
        rows = None if labels is None else list(labels)
        derived = [self._layer_tree.get(target).label]
        if rows is not None and len(rows) != len(derived):
            # Refused before the guide is recorded, so a rejected call leaves the description untouched.
            raise ValueError(
                f"legend(labels=...) has {len(rows)} entries and layer {target!r} contributes "
                f"{len(derived)} row to a Bokeh legend ({derived[0]!r}); zip would drop the difference and "
                "leave rows out of the key"
            )
        described = self._record_guide(
            target, Guide(show=bool(visible), title=title), kind="legend", labels=rows
        )
        self._draw_guide(
            target, "legend", **fold_guide(described.symbology, described.label)
        )
        return self
