"""What the static tier can read off a drawn artist, and what it refuses to guess (order 24).

`tests/static/test_static_guides.py` covers the *API* — that a key follows its layer, hides with it and goes
away with it. This covers the layer underneath: the two readings the tier takes off a matplotlib artist to
decide whether a colour key is possible at all, and the three paths a key takes once it is.

The readings matter because they are the only thing standing between "this layer can be keyed" and a bar
labelled with numbers the picture never drew. :func:`~digitalearth.static.guides.drawn_scale` has **four**
separate ways to answer "no scale", and a wrong `None` from any of them silently downgrades a keyable layer
to an unkeyable one, while a wrong `Scale` labels the key with invented limits. Neither shows up as an
exception — both draw a picture.

The **anchored** paths are here for a different reason. This tier never sets `Guide.anchor` itself: its
`colorbar`/`legend` take no placement keyword, because the Core shape has none. An anchor can only arrive on
a figure built somewhere else and read back — which is the portability the whole order rests on, and which
nothing exercised. The same line is untested on the 3-D tier for the same reason.
"""

from math import inf, nan
from typing import Any

import pytest
from matplotlib.colors import BoundaryNorm, Normalize

from digitalearth.base.spec import Encoding, Scale, Symbology
from digitalearth.static import Map
from digitalearth.static.guides import (
    GUIDE_KIND_KEY,
    _swatches,
    drawn_scale,
    guide_kind,
)


class _Handle:
    """A legend handle that may or may not answer `get_facecolor`.

    Stands in for a matplotlib artist because the two shapes this exercises — a handle with no face colour at
    all, and a handle count that disagrees with the label count — are both states a real legend reaches only
    through an artist type the tier does not draw. Building one by hand says what is being tested; coaxing
    matplotlib into producing one would say only that it can be coaxed.
    """

    def __init__(self, face: str | None) -> None:
        """Store the face colour this handle reports, or `None` to report none at all.

        Args:
            face: A matplotlib colour, or `None` for a handle with no `get_facecolor` attribute.
        """
        if face is not None:
            self.get_facecolor = lambda: face  # noqa: E731 - a stub, not a method


class _Legend:
    """The two accessors :func:`~digitalearth.static.guides._swatches` reads a drawn legend through.

    Args:
        texts: The label strings the legend holds.
        handles: The handles it holds, one per swatch.
    """

    def __init__(self, texts: list[str], handles: list[Any]) -> None:
        """Store the texts and handles this stand-in reports."""
        self._texts = [
            type("T", (), {"get_text": staticmethod(lambda t=t: t)})() for t in texts
        ]
        self.legend_handles = handles

    def get_texts(self) -> list[Any]:
        """Return the label objects, as matplotlib's `Legend.get_texts` does.

        Returns:
            One object per label, each answering `get_text()`.
        """
        return self._texts


class TestScaleOf:
    """`drawn_scale` — the scale a layer publishes, read off the artist's own norm."""

    def test_a_classified_norm_publishes_its_edges(self):
        """A `BoundaryNorm`'s boundaries become a classified scale carrying those very edges.

        Test scenario:
            This is the reading that makes a graduated legend possible. The edges are taken from the norm the
            artist draws through, so the rows cannot disagree with the picture.
        """
        scale = drawn_scale(
            type("A", (), {"norm": BoundaryNorm([0.0, 5.0, 10.0], 2)})()
        )
        assert scale is not None, (
            "a BoundaryNorm carries class edges and must publish a scale"
        )
        assert scale.breaks == (0.0, 5.0, 10.0), (
            f"edges not published verbatim: {scale.breaks}"
        )
        assert scale.is_classified, (
            "a scale built from boundaries must report itself classified"
        )

    def test_a_continuous_norm_publishes_its_limits(self):
        """A plain `Normalize` publishes a continuous scale over vmin/vmax.

        Test scenario:
            The ordinary raster case — no classes, so the key is a ramp between the two limits the artist
            normalises through.
        """
        scale = drawn_scale(type("A", (), {"norm": Normalize(vmin=-3.0, vmax=7.0)})())
        assert scale is not None, (
            "a normalised artist must publish the limits it draws through"
        )
        assert scale.as_limits() == (-3.0, 7.0), (
            f"limits not published verbatim: {scale.as_limits()}"
        )
        assert not scale.is_classified, "a plain Normalize cuts no classes"

    def test_an_artist_with_no_norm_publishes_nothing(self):
        """An artist that colours by nothing has no scale to publish.

        Test scenario:
            An outline-only artist — the case `quickmap` already tolerates rather than raising on. Answering
            with a scale here would offer a colorbar over a layer with no colour to explain.
        """
        assert drawn_scale(object()) is None, (
            "an artist with no norm must publish no scale"
        )

    @pytest.mark.parametrize(
        "boundaries, why",
        [
            ([1.0], "a single edge cuts no class"),
            ([5.0, 5.0], "a zero-width range has no interior to label"),
            ([10.0, 1.0], "edges that descend are not a range"),
            ([0.0, inf], "a non-finite edge cannot be labelled"),
            ([nan, 1.0], "a NaN edge cannot be labelled"),
        ],
    )
    def test_unusable_boundaries_publish_nothing_rather_than_a_wrong_scale(
        self, boundaries, why
    ):
        """Boundaries that cannot key anything answer `None`, not a scale built from them.

        Args:
            boundaries: The edge list the norm reports.
            why: What makes it unusable, for the failure message.

        Test scenario:
            Each of these would produce a `Scale` that draws — and labels the key with an edge the picture
            never cut. The honest answer is no scale, which downgrades the layer to an unkeyed one rather
            than keying it wrongly.
        """
        norm = type("N", (), {"boundaries": boundaries})()
        published = drawn_scale(type("A", (), {"norm": norm})())
        assert published is None, f"{why}; expected no scale, got {published}"

    @pytest.mark.parametrize(
        "low, high, why",
        [
            (None, 5.0, "an unmeasured floor"),
            (5.0, None, "an unmeasured ceiling"),
            (nan, 5.0, "a NaN floor"),
            (0.0, inf, "an infinite ceiling"),
        ],
    )
    def test_unusable_limits_publish_nothing(self, low, high, why):
        """Limits that are absent or non-finite publish no scale.

        Args:
            low: The norm's `vmin`.
            high: The norm's `vmax`.
            why: What makes the pair unusable, for the failure message.

        Test scenario:
            A norm is created before it is scaled to data, so `vmin`/`vmax` are `None` until an artist draws.
            Publishing `Scale.from_limits(None, None)` would raise; publishing invented numbers would label
            the key with them. Both are worse than no key.
        """
        norm = type("N", (), {"boundaries": None, "vmin": low, "vmax": high})()
        published = drawn_scale(type("A", (), {"norm": norm})())
        assert published is None, f"{why} must publish no scale, got {published}"


class TestSwatches:
    """`_swatches` — the categories and colours read back off a drawn swatch legend."""

    def test_a_drawn_legend_answers_its_labels_and_colours(self):
        """The ordinary case: one colour per label, read off the handles.

        Test scenario:
            The colours come from the handles the glyph actually drew, which is what makes a categorical key
            agree with the fill.
        """
        legend = _Legend(["river", "ridge"], [_Handle("#ff0000"), _Handle("#00ff00")])
        labels, colors = _swatches(legend)
        assert labels == ["river", "ridge"], f"labels not read back: {labels}"
        assert colors == ["#ff0000", "#00ff00"], f"colours not read back: {colors}"

    def test_a_handle_with_no_face_colour_answers_nothing(self):
        """One unreadable handle discards the whole reading rather than keying a partial legend.

        Test scenario:
            A handle type with no `get_facecolor` — a line or a marker rather than a patch. Returning the
            colours read so far would key some categories and silently drop the rest, which reads as a
            complete legend.
        """
        legend = _Legend(["a", "b"], [_Handle("#ff0000"), _Handle(None)])
        assert _swatches(legend) == ([], []), (
            "a partial reading must be discarded whole"
        )

    def test_a_label_count_that_disagrees_answers_nothing(self):
        """Labels and colours that do not pair up answer nothing rather than zipping to the shorter.

        Test scenario:
            `zip` would pair two of three and drop the third — a legend missing a category, which looks
            exactly like a legend. This is the failure mode `LegendSpec` exists to make impossible.
        """
        legend = _Legend(["a", "b", "c"], [_Handle("#ff0000"), _Handle("#00ff00")])
        assert _swatches(legend) == ([], []), (
            "an unpaired reading must be discarded whole"
        )

    def test_no_labels_at_all_answers_nothing(self):
        """An empty legend keys nothing.

        Test scenario:
            The boundary of the pairing check — zero labels and zero colours pair up trivially, and keying
            an empty legend would draw an empty box.
        """
        assert _swatches(_Legend([], [])) == ([], []), (
            "an empty legend must key nothing"
        )


class TestGuideKind:
    """`guide_kind` — which of the tier's two keys a layer asks for."""

    def test_a_recorded_kind_is_honoured(self):
        """A layer that recorded which key it asked for gets that one.

        Test scenario:
            The normal path: `colorbar()` and `legend()` each record their kind, because this tier draws two
            kinds where the others draw one and `Guide` has no field for it.
        """
        layer = type(
            "L", (), {"symbology": Symbology(props={GUIDE_KIND_KEY: "legend"})}
        )()
        assert guide_kind(layer) == "legend", (
            "a recorded kind must be honoured verbatim"
        )

    @pytest.mark.parametrize(
        "scale, expected",
        [
            (Scale.categorical(["a", "b"], ["#f00", "#0f0"]), "legend"),
            (Scale.from_limits(0.0, 10.0), "colorbar"),
        ],
    )
    def test_a_guide_from_a_foreign_figure_is_decided_by_its_scale(
        self, scale, expected
    ):
        """A guide that arrived without a kind is decided by whether its scale is categorical.

        Args:
            scale: The colour scale the layer publishes.
            expected: The key kind that follows from it.

        Test scenario:
            **A figure built where the distinction does not exist** — the other three tiers draw one kind of
            key and record no kind, so a figure carried here has none to read. The fallback applies the rule
            this tier already used through ``add_colorbar=categorical``: a categorical scale is a swatch
            list, everything else is a bar.
        """
        layer = type(
            "L",
            (),
            {
                "symbology": Symbology(
                    encodings={"color": Encoding.by_field("color", "v", scale=scale)}
                )
            },
        )()
        assert guide_kind(layer) == expected, (
            f"a {'categorical' if scale.is_categorical else 'continuous'} scale with no recorded kind "
            f"should key as {expected!r}, got {guide_kind(layer)!r}"
        )

    def test_a_layer_with_no_colour_at_all_keys_as_a_bar(self):
        """The last resort, for a layer carrying neither a kind nor a colour encoding.

        Test scenario:
            Reached only by a hand-built description. A bar is the safer default of the two: it needs the
            mappable and nothing else, where a swatch list needs rows this layer cannot supply.
        """
        layer = type("L", (), {"symbology": Symbology()})()
        assert guide_kind(layer) == "colorbar", (
            "a colourless layer must not key as a swatch list"
        )


class TestAnAnchorArrivingFromAnotherFigure:
    """`Guide.anchor` is honoured though this tier never sets one (order 24).

    `colorbar`/`legend` take no placement keyword here — the Core shape has none — so an anchor can only
    arrive on a figure built elsewhere and read back. That is the portability the guide-on-the-encoding model
    is *for*, and it was the one part of it nothing exercised.
    """

    #: How `Guide`'s corner words translate into matplotlib's. The two vocabularies disagree on the vertical
    #: axis — `Guide` says `top`/`bottom` (the word the web tier's MapLibre corners use, which is where the
    #: shared four came from) and matplotlib says `upper`/`lower`. A test that compared the words directly
    #: would fail on a correct mapping, which is what the first draft of this file did.
    _VERTICAL = {"top": "upper", "bottom": "lower"}

    @pytest.mark.parametrize(
        "anchor", ["top-left", "top-right", "bottom-left", "bottom-right"]
    )
    def test_every_shared_anchor_places_a_bar_on_the_side_it_names(self, anchor):
        """Each of the four shared anchors reaches matplotlib as the side its own name gives.

        Args:
            anchor: One of the four corners `Guide` accepts.

        Test scenario:
            The four anchors are the vocabulary `Guide` shares with the web tier's `position`. Matplotlib has
            no corners for a colorbar — only sides — so the mapping is lossy by nature, but it is lossy in
            one specific way: the **horizontal** half of the corner is what survives.

            Expected side is derived from the anchor's own spelling rather than read back out of the table.
            The first version of this test asserted `anchor in _COLORBAR_SIDES` and
            `_COLORBAR_SIDES[anchor] in {"left", "right", "top", "bottom"}` — both true of a table mapping
            all four anchors to `"left"`, so it restated the constant it imports and could not tell a
            correct table from a broken one (review N2).
        """
        from digitalearth.static.guides import _COLORBAR_SIDES

        assert anchor in _COLORBAR_SIDES, (
            f"{anchor!r} is one of Guide's four corners and has no colorbar side, so a figure carrying it "
            f"would silently lose its placement; the table has {sorted(_COLORBAR_SIDES)}"
        )
        assert _COLORBAR_SIDES[anchor] == anchor.split("-")[1], (
            f"{anchor!r} maps to {_COLORBAR_SIDES[anchor]!r}, which is not the side it names — a figure "
            f"carrying this anchor would be drawn with its key on the wrong side of the map, which looks "
            f"deliberate and is the harder error to see"
        )

    def test_the_bar_side_table_is_every_shared_anchor_s_horizontal_half(self):
        """The table is exactly that projection, over exactly the anchors the package shares.

        Test scenario:
            The parametrised test above says each of the four anchors it names is right. This says there is
            nothing else in the table and nothing missing from it, against
            `FURNITURE_ANCHORS` rather than against a list repeated here — so an anchor added to the shared
            vocabulary and forgotten in this tier reddens without anyone remembering to come back.

            It also pins the coarsening as deliberate: two anchors per side, so the vertical half is dropped
            for every anchor rather than for the ones nobody looked at. A colorbar is a strip beside the
            axes and has no corners to keep it in.
        """
        from digitalearth.base.registry import FURNITURE_ANCHORS
        from digitalearth.static.guides import _COLORBAR_SIDES

        assert _COLORBAR_SIDES == {
            anchor: anchor.split("-")[1] for anchor in FURNITURE_ANCHORS
        }, (
            "the colorbar side table is no longer the horizontal half of each shared anchor; the shared "
            f"anchors are {list(FURNITURE_ANCHORS)} and the table is {_COLORBAR_SIDES}"
        )
        assert len(set(_COLORBAR_SIDES.values())) == 2, (
            "a colorbar has sides and not corners, so the four anchors must collapse onto two sides"
        )

    @pytest.mark.parametrize(
        "anchor", ["top-left", "top-right", "bottom-left", "bottom-right"]
    )
    def test_every_shared_anchor_places_a_swatch_legend(self, anchor):
        """And each reaches matplotlib as a legend `loc`.

        Args:
            anchor: One of the four corners `Guide` accepts.

        Test scenario:
            The legend half of the same mapping. A legend *does* have corners, so this one is lossless and
            the corner must survive rather than collapsing to a side.
        """
        from digitalearth.static.guides import _LEGEND_LOCATIONS

        assert anchor in _LEGEND_LOCATIONS, (
            f"{anchor!r} has no legend location, so a figure carrying it would lose its placement; the "
            f"table has {sorted(_LEGEND_LOCATIONS)}"
        )
        placed = _LEGEND_LOCATIONS[anchor]
        vertical, horizontal = anchor.split("-")
        # Split rather than `and`-ed, so a mapping that gets one axis right and the other wrong says which.
        # A key drawn in the wrong corner is the harder error to see — it looks deliberate.
        assert self._VERTICAL[vertical] in placed, (
            f"{anchor!r} maps to {placed!r}, which does not name the {self._VERTICAL[vertical]!r} half of "
            f"its corner; a figure carrying it would be drawn with its key vertically misplaced"
        )
        assert horizontal in placed, (
            f"{anchor!r} maps to {placed!r}, which does not name the {horizontal!r} half of its corner; a "
            f"figure carrying it would be drawn with its key horizontally misplaced"
        )


class TestAnAnchoredGuideIsActuallyDrawnThere:
    """The end of the same story: an anchored guide reaches matplotlib, not just the lookup table.

    The class above checks the two mappings are complete and point at the right corner. That is necessary and
    not sufficient — a correct table read by nobody places nothing. These build the state a foreign figure
    arrives in (a layer whose encoding carries a `Guide` with an anchor) and drive the **one** drawer,
    `Renderer.draw_guide`, which is the path `from_dict` takes.
    """

    @staticmethod
    def _anchor(canvas, layer_id: str, anchor: str, kind: str):
        """Put an anchored guide on a drawn layer, the way `_record_key` does, and draw it.

        Args:
            canvas: The map holding the layer.
            layer_id: The layer to key.
            anchor: The corner the guide asks for.
            kind: ``"colorbar"`` or ``"legend"``.

        Returns:
            What `Renderer.draw_guide` drew.
        """
        from dataclasses import replace as with_fields

        from digitalearth.base.spec import Guide
        from digitalearth.static.guides import GUIDE_KIND_KEY

        layer = canvas.get_layer(layer_id)
        props = dict(layer.symbology.props)
        props[GUIDE_KIND_KEY] = kind
        symbology = with_fields(layer.symbology, props=props).with_guide(
            Guide(show=True, title="m", anchor=anchor)
        )
        canvas._layer_tree = canvas._layer_tree.replace(
            with_fields(layer, symbology=symbology)
        )
        return canvas._renderer.draw_guide(canvas._layer_tree.get(layer_id))

    @pytest.mark.parametrize("anchor", ["top-left", "bottom-left"])
    def test_a_bar_lands_on_the_side_its_anchor_names(self, anchor, dataset):
        """A colorbar anchored to a left-hand corner is drawn to the **left** of the axes.

        Args:
            anchor: The corner the foreign figure asked for.
            dataset: The committed ``acc4000`` raster.

        Test scenario:
            Both left-hand corners, and deliberately not the right-hand ones. All four anchors map to a
            matplotlib *side* — `left` or `right`, since a bar has no corners — and `right` is also
            matplotlib's own default. So a right-anchored bar is indistinguishable from an ignored anchor,
            and a test parametrised over one of each would pass with the placement deleted. The first draft
            of this test did exactly that: it asserted `orientation`, which is `vertical` for all four
            anchors *and* for the default, and it survived the mutation that dropped the anchor entirely.

            The reading is the drawn bar's own axes geometry, not the keyword that asked for it — a keyword
            proves what was requested, and what matters is where the bar went.
        """
        canvas = Map(crs=dataset.epsg)
        try:
            canvas.field(dataset, name="acc")
            bar = self._anchor(canvas, "acc", anchor, "colorbar")
            assert bar is not None, (
                f"an anchored guide on 'acc' drew nothing for {anchor!r}"
            )
            bar_x = bar.ax.get_position().x0
            axes_x = canvas.ax.get_position().x0
            assert bar_x < axes_x, (
                f"{anchor!r} asks for the left side, so the bar should start left of the axes "
                f"(x0={axes_x:.3f}); it starts at x0={bar_x:.3f}, which is where matplotlib's default "
                "right-hand placement puts it — the guide's anchor did not reach the draw"
            )
        finally:
            canvas.close()

    def test_an_anchorless_guide_leaves_matplotlib_its_default(self, dataset):
        """A guide with no anchor does not force a side.

        Args:
            dataset: The committed ``acc4000`` raster.

        Test scenario:
            The complement, and the case every key built on this tier actually takes, since `colorbar()`
            exposes no placement. It must not pick a corner on the caller's behalf — `Guide` documents `None`
            as "the tier's own default", and inventing one here would make a figure drawn on this tier
            disagree with the same figure drawn anywhere else.

            Read as geometry, like its sibling, and **not** as `bar.orientation` — which is the reading the
            sibling's own docstring condemns and which this test carried until review M11 (its second
            catch). `_COLORBAR_SIDES` holds only `left` and `right`, so every placement the tier can force
            is vertical: `orientation` survived forcing `location='left'` on every bar, the very defect
            named above. The drawn bar's `x0` beyond the axes' `x1` is where matplotlib's own default puts
            it and is where none of `left`, `top` or `bottom` can put it (measured: default and `right`
            start at x0=0.784 with the axes ending at x1=0.745, while `left` starts at 0.125 and `top` and
            `bottom` span the axes' full width from 0.125).
        """
        canvas = Map(crs=dataset.epsg)
        try:
            canvas.field(dataset, name="acc")
            bar = canvas.colorbar("acc") and canvas._renderer.drawn["acc"].guides[0]
            bar_x = bar.ax.get_position().x0
            axes_x = canvas.ax.get_position().x1
            assert bar_x > axes_x, (
                "an unanchored guide must leave the placement to matplotlib, which puts the bar clear of "
                f"the right-hand edge of the axes (x1={axes_x:.3f}); this bar starts at x0={bar_x:.3f}, so "
                "a side was picked on the caller's behalf"
            )
        finally:
            canvas.close()
