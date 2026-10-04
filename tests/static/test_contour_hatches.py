"""Hatch encoding on filled contours, and the hatch legend that keys it (ST-23).

cleopatra 0.38's ``Contour`` group carries a hatch pattern per band (``hatches``), an uncoloured overlay form
(``fill=False``) and a stroke colour (``hatch_color``) on the ``contourf`` path — the usual way a significance
or uncertainty mask is drawn over a field without spending the colour channel. These pin the plain keywords
the static ``contours`` builder takes for it, and that ``legend()`` on a hatched layer keys its bands by
pattern rather than reading a continuous ramp off a layer that has no colours.
"""

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.path import Path
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map
from digitalearth.static.guides import GuidePlan, _marked_bands, paint_guide

#: The p-value bands of a significance overlay: significant below 0.05, not above it.
P_LEVELS = [0.0, 0.05, 1.0]


@pytest.fixture(scope="module")
def p_values() -> Dataset:
    """A 4 x 4 field of p-values in lon/lat, half of it below 0.05.

    Returns:
        A pyramids `Dataset`.
    """
    arr = np.array(
        [
            [0.01, 0.02, 0.30, 0.40],
            [0.01, 0.03, 0.50, 0.60],
            [0.02, 0.04, 0.70, 0.80],
            [0.03, 0.04, 0.90, 0.95],
        ],
        dtype="float64",
    )
    return Dataset.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
        no_data_value=-9999.0,
    )


def _overlay(canvas: Map, dataset: Dataset, **kwargs):
    """Draw the significance overlay: hatched below 0.05, nothing above.

    Args:
        canvas: The map.
        dataset: The p-values.
        **kwargs: Extra keywords for ``contours``.

    Returns:
        The contour set.
    """
    return canvas.contours(
        dataset,
        levels=P_LEVELS,
        filled=True,
        hatches=["///", ""],
        fill=False,
        name="significance",
        **kwargs,
    )


class TestTheHatchKeywords:
    """``hatches=``, ``fill=`` and ``hatch_color=`` reach the drawn contour set."""

    def test_the_bands_carry_the_patterns_asked_for(self, p_values):
        """One pattern per band, in level order.

        Args:
            p_values: The p-value field.
        """
        artist = _overlay(Map(crs=4326), p_values)
        assert list(artist.hatches) == ["///", ""], artist.hatches

    def test_an_unfilled_overlay_paints_no_colour(self, p_values):
        """``fill=False`` leaves every band transparent, so only the hatching marks the map.

        Args:
            p_values: The p-value field.
        """
        artist = _overlay(Map(crs=4326), p_values)
        alphas = np.asarray(artist.get_facecolor())[:, 3]
        assert np.all(alphas == 0.0), alphas

    def test_a_filled_hatched_layer_keeps_its_colours(self, p_values):
        """Without ``fill=False`` the bands are coloured and hatched both.

        Args:
            p_values: The p-value field.
        """
        artist = Map(crs=4326).contours(
            p_values, levels=P_LEVELS, filled=True, hatches=["///", ""]
        )
        alphas = np.asarray(artist.get_facecolor())[:, 3]
        assert np.all(alphas > 0.0), alphas

    def test_the_stroke_colour_reaches_the_hatching(self, p_values):
        """``hatch_color`` recolours the strokes.

        Args:
            p_values: The p-value field.
        """
        artist = _overlay(Map(crs=4326), p_values, hatch_color="red")
        assert tuple(artist.get_hatchcolor()[0]) == to_rgba("red"), (
            artist.get_hatchcolor()
        )

    def test_the_patterns_are_described(self, p_values):
        """A figure records the patterns, so it can be drawn again with them.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        opts = canvas.figure_spec.layers.get("significance").symbology.props["opts"]
        assert list(opts["hatches"]) == ["///", ""], opts

    def test_hatches_on_contour_lines_are_refused(self, p_values):
        """A line contour has no bands to hatch, so the request is an error rather than a no-op.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        with pytest.raises(ValueError, match="filled=True"):
            canvas.contours(p_values, levels=P_LEVELS, hatches=["///", ""])
        assert canvas.layer_ids == [], canvas.layer_ids

    def test_a_stroke_colour_with_no_patterns_to_stroke_is_refused(self, p_values):
        """``hatch_color`` without ``hatches`` colours nothing, so it is refused rather than dropped.

        Args:
            p_values: The p-value field.

        Test scenario:
            - Ask for a filled render with a hatch colour and no patterns — the other half of the guard
              that already refuses ``hatch_color`` on a line contour.
            - cleopatra answers that spelling with a ``UserWarning`` and draws the bands plain, which is
              the silent drop this builder refuses on the ``filled=`` axis; the error names
              ``hatch_color``, and no layer is left on the figure.
        """
        canvas = Map(crs=4326)
        with pytest.raises(ValueError, match="hatch_color"):
            canvas.contours(p_values, levels=P_LEVELS, filled=True, hatch_color="red")
        assert canvas.layer_ids == [], canvas.layer_ids

    def test_an_uncoloured_render_with_no_patterns_is_refused(self, p_values):
        """``fill=False`` with no ``hatches`` draws an invisible layer, so it is refused by name.

        Args:
            p_values: The p-value field.

        Test scenario:
            - ``fill=False`` is the overlay form: it leaves every band transparent so only the hatching
              marks the map. With no patterns there is nothing left to mark it with.
            - cleopatra warns and draws the invisible set; the builder refuses it instead, naming
              ``fill=False``, and leaves the figure empty.
        """
        canvas = Map(crs=4326)
        with pytest.raises(ValueError, match="fill=False"):
            canvas.contours(p_values, levels=P_LEVELS, filled=True, fill=False)
        assert canvas.layer_ids == [], canvas.layer_ids

    def test_fill_on_contour_lines_is_refused(self, p_values):
        """``fill`` is a band keyword, so a line render refuses it as it already refuses ``hatches``.

        Args:
            p_values: The p-value field.

        Test scenario:
            - Pass ``fill=False`` with the default ``filled=False``.
            - cleopatra's answer is ``"hatches/fill/hatch_color are contourf-only and are ignored for
              kind='contour'"`` — a warning and then the plain lines. The builder refuses it, so all three
              band keywords are guarded on that axis rather than two of them.
        """
        canvas = Map(crs=4326)
        with pytest.raises(ValueError, match="filled=True"):
            canvas.contours(p_values, levels=P_LEVELS, fill=False)
        assert canvas.layer_ids == [], canvas.layer_ids


class TestTheHatchLegend:
    """``legend()`` on a hatched layer keys its bands by pattern."""

    def test_an_overlay_keys_only_the_bands_it_marks(self, p_values):
        """The unhatched, uncoloured band draws nothing on the map, so it has no row either.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        canvas.legend("significance")
        legend = canvas.ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == ["0.0 – 0.05"], [
            text.get_text() for text in legend.get_texts()
        ]
        assert [handle.get_hatch() for handle in legend.legend_handles] == ["///"]

    def test_a_hatched_band_holding_no_data_is_not_keyed(self, p_values):
        """Levels past the field leave bands with no geometry, and a band that marks nothing has no row.

        Args:
            p_values: The p-value field, whose values lie in [0.01, 0.95].

        Test scenario:
            - Ask for levels that run well past the data — ``[0, 0.05, 1, 5, 10]`` over p-values that stop
              at 0.95 — with a pattern on three of the four bands, drawn as an uncoloured overlay.
            - The last two bands hold no geometry at all (measured: ``get_paths()`` reports 11 vertices for
              band 0, 11 for band 1 and 0 for bands 2 and 3), so the only band the map marks is the first.
            - The key used to read ``['0.0 – 0.05', '1.0 – 5.0', '5.0 – 10.0']``, inventing two
              significance classes a reader would look for and not find.
        """
        canvas = Map(crs=4326)
        canvas.contours(
            p_values,
            levels=[0.0, 0.05, 1.0, 5.0, 10.0],
            filled=True,
            hatches=["///", "", "..", "xx"],
            fill=False,
            name="beyond",
        )
        canvas.legend("beyond")
        legend = canvas.ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == ["0.0 – 0.05"], [
            text.get_text() for text in legend.get_texts()
        ]
        assert [handle.get_hatch() for handle in legend.legend_handles] == ["///"]

    def test_a_coloured_band_holding_no_data_is_not_keyed_either(self, p_values):
        """A band with a colour and a pattern but no geometry still draws nothing, so it is not keyed.

        Args:
            p_values: The p-value field, whose values lie in [0.01, 0.95].

        Test scenario:
            - The same over-wide levels, but filled: every band now has an opaque face as well as a
              pattern, so the old ``pattern or face[3] > 0.0`` filter kept all four rows.
            - Only the two bands the data reaches draw anything, so the key has two rows, labelled for the
              two level pairs the field spans.
        """
        canvas = Map(crs=4326)
        canvas.contours(
            p_values,
            levels=[0.0, 0.05, 1.0, 5.0, 10.0],
            filled=True,
            hatches=["///", "||", "..", "xx"],
            name="filled-beyond",
        )
        canvas.legend("filled-beyond")
        legend = canvas.ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            "0.0 – 0.05",
            "0.05 – 1.0",
        ], [text.get_text() for text in legend.get_texts()]
        assert [handle.get_hatch() for handle in legend.legend_handles] == ["///", "||"]

    def test_an_overlay_swatch_is_unfilled(self, p_values):
        """The swatch shows the pattern on the background, as the map does.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        canvas.legend("significance")
        (handle,) = canvas.ax.get_legend().legend_handles
        assert handle.get_facecolor()[3] == 0.0, handle.get_facecolor()

    def test_the_swatch_strokes_take_the_hatch_colour(self, p_values):
        """A red hatch on the map is a red hatch in the key.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values, hatch_color="red")
        canvas.legend("significance")
        (handle,) = canvas.ax.get_legend().legend_handles
        assert tuple(handle.get_hatchcolor()) == to_rgba("red"), handle.get_hatchcolor()

    def test_a_filled_hatched_layer_keys_every_band_with_colour_and_pattern(
        self, p_values
    ):
        """Both bands are drawn, so both are keyed, each with its colour and its pattern.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        artist = canvas.contours(
            p_values, levels=P_LEVELS, filled=True, hatches=["///", ".."], name="p"
        )
        canvas.legend("p")
        handles = canvas.ax.get_legend().legend_handles
        assert [handle.get_hatch() for handle in handles] == ["///", ".."]
        drawn = np.asarray(artist.get_facecolor())
        keyed = np.asarray([handle.get_facecolor() for handle in handles])
        # A swatch colour travels as hex, so it agrees with the band to within one 8-bit step.
        assert np.allclose(keyed, drawn, atol=1 / 255), (keyed, drawn)

    def test_the_callers_labels_name_the_bands_the_levels_declare(self, p_values):
        """``labels=`` is one label per band between ``levels``, and the drawn row takes its own.

        Args:
            p_values: The p-value field.

        Test scenario:
            - ``_overlay`` declares ``levels=[0, 0.05, 1]`` — two bands — and only the first is keyed,
              since the second is neither hatched nor filled.
            - ``labels=`` is counted against the two declared bands, so both are named; the label of the
              band that is not keyed is dropped with its row and only ``"p < 0.05"`` is drawn.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        canvas.legend("significance", labels=["p < 0.05", "not significant"])
        texts = [text.get_text() for text in canvas.ax.get_legend().get_texts()]
        assert texts == ["p < 0.05"], texts

    def test_a_label_count_the_levels_do_not_declare_is_refused(self, p_values):
        """One label for a two-band layer leaves a declared band unnamed.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        with pytest.raises(ValueError, match="2 bands"):
            canvas.legend("significance", labels=["p < 0.05"])

    def test_the_label_count_does_not_follow_the_data(self, p_values):
        """``labels=`` numbers the declared bands, so a saved list survives data that marks fewer of them.

        Args:
            p_values: The p-value field, whose values lie in [0.01, 0.95].

        Test scenario:
            - Four bands are declared, ``levels=[0, 0.05, 1, 5, 10]``, each with a pattern, over a field
              that stops at 0.95 — so only the first two bands hold geometry and only those two are keyed.
            - Four labels, one per declared band, are accepted: the count is a property of the call, not of
              the raster, so the same ``labels=`` keeps working when the data stops reaching a band.
            - Only the labels of the two keyed bands are drawn; ``"c"`` and ``"d"`` go with their rows.
            - Before this, four labels were refused with "has 2 rows, so labels= needs 2 labels; got 4" and
              only a two-item list — derived from the data — was accepted.
        """
        canvas = Map(crs=4326)
        canvas.contours(
            p_values,
            levels=[0.0, 0.05, 1.0, 5.0, 10.0],
            filled=True,
            hatches=["///", "..", "xx", "oo"],
            fill=False,
            name="declared",
        )
        canvas.legend("declared", labels=["a", "b", "c", "d"])
        texts = [text.get_text() for text in canvas.ax.get_legend().get_texts()]
        assert texts == ["a", "b"], texts

    def test_a_label_list_the_data_happens_to_number_is_refused_too(self, p_values):
        """The count that the data would give is not the count the call declares.

        Args:
            p_values: The p-value field, whose values lie in [0.01, 0.95].

        Test scenario:
            - The same four declared bands, of which the data marks two — and two labels, which the old
              data-derived rule accepted.
            - It is refused now, naming the four bands ``levels=`` declares, so the accepted spelling is
              the one that stays accepted.
        """
        canvas = Map(crs=4326)
        canvas.contours(
            p_values,
            levels=[0.0, 0.05, 1.0, 5.0, 10.0],
            filled=True,
            hatches=["///", "..", "xx", "oo"],
            fill=False,
            name="declared",
        )
        with pytest.raises(ValueError, match="4 bands"):
            canvas.legend("declared", labels=["a", "b"])

    def test_a_layer_that_marks_no_band_gets_no_legend_box(self, p_values, caplog):
        """With nothing keyed there is nothing to key, so no box goes on the axes at all.

        Args:
            p_values: The p-value field, whose values lie in [0.01, 0.95].
            caplog: pytest's log capture.

        Test scenario:
            - Levels that sit entirely above the data — ``[5, 6, 7]`` over p-values that stop at 0.95 — so
              both declared bands are drawn as paths with no vertices and no band is keyed.
            - An empty framed box carrying only its title used to go on the axes: measured
              ``legend present: True  texts: []  title: Significance``. That heading is itself the class a
              reader looks for and does not find, which is what round 1's M1 fix removed rows for.
            - Nothing is drawn now, and the skipped key is logged rather than left silent.
        """
        canvas = Map(crs=4326)
        canvas.contours(
            p_values,
            levels=[5.0, 6.0, 7.0],
            filled=True,
            hatches=["///", ".."],
            fill=False,
            name="none",
        )
        with caplog.at_level("WARNING", logger="digitalearth.static.guides"):
            canvas.legend("none", title="Significance")
        assert canvas.ax.get_legend() is None, (
            "a layer that marks no band must get no legend box, got "
            f"{canvas.ax.get_legend()}"
        )
        assert "marks none of its" in caplog.text, caplog.text

    def test_an_unhatched_filled_layer_is_keyed_as_before(self, p_values):
        """Hatching changes nothing for a layer that has none.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        canvas.contours(p_values, levels=P_LEVELS, filled=True, name="p")
        canvas.legend("p")
        handles = canvas.ax.get_legend().legend_handles
        assert all(not handle.get_hatch() for handle in handles), [
            handle.get_hatch() for handle in handles
        ]

    def test_a_hatch_key_switched_off_is_checked_and_then_not_drawn(self, p_values):
        """``visible=False`` records the guide and draws nothing — after the rows have been derived.

        Args:
            p_values: The p-value field.

        Test scenario:
            The ``show`` gate sits *behind* the derivation for the hatched path too, which is what keeps one
            spelling of a call from being valid only half the time (review M1): labels that do not number
            the declared bands are refused whether or not the key is drawn. So the same overlay must leave
            no legend on the axes with ``visible=False``, and still refuse a bad ``labels=`` through it.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        canvas.legend("significance", visible=False)
        assert canvas.ax.get_legend() is None, (
            f"a key switched off must not be on the axes, got {canvas.ax.get_legend()}"
        )
        with pytest.raises(ValueError, match="2 bands"):
            canvas.legend("significance", labels=["a", "b", "c"], visible=False)

    def test_a_plan_naming_no_hatch_colour_leaves_the_strokes_at_the_engine_default(
        self,
    ):
        """A derived key with patterns but no stroke colour paints the patterns and recolours nothing.

        Test scenario:
            ``GuidePlan`` is the seam between deciding what a key is and putting it on the figure, and its
            ``hatch_color`` is optional. Painting a coloured, patterned plan that names no stroke colour
            must still set every swatch's pattern — the patterns are the whole point of the key — while
            leaving the strokes wherever matplotlib puts them, which a bare ``Patch`` built here reports
            independently of the code under test.
        """
        rows, hatches = ("low", "high"), ("///", "..")
        plan = GuidePlan(
            "legend",
            colors=("#1f77b4ff", "#ff7f0eff"),
            rows=rows,
            hatches=hatches,
        )
        canvas = Map(crs=4326)
        handles = paint_guide(canvas, plan).legend_handles
        default_stroke = tuple(Patch(hatch="///").get_hatchcolor())
        assert [handle.get_hatch() for handle in handles] == list(hatches), (
            f"every swatch must carry its band's pattern, got "
            f"{[handle.get_hatch() for handle in handles]}"
        )
        assert [tuple(handle.get_hatchcolor()) for handle in handles] == [
            default_stroke
        ] * len(hatches), (
            f"a plan naming no hatch colour must leave the strokes at {default_stroke}, got "
            f"{[tuple(handle.get_hatchcolor()) for handle in handles]}"
        )

    def test_an_opaque_colour_whose_hex_ends_in_00_still_fills_its_swatch(self):
        """Transparency is a number, not a string suffix, so an opaque ``#…00`` keeps its colour.

        Test scenario:
            - ``GuidePlan`` + ``paint_guide`` is a public seam, and a plan built by hand may carry 6-digit
              hex. ``#aabb00`` and ``#112200`` are fully opaque colours whose last two characters happen
              to be the two the old check read as "alpha 00".
            - Both swatches came back ``(0, 0, 0, 0)`` — the uncoloured ``hatch_legend`` form — so two
              opaque colours were dropped. Each swatch must now carry its own colour, computed here from
              the hex by matplotlib rather than read back out of the code under test.
        """
        colors = ("#aabb00", "#112200")
        plan = GuidePlan(
            "legend", colors=colors, rows=("a", "b"), hatches=("///", "..")
        )
        canvas = Map(crs=4326)
        handles = paint_guide(canvas, plan).legend_handles
        assert [tuple(handle.get_facecolor()) for handle in handles] == [
            to_rgba(color) for color in colors
        ], [tuple(handle.get_facecolor()) for handle in handles]

    def test_the_faintest_colour_an_8_bit_channel_can_carry_still_fills_its_swatch(
        self,
    ):
        """``#00000001`` is alpha 1/255, the smallest non-zero an 8-digit hex can spell — and it fills.

        Test scenario:
            - The transparency check is a tolerance rather than an exact ``== 0.0`` comparison
              (SonarCloud ``S1244``), and this is the boundary that says the tolerance is tight enough:
              every colour the plan carries round-trips through 8-bit hex, so the smallest alpha above
              zero it can hold is ``1/255 == 0.00392156862745098``.
            - A tolerance at or above that would read this colour as see-through and route the key into
              the uncoloured ``hatch_legend`` form, which is the same class of mistake as the
              ``endswith("00")`` reading it replaced.
            - The expected face is computed here by matplotlib from the hex, not read back out of the
              code under test.
        """
        plan = GuidePlan("legend", colors=("#00000001",), rows=("a",), hatches=("///",))
        canvas = Map(crs=4326)
        (handle,) = paint_guide(canvas, plan).legend_handles
        assert tuple(handle.get_facecolor()) == to_rgba("#00000001"), (
            f"the faintest 8-bit alpha must still colour its swatch, got "
            f"{tuple(handle.get_facecolor())}"
        )

    def test_a_fully_transparent_colour_leaves_its_swatch_unfilled(self):
        """Alpha exactly zero is see-through, so the key is drawn in the uncoloured form.

        Test scenario:
            - The other side of the boundary above: ``#00000000`` has alpha ``0.0`` exactly, so the whole
              plan is uncoloured and ``_paint_hatch_legend`` hands it to ``hatch_legend``, whose swatches
              carry no face at all.
        """
        plan = GuidePlan("legend", colors=("#00000000",), rows=("a",), hatches=("///",))
        canvas = Map(crs=4326)
        (handle,) = paint_guide(canvas, plan).legend_handles
        assert tuple(handle.get_facecolor()) == to_rgba("none"), (
            f"a fully transparent colour must leave the swatch unfilled, got "
            f"{tuple(handle.get_facecolor())}"
        )

    def test_a_colour_matplotlib_cannot_read_is_not_taken_for_transparent(self):
        """An unreadable swatch colour reaches the engine that can name it, rather than being dropped.

        Test scenario:
            - The transparency check parses the colour, and a colour matplotlib cannot parse has no alpha
              to read. Answering "transparent" for it would route a plan of unreadable colours into the
              uncoloured ``hatch_legend`` form, which draws happily — so a typo in a hand-built plan drew
              a legend that was merely wrong instead of saying so.
            - ``paint_guide`` must therefore still hand the colour to the swatch legend, where matplotlib
              refuses it by name. The one call in the block is the paint; the plan and the map are built
              above it.
        """
        plan = GuidePlan(
            "legend", colors=("nosuchcolour",), rows=("a",), hatches=("///",)
        )
        canvas = Map(crs=4326)
        with pytest.raises(ValueError, match="nosuchcolour") as refusal:
            paint_guide(canvas, plan)
        assert canvas.ax.get_legend() is None, (
            "a refused colour must leave no legend on the axes"
        )
        assert "nosuchcolour" in str(refusal.value), (
            f"the refusal must name the colour it could not read, got {refusal.value}"
        )


class TestTheBandsAMapActuallyMarks:
    """``_marked_bands`` reads the geometry per band off the artist, and says so when it cannot.

    The tests above drive the filter through a real ``contours`` call, which is how a caller meets it. This
    one pins the answer for the shape those calls never produce: an artist whose path count does not number
    its bands, which the matplotlib contract allows to change under us.
    """

    class _PathsOnly:
        """The one method ``_marked_bands`` reads off a filled contour set."""

        def __init__(self, paths):
            """Hold the paths to hand back.

            Args:
                paths: The compound paths, one per band in a well-formed set.
            """
            self._paths = list(paths)

        def get_paths(self):
            """Return the paths, the way ``ContourSet.get_paths`` does.

            Returns:
                The list given at construction.
            """
            return list(self._paths)

    @pytest.mark.parametrize(
        "paths, bands",
        [(1, 3), (4, 3), (0, 2)],
        ids=["fewer", "more", "none"],
    )
    def test_paths_that_do_not_number_the_bands_keep_every_band(self, paths, bands):
        """When the reading no longer holds, every band is reported marked rather than none.

        Args:
            paths: How many compound paths the stand-in artist reports.
            bands: How many bands the set's levels declare.

        Test scenario:
            - The filter is a subtraction: a band reported unmarked loses its row in the key. If a future
              matplotlib stopped publishing one path per band, reading the flags positionally would drop
              rows that the map does draw — a key missing real classes, which is worse than one carrying
              an empty band.
            - So a path count that does not number the bands answers "all marked", for a count below, above
              and at zero.
        """
        artist = self._PathsOnly([Path(np.zeros((0, 2)))] * paths)
        assert _marked_bands(artist, bands) == [True] * bands, (
            f"{paths} paths over {bands} bands must keep every band, got "
            f"{_marked_bands(artist, bands)}"
        )
