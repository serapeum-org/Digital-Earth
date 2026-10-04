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
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map
from digitalearth.static.guides import GuidePlan, paint_guide

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

    def test_the_callers_labels_name_the_keyed_rows(self, p_values):
        """``labels=`` numbers the rows the key draws — one, for the overlay.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        canvas.legend("significance", labels=["p < 0.05"])
        texts = [text.get_text() for text in canvas.ax.get_legend().get_texts()]
        assert texts == ["p < 0.05"], texts

    def test_labels_that_do_not_number_the_rows_are_refused(self, p_values):
        """Two labels for a one-row key would label a swatch nobody drew.

        Args:
            p_values: The p-value field.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        with pytest.raises(ValueError, match="1 rows"):
            canvas.legend("significance", labels=["p < 0.05", "not significant"])

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
            the keyed rows are refused whether or not the key is drawn. So the same overlay must leave no
            legend on the axes with ``visible=False``, and still refuse a bad ``labels=`` through it.
        """
        canvas = Map(crs=4326)
        _overlay(canvas, p_values)
        canvas.legend("significance", visible=False)
        assert canvas.ax.get_legend() is None, (
            f"a key switched off must not be on the axes, got {canvas.ax.get_legend()}"
        )
        with pytest.raises(ValueError, match="1 rows"):
            canvas.legend("significance", labels=["a", "b"], visible=False)

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
