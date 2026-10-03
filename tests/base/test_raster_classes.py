"""Tests for :mod:`digitalearth.base.raster_classes` — the one raster classifier the three 2-D tiers share.

Expected values are computed independently of the module under test — sorted codes by hand, edges from
``numpy.quantile``/``linspace``, colours from ``sample_cmap``/``categorical_colors`` — so a test cannot pass by
agreeing with the code it checks.
"""

import numpy as np
import pytest
from digitalearth.base.raster_classes import (
    DEFAULT_CLASS_CMAP,
    MAX_RASTER_CATEGORIES,
    _is_code,
    asks_categorical,
    class_index,
    classes_of,
    classify_band,
    code_edges,
    raster_categories,
)

from digitalearth.base.spec import Scale
from digitalearth.base.symbology import (
    categorical_colors,
    resolve_categorical_cmap,
    sample_cmap,
)


class TestClassifyBand:
    """``classify_band`` cuts a band into classes and colours each one."""

    def test_codes_become_one_class_each_in_the_shared_palette(self):
        """A categorical band's classes are its codes, coloured as a categorical vector column would be.

        Test scenario:
            Codes 5, 1, 2 (with a NaN) give categories ``[1, 2, 5]``, edges half a step either side and midway
            between neighbours, and the shared default palette's first three colours.
        """
        classes = classify_band(
            np.array([[5, 1], [2, np.nan]]), "categorical", None, None
        )
        palette = categorical_colors([1, 2, 5], resolve_categorical_cmap(None))[1]
        assert list(classes.scale.categories) == [1, 2, 5], (
            f"categories {classes.scale.categories}"
        )
        assert classes.edges == (0.5, 1.5, 3.5, 5.5), f"edges {classes.edges}"
        assert list(classes.colors) == list(palette), (
            f"colours {classes.colors} != {palette}"
        )
        assert classes.is_categorical, "a categorical band should say so"

    def test_a_named_scheme_cuts_its_edges_and_samples_the_colormap(self):
        """Quantile classes take ``numpy``'s quantiles as edges and ``sample_cmap``'s colours.

        Test scenario:
            ``arange(25)`` cut into four quantile classes; the edges are the 0/25/50/75/100th percentiles and
            the colours four evenly spaced samples of Set2.
        """
        values = np.arange(25.0)
        classes = classify_band(values, "quantiles", 4, "Set2")
        expected = tuple(np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0]))
        assert classes.edges == pytest.approx(expected), (
            f"edges {classes.edges} != {expected}"
        )
        assert list(classes.colors) == sample_cmap("Set2", 4), (
            f"colours {classes.colors}"
        )
        assert list(classes.scale.breaks) == list(classes.edges), (
            "the scale's breaks are the edges"
        )
        assert not classes.is_categorical, "a graduated band is not categorical"

    def test_k_none_takes_five_classes_and_cmap_none_takes_the_default(self):
        """Without ``k`` a named scheme cuts five classes, and without ``cmap`` they are sampled from viridis.

        Test scenario:
            Equal intervals over 0..10 with neither argument: six edges at every 2, viridis colours.
        """
        classes = classify_band(
            np.linspace(0.0, 10.0, 11), "equal_interval", None, None
        )
        assert classes.edges == pytest.approx(tuple(np.linspace(0.0, 10.0, 6))), (
            f"edges {classes.edges}"
        )
        assert list(classes.colors) == sample_cmap(DEFAULT_CLASS_CMAP, 5), (
            f"colours {classes.colors}"
        )

    def test_masked_cells_are_not_classified(self):
        """A masked cell's value does not move a graduated scheme's edges.

        Test scenario:
            The masked ``1000`` would stretch an equal-interval cut to 1000; masked, the edges span 0..4.
        """
        values = np.ma.masked_array([0.0, 1.0, 2.0, 4.0, 1000.0], mask=[0, 0, 0, 0, 1])
        classes = classify_band(values, "equal_interval", 2, "viridis")
        assert classes.edges == pytest.approx((0.0, 2.0, 4.0)), f"edges {classes.edges}"

    def test_an_unknown_scheme_is_refused_by_name(self):
        """A scheme the classifier does not know is refused, naming it.

        Test scenario:
            ``"nope"`` must not fall back to a ramp.
        """
        values = np.arange(5.0)
        with pytest.raises(ValueError, match="nope"):
            classify_band(values, "nope", 3, "viridis")


class TestClassesOf:
    """``classes_of`` recovers a band's classes from the colour scale a figure recorded."""

    def test_a_categorical_scale_gives_its_codes_in_ascending_order(self):
        """Categories recorded out of order still bin by ascending code, each keeping its own colour.

        Test scenario:
            ``[5, 1]`` with colours red/blue gives edges for ``[1, 5]`` and colours blue/red.
        """
        classes = classes_of(Scale.categorical([5, 1], ["#ff0000", "#0000ff"]), None)
        assert classes.edges == (0.5, 3.0, 5.5), f"edges {classes.edges}"
        assert classes.colors == ("#0000ff", "#ff0000"), f"colours {classes.colors}"

    def test_named_categories_have_no_cells_to_colour(self):
        """A categorical scale of names cannot classify a raster, so the band keeps its ramp.

        Test scenario:
            ``["forest", "water"]`` are no cell's value; the answer is ``None`` rather than a crash.
        """
        scale = Scale.categorical(["forest", "water"], ["#00ff00", "#0000ff"])
        assert classes_of(scale, None) is None, (
            "named categories should give no classes"
        )

    def test_a_classified_scale_gives_its_breaks_and_the_sampled_colours(self):
        """A recorded graduated scale's breaks are the edges, and its colours are sampled from ``cmap``.

        Test scenario:
            Breaks ``(0, 1, 3)`` with ``cmap="Set2"``: two classes, Set2's two samples.
        """
        scale = Scale(0.0, 3.0, scheme=[0.0, 1.0, 3.0], breaks=(0.0, 1.0, 3.0))
        classes = classes_of(scale, "Set2")
        assert classes.edges == (0.0, 1.0, 3.0), f"edges {classes.edges}"
        assert list(classes.colors) == sample_cmap("Set2", 2), (
            f"colours {classes.colors}"
        )

    @pytest.mark.parametrize(
        "scale", [None, Scale.from_limits(0.0, 1.0)], ids=["no-scale", "continuous"]
    )
    def test_no_scale_or_a_continuous_one_has_no_classes(self, scale):
        """Nothing classified was recorded, so the band is drawn along its ramp.

        Args:
            scale: The recorded scale.
        """
        assert classes_of(scale, "viridis") is None, f"{scale!r} should give no classes"


class TestClassIndex:
    """``class_index`` bins each cell the way matplotlib's ``BoundaryNorm`` bins the static tier's."""

    def test_cells_bin_into_their_classes_and_out_of_range_cells_take_the_ends(self):
        """An inner edge belongs to the class above it; cells past either end take the end classes.

        Test scenario:
            Edges ``[0, 1, 2]``: ``-5`` and ``0.5`` → 0, ``1.0`` → 1 (on the edge), ``9`` → 1, NaN → -1.
        """
        index = class_index(np.array([-5.0, 0.5, 1.0, 9.0, np.nan]), [0.0, 1.0, 2.0])
        assert index.tolist() == [0, 0, 1, 1, -1], f"classes {index.tolist()}"

    def test_a_masked_cell_is_in_no_class(self):
        """A masked cell gets ``-1``, like a NaN.

        Test scenario:
            The masked ``1.5`` would be class 1; masked it is no class.
        """
        values = np.ma.masked_array([0.5, 1.5], mask=[False, True])
        assert class_index(values, [0.0, 1.0, 2.0]).tolist() == [0, -1], (
            "the masked cell is no class"
        )


class TestHelpers:
    """The small predicates the classifier is built from."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (3, True),
            (4.0, True),
            (np.int64(2), True),
            (2.5, False),
            ("a", False),
            (True, False),
        ],
        ids=["int", "whole-float", "np-int", "fraction", "name", "bool"],
    )
    def test_is_code(self, value, expected):
        """Only whole numbers — never names, fractions or ``bool`` — are a raster cell's code.

        Args:
            value: A recorded category.
            expected: Whether it is a code.
        """
        assert _is_code(value) is expected, f"_is_code({value!r}) should be {expected}"

    @pytest.mark.parametrize(
        "scheme, expected",
        [
            ("categorical", True),
            ("CATEGORICAL", True),
            ("quantiles", False),
            ([0, 1], False),
            (None, False),
        ],
    )
    def test_asks_categorical(self, scheme, expected):
        """``"categorical"`` in any case asks for codes; nothing else does.

        Args:
            scheme: The caller's scheme.
            expected: Whether it asks for codes.
        """
        assert asks_categorical(scheme) is expected, f"asks_categorical({scheme!r})"

    def test_code_edges_put_each_code_in_its_own_class(self):
        """Every code lies strictly inside its own class.

        Test scenario:
            Codes with a gap and a negative: each code's class is the one between its two edges.
        """
        codes = [-3, 0, 7]
        edges = code_edges(codes)
        assert all(edges[i] < code < edges[i + 1] for i, code in enumerate(codes)), (
            f"edges {edges}"
        )

    def test_the_category_limit_is_exported(self):
        """The limit the refusal names is the one every tier enforces."""
        codes = np.arange(MAX_RASTER_CATEGORIES + 1)
        with pytest.raises(ValueError, match=f"at most {MAX_RASTER_CATEGORIES}"):
            raster_categories(codes)

    @pytest.mark.parametrize(
        "values, message",
        [
            (np.array([np.nan, np.nan]), "no valid cells"),
            (np.array([1.0, 2.5]), "integer class codes"),
            (np.array([2**52, 2**52 + 1], dtype=np.int64), r"2\*\*52"),
        ],
        ids=["all-nodata", "fractional", "too-large"],
    )
    def test_a_band_that_is_not_nominal_is_refused(self, values, message):
        """A band with no cell, or with a magnitude rather than a code, is refused by name.

        Args:
            values: The band.
            message: What the refusal says.
        """
        with pytest.raises(ValueError, match=message):
            raster_categories(values)
