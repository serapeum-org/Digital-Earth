"""The engine-neutral bivariate classifier (#492).

`bivariate_rgba(a, b, scheme, n, matrix)` cuts two variables into n classes with the shared classifier and
colours each element from an n×n matrix. These cover the classification, the no-data colour, the legend grid,
and the refusals — all without importing a renderer (the module is `base/`).
"""

import numpy as np
import pytest

from digitalearth.base.bivariate import (
    DEFAULT_BIVARIATE_MATRIX,
    MISSING_BIVARIATE_COLOR,
    bivariate_rgba,
)


class TestBivariateRgba:
    """`bivariate_rgba` classifies two variables and colours each element."""

    def test_it_returns_one_rgba_per_element(self):
        """The result has one RGBA row per input element."""
        a = np.linspace(0.0, 1.0, 12)
        b = np.linspace(1.0, 0.0, 12)
        classes = bivariate_rgba(a, b, n=3)
        assert classes.rgba.shape == (12, 4), classes.rgba.shape

    def test_the_corner_classes_take_the_corner_colours(self):
        """The low/low element takes ``matrix[0][0]`` and the high/high element ``matrix[2][2]``."""
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        classes = bivariate_rgba(a, b, scheme="equal_interval", n=3)
        low = tuple(int(c) for c in classes.rgba[0])
        high = tuple(int(c) for c in classes.rgba[-1])
        from matplotlib.colors import to_rgba

        assert low == tuple(
            int(c * 255) for c in to_rgba(DEFAULT_BIVARIATE_MATRIX[0][0])
        ), low
        assert high == tuple(
            int(c * 255) for c in to_rgba(DEFAULT_BIVARIATE_MATRIX[2][2])
        ), high

    def test_a_no_data_element_takes_the_missing_colour(self):
        """An element whose value is NaN in either variable is drawn in the neutral no-data colour."""
        from matplotlib.colors import to_rgba

        a = np.array([1.0, 2.0, 3.0, np.nan])
        b = np.array([3.0, 2.0, 1.0, 1.0])
        classes = bivariate_rgba(a, b, n=3)
        missing = tuple(int(c * 255) for c in to_rgba(MISSING_BIVARIATE_COLOR))
        assert tuple(int(c) for c in classes.rgba[-1]) == missing, classes.rgba[-1]

    def test_the_two_variables_are_classified_independently(self):
        """Each variable gets its own edges, so swapping them is not the same picture.

        Test scenario:
            A pair where a rises and b falls colours the opposite diagonal of the matrix from where a
            constant b would, which only holds if the two variables are cut on their own data.
        """
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        rising = bivariate_rgba(a, a, scheme="equal_interval", n=3)
        falling = bivariate_rgba(a, a[::-1], scheme="equal_interval", n=3)
        assert not np.array_equal(rising.rgba, falling.rgba), (
            "b must be classified on its own values"
        )


class TestBivariateLegend:
    """The legend grid describes the matrix for a tier to draw a key from."""

    def test_the_legend_has_one_swatch_per_matrix_cell(self):
        """A 3×3 matrix yields nine swatches."""
        classes = bivariate_rgba(np.arange(9.0), np.arange(9.0)[::-1], n=3)
        assert len(classes.legend()) == 9, classes.legend()

    def test_a_swatch_names_its_row_column_and_colour(self):
        """The top-left swatch is row 0, col 0, the matrix's first colour."""
        classes = bivariate_rgba(np.arange(9.0), np.arange(9.0)[::-1], n=3)
        assert classes.legend()[0] == {
            "row": 0,
            "col": 0,
            "color": DEFAULT_BIVARIATE_MATRIX[0][0],
        }, classes.legend()[0]


class TestBivariateRefusals:
    """The refusals, each raising before any classification."""

    def test_a_matrix_that_is_not_n_by_n_is_refused(self):
        """A 3×3 matrix cannot serve n=4."""
        values = np.arange(4.0)
        with pytest.raises(ValueError, match="matrix for n=4"):
            bivariate_rgba(values, values, n=4)

    def test_mismatched_sizes_are_refused(self):
        """`a` and `b` must have the same number of elements."""
        five = np.arange(5.0)
        four = np.arange(4.0)
        with pytest.raises(ValueError, match="same size"):
            bivariate_rgba(five, four, n=3)

    def test_a_constant_variable_is_refused_naming_the_function(self):
        """A variable with no spread cannot be classified; the error is prefixed with bivariate_rgba."""
        constant = np.ones(6)
        spread = np.arange(6.0)
        with pytest.raises(ValueError, match="bivariate_rgba.. cannot classify"):
            bivariate_rgba(constant, spread, n=3)
