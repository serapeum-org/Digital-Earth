"""Bivariate classification: two variables encoded as one colour from an n×n matrix (engine-neutral).

A bivariate raster style in the shared, renderer-free layer. Each variable is cut into ``n`` classes
by the **shared classifier** (the same ``scheme``/``k`` every tier uses, through
:meth:`~digitalearth.base.spec.scale.Scale.breaks_of`), and the ``(class_a, class_b)`` pair picks a colour from
an ``n×n`` matrix. The result is per-element RGBA plus the legend grid, so any tier can draw it.

This module imports no renderer — only matplotlib's colour registry, the one carve-out ``base/`` allows (the
same one :mod:`digitalearth.base.symbology` uses to resolve a colour name to hex), so
``tests/test_base_is_engine_neutral.py`` stays satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from digitalearth.base.raster_classes import class_index
from digitalearth.base.spec import Scale

__all__ = [
    "DEFAULT_BIVARIATE_MATRIX",
    "MISSING_BIVARIATE_COLOR",
    "BivariateClasses",
    "bivariate_rgba",
]

#: A standard 3×3 bivariate palette (the GeoViz "DkBlue/Pink" scheme), indexed ``matrix[class_a][class_b]`` —
#: the first axis is the first variable's class, the second axis the second variable's. A corner is one
#: variable high and the other low; the opposite corner is the reverse; the far corner is both high.
DEFAULT_BIVARIATE_MATRIX: tuple[tuple[str, ...], ...] = (
    ("#e8e8e8", "#ace4e4", "#5ac8c8"),
    ("#dfb0d6", "#a5add3", "#5698b9"),
    ("#be64ac", "#8c62aa", "#3b4994"),
)

#: The colour an element missing either variable is drawn in — neutral, so no-data reads as absent rather than
#: as one of the classes.
MISSING_BIVARIATE_COLOR: str = "#cccccc"


@dataclass(frozen=True)
class BivariateClasses:
    """The result of a bivariate classification: the two edge sets, the matrix, and per-element RGBA.

    Attributes:
        edges_a: The class edges the first variable was cut at (``n + 1`` of them).
        edges_b: The class edges the second variable was cut at.
        matrix: The ``n×n`` colour matrix the classes were coloured from, as hex strings.
        rgba: The ``(N, 4)`` ``uint8`` colour of every element, in the input's flattened order.
        classes_a: The first variable's class per element (``0..n-1``, ``-1`` for no data).
        classes_b: The second variable's class per element.
    """

    edges_a: tuple[float, ...]
    edges_b: tuple[float, ...]
    matrix: tuple[tuple[str, ...], ...]
    rgba: np.ndarray
    classes_a: np.ndarray
    classes_b: np.ndarray

    @property
    def n(self) -> int:
        """The number of classes per axis (the matrix side length).

        Returns:
            The class count ``n``.
        """
        return len(self.matrix)

    def legend(self) -> list[dict[str, Any]]:
        """Return the ``n×n`` swatch grid as ``{row, col, color}`` dicts, for a tier to draw a key from.

        Returns:
            One dict per cell — ``row`` is the first variable's class, ``col`` the second, ``color`` the hex
            swatch — so a tier draws the matrix without re-deriving it.

        Examples:
            - The default 3×3 matrix gives nine swatches:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.bivariate import bivariate_rgba
                >>> classes = bivariate_rgba(np.arange(9.0), np.arange(9.0)[::-1])
                >>> len(classes.legend())
                9
                >>> classes.legend()[0]
                {'row': 0, 'col': 0, 'color': '#e8e8e8'}

                ```
        """
        return [
            {"row": row, "col": col, "color": self.matrix[row][col]}
            for row in range(self.n)
            for col in range(self.n)
        ]


def _rgba8(color: str) -> tuple[int, int, int, int]:
    """Return a colour as an ``(r, g, b, a)`` tuple of 0–255 ints.

    Args:
        color: Any colour matplotlib reads — a hex string, a name.

    Returns:
        The colour as four ``uint8`` channels.
    """
    from matplotlib.colors import to_rgba

    red, green, blue, alpha = to_rgba(color)
    return int(red * 255), int(green * 255), int(blue * 255), int(alpha * 255)


def bivariate_rgba(
    a: Any,
    b: Any,
    *,
    scheme: str = "quantiles",
    n: int = 3,
    matrix: Any = None,
) -> BivariateClasses:
    """Classify two variables and colour each element from an ``n×n`` matrix.

    Args:
        a: The first variable's values, any shape; flattened for the result.
        b: The second variable's values, the same size as `a`.
        scheme: The classification scheme both variables are cut with — one of the shared schemes
            (``"quantiles"``, ``"equal_interval"``, ``"natural_breaks"``, …), so one ``scheme``/``n`` pair cuts
            the same classes here as anywhere else.
        n: The number of classes per variable, and the side length of `matrix`.
        matrix: The ``n×n`` colour matrix, as rows of hex strings, indexed ``matrix[class_a][class_b]``.
            ``None`` uses :data:`DEFAULT_BIVARIATE_MATRIX` (which requires ``n == 3``).

    Returns:
        A :class:`BivariateClasses` carrying the edges, the matrix, and the per-element RGBA.

    Raises:
        ValueError: when `matrix` is not ``n×n``, when `a` and `b` are not the same size, or when a variable
            has no spread for the scheme to classify (propagated from the shared classifier, prefixed with the
            scheme and class count).

    Examples:
        - Two variables classify to one colour per element, with no-data elements left neutral:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.bivariate import bivariate_rgba
            >>> a = np.array([1.0, 2.0, 3.0, np.nan])
            >>> b = np.array([3.0, 2.0, 1.0, 1.0])
            >>> classes = bivariate_rgba(a, b, n=3)
            >>> classes.rgba.shape
            (4, 4)
            >>> [int(channel) for channel in classes.rgba[-1]]
            [204, 204, 204, 255]

            ```
    """
    grid = (
        DEFAULT_BIVARIATE_MATRIX
        if matrix is None
        else tuple(tuple(row) for row in matrix)
    )
    if len(grid) != n or any(len(row) != n for row in grid):
        raise ValueError(
            f"bivariate_rgba() needs an {n}×{n} matrix for n={n}; got {len(grid)} rows"
        )
    values_a = np.asarray(a, dtype=float).ravel()
    values_b = np.asarray(b, dtype=float).ravel()
    if values_a.shape != values_b.shape:
        raise ValueError(
            f"bivariate_rgba() needs a and b the same size; got {values_a.size} and {values_b.size}"
        )
    try:
        edges_a = Scale.breaks_of(values_a, scheme, n)
        edges_b = Scale.breaks_of(values_b, scheme, n)
    except ValueError as error:
        raise ValueError(
            f"bivariate_rgba() cannot classify a variable with scheme={scheme!r}, n={n}: {error}"
        ) from error
    classes_a = class_index(values_a, edges_a)
    classes_b = class_index(values_b, edges_b)
    lut = np.array(
        [[_rgba8(grid[row][col]) for col in range(n)] for row in range(n)],
        dtype="uint8",
    )
    missing = np.array(_rgba8(MISSING_BIVARIATE_COLOR), dtype="uint8")
    valid = (classes_a >= 0) & (classes_b >= 0)
    picked = lut[np.clip(classes_a, 0, n - 1), np.clip(classes_b, 0, n - 1)]
    rgba = np.where(valid[:, None], picked, missing).astype("uint8")
    return BivariateClasses(
        tuple(float(edge) for edge in edges_a),
        tuple(float(edge) for edge in edges_b),
        grid,
        rgba,
        classes_a,
        classes_b,
    )
