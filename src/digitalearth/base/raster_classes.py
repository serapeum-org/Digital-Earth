"""Cut a raster band into colour classes once, for every tier that draws it.

A raster drawn with ``scheme=`` is coloured class by class rather than along a continuous ramp. Two things
decide what a viewer sees, and both are decided here so that no tier can answer them differently:

- **the classes** — the class edges of a graduated scheme (``"quantiles"``, ``"equal_interval"``, an explicit
  list of edges, …), cut by :meth:`~digitalearth.base.spec.scale.Scale.breaks_of`, the classifier the
  graduated vector layers already share; or, for ``scheme="categorical"``, one class per integer code;
- **the colour of each class** — the shared categorical palette for codes, the same one a categorical vector
  column takes, and :func:`~digitalearth.base.symbology.sample_cmap` for graduated classes, the sampler every
  tier's graduated vector layer uses.

The result is a :class:`BandClasses`: the layer's colour :class:`~digitalearth.base.spec.scale.Scale` (what a
figure records and a legend is derived from), the edges each cell is binned by, and one colour per class.
:func:`classes_of` turns a *recorded* scale back into the same thing, which is how a figure one tier
described draws on another: the classes travel in the layer's colour encoding, so the drawing tier does not
classify the band a second time.

Engine-neutral: numpy and :mod:`digitalearth.base` only — no renderer is imported, so the static, interactive
and web tiers can all build on it.
"""

import numbers
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from digitalearth.base.spec.scale import DEFAULT_CLASS_COUNT, Scale
from digitalearth.base.symbology import (
    categorical_colors,
    resolve_categorical_cmap,
    sample_cmap,
)

__all__ = [
    "MAX_RASTER_CATEGORIES",
    "DEFAULT_CLASS_CMAP",
    "BandClasses",
    "asks_categorical",
    "class_index",
    "classes_of",
    "classify_band",
    "code_edges",
    "raster_categories",
]

#: The most distinct codes a ``scheme="categorical"`` raster is drawn with. A key of more swatches than this
#: stops being readable, and a band with that many distinct values is almost always a magnitude rather than a
#: set of classes — so it is refused, pointing at a graduated scheme.
MAX_RASTER_CATEGORIES = 24

#: The colormap graduated classes are sampled from when none is given — every tier's default field colormap.
DEFAULT_CLASS_CMAP = "viridis"

#: The magnitude a class code must stay below: past ``2**52`` a float no longer resolves the half step the class
#: edges sit on (see :func:`code_edges`).
_EXACT_CODE_LIMIT = 2.0**52


@dataclass(frozen=True)
class BandClasses:
    """How a classified raster band is coloured: its scale, the edges its cells are binned by, and the colours.

    Attributes:
        scale: The layer's colour scale — categorical (the codes and their colours) or graduated (the class
            edges as its breaks). It is what a figure records in the layer's colour encoding, and what every
            tier's legend is derived from.
        edges: The class edges each cell is binned by, ascending: one more than there are classes. For codes,
            half a step either side of each code and midway between neighbours (:func:`code_edges`).
        colors: One ``#rrggbb`` colour per class, in class order.

    Examples:
        - Three equal-interval classes of a ramp, coloured from viridis:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classify_band
            >>> classes = classify_band(np.arange(10.0), "equal_interval", 3, "viridis")
            >>> classes.edges
            (0.0, 3.0, 6.0, 9.0)
            >>> classes.colors
            ('#440154', '#21918c', '#fde725')

            ```
        - A band of codes gets one class per code, and its scale records the codes as categories:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classify_band
            >>> classes = classify_band(np.array([[1, 3], [3, 1]]), "categorical", None, None)
            >>> classes.scale.categories, classes.edges, classes.colors
            ((1, 3), (0.5, 2.0, 3.5), ('#1f77b4', '#ff7f0e'))

            ```
    """

    scale: Scale
    edges: Tuple[float, ...]
    colors: Tuple[str, ...]

    @property
    def is_categorical(self) -> bool:
        """Whether each class is one integer code rather than a range of values.

        Returns:
            ``True`` for a ``scheme="categorical"`` band.

        Examples:
            - Codes are categorical; a graduated scheme is not:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.raster_classes import classify_band
                >>> classify_band(np.array([1, 2, 2]), "categorical", None, None).is_categorical
                True
                >>> classify_band(np.arange(4.0), "quantiles", 2, "viridis").is_categorical
                False

                ```
        """
        return self.scale.is_categorical


def asks_categorical(scheme: Any) -> bool:
    """Whether a ``scheme`` asks for one class per code, in any spelling of ``"categorical"``.

    Args:
        scheme: The caller's ``scheme`` — a name, a list of edges, or ``None``.

    Returns:
        ``True`` for ``"categorical"`` in any case; ``False`` for every other scheme, an edge list and ``None``.

    Examples:
        - The spelling is case-insensitive, and an edge list is never categorical:
            ```python
            >>> from digitalearth.base.raster_classes import asks_categorical
            >>> asks_categorical("Categorical"), asks_categorical("quantiles"), asks_categorical([0, 1])
            (True, False, False)

            ```
    """
    return isinstance(scheme, str) and scheme.lower() == "categorical"


def raster_categories(values: Any) -> List[int]:
    """Return a raster band's distinct integer class codes, refusing a band that is not nominal.

    Args:
        values: The band's cell values; ``NaN`` and masked cells are nodata and fall in no class.

    Returns:
        The distinct codes, ascending, as ``int``.

    Raises:
        ValueError: when every cell is nodata, when a value is not a whole number (a magnitude, not a code),
            when there are more than :data:`MAX_RASTER_CATEGORIES` distinct codes, or when a code reaches
            ``2**52`` in magnitude, past which a float cannot place a class edge half a step from it.

    Examples:
        - Codes come back sorted and distinct; a NaN cell is no code:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import raster_categories
            >>> raster_categories(np.array([[3.0, 1.0], [np.nan, 3.0]]))
            [1, 3]

            ```
        - A fractional value is a magnitude, so the band is refused:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import raster_categories
            >>> raster_categories(np.array([0.5, 1.0]))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: scheme='categorical' needs integer class codes ... such as 0.5; ...

            ```
        - A band of more distinct codes than a key can show is refused, pointing at a graduated scheme:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import raster_categories
            >>> raster_categories(np.arange(30))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: scheme='categorical' draws one swatch per distinct code, and this band has 30 (at most 24); ...

            ```
    """
    valid = np.ma.compressed(np.ma.masked_invalid(np.ma.asarray(values, dtype=float)))
    if valid.size == 0:
        raise ValueError(
            "scheme='categorical' found no valid cells to classify: every cell is nodata"
        )
    codes = np.unique(valid)
    fractional = codes[codes != np.round(codes)]
    if fractional.size:
        raise ValueError(
            "scheme='categorical' needs integer class codes (land cover, zone ids), but this band holds "
            f"non-integer values such as {fractional[0]:g}; classify a continuous magnitude with a graduated "
            "scheme ('quantiles', 'equal_interval', ...) instead"
        )
    if codes.size > MAX_RASTER_CATEGORIES:
        raise ValueError(
            f"scheme='categorical' draws one swatch per distinct code, and this band has {codes.size} "
            f"(at most {MAX_RASTER_CATEGORIES}); classify it with a graduated scheme instead"
        )
    # Each class is bounded half a step either side of its code, in floats. From 2**52 on a float cannot hold
    # that half step (and from 2**53 not even adjacent codes), so edges would merge classes; refused instead.
    if np.abs(codes).max() >= _EXACT_CODE_LIMIT:
        raise ValueError(
            "scheme='categorical' bounds each class half a step either side of its code, which a float holds "
            f"only below 2**52 in magnitude; this band has a code of {codes[np.abs(codes).argmax()]:.0f}"
        )
    return [int(code) for code in codes]


def code_edges(codes: Sequence[int]) -> List[float]:
    """Return class edges that put each integer code in a class of its own.

    Half a step below the first code, midway between each pair of neighbours, and half a step above the last
    — so a gap in the codes (``1, 2, 5``) widens a class without letting it swallow a code.

    Args:
        codes: The distinct codes, ascending.

    Returns:
        ``len(codes) + 1`` ascending edges.

    Examples:
        - A gap widens the class beside it, and every code stays inside its own:
            ```python
            >>> from digitalearth.base.raster_classes import code_edges
            >>> code_edges([1, 2, 5])
            [0.5, 1.5, 3.5, 5.5]

            ```
        - A single code gets one class, a step wide:
            ```python
            >>> from digitalearth.base.raster_classes import code_edges
            >>> code_edges([7])
            [6.5, 7.5]

            ```
    """
    middles = [(low + high) / 2 for low, high in zip(codes[:-1], codes[1:])]
    return [codes[0] - 0.5, *middles, codes[-1] + 0.5]


def _is_code(category: Any) -> bool:
    """Whether a recorded category can be a raster cell's class code — a whole number, not a ``bool``.

    Args:
        category: One category of a recorded categorical scale.

    Returns:
        ``True`` for an integer, or a float holding a whole number; ``False`` for a name, a fraction or a
        ``bool`` (which Python counts as an integer, and which no raster band holds as a code).

    Examples:
        - Numbers that are whole are codes; names and fractions are not:
            ```python
            >>> from digitalearth.base.raster_classes import _is_code
            >>> [_is_code(value) for value in (3, 4.0, 2.5, "forest", True)]
            [True, True, False, False, False]

            ```
    """
    if isinstance(category, bool) or not isinstance(category, numbers.Real):
        return False
    return float(category).is_integer()


def classify_band(values: Any, scheme: Any, k: Any, cmap: Any) -> BandClasses:
    """Cut a band into the classes ``scheme`` asks for, and colour each one.

    Args:
        values: The band's cell values; ``NaN`` and masked cells are nodata and are not classified.
        scheme: ``"categorical"`` for one class per integer code, a named graduated scheme
            (``"quantiles"``, ``"equal_interval"``, ``"fisher_jenks"``, …), or an explicit sequence of class
            edges, which is sorted and de-duplicated.
        k: The number of classes a named graduated scheme cuts; ignored by ``"categorical"`` and by explicit
            edges. ``None`` takes
            :data:`~digitalearth.base.spec.scale.DEFAULT_CLASS_COUNT`.
        cmap: The colours: a palette for codes (``None`` takes the shared categorical default), or the
            colormap graduated classes are sampled from — a name, a ``Colormap`` or a list of colours
            (``None`` takes :data:`DEFAULT_CLASS_CMAP`).

    Returns:
        The band's :class:`BandClasses`.

    Raises:
        ValueError: from :func:`raster_categories` for a band that is not nominal, or from
            :meth:`~digitalearth.base.spec.scale.Scale.breaks_of` for an unknown scheme, ``k < 1`` or a band
            with no valid cell.

    Examples:
        - One class per code, coloured from the shared categorical palette:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classify_band
            >>> classes = classify_band(np.array([[1, 5], [2, 1]]), "categorical", None, None)
            >>> list(classes.scale.categories), classes.edges
            ([1, 2, 5], (0.5, 1.5, 3.5, 5.5))

            ```
        - Explicit edges are sorted and de-duplicated:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classify_band
            >>> classify_band(np.arange(25.0), [10, 0, 20, 10], None, "viridis").edges
            (0.0, 10.0, 20.0)

            ```
        - ``k=None`` cuts the default five classes, each sampled from the colormap:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classify_band
            >>> classes = classify_band(np.arange(11.0), "equal_interval", None, "viridis")
            >>> classes.edges, len(classes.colors)
            ((0.0, 2.0, 4.0, 6.0, 8.0, 10.0), 5)

            ```
        - A class count below one is refused:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classify_band
            >>> classify_band(np.arange(10.0), "quantiles", 0, "viridis")
            Traceback (most recent call last):
                ...
            ValueError: scheme='quantiles' with k=0 cannot classify this data: `k` must be >= 1, got 0.

            ```
    """
    if asks_categorical(scheme):
        codes = raster_categories(values)
        categories, colors = categorical_colors(codes, resolve_categorical_cmap(cmap))
        return BandClasses(
            Scale.categorical(categories, colors),
            tuple(code_edges(codes)),
            tuple(colors),
        )
    valid = np.ma.compressed(np.ma.masked_invalid(np.ma.asarray(values, dtype=float)))
    edges = Scale.breaks_of(valid, scheme, DEFAULT_CLASS_COUNT if k is None else k)
    scale = Scale(edges[0], edges[-1], scheme=edges, breaks=edges)
    colors = sample_cmap(DEFAULT_CLASS_CMAP if cmap is None else cmap, len(edges) - 1)
    return BandClasses(scale, edges, tuple(colors))


def classes_of(scale: Optional[Scale], cmap: Any) -> Optional[BandClasses]:
    """Recover a band's classes from the colour scale a figure recorded, without reading the band again.

    This is how a classified raster one tier described draws on another: the classes travel in the layer's
    colour encoding, and each tier colours them the same way :func:`classify_band` would.

    Args:
        scale: The layer's recorded colour scale, or ``None``.
        cmap: The colormap graduated classes are sampled from; unused for a categorical scale, which carries
            its own colours.

    Returns:
        The :class:`BandClasses`, or ``None`` for a continuous scale or none at all — the band is then drawn
        along a ramp, as it always was.

    Examples:
        - A recorded categorical scale gives back its codes' classes and colours:
            ```python
            >>> from digitalearth.base.raster_classes import classes_of
            >>> from digitalearth.base.spec import Scale
            >>> classes = classes_of(Scale.categorical([1, 4], ["#111111", "#222222"]), None)
            >>> classes.edges, classes.colors
            ((0.5, 2.5, 4.5), ('#111111', '#222222'))

            ```
        - A continuous scale has no classes:
            ```python
            >>> from digitalearth.base.raster_classes import classes_of
            >>> from digitalearth.base.spec import Scale
            >>> classes_of(Scale.from_limits(0.0, 1.0), "viridis") is None
            True

            ```
        - A recorded graduated scale gives back its edges, recoloured from `cmap` as the classifier did:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import classes_of, classify_band
            >>> recorded = classify_band(np.arange(10.0), "equal_interval", 3, "viridis").scale
            >>> classes = classes_of(recorded, "viridis")
            >>> classes.edges, classes.colors
            ((0.0, 3.0, 6.0, 9.0), ('#440154', '#21918c', '#fde725'))

            ```
        - Categories named by text are no cell's code, so they give no classes:
            ```python
            >>> from digitalearth.base.raster_classes import classes_of
            >>> from digitalearth.base.spec import Scale
            >>> classes_of(Scale.categorical(["forest", "water"], ["#111111", "#222222"]), None) is None
            True

            ```
    """
    if scale is None:
        return None
    if scale.is_categorical:
        if not all(_is_code(category) for category in scale.categories):
            # A raster cell holds a number, so categories named by anything else — "forest", "water" — have
            # no cells to colour; the band is drawn along its ramp, as a scale it cannot classify by.
            return None
        pairs = sorted(
            (int(code), str(scale.color_for(code))) for code in scale.categories
        )
        codes = [code for code, _ in pairs]
        return BandClasses(
            scale, tuple(code_edges(codes)), tuple(color for _, color in pairs)
        )
    if scale.is_classified:
        edges = tuple(float(edge) for edge in scale.breaks)
        colors = sample_cmap(
            DEFAULT_CLASS_CMAP if cmap is None else cmap, len(edges) - 1
        )
        return BandClasses(scale, edges, tuple(colors))
    return None


def class_index(values: Any, edges: Sequence[float]) -> np.ndarray:
    """Return the class each cell falls in, ``-1`` for a cell with no value.

    A cell below the first edge takes the first class and one above the last takes the last — the end
    colours, as matplotlib's ``BoundaryNorm`` gives the static tier — so the tiers agree on cells an explicit
    list of edges does not reach.

    Args:
        values: The band's cell values; ``NaN`` and masked cells are nodata.
        edges: The ascending class edges.

    Returns:
        An ``int`` array shaped like `values`.

    Examples:
        - Each cell lands in its class; out-of-range cells take the end classes, and NaN none:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import class_index
            >>> class_index(np.array([-5.0, 0.5, 1.5, 9.0, np.nan]), [0.0, 1.0, 2.0])
            array([ 0,  0,  1,  1, -1])

            ```
        - A masked cell is nodata too, whatever value sits under the mask:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.raster_classes import class_index
            >>> class_index(np.ma.masked_array([1.0, 2.0], mask=[True, False]), [0.0, 1.5, 3.0])
            array([-1,  1])

            ```
    """
    array = np.ma.masked_invalid(np.ma.asarray(values, dtype=float))
    data = array.filled(np.nan)
    classes = len(edges) - 1
    index = np.clip(
        np.searchsorted(np.asarray(edges[1:-1]), data, side="right"), 0, classes - 1
    )
    return np.where(np.isfinite(data), index, -1)
