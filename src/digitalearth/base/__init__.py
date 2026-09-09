"""base — the engine-neutral layer every rendering backend shares.

Everything in this subpackage is used by two or more of the four backends (:mod:`digitalearth.static` /
matplotlib, :mod:`digitalearth.interactive` / HoloViz, :mod:`digitalearth.three_d` / PyVista and
:mod:`digitalearth.web` / MapLibre + deck.gl), so it must stay free of any renderer import: matplotlib,
cleopatra, pyvista, geovista, holoviews, geoviews, hvplot, panel, datashader, bokeh, maplibre, lonboard. Only
numpy/pandas, PyYAML, pyramids and the standard library belong here — and never a sibling backend
either, since importing one pulls its engine in behind it.

The one carve-out is matplotlib's **colour** registry — ``matplotlib.colors`` / ``matplotlib.colormaps`` /
``matplotlib.cm`` return plain data (a named-colormap lookup, hex conversion), never figures, and are what let
:mod:`~digitalearth.base.symbology` hand every backend the *same* colours for one ``cmap``. ``pyplot``,
``figure``, ``axes`` and the matplotlib backends stay banned, as does a bare ``import matplotlib``. The whole
rule — bans and carve-out — is enforced by ``tests/test_base_is_engine_neutral.py``, which walks these modules
with :mod:`ast` rather than importing them, so it also catches a lazy import inside a function.

What lives here:

* :mod:`~digitalearth.base.sources` — the uniform ``Source`` / ``DimensionInfo`` view over pyramids inputs
  (``get_source`` single-band, ``get_stack`` multiband); the extraction seam every backend reads through.
* :mod:`~digitalearth.base.autostyle` — variable/units → style resolution, including the ECMWF-Magics
  operational identity matching.
* :mod:`~digitalearth.base.symbology` — categorical (distinct value → colour) symbology, and the ``cmap``
  sentinel all tiers resolve through.
* :mod:`~digitalearth.base.chartdata` — chart data preparation: a column name or array-like → plottable values.
* :mod:`~digitalearth.base.arrays` — nodata masking and the finite-value reductions.
* :mod:`~digitalearth.base.crs` — the best-effort EPSG lookup.
* :mod:`~digitalearth.base.preprocess` — longitude wrapping and the cyclic column for global fields.
* :mod:`~digitalearth.base.stretch` — the composite contrast stretch: per-channel bounds every backend
  shares, so a true-colour render is identical across tiers and a sequence of frames can be frozen on
  one stretch.
* :mod:`~digitalearth.base.types` — the ``RasterLike`` / ``VectorLike`` structural Protocols.

Anything a *single* backend needs belongs in that backend instead: the matplotlib figure lookup that used to sit
beside :mod:`~digitalearth.base.arrays` lives in :mod:`digitalearth.static.figures` for exactly that reason.
"""
