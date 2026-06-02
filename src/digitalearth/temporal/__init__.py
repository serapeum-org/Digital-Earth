"""temporal — time-series and climatology products over pyramids time data.

Orchestration only: per-time-step reduction of a pyramids ``DatasetCollection`` into a 1-D series, rendered
through cleopatra ``LineGlyph`` (and :mod:`digitalearth.series` for climatology plumes). The heavy GIS/time
work stays in pyramids; the numeric reduction here is plain array aggregation for plotting.
"""
from digitalearth.temporal.climatology import Climatology
from digitalearth.temporal.timeseries import TimeSeries

__all__ = ["TimeSeries", "Climatology"]
