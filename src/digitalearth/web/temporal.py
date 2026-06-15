"""TemporalMixin — web-tier temporal builder (DW.5).

Empty skeleton: DW.5 adds ``timeslider`` (recipe W6 — an anywidget slider whose value swaps the active
source per ``DatasetCollection`` member, or sets a MapLibre filter on a time field, with a shared colour
range). Registers via ``self.add_layer``.
"""


class TemporalMixin:
    """Temporal builder for :class:`~digitalearth.web.map.WebMap` (populated in DW.5)."""
