"""Export modelled, not merely described (IN-10).

``export_plan`` states what a ``save_app(embed=True)`` would bake — the per-widget state counts and their
product — against a cap, so a caller finds out whether the offline file survives the trip before writing it.
"""

import pytest

pytest.importorskip("geoviews")
pytest.importorskip("panel")

from digitalearth.interactive import InteractiveMap  # noqa: E402


@pytest.fixture
def m(dataset) -> InteractiveMap:
    """A Web-Mercator map carrying one raster layer.

    Args:
        dataset: The raster fixture.

    Returns:
        The map.
    """
    return InteractiveMap().field(dataset)


class TestExportPlan:
    """The plan counts the baked states and judges them against the cap."""

    def test_the_state_count_is_the_product_of_the_widgets(self, m):
        """A colormap Select (8 options) and a sampled opacity slider (3) make 24 states.

        Args:
            m: The map fixture.
        """
        plan = m.export_plan(widgets=("cmap", "alpha"))
        assert plan.total_states == 24
        assert plan.embeddable

    def test_an_oversize_export_is_flagged_with_a_breakdown(self, m):
        """Past the cap the plan is not embeddable and its warning names the counts.

        Args:
            m: The map fixture.
        """
        plan = m.export_plan(widgets=("cmap",), max_states=5)
        assert not plan.embeddable
        assert plan.warning is not None
        assert "cmap:8" in plan.warning

    def test_within_the_cap_there_is_no_warning(self, m):
        """A plan within the cap carries no warning.

        Args:
            m: The map fixture.
        """
        assert m.export_plan(widgets=("alpha",)).warning is None

    def test_export_plan_reuses_the_widget_vocabulary(self, m):
        """An unknown widget is refused by ``export_plan`` the same way ``dashboard`` refuses it.

        Args:
            m: The map fixture.
        """
        with pytest.raises(ValueError, match="unknown dashboard widget"):
            m.export_plan(widgets=("bogus",))
