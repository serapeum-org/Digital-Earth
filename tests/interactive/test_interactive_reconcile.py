"""The one reconcile path every widget event takes (IN-2).

The dashboard's cmap/alpha/basemap widgets and the layer switcher's visibility/opacity widgets used to
rebuild the overlay three different ways. They route through one
:meth:`~digitalearth.interactive.dashboard.DashboardMixin._reconcile_view` now, which re-styles only the
layers a widget value reaches and hands every other layer's element back untouched — a reconcile, not a
rebuild.
"""

import pytest

pytest.importorskip("geoviews")
pytest.importorskip("panel")

from digitalearth.interactive import InteractiveMap  # noqa: E402


@pytest.fixture
def point_fc():
    """Return the point fixture as a pyramids ``FeatureCollection``.

    Returns:
        A ``FeatureCollection`` of points.
    """
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def two_layers(dataset, point_fc):
    """Return a Web-Mercator map carrying a raster field and a point overlay.

    Args:
        dataset: The raster fixture.
        point_fc: The point ``FeatureCollection`` fixture.

    Returns:
        An ``InteractiveMap`` with two layers.
    """
    return InteractiveMap().field(dataset).points(point_fc)


def _points_index(layers) -> int:
    """Return the position of the ``Points`` element in a layer list.

    Args:
        layers: The map's composed elements.

    Returns:
        The index of the (non colour-mapped) points layer.
    """
    return next(i for i, layer in enumerate(layers) if type(layer).__name__ == "Points")


class TestReconcileReusesElements:
    """A widget value re-styles the layers it reaches and reuses the rest (not a full rebuild)."""

    def test_an_untouched_layer_keeps_its_element(self, two_layers):
        """An alpha override reaches the raster but not the points, so the points element is reused.

        Args:
            two_layers: The map under test.
        """
        before = list(two_layers.layers)
        restyled = two_layers._restyled_layers({"alpha": 0.5})
        idx = _points_index(before)
        assert restyled[idx] is before[idx]

    def test_no_override_reuses_every_element(self, two_layers):
        """A basemap-only change carries no style override, so nothing is re-``.opts()``'d.

        Args:
            two_layers: The map under test.
        """
        before = list(two_layers.layers)
        view = two_layers._reconcile_view(before, {})
        # Every element composed is the very object the map already held.
        assert [element for element in view] == before


class TestOneComposePath:
    """Both widget entry points go through the single reconcile, so there is one recipe to fix."""

    def test_both_widget_methods_delegate_to_reconcile(self, two_layers, monkeypatch):
        """``_render_with_overrides`` and ``_compose_visible_layers`` both call ``_reconcile_view``.

        Args:
            two_layers: The map under test.
            monkeypatch: Pytest's monkeypatch, to count the shared path's calls.
        """
        calls = []
        original = two_layers._reconcile_view

        def _counting(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(two_layers, "_reconcile_view", _counting)
        two_layers._render_with_overrides({"alpha": 0.5})
        two_layers._compose_visible_layers(list(two_layers.layer_ids), op=0.5)
        assert len(calls) == 2
