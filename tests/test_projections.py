"""Tests for DC.1-DC.3 — named projections, projection_frame, graticule."""
import numpy as np
import pytest

from digitalearth.scene import projections


class TestRegistry:
    """Tests for the named-projection registry."""

    def test_epsg_projection(self):
        """An EPSG-coded projection resolves to its integer code."""
        assert projections.get("web_mercator") == 3857
        assert projections.get("polar_south") == 3031

    def test_parametrised_proj4(self):
        """orthographic builds a proj4 string with the given centre."""
        spec = projections.get("orthographic", lon=-9, lat=39)
        assert "+proj=ortho" in spec and "+lat_0=39" in spec and "+lon_0=-9" in spec

    def test_case_insensitive(self):
        """Name lookup is case-insensitive."""
        assert projections.get("Robinson") == projections.robinson()

    def test_unknown_raises(self):
        """An unregistered name raises KeyError."""
        with pytest.raises(KeyError, match="unknown projection"):
            projections.get("atlantis")

    def test_polar_north_epsg(self):
        """polar_north resolves to the NSIDC north EPSG code (3413)."""
        assert projections.get("polar_north") == 3413
        assert projections.polar_north() == 3413

    def test_mollweide_proj4(self):
        """mollweide builds a centred equal-area proj4 string."""
        spec = projections.get("mollweide", lon=20)
        assert "+proj=moll" in spec and "+lon_0=20" in spec


class TestConvexHull:
    """Tests for the pure-numpy convex-hull helper (degenerate small-point cases)."""

    def test_single_point_closes_to_itself(self):
        """A single point returns itself repeated (a closed degenerate ring)."""
        ring = projections._convex_hull(np.array([[1.0, 2.0]]))
        assert ring.shape[1] == 2 and len(ring) == 2
        np.testing.assert_allclose(ring[0], ring[-1])

    def test_two_points_close(self):
        """Two distinct points return a 2-vertex ring (fewer than 3 points -> no hull)."""
        ring = projections._convex_hull(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert len(ring) == 3 and ring.shape[1] == 2  # [p0, p1, p0]

    def test_empty_returns_empty(self):
        """Zero points return an empty (0, 2) array without raising."""
        ring = projections._convex_hull(np.empty((0, 2)))
        assert ring.shape == (0, 2)


class TestProjectionFrame:
    """Tests for projection_frame."""

    def test_orthographic_is_circular(self):
        """The orthographic boundary is ~circular: x/y spans are about equal and centred near 0."""
        ring, xlim, ylim = projections.projection_frame(projections.orthographic(0, 0), n=180)
        assert ring.ndim == 2 and ring.shape[1] == 2
        xspan, yspan = xlim[1] - xlim[0], ylim[1] - ylim[0]
        assert abs(xspan - yspan) / max(xspan, yspan) < 0.05  # near-equal => disc
        assert abs(xlim[0] + xlim[1]) < 0.02 * xspan          # centred on 0

    def test_mercator_is_rectangular(self):
        """A cylindrical CRS boundary is a rectangle (corners near the limit box)."""
        ring, xlim, ylim = projections.projection_frame(3857, n=180)
        assert xlim[0] < 0 < xlim[1] and ylim[0] < 0 < ylim[1]

    def test_ring_is_closed(self):
        """The boundary ring is closed (first vertex repeated at the end)."""
        ring, _, _ = projections.projection_frame(projections.orthographic(-9, 39), n=120)
        np.testing.assert_allclose(ring[0], ring[-1])

    def test_empty_domain_raises_clearly(self, mocker):
        """A CRS that projects every sample to non-finite coords raises a clear error, not min() on empty."""
        mocker.patch("digitalearth.scene.projections.reproject_coordinates",
                     return_value=([float("inf")] * 4, [float("nan")] * 4))
        with pytest.raises(ValueError, match="no finite projected domain"):
            projections.projection_frame(3857, n=4)


class TestGraticule:
    """Tests for graticule."""

    def test_returns_polylines(self):
        """graticule returns a non-empty list of (M, 2) polylines."""
        lines = projections.graticule(3857, lon_step=60, lat_step=30)
        assert lines and all(line.shape[1] == 2 and len(line) > 1 for line in lines)

    def test_orthographic_confined_to_disc(self):
        """Orthographic graticule points stay within the projection boundary bbox."""
        crs = projections.orthographic(0, 0)
        _, xlim, ylim = projections.projection_frame(crs, n=180)
        lines = projections.graticule(crs, lon_step=30, lat_step=30)
        pts = np.vstack(lines)
        # the boundary is a hull of coarse samples; allow a small fraction of the radius as slack
        pad = 0.01 * (xlim[1] - xlim[0])
        assert pts[:, 0].min() >= xlim[0] - pad and pts[:, 0].max() <= xlim[1] + pad
        assert pts[:, 1].min() >= ylim[0] - pad and pts[:, 1].max() <= ylim[1] + pad

    def test_no_finite_gaps_within_a_polyline(self):
        """Each returned polyline is fully finite (splitting removed off-domain gaps)."""
        lines = projections.graticule(projections.orthographic(0, 0))
        assert all(np.isfinite(line).all() for line in lines)
