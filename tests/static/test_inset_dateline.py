"""A dateline-crossing locator extent is supported by recentring the locator, not refused (#395).

`Map.inset(extent=[west, south, east, north])` reads a `west > east` pair as a dateline-crossing region on a
geographic CRS — a Pacific-centred overview such as `[170, -40, -170, 40]` — and draws it on an
equidistant-cylindrical CRS recentred over the crossing, so the two longitude bands `[west, 180]` and
`[-180, east]` sit either side of the frame instead of wrapping. On a projected CRS the same pair has no
antimeridian to wrap around and stays the refusal it was.
"""

import matplotlib
import pytest

matplotlib.use("Agg")

from digitalearth.static import Map  # noqa: E402 - after the Agg backend is set

DATELINE = [170.0, -40.0, -170.0, 40.0]


class TestADatelineCrossingExtentIsDrawn:
    """A geographic locator given `west > east` recentres onto an equidistant-cylindrical CRS."""

    def test_the_locator_is_built_on_a_recentred_eqc_crs(self):
        """The locator's CRS is an equidistant-cylindrical projection centred on the crossing.

        Test scenario:
            `[170, -40, -170, 40]` straddles the antimeridian; its middle longitude is 180, so the locator
            is drawn on `+proj=eqc +lon_0=180`, where the region is one contiguous span rather than two that
            wrap.
        """
        with Map(crs=4326) as m:
            m.set_bounds([120.0, -10.0, 160.0, 10.0])
            m.inset(extent=DATELINE)
            crs = str(m.locator.crs)
            assert "eqc" in crs and "lon_0=180" in crs, (
                f"the locator was not recentred onto an eqc CRS over the crossing: {crs!r}"
            )

    def test_the_frame_is_a_contiguous_west_to_east_span(self):
        """The recentred frame runs west-to-east across the centre, not inverted.

        Test scenario:
            After recentring, the extent's corners reproject to a true `x_west < x_east` box, so the
            locator's x-limits are a normal increasing pair symmetric about the centre meridian.
        """
        with Map(crs=4326) as m:
            m.set_bounds([120.0, -10.0, 160.0, 10.0])
            m.inset(extent=DATELINE)
            xlim = [float(v) for v in m.locator.ax.get_xlim()]
            assert xlim[0] < 0 < xlim[1], (
                f"the recentred frame is not a contiguous span: {xlim}"
            )

    def test_reference_geography_is_drawn(self):
        """Reference coastlines are drawn into the recentred locator rather than left empty.

        Test scenario:
            The point of recentring is that geography draws across the seam; the locator holds at least one
            coastline collection once it is built.
        """
        with Map(crs=4326) as m:
            m.set_bounds([120.0, -10.0, 160.0, 10.0])
            m.inset(extent=DATELINE)
            assert len(m.locator.ax.collections) >= 1, (
                "the recentred locator drew no reference geography"
            )


class TestTheOrdinaryInsetPathsAreUnaffected:
    """`inset()` without a crossing extent still frames normally, with no recentring."""

    def test_no_extent_shows_the_whole_projection_domain(self):
        """`inset()` with no extent builds a locator on the map's own CRS, framed to the whole domain.

        Test scenario:
            The default (no `extent`) path skips the recentring branch entirely: the locator is drawn on the
            map's CRS and shown globally, so its CRS is never an equidistant-cylindrical recentring.
        """
        with Map(crs=4326) as m:
            m.set_bounds([120.0, -10.0, 160.0, 10.0])
            m.inset()
            assert m.locator is not None and "eqc" not in str(m.locator.crs), (
                f"a no-extent inset recentred or failed to build: {m.locator!r}"
            )

    def test_an_ordinary_extent_keeps_the_map_crs(self):
        """A plain `west < east` extent frames the locator without recentring it.

        Test scenario:
            `[130, -5, 150, 5]` is an ordinary rectangle, so `_checked_extent` returns no recentring CRS and
            the locator keeps the map's CRS rather than an equidistant-cylindrical one.
        """
        with Map(crs=4326) as m:
            m.set_bounds([120.0, -10.0, 160.0, 10.0])
            m.inset(extent=[130.0, -5.0, 150.0, 5.0])
            assert "eqc" not in str(m.locator.crs), (
                f"an ordinary extent recentred onto an eqc CRS: {m.locator.crs!r}"
            )


class TestTheRefusalsThatStand:
    """A `west > east` pair without an antimeridian, and a flipped latitude, are still refused."""

    def test_a_crossing_the_projection_cannot_place_is_refused(self):
        """A recentred box that collapses to zero width is refused, naming the failed recentring.

        Test scenario:
            A pair whose longitudes coincide reprojects to a zero-width box on the equidistant-cylindrical
            CRS, so `_recentre_dateline` cannot frame a `west < east` rectangle and refuses rather than
            drawing an empty locator.
        """
        with Map(crs=4326) as m:
            coincident = [10.0, -40.0, 10.0, 40.0]
            with pytest.raises(ValueError, match=r"could not be recentred"):
                m._recentre_dateline(coincident, 4326)

    def test_a_projected_crs_refuses_west_over_east(self):
        """On a projected CRS `west > east` is an inverted rectangle, not a crossing.

        Test scenario:
            EPSG:3857 has no antimeridian wrap, so the same values are a caller's mistake and are refused by
            name — the hardening #394 added, kept for the projected case.
        """
        with Map(crs=3857) as m:
            with pytest.raises(ValueError, match=r"east has to be above west"):
                m.inset(extent=DATELINE)

    def test_a_flipped_latitude_is_refused_everywhere(self):
        """`north <= south` frames no area whatever the CRS.

        Test scenario:
            A flipped or zero-height latitude pair is never a crossing, so it is refused on a geographic CRS
            too, naming the north/south edge.
        """
        with Map(crs=4326) as m:
            with pytest.raises(ValueError, match=r"north has to be above south"):
                m.inset(extent=[10.0, 50.0, 20.0, 10.0])


class TestCheckedExtentResolution:
    """`_checked_extent` returns the recentred box only for a geographic crossing."""

    def test_an_ordinary_rectangle_keeps_its_crs(self):
        """A normal `west < east` extent is returned unchanged, with no recentring CRS.

        Test scenario:
            The common case must not recentre: the box comes back as given and the second element is
            ``None``, so the locator keeps the CRS it was built with.
        """
        with Map(crs=4326) as m:
            box, recentre = m._checked_extent([10.0, 40.0, 20.0, 60.0], 4326)
            assert box == [10.0, 40.0, 20.0, 60.0], f"the box was altered: {box}"
            assert recentre is None, (
                f"an ordinary rectangle recentred onto {recentre!r}"
            )

    def test_a_crossing_returns_a_box_in_the_recentred_crs(self):
        """A dateline crossing returns `(box, eqc_crs)` with the box a contiguous west<east pair.

        Test scenario:
            The returned box is in the recentred CRS (metres, not the lon/lat degrees passed in), so
            `x_west < x_east`, and the second element is the eqc CRS the locator must use.
        """
        with Map(crs=4326) as m:
            box, recentre = m._checked_extent(DATELINE, 4326)
            assert recentre is not None and "eqc" in str(recentre), (
                f"a crossing did not recentre: {recentre!r}"
            )
            assert box[0] < box[2] and box[1] < box[3], (
                f"the recentred box is not a contiguous west<east/south<north rectangle: {box}"
            )

    def test_a_crossing_past_the_antimeridian_folds_the_centre_back(self):
        """A crossing whose midpoint lands beyond +180 folds the centre meridian back into range.

        Test scenario:
            `[179, -40, -170, 40]` has a midpoint longitude of 184.5, past the antimeridian; the centre is
            folded by 360 to -175.5, so the locator is recentred onto `+proj=eqc +lon_0=-175.5` rather than
            an out-of-range meridian.
        """
        with Map(crs=4326) as m:
            box, recentre = m._checked_extent([179.0, -40.0, -170.0, 40.0], 4326)
            assert recentre is not None and "lon_0=-175.5" in str(recentre), (
                f"the past-antimeridian centre was not folded back: {recentre!r}"
            )
            assert box[0] < box[2], f"the folded crossing is not a contiguous span: {box}"
