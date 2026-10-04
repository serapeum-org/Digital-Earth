"""``facet`` — a raster stack as small multiples on one shared colour scale (ST-8, #223).

Small multiples are read against each other, so every panel has to colour one value the same way: the scale
is resolved **once over the whole stack**, not per panel. These pin that, the panel layout (``col_wrap``),
the per-panel titles and the one colorbar spanning the panels. Expected limits and class edges are computed
from the arrays the test builds, never through the code under test.
"""

import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm, to_hex
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map, facet, projections

#: Four 4 x 4 frames whose values climb frame by frame, so a per-panel scale would differ on every panel.
FRAMES = [np.arange(16, dtype="float64").reshape(4, 4) + 10.0 * i for i in range(4)]

#: Two frames of class codes that share no code, so a per-panel categorical scale paints them the same way.
CODE_FRAMES = [
    np.array([[1, 2], [2, 1]], dtype="float64"),
    np.array([[3, 4], [4, 3]], dtype="float64"),
]

#: The lon/lat placement every frame shares.
GEO = GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)

#: The same 4 x 4 footprint moved to lon 170-174, which is behind a globe centred on lon 0.
GEO_FAR_SIDE = GeoReference(geo=(170.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)


def _dataset(arr: np.ndarray, geo_ref: GeoReference = GEO) -> Dataset:
    """Wrap one array as a lon/lat dataset.

    Args:
        arr: A 2-D frame, or a 3-D ``(bands, rows, cols)`` stack.
        geo_ref: Where it sits. The shared lon/lat placement by default.

    Returns:
        A pyramids `Dataset`.
    """
    return Dataset.from_array(arr=arr, geo_ref=geo_ref, no_data_value=-9999.0)


@pytest.fixture
def stack():
    """Four single-band frames.

    Returns:
        A list of pyramids `Dataset`.
    """
    return [_dataset(arr) for arr in FRAMES]


@pytest.fixture
def code_stack():
    """Two single-band frames of class codes that share no code.

    Returns:
        A list of pyramids `Dataset`.
    """
    return [_dataset(arr) for arr in CODE_FRAMES]


def _code_colors(artist, codes) -> dict:
    """Return the colour one drawn panel paints each class code in.

    Args:
        artist: The panel's mappable.
        codes: The class codes to look up.

    Returns:
        A ``{code: "#rrggbb"}`` mapping, read through the artist's own norm and colormap.
    """
    return {int(code): to_hex(artist.cmap(artist.norm(float(code)))) for code in codes}


def _colorbars(fig) -> list:
    """Return the colorbar axes on a figure.

    Args:
        fig: The figure.

    Returns:
        The axes matplotlib created for a colorbar.
    """
    return [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]


class TestOneSharedScale:
    """Every panel colours one value the same way."""

    def test_one_map_per_frame_on_one_figure(self, stack):
        """Four frames, four panels, one figure.

        Args:
            stack: The frames.
        """
        fig, maps = facet(stack, crs=4326)
        assert len(maps) == 4, len(maps)
        assert all(isinstance(m, Map) and m.fig is fig for m in maps), maps

    def test_every_panel_shares_the_stack_s_limits(self, stack):
        """Each panel's colour limits are the whole stack's minimum and maximum.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326)
        expected = (
            float(min(a.min() for a in FRAMES)),
            float(max(a.max() for a in FRAMES)),
        )
        clims = [m.layers[-1][1].get_clim() for m in maps]
        assert clims == [expected] * 4, clims

    def test_explicit_limits_win(self, stack):
        """A caller's ``vmin``/``vmax`` are the shared scale.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326, vmin=5.0, vmax=25.0)
        clims = {m.layers[-1][1].get_clim() for m in maps}
        assert clims == {(5.0, 25.0)}, clims

    def test_a_scheme_is_cut_once_over_the_stack(self, stack):
        """``scheme=``/``k=`` classify every panel with the same edges, cut over all the frames.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326, scheme="equal_interval", k=4)
        lo, hi = min(a.min() for a in FRAMES), max(a.max() for a in FRAMES)
        expected = np.linspace(lo, hi, 5)
        for m in maps:
            norm = m.layers[-1][1].norm
            assert isinstance(norm, BoundaryNorm), type(norm)
            assert np.allclose(norm.boundaries, expected), norm.boundaries

    def test_filled_contours_share_the_levels_given(self, stack):
        """``kind="contourf"`` takes explicit levels, which every panel draws.

        Args:
            stack: The frames.
        """
        levels = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        _, maps = facet(stack, crs=4326, kind="contourf", levels=levels)
        for m in maps:
            assert list(m.layers[-1][1].levels) == levels, m.layers[-1][1].levels

    def test_filled_contours_without_levels_are_refused(self, stack):
        """Each panel would pick its own levels from its own frame, which is the bug facets exist to avoid.

        Args:
            stack: The frames.
        """
        with pytest.raises(ValueError, match="levels"):
            facet(stack, crs=4326, kind="contourf")

    def test_limits_passed_as_none_are_still_filled_from_the_stack(self, stack):
        """``vmin=None``/``vmax=None`` is the default spelt out, not a request for per-panel limits.

        Args:
            stack: The frames.

        Test scenario:
            A caller forwarding its own optional arguments passes ``vmin=None`` rather than omitting it,
            and ``dict.setdefault`` leaves a key that is present and ``None`` exactly as it found it. Were
            the explicit ``None`` not filled in as well, each panel would scale itself off its own frame —
            the one thing a facet exists to prevent — so the limits must come out the same as when the
            keywords are left off altogether.
        """
        _, maps = facet(stack, crs=4326, vmin=None, vmax=None)
        expected = (
            float(min(a.min() for a in FRAMES)),
            float(max(a.max() for a in FRAMES)),
        )
        clims = {m.layers[-1][1].get_clim() for m in maps}
        assert clims == {expected}, (
            f"an explicit vmin=None/vmax=None must pool the stack to {expected}, got {clims}"
        )

    def test_a_kind_that_is_not_a_render_is_refused(self, stack):
        """``kind=`` names one of the four renders; anything else is refused by name.

        Args:
            stack: The frames.

        Test scenario:
            The refusal arrives before any axes are made, and it lists what is on offer — a misspelt
            render must not fall through to ``getattr(panel, ...)`` and fail as a missing attribute.
        """
        with pytest.raises(ValueError, match="is not a render"):
            facet(stack, crs=4326, kind="heatmap")

    def test_each_panel_describes_its_layer(self, stack):
        """A panel is an ordinary ``Map``: its frame is a described raster layer.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326)
        kinds = [[m.figure_spec.layers.get(i).kind for i in m.layer_ids] for m in maps]
        assert kinds == [["raster"]] * 4, kinds


class TestTheLayout:
    """One row by default; ``col_wrap`` folds it."""

    def test_one_row_by_default(self, stack):
        """Four frames side by side.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326)
        geometry = {m.ax.get_subplotspec().get_geometry()[:2] for m in maps}
        assert geometry == {(1, 4)}, geometry

    def test_col_wrap_folds_the_row(self, stack):
        """``col_wrap=3`` puts four frames on two rows of three, the last slots hidden.

        Args:
            stack: The frames.
        """
        fig, maps = facet(stack, crs=4326, col_wrap=3)
        geometry = {m.ax.get_subplotspec().get_geometry()[:2] for m in maps}
        assert geometry == {(2, 3)}, geometry
        panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        hidden = [ax for ax in panels if not ax.get_visible()]
        assert (len(panels), len(hidden)) == (6, 2), (len(panels), len(hidden))

    def test_a_col_wrap_below_one_is_refused(self, stack):
        """Zero columns is no layout.

        Args:
            stack: The frames.
        """
        with pytest.raises(ValueError, match="col_wrap"):
            facet(stack, crs=4326, col_wrap=0)

    def test_an_empty_stack_is_refused(self):
        """No frames, no panels."""
        with pytest.raises(ValueError, match="no frames"):
            facet([], crs=4326)


class TestTitlesAndKey:
    """Each panel says which frame it is; one bar keys them all."""

    def test_panels_are_titled_by_index(self, stack):
        """With no labels, a panel is titled ``"<col> = <index>"``.

        Args:
            stack: The frames.
        """
        _, maps = facet(stack, crs=4326, col="time")
        titles = [m.ax.get_title() for m in maps]
        assert titles == ["time = 0", "time = 1", "time = 2", "time = 3"], titles

    def test_panels_are_titled_by_label(self, stack):
        """``labels=`` names each frame.

        Args:
            stack: The frames.
        """
        _, maps = facet(
            stack, crs=4326, col="month", labels=["Jan", "Feb", "Mar", "Apr"]
        )
        titles = [m.ax.get_title() for m in maps]
        assert titles == ["month = Jan", "month = Feb", "month = Mar", "month = Apr"], (
            titles
        )

    def test_labels_that_do_not_number_the_frames_are_refused(self, stack):
        """Three labels for four frames would title a panel with the wrong frame's name.

        Args:
            stack: The frames.
        """
        with pytest.raises(ValueError, match="labels"):
            facet(stack, crs=4326, labels=["Jan", "Feb", "Mar"])

    def test_one_colorbar_spans_the_panels(self, stack):
        """Exactly one bar, labelled as asked.

        Args:
            stack: The frames.
        """
        fig, _ = facet(stack, crs=4326, cbar_label="mm")
        bars = _colorbars(fig)
        assert len(bars) == 1, len(bars)
        assert bars[0].get_ylabel() == "mm", bars[0].get_ylabel()

    def test_the_bar_can_be_left_off(self, stack):
        """``colorbar=False`` draws none.

        Args:
            stack: The frames.
        """
        fig, _ = facet(stack, crs=4326, colorbar=False)
        assert _colorbars(fig) == [], _colorbars(fig)


class TestABandStack:
    """A multi-band dataset facets over its bands."""

    def test_one_panel_per_band_on_one_scale(self):
        """Three bands, three panels titled by band number, one shared scale."""
        cube = np.stack(FRAMES[:3])
        _, maps = facet(_dataset(cube), crs=4326)
        titles = [m.ax.get_title() for m in maps]
        clims = {m.layers[-1][1].get_clim() for m in maps}
        assert titles == ["band = 1", "band = 2", "band = 3"], titles
        assert clims == {(float(cube.min()), float(cube.max()))}, clims


class TestAFrameTheDisplayCrsCannotPlace:
    """A frame off the limb draws nothing, so it bounds nothing either."""

    def test_an_off_limb_frame_is_left_out_of_the_shared_scale(self):
        """On a globe centred on lon 0, a frame at lon 170 is behind it and contributes no limits.

        Test scenario:
            The shared scale is pooled over the frames that actually land. A frame the projection cannot
            place draws no panel at all, and letting its values into the pool would stretch every visible
            panel's colours over a range nothing on the figure shows. The far frame's values sit 100 above
            the near frame's, so a pooled scale would be unmistakable: the near panel's limits must stay
            the near frame's own.
        """
        near = _dataset(FRAMES[0])
        far = _dataset(FRAMES[0] + 100.0, GEO_FAR_SIDE)
        _, maps = facet(
            [near, far], crs=projections.orthographic(lon=0, lat=0), globe=True
        )
        expected = (float(FRAMES[0].min()), float(FRAMES[0].max()))
        assert maps[0].layers[-1][1].get_clim() == expected, (
            f"the near panel must scale to the frames that land, {expected}, got "
            f"{maps[0].layers[-1][1].get_clim()}"
        )
        assert not maps[1].layers, (
            f"the far-side panel draws nothing, got {maps[1].layers}"
        )


class TestAPooledCategoricalScale:
    """``scheme="categorical"`` is pooled over the stack, so one code is one colour on every panel."""

    def test_the_class_codes_are_pooled_over_every_frame(self, code_stack):
        """Every panel is cut by the union of the stack's codes, not by its own frame's.

        Args:
            code_stack: Two frames of disjoint class codes.

        Test scenario:
            A categorical scheme used to be forwarded to each panel untouched, so each panel derived its
            class codes from its own frame: the frames ``{1, 2}`` and ``{3, 4}`` gave panel 0 the boundaries
            ``[0.5, 1.5, 2.5]`` and panel 1 ``[2.5, 3.5, 4.5]``. The codes are pooled instead, so both
            panels carry the one set of edges the whole stack cuts.
        """
        codes = np.unique(np.concatenate([arr.ravel() for arr in CODE_FRAMES]))
        expected = [*(code - 0.5 for code in codes), codes[-1] + 0.5]
        _, maps = facet(code_stack, crs=4326, scheme="categorical")
        for panel in maps:
            norm = panel.layers[-1][1].norm
            assert isinstance(norm, BoundaryNorm), type(norm)
            assert np.allclose(norm.boundaries, expected), norm.boundaries

    def test_one_code_is_one_colour_on_every_panel(self, code_stack):
        """A code painted on two panels is painted the same colour, and two codes are told apart.

        Args:
            code_stack: Two frames of disjoint class codes.

        Test scenario:
            This is the failure a facet exists to prevent, and the one the per-panel categorical scale
            produced: with disjoint codes, class 1 (panel 0's first code) and class 3 (panel 1's first
            code) both came out as the palette's first colour, so two different classes read as one. The
            pooled scale has to give the whole stack's codes distinct colours and agree on them panel by
            panel.
        """
        codes = [
            int(code)
            for code in np.unique(np.concatenate([a.ravel() for a in CODE_FRAMES]))
        ]
        _, maps = facet(code_stack, crs=4326, scheme="categorical")
        painted = [_code_colors(panel.layers[-1][1], codes) for panel in maps]
        assert painted[0] == painted[1], painted
        assert len(set(painted[0].values())) == len(codes), painted[0]

    def test_the_spanning_bar_describes_every_panel(self, code_stack):
        """The one bar is keyed by the pooled edges, so it labels panel 1 as truly as panel 0.

        Args:
            code_stack: Two frames of disjoint class codes.

        Test scenario:
            ``facet`` keys the spanning bar off the *first* panel's artist, which is only a fair
            representative once every panel shares one norm. With a per-panel categorical scale the bar ran
            0.5-2.5 — panel 0's range — and mislabelled panel 1 entirely.
        """
        codes = np.unique(np.concatenate([arr.ravel() for arr in CODE_FRAMES]))
        expected = [*(code - 0.5 for code in codes), codes[-1] + 0.5]
        fig, _ = facet(code_stack, crs=4326, scheme="categorical")
        bars = _colorbars(fig)
        assert len(bars) == 1, len(bars)
        assert np.allclose(bars[0].get_ylim(), (expected[0], expected[-1])), bars[
            0
        ].get_ylim()

    def test_a_palette_given_outright_colours_the_pooled_codes(self, code_stack):
        """``cmap=`` as a list of colours is spread over the pooled codes, one colour per code.

        Args:
            code_stack: Two frames of disjoint class codes.

        Test scenario:
            The pooled palette has to be built from the caller's ``cmap`` rather than the categorical
            default, and it has to be as long as the *pooled* code list — a palette sized to one frame's
            codes would recycle colours across the stack.
        """
        palette = ["#111111", "#222222", "#333333", "#444444"]
        codes = [
            int(code)
            for code in np.unique(np.concatenate([a.ravel() for a in CODE_FRAMES]))
        ]
        _, maps = facet(code_stack, crs=4326, scheme="categorical", cmap=palette)
        painted = _code_colors(maps[1].layers[-1][1], codes)
        assert painted == dict(zip(codes, palette)), painted


class TestTheMeasurePass:
    """The scale is measured by reading each frame once, not once per panel."""

    @staticmethod
    def _count_warps(monkeypatch) -> list:
        """Record every dataset handed to a panel's warp.

        Args:
            monkeypatch: pytest's monkeypatch fixture.

        Returns:
            The list the recorder appends to, one entry per ``_reproject`` call.
        """
        warped: list = []
        original = Map._reproject

        def counting(self, dataset):
            warped.append(dataset)
            return original(self, dataset)

        monkeypatch.setattr(Map, "_reproject", counting)
        return warped

    def test_a_multi_band_cube_is_warped_once_to_measure_it(self, monkeypatch):
        """One warp measures the whole cube, however many bands it has.

        Args:
            monkeypatch: pytest's monkeypatch fixture.

        Test scenario:
            A multi-band dataset is one panel per band, and the stack is the *same* dataset handed over
            once per band. Warping it per band made the measure pass ``n`` warps of ``n`` bands — quadratic
            — so a 32-band cube took 1.3 s to measure what one warp answers. The cube is read once here, so
            the only warps left are the one measurement plus the one each panel makes to draw.
        """
        cube = np.stack(FRAMES)
        warped = self._count_warps(monkeypatch)
        facet(_dataset(cube), crs=4326)
        expected = 1 + len(
            FRAMES
        )  # one warp to measure the cube, then one per panel to draw it
        assert len(warped) == expected, len(warped)

    def test_separate_frames_are_each_measured(self, monkeypatch):
        """Distinct frames are not collapsed: each is still read for the shared scale.

        Args:
            monkeypatch: pytest's monkeypatch fixture.

        Test scenario:
            The measure pass reads one warp per *frame*, which for a sequence of separate datasets is one
            each — the saving is only in not re-reading a frame already read. Were the frames deduplicated
            by anything coarser than identity, a stack of equal-valued frames would be measured off one of
            them and the shared scale would stop covering the others.
        """
        frames = [_dataset(arr) for arr in FRAMES]
        warped = self._count_warps(monkeypatch)
        facet(frames, crs=4326)
        expected = 2 * len(frames)  # one warp each to measure, then one each to draw
        assert len(warped) == expected, len(warped)
