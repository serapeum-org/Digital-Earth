"""``ProjectionMixin``'s builders hand back the map, so the whole static tier chains (L6).

``set_bounds`` already returned ``Self`` while the three builders beside it returned ``None``, so
``Map(...).graticule(spacing=30.0).set_global()`` raised ``AttributeError`` on a tier whose every other
builder chains. Measured on the parent commit (``842a2d95``):

```
globe  graticule  returns NoneType  is-map=False  layers=['graticule-1']
flat   graticule  returns NoneType  is-map=False  layers=['graticule-1']
flat   set_global returns NoneType  is-map=False
flat   set_domain returns NoneType  is-map=False
declared returns: graticule -> None, set_global -> None, set_domain -> None, set_bounds -> Self
```

``render``, ``show`` and ``save`` are deliberately not converted: the first two are terminal actions and
``save``'s return *is* the point (it answers the path it wrote).

Each claim is read off the returned object's **identity and type** — ``Self`` promises the composed `Map`,
not the mixin that defines the method — and the layer a converted builder drew is reached through
:meth:`~digitalearth.static.scene.Scene.artist`, which **raises** ``KeyError`` for a layer that drew
nothing, so the replacement for "assert the return value" is not vacuous.
"""

import pytest

from digitalearth.static import Map

#: A window, as ``(west, south, east, north)`` in EPSG:4326, the chained calls are measured against.
WINDOW = [-35.0, -5.0, 35.0, 65.0]


@pytest.fixture
def canvas():
    """Yield a flat lon/lat map, closed on the way out.

    Yields:
        A :class:`~digitalearth.static.map.Map` for the chaining claims.
    """
    scene = Map(crs=4326)
    yield scene
    scene.close()


class TestTheBuildersHandBackTheMap:
    """Identity and type, for each of the three converted methods."""

    def test_graticule_returns_the_same_map(self, canvas):
        """``graticule()`` is a layer builder, so it chains like every other one.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            It registers ``graticule-1`` and draws, which makes it a builder by every other test on the
            tier; it simply handed back nothing.
        """
        assert canvas.graticule(spacing=30.0) is canvas, (
            "graticule() must return the map it drew on"
        )

    def test_graticule_returns_the_composed_map_and_not_the_mixin(self, canvas):
        """``Self`` promises the composed class, which is what makes the next call in a chain legal.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            An annotation naming the mixin would type-check a chained call against ``ProjectionMixin``
            and lose every method the other five mixins contribute.
        """
        assert type(canvas.graticule(spacing=30.0)) is Map, (
            f"graticule() must return a Map; got {type(canvas.graticule(spacing=30.0)).__name__}"
        )

    def test_set_global_returns_the_same_map(self, canvas):
        """``set_global()`` frames the view, which is what ``set_bounds`` does and already chains.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            The two are the same call with and without an argument — ``set_global`` is one line over
            ``set_bounds`` — so answering differently was an inconsistency rather than a contract.
        """
        assert canvas.set_global() is canvas, (
            "set_global() must return the map it framed"
        )

    def test_set_domain_returns_the_same_map(self, canvas):
        """``set_domain(name)`` frames the view too.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            The named-region spelling of ``set_bounds``, reached through it.
        """
        assert canvas.set_domain("europe") is canvas, (
            "set_domain() must return the map it framed"
        )

    def test_set_domain_returns_the_map_even_when_there_is_no_domain(self, canvas):
        """The no-op path returns the map as well, which is the branch a conversion forgets.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            ``set_domain()`` with nothing to resolve returns early and does not frame anything. A chain
            must survive it: dropping ``self`` on that branch alone would make the method return `None`
            exactly when the map was constructed without a domain.
        """
        assert canvas.set_domain() is canvas, (
            "set_domain() with no domain must still return the map"
        )


class TestTheWholeChainRuns:
    """The calls in sequence, with their side effects still landing."""

    def test_a_graticule_then_a_global_frame_chains(self, canvas):
        """The sequence from the tier's own docs, as one expression.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            This raised ``AttributeError: 'NoneType' object has no attribute 'set_global'`` before the
            conversion, which is the whole defect.
        """
        assert canvas.graticule(spacing=30.0).set_global() is canvas, (
            "graticule().set_global() must run and answer the map"
        )

    def test_the_chain_still_frames_the_view(self, canvas):
        """Chaining is only worth having if each call lands its side effect.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            ``set_bounds`` is last in the chain, so the limits are the ones it asked for rather than the
            world ``set_global`` set one call earlier.
        """
        held = [
            float(value)
            for value in canvas.graticule(spacing=30.0)
            .set_global()
            .set_bounds(WINDOW)
            .ax.get_xlim()
        ]
        assert held == [WINDOW[0], WINDOW[2]], (
            f"the last frame should stand; got {held}"
        )

    def test_the_chained_graticule_is_still_on_the_axes(self, canvas):
        """The layer the chained call drew is reached by id, not by the call's value.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            ``Map.artist`` **raises** ``KeyError`` for a layer that drew nothing, so reading the grid
            through it is the non-vacuous replacement for asserting on a return value that is now the
            map. A flat map's graticule owns one ``LineCollection`` plus one ``Text`` per labelled line.
        """
        canvas.set_bounds(WINDOW).graticule(spacing=30.0)
        assert canvas.artist("graticule-1") is not None, (
            "the chained graticule should own an artist reachable by its id"
        )

    def test_a_domain_then_a_graticule_chains(self, canvas):
        """The other order, so neither method is only tested as the last link.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            ``set_domain`` frames through ``set_bounds`` and then hands the map on, so the grid is drawn
            on the framed view rather than on the world.
        """
        framed = canvas.set_domain("europe").graticule(spacing=30.0)
        assert framed.layer_ids == ["graticule-1"], (
            f"the chain should have drawn one grid; got {framed.layer_ids}"
        )


class TestWhatIsNotAChainableBuilder:
    """The three methods on the mixin that deliberately do not hand back the map."""

    def test_save_answers_the_path_it_wrote(self, canvas, tmp_path):
        """``save`` is the call whose return *is* the point, so it is left alone.

        Args:
            canvas: A flat lon/lat map.
            tmp_path: Destination directory.

        Test scenario:
            The same reason ``stock_img`` was not converted under L6: a caller wants the artefact, not
            the map. Converting it would take the path away with nothing to replace it.
        """
        written = canvas.graticule(spacing=30.0).save(tmp_path / "map.png")
        assert written.name == "map.png", (
            f"save() must answer the path; got {written!r}"
        )

    def test_render_is_a_terminal_action(self, canvas):
        """``render`` applies the frame and ends the sentence.

        Args:
            canvas: A flat lon/lat map.

        Test scenario:
            Pinned so a later sweep does not convert it for symmetry: nothing follows a render, and a
            terminal action that answered the map would invite a chain that means nothing.
        """
        assert canvas.graticule(spacing=30.0).render() is None, (
            "render() is a terminal action and answers nothing"
        )
