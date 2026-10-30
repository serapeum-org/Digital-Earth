"""`scene_kw` on the interactive (HoloViz) tier — a scene setting the render never sees.

This tier is the one where a name-based routing rule is provably impossible: it sizes a Datashader
canvas from ``width``/``height`` taken out of a builder's ``**opts``, declaring neither. So the
destination has to come from the caller, and this pins that it arrives.
"""

import pytest

pytest.importorskip("holoviews")


class TestInteractiveSceneKw:
    """`quickmap(backend="interactive", scene_kw=...)` reaches `InteractiveMap`."""

    def test_a_scene_setting_reaches_the_constructor(self, dataset, mocker):
        """`scene_kw={"width": 1100}` is applied when the scene is built.

        Args:
            dataset: The shared raster fixture.
            mocker: pytest-mock fixture, used to stub the builder.

        Test scenario:
            `width` is read only at construction, and `quickmap` builds the scene itself, so without
            `scene_kw` it was out of reach of a one-call plot entirely.
        """
        from digitalearth.api import quickmap
        from digitalearth.interactive import InteractiveMap

        mocker.patch.object(InteractiveMap, "field")

        scene = quickmap(
            dataset, backend="interactive", band=1, scene_kw={"width": 1100}
        )

        assert scene.width == 1100, (
            f"scene_kw's width must reach the constructor, got {scene.width}"
        )
