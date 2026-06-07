"""Structural tests for the Map mixin composition (PB-1/PB-2).

Pins the architecture: Map is GeoLayerBase(Scene) plus five capability mixins, and each public method is
contributed by the expected mixin. Behaviour is covered by the per-feature test modules; this only guards the
class structure so an accidental re-flattening or mis-wiring is caught.
"""
from digitalearth.scene import Map, Scene
from digitalearth.scene.maps.animation import AnimationMixin
from digitalearth.scene.maps.base import GeoLayerBase
from digitalearth.scene.maps.decoration import DecorationMixin
from digitalearth.scene.maps.projection import ProjectionMixin
from digitalearth.scene.maps.raster import RasterMixin
from digitalearth.scene.maps.vector import VectorMixin


class TestMapComposition:
    """Tests for Map's class composition."""

    def test_mro_is_base_plus_five_mixins(self):
        """Map's MRO is the five mixins, then GeoLayerBase, then Scene.

        Test scenario:
            The class is assembled from exactly the documented bases in the documented order.
        """
        names = [c.__name__ for c in Map.__mro__]
        assert names[:9] == [
            "Map", "RasterMixin", "VectorMixin", "DecorationMixin", "ProjectionMixin",
            "AnimationMixin", "GeoLayerBase", "Scene", "object",
        ], f"unexpected MRO: {names}"

    def test_geolayerbase_subclasses_scene(self):
        """GeoLayerBase is the Scene subclass the mixins layer onto.

        Test scenario:
            The geo base derives from Scene (so colorbar/legend/save/show are available to every mixin).
        """
        assert issubclass(GeoLayerBase, Scene), "GeoLayerBase must subclass Scene"
        assert issubclass(Map, GeoLayerBase), "Map must subclass GeoLayerBase"

    def test_public_methods_come_from_expected_mixins(self):
        """Each representative public method is contributed by the expected mixin.

        Test scenario:
            Method-resolution finds imshow on RasterMixin, scatter on VectorMixin, coastlines on
            DecorationMixin, set_domain on ProjectionMixin, animate on AnimationMixin.
        """
        owner = {
            "imshow": RasterMixin, "rgb_composite": RasterMixin,
            "scatter": VectorMixin, "choropleth": VectorMixin, "quiver": VectorMixin,
            "coastlines": DecorationMixin, "text": DecorationMixin,
            "set_domain": ProjectionMixin, "set_global": ProjectionMixin,
            "animate": AnimationMixin, "rotate": AnimationMixin,
        }
        for method, mixin in owner.items():
            assert method in mixin.__dict__, f"{method} should be defined on {mixin.__name__}"

    def test_map_constructs_and_renders(self, dataset):
        """The composed Map still constructs and draws a layer end to end.

        Test scenario:
            A smoke check that the mixins cooperate on a real render through the composed class.
        """
        m = Map(crs=dataset.epsg)
        m.imshow(dataset)
        assert len(m.layers) == 1, f"expected one layer, got {len(m.layers)}"
