"""PhotorealMixin — render the 3-D scene as a photorealistic path-traced still (Mitsuba 3 + Intel OIDN).

A **second renderer** beside the PyVista/VTK OpenGL rasterizer the rest of ``three_d`` draws with. It reads the
scene's geometry, its per-vertex colours, the camera and a sun, and renders a global-illumination still with
Mitsuba 3's path tracer — on the CPU, headless — optionally denoised with Intel Open Image Denoise. This is the
``#496`` spike's **option B**: path tracing through an installable, CI-viable engine, since an OSPRay-enabled VTK
is not distributed on any channel (so the ``RenderingRayTracing`` module PyVista would need never loads).

Mitsuba and OIDN are imported lazily through :func:`_require_photoreal` and live behind the ``[photoreal]`` extra,
so importing the tier — or using any other 3-D verb — never requires them. Mitsuba is a renderer, not a GIS /
point-cloud / matplotlib engine, so this module touches none of the pyramids / cleopatra / pointclouds
boundaries: it only translates meshes the tier already built into a Mitsuba scene.
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

#: Default sky colour (a cool daylight fill) and sun direction (pointing from the light toward the scene).
_DEFAULT_SKY = (0.5, 0.56, 0.72)
_DEFAULT_SUN_DIRECTION = (-0.5, 0.42, -0.76)
_DEFAULT_SOLID_RGB = (0.6, 0.6, 0.62)


def _require_photoreal():
    """Import and return Mitsuba, or raise a clear, actionable error when the photoreal extra is missing.

    Returns:
        module: the imported ``mitsuba`` package (with a variant not yet set).

    Raises:
        ImportError: when ``mitsuba`` or ``oidn`` is not installed, naming the install command — the same shape
            :func:`~digitalearth.three_d.builders.globe._require_geovista` uses.
    """
    try:
        import mitsuba
        import oidn  # noqa: F401 - imported only to confirm the denoiser is installed; used in _denoise
    except ImportError as exc:
        raise ImportError(
            "Scene3D.photoreal() needs the optional photorealistic-rendering dependencies 'mitsuba' and 'oidn'. "
            "Install them with:  pip install 'digitalearth[photoreal]'"
        ) from exc
    return mitsuba


def _triangulated_surface(mesh: Any) -> Any:
    """Return the triangulated outer surface of a PyVista mesh, with per-vertex normals computed."""
    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    return surf.compute_normals(
        point_normals=True, cell_normals=False, auto_orient_normals=True
    )


def _point_scalars(surf: Any) -> np.ndarray | None:
    """Return the surface's 1-D per-vertex colour scalars, interpolating cell scalars to points if needed.

    Args:
        surf: A triangulated PyVista surface.

    Returns:
        numpy.ndarray | None: an ``(n_points,)`` scalar array to colour by, or ``None`` when the surface carries
        no usable scalars (neither point nor cell).
    """
    scalars = surf.active_scalars
    if scalars is not None and np.ndim(scalars) == 1 and len(scalars) == surf.n_points:
        return np.asarray(scalars)
    # A layer coloured by cell data (len == n_cells) would otherwise fall through to the solid colour; averaging
    # it to the points (same geometry, same value range, so the actor's LUT still applies) keeps its colouring.
    name = surf.active_scalars_name
    if name and name in surf.cell_data:
        try:
            converted = surf.cell_data_to_point_data().point_data.get(name)
        except Exception:  # noqa: BLE001 - a VTK conversion failure falls back to solid, never aborts the render
            return None
        if (
            converted is not None
            and np.ndim(converted) == 1
            and len(converted) == surf.n_points
        ):
            return np.asarray(converted)
    return None


def _vertex_rgb(surf: Any, actor: Any, layer: str = "") -> np.ndarray:
    """Per-vertex float RGB in ``[0, 1]`` for a surface, mapped through the actor's LUT (solid colour otherwise).

    Point scalars map directly; cell scalars are interpolated to the points first (see :func:`_point_scalars`).
    A surface with no mappable scalars falls back to the actor's solid colour and **warns** (naming ``layer`` so
    each fallback layer warns distinctly), so the fallback is observable rather than a silently grey layer.

    Args:
        surf: A triangulated PyVista surface (``n_points`` vertices).
        actor: The PyVista actor the tier drew it with, whose mapper carries the colour lookup table.
        layer: The layer's name, named in the fallback warning so per-layer warnings are not de-duplicated.

    Returns:
        numpy.ndarray: an ``(n_points, 3)`` float32 array of RGB in ``[0, 1]``.
    """
    scalars = _point_scalars(surf)
    if scalars is not None:
        try:
            from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

            lut = actor.mapper.lookup_table
            varr = numpy_to_vtk(
                np.ascontiguousarray(np.asarray(scalars, dtype="float64"))
            )
            mapped = lut.MapScalars(varr, 0, -1)
            rgb = vtk_to_numpy(mapped)[:, :3].astype("float32") / 255.0
            return np.ascontiguousarray(rgb)
        except Exception:  # noqa: BLE001 - any LUT/VTK hiccup falls back to a solid colour, never fails the render
            pass
    # Name the layer so each fallback warns distinctly: identical messages are de-duplicated under Python's
    # default warning filter, which would otherwise hide every fallback layer after the first.
    where = f" ({layer})" if layer else ""
    warnings.warn(
        f"photoreal(): a layer{where} has no point-mappable scalars; rendering it in a solid colour",
        stacklevel=2,
    )
    try:
        solid = np.asarray(actor.prop.color, dtype="float32")[:3]
    except Exception:  # noqa: BLE001
        solid = np.asarray(_DEFAULT_SOLID_RGB, dtype="float32")
    return np.ascontiguousarray(np.tile(solid, (surf.n_points, 1)))


def _write_colored_ply(
    path: str,
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    rgb: np.ndarray,
) -> None:
    """Write a binary PLY carrying positions, normals and float ``red/green/blue`` (read as ``vertex_color``).

    Args:
        path: Destination ``.ply`` path.
        verts: ``(N, 3)`` float vertex positions.
        faces: ``(F, 3)`` int triangle indices.
        normals: ``(N, 3)`` float vertex normals.
        rgb: ``(N, 3)`` float vertex colours in ``[0, 1]`` (Mitsuba reads these as the ``vertex_color`` attribute).
    """
    n, f = len(verts), len(faces)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float red\nproperty float green\nproperty float blue\n"
        f"element face {f}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    vbuf = np.concatenate([verts, normals, rgb], axis=1).astype("<f4")
    face_dtype = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    fbuf = np.empty(f, dtype=face_dtype)
    fbuf["n"] = 3
    fbuf["a"], fbuf["b"], fbuf["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vbuf.tobytes())
        fh.write(fbuf.tobytes())


def _denoise(beauty: np.ndarray, albedo: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Denoise a path-traced frame with Intel OIDN, guided by albedo and shading-normal AOVs.

    Args:
        beauty: ``(H, W, 3)`` linear HDR colour.
        albedo: ``(H, W, 3)`` surface albedo AOV.
        normal: ``(H, W, 3)`` shading-normal AOV.

    Returns:
        numpy.ndarray: the ``(H, W, 3)`` denoised colour, clipped to ``[0, 1]``.
    """
    import oidn

    h, w = beauty.shape[:2]
    color = np.ascontiguousarray(np.clip(beauty, 0.0, 1.0).astype("float32"))
    alb = np.ascontiguousarray(np.clip(albedo, 0.0, 1.0).astype("float32"))
    nrm = np.ascontiguousarray(normal.astype("float32"))
    out = np.zeros_like(color)
    dev = oidn.NewDevice(oidn.DEVICE_TYPE_CPU)
    oidn.CommitDevice(dev)
    flt = oidn.NewFilter(dev, "RT")
    oidn.SetSharedFilterImage(flt, "color", color, oidn.FORMAT_FLOAT3, w, h)
    oidn.SetSharedFilterImage(flt, "albedo", alb, oidn.FORMAT_FLOAT3, w, h)
    oidn.SetSharedFilterImage(flt, "normal", nrm, oidn.FORMAT_FLOAT3, w, h)
    oidn.SetSharedFilterImage(flt, "output", out, oidn.FORMAT_FLOAT3, w, h)
    oidn.CommitFilter(flt)
    oidn.ExecuteFilter(flt)
    oidn.ReleaseFilter(flt)
    oidn.ReleaseDevice(dev)
    return np.clip(out, 0.0, 1.0)


def _split_aovs(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an `aov`-integrator render tensor into ``(beauty, albedo, normal)``, failing loud on a layout change.

    The scene's `aov` integrator yields 9 channels — the nested path integrator's RGB beauty (its alpha dropped
    at develop), then the ``albedo`` and shading-``normal`` AOVs. If a future Mitsuba kept the alpha (10 channels)
    the fixed slices would silently misalign, so this raises instead.

    Args:
        arr: The ``(H, W, C)`` render tensor from :func:`mitsuba.render`.

    Returns:
        tuple: the ``(H, W, 3)`` beauty, albedo and normal slices.

    Raises:
        RuntimeError: when ``arr`` does not have exactly 9 channels.
    """
    if arr.shape[-1] != 9:
        raise RuntimeError(
            f"photoreal() expected 9 render channels (beauty + albedo + normals), got {arr.shape[-1]}; "
            "the Mitsuba AOV channel layout changed"
        )
    return arr[..., 0:3], arr[..., 3:6], arr[..., 6:9]


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.core.base import Scene3DBase as _MixinBase
else:
    _MixinBase = object


class PhotorealMixin(_MixinBase):
    """Adds :meth:`photoreal` — render the scene as a path-traced still — to a :class:`Scene3D`.

    A capability mixin of :class:`~digitalearth.three_d.scene3d.Scene3D`, only ever composed into that scene. It
    reaches the registered ``layers``, the wrapped ``pyvista.Plotter`` and its camera through ``self``.
    """

    def photoreal(
        self,
        path: Any = None,
        *,
        spp: int = 64,
        width: int = 1280,
        height: int = 800,
        denoise: bool = True,
        sun_direction: Any = _DEFAULT_SUN_DIRECTION,
        sun_irradiance: float = 6.0,
        sky: Any = _DEFAULT_SKY,
        max_depth: int = 8,
    ) -> Any:
        """Render the scene as a photorealistic path-traced still with Mitsuba 3, optionally denoised.

        Every registered layer's mesh is translated into a Mitsuba triangle mesh carrying its per-vertex colours
        (mapped through the layer's lookup table), lit by a single directional sun plus a constant sky and seen
        through the scene's current camera. The frame is path-traced on the CPU (no GPU, no display); with
        ``denoise=True`` a low sample count is cleaned by Intel OIDN using albedo and normal AOVs.

        Args:
            path: Where to write the image (``.png``). ``None`` renders without writing and only returns the array.
            spp: Samples per pixel. With ``denoise=True``, 32-64 is usually enough; without it, use more.
            width: Image width in pixels.
            height: Image height in pixels.
            denoise: Denoise the frame with Intel OIDN (albedo + normal guided). Needs the ``oidn`` dependency.
            sun_direction: ``(x, y, z)`` direction the sunlight travels (from the light toward the scene).
            sun_irradiance: Scalar brightness of the sun.
            sky: ``(r, g, b)`` radiance of the constant sky fill.
            max_depth: Maximum path-tracer bounce depth (higher = more indirect light, slower).

        Returns:
            The rendered ``(H, W, 3)`` uint8 RGB array; also written to ``path`` when one was given.

        Raises:
            ImportError: when the ``[photoreal]`` extra (``mitsuba`` + ``oidn``) is not installed.
            ValueError: when the scene has no renderable geometry.

        Examples:
            - Render a terrain scene to a photorealistic still (needs the ``photoreal`` extra):
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> from digitalearth.base.sources import get_source  # doctest: +SKIP
                >>> from digitalearth.three_d import Scene3D  # doctest: +SKIP
                >>> scene = Scene3D(off_screen=True)  # doctest: +SKIP
                >>> _ = scene.terrain(get_source(Dataset.read_file("dem.tif")))  # doctest: +SKIP
                >>> img = scene.photoreal("relief.png", spp=48)  # doctest: +SKIP
                >>> img.shape[2]  # doctest: +SKIP
                3

                ```
        """
        mi = _require_photoreal()
        mi.set_variant("scalar_rgb")

        center = self._photoreal_center()
        tmpdir = Path(tempfile.mkdtemp(prefix="digitalearth-photoreal-"))
        try:
            meshes = self._photoreal_meshes(center, tmpdir)
            if not meshes:
                raise ValueError(
                    "photoreal() has no renderable geometry; add a layer (e.g. terrain) before rendering"
                )

            cam = self.plotter.camera
            sensor = {
                "type": "perspective",
                "fov": float(cam.view_angle),
                "fov_axis": "y",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[float(p) - float(c) for p, c in zip(cam.position, center)],
                    target=[
                        float(p) - float(c) for p, c in zip(cam.focal_point, center)
                    ],
                    up=[float(v) for v in cam.up],
                ),
                "sampler": {"type": "independent", "sample_count": int(spp)},
                "film": {"type": "hdrfilm", "width": int(width), "height": int(height)},
            }
            scene_dict: dict[str, Any] = {
                "type": "scene",
                "integrator": {
                    "type": "aov",
                    "aovs": "albedo:albedo,normals:sh_normal",
                    "img": {"type": "path", "max_depth": int(max_depth)},
                },
                "sensor": sensor,
                "sun": {
                    "type": "directional",
                    "direction": [float(v) for v in sun_direction],
                    "irradiance": {"type": "rgb", "value": float(sun_irradiance)},
                },
                "sky": {
                    "type": "constant",
                    "radiance": {"type": "rgb", "value": [float(v) for v in sky]},
                },
            }
            scene_dict.update(meshes)

            arr = np.array(mi.render(mi.load_dict(scene_dict)), dtype="float32")
            beauty, albedo, normal = _split_aovs(arr)
            color = (
                _denoise(beauty, albedo, normal)
                if denoise
                else np.clip(beauty, 0.0, 1.0)
            )

            bmp = mi.Bitmap(np.ascontiguousarray(color)).convert(
                mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True
            )
            out = np.array(bmp)
            if path is not None:
                bmp.write(str(path))
            return out
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _photoreal_center(self) -> tuple[float, float, float]:
        """Return the scene bounding-box centre ``(x, y, z)``, used to recenter geometry and camera."""
        b = self.plotter.bounds
        return (
            float(b[0] + b[1]) / 2.0,
            float(b[2] + b[3]) / 2.0,
            float(b[4] + b[5]) / 2.0,
        )

    def _photoreal_meshes(
        self, center: tuple[float, float, float], tmpdir: Path
    ) -> dict[str, Any]:
        """Translate every registered layer's mesh into a Mitsuba PLY shape with per-vertex colours.

        Each mesh is recentered by subtracting ``center`` so large projected-CRS coordinates do not lose float
        precision in the renderer; :meth:`photoreal` subtracts the same ``center`` from the camera to match. The
        PLYs are written into ``tmpdir``, whose lifetime :meth:`photoreal` owns (it removes it after the render).

        Args:
            center: The scene bounding-box centre to subtract from every vertex (see :meth:`_photoreal_center`).
            tmpdir: A directory to write the per-layer PLY files into, cleaned up by the caller.

        Returns:
            dict: ``{layer_name: shape_dict}`` for :func:`mitsuba.load_dict`, each a ``ply`` shape with a diffuse
            BSDF reading its ``vertex_color`` attribute. Empty when the scene has no geometry.
        """
        offset = np.asarray(center, dtype="float64")
        shapes: dict[str, Any] = {}
        for index, (mesh, actor) in enumerate(self.layers):
            if mesh is None or getattr(mesh, "n_points", 0) == 0:
                continue
            surf = _triangulated_surface(mesh)
            faces = np.asarray(surf.faces).reshape(-1, 4)[:, 1:4].astype("int32")
            if len(faces) == 0:
                continue
            verts = (np.asarray(surf.points, dtype="float64") - offset).astype(
                "float32"
            )
            normals = np.asarray(surf.point_data["Normals"], dtype="float32")
            rgb = _vertex_rgb(surf, actor, layer=f"layer_{index}")
            ply = str(tmpdir / f"layer_{index}.ply")
            _write_colored_ply(ply, verts, faces, normals, rgb)
            shapes[f"layer_{index}"] = {
                "type": "ply",
                "filename": ply,
                # twosided so the surface lights regardless of which way the triangulated normals ended up
                # facing — a one-sided diffuse renders the camera-facing back of a terrain surface black.
                "bsdf": {
                    "type": "twosided",
                    "material": {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "mesh_attribute",
                            "name": "vertex_color",
                        },
                    },
                },
            }
        return shapes
