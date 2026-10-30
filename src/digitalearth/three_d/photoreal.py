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

import math
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

#: Default sky colour (a cool daylight fill) and sun direction (pointing from the light toward the scene).
_DEFAULT_SKY = (0.5, 0.56, 0.72)
_DEFAULT_SUN_DIRECTION = (-0.5, 0.42, -0.76)
_DEFAULT_SOLID_RGB = (0.6, 0.6, 0.62)
#: Point-cloud sphere radius as a fraction of the scene bounding-box diagonal (when none is given).
_POINT_RADIUS_FRACTION = 0.004
#: Above this many points, a cloud is subsampled before glyphing — each point becomes a 20-triangle icosphere,
#: so the uncapped triangle count (up to the tier's 500k-point budget x 20) is far too heavy for the CPU tracer.
_MAX_GLYPH_POINTS = 50_000


@dataclass(frozen=True)
class PhotorealResult:
    """The photoreal render together with its HDR AOV passes (returned by :meth:`photoreal` when ``return_aovs``).

    The AOV passes are the linear, unclipped arrays the renderer produces before tonemapping, so they can feed
    an HDR/compositing workflow; ``image`` is the same tonemapped frame a plain :meth:`photoreal` call returns.

    Attributes:
        image: The tonemapped ``(H, W, 3)`` uint8 RGB render.
        beauty: The linear HDR ``(H, W, 3)`` beauty pass (unclipped), as written to EXR and fed to the denoiser.
        albedo: The ``(H, W, 3)`` albedo AOV.
        normal: The ``(H, W, 3)`` shading-normal AOV.
    """

    image: np.ndarray
    beauty: np.ndarray
    albedo: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class SkyOptions:
    """Illumination for :meth:`photoreal` — the sun and the sky it lights the scene with.

    Attributes:
        model: ``"constant"`` (a hard directional sun plus a flat constant sky) or ``"sunsky"`` (Mitsuba's
            physical sun disk plus sky dome, which casts soft shadows), driven by the same sun azimuth/elevation.
        color: ``(r, g, b)`` radiance of the constant sky fill (``model="constant"`` only).
        turbidity: Atmospheric haze for ``model="sunsky"`` (about 2 for a clear sky, up to 10 for a hazy one).
        sun_direction: ``(x, y, z)`` the sunlight travels (from the light toward the scene). ``None`` derives it
            from the scene's :meth:`~...decoration.DecorationMixin.sun` azimuth/elevation when one is set,
            falling back to a fixed default otherwise.
        sun_irradiance: Scalar sun brightness (``model="constant"`` only).
    """

    model: str = "constant"
    color: Any = _DEFAULT_SKY
    turbidity: float = 3.0
    sun_direction: Any = None
    sun_irradiance: float = 6.0


@dataclass(frozen=True)
class WaterOptions:
    """A reflective water plane for :meth:`photoreal`.

    Attributes:
        level: World ``z`` of the water surface; ``None`` adds no water.
        roughness: ``0`` is a mirror-smooth surface; ``> 0`` gives a rippled rough water.
    """

    level: float | None = None
    roughness: float = 0.0


@dataclass(frozen=True)
class SmokeOptions:
    """A volumetric smoke/cloud for :meth:`photoreal`.

    Attributes:
        density: A 3-D ``(nz, ny, nx)`` density grid (non-negative); ``None`` renders no smoke.
        scale: Extinction multiplier (optical thickness — higher reads as denser).
        albedo: Single-scattering albedo in ``[0, 1]`` (near 1 = bright cloud).
    """

    density: Any = None
    scale: float = 8.0
    albedo: float = 0.9


@dataclass(frozen=True)
class OutputOptions:
    """Extra outputs for :meth:`photoreal` beyond the tonemapped image.

    Attributes:
        exr: Where to also write the linear HDR beauty as OpenEXR (``.exr``); ``None`` writes none.
        return_aovs: When ``True``, :meth:`photoreal` returns a :class:`PhotorealResult` (image plus the HDR
            beauty/albedo/normal passes) instead of the bare uint8 image.
    """

    exr: Any = None
    return_aovs: bool = False


def _scene_sun_direction(decoration: Any) -> tuple[float, float, float] | None:
    """Mitsuba sun travel-direction from a scene's recorded sun (azimuth/elevation), or ``None`` if none is set.

    The tier's :meth:`~digitalearth.three_d.decoration.decoration.DecorationMixin.sun` records
    ``{azimuth, elevation, ...}`` in the scene's decoration state and places a light *along*
    ``(cos el sin az, cos el cos az, sin el)`` shining at the scene (#511), so the light **travels** the opposite
    way — which is what Mitsuba's ``directional`` emitter takes as its ``direction``.

    Args:
        decoration: The scene's decoration state dict (``self._decoration``), or ``None``.

    Returns:
        The ``(x, y, z)`` travel-direction, or ``None`` when no sun is recorded.
    """
    sun = (decoration or {}).get("sun") if isinstance(decoration, dict) else None
    if not sun:
        return None
    az = math.radians(float(sun["azimuth"]))
    el = math.radians(float(sun["elevation"]))
    return (-math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), -math.sin(el))


def _points_to_spheres(mesh: Any, radius: float, layer: str = "a point cloud") -> Any:
    """Replace a points-only layer (e.g. a point cloud) with a small sphere at each point, keeping its scalars.

    A point-cloud mesh has no polygon faces, so it would triangulate to nothing and be dropped (#510). Glyphing a
    low-resolution icosphere onto each point gives a renderable surface; the point scalars propagate, but glyph
    makes its own ``GlyphScale`` array active, so the original colour scalars are re-activated here.

    Args:
        mesh: The points-only PyVista mesh.
        radius: Sphere radius in the mesh's own (world) units.
        layer: The layer's name, named in the subsample warning so each layer warns distinctly (not de-duplicated).

    Returns:
        The glyphed PyVista mesh (spheres at every point), or ``None`` when it has no geometry.
    """
    import pyvista as pv

    name = mesh.active_scalars_name
    source = mesh
    count = int(getattr(mesh, "n_points", 0))
    if count > _MAX_GLYPH_POINTS:
        # Each point becomes a 20-triangle sphere, so cap the count for the CPU path tracer (an even stride
        # keeps the render deterministic). The PyVista raster can draw far more as flat dots; this is the
        # photoreal-specific budget the tier's point budget does not provide (#514).
        index = np.arange(0, count, int(np.ceil(count / _MAX_GLYPH_POINTS)))
        source = pv.PolyData(np.asarray(mesh.points)[index])
        if name:
            source[name] = np.asarray(mesh[name])[index]
            source.set_active_scalars(name)
        warnings.warn(
            f"photoreal(): {layer} of {count} points is subsampled to {source.n_points} for path tracing "
            "(each point becomes a sphere); decimate the cloud upstream to render more",
            stacklevel=2,
        )
    glyphed = source.glyph(
        geom=pv.Icosphere(radius=radius, nsub=0), scale=False, orient=False
    )
    if name and name in glyphed.point_data:
        glyphed.set_active_scalars(name)
    return glyphed if getattr(glyphed, "n_points", 0) else None


def _material(roughness: float | None) -> dict[str, Any]:
    """The twosided BSDF reading the per-vertex ``vertex_color``: diffuse by default, rough-PBR when asked.

    ``twosided`` so a surface lights regardless of its triangulated normal facing. With ``roughness`` set, the
    surface uses a Disney-style ``principled`` BSDF (so it reads as matte-to-glossy) instead of pure Lambertian
    diffuse (#512).

    Args:
        roughness: ``None`` for a plain diffuse surface, else the ``principled`` BSDF roughness in ``[0, 1]``.

    Returns:
        dict: the ``twosided`` BSDF spec for :func:`mitsuba.load_dict`.
    """
    material: dict[str, Any]
    if roughness is None:
        material = {
            "type": "diffuse",
            "reflectance": {"type": "mesh_attribute", "name": "vertex_color"},
        }
    else:
        material = {
            "type": "principled",
            "base_color": {"type": "mesh_attribute", "name": "vertex_color"},
            "roughness": float(roughness),
        }
    return {"type": "twosided", "material": material}


def _sky_emitters(
    sky_model: str,
    sun_direction: tuple[float, float, float],
    sun_irradiance: float,
    sky: Any,
    turbidity: float,
) -> dict[str, Any]:
    """Build the scene's light emitters for :func:`mitsuba.load_dict`, per the chosen sky model (A2).

    ``"constant"`` is the original pairing — a ``directional`` sun plus a flat ``constant`` sky fill, which
    gives hard shadows and no sky variation. ``"sunsky"`` replaces both with Mitsuba's physical ``sunsky``
    emitter: its finite sun disk casts **soft** shadows and its sky dome supplies physically-based ambient
    light. ``sunsky`` points *at* the sun, which is the negation of the travel direction the directional sun
    uses, so the same scene sun azimuth/elevation drives either model.

    Args:
        sky_model: ``"constant"`` or ``"sunsky"``.
        sun_direction: ``(x, y, z)`` the sunlight travels (from the light toward the scene).
        sun_irradiance: Directional-sun brightness (used by ``"constant"`` only).
        sky: ``(r, g, b)`` constant-sky radiance (used by ``"constant"`` only).
        turbidity: Atmospheric haze for ``sunsky`` (about 2 for a clear sky, up to 10 for a hazy one).

    Returns:
        dict: the emitter entries to merge into the scene dict.

    Raises:
        ValueError: when ``sky_model`` is neither ``"constant"`` nor ``"sunsky"``.
    """
    if sky_model == "constant":
        return {
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
    if sky_model == "sunsky":
        return {
            "environment": {
                "type": "sunsky",
                "sun_direction": [-float(v) for v in sun_direction],
                "turbidity": float(turbidity),
            }
        }
    raise ValueError(f"sky_model must be 'constant' or 'sunsky', got {sky_model!r}")


def _water_shape(
    mi: Any,
    bounds: Any,
    center: tuple[float, float, float],
    water_level: float,
    roughness: float,
) -> dict[str, Any]:
    """A horizontal water plane for :func:`mitsuba.load_dict` — a rectangle with a (rough) water dielectric (A3).

    A Mitsuba ``rectangle`` (the unit ``xy`` square) scaled to span the scene's horizontal extent and translated
    to ``water_level`` (recentered by ``center`` like every other shape), carrying a water ``dielectric`` BSDF so
    the path tracer reflects the sky and terrain off it — with Fresnel reflectance that rises toward grazing
    angles, as real water does. ``roughness > 0`` switches to ``roughdielectric`` for a rippled surface.

    Args:
        mi: The imported ``mitsuba`` module (for :class:`ScalarTransform4f`).
        bounds: The scene ``(xmin, xmax, ymin, ymax, zmin, zmax)`` bounds.
        center: The scene centre subtracted from every shape (see :meth:`_photoreal_center`).
        water_level: World ``z`` of the water surface.
        roughness: ``0`` for a mirror-smooth surface, ``> 0`` for a rough (rippled) ``roughdielectric``.

    Returns:
        dict: the ``rectangle`` water-shape spec.
    """
    half_x = max((bounds[1] - bounds[0]) / 2.0, 1e-6)
    half_y = max((bounds[3] - bounds[2]) / 2.0, 1e-6)
    z = float(water_level) - float(center[2])
    bsdf: dict[str, Any] = (
        {"type": "dielectric", "int_ior": "water", "ext_ior": "air"}
        if roughness <= 0.0
        else {
            "type": "roughdielectric",
            "int_ior": "water",
            "ext_ior": "air",
            "alpha": float(roughness),
        }
    )
    return {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f()
        .translate([0.0, 0.0, z])
        .scale([half_x, half_y, 1.0]),
        "bsdf": bsdf,
    }


def _smoke_medium(
    mi: Any,
    density: np.ndarray,
    bounds: Any,
    scale: float,
    albedo: float,
) -> dict[str, Any]:
    """A smoke/cloud volume for :func:`mitsuba.load_dict` — a cube filled with a heterogeneous medium (A4).

    The density grid becomes a Mitsuba ``gridvolume`` wrapped in a ``heterogeneous`` participating medium inside
    a ``cube`` with a ``null`` BSDF (the cube is only the medium's bounds, not a visible surface). The cube spans
    the scene's extent, recentred to the origin like every other shape, and the grid is mapped across the cube's
    local ``[-1, 1]^3`` box. With the ``volpath`` integrator this scatters light, so the smoke both self-shadows
    and casts a shadow on the terrain below. ``scale`` multiplies the extinction (optical thickness); ``albedo``
    sets how much light scatters vs is absorbed (near 1 = bright smoke/cloud).

    Args:
        mi: The imported ``mitsuba`` module.
        density: A 3-D ``(nz, ny, nx)`` density grid, non-negative.
        bounds: The scene ``(xmin, xmax, ymin, ymax, zmin, zmax)`` bounds.
        scale: Extinction multiplier (optical thickness of the medium).
        albedo: Single-scattering albedo in ``[0, 1]``.

    Returns:
        dict: the ``cube`` smoke-shape spec with its interior medium.
    """
    half = [
        max((bounds[1] - bounds[0]) / 2.0, 1e-6),
        max((bounds[3] - bounds[2]) / 2.0, 1e-6),
        max((bounds[5] - bounds[4]) / 2.0, 1e-6),
    ]
    grid = np.ascontiguousarray(np.asarray(density, dtype="float32"))
    grid_to_world = (
        mi.ScalarTransform4f().translate([-1.0, -1.0, -1.0]).scale([2.0, 2.0, 2.0])
    )
    return {
        "type": "cube",
        "to_world": mi.ScalarTransform4f().scale(half),
        "bsdf": {"type": "null"},
        "interior": {
            "type": "heterogeneous",
            "sigma_t": {
                "type": "gridvolume",
                "grid": mi.VolumeGrid(grid),
                "to_world": grid_to_world,
            },
            "scale": float(scale),
            "albedo": float(albedo),
        },
    }


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
    """Return the triangulated outer surface of a PyVista mesh, with per-vertex normals when it has faces.

    A points-only mesh (e.g. a point cloud) triangulates to no polygons, and VTK cannot compute normals for it,
    so the surface is returned without normals — the caller detects the empty-face case and glyphs it (#510).
    """
    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if np.asarray(surf.faces).reshape(-1, 4).shape[0] == 0:
        return surf
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
        except Exception:  # noqa: BLE001
            # A VTK cell->point conversion failure degrades to the solid colour, never aborts the render.
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
        except Exception:  # noqa: BLE001
            # Any LUT/VTK hiccup falls back to the solid colour below, never failing the render.
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


def _median_3x3(plane: np.ndarray) -> np.ndarray:
    """Return the 3x3 neighbourhood median of a 2-D array, with edge padding at the borders."""
    padded = np.pad(plane, 1, mode="edge")
    rows, cols = plane.shape
    neighbours = np.stack(
        [padded[i : i + rows, j : j + cols] for i in range(3) for j in range(3)],
        axis=0,
    )
    return np.asarray(np.median(neighbours, axis=0))


def _despeckle(color: np.ndarray, threshold: float) -> np.ndarray:
    """Replace pixels whose luminance deviates from their 3x3 neighbourhood median by more than ``threshold``.

    A selective median filter that removes the isolated bright/dark speckles (fireflies) a low-sample path trace
    leaves (#512). The outlier decision is made once per pixel on luminance and all three channels of a flagged
    pixel are replaced by their local median together, so a speckle bright in only one channel is removed cleanly
    rather than left as a hue-shifted residual. Step edges keep a 3-pixel majority and are preserved, but an
    isolated one-pixel-wide feature (a thin line, a lone point, a specular glint) is indistinguishable from a
    firefly to a 3x3 median and is also removed — so enable this only when that trade-off is acceptable. Operates
    on an LDR ``[0, 1]`` image.

    Args:
        color: An ``(H, W, 3)`` image in ``[0, 1]``.
        threshold: Maximum allowed luminance deviation from the local median before a pixel is a speckle.

    Returns:
        numpy.ndarray: the despeckled ``(H, W, 3)`` image.
    """
    luminance = color.mean(axis=2)
    outlier = np.abs(luminance - _median_3x3(luminance)) > float(threshold)
    out = color.copy()
    for channel in range(color.shape[2]):
        out[..., channel] = np.where(
            outlier, _median_3x3(color[..., channel]), color[..., channel]
        )
    return out


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
        max_depth: int = 8,
        despeckle: float | None = None,
        roughness: float | None = None,
        point_radius: float | None = None,
        sky: SkyOptions | None = None,
        water: WaterOptions | None = None,
        smoke: SmokeOptions | None = None,
        output: OutputOptions | None = None,
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
            max_depth: Maximum path-tracer bounce depth (higher = more indirect light, slower). Water and smoke
                need it high enough for the extra bounce.
            despeckle: Remove isolated bright/dark speckles (fireflies) a low-sample path trace leaves: a pixel
                deviating from its 3x3 neighbourhood median by more than this (in the ``[0, 1]`` output) is
                replaced by the median, leaving edges and smooth regions untouched. ``None`` (the default) is
                off; ``~0.2-0.4`` is a typical strength.
            roughness: ``None`` renders a plain Lambertian diffuse surface; a value in ``[0, 1]`` uses a
                Disney-style ``principled`` BSDF of that roughness so the surface reads matte-to-glossy.
            point_radius: Sphere radius (world units) for points-only layers such as point clouds. ``None``
                (the default) uses a small fraction of the scene bounding-box diagonal.
            sky: A :class:`SkyOptions` bundling the sun and sky — model (constant/sunsky), colour, turbidity,
                sun direction and irradiance. ``None`` (the default) is a constant sky lit by the scene-derived
                sun (A2, #511).
            water: A :class:`WaterOptions` reflective water plane (level + roughness). ``None`` adds no water (A3).
            smoke: A :class:`SmokeOptions` volumetric smoke/cloud (3-D density grid + scale + albedo), traced with
                ``volpath`` so it self-shadows and shadows the terrain. ``None`` renders no smoke (A4).
            output: An :class:`OutputOptions` for extra outputs — an EXR of the HDR beauty, and whether to return
                the AOV passes. ``None`` (the default) writes only the tonemapped image (A1).

        Returns:
            The rendered ``(H, W, 3)`` uint8 RGB array (also written to ``path`` when one was given); or, when
            ``output.return_aovs`` is set, a :class:`PhotorealResult` with that image plus the HDR AOV passes.

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

        sky = sky or SkyOptions()
        water = water or WaterOptions()
        smoke = smoke or SmokeOptions()
        output = output or OutputOptions()
        center = self._photoreal_center()
        sun_direction = sky.sun_direction
        if sun_direction is None:
            sun_direction = (
                _scene_sun_direction(getattr(self, "_decoration", None))
                or _DEFAULT_SUN_DIRECTION
            )
        if point_radius is None:
            point_radius = self._photoreal_point_radius()
        tmpdir = Path(tempfile.mkdtemp(prefix="digitalearth-photoreal-"))
        try:
            meshes = self._photoreal_meshes(
                center, tmpdir, point_radius=float(point_radius), roughness=roughness
            )
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
                    "img": {
                        "type": "volpath" if smoke.density is not None else "path",
                        "max_depth": int(max_depth),
                    },
                },
                "sensor": sensor,
            }
            scene_dict.update(
                _sky_emitters(
                    sky.model,
                    sun_direction,
                    sky.sun_irradiance,
                    sky.color,
                    sky.turbidity,
                )
            )
            scene_dict.update(meshes)
            if water.level is not None:
                scene_dict["water"] = _water_shape(
                    mi, self.plotter.bounds, center, water.level, water.roughness
                )
            if smoke.density is not None:
                density = np.asarray(smoke.density, dtype="float32")
                if density.ndim != 3:
                    raise ValueError(
                        f"smoke density must be a 3-D (nz, ny, nx) grid, got {density.ndim}-D"
                    )
                scene_dict["smoke"] = _smoke_medium(
                    mi, density, self.plotter.bounds, smoke.scale, smoke.albedo
                )

            arr = np.array(mi.render(mi.load_dict(scene_dict)), dtype="float32")
            beauty, albedo, normal = _split_aovs(arr)
            color = (
                _denoise(beauty, albedo, normal)
                if denoise
                else np.clip(beauty, 0.0, 1.0)
            )
            if despeckle is not None:
                # Remove the isolated bright/dark speckles a low-sample path trace leaves, after the denoise,
                # on the LDR image where a firefly is a lone outlier against its neighbours (#512).
                color = _despeckle(color, float(despeckle))

            bmp = mi.Bitmap(np.ascontiguousarray(color)).convert(
                mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True
            )
            out = np.array(bmp)
            if output.exr is not None:
                # The linear, unclipped HDR beauty (pre-tonemap, pre-despeckle) is what an EXR workflow wants.
                mi.Bitmap(np.ascontiguousarray(beauty)).write(str(output.exr))
            if path is not None:
                bmp.write(str(path))
            if output.return_aovs:
                return PhotorealResult(
                    image=out, beauty=beauty, albedo=albedo, normal=normal
                )
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

    def _photoreal_point_radius(self) -> float:
        """Default point-cloud sphere radius: a small fraction of the scene bounding-box diagonal.

        A degenerate scene (a single point, all-coincident points) has a zero diagonal, which would collapse
        every icosphere to its centre, so the radius falls back to ``1.0`` in that case.
        """
        b = self.plotter.bounds
        diag = math.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
        radius = float(diag) * _POINT_RADIUS_FRACTION
        return radius if radius > 0.0 else 1.0

    def _photoreal_meshes(
        self,
        center: tuple[float, float, float],
        tmpdir: Path,
        *,
        point_radius: float,
        roughness: float | None,
    ) -> dict[str, Any]:
        """Translate every registered layer's mesh into a Mitsuba PLY shape with per-vertex colours.

        Each mesh is recentered by subtracting ``center`` so large projected-CRS coordinates do not lose float
        precision in the renderer; :meth:`photoreal` subtracts the same ``center`` from the camera to match. The
        PLYs are written into ``tmpdir``, whose lifetime :meth:`photoreal` owns (it removes it after the render).
        A points-only layer (e.g. a point cloud) has no faces, so it is glyphed into a sphere at each point
        (:func:`_points_to_spheres`) rather than dropped.

        Args:
            center: The scene bounding-box centre to subtract from every vertex (see :meth:`_photoreal_center`).
            tmpdir: A directory to write the per-layer PLY files into, cleaned up by the caller.
            point_radius: Sphere radius (world units) for a points-only layer's per-point glyph.
            roughness: ``None`` for a diffuse BSDF, else the ``principled`` BSDF roughness (see :func:`_material`).

        Returns:
            dict: ``{layer_name: shape_dict}`` for :func:`mitsuba.load_dict`, each a ``ply`` shape with the
            twosided BSDF :func:`_material` builds (diffuse, or ``principled`` when ``roughness`` is set), reading
            its ``vertex_color`` attribute. Empty when the scene has no geometry.
        """
        offset = np.asarray(center, dtype="float64")
        shapes: dict[str, Any] = {}
        for index, (mesh, actor) in enumerate(self.layers):
            if mesh is None or getattr(mesh, "n_points", 0) == 0:
                continue
            surf = _triangulated_surface(mesh)
            faces = np.asarray(surf.faces).reshape(-1, 4)[:, 1:4].astype("int32")
            if len(faces) == 0:
                # A points-only layer (e.g. a point cloud) triangulates to nothing; render each point as a
                # small sphere so it is drawn rather than silently dropped (#510).
                spheres = _points_to_spheres(mesh, point_radius, layer=f"layer_{index}")
                if spheres is None:
                    continue
                surf = _triangulated_surface(spheres)
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
                "bsdf": _material(roughness),
            }
        return shapes
