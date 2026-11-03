"""Render a terrain scene on each photoreal device backend (cpu / llvm / gpu).

``Scene3D.photoreal()`` path-traces with Mitsuba 3, and ``RenderOptions(device=...)`` picks the backend:

- ``"cpu"``  -> ``scalar_rgb``   : always available (no GPU, no display), the default.
- ``"llvm"`` -> ``llvm_ad_rgb``  : CPU-vectorised, needs drjit's LLVM runtime (the LLVM shared library).
- ``"gpu"``  -> ``cuda_ad_rgb``  : needs an NVIDIA GPU with the CUDA runtime.

This script tries all three and writes ``<device>.png`` for every backend that is present, printing a clear
"skipped" line for the ones that are not — so it runs anywhere, mirroring how the tests gate on the runtime.

Run it in the photoreal env::

    pixi run -e photoreal python examples/3d-maps/photoreal_device_render.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d.photoreal import RenderOptions


def _terrain_scene() -> Scene3D:
    """Build a small off-screen coloured-terrain scene for a quick path-traced render."""
    rng = np.random.default_rng(0)
    z = rng.random((40, 40)) * 6.0
    scene = Scene3D(off_screen=True)
    scene.terrain(get_source(z, x=np.arange(40.0), y=np.arange(40.0)), cmap="viridis")
    scene.view_angles(azimuth=225.0, elevation=30.0)
    return scene


def main() -> None:
    """Render the scene on cpu, llvm and gpu, skipping any backend whose runtime is unavailable."""
    out_dir = Path(__file__).with_suffix("")
    out_dir.mkdir(exist_ok=True)
    for device in ("cpu", "llvm", "gpu"):
        scene = _terrain_scene()
        try:
            img = scene.photoreal(
                path=str(out_dir / f"{device}.png"),
                spp=64,
                width=480,
                height=360,
                denoise=True,
                render=RenderOptions(device=device),
            )
            print(
                f"{device:>4} -> rendered {img.shape[1]}x{img.shape[0]} to {out_dir / f'{device}.png'}"
            )
        except ImportError as exc:
            print(f"{device:>4} -> skipped (photoreal extra missing): {exc}")
        except RuntimeError as exc:
            print(f"{device:>4} -> skipped (backend unavailable): {exc}")
        finally:
            scene.close()


if __name__ == "__main__":
    main()
