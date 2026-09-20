"""Compare a rendered 3-D scene with its committed baseline image (DE-47, #290).

The static tier compares its figures through pytest-mpl; the 3-D tier had no comparison at all, so a change that
moved every vertex kept the suite green. This is the 3-D counterpart, on PyVista's own `compare_images`.

Three modes, chosen on the pytest command line (see the options in `tests/conftest.py`):

* no option — the scene is rendered and must not be empty, but nothing is compared. A developer on another
  platform, whose OpenGL rasterises differently, is never blocked by the baselines.
* `--image3d-compare` — the render is compared with `tests/baseline3d/<name>.png`. This is what CI runs, on the Linux
  job whose software OpenGL the baselines were rendered with.
* `--image3d-generate` — the render is written as the new baseline. Review the image diff before committing it.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

#: Where the committed baseline images live.
BASELINES = Path(__file__).resolve().parents[1] / "baseline3d"

#: The render size every baseline uses, so a comparison is never between two sizes.
WINDOW_SIZE = (400, 300)

#: The largest `pyvista.compare_images` error accepted as "the same picture".
#:
#: Measured on a 24x32 terrain with a 50 m bump, rendered at `WINDOW_SIZE` from the isometric camera:
#:   * noise floor: the same scene rendered twice scores exactly 0.0, and so does each of the eight baseline scenes
#:     compared with its own saved PNG;
#:   * the weakest real changes tried: 216.5 (the two edge columns of the DEM zeroed) and 416.9 (one cell raised by
#:     1 m); then 1748.3 (one cell +5 m), 2835.8 (the colormap swapped) and 4824.7 (one cell +20 m).
#:
#: 100 leaves 2x under the weakest change. The baselines are rendered and compared on the same CI image, so the
#: cross-platform difference is not in the budget; a rasteriser change there is answered by regenerating the
#: baselines, never by raising this number.
ERROR_THRESHOLD = 100.0


def render(scene: Any) -> np.ndarray:
    """Render a scene from the fixed isometric camera every baseline is taken from.

    Args:
        scene: A `Scene3D` built at `WINDOW_SIZE` with its layers added.

    Returns:
        The rendered RGB image.
    """
    scene.plotter.camera_position = "iso"
    scene.plotter.reset_camera()
    return scene.screenshot()


def _keep(
    results: Optional[str],
    name: str,
    image: np.ndarray,
    baseline: Optional[Path] = None,
) -> None:
    """Write the render, and the baseline it was compared with, where CI uploads them from.

    Args:
        results: The `--image3d-results` directory, or `None` to keep nothing.
        name: The baseline's name.
        image: The rendered image.
        baseline: The baseline file, when one exists.
    """
    if results is None:
        return
    from PIL import Image

    folder = Path(results)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(folder / f"{name}.png")
    if baseline is not None:
        (folder / f"{name}-baseline.png").write_bytes(baseline.read_bytes())


def check_against_baseline(config: Any, image: np.ndarray, name: str) -> None:
    """Render-check, compare or regenerate one baseline, as the command line asks.

    Args:
        config: The pytest config, for the `--image3d-*` options.
        image: The rendered image.
        name: The baseline's file name, without the suffix.
    """
    import pyvista as pv
    from PIL import Image

    assert (image < 250).any(), f"{name} rendered an empty frame"
    baseline = BASELINES / f"{name}.png"
    if config.getoption("--image3d-generate", default=False):
        BASELINES.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(baseline)
        return
    if not config.getoption("--image3d-compare", default=False):
        return
    results = config.getoption("--image3d-results", default=None)
    if not baseline.exists():
        _keep(results, name, image)
        pytest.fail(
            f"no baseline {baseline.name} in tests/baseline3d; generate it with `pixi run -e viz3d "
            "test-3d-images-generate` on the CI image and review it before committing"
        )
    stored = np.asarray(Image.open(baseline).convert("RGB"))
    if stored.shape != image.shape:
        _keep(results, name, image, baseline)
        pytest.fail(
            f"{name} rendered at {image.shape}, but its baseline is {stored.shape}"
        )
    error = pv.compare_images(image, stored)
    if error > ERROR_THRESHOLD:
        _keep(results, name, image, baseline)
        pytest.fail(
            f"{name} differs from its baseline: error {error:.1f} > {ERROR_THRESHOLD}"
        )
