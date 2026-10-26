"""Regenerate the tiny LAS fixture used by ``tests/three_d/test_point_cloud3d.py::TestLasPath``.

Writes 8 points whose real-world coordinates are exact at the 0.01 LAS scale, over a 155000/463000 offset, so a
reader applying the LAS scale/offset reproduces them and a reader using the raw integer accessors does not.
EPSG:28992 (Amersfoort / RD New) is carried in the header so the scene-CRS-adoption test has a code to assert.
Needs ``laspy`` and ``pyproj`` (present in the ``viz3d`` env via the ``pointclouds`` dependency):

    python tests/data/make_point_cloud_fixture.py tests/data/point_cloud_28992.las
"""

import sys

import laspy
import numpy as np
import pyproj


def write_fixture(out: str) -> None:
    """Write the 8-point EPSG:28992 LAS fixture to ``out``."""
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [155000.0, 463000.0, 0.0]
    header.add_crs(pyproj.CRS.from_epsg(28992))

    las = laspy.LasData(header)
    las.x = np.array(
        [
            155000.00,
            155000.50,
            155001.00,
            155001.50,
            155002.00,
            155002.50,
            155003.00,
            155003.50,
        ]
    )
    las.y = np.array(
        [
            463000.00,
            463000.25,
            463000.50,
            463000.75,
            463001.00,
            463001.25,
            463001.50,
            463001.75,
        ]
    )
    las.z = np.array([10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00, 13.50])
    las.classification = np.array([2, 2, 6, 6, 2, 5, 5, 2], dtype=np.uint8)
    las.intensity = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.uint16)
    las.return_number = np.array([1, 1, 1, 2, 1, 1, 2, 1], dtype=np.uint8)
    las.write(out)


if __name__ == "__main__":
    write_fixture(sys.argv[1])
