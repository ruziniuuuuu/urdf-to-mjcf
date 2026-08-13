"""Tests for geometry utilities."""

import math
import xml.etree.ElementTree as ET

from urdf_to_mjcf.core.geometry import (
    build_transform,
    compute_min_z,
    format_value,
    parse_vector,
    quat_from_str,
    quat_to_rot,
    rpy_to_quat,
)


def test_format_value_basic() -> None:
    assert format_value(1.0) == "1"
    assert format_value(1.23456) == "1.23456"
    assert format_value(-0.0) == "0"
    assert format_value(0.1000) == "0.1"


def test_format_value_keeps_precision_beyond_four_decimals() -> None:
    assert format_value(0.034366) == "0.034366"
    assert format_value(math.sqrt(0.5)) == "0.707106781187"


def test_format_value_snaps_trigonometric_noise_to_zero() -> None:
    assert format_value(math.cos(math.pi / 2)) == "0"
    assert format_value(-6.123233995736766e-17) == "0"


def test_format_value_never_uses_scientific_notation() -> None:
    assert format_value(5e-6) == "0.000005"
    assert format_value(-1.5e-9) == "-0.0000000015"


def test_parse_vector() -> None:
    assert parse_vector("1 2 3") == [1.0, 2.0, 3.0]
    assert parse_vector("0.5 -1.5 2.0") == [0.5, -1.5, 2.0]


def test_quat_from_str() -> None:
    assert quat_from_str("1 0 0 0") == [1.0, 0.0, 0.0, 0.0]


def test_quat_to_rot_identity() -> None:
    rot = quat_to_rot([1.0, 0.0, 0.0, 0.0])
    assert rot[0][0] == 1.0
    assert rot[1][1] == 1.0
    assert rot[2][2] == 1.0


def test_build_transform() -> None:
    t = build_transform("1 2 3", "1 0 0 0")
    assert t[0][3] == 1.0
    assert t[1][3] == 2.0
    assert t[2][3] == 3.0
    assert t[3] == [0.0, 0.0, 0.0, 1.0]


def test_compute_min_z_uses_rotated_cylinder_radius() -> None:
    body = ET.fromstring(
        """
        <body>
          <body pos="0 0 0.1">
            <geom type="cylinder" size="0.105 0.025" quat="0.70710678 0 0.70710678 0"/>
          </body>
        </body>
        """
    )

    assert abs(compute_min_z(body) - -0.005) < 1e-6


def test_rpy_to_quat_zero() -> None:
    q = rpy_to_quat("0 0 0")
    assert q == "1 0 0 0"


def test_rpy_to_quat_nonzero() -> None:
    q = rpy_to_quat(f"{math.pi} 0 0")
    parts = list(map(float, q.split()))
    assert abs(parts[0]) < 1e-6
    assert abs(parts[1] - 1.0) < 1e-6
    assert abs(parts[2]) < 1e-6
    assert abs(parts[3]) < 1e-6
