from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def polygon_to_ring_xy(poly: Polygon) -> list[tuple[float, float]]:
    coords = list(poly.exterior.coords)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return [(float(x), float(y)) for x, y in coords]


def ring_xyz(ring_xy: list[tuple[float, float]], z: float) -> list[tuple[float, float, float]]:
    return [(x, y, float(z)) for x, y in ring_xy]


def wall_faces(ring_xy: list[tuple[float, float]], z0: float, z1: float) -> list[list[tuple[float, float, float]]]:
    faces: list[list[tuple[float, float, float]]] = []
    for i in range(len(ring_xy) - 1):
        x1, y1 = ring_xy[i]
        x2, y2 = ring_xy[i + 1]
        faces.append([(x1, y1, z0), (x2, y2, z0), (x2, y2, z1), (x1, y1, z1), (x1, y1, z0)])
    return faces


def _project_footprint(footprint: Polygon, long_axis: np.ndarray, short_axis: np.ndarray):
    """Project footprint coords onto (long, short) axes; return (coords, cx, cy, l_ext, s_ext)."""
    coords = np.array(footprint.exterior.coords[:-1])
    cx, cy = coords.mean(axis=0)
    rel = coords - np.array([cx, cy])
    ls = rel @ np.column_stack([long_axis, short_axis])  # (N, 2): long, short
    return coords, cx, cy, ls


def _gable_faces(
    footprint: Polygon,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    eave_z: float,
    ridge_z: float,
) -> list[list[tuple[float, float, float]]]:
    """
    Gable roof: two sloped rectangular faces + two triangular gable ends.
    Ridge runs along the long axis through the centroid.
    """
    coords, cx, cy, ls = _project_footprint(footprint, long_axis, short_axis)

    l_min, l_max = ls[:, 0].min(), ls[:, 0].max()
    s_min, s_max = ls[:, 1].min(), ls[:, 1].max()
    s_mid = (s_min + s_max) / 2.0

    def world(l: float, s: float) -> tuple[float, float]:
        pt = np.array([cx, cy]) + l * long_axis + s * short_axis
        return (float(pt[0]), float(pt[1]))

    # Ridge endpoints
    r0 = world(l_min, s_mid)
    r1 = world(l_max, s_mid)

    # Eave corners (four corners of the footprint bounding box in rotated frame)
    a = world(l_min, s_min)
    b = world(l_max, s_min)
    c = world(l_max, s_max)
    d = world(l_min, s_max)

    faces: list[list[tuple[float, float, float]]] = []

    # Slope face 1: a → b → r1 → r0
    faces.append([
        (*a, eave_z), (*b, eave_z), (*r1, ridge_z), (*r0, ridge_z), (*a, eave_z),
    ])
    # Slope face 2: d → r0 → r1 → c  (opposite side)
    faces.append([
        (*d, eave_z), (*r0, ridge_z), (*r1, ridge_z), (*c, eave_z), (*d, eave_z),
    ])
    # Gable triangle left: a → r0 → d
    faces.append([
        (*a, eave_z), (*r0, ridge_z), (*d, eave_z), (*a, eave_z),
    ])
    # Gable triangle right: b → c → r1
    faces.append([
        (*b, eave_z), (*c, eave_z), (*r1, ridge_z), (*b, eave_z),
    ])

    return faces


def _hip_faces(
    footprint: Polygon,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    eave_z: float,
    ridge_z: float,
) -> list[list[tuple[float, float, float]]]:
    """
    Hip roof: two trapezoidal long faces + two triangular hip ends.
    Ridge is shortened by the hip inset (= half the short span).
    """
    coords, cx, cy, ls = _project_footprint(footprint, long_axis, short_axis)

    l_min, l_max = ls[:, 0].min(), ls[:, 0].max()
    s_min, s_max = ls[:, 1].min(), ls[:, 1].max()
    s_mid = (s_min + s_max) / 2.0
    hip_inset = (s_max - s_min) / 2.0  # inset from each end

    def world(l: float, s: float) -> tuple[float, float]:
        pt = np.array([cx, cy]) + l * long_axis + s * short_axis
        return (float(pt[0]), float(pt[1]))

    rl0 = world(l_min + hip_inset, s_mid)
    rl1 = world(l_max - hip_inset, s_mid)

    a = world(l_min, s_min)
    b = world(l_max, s_min)
    c = world(l_max, s_max)
    d = world(l_min, s_max)

    faces: list[list[tuple[float, float, float]]] = []

    # Long slope 1: a → b → rl1 → rl0
    faces.append([
        (*a, eave_z), (*b, eave_z), (*rl1, ridge_z), (*rl0, ridge_z), (*a, eave_z),
    ])
    # Long slope 2: d → rl0 → rl1 → c
    faces.append([
        (*d, eave_z), (*rl0, ridge_z), (*rl1, ridge_z), (*c, eave_z), (*d, eave_z),
    ])
    # Hip triangle left: a → rl0 → d
    faces.append([
        (*a, eave_z), (*rl0, ridge_z), (*d, eave_z), (*a, eave_z),
    ])
    # Hip triangle right: b → c → rl1
    faces.append([
        (*b, eave_z), (*c, eave_z), (*rl1, ridge_z), (*b, eave_z),
    ])

    return faces


def parametric_roof_faces(
    footprint: Polygon,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    roof_kind: str,
    eave_z: float,
    ridge_z: float,
) -> list[list[tuple[float, float, float]]]:
    """Return clean planar roof face polygons for gable or hip roofs."""
    if roof_kind == "gable":
        return _gable_faces(footprint, long_axis, short_axis, eave_z, ridge_z)
    if roof_kind == "hip":
        return _hip_faces(footprint, long_axis, short_axis, eave_z, ridge_z)
    return []
