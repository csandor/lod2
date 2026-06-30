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
    """Project footprint coords onto (long, short) axes."""
    coords = np.array(footprint.exterior.coords[:-1])
    cx, cy = coords.mean(axis=0)
    rel = coords - np.array([cx, cy])
    ls = rel @ np.column_stack([long_axis, short_axis])  # (N, 2): long, short
    return coords, cx, cy, ls


# ---------------------------------------------------------------------------
# MBR mode — fast rectangular approximation
# ---------------------------------------------------------------------------

def _gable_faces_mbr(
    footprint: Polygon,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    eave_z: float,
    ridge_z: float,
) -> list[list[tuple[float, float, float]]]:
    coords, cx, cy, ls = _project_footprint(footprint, long_axis, short_axis)
    l_min, l_max = ls[:, 0].min(), ls[:, 0].max()
    s_min, s_max = ls[:, 1].min(), ls[:, 1].max()
    s_mid = (s_min + s_max) / 2.0

    def w(l: float, s: float) -> tuple[float, float]:
        pt = np.array([cx, cy]) + l * long_axis + s * short_axis
        return (float(pt[0]), float(pt[1]))

    r0, r1 = w(l_min, s_mid), w(l_max, s_mid)
    a, b = w(l_min, s_min), w(l_max, s_min)
    c, d = w(l_max, s_max), w(l_min, s_max)

    return [
        [(*a, eave_z), (*b, eave_z), (*r1, ridge_z), (*r0, ridge_z), (*a, eave_z)],
        [(*d, eave_z), (*r0, ridge_z), (*r1, ridge_z), (*c, eave_z), (*d, eave_z)],
        [(*a, eave_z), (*r0, ridge_z), (*d, eave_z), (*a, eave_z)],
        [(*b, eave_z), (*c, eave_z), (*r1, ridge_z), (*b, eave_z)],
    ]


def _hip_faces_mbr(
    footprint: Polygon,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    eave_z: float,
    ridge_z: float,
) -> list[list[tuple[float, float, float]]]:
    coords, cx, cy, ls = _project_footprint(footprint, long_axis, short_axis)
    l_min, l_max = ls[:, 0].min(), ls[:, 0].max()
    s_min, s_max = ls[:, 1].min(), ls[:, 1].max()
    s_mid = (s_min + s_max) / 2.0
    hip_inset = (s_max - s_min) / 2.0

    def w(l: float, s: float) -> tuple[float, float]:
        pt = np.array([cx, cy]) + l * long_axis + s * short_axis
        return (float(pt[0]), float(pt[1]))

    rl0, rl1 = w(l_min + hip_inset, s_mid), w(l_max - hip_inset, s_mid)
    a, b = w(l_min, s_min), w(l_max, s_min)
    c, d = w(l_max, s_max), w(l_min, s_max)

    return [
        [(*a, eave_z), (*b, eave_z), (*rl1, ridge_z), (*rl0, ridge_z), (*a, eave_z)],
        [(*d, eave_z), (*rl0, ridge_z), (*rl1, ridge_z), (*c, eave_z), (*d, eave_z)],
        [(*a, eave_z), (*rl0, ridge_z), (*d, eave_z), (*a, eave_z)],
        [(*b, eave_z), (*c, eave_z), (*rl1, ridge_z), (*b, eave_z)],
    ]


# ---------------------------------------------------------------------------
# Footprint mode — eave follows actual polygon outline
# ---------------------------------------------------------------------------

def _ridge_point(cx: float, cy: float, l: float, s_mid: float,
                 long_axis: np.ndarray, short_axis: np.ndarray) -> tuple[float, float]:
    pt = np.array([cx, cy]) + l * long_axis + s_mid * short_axis
    return (float(pt[0]), float(pt[1]))


def _gable_faces_footprint(
    footprint: Polygon,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    eave_z: float,
    ridge_z: float,
) -> list[list[tuple[float, float, float]]]:
    """
    Gable roof following the actual footprint outline.

    Strategy: split footprint vertices into two sides by their sign along the
    short axis. Each side forms a slope face (eave vertices at eave_z, all
    rising to the ridge line at ridge_z). The two gable ends are triangles/
    polygons at the long-axis extremes.
    """
    coords, cx, cy, ls = _project_footprint(footprint, long_axis, short_axis)
    l_vals, s_vals = ls[:, 0], ls[:, 1]
    s_mid = (s_vals.min() + s_vals.max()) / 2.0
    l_min, l_max = l_vals.min(), l_vals.max()

    r0 = _ridge_point(cx, cy, l_min, s_mid, long_axis, short_axis)
    r1 = _ridge_point(cx, cy, l_max, s_mid, long_axis, short_axis)

    # Each vertex projects to a ridge point at its long-axis position
    def ridge_at(l: float) -> tuple[float, float]:
        return _ridge_point(cx, cy, l, s_mid, long_axis, short_axis)

    # Build one triangular face per footprint edge: eave edge + two ridge points
    faces: list[list[tuple[float, float, float]]] = []
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        ex1, ey1 = float(coords[i, 0]), float(coords[i, 1])
        ex2, ey2 = float(coords[j, 0]), float(coords[j, 1])
        rx1, ry1 = ridge_at(l_vals[i])
        rx2, ry2 = ridge_at(l_vals[j])

        # Degenerate if the ridge points coincide with each other and the eave
        if abs(l_vals[i] - l_vals[j]) < 1e-6 and abs(s_vals[i] - s_mid) < 1e-6:
            continue

        face = [
            (ex1, ey1, eave_z),
            (ex2, ey2, eave_z),
            (rx2, ry2, ridge_z),
            (rx1, ry1, ridge_z),
            (ex1, ey1, eave_z),
        ]
        faces.append(face)

    return faces


def _hip_faces_footprint(
    footprint: Polygon,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    eave_z: float,
    ridge_z: float,
) -> list[list[tuple[float, float, float]]]:
    """
    Hip roof following the actual footprint outline.

    The ridge is shortened by hip_inset = half the short span.
    Vertices beyond the ridge endpoints get connected to the nearest ridge end.
    """
    coords, cx, cy, ls = _project_footprint(footprint, long_axis, short_axis)
    l_vals, s_vals = ls[:, 0], ls[:, 1]
    s_mid = (s_vals.min() + s_vals.max()) / 2.0
    l_min, l_max = l_vals.min(), l_vals.max()
    hip_inset = (s_vals.max() - s_vals.min()) / 2.0
    rl_min = l_min + hip_inset
    rl_max = l_max - hip_inset

    # If the footprint is too narrow for a ridge, fall back to a point (pyramid)
    if rl_min >= rl_max:
        rl_min = rl_max = (l_min + l_max) / 2.0

    rl0 = _ridge_point(cx, cy, rl_min, s_mid, long_axis, short_axis)
    rl1 = _ridge_point(cx, cy, rl_max, s_mid, long_axis, short_axis)

    def ridge_at(l: float) -> tuple[float, float]:
        # Clamp to the ridge segment
        l_clamped = max(rl_min, min(rl_max, l))
        return _ridge_point(cx, cy, l_clamped, s_mid, long_axis, short_axis)

    faces: list[list[tuple[float, float, float]]] = []
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        ex1, ey1 = float(coords[i, 0]), float(coords[i, 1])
        ex2, ey2 = float(coords[j, 0]), float(coords[j, 1])
        rx1, ry1 = ridge_at(l_vals[i])
        rx2, ry2 = ridge_at(l_vals[j])

        # Collapse degenerate faces where ridge points equal eave points
        unique = [(ex1, ey1, eave_z), (ex2, ey2, eave_z),
                  (rx2, ry2, ridge_z), (rx1, ry1, ridge_z)]
        seen: list[tuple[float, float, float]] = []
        for pt in unique:
            if not seen or (abs(pt[0] - seen[-1][0]) > 1e-6 or abs(pt[1] - seen[-1][1]) > 1e-6):
                seen.append(pt)
        if len(seen) < 3:
            continue
        seen.append(seen[0])
        faces.append(seen)

    return faces


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parametric_roof_faces(
    footprint: Polygon,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    roof_kind: str,
    eave_z: float,
    ridge_z: float,
    roof_shape: str = "mbr",
) -> list[list[tuple[float, float, float]]]:
    """Return clean planar roof face polygons for gable or hip roofs."""
    if roof_kind == "gable":
        if roof_shape == "footprint":
            return _gable_faces_footprint(footprint, long_axis, short_axis, eave_z, ridge_z)
        return _gable_faces_mbr(footprint, long_axis, short_axis, eave_z, ridge_z)
    if roof_kind == "hip":
        if roof_shape == "footprint":
            return _hip_faces_footprint(footprint, long_axis, short_axis, eave_z, ridge_z)
        return _hip_faces_mbr(footprint, long_axis, short_axis, eave_z, ridge_z)
    return []
