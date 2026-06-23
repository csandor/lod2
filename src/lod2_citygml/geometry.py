from __future__ import annotations

import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import xy as transform_xy
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, mapping


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
        faces.append(
            [
                (x1, y1, z0),
                (x2, y2, z0),
                (x2, y2, z1),
                (x1, y1, z1),
                (x1, y1, z0),
            ]
        )
    return faces


def roof_from_dsm(
    footprint: Polygon, dsm: rasterio.DatasetReader, base_z: float, eave_z: float, roof_z: float
) -> list[list[tuple[float, float, float]]]:
    data, transform = rasterio_mask(dsm, [mapping(footprint)], crop=True, filled=False)

    if data.size == 0:
        return []

    dsm_values = data[0]
    mask_values = data.mask[0] if hasattr(data, "mask") else np.zeros_like(dsm_values, dtype=bool)

    rows, cols = np.where(~mask_values & np.isfinite(dsm_values))

    if len(rows) < 3:
        return []

    xs, ys = transform_xy(transform, rows, cols)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    zs = dsm_values[rows, cols].astype(np.float64)

    # Normalize heights: map DSM range to [eave_z, roof_z]
    z_min = np.nanmin(zs)
    z_max = np.nanmax(zs)
    if z_max - z_min < 1e-3:
        return []

    # Scale relative to DSM range
    z_normalized = (zs - z_min) / (z_max - z_min)
    z_scaled = eave_z + z_normalized * (roof_z - eave_z)

    try:
        tri = Delaunay(np.column_stack([xs, ys]))
    except Exception:
        return []

    triangles: list[list[tuple[float, float, float]]] = []
    for simplex in tri.simplices:
        pts = [(float(xs[i]), float(ys[i]), float(z_scaled[i])) for i in simplex]
        pts.append(pts[0])
        triangles.append(pts)

    return triangles
