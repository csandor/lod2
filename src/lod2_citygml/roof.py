from __future__ import annotations

import cv2
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import LineString, mapping

from lod2_citygml.models import BuildingRecord


def _extract_lines(dataset: rasterio.DatasetReader, geom) -> list[LineString]:
    data, transform = mask(dataset, [mapping(geom)], crop=True, filled=True)
    arr = np.asarray(data[0], dtype=np.float32)
    if arr.size == 0:
        return []

    # Normalize local elevation patch to [0, 255] for edge extraction.
    finite = np.isfinite(arr)
    if not finite.any():
        return []
    min_v = float(np.nanmin(arr[finite]))
    max_v = float(np.nanmax(arr[finite]))
    if max_v - min_v < 1e-6:
        return []

    img = ((arr - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    edges = cv2.Canny(img, threshold1=30, threshold2=90)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=15, minLineLength=5, maxLineGap=5)

    if lines is None:
        return []

    from rasterio.transform import xy as transform_xy

    result: list[LineString] = []
    for line in lines[:20]:
        x1, y1, x2, y2 = line[0]
        if x1 == x2 and y1 == y2:
            continue
        # Convert pixel coordinates to world coordinates
        wx1, wy1 = transform_xy(transform, y1, x1)
        wx2, wy2 = transform_xy(transform, y2, x2)
        result.append(LineString([(wx1, wy1), (wx2, wy2)]))
    return result


def infer_roof_kind(records: list[BuildingRecord], dsm: rasterio.DatasetReader, min_confidence: float) -> list[BuildingRecord]:
    updated: list[BuildingRecord] = []
    for rec in records:
        lines = _extract_lines(dsm, rec.footprint)
        conf = rec.roof_confidence

        if len(lines) >= 6:
            kind = "complex"
        elif len(lines) >= 2:
            kind = "gable_or_hip"
        else:
            kind = "flat"

        updated.append(
            BuildingRecord(
                building_id=rec.building_id,
                footprint=rec.footprint,
                base_z=rec.base_z,
                roof_z=rec.roof_z,
                height=rec.height,
                roof_confidence=conf,
                roof_kind=kind,
                roof_lines=lines if lines else None,
            )
        )

    return updated
