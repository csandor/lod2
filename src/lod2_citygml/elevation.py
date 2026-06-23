from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

from lod2_citygml.models import BuildingRecord


def _masked_values(dataset: rasterio.DatasetReader, geom) -> np.ndarray:
    data, _ = mask(dataset, [mapping(geom)], crop=True, filled=False)
    arr = np.asarray(data[0]).astype(float)
    if hasattr(data, "mask"):
        arr = arr[~data.mask[0]]
    return arr[np.isfinite(arr)]


def _robust_base_z(dtm_vals: np.ndarray) -> float:
    if dtm_vals.size == 0:
        return float("nan")
    return float(np.nanmin(dtm_vals))


def _robust_roof_z(dsm_vals: np.ndarray) -> float:
    if dsm_vals.size == 0:
        return float("nan")
    return float(np.nanmax(dsm_vals))


def estimate_buildings(footprints: gpd.GeoDataFrame, dsm: rasterio.DatasetReader, dtm: rasterio.DatasetReader) -> list[BuildingRecord]:
    records: list[BuildingRecord] = []

    id_col = next((c for c in ["id", "ID", "fid", "FID", "OBJECTID"] if c in footprints.columns), None)

    for i, row in footprints.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        if geom.geom_type != "Polygon" or geom.area <= 0.0:
            continue

        dtm_vals = _masked_values(dtm, geom)
        dsm_vals = _masked_values(dsm, geom)

        base_z = _robust_base_z(dtm_vals)
        roof_z = _robust_roof_z(dsm_vals)

        if not (math.isfinite(base_z) and math.isfinite(roof_z)):
            continue

        height = max(roof_z - base_z, 0.1)

        local_spread = float(np.nanpercentile(dsm_vals, 95.0) - np.nanpercentile(dsm_vals, 50.0)) if dsm_vals.size else 0.0
        roof_confidence = float(min(1.0, max(0.0, local_spread / 6.0)))

        building_id = str(row[id_col]) if id_col is not None else str(i)
        records.append(
            BuildingRecord(
                building_id=building_id,
                footprint=geom,
                base_z=base_z,
                roof_z=roof_z,
                height=height,
                roof_confidence=roof_confidence,
                roof_kind="unknown",
            )
        )

    return records
