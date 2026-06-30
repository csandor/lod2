from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from pyproj import CRS
import rasterio
from shapely.geometry import box

from lod2_citygml.config import ConfigError, RunConfig


@dataclass(slots=True)
class WorkloadEstimate:
    buildings: int
    area_km2: float


def _crs_equal(a: CRS, b: CRS) -> bool:
    return a == b


def validate_inputs(
    config: RunConfig,
    footprints: gpd.GeoDataFrame,
    ortho: rasterio.DatasetReader,
    dsm: rasterio.DatasetReader,
    dtm: rasterio.DatasetReader,
    rel_height: rasterio.DatasetReader,
) -> WorkloadEstimate:
    if footprints.crs is None:
        raise ConfigError("Footprints CRS is missing.")
    if ortho.crs is None or dsm.crs is None or dtm.crs is None or rel_height.crs is None:
        raise ConfigError("One or more raster CRSs are missing.")

    fp_crs = CRS.from_user_input(footprints.crs)
    o_crs = CRS.from_user_input(ortho.crs)
    dsm_crs = CRS.from_user_input(dsm.crs)
    dtm_crs = CRS.from_user_input(dtm.crs)
    rh_crs = CRS.from_user_input(rel_height.crs)

    if not (_crs_equal(fp_crs, o_crs) and _crs_equal(fp_crs, dsm_crs) and _crs_equal(fp_crs, dtm_crs) and _crs_equal(fp_crs, rh_crs)):
        raise ConfigError("All inputs must use the same projected CRS.")
    if not fp_crs.is_projected:
        raise ConfigError("Input CRS must be projected (uniform cartesian reference system).")

    if ortho.count not in (3, 4):
        raise ConfigError(f"Orthophoto must have 3 or 4 bands (RGB or RGBA), got {ortho.count}.")
    if dsm.count != 1:
        raise ConfigError(f"DSM must have 1 band, got {dsm.count}.")
    if dtm.count != 1:
        raise ConfigError(f"DTM must have 1 band, got {dtm.count}.")
    if rel_height.count != 1:
        raise ConfigError(f"Relative height must have 1 band, got {rel_height.count}.")

    fp_bounds = box(*footprints.total_bounds)
    raster_union = box(*ortho.bounds).intersection(box(*dsm.bounds)).intersection(box(*dtm.bounds)).intersection(box(*rel_height.bounds))
    if raster_union.is_empty or not raster_union.intersects(fp_bounds):
        raise ConfigError("No spatial overlap between footprints and shared raster extent.")

    invalid_count = (~footprints.geometry.is_valid).sum()
    if invalid_count > 0:
        raise ConfigError(f"Found {invalid_count} invalid footprint geometries.")

    area_km2 = footprints.geometry.area.sum() / 1_000_000.0
    estimate = WorkloadEstimate(buildings=len(footprints), area_km2=area_km2)

    if estimate.buildings > config.max_buildings or estimate.area_km2 > config.max_area_km2:
        msg = (
            "Input area is larger than configured limits: "
            f"{estimate.buildings} buildings, {estimate.area_km2:.3f} km² "
            f"(limits {config.max_buildings}, {config.max_area_km2:.3f} km²)."
        )
        if config.on_large_area == "abort":
            raise ConfigError(msg)
        if config.on_large_area == "warn":
            print(f"WARNING: {msg}")
        if config.on_large_area == "tile":
            print(f"INFO: {msg} Processing should be tiled (--tile-size={config.tile_size}).")

    return estimate
