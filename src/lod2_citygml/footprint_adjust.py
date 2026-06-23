from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

from lod2_citygml.config import ConfigError
from lod2_citygml.footprint_adjust_config import AdjustConfig
from lod2_citygml.footprint_adjust_reporting import summarize, write_report

logger = logging.getLogger("footprint_adjust")

SUPPORTED_RASTER_SUFFIXES = {".tif", ".tiff", ".vrt", ".ecw"}


def run_adjust(config: AdjustConfig) -> gpd.GeoDataFrame:
    """Filter footprints by area, estimate height from a DSM-DTM difference
    raster, and write the surviving footprints (with a height column) to SHP."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    _validate_input_raster(config.dsm_dtm_diff)
    if not config.footprints.exists():
        raise ConfigError(f"Footprints path does not exist: {config.footprints}")

    read_kwargs = {"layer": config.input_layer} if config.input_layer else {}
    gdf = gpd.read_file(config.footprints, engine="pyogrio", **read_kwargs)
    n_input = len(gdf)
    logger.info(
        "Read %d footprint(s) from %s%s",
        n_input,
        config.footprints,
        f" (layer={config.input_layer})" if config.input_layer else "",
    )

    if config.crs:
        gdf = gdf.to_crs(config.crs)

    with rasterio.open(config.dsm_dtm_diff) as diff:
        # Footprints must share the raster CRS for masking and metric areas.
        if diff.crs is not None and gdf.crs is not None and gdf.crs != diff.crs:
            logger.info("Reprojecting footprints to raster CRS %s", diff.crs)
            gdf = gdf.to_crs(diff.crs)

        # 1. Drop small footprints up front so we never sample the raster for them.
        areas = gdf.geometry.area
        keep_area = areas >= config.min_area
        n_small = int((~keep_area).sum())
        gdf = gdf[keep_area].copy()
        if n_small:
            logger.info(
                "Dropped %d footprint(s) below min_area=%.1f m^2", n_small, config.min_area
            )

        # 2. Estimate height per footprint from the difference raster.
        heights: list[float] = []
        for geom in gdf.geometry:
            heights.append(_estimate_height(diff, geom, config.buffer))

    gdf[config.height_column] = heights

    # 3. Drop footprints below the height threshold (NaN heights also dropped:
    #    no usable raster coverage means no defensible height).
    height_vals = gdf[config.height_column]
    keep_height = height_vals.notna() & (height_vals >= config.min_height)
    n_low = int((~keep_height).sum())
    gdf = gdf[keep_height].copy()
    if n_low:
        logger.info(
            "Dropped %d footprint(s) below min_height=%.1f m (incl. no raster coverage)",
            n_low,
            config.min_height,
        )

    gdf = gdf.reset_index(drop=True)

    _write_footprints(gdf, config.out_footprints, config.output_layer)
    logger.info(
        "Wrote %d footprint(s) to %s%s",
        len(gdf),
        config.out_footprints,
        f" (layer={config.output_layer})" if config.output_layer else "",
    )

    stats = summarize(gdf, n_input, config.height_column)
    write_report(stats, config.report)
    return gdf


def _estimate_height(diff: rasterio.DatasetReader, geom, buffer_m: float) -> float:
    """Median DSM-DTM difference value inside an inward-buffered footprint."""
    if geom is None or geom.is_empty:
        return float("nan")

    sample_geom = geom
    if buffer_m > 0:
        eroded = geom.buffer(-buffer_m)
        # A footprint thinner than 2*buffer collapses to empty; fall back to the
        # original geometry so narrow buildings still get a height.
        if not eroded.is_empty and eroded.area > 0:
            sample_geom = eroded

    try:
        vals = _masked_values(diff, sample_geom)
    except ValueError:
        # Geometry falls entirely outside the raster extent.
        return float("nan")

    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def _masked_values(dataset: rasterio.DatasetReader, geom) -> np.ndarray:
    data, _ = mask(dataset, [mapping(geom)], crop=True, filled=False)
    arr = np.asarray(data[0]).astype(float)
    if hasattr(data, "mask"):
        arr = arr[~data.mask[0]]
    return arr[np.isfinite(arr)]


def _validate_input_raster(path: Path) -> None:
    if not path.exists():
        raise ConfigError(f"DSM-DTM difference raster does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
        suffixes = ", ".join(sorted(SUPPORTED_RASTER_SUFFIXES))
        raise ConfigError(f"DSM-DTM difference raster must be one of [{suffixes}]: {path}")


VECTOR_DRIVERS = {
    ".shp": "ESRI Shapefile",
    ".gpkg": "GPKG",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
}


def _driver_for(path: Path) -> str:
    driver = VECTOR_DRIVERS.get(path.suffix.lower())
    if driver is None:
        suffixes = ", ".join(sorted(VECTOR_DRIVERS))
        raise ConfigError(f"Unsupported output format [{path.suffix}]; use one of: {suffixes}")
    return driver


def _write_footprints(
    gdf: gpd.GeoDataFrame, out_path: Path, output_layer: str | None = None
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    driver = _driver_for(out_path)
    write_kwargs: dict = {"driver": driver, "engine": "pyogrio"}
    # Only GPKG carries named layers; Shapefile/GeoJSON are single-layer.
    if output_layer:
        if driver != "GPKG":
            raise ConfigError("--output-layer is only supported for .gpkg output.")
        write_kwargs["layer"] = output_layer
    if gdf.empty:
        # Preserve an empty but valid dataset so downstream steps don't choke.
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=gdf.crs)
        empty.to_file(out_path, **write_kwargs)
        return
    gdf.to_file(out_path, **write_kwargs)
