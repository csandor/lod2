from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio

from lod2_citygml.config import ConfigError, RunConfig


SUPPORTED_RASTER_SUFFIXES = {".tif", ".tiff", ".vrt", ".ecw"}


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise ConfigError(f"{label} path does not exist: {path}")


def _check_raster_suffix(path: Path, label: str) -> None:
    if path.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
        suffixes = ", ".join(sorted(SUPPORTED_RASTER_SUFFIXES))
        raise ConfigError(f"{label} must be one of [{suffixes}]: {path}")


def _open_raster(path: Path, label: str) -> rasterio.DatasetReader:
    try:
        return rasterio.open(path)
    except Exception as exc:  # pragma: no cover - depends on local GDAL drivers
        msg = f"Failed to open {label}: {path} ({exc})"
        if path.suffix.lower() == ".ecw":
            msg += " - ECW driver support is not available in the current GDAL/Rasterio environment."
        raise ConfigError(msg) from exc


def load_inputs(config: RunConfig) -> tuple[gpd.GeoDataFrame, rasterio.DatasetReader, rasterio.DatasetReader, rasterio.DatasetReader]:
    _check_exists(config.footprints, "Footprints")
    _check_exists(config.orthophoto, "Orthophoto")
    _check_exists(config.dsm, "DSM")
    _check_exists(config.dtm, "DTM")

    _check_raster_suffix(config.orthophoto, "Orthophoto")
    _check_raster_suffix(config.dsm, "DSM")
    _check_raster_suffix(config.dtm, "DTM")

    footprints = gpd.read_file(config.footprints, engine="pyogrio")
    if footprints.empty:
        raise ConfigError("Footprints dataset is empty.")
    if "geometry" not in footprints.columns:
        raise ConfigError("Footprints dataset has no geometry column.")

    ortho = _open_raster(config.orthophoto, "Orthophoto")
    dsm = _open_raster(config.dsm, "DSM")
    dtm = _open_raster(config.dtm, "DTM")

    return footprints, ortho, dsm, dtm
