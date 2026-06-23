from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lod2_citygml.config import ConfigError


@dataclass(slots=True)
class AdjustConfig:
    """Configuration for footprint area-filtering and height estimation."""

    footprints: Path
    dsm_dtm_diff: Path
    out_footprints: Path
    # Layer to read from a multi-layer input (e.g. GPKG). None = default/first layer.
    input_layer: str | None = None
    # Layer name to write in a multi-layer output (e.g. GPKG). None = file stem.
    output_layer: str | None = None
    # Drop footprints with a 2D area smaller than this (m²).
    min_area: float = 30.0
    # Negative (inward) buffer applied before sampling the height raster (m).
    # Erodes footprint edges so overhanging roofs / mixed edge pixels don't
    # pollute the median height.
    buffer: float = 1.0
    # Drop footprints whose estimated height is below this (m).
    min_height: float = 2.0
    # Output column name for the estimated height.
    height_column: str = "height"
    crs: str | None = None
    log_level: str = "INFO"
    report: Path | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("Config file root must be a mapping/object.")
    return data


def load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    return _load_yaml(path)


def build_adjust_config(
    cli_values: dict[str, Any], file_values: dict[str, Any]
) -> AdjustConfig:
    merged = {**file_values, **{k: v for k, v in cli_values.items() if v is not None}}

    required = ["footprints", "dsm_dtm_diff", "out_footprints"]
    missing = [name for name in required if not merged.get(name)]
    if missing:
        raise ConfigError(f"Missing required input(s): {', '.join(missing)}")

    config = AdjustConfig(
        footprints=Path(merged["footprints"]),
        dsm_dtm_diff=Path(merged["dsm_dtm_diff"]),
        out_footprints=Path(merged["out_footprints"]),
        input_layer=merged.get("input_layer"),
        output_layer=merged.get("output_layer"),
        min_area=float(merged.get("min_area", 30.0)),
        buffer=float(merged.get("buffer", 1.0)),
        min_height=float(merged.get("min_height", 2.0)),
        height_column=str(merged.get("height_column", "height")),
        crs=merged.get("crs"),
        log_level=str(merged.get("log_level", "INFO")),
        report=Path(merged["report"]) if merged.get("report") else None,
    )

    validate_adjust_config(config)
    return config


def validate_adjust_config(config: AdjustConfig) -> None:
    if config.min_area < 0:
        raise ConfigError("--min-area must be >= 0")
    if config.buffer < 0:
        raise ConfigError("--buffer must be >= 0 (it is applied inward).")
    if config.min_height < 0:
        raise ConfigError("--min-height must be >= 0")
    if not config.height_column:
        raise ConfigError("--height-column must be a non-empty column name.")
    # Shapefile (.dbf) column names are limited to 10 characters; other formats
    # (GPKG, GeoJSON) have no such limit, so only enforce it for .shp output.
    if config.out_footprints.suffix.lower() == ".shp" and len(config.height_column) > 10:
        raise ConfigError(
            "--height-column must be <= 10 chars for ESRI Shapefile output: "
            f"{config.height_column!r}"
        )
