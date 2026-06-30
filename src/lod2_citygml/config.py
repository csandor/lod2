from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when user configuration is invalid."""


@dataclass(slots=True)
class RunConfig:
    footprints: Path
    orthophoto: Path
    dsm: Path
    dtm: Path
    relative_height: Path
    out_citygml: Path | None = None
    out_cityjson: Path | None = None
    citygml_version: str = "2.0"
    crs: str | None = None
    aoi: str | None = None
    max_buildings: int = 50_000
    max_area_km2: float = 25.0
    on_large_area: str = "warn"
    min_roof_confidence: float = 0.45
    roof_shape: str = "mbr"
    lod1_intermediate: Path | None = None
    tile_size: int = 2000
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


def build_config(cli_values: dict[str, Any], file_values: dict[str, Any]) -> RunConfig:
    merged = {**file_values, **{k: v for k, v in cli_values.items() if v is not None}}

    required = ["footprints", "orthophoto", "dsm", "dtm", "relative_height"]
    missing = [name for name in required if not merged.get(name)]
    if missing:
        raise ConfigError(f"Missing required input(s): {', '.join(missing)}")
    if not merged.get("out_citygml") and not merged.get("out_cityjson"):
        raise ConfigError("At least one output is required: --out-citygml and/or --out-cityjson")

    config = RunConfig(
        footprints=Path(merged["footprints"]),
        orthophoto=Path(merged["orthophoto"]),
        dsm=Path(merged["dsm"]),
        dtm=Path(merged["dtm"]),
        relative_height=Path(merged["relative_height"]),
        out_citygml=Path(merged["out_citygml"]) if merged.get("out_citygml") else None,
        out_cityjson=Path(merged["out_cityjson"]) if merged.get("out_cityjson") else None,
        citygml_version=str(merged.get("citygml_version", "2.0")),
        crs=merged.get("crs"),
        aoi=merged.get("aoi"),
        max_buildings=int(merged.get("max_buildings", 50_000)),
        max_area_km2=float(merged.get("max_area_km2", 25.0)),
        on_large_area=str(merged.get("on_large_area", "warn")),
        min_roof_confidence=float(merged.get("min_roof_confidence", 0.45)),
        roof_shape=str(merged.get("roof_shape", "mbr")),
        lod1_intermediate=Path(merged["lod1_intermediate"]) if merged.get("lod1_intermediate") else None,
        tile_size=int(merged.get("tile_size", 2000)),
        log_level=str(merged.get("log_level", "INFO")),
        report=Path(merged["report"]) if merged.get("report") else None,
    )

    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    if config.citygml_version not in {"2.0", "3.0"}:
        raise ConfigError("--citygml-version must be one of: 2.0, 3.0")
    if config.on_large_area not in {"warn", "abort", "tile"}:
        raise ConfigError("--on-large-area must be one of: warn, abort, tile")
    if config.roof_shape not in {"mbr", "footprint"}:
        raise ConfigError("--roof-shape must be one of: mbr, footprint")
    if not (0.0 <= config.min_roof_confidence <= 1.0):
        raise ConfigError("--min-roof-confidence must be in [0, 1].")
    if config.tile_size <= 0:
        raise ConfigError("--tile-size must be > 0")
    if config.max_buildings <= 0:
        raise ConfigError("--max-buildings must be > 0")
    if config.max_area_km2 <= 0:
        raise ConfigError("--max-area-km2 must be > 0")
