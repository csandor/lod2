from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lod2_citygml.config import ConfigError


@dataclass(slots=True)
class DetectConfig:
    """Configuration for building-footprint detection from an orthophoto."""

    orthophoto: Path
    out_footprints: Path
    backend: str = "mask_rcnn"
    model_path: str = "building_footprints_usa.pth"
    model_repo_id: str | None = None
    device: str | None = None
    # mask_rcnn (BuildingFootprintExtractor.process_raster) parameters
    batch_size: int = 4
    confidence_threshold: float = 0.5
    overlap: float = 0.25
    nms_iou_threshold: float = 0.5
    mask_threshold: float = 0.5
    min_object_area: float = 100.0
    max_object_area: float | None = None
    simplify_tolerance: float = 1.0
    filter_edges: bool = True
    edge_buffer: int = 20
    band_indexes: list[int] | None = None
    # Resample the orthophoto to this ground sample distance (raster CRS units,
    # typically metres/pixel) before detection. The pretrained models expect
    # ~0.3-1 m/px imagery; very high-res input (e.g. 5 cm) must be downsampled
    # or buildings are detected as tiny fragments. None = use native resolution.
    target_resolution: float | None = None
    # timm semantic-segmentation backend parameters
    window_size: int = 512
    seg_overlap: int = 256
    probability_threshold: float | None = None
    orthogonalize_epsilon: float = 2.0
    # regularization
    regularize: str = "hybrid"
    min_area: float = 100.0
    angle_threshold: float = 15.0
    orthogonality_threshold: float = 0.3
    rectangularity_threshold: float = 0.7
    angle_tolerance: float = 10.0
    reg_simplify_tolerance: float = 0.5
    area_threshold: float = 0.9
    # output / misc
    crs: str | None = None
    aoi: str | None = None
    aoi_mode: str = "crop"
    keep_mask: bool = False
    log_level: str = "INFO"
    report: Path | None = None


VALID_BACKENDS = {"mask_rcnn", "timm"}
VALID_REGULARIZE = {"none", "extractor", "regularization", "hybrid", "adaptive"}
VALID_AOI_MODES = {"crop", "filter"}


def _parse_optional_resolution(value: Any) -> float | None:
    """Parse target_resolution: None/null, 'none', or 0 mean no resampling."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "native", "0"}:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ConfigError(
                f"--target-resolution must be a number or 'none': {value!r}"
            ) from exc
    value = float(value)
    return None if value == 0 else value


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


def build_detect_config(
    cli_values: dict[str, Any], file_values: dict[str, Any]
) -> DetectConfig:
    merged = {**file_values, **{k: v for k, v in cli_values.items() if v is not None}}

    required = ["orthophoto", "out_footprints"]
    missing = [name for name in required if not merged.get(name)]
    if missing:
        raise ConfigError(f"Missing required input(s): {', '.join(missing)}")

    config = DetectConfig(
        orthophoto=Path(merged["orthophoto"]),
        out_footprints=Path(merged["out_footprints"]),
        backend=str(merged.get("backend", "mask_rcnn")),
        model_path=str(merged.get("model_path", "building_footprints_usa.pth")),
        model_repo_id=merged.get("model_repo_id"),
        device=merged.get("device"),
        batch_size=int(merged.get("batch_size", 4)),
        confidence_threshold=float(merged.get("confidence_threshold", 0.5)),
        overlap=float(merged.get("overlap", 0.25)),
        nms_iou_threshold=float(merged.get("nms_iou_threshold", 0.5)),
        mask_threshold=float(merged.get("mask_threshold", 0.5)),
        min_object_area=float(merged.get("min_object_area", 100.0)),
        max_object_area=(
            float(merged["max_object_area"])
            if merged.get("max_object_area") is not None
            else None
        ),
        simplify_tolerance=float(merged.get("simplify_tolerance", 1.0)),
        filter_edges=bool(merged.get("filter_edges", True)),
        edge_buffer=int(merged.get("edge_buffer", 20)),
        band_indexes=merged.get("band_indexes"),
        target_resolution=_parse_optional_resolution(merged.get("target_resolution")),
        window_size=int(merged.get("window_size", 512)),
        seg_overlap=int(merged.get("seg_overlap", 256)),
        probability_threshold=(
            float(merged["probability_threshold"])
            if merged.get("probability_threshold") is not None
            else None
        ),
        orthogonalize_epsilon=float(merged.get("orthogonalize_epsilon", 2.0)),
        regularize=str(merged.get("regularize", "hybrid")),
        min_area=float(merged.get("min_area", 100.0)),
        angle_threshold=float(merged.get("angle_threshold", 15.0)),
        orthogonality_threshold=float(merged.get("orthogonality_threshold", 0.3)),
        rectangularity_threshold=float(merged.get("rectangularity_threshold", 0.7)),
        angle_tolerance=float(merged.get("angle_tolerance", 10.0)),
        reg_simplify_tolerance=float(merged.get("reg_simplify_tolerance", 0.5)),
        area_threshold=float(merged.get("area_threshold", 0.9)),
        crs=merged.get("crs"),
        aoi=merged.get("aoi"),
        aoi_mode=str(merged.get("aoi_mode", "crop")),
        keep_mask=bool(merged.get("keep_mask", False)),
        log_level=str(merged.get("log_level", "INFO")),
        report=Path(merged["report"]) if merged.get("report") else None,
    )

    validate_detect_config(config)
    return config


def validate_detect_config(config: DetectConfig) -> None:
    if config.backend not in VALID_BACKENDS:
        raise ConfigError(
            f"--backend must be one of: {', '.join(sorted(VALID_BACKENDS))}"
        )
    if config.regularize not in VALID_REGULARIZE:
        raise ConfigError(
            f"--regularize must be one of: {', '.join(sorted(VALID_REGULARIZE))}"
        )
    if config.aoi_mode not in VALID_AOI_MODES:
        raise ConfigError(
            f"--aoi-mode must be one of: {', '.join(sorted(VALID_AOI_MODES))}"
        )
    if config.backend == "timm" and not config.model_repo_id:
        raise ConfigError(
            "--model-repo-id is required when --backend=timm "
            "(e.g. giswqs/whu-building-unetplusplus-efficientnet-b4)."
        )
    if config.backend == "timm" and config.regularize == "extractor":
        raise ConfigError(
            "--regularize=extractor is only available with --backend=mask_rcnn."
        )
    if not (0.0 <= config.confidence_threshold <= 1.0):
        raise ConfigError("--confidence-threshold must be in [0, 1].")
    if not (0.0 <= config.mask_threshold <= 1.0):
        raise ConfigError("--mask-threshold must be in [0, 1].")
    if not (0.0 <= config.overlap < 1.0):
        raise ConfigError("--overlap must be in [0, 1).")
    if config.batch_size <= 0:
        raise ConfigError("--batch-size must be > 0")
    if config.window_size <= 0:
        raise ConfigError("--window-size must be > 0")
    if config.min_object_area < 0 or config.min_area < 0:
        raise ConfigError("area thresholds must be >= 0")
    if config.target_resolution is not None and config.target_resolution <= 0:
        raise ConfigError("--target-resolution must be > 0")
