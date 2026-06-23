from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from lod2_citygml.config import ConfigError, build_config
from lod2_citygml.config import load_config_file as load_build_config_file
from lod2_citygml.footprint_adjust_config import build_adjust_config
from lod2_citygml.footprint_adjust_config import load_config_file as load_adjust_config_file
from lod2_citygml.footprints_config import build_detect_config
from lod2_citygml.footprints_config import load_config_file as load_detect_config_file
from lod2_citygml.pipeline import run_pipeline

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Create LOD2 CityGML buildings from footprints + orthophoto + DSM + DTM.",
)


@app.command()
def build(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Optional YAML config file. CLI values override config values.",
    ),
    footprints: Optional[Path] = typer.Option(None, "--footprints", help="Building footprints (SHP)."),
    orthophoto: Optional[Path] = typer.Option(
        None,
        "--orthophoto",
        help="Orthophoto raster (3-band TIFF, VRT, or ECW).",
    ),
    dsm: Optional[Path] = typer.Option(None, "--dsm", help="DSM raster (single-band TIFF, VRT, or ECW)."),
    dtm: Optional[Path] = typer.Option(None, "--dtm", help="DTM raster (single-band TIFF, VRT, or ECW)."),
    out_citygml: Optional[Path] = typer.Option(None, "--out-citygml", help="Output CityGML file path."),
    out_cityjson: Optional[Path] = typer.Option(None, "--out-cityjson", help="Output CityJSON file path."),
    citygml_version: Optional[str] = typer.Option("2.0", "--citygml-version", help="2.0 or 3.0."),
    crs: Optional[str] = typer.Option(None, "--crs", help="Optional override CRS (EPSG code or WKT)."),
    aoi: Optional[str] = typer.Option(
        None,
        "--aoi",
        help="Optional AOI path or bbox as minx,miny,maxx,maxy.",
    ),
    max_buildings: Optional[int] = typer.Option(50_000, "--max-buildings", help="Large-area threshold."),
    max_area_km2: Optional[float] = typer.Option(25.0, "--max-area-km2", help="Large-area threshold in km²."),
    on_large_area: Optional[str] = typer.Option("warn", "--on-large-area", help="warn | abort | tile"),
    quality: Optional[str] = typer.Option("high_precision", "--quality", help="Supported: high_precision"),
    roof_detail: Optional[str] = typer.Option("fine", "--roof-detail", help="Supported: fine"),
    min_roof_confidence: Optional[float] = typer.Option(0.45, "--min-roof-confidence", help="0-1 threshold."),
    lod1_intermediate: Optional[Path] = typer.Option(
        None,
        "--lod1-intermediate",
        help="Optional output for intermediate LOD1 footprints (GPKG).",
    ),
    fallback_roof_mode: Optional[str] = typer.Option(
        "simple_planes",
        "--fallback-roof-mode",
        help="flat | simple_planes",
    ),
    tile_size: Optional[int] = typer.Option(2000, "--tile-size", help="Tile width for chunking mode."),
    workers: Optional[int] = typer.Option(1, "--workers", help="Worker count."),
    log_level: Optional[str] = typer.Option("INFO", "--log-level", help="DEBUG|INFO|WARNING|ERROR"),
    report: Optional[Path] = typer.Option(None, "--report", help="Optional QA JSON report path."),
) -> None:
    """Run full LOD2 reconstruction and export CityGML and/or CityJSON."""
    cli_values = {
        "footprints": footprints,
        "orthophoto": orthophoto,
        "dsm": dsm,
        "dtm": dtm,
        "out_citygml": out_citygml,
        "out_cityjson": out_cityjson,
        "citygml_version": citygml_version,
        "crs": crs,
        "aoi": aoi,
        "max_buildings": max_buildings,
        "max_area_km2": max_area_km2,
        "on_large_area": on_large_area,
        "quality": quality,
        "roof_detail": roof_detail,
        "min_roof_confidence": min_roof_confidence,
        "lod1_intermediate": lod1_intermediate,
        "fallback_roof_mode": fallback_roof_mode,
        "tile_size": tile_size,
        "workers": workers,
        "log_level": log_level,
        "report": report,
    }

    try:
        file_values = load_build_config_file(config)
        run_config = build_config(cli_values=cli_values, file_values=file_values)
        run_pipeline(run_config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("detect-footprints")
def detect_footprints(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Optional YAML config file. CLI values override config values.",
    ),
    orthophoto: Optional[Path] = typer.Option(
        None,
        "--orthophoto",
        help="Input orthophoto raster (3/4-band TIFF, VRT, or ECW).",
    ),
    out_footprints: Optional[Path] = typer.Option(
        None,
        "--out-footprints",
        help="Output building footprints (SHP).",
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="mask_rcnn | timm (default: mask_rcnn)."
    ),
    model_path: Optional[str] = typer.Option(
        None, "--model-path", help="Mask R-CNN weights path or filename (auto-downloaded)."
    ),
    model_repo_id: Optional[str] = typer.Option(
        None,
        "--model-repo-id",
        help="HuggingFace repo id (required for --backend=timm).",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Torch device, e.g. cuda, cpu (default: auto)."
    ),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Inference batch size."),
    confidence_threshold: Optional[float] = typer.Option(
        None, "--confidence-threshold", help="0-1 detection confidence (mask_rcnn)."
    ),
    overlap: Optional[float] = typer.Option(
        None, "--overlap", help="Tile overlap fraction 0-1 (mask_rcnn)."
    ),
    nms_iou_threshold: Optional[float] = typer.Option(
        None, "--nms-iou-threshold", help="NMS IoU threshold (mask_rcnn)."
    ),
    mask_threshold: Optional[float] = typer.Option(
        None, "--mask-threshold", help="0-1 mask binarization threshold."
    ),
    min_object_area: Optional[float] = typer.Option(
        None, "--min-object-area", help="Min detected object area (px/m²)."
    ),
    max_object_area: Optional[float] = typer.Option(
        None, "--max-object-area", help="Max detected object area."
    ),
    simplify_tolerance: Optional[float] = typer.Option(
        None, "--simplify-tolerance", help="Polygon simplification tolerance."
    ),
    target_resolution: Optional[str] = typer.Option(
        None,
        "--target-resolution",
        help="Resample orthophoto to this ground sample distance (m/px) before "
        "detection. Models expect ~0.3-1 m/px; downsample finer imagery. "
        "Use 'none' or 0 to keep native resolution (no resampling).",
    ),
    window_size: Optional[int] = typer.Option(
        None, "--window-size", help="Tile window size (timm)."
    ),
    seg_overlap: Optional[int] = typer.Option(
        None, "--seg-overlap", help="Tile overlap in pixels (timm)."
    ),
    orthogonalize_epsilon: Optional[float] = typer.Option(
        None, "--orthogonalize-epsilon", help="orthogonalize() epsilon (timm)."
    ),
    regularize: Optional[str] = typer.Option(
        None,
        "--regularize",
        help="none | extractor | regularization | hybrid | adaptive (default: hybrid).",
    ),
    min_area: Optional[float] = typer.Option(
        None, "--min-area", help="Min footprint area (m²) kept in output."
    ),
    angle_threshold: Optional[float] = typer.Option(
        None, "--angle-threshold", help="Angle threshold for extractor regularization."
    ),
    angle_tolerance: Optional[float] = typer.Option(
        None, "--angle-tolerance", help="Angle tolerance for regularization()."
    ),
    crs: Optional[str] = typer.Option(None, "--crs", help="Optional output CRS (EPSG/WKT)."),
    aoi: Optional[str] = typer.Option(
        None, "--aoi", help="Optional AOI path or bbox minx,miny,maxx,maxy."
    ),
    aoi_mode: Optional[str] = typer.Option(
        None,
        "--aoi-mode",
        help="crop (crop orthophoto to AOI before detection) | filter (detect full raster, clip after). Default: crop.",
    ),
    keep_mask: Optional[bool] = typer.Option(
        None, "--keep-mask/--no-keep-mask", help="Keep intermediate mask raster (timm)."
    ),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="DEBUG|INFO|WARNING|ERROR"),
    report: Optional[Path] = typer.Option(None, "--report", help="Optional JSON report path."),
) -> None:
    """Detect building footprints in an orthophoto and export them as SHP."""
    cli_values = {
        "orthophoto": orthophoto,
        "out_footprints": out_footprints,
        "backend": backend,
        "model_path": model_path,
        "model_repo_id": model_repo_id,
        "device": device,
        "batch_size": batch_size,
        "confidence_threshold": confidence_threshold,
        "overlap": overlap,
        "nms_iou_threshold": nms_iou_threshold,
        "mask_threshold": mask_threshold,
        "min_object_area": min_object_area,
        "max_object_area": max_object_area,
        "simplify_tolerance": simplify_tolerance,
        "target_resolution": target_resolution,
        "window_size": window_size,
        "seg_overlap": seg_overlap,
        "orthogonalize_epsilon": orthogonalize_epsilon,
        "regularize": regularize,
        "min_area": min_area,
        "angle_threshold": angle_threshold,
        "angle_tolerance": angle_tolerance,
        "crs": crs,
        "aoi": aoi,
        "aoi_mode": aoi_mode,
        "keep_mask": keep_mask,
        "log_level": log_level,
        "report": report,
    }

    try:
        file_values = load_detect_config_file(config)
        detect_config = build_detect_config(cli_values=cli_values, file_values=file_values)
        # Imported lazily: pulls in torch/geoai only when this command runs.
        from lod2_citygml.footprints_detect import run_detection

        run_detection(detect_config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("footprint-adjust")
def footprint_adjust(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Optional YAML config file. CLI values override config values.",
    ),
    footprints: Optional[Path] = typer.Option(
        None, "--footprints", help="Input building footprints (SHP, GPKG, or GeoJSON)."
    ),
    input_layer: Optional[str] = typer.Option(
        None, "--input-layer", help="Layer to read from a multi-layer input (e.g. GPKG)."
    ),
    dsm_dtm_diff: Optional[Path] = typer.Option(
        None,
        "--dsm-dtm-diff",
        help="DSM-DTM difference raster (single-band TIFF, VRT, or ECW).",
    ),
    out_footprints: Optional[Path] = typer.Option(
        None, "--out-footprints", help="Output filtered footprints (SHP, GPKG, or GeoJSON)."
    ),
    output_layer: Optional[str] = typer.Option(
        None, "--output-layer", help="Layer name to write in a GPKG output."
    ),
    min_area: Optional[float] = typer.Option(
        None, "--min-area", help="Drop footprints with 2D area below this (m²)."
    ),
    buffer: Optional[float] = typer.Option(
        None,
        "--buffer",
        help="Inward (negative) buffer applied before sampling height (m).",
    ),
    min_height: Optional[float] = typer.Option(
        None, "--min-height", help="Drop footprints with estimated height below this (m)."
    ),
    height_column: Optional[str] = typer.Option(
        None, "--height-column", help="Output column name for height (<=10 chars)."
    ),
    crs: Optional[str] = typer.Option(None, "--crs", help="Optional output CRS (EPSG/WKT)."),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="DEBUG|INFO|WARNING|ERROR"),
    report: Optional[Path] = typer.Option(None, "--report", help="Optional JSON report path."),
) -> None:
    """Filter footprints by area and estimate building height from a DSM-DTM
    difference raster, writing the result as SHP."""
    cli_values = {
        "footprints": footprints,
        "input_layer": input_layer,
        "dsm_dtm_diff": dsm_dtm_diff,
        "out_footprints": out_footprints,
        "output_layer": output_layer,
        "min_area": min_area,
        "buffer": buffer,
        "min_height": min_height,
        "height_column": height_column,
        "crs": crs,
        "log_level": log_level,
        "report": report,
    }

    try:
        file_values = load_adjust_config_file(config)
        adjust_config = build_adjust_config(cli_values=cli_values, file_values=file_values)
        from lod2_citygml.footprint_adjust import run_adjust

        run_adjust(adjust_config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()
