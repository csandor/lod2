from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from lod2_citygml.config import ConfigError
from lod2_citygml.footprints_config import DetectConfig
from lod2_citygml.footprints_reporting import summarize, write_report

logger = logging.getLogger("footprints")

SUPPORTED_RASTER_SUFFIXES = {".tif", ".tiff", ".vrt", ".ecw"}


def run_detection(config: DetectConfig) -> gpd.GeoDataFrame:
    """Detect building footprints in an orthophoto and write them as SHP."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    _validate_input_raster(config.orthophoto)
    config.out_footprints.parent.mkdir(parents=True, exist_ok=True)

    # geoai pulls in torch and large model deps; import lazily so config errors
    # surface fast without paying the import cost.
    import geoai

    device = _resolve_device(config.device)
    logger.info("Using device: %s", device)

    # Prepare the raster fed to the detector:
    #  - AOI "crop" mode window-reads only the AOI, so we never run inference over
    #    the whole image. "filter" mode detects on the full raster and clips after.
    #  - target_resolution resamples to a target ground sample distance, because
    #    the pretrained models expect ~0.3-1 m/px imagery.
    # Both happen in a single read when applicable; the result is a temp GeoTIFF.
    detect_path = config.orthophoto
    prepared_tmp: Path | None = None
    crop_aoi = config.aoi if config.aoi_mode == "crop" else None
    if crop_aoi or config.target_resolution is not None:
        detect_path, prepared_tmp = _prepare_raster(config, crop_aoi)

    try:
        if config.backend == "mask_rcnn":
            gdf = _detect_mask_rcnn(geoai, config, detect_path, device)
        else:
            gdf = _detect_timm(geoai, config, detect_path, device)

        logger.info("Detected %d raw footprint(s).", len(gdf))

        if config.crs:
            gdf = gdf.to_crs(config.crs)

        # In "filter" mode the AOI still has to clip the full-raster detections.
        # In "crop" mode we already restricted the input, so clipping is a no-op
        # but harmless (and trims any footprints touching the crop edge).
        if config.aoi:
            gdf = _clip_to_aoi(gdf, config.aoi)
            logger.info("After AOI clip: %d footprint(s).", len(gdf))

        gdf = _regularize(geoai, gdf, config)

        gdf = _finalize(geoai, gdf, config)

        _write_footprints(gdf, config.out_footprints)
        logger.info("Wrote %d footprint(s) to %s", len(gdf), config.out_footprints)

        stats = summarize(gdf)
        write_report(stats, config.report)
        return gdf
    finally:
        # The prepared (cropped/resampled) raster is an internal intermediate;
        # keep it only when the user asked to keep intermediates (same flag that
        # preserves the timm mask).
        if prepared_tmp is not None and not config.keep_mask:
            prepared_tmp.unlink(missing_ok=True)
        elif prepared_tmp is not None:
            logger.info("Kept prepared raster: %s", prepared_tmp)


def _resolve_device(requested: str | None) -> str:
    """Return a usable torch device string.

    If the user requested one, honor it. Otherwise prefer CUDA, but only when it
    truly initializes — a torch built against a newer CUDA than the installed
    driver reports is_available() inconsistently and crashes on first use (as the
    timm backend does), so probe defensively and fall back to CPU.
    """
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros(1).cuda()  # force CUDA init; raises on driver mismatch
            return "cuda"
    except Exception as exc:  # pragma: no cover - depends on local GPU/driver
        logger.warning("CUDA unavailable (%s); using CPU.", exc)
    return "cpu"


def _validate_input_raster(path: Path) -> None:
    if not path.exists():
        raise ConfigError(f"Orthophoto path does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
        suffixes = ", ".join(sorted(SUPPORTED_RASTER_SUFFIXES))
        raise ConfigError(f"Orthophoto must be one of [{suffixes}]: {path}")


def _detect_mask_rcnn(
    geoai, config: DetectConfig, raster_path: Path, device: str
) -> gpd.GeoDataFrame:
    """BuildingFootprintExtractor (Mask R-CNN) -> vector footprints directly."""
    extractor = geoai.BuildingFootprintExtractor(
        model_path=config.model_path,
        repo_id=config.model_repo_id,
        device=device,
    )
    logger.info("Running BuildingFootprintExtractor.process_raster on %s ...", raster_path)
    gdf = extractor.process_raster(
        str(raster_path),
        batch_size=config.batch_size,
        filter_edges=config.filter_edges,
        edge_buffer=config.edge_buffer,
        band_indexes=config.band_indexes,
        confidence_threshold=config.confidence_threshold,
        overlap=config.overlap,
        nms_iou_threshold=config.nms_iou_threshold,
        min_object_area=config.min_object_area,
        max_object_area=config.max_object_area,
        mask_threshold=config.mask_threshold,
        simplify_tolerance=config.simplify_tolerance,
    )
    # Stash the extractor so extractor-based regularization can reuse it.
    config_extractor_cache[id(config)] = extractor
    return gdf


def _detect_timm(
    geoai, config: DetectConfig, raster_path: Path, device: str
) -> gpd.GeoDataFrame:
    """timm semantic segmentation -> mask raster -> orthogonalized vector."""
    out_dir = config.out_footprints.parent
    mask_path = out_dir / (config.out_footprints.stem + "_mask.tif")

    logger.info(
        "Running timm_segmentation_from_hub (repo_id=%s) on %s ...",
        config.model_repo_id,
        raster_path,
    )
    geoai.timm_segmentation_from_hub(
        input_path=str(raster_path),
        output_path=str(mask_path),
        repo_id=config.model_repo_id,
        window_size=config.window_size,
        overlap=config.seg_overlap,
        batch_size=config.batch_size,
        device=device,
        probability_threshold=config.probability_threshold,
    )

    logger.info("Orthogonalizing segmentation mask ...")
    gdf = geoai.orthogonalize(
        input_path=str(mask_path),
        output_path=None,
        epsilon=config.orthogonalize_epsilon,
        min_area=int(config.min_object_area),
    )

    if not config.keep_mask:
        mask_path.unlink(missing_ok=True)
    else:
        logger.info("Kept intermediate mask raster: %s", mask_path)

    if gdf is None:
        return gpd.GeoDataFrame(geometry=[], crs=config.crs)
    return gdf


config_extractor_cache: dict[int, object] = {}


def _regularize(geoai, gdf: gpd.GeoDataFrame, config: DetectConfig) -> gpd.GeoDataFrame:
    if gdf.empty or config.regularize == "none":
        return gdf

    logger.info("Regularizing footprints with method=%s", config.regularize)

    if config.regularize == "extractor":
        extractor = config_extractor_cache.get(id(config))
        if extractor is None:
            extractor = geoai.BuildingFootprintExtractor(
                model_path=config.model_path,
                repo_id=config.model_repo_id,
                device=config.device,
            )
        return extractor.regularize_buildings(
            gdf=gdf,
            min_area=int(config.min_area),
            angle_threshold=int(config.angle_threshold),
            orthogonality_threshold=config.orthogonality_threshold,
            rectangularity_threshold=config.rectangularity_threshold,
        )

    if config.regularize == "regularization":
        return geoai.regularization(
            building_polygons=gdf,
            angle_tolerance=config.angle_tolerance,
            simplify_tolerance=config.reg_simplify_tolerance,
            orthogonalize=True,
            preserve_topology=True,
        )

    if config.regularize == "hybrid":
        return geoai.hybrid_regularization(gdf)

    if config.regularize == "adaptive":
        return geoai.adaptive_regularization(
            building_polygons=gdf,
            simplify_tolerance=config.reg_simplify_tolerance,
            area_threshold=config.area_threshold,
            preserve_shape=True,
        )

    return gdf


def _finalize(geoai, gdf: gpd.GeoDataFrame, config: DetectConfig) -> gpd.GeoDataFrame:
    """Add area attributes and drop sub-threshold artifacts."""
    if gdf.empty:
        return gdf

    gdf = geoai.add_geometric_properties(gdf, area_unit="m2", length_unit="m")
    if "area_m2" in gdf.columns and config.min_area > 0:
        before = len(gdf)
        gdf = gdf[gdf["area_m2"] >= config.min_area].copy()
        if len(gdf) != before:
            logger.info(
                "Filtered %d footprint(s) below min_area=%.1f m^2",
                before - len(gdf),
                config.min_area,
            )
    return gdf.reset_index(drop=True)


def _aoi_bounds(aoi: str, raster_crs) -> tuple[float, float, float, float]:
    """Resolve an AOI (bbox string or vector path) to bounds in the raster CRS."""
    if "," in aoi:
        vals = [float(v.strip()) for v in aoi.split(",")]
        if len(vals) != 4:
            raise ConfigError("--aoi bbox format must be: minx,miny,maxx,maxy")
        return vals[0], vals[1], vals[2], vals[3]

    aoi_gdf = gpd.read_file(aoi, engine="pyogrio")
    if aoi_gdf.empty:
        raise ConfigError(f"AOI vector is empty: {aoi}")
    if raster_crs is not None and aoi_gdf.crs is not None and aoi_gdf.crs != raster_crs:
        aoi_gdf = aoi_gdf.to_crs(raster_crs)
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _prepare_raster(config: DetectConfig, crop_aoi: str | None) -> tuple[Path, Path]:
    """Window-read the orthophoto (optionally cropping to an AOI and/or resampling
    to ``target_resolution``) and write the result to a temporary GeoTIFF.

    Returns (prepared_path, prepared_path); the second element is the temp file to
    clean up. Cropping restricts inference to the AOI window; resampling brings
    high-resolution imagery to a ground sample distance the model expects.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window, from_bounds

    out_dir = config.out_footprints.parent
    prepared = out_dir / (config.out_footprints.stem + "_prep.tif")

    with rasterio.open(config.orthophoto) as src:
        if crop_aoi:
            minx, miny, maxx, maxy = _aoi_bounds(crop_aoi, src.crs)
            rminx, rminy, rmaxx, rmaxy = src.bounds
            minx, miny = max(minx, rminx), max(miny, rminy)
            maxx, maxy = min(maxx, rmaxx), min(maxy, rmaxy)
            if minx >= maxx or miny >= maxy:
                raise ConfigError(
                    "AOI does not overlap the orthophoto extent: "
                    f"aoi=({crop_aoi}) raster_bounds=({rminx},{rminy},{rmaxx},{rmaxy})"
                )
            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            window = window.round_offsets().round_lengths()
        else:
            window = Window(0, 0, src.width, src.height)

        base_transform = src.window_transform(window)
        out_w, out_h = int(window.width), int(window.height)

        if config.target_resolution is not None:
            src_res = abs(src.transform.a)
            scale = src_res / config.target_resolution
            out_w = max(1, int(round(window.width * scale)))
            out_h = max(1, int(round(window.height * scale)))
            logger.info(
                "Resampling from %.4f to %.4f units/px (scale %.3f)",
                src_res,
                config.target_resolution,
                scale,
            )

        data = src.read(
            window=window,
            out_shape=(src.count, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        # Scale the window transform to the actual output pixel grid.
        transform = base_transform * base_transform.scale(
            window.width / out_w, window.height / out_h
        )

        profile = src.profile.copy()
        # Drop source blocking hints: a non-tiled output with BLOCKXSIZE/BLOCKYSIZE
        # set raises a GDAL warning. Let GDAL pick defaults.
        for key in ("blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
        profile.update(height=out_h, width=out_w, transform=transform, driver="GTiff")

        logger.info(
            "Prepared raster %d x %d px (from %d x %d)",
            out_w,
            out_h,
            src.width,
            src.height,
        )
        with rasterio.open(prepared, "w", **profile) as dst:
            dst.write(data)

    return prepared, prepared


def _clip_to_aoi(gdf: gpd.GeoDataFrame, aoi: str) -> gpd.GeoDataFrame:
    if "," in aoi:
        vals = [float(v.strip()) for v in aoi.split(",")]
        if len(vals) != 4:
            raise ConfigError("--aoi bbox format must be: minx,miny,maxx,maxy")
        minx, miny, maxx, maxy = vals
        return gdf.cx[minx:maxx, miny:maxy]

    aoi_gdf = gpd.read_file(aoi, engine="pyogrio")
    if aoi_gdf.empty:
        return gdf.iloc[0:0]
    aoi_geom = aoi_gdf.union_all()
    return gdf[gdf.intersects(aoi_geom)]


def _write_footprints(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        # Preserve an empty but valid shapefile so downstream steps don't choke.
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=gdf.crs)
        empty.to_file(out_path, driver="ESRI Shapefile", engine="pyogrio")
        return
    gdf.to_file(out_path, driver="ESRI Shapefile", engine="pyogrio")
