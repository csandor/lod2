# LOD2 Building Reconstruction Plan (Pilot, High-Precision, CLI Tool)

## 0. Deliverable shape: command-line tool
- Implement as a Python CLI tool (single entry point) for reproducible batch processing.
- Provide clear, documented CLI switches with defaults, required/optional flags, and examples.
- Include `--help` output, usage examples, and error messages for invalid combinations.

## 1. Project and environment management (uv)
- Use `uv` as the standard tool for Python environment and dependency management.
- Project setup and execution model:
  - initialize/manage project metadata with `uv` (`pyproject.toml`)
  - create/sync environment with `uv sync`
  - add/remove dependencies with `uv add` / `uv remove`
  - run CLI with `uv run ...`
- Commit dependency lock state for reproducibility (`uv.lock`).
- Document minimum Python version and exact setup commands in README.

## 2. Config-first pipeline (pilot oriented)
- Build a config-driven script with explicit options:
  - `citygml_version: 2.0 | 3.0`
  - processing CRS, file paths, output path
  - quality mode fixed to `high_precision`
  - pilot AOI boundary (optional clip polygon)
  - max buildings / max km² per batch thresholds
- Allow parameters from CLI switches and optional config file (CLI overrides config).
- Before processing, estimate workload and warn/stop if AOI exceeds thresholds for a single run.

## 3. Input loading + strict QA
- Read building footprints (`SHP`) and rasters (orthophoto/DSM/DTM) from either `TIFF` or `VRT`.
- Verify same projected Cartesian CRS and overlap.
- Check nodata, pixel resolution consistency (5 cm RGB for orthophoto, 40 cm DSM/DTM), and footprint validity.
- Validate raster characteristics:
  - orthophoto: 3 bands,
  - DSM/DTM: single band,
  - VRT-backed rasters must resolve correctly to source data.
- If AOI is large, auto-suggest chunked/tiled processing.

## 4. High-precision elevation normalization
- Compute `nDSM = DSM - DTM`.
- Per building, derive:
  - robust base elevation (DTM, footprint edge-aware sampling),
  - robust roof elevation candidates (DSM/nDSM percentiles + outlier filtering),
  - initial building height.
- Keep uncertainty metrics per building.

## 5. LOD1 baseline generation (mandatory first stage)
- Extrude footprint to flat roof using detected height/base.
- Store LOD1 geometry as intermediate output for QA and fallback.
- Validate geometry and flag problematic footprints early.

## 6. Fine roof-structure reconstruction (LOD2-focused)
- Inside each footprint:
  - detect roof edges/ridges from nDSM gradients + RGB edges,
  - extract line network, snap/merge/regularize,
  - segment roof into multiple planar patches (RANSAC + adjacency constraints),
  - reconstruct detailed roof facets to mimic true structure (not just simple gable/hip).
- Use confidence scoring; if reconstruction is weak, fall back to best plausible simplified roof while preserving precision where reliable.

## 7. LOD2 solid assembly
- Combine roof facets + walls from footprint to create building solids.
- Prioritize geometric precision over speed:
  - tighter tolerances,
  - iterative cleanup,
  - intersection/gap checks.
- No strict requirement to force full semantic completeness, but include semantics when confidently available.

## 8. CityGML output (2.0 or 3.0 selectable)
- Export path supports both target versions via config option.
- Prefer robust workflow: construct validated intermediate city model, then emit CityGML in selected version.
- Include core attributes (id, measured heights, reconstruction confidence, processing flags).

## 9. Validation + reporting
- Run geometry validity checks and summary stats:
  - success/fallback rates,
  - invalid/empty buildings,
  - average roof-plane count,
  - uncertainty distribution.
- Output a pilot report and explicit recommendation for scaling (tile size, batch size, expected runtime).

## 10. CLI switch specification (to be documented)
- Required inputs:
  - `--footprints PATH` (SHP)
  - `--orthophoto PATH` (RGB raster: TIFF or VRT)
  - `--dsm PATH` (single-band raster: TIFF or VRT)
  - `--dtm PATH` (single-band raster: TIFF or VRT)
  - `--out-citygml PATH`
- Core options:
  - `--citygml-version {2.0,3.0}`
  - `--crs EPSG_CODE_OR_WKT`
  - `--aoi PATH_OR_BBOX` (optional pilot clip)
  - `--max-buildings N`
  - `--max-area-km2 FLOAT`
  - `--on-large-area {warn,abort,tile}`
  - `--quality high_precision` (default)
- Precision/reconstruction options:
  - `--roof-detail fine` (default)
  - `--min-roof-confidence FLOAT`
  - `--lod1-intermediate PATH` (optional save)
  - `--fallback-roof-mode {flat,simple_planes}`
- Runtime/output options:
  - `--tile-size M` (for optional chunking)
  - `--workers N` (default conservative)
  - `--log-level {DEBUG,INFO,WARNING,ERROR}`
  - `--report PATH` (QA summary)
- CLI UX requirements:
  - clear `--help` text for every switch,
  - validation of required combinations,
  - actionable errors and warnings,
  - explicit support note for TIFF/VRT rasters,
  - at least 3 end-to-end usage examples in docs.

## 11. Dependency plan (managed with uv)
- Core geospatial stack:
  - `geopandas`, `pyogrio`, `shapely`, `rasterio`, `pyproj`
- Numeric and reconstruction stack:
  - `numpy`, `scipy`, `scikit-image`, `opencv-python`, `scikit-learn`
- City model and export stack:
  - `cjio`, `lxml` (if direct XML handling needed)
- CLI/dev stack:
  - `typer` (or `argparse` if keeping stdlib-only), `rich` (optional), `pydantic` (optional config validation)
- QA/testing (optional but recommended):
  - `pytest`
- Add all packages via `uv add ...` and keep lockfile updated.

## Changes Integrated from User Decisions
- Added explicit CityGML `2.0/3.0` switch.
- Elevated roof modeling to fine-structure mimicry (multi-facet priority).
- Optimized for small pilot areas first.
- Added automatic "area too large" warning/gate before processing.
- Set strategy to high-precision, slower by default.
- Relaxed semantics to optional/non-strict.
- Defined deliverable as a documented CLI tool with explicit switches.
- Added support for raster inputs in both `TIFF` and `VRT` formats.
- Standardized Python environment and dependencies around `uv`.
