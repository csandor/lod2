# CLAUDE.md

Guidance for working in this repository.

## What this is

`lod2-citygml` — a Python CLI that reconstructs LOD2-like building models from
2D footprints + orthophoto + DSM + DTM, and provides supporting commands for
detecting and post-processing footprints. Built on geopandas / rasterio /
shapely / typer, managed with `uv`.

## Environment & commands

This project is managed with `uv`. There is no `pip install` flow.

```bash
uv sync                                   # install (incl. heavy geoai/torch deps)
uv run lod2-citygml --help                # run the CLI
uv run --extra dev ruff check src/        # lint (ruff config in pyproject.toml)
uv run --extra dev pytest                 # tests
```

Ruff: line length 100, rules `E`, `F`, `I`. `ruff` is only available via the
`dev` extra, so always invoke it as `uv run --extra dev ruff ...`.

## CLI subcommands

Entry point: `src/lod2_citygml/cli.py` (`app` = the Typer app). Three commands:

- `build` — full LOD2 reconstruction → CityGML/CityJSON. Logic in `pipeline.py`.
- `detect-footprints` — deep-learning building detection from an orthophoto
  (uses `geoai-py`). Logic in `footprints_detect.py`.
- `footprint-adjust` — filter footprints by 2D area, estimate height from a
  DSM-DTM **difference** raster (median of inward-buffered samples), write a
  height column. Logic in `footprint_adjust.py`.

## Architecture conventions (follow these for new commands)

Each command is a triplet of modules, kept separate from the others:

- `<feature>_config.py` — a `@dataclass(slots=True)` config + `load_config_file`
  + `build_<feature>_config(cli_values, file_values)` + `validate_*`.
- `<feature>.py` — the algorithm; a single `run_<feature>(config)` entry point.
- `<feature>_reporting.py` — a stats dataclass + `summarize()` + `write_report()`
  that writes an optional JSON report.

`config.py` holds the shared `ConfigError` (subclass of `ValueError`) and the
`build` command's config. Reuse `ConfigError` everywhere; the CLI catches it and
re-raises as `typer.BadParameter`.

### The config-merge pattern (important — keep it uniform)

Every command merges file + CLI values the same way:

```python
merged = {**file_values, **{k: v for k, v in cli_values.items() if v is not None}}
```

So **CLI options override the YAML config**, and a `None` CLI value falls back to
the config / dataclass default. CLI option defaults are therefore mostly `None`
(not the real default) so "unset" is distinguishable. The real defaults live in
the dataclass and are applied via `merged.get(key, default)` in the builder.
Required keys are checked explicitly and raise `ConfigError` listing what's
missing.

Each command has a matching example config at the repo root
(`*.config.example.yaml`) and a working config (`*.config.yaml`).

### Lazy heavy imports

`geoai` / `torch` are imported **inside** the command body (after config
validation), not at module top level — so config errors surface fast without
paying the multi-second import cost. Keep this for any torch/geoai-touching code.

## Vector I/O conventions

- Always read/write vectors with `engine="pyogrio"`.
- Output format is chosen from the **file extension** via the `VECTOR_DRIVERS`
  map in `footprint_adjust.py` (`.shp`→ESRI Shapefile, `.gpkg`→GPKG,
  `.geojson`/`.json`→GeoJSON). An unknown extension raises `ConfigError`.
- **Shapefile field names cap at 10 chars.** The `height_column` length check is
  enforced *only* for `.shp` output; GPKG/GeoJSON allow longer names.
- Multi-layer GPKG: `--input-layer` selects the read layer (default: first);
  `--output-layer` names the written layer (default: file stem; only valid for
  `.gpkg`, errors otherwise).
- Empty results still write a valid (empty) dataset so downstream steps don't
  choke.
- Footprints are reprojected to the relevant raster CRS before masking
  (`gpd.read_file` auto-detects CRS; masking and metric areas need a common CRS).

## Raster sampling

Height/elevation sampling masks a raster to a polygon with
`rasterio.mask.mask(..., crop=True, filled=False)` and keeps finite, unmasked
values (see `_masked_values` in both `elevation.py` and `footprint_adjust.py`).
`footprint-adjust` takes the **median** of the DSM-DTM difference inside an
inward-buffered footprint; a footprint too thin to survive the buffer falls back
to its full geometry.

## Gotchas

- **`footprint-adjust` in-place output**: a config may set `out_footprints` to
  the **same GPKG file** as `footprints`, writing a different `output_layer`
  (e.g. input layer `buildings`, output layer `buildings_filtered`). This is
  intentional and supported, but only safe because input and output layers
  differ — never point `--output-layer` at the layer being read, or you risk
  truncating the source mid-run.
- All `build` inputs must share the same **projected** CRS; DSM/DTM are
  single-band, orthophoto 3- or 4-band.
- Supported raster suffixes: `.tif`, `.tiff`, `.vrt`, `.ecw` (ECW needs a
  GDAL/rasterio build with the ECW driver).
- `sample data/` holds EOV (EPSG:23700) test data, including a 40 cm relative
  height model usable as the DSM-DTM difference for `footprint-adjust`.
