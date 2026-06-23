# lod2-citygml

Python CLI tool to reconstruct building models from:
- 2D building footprints (`.shp`)
- RGB orthophoto (`.tif/.tiff`, `.vrt`, or `.ecw`, 3-band)
- DSM (`.tif/.tiff`, `.vrt`, or `.ecw`, 1-band)
- DTM (`.tif/.tiff`, `.vrt`, or `.ecw`, 1-band)

Output:
- CityGML and/or CityJSON with LOD2-like building solids and roof-type classification.

## Environment management (uv)

This project is managed with `uv`.

```bash
uv sync
```

Run CLI:

```bash
uv run lod2-citygml --help
```

The CLI exposes three subcommands:

- `build` — reconstruct LOD2 CityGML/CityJSON (see below).
- `detect-footprints` — detect building footprints in an orthophoto and export SHP (see [Building-footprint detection](#building-footprint-detection)).
- `footprint-adjust` — filter footprints by area and estimate building height from a DSM-DTM difference raster (see [Footprint adjustment](#footprint-adjustment)).

## Main command

```bash
uv run lod2-citygml build \
  --footprints data/buildings.shp \
  --orthophoto data/ortho.vrt \
  --dsm data/dsm.tif \
  --dtm data/dtm.tif \
  --out-citygml out/city_lod2.gml \
  --out-cityjson out/city_lod2.city.json
```

## Examples

1. Basic pilot run:

```bash
uv run lod2-citygml build \
  --footprints data/buildings.shp \
  --orthophoto data/ortho.tif \
  --dsm data/dsm.vrt \
  --dtm data/dtm.vrt \
  --out-citygml out/pilot.gml \
  --citygml-version 2.0
```

2. CityGML 3.0 requested with AOI bbox, intermediate LOD1, and report:

```bash
uv run lod2-citygml build \
  --footprints data/buildings.shp \
  --orthophoto data/ortho.vrt \
  --dsm data/dsm.vrt \
  --dtm data/dtm.vrt \
  --out-citygml out/pilot_3.gml \
  --citygml-version 3.0 \
  --aoi 648000,237000,649500,238500 \
  --lod1-intermediate out/lod1.gpkg \
  --report out/report.json
```

3. Large-area safety in abort mode:

```bash
uv run lod2-citygml build \
  --footprints data/buildings.shp \
  --orthophoto data/ortho.vrt \
  --dsm data/dsm.tif \
  --dtm data/dtm.tif \
  --out-citygml out/run.gml \
  --max-buildings 50000 \
  --max-area-km2 25 \
  --on-large-area abort
```

## Config file support

You can pass a YAML config file. CLI options override config values.

```bash
uv run lod2-citygml build --config config.example.yaml --out-citygml out/override.gml
```

## Building-footprint detection

`detect-footprints` runs deep-learning building detection on an orthophoto
(using [`geoai-py`](https://opengeoai.org)) and writes footprints as a shapefile.
It shares this project's `uv` environment.

```bash
uv run lod2-citygml detect-footprints \
  --orthophoto data/ortho.tif \
  --out-footprints out/building_footprints.shp \
  --backend mask_rcnn \
  --confidence-threshold 0.5 \
  --regularize hybrid \
  --report out/footprints_report.json
```

Backends:

- `timm` — `geoai.timm_segmentation_from_hub` semantic segmentation to a mask
  raster, then `orthogonalize` to polygons. Requires `--model-repo-id`, e.g.
  `giswqs/whu-building-unetplusplus-efficientnet-b4`. **Best for high-resolution
  aerial rooftop imagery** (the WHU model aligns well on such data).
- `mask_rcnn` — `geoai.BuildingFootprintExtractor`; instance segmentation
  straight to vector footprints. Uses the USA/NAIP model, which is poorly matched
  to non-US, very-high-resolution imagery and may misdetect.

Resolution matters: the pretrained models expect ~0.3–1 m/px. Use
`--target-resolution` (e.g. `0.3`) to downsample finer input (e.g. 5 cm/px) — at
native 5 cm the models otherwise detect tiny fragments. Detection runs on GPU when
a CUDA-capable torch/driver is present, else CPU (auto-detected; force with
`--device cpu`).

Regularization (`--regularize`): `none | extractor | regularization | hybrid | adaptive`
(default `hybrid`). `extractor` requires the `mask_rcnn` backend.

Process only an area of interest with `--aoi` (a `minx,miny,maxx,maxy` bbox in the
raster CRS, or a path to a vector AOI). `--aoi-mode crop` (default) crops the
orthophoto to the AOI **before** detection, so inference runs only on that window —
much faster than processing a full mosaic. `--aoi-mode filter` instead detects over
the whole raster and clips the resulting footprints to the AOI afterwards.

```bash
uv run lod2-citygml detect-footprints \
  --orthophoto data/ortho.tif \
  --out-footprints out/building_footprints.shp \
  --aoi 633048.14,236919.59,633180.06,237051.47 \
  --aoi-mode crop
```

Using a config file (CLI options override config values):

```bash
uv run lod2-citygml detect-footprints --config footprints.config.yaml
```

The detected footprints SHP can be fed straight into `build --footprints`.

References:
[building_detection_whu](https://opengeoai.org/examples/building_detection_whu/),
[building_footprints_usa](https://opengeoai.org/examples/building_footprints_usa/),
[building_regularization](https://opengeoai.org/examples/building_regularization/).

## Footprint adjustment

`footprint-adjust` cleans a footprint shapefile and attaches a building height,
using a **DSM-DTM difference raster** (a relative-height / normalized-DSM model).
It is a useful post-process for detected footprints before `build`.

The algorithm:

1. Drops footprints whose 2D area is below `--min-area` (m²).
2. For each remaining footprint, applies an inward (negative) `--buffer` (m),
   masks the difference raster to that eroded polygon, and takes the **median**
   value as the building height. The inward buffer keeps overhanging roof edges
   and mixed boundary pixels out of the estimate. (Footprints too narrow to
   survive the buffer fall back to their full geometry so they still get a
   height.)
3. Writes the height into a configurable column (`--height-column`, default
   `height`; max 10 chars due to the ESRI Shapefile field-name limit).
4. Drops footprints whose estimated height is below `--min-height` (m). Footprints
   with no usable raster coverage (no finite pixels) are dropped too.
5. Writes the surviving footprints to `--out-footprints` and an optional JSON
   report (`--report`).

```bash
uv run lod2-citygml footprint-adjust \
  --footprints out/building_footprints.shp \
  --dsm-dtm-diff data/dsm_dtm_diff.tif \
  --out-footprints out/footprints_adjusted.shp \
  --min-area 30 \
  --buffer 1.0 \
  --min-height 2.0 \
  --height-column height \
  --report out/footprint_adjust_report.json
```

Input and output may be `.shp`, `.gpkg`, or `.geojson` — the format is chosen
from each path's extension. Note that ESRI Shapefile caps field names at 10
characters, so a longer `--height-column` (e.g. `building_height`) requires a
`.gpkg` or `.geojson` output.

For multi-layer GeoPackages, select the input layer with `--input-layer` (defaults
to the first layer) and name the output layer with `--output-layer` (defaults to
the output file stem; only valid for `.gpkg` output):

```bash
uv run lod2-citygml footprint-adjust \
  --footprints buildings.gpkg --input-layer footprints \
  --dsm-dtm-diff diff.tif \
  --out-footprints out/adjusted.gpkg --output-layer footprints_adjusted
```

Footprints are automatically reprojected to the difference raster's CRS for
masking. Use `--crs` to set the output CRS (defaults to the input CRS).

Using a config file (CLI options override config values):

```bash
uv run lod2-citygml footprint-adjust --config footprint_adjust.config.yaml
```

## Important notes

- All inputs must share the same **projected CRS**.
- Orthophoto must be 3-band (RGB) or 4-band (RGBA); if 4-band, the alpha channel is ignored. DSM/DTM must be single-band.
- ECW inputs require GDAL/Rasterio built with ECW driver support.
- Provide at least one output path: `--out-citygml` and/or `--out-cityjson`.
- `--quality high_precision` and `--roof-detail fine` are currently the supported modes.
- `--citygml-version 3.0` is accepted and tagged in metadata, while the current serializer uses a baseline interoperable structure for output geometry.
