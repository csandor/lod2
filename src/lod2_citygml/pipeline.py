from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping

from lod2_citygml.citygml_writer import write_citygml
from lod2_citygml.cityjson_writer import write_cityjson
from lod2_citygml.config import RunConfig
from lod2_citygml.elevation import estimate_buildings
from lod2_citygml.geometry import roof_from_dsm
from lod2_citygml.io_data import load_inputs
from lod2_citygml.reporting import summarize, write_report
from lod2_citygml.roof import infer_roof_kind
from lod2_citygml.qa import validate_inputs


def run_pipeline(config: RunConfig) -> None:
    footprints, ortho, dsm, dtm = load_inputs(config)

    try:
        validate_inputs(config, footprints, ortho, dsm, dtm)

        # Optional CRS override for the vector layer.
        if config.crs:
            footprints = footprints.to_crs(config.crs)

        # AOI support: path to vector or "minx,miny,maxx,maxy".
        if config.aoi:
            footprints = _clip_to_aoi(footprints, config.aoi)

        records = estimate_buildings(footprints, dsm, dtm)
        records = infer_roof_kind(records, dsm, config.min_roof_confidence)

        # Export DSM/DTM cutouts for debugging
        _export_building_rasters(records, dsm, dtm, config.out_citygml.parent if config.out_citygml else Path("out"))

        # Add triangulated roof surface for pitched roofs from DSM data.
        records = [
            replace(
                rec,
                eave_z=rec.base_z + (rec.roof_z - rec.base_z) * 0.3,
                roof_triangles=roof_from_dsm(
                    rec.footprint,
                    dsm,
                    rec.base_z,
                    rec.base_z + (rec.roof_z - rec.base_z) * 0.3,
                    rec.roof_z
                )
            )
            if rec.roof_kind != "flat"
            else rec
            for rec in records
        ]

        # Optional LOD1 intermediate saved as GeoPackage for inspection.
        if config.lod1_intermediate:
            _write_lod1_intermediate(records, config.lod1_intermediate, footprints.crs)

        if config.out_citygml is not None:
            write_citygml(records, config.out_citygml, config.citygml_version)
        if config.out_cityjson is not None:
            epsg = footprints.crs.to_epsg() if footprints.crs is not None else None
            write_cityjson(records, config.out_cityjson, epsg=epsg)

        stats = summarize(records)
        write_report(stats, config.report)
    finally:
        ortho.close()
        dsm.close()
        dtm.close()


def _export_building_rasters(records, dsm, dtm, output_dir: Path) -> None:
    import numpy as np

    debug_dir = output_dir / "debug_rasters"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting relative height (DSM - DTM) for {len(records)} buildings...")

    for i, rec in enumerate(records):
        geom_list = [mapping(rec.footprint)]

        dsm_cut, dsm_transform = rasterio_mask(dsm, geom_list, crop=True)
        dtm_cut, _ = rasterio_mask(dtm, geom_list, crop=True)

        if dsm_cut.size == 0:
            continue

        building_id = str(rec.building_id).replace("/", "_")

        # Compute relative height (DSM - DTM)
        relative_height = dsm_cut[0].astype(np.float32) - dtm_cut[0].astype(np.float32)

        height_path = debug_dir / f"height_b{building_id}.tif"
        with rasterio.open(
            height_path,
            "w",
            driver="GTiff",
            height=relative_height.shape[0],
            width=relative_height.shape[1],
            count=1,
            dtype=np.float32,
            transform=dsm_transform,
            crs=dsm.crs,
        ) as dst:
            dst.write(relative_height, 1)

        print(f"  {i+1}. Building {rec.building_id}: height={rec.height:.2f}m (min={np.nanmin(relative_height):.2f}, max={np.nanmax(relative_height):.2f})")


def _clip_to_aoi(footprints: gpd.GeoDataFrame, aoi: str) -> gpd.GeoDataFrame:
    if "," in aoi:
        vals = [float(v.strip()) for v in aoi.split(",")]
        if len(vals) != 4:
            raise ValueError("--aoi bbox format must be: minx,miny,maxx,maxy")
        minx, miny, maxx, maxy = vals
        return footprints.cx[minx:maxx, miny:maxy]

    aoi_gdf = gpd.read_file(aoi, engine="pyogrio")
    if aoi_gdf.empty:
        return footprints.iloc[0:0]
    aoi_geom = aoi_gdf.unary_union
    return footprints[footprints.intersects(aoi_geom)]


def _write_lod1_intermediate(records, out_path, crs) -> None:
    rows = []
    for rec in records:
        rows.append(
            {
                "building_id": rec.building_id,
                "base_z": rec.base_z,
                "roof_z": rec.roof_z,
                "height": rec.height,
                "roof_conf": rec.roof_confidence,
                "roof_kind": rec.roof_kind,
                "geometry": rec.footprint,
            }
        )
    if rows:
        lod1_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    else:
        lod1_gdf = gpd.GeoDataFrame(
            {
                "building_id": [],
                "base_z": [],
                "roof_z": [],
                "height": [],
                "roof_conf": [],
                "roof_kind": [],
                "geometry": [],
            },
            geometry="geometry",
            crs=crs,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lod1_gdf.to_file(out_path, driver="GPKG")
