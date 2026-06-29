from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import geopandas as gpd

from lod2_citygml.citygml_writer import write_citygml
from lod2_citygml.cityjson_writer import write_cityjson
from lod2_citygml.config import RunConfig
from lod2_citygml.elevation import estimate_buildings
from lod2_citygml.geometry import parametric_roof_faces
from lod2_citygml.io_data import load_inputs
from lod2_citygml.reporting import summarize, write_report
from lod2_citygml.roof import infer_roof_kind
from lod2_citygml.qa import validate_inputs


def run_pipeline(config: RunConfig) -> None:
    footprints, ortho, dsm, dtm = load_inputs(config)

    try:
        validate_inputs(config, footprints, ortho, dsm, dtm)

        if config.crs:
            footprints = footprints.to_crs(config.crs)

        if config.aoi:
            footprints = _clip_to_aoi(footprints, config.aoi)

        records = estimate_buildings(footprints, dsm, dtm)
        records = infer_roof_kind(records, dsm, config.min_roof_confidence)

        records = [
            replace(
                rec,
                eave_z=rec.base_z + (rec.roof_z - rec.base_z) * 0.3,
                roof_faces=parametric_roof_faces(
                    rec.footprint,
                    rec.long_axis,
                    rec.short_axis,
                    rec.roof_kind,
                    eave_z=rec.base_z + (rec.roof_z - rec.base_z) * 0.3,
                    ridge_z=rec.roof_z,
                ),
            )
            if rec.roof_kind != "flat"
            else rec
            for rec in records
        ]

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
