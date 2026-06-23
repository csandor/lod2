from __future__ import annotations

import json
from pathlib import Path

from lod2_citygml.geometry import polygon_to_ring_xy, ring_xyz, wall_faces
from lod2_citygml.models import BuildingRecord


def write_cityjson(records: list[BuildingRecord], output: Path, epsg: int | None = None) -> None:
    vertices: list[list[float]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}

    def v_idx(coord: tuple[float, float, float]) -> int:
        key = (round(coord[0], 3), round(coord[1], 3), round(coord[2], 3))
        existing = vertex_index.get(key)
        if existing is not None:
            return existing
        idx = len(vertices)
        vertices.append([key[0], key[1], key[2]])
        vertex_index[key] = idx
        return idx

    city_objects: dict[str, dict] = {}

    for i, rec in enumerate(records):
        ring = polygon_to_ring_xy(rec.footprint)
        bottom = ring_xyz(ring, rec.base_z)
        top = ring_xyz(ring, rec.roof_z)
        wall_top = rec.eave_z if rec.eave_z is not None else rec.roof_z
        walls = wall_faces(ring, rec.base_z, wall_top)

        # CityJSON Solid boundaries:
        # [ shell [ surface [ ring [vertex indices] ] ] ]
        shell = []
        shell.append([[v_idx(c) for c in bottom]])

        if rec.roof_triangles:
            for tri in rec.roof_triangles:
                shell.append([[v_idx(c) for c in tri]])
        else:
            shell.append([[v_idx(c) for c in top]])

        for face in walls:
            shell.append([[v_idx(c) for c in face]])

        city_objects[f"b_{rec.building_id}_{i}"] = {
            "type": "Building",
            "attributes": {
                "roof_kind": rec.roof_kind,
                "roof_confidence": round(rec.roof_confidence, 4),
                "height": round(rec.height, 3),
                "base_z": round(rec.base_z, 3),
                "roof_z": round(rec.roof_z, 3),
            },
            "geometry": [
                {
                    "type": "Solid",
                    "lod": "2",
                    "boundaries": [shell],
                }
            ],
        }

    model = {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": city_objects,
        "vertices": vertices,
    }
    if epsg is not None:
        model["metadata"] = {"referenceSystem": f"https://www.opengis.net/def/crs/EPSG/0/{epsg}"}

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2)
