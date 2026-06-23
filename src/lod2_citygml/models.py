from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString, Polygon


@dataclass(slots=True)
class BuildingRecord:
    building_id: str
    footprint: Polygon
    base_z: float
    roof_z: float
    height: float
    roof_confidence: float
    roof_kind: str
    eave_z: float | None = None
    roof_lines: list[LineString] | None = None
    roof_triangles: list[list[tuple[float, float, float]]] | None = None


@dataclass(slots=True)
class PipelineStats:
    buildings_total: int = 0
    buildings_processed: int = 0
    buildings_fallback_roof: int = 0
    buildings_invalid: int = 0
    mean_height: float = 0.0
