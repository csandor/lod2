from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd


@dataclass(slots=True)
class DetectStats:
    footprints_total: int = 0
    total_area_m2: float = 0.0
    mean_area_m2: float = 0.0
    min_area_m2: float = 0.0
    max_area_m2: float = 0.0
    crs: str | None = None


def summarize(gdf: gpd.GeoDataFrame) -> DetectStats:
    stats = DetectStats()
    stats.footprints_total = len(gdf)
    stats.crs = str(gdf.crs) if gdf.crs is not None else None

    if gdf.empty:
        return stats

    if "area_m2" in gdf.columns:
        areas = gdf["area_m2"].astype(float)
    else:
        areas = gdf.geometry.area

    stats.total_area_m2 = float(areas.sum())
    stats.mean_area_m2 = float(areas.mean())
    stats.min_area_m2 = float(areas.min())
    stats.max_area_m2 = float(areas.max())
    return stats


def write_report(stats: DetectStats, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "footprints_total": stats.footprints_total,
                "total_area_m2": round(stats.total_area_m2, 3),
                "mean_area_m2": round(stats.mean_area_m2, 3),
                "min_area_m2": round(stats.min_area_m2, 3),
                "max_area_m2": round(stats.max_area_m2, 3),
                "crs": stats.crs,
            },
            handle,
            indent=2,
        )
