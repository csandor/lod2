from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd


@dataclass(slots=True)
class AdjustStats:
    footprints_input: int = 0
    footprints_kept: int = 0
    footprints_dropped: int = 0
    total_area_m2: float = 0.0
    mean_height_m: float = 0.0
    min_height_m: float = 0.0
    max_height_m: float = 0.0
    crs: str | None = None


def summarize(gdf: gpd.GeoDataFrame, n_input: int, height_column: str) -> AdjustStats:
    stats = AdjustStats()
    stats.footprints_input = n_input
    stats.footprints_kept = len(gdf)
    stats.footprints_dropped = n_input - len(gdf)
    stats.crs = str(gdf.crs) if gdf.crs is not None else None

    if gdf.empty:
        return stats

    stats.total_area_m2 = float(gdf.geometry.area.sum())

    if height_column in gdf.columns:
        heights = gdf[height_column].astype(float)
        stats.mean_height_m = float(heights.mean())
        stats.min_height_m = float(heights.min())
        stats.max_height_m = float(heights.max())
    return stats


def write_report(stats: AdjustStats, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "footprints_input": stats.footprints_input,
                "footprints_kept": stats.footprints_kept,
                "footprints_dropped": stats.footprints_dropped,
                "total_area_m2": round(stats.total_area_m2, 3),
                "mean_height_m": round(stats.mean_height_m, 3),
                "min_height_m": round(stats.min_height_m, 3),
                "max_height_m": round(stats.max_height_m, 3),
                "crs": stats.crs,
            },
            handle,
            indent=2,
        )
