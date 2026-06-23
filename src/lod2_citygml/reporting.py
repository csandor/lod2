from __future__ import annotations

import json
from pathlib import Path

from lod2_citygml.models import BuildingRecord, PipelineStats


def summarize(records: list[BuildingRecord]) -> PipelineStats:
    stats = PipelineStats()
    stats.buildings_total = len(records)
    stats.buildings_processed = len(records)

    if records:
        stats.mean_height = sum(r.height for r in records) / len(records)
    stats.buildings_fallback_roof = sum(1 for r in records if r.roof_kind == "flat")

    return stats


def write_report(stats: PipelineStats, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "buildings_total": stats.buildings_total,
                "buildings_processed": stats.buildings_processed,
                "buildings_fallback_roof": stats.buildings_fallback_roof,
                "buildings_invalid": stats.buildings_invalid,
                "mean_height": round(stats.mean_height, 3),
            },
            handle,
            indent=2,
        )
