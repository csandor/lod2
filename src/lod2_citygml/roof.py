from __future__ import annotations

import warnings

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy as transform_xy
from shapely.geometry import mapping
from shapely.geometry import Polygon

from lod2_citygml.models import BuildingRecord


def _dominant_axis(footprint: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Return (long_axis_unit, short_axis_unit) from the minimum bounding rectangle."""
    mbr = footprint.minimum_rotated_rectangle
    coords = np.array(mbr.exterior.coords[:-1])
    edges = [coords[(i + 1) % 4] - coords[i] for i in range(4)]
    lengths = [np.linalg.norm(e) for e in edges]
    long_idx = int(np.argmax(lengths))
    long = edges[long_idx] / lengths[long_idx]
    short = edges[(long_idx + 1) % 4] / lengths[(long_idx + 1) % 4]
    return long, short


def _dsm_points(
    dsm: rasterio.DatasetReader, geom: Polygon
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (world_x, world_y, z) arrays for all valid DSM pixels inside geom."""
    data, transform = mask(dsm, [mapping(geom)], crop=True, filled=False)
    arr = np.asarray(data[0], dtype=np.float32)
    msk = data.mask[0] if hasattr(data, "mask") else np.zeros_like(arr, dtype=bool)
    valid = ~msk & np.isfinite(arr)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return np.array([]), np.array([]), np.array([])
    wx, wy = transform_xy(transform, rows, cols)
    return np.array(wx), np.array(wy), arr[rows, cols]


def _axis_profile_variance(
    wx: np.ndarray, wy: np.ndarray, z: np.ndarray, axis: np.ndarray, n_bins: int = 10
) -> float:
    """Bin points by projection onto axis, return variance of per-bin median z."""
    proj = wx * axis[0] + wy * axis[1]
    if proj.size < n_bins:
        return 0.0
    edges = np.linspace(proj.min(), proj.max(), n_bins + 1)
    medians = []
    for j in range(n_bins):
        sel = (proj >= edges[j]) & (proj < edges[j + 1])
        if sel.sum() >= 2:
            medians.append(float(np.median(z[sel])))
    return float(np.var(medians)) if len(medians) >= 3 else 0.0


def _classify(
    wx: np.ndarray,
    wy: np.ndarray,
    z: np.ndarray,
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    height: float,
    min_confidence: float,
) -> tuple[str, float]:
    if wx.size == 0 or height < 1.5:
        return "flat", 0.0

    spread = float(np.percentile(z, 95) - np.percentile(z, 5))
    if spread < 1.0:
        return "flat", 0.0

    # Variance of median-z when sliced along each axis.
    # A gable roof has high variance along the SHORT axis (profile rises to ridge)
    # and low variance along the LONG axis (ridge is flat along its length).
    var_short = _axis_profile_variance(wx, wy, z, short_axis)  # gable signal
    var_long = _axis_profile_variance(wx, wy, z, long_axis)    # hip signal

    total = var_short + var_long
    if total < 1e-6:
        return "flat", 0.0

    # Confidence: how much of the height spread is explained by a systematic profile
    confidence = min(1.0, max(var_short, var_long) / (spread + 1e-6))

    if confidence < min_confidence:
        return "flat", confidence

    # Gable: variance concentrated on short-axis profile, long-axis profile is flat
    # Hip: both axes show variance
    ratio = var_short / (var_long + 1e-6)
    if ratio > 1.5:
        return "gable", confidence
    return "hip", confidence


def infer_roof_kind(
    records: list[BuildingRecord],
    dsm: rasterio.DatasetReader,
    min_confidence: float,
) -> list[BuildingRecord]:
    updated: list[BuildingRecord] = []
    for rec in records:
        long_axis, short_axis = _dominant_axis(rec.footprint)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice")
            wx, wy, z = _dsm_points(dsm, rec.footprint)
        kind, conf = _classify(wx, wy, z, long_axis, short_axis, rec.height, min_confidence)
        updated.append(
            BuildingRecord(
                building_id=rec.building_id,
                footprint=rec.footprint,
                base_z=rec.base_z,
                roof_z=rec.roof_z,
                height=rec.height,
                roof_confidence=conf,
                roof_kind=kind,
                long_axis=long_axis,
                short_axis=short_axis,
            )
        )
    return updated
