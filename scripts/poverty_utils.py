"""
Shared utilities for poverty–residual analysis (03a, 03b).

Provides: load_and_prepare_gdf() — validates input, filters to valid quadkeys,
adds Distance and PopulationDensity for regression.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np


def load_and_prepare_gdf(input_path, project_crs: str, residual_col: str = "residual"):
    """
    Load harmonised gpkg from script 02, filter to valid quadkeys, add controls.

    Args:
        residual_col: Column to use as dependent variable (residual, log_ratio,
                      scaled_log_ratio, scaled_relative). Default: residual.

    Returns:
        gdf_analysis: GeoDataFrame with residual_col, poverty_mean, Distance, PopulationDensity
    """
    gdf = gpd.read_file(input_path)
    if "poverty_mean" not in gdf.columns:
        raise ValueError("Input must include poverty_mean. Run script 01 with --poverty, then script 02.")
    if residual_col not in gdf.columns:
        raise ValueError(f"Input must include {residual_col}. Run script 02 (with --normalize for scaled_*).")

    valid = gdf[residual_col].notna() & gdf["poverty_mean"].notna()
    if "poverty_n_pixels" in gdf.columns:
        valid = valid & (gdf["poverty_n_pixels"] > 0)
    gdf_analysis = gdf[valid].copy()

    # Distance and population density for regression controls
    gdf_proj = gdf_analysis.to_crs(project_crs)
    centroid_study = gdf_proj.geometry.centroid.unary_union.centroid
    gdf_analysis["Distance"] = gdf_proj.geometry.centroid.distance(centroid_study)
    gdf_analysis["area_km2"] = gdf_proj.geometry.area / 1e6
    gdf_analysis["PopulationDensity"] = gdf_analysis["worldpop_count"] / gdf_analysis["area_km2"].clip(lower=1e-6)

    return gdf_analysis
