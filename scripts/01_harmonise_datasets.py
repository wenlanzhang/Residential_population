#!/usr/bin/env python3
"""
Step 1 — Harmonise all rasters to the quadkey grid.

Aggregates to the exact Meta quadkey grid:
  - WorldPop (zonal sum)
  - Meta baseline (already in quadkeys)
  - Poverty (zonal mean, optional via --poverty)

Usage:
  python scripts/01_harmonise_datasets.py --worldpop /path/to.tif --meta /path/to.gpkg
  python scripts/01_harmonise_datasets.py --worldpop ... --meta ... --poverty /path/to/poverty.tif
  python scripts/01_harmonise_datasets.py --filter-by meta --filter-min 50   # keep quadkeys with meta_baseline > 50
  python scripts/01_harmonise_datasets.py --filter-by worldpop --filter-min 50   # keep quadkeys with worldpop_count > 50
  python scripts/01_harmonise_datasets.py --filter-by both --filter-min 50   # keep quadkeys where BOTH meta and worldpop >= 50
"""

# python scripts/01_harmonise_datasets.py --filter-by both --filter-min 30

import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats

# Default paths
DEFAULT_BASE = Path("/Users/wenlanzhang/Downloads/PhD_UCL/Data/Residential_population")
DEFAULT_WORLDPOP = DEFAULT_BASE / "ken_pop_2024_CN_100m_R2025A_v1.tif"
DEFAULT_META = DEFAULT_BASE / "3_fb_baseline_median.gpkg"
DEFAULT_POVERTY = DEFAULT_BASE / "ken08povmpi.tif"


def filter_quadkeys(gdf, by=None, min_val=50):
    """
    Keep quadkeys only where the specified variable(s) exceed the threshold.

    Args:
        gdf: GeoDataFrame with meta_baseline and worldpop_count columns
        by: "meta"/"fb" = meta_baseline; "worldpop" = worldpop_count; "both" = both >= min_val
        min_val: minimum value (default 50)

    Returns:
        Filtered GeoDataFrame. If by is None, returns gdf unchanged.
    """
    if by is None:
        return gdf
    if by == "both":
        if "meta_baseline" not in gdf.columns or "worldpop_count" not in gdf.columns:
            return gdf
        return gdf[(gdf["meta_baseline"] >= min_val) & (gdf["worldpop_count"] >= min_val)].copy()
    col = "meta_baseline" if by in ("meta", "fb") else "worldpop_count"
    if col not in gdf.columns:
        return gdf
    return gdf[gdf[col] >= min_val].copy()


def parse_args():
    p = argparse.ArgumentParser(description="Harmonise rasters to quadkey grid")
    p.add_argument("--worldpop", type=Path, default=DEFAULT_WORLDPOP)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--poverty", type=Path, default=DEFAULT_POVERTY, help="Poverty raster (zonal mean to quadkeys)")
    p.add_argument("--no-poverty", action="store_true", help="Skip poverty aggregation")
    p.add_argument("--poverty-nodata", type=float, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--filter-by", type=str, default=None, choices=["meta", "fb", "worldpop", "both"],
                   help="Keep quadkeys where variable(s) >= threshold. meta/fb = meta_baseline; worldpop = worldpop_count; both = both >= threshold")
    p.add_argument("--filter-min", type=float, default=50,
                   help="Minimum threshold when --filter-by is set (default: 50)")
    p.add_argument("--min-meta", type=float, default=None, help="(Deprecated) Use --filter-by meta --filter-min N")
    p.add_argument("--plot", action="store_true", help="Create data overview figure (R)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "01"
    out_gpkg = args.output or (out_dir / "harmonised_meta_worldpop.gpkg")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    meta = gpd.read_file(args.meta)
    print(f"  Meta: {len(meta)} quadkeys, CRS={meta.crs}")

    # -------------------------------------------------------------------------
    # 2. Harmonise CRS and extent
    # -------------------------------------------------------------------------
    target_crs = "EPSG:4326"
    if meta.crs != target_crs:
        meta = meta.to_crs(target_crs)
        print("  Meta reprojected to EPSG:4326")

    # WorldPop is already EPSG:4326; rasterstats will handle CRS alignment
    # when zones are in same CRS as raster

    # -------------------------------------------------------------------------
    # 3. Aggregate WorldPop to Meta quadkey grid (zonal sum)
    # -------------------------------------------------------------------------
    print("Aggregating WorldPop to Meta quadkey grid...")
    # Pass path so rasterstats uses raster nodata from file
    stats = zonal_stats(
        meta.geometry,
        str(args.worldpop),
        stats=["sum", "count", "min", "max", "mean"],
        nodata=-99999.0,  # WorldPop NoData value
        all_touched=True,  # include edge pixels
    )

    # -------------------------------------------------------------------------
    # 4. Join and harmonise units
    # -------------------------------------------------------------------------
    meta = meta.copy()
    meta["worldpop_count"] = [s["sum"] if s["sum"] is not None else np.nan for s in stats]
    meta["worldpop_n_pixels"] = [s["count"] if s["count"] is not None else 0 for s in stats]
    meta["worldpop_min"] = [s["min"] if s["min"] is not None else np.nan for s in stats]
    meta["worldpop_max"] = [s["max"] if s["max"] is not None else np.nan for s in stats]
    meta["worldpop_mean"] = [s["mean"] if s["mean"] is not None else np.nan for s in stats]

    # Rename Meta baseline column (auto-detect first numeric non-geometry column)
    meta_col = next((c for c in meta.columns if c not in ("geometry", "quadkey") and pd.api.types.is_numeric_dtype(meta[c])), None)
    if meta_col and meta_col != "meta_baseline":
        meta = meta.rename(columns={meta_col: "meta_baseline"})

    # -------------------------------------------------------------------------
    # 5. Validation: zero-division and sparsity
    # -------------------------------------------------------------------------
    print("\n--- Harmonisation checks ---")

    # Same projection
    print(f"CRS: {meta.crs} (both datasets)")

    # Spatial extent: Meta defines extent; WorldPop covers Kenya so Nairobi is inside
    meta_bounds = meta.total_bounds
    print(f"Spatial extent (Meta): {meta_bounds}")

    # Unit: both in counts per quadkey
    print("Unit: counts per quadkey (WorldPop = sum of 100m cells; Meta = midnight baseline)")

    # Zero-division: cells with 0 in either dataset
    wp_zeros = (meta["worldpop_count"] == 0) | meta["worldpop_count"].isna()
    meta_zeros = meta["meta_baseline"] == 0
    both_valid = ~wp_zeros & ~meta_zeros
    print(f"WorldPop zero/NaN cells: {wp_zeros.sum()}")
    print(f"Meta zero cells: {meta_zeros.sum()}")
    print(f"Cells with both non-zero: {both_valid.sum()} (safe for ratio comparisons)")

    # Sparsity: cells with very few pixels
    sparse = meta["worldpop_n_pixels"] < 10
    print(f"Extreme sparsity (n_pixels < 10): {sparse.sum()} quadkeys")

    # -------------------------------------------------------------------------
    # 6. Optional: Aggregate poverty raster to quadkey grid
    # -------------------------------------------------------------------------
    poverty_path = None if args.no_poverty else args.poverty
    if poverty_path and poverty_path.exists():
        print("\n--- Aggregating poverty raster to quadkeys ---")
        zs_kw = {"stats": ["mean", "count"], "all_touched": True}
        if args.poverty_nodata is not None:
            zs_kw["nodata"] = args.poverty_nodata
        stats_pov = zonal_stats(meta.geometry, str(poverty_path), **zs_kw)
        meta["poverty_mean"] = [s["mean"] if s["mean"] is not None else np.nan for s in stats_pov]
        meta["poverty_n_pixels"] = [s["count"] if s["count"] is not None else 0 for s in stats_pov]
        print(f"  Poverty: mean per quadkey, valid cells: {(meta['poverty_n_pixels'] > 0).sum()}")

    # -------------------------------------------------------------------------
    # 6b. Optional: Filter quadkeys by minimum count
    # -------------------------------------------------------------------------
    filter_by = args.filter_by
    if args.min_meta is not None:
        filter_by = "meta"
        args.filter_min = args.min_meta
        print("  Note: --min-meta is deprecated, use --filter-by meta --filter-min N")
    if filter_by is not None:
        before = len(meta)
        meta = filter_quadkeys(meta, by=filter_by, min_val=args.filter_min)
        if filter_by == "both":
            print(f"\n--- Filter: meta_baseline >= {args.filter_min} AND worldpop_count >= {args.filter_min} ---")
        else:
            col = "meta_baseline" if filter_by in ("meta", "fb") else "worldpop_count"
            print(f"\n--- Filter: {col} >= {args.filter_min} ---")
        print(f"  Kept {len(meta)} / {before} quadkeys (dropped {before - len(meta)})")

    # Summary stats
    print("\n--- Harmonised summary ---")
    cols = ["worldpop_count", "meta_baseline"]
    if "poverty_mean" in meta.columns:
        cols.append("poverty_mean")
    print(meta[cols].describe())

    # -------------------------------------------------------------------------
    # 7. Save
    # -------------------------------------------------------------------------
    meta.to_file(out_gpkg, driver="GPKG")
    print(f"\nSaved: {out_gpkg}")

    # -------------------------------------------------------------------------
    # 8. Optional: Data overview figure (R)
    # -------------------------------------------------------------------------
    if args.plot:
        import subprocess
        script_dir = Path(__file__).resolve().parent
        r_script = script_dir / "01_plot_descriptive.R"
        if r_script.exists():
            out_path = out_dir / "01_descriptive_overview.png"
            result = subprocess.run(
                ["Rscript", str(r_script), "-i", str(out_gpkg)],
                cwd=script_dir.parent,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  Figures: 01_data_overview.png, 01_bivariate_worldpop_meta.png, 01_bivariate_worldpop_meta_basemap.png")
            else:
                print(f"  R figure failed: {result.stderr[:200] if result.stderr else result.stdout}")
        else:
            print("  01_plot_descriptive.R not found, skipping figure")

    return meta


if __name__ == "__main__":
    main()
