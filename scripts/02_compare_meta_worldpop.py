#!/usr/bin/env python3
"""
Step 2 — Compare Meta and WorldPop (after harmonisation).

1. Spatial Pattern Agreement (Spearman rank correlation)
2. Distribution Similarity: histograms, KDE, KS test, Earth Mover's Distance
3. Inequality Metrics: Gini coefficients, ΔGini, Lorenz curves
4. Residual Spatial Bias: residual = log(Meta) - log(WorldPop), map, contextual tests

Requires outputs from 01_harmonise_datasets.py.

Usage:
  conda activate geo_env_LLM
  python scripts/02_compare_meta_worldpop.py
  python scripts/02_compare_meta_worldpop.py -i outputs/01_harmonised_meta_worldpop.gpkg
  # With optional context for residual tests:
  python scripts/02_compare_meta_worldpop.py --informal informal.gpkg --rural rural.gpkg --nightlight viirs.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "01" / "harmonised_meta_worldpop.gpkg"
OUT_DIR = PROJECT_ROOT / "outputs" / "02"


def gini_coefficient(x):
    """Gini coefficient (0 = perfect equality, 1 = maximal inequality)."""
    x = np.asarray(x)
    x = x[~np.isnan(x) & (x >= 0)]
    if len(x) == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def lorenz_curve(x):
    """Return (cumulative share of population, cumulative share of value) for Lorenz curve."""
    x = np.asarray(x)
    x = x[~np.isnan(x) & (x >= 0)]
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return np.array([0]), np.array([0])
    cumx = np.cumsum(x)
    return np.arange(1, n + 1) / n, cumx / cumx[-1]


def _load_context(gdf_valid, path: Path, key_col="quadkey"):
    """Load auxiliary context and join to gdf by quadkey."""
    path = Path(path)
    if not path.exists():
        return None
    import pandas as pd
    if path.suffix.lower() in (".gpkg", ".geojson", ".shp"):
        aux = gpd.read_file(path)
    else:
        aux = pd.read_csv(path)
    if key_col not in aux.columns or key_col not in gdf_valid.columns:
        return None
    # Drop geometry from aux to avoid merge conflicts
    cols = [c for c in aux.columns if c != key_col and c != "geometry"]
    if not cols:
        return None
    return gdf_valid.merge(aux[[key_col] + cols], on=key_col, how="left")


def run_comparison(input_gpkg: Path, out_dir: Path, args=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(input_gpkg)

    # Filter to valid pairs (both > 0) for log-based analyses
    wp = gdf["worldpop_count"].values
    meta = gdf["meta_baseline"].values
    valid = (wp > 0) & (meta > 0)
    wp_v = wp[valid]
    meta_v = meta[valid]
    gdf_valid = gdf[valid].copy()

    print("=" * 60)
    print("STEP 2: Meta vs WorldPop Comparison")
    print("=" * 60)
    print(f"Total quadkeys: {len(gdf)}, valid (both > 0): {valid.sum()}")

    # -------------------------------------------------------------------------
    # 1. Spatial Pattern Agreement (Spearman)
    # -------------------------------------------------------------------------
    print("\n--- 1. Spatial Pattern Agreement ---")
    r_spearman, p_spearman = stats.spearmanr(wp_v, meta_v)
    print(f"Spearman ρ = {r_spearman:.4f}, p-value = {p_spearman:.2e}")
    print("Interpretation: positive ρ = similar spatial ranking; ρ near 0 = little agreement")

    # -------------------------------------------------------------------------
    # 2. Distribution Similarity (normalized)
    # -------------------------------------------------------------------------
    print("\n--- 2. Distribution Similarity ---")

    # Normalize to [0,1] for shape comparison
    wp_norm = (wp_v - wp_v.min()) / (wp_v.max() - wp_v.min() + 1e-10)
    meta_norm = (meta_v - meta_v.min()) / (meta_v.max() - meta_v.min() + 1e-10)

    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(wp_norm, bins=30, alpha=0.7, label="WorldPop", color="steelblue", density=True)
    axes[0].hist(meta_norm, bins=30, alpha=0.7, label="Meta", color="coral", density=True)
    axes[0].set_xlabel("Normalized value")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Histograms (normalized)")
    axes[0].legend()

    # KDE
    x = np.linspace(0, 1, 200)
    kde_wp = stats.gaussian_kde(wp_norm, bw_method="scott")
    kde_meta = stats.gaussian_kde(meta_norm, bw_method="scott")
    axes[1].plot(x, kde_wp(x), label="WorldPop", color="steelblue", lw=2)
    axes[1].plot(x, kde_meta(x), label="Meta", color="coral", lw=2)
    axes[1].set_xlabel("Normalized value")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Kernel density curves")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_distribution_histogram_kde.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / '02_distribution_histogram_kde.png'}")

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(wp_norm, meta_norm)
    print(f"Kolmogorov–Smirnov: statistic = {ks_stat:.4f}, p-value = {ks_pval:.2e}")
    print("  (H0: same distribution; low p = distributions differ)")

    # Earth Mover's Distance (Wasserstein)
    emd = stats.wasserstein_distance(wp_norm, meta_norm)
    print(f"Earth Mover's Distance (Wasserstein): {emd:.4f}")
    print("  (0 = identical; higher = more dissimilar)")

    # -------------------------------------------------------------------------
    # 3. Inequality Metrics
    # -------------------------------------------------------------------------
    print("\n--- 3. Inequality Metrics ---")

    gini_wp = gini_coefficient(wp_v)
    gini_meta = gini_coefficient(meta_v)
    delta_gini = gini_meta - gini_wp

    print(f"Gini (WorldPop): {gini_wp:.4f}")
    print(f"Gini (Meta):     {gini_meta:.4f}")
    print(f"ΔGini (Meta - WorldPop): {delta_gini:.4f}")
    print("  (positive ΔGini = Meta shows more inequality)")

    # Lorenz curves
    fig, ax = plt.subplots(figsize=(6, 6))
    pop_wp, val_wp = lorenz_curve(wp_v)
    pop_meta, val_meta = lorenz_curve(meta_v)
    ax.plot(np.concatenate([[0], pop_wp]), np.concatenate([[0], val_wp]), label="WorldPop", color="steelblue", lw=2)
    ax.plot(np.concatenate([[0], pop_meta]), np.concatenate([[0], val_meta]), label="Meta", color="coral", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect equality")
    ax.set_xlabel("Cumulative share of quadkeys (by count)")
    ax.set_ylabel("Cumulative share of population")
    ax.set_title("Lorenz curves")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "02_lorenz_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / '02_lorenz_curves.png'}")

    # -------------------------------------------------------------------------
    # 4. Residual Spatial Bias
    # -------------------------------------------------------------------------
    print("\n--- 4. Residual Spatial Bias ---")

    # --- Core residual definitions ---
    residual_raw = meta_v - wp_v
    log_ratio = np.log(meta_v) - np.log(wp_v)  # log(Meta / WorldPop)
    residual_relative = (meta_v - wp_v) / np.maximum(wp_v, 1e-10)

    # --- Proper global total scaling ---
    total_wp = wp_v.sum()
    total_meta = meta_v.sum()
    scale_factor = total_wp / total_meta

    meta_scaled = meta_v * scale_factor
    log_ratio_scaled = np.log(meta_scaled) - np.log(wp_v)
    residual_relative_scaled = (meta_scaled - wp_v) / np.maximum(wp_v, 1e-10)

    print(f"Total WorldPop: {total_wp:.0f}")
    print(f"Total Meta:     {total_meta:.0f}")
    print(f"Global scale factor (WP / Meta): {scale_factor:.4f}")

    print("Raw residual: mean = {:.4f}, std = {:.4f}".format(residual_raw.mean(), residual_raw.std()))
    print("Log-ratio: mean = {:.4f}, std = {:.4f} (log(Meta/WP))".format(log_ratio.mean(), log_ratio.std()))
    print("Relative residual: mean = {:.4f}, std = {:.4f}".format(residual_relative.mean(), residual_relative.std()))
    print("Scaled log-ratio: mean = {:.4f}, std = {:.4f} (after global alignment)".format(
        log_ratio_scaled.mean(), log_ratio_scaled.std()))
    print("Scaled relative residual: mean = {:.4f}, std = {:.4f}".format(
        residual_relative_scaled.mean(), residual_relative_scaled.std()))

    # Store in GeoDataFrame
    gdf_valid = gdf_valid.copy()
    gdf_valid["residual_raw"] = residual_raw
    gdf_valid["log_ratio"] = log_ratio
    gdf_valid["residual"] = log_ratio  # alias for downstream (03a, 03b, poverty_utils)
    gdf_valid["residual_relative"] = residual_relative
    gdf_valid["log_ratio_scaled"] = log_ratio_scaled
    gdf_valid["scaled_log_ratio"] = log_ratio_scaled  # alias for 03a --residual-var
    gdf_valid["residual_relative_scaled"] = residual_relative_scaled
    gdf_valid["scaled_relative"] = residual_relative_scaled  # alias for 03a --residual-var

    def _residual_map(gdf, col, title, fname):
        v = gdf[col].values
        lim = max(abs(np.nanmin(v)), abs(np.nanmax(v)), 1e-6)
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(ax=ax, column=col, legend=True, cmap="RdBu_r", legend_kwds={"shrink": 0.6},
                 vmin=-lim, vmax=lim)
        ax.set_title(title)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()

    _residual_map(gdf_valid, "residual_raw", "Residual (raw): Meta − WorldPop", "02_residual_raw.png")
    _residual_map(gdf_valid, "log_ratio", "Log-ratio: log(Meta / WorldPop)", "02_log_ratio.png")
    _residual_map(gdf_valid, "residual_relative", "Relative residual: (Meta − WP) / WP", "02_residual_relative.png")
    _residual_map(gdf_valid, "log_ratio_scaled", "Scaled log-ratio (totals aligned)", "02_log_ratio_scaled.png")
    _residual_map(gdf_valid, "residual_relative_scaled", "Scaled relative residual (totals aligned)", "02_relative_scaled.png")

    print("  Saved residual maps.")

    # Save gdf with residual for downstream use
    out_gpkg = out_dir / "harmonised_with_residual.gpkg"
    gdf_valid.to_file(out_gpkg, driver="GPKG")
    print(f"  Saved: {out_gpkg}")


    # -------------------------------------------------------------------------
    # 4b. Contextual tests (residuals by context)
    # -------------------------------------------------------------------------
    print("\n--- 4b. Contextual Tests (Residual Spatial Bias) ---")

    # Compute peripheral zones: distance from centroid to study-area centroid (in metres)
    gdf_proj = gdf_valid.to_crs("EPSG:32737")  # UTM 37S (Nairobi)
    # centroid_study = gdf_proj.geometry.centroid.unary_union.centroid
    centroid_study = gdf_proj.geometry.centroid.unary_union.centroid
    gdf_valid["dist_centroid"] = gdf_proj.geometry.centroid.distance(centroid_study)
    gdf_valid["Distance"] = gdf_valid["dist_centroid"]
    gdf_valid["area_km2"] = gdf_proj.geometry.area / 1e6
    gdf_valid["PopulationDensity"] = gdf_valid["worldpop_count"] / gdf_valid["area_km2"].clip(lower=1e-6)
    # Binary: peripheral = top 25% by distance
    gdf_valid["peripheral"] = gdf_valid["dist_centroid"] >= gdf_valid["dist_centroid"].quantile(0.75)

    # Test: Are residuals larger in peripheral zones?
    res_peripheral = gdf_valid[gdf_valid["peripheral"]]["log_ratio_scaled"].values
    res_central = gdf_valid[~gdf_valid["peripheral"]]["log_ratio_scaled"].values
    t_peripheral, p_peripheral = stats.ttest_ind(res_peripheral, res_central)
    mean_periph = res_peripheral.mean()
    mean_cent = res_central.mean()
    print(f"Peripheral zones (top 25% by distance from centroid):")
    print(f"  Mean scaled log-ratio (peripheral): {mean_periph:.4f}")
    print(f"  Mean scaled log-ratio (central):   {mean_cent:.4f}")
    print(f"  t-test p-value: {p_peripheral:.4f}")
    diff = mean_periph - mean_cent
    print(f"  Difference (peripheral − central): {diff:.4f}")
    print("  Interpretation: negative difference = stronger Meta underrepresentation in periphery")


    # Optional: auxiliary context (informal, rural, nightlight)
    if args:
        for path_arg, col, label in [
            (getattr(args, "informal", None), "informal", "Informal settlements"),
            (getattr(args, "rural", None), "rural", "Rural areas"),
        ]:
            if path_arg:
                merged = _load_context(gdf_valid, path_arg)
                if merged is not None and col in merged.columns:
                    run_contextual_test(merged, col, label)
        # Nightlight: treat as continuous; test high vs low (median split)
        nl_path = getattr(args, "nightlight", None) if args else None
        if nl_path:
            merged = _load_context(gdf_valid, nl_path)
            if merged is not None:
                cand = [c for c in merged.columns if "night" in c.lower() or "viirs" in c.lower() or c == "nightlight"]
                nc = cand[0] if cand else None
                if nc:
                    merged["low_nightlight"] = merged[nc] < merged[nc].median()
                    run_contextual_test(merged, "low_nightlight", "Low vs high nightlight")

    # -------------------------------------------------------------------------
    # 4c. Table 2 — OLS Regression (Area-weighted)
    # -------------------------------------------------------------------------
    import pandas as pd
    if "poverty_mean" in gdf_valid.columns:
        print("\n--- 4c. OLS Regression (scaled log-ratio ~ Poverty + Distance + Log Pop Density) ---")
        try:
            import statsmodels.api as sm
            gdf_reg = gdf_valid.dropna(subset=["poverty_mean", "scaled_log_ratio", "Distance", "PopulationDensity"])
            if "poverty_n_pixels" in gdf_reg.columns:
                gdf_reg = gdf_reg[gdf_reg["poverty_n_pixels"] > 0]
            if len(gdf_reg) < 10:
                print("  Insufficient valid rows for regression, skipping Table 2")
            else:
                def _z(x):
                    x = np.asarray(x, dtype=float)
                    m, s = np.nanmean(x), np.nanstd(x)
                    return (x - m) / (s + 1e-10)
                pov_z = _z(gdf_reg["poverty_mean"])
                dist_z = _z(gdf_reg["Distance"])
                logpop_z = _z(np.log(gdf_reg["PopulationDensity"].values + 1))
                X = sm.add_constant(np.column_stack([pov_z, dist_z, logpop_z]))
                X_df = pd.DataFrame(X, columns=["const", "Poverty_z", "Distance_z", "LogPopDensity_z"])
                model = sm.OLS(gdf_reg["scaled_log_ratio"].values, X_df).fit()
                def _p_fmt(p):
                    return "<0.001" if p < 0.001 else f"{p:.4f}"
                def _num(x):
                    return f"{float(x):.4f}"
                tbl2 = pd.DataFrame([
                    {"Variable": "Constant", "Coefficient": _num(model.params["const"]), "Std. Error": _num(model.bse["const"]), "p-value": _p_fmt(model.pvalues["const"])},
                    {"Variable": "Poverty (z)", "Coefficient": _num(model.params["Poverty_z"]), "Std. Error": _num(model.bse["Poverty_z"]), "p-value": _p_fmt(model.pvalues["Poverty_z"])},
                    {"Variable": "Distance (z)", "Coefficient": _num(model.params["Distance_z"]), "Std. Error": _num(model.bse["Distance_z"]), "p-value": _p_fmt(model.pvalues["Distance_z"])},
                    {"Variable": "Log Population Density (z)", "Coefficient": _num(model.params["LogPopDensity_z"]), "Std. Error": _num(model.bse["LogPopDensity_z"]), "p-value": _p_fmt(model.pvalues["LogPopDensity_z"])},
                ])
                tbl2.to_csv(out_dir / "Table2_OLS_regression.csv", index=False)
                print("\nTable 2. OLS Regression Results (Area-weighted)")
                print(tbl2.to_string(index=False))
                print(f"  Saved: {out_dir / 'Table2_OLS_regression.csv'}")
                print(f"  N = {len(gdf_reg)}, R² = {model.rsquared:.4f}")
        except ImportError:
            print("  (Install statsmodels for OLS: pip install statsmodels)")
        except Exception as e:
            print(f"  OLS regression failed: {e}")
    else:
        print("\n--- 4c. OLS Regression ---")
        print("  Skipped: poverty_mean not in input. Run 01 with --poverty.")

    # -------------------------------------------------------------------------
    # Summary & Table 1
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Spearman ρ: {r_spearman:.4f}")
    print(f"KS p-value: {ks_pval:.2e}")
    print(f"EMD: {emd:.4f}")
    print(f"Gini WP / Meta / ΔGini: {gini_wp:.4f} / {gini_meta:.4f} / {delta_gini:.4f}")
    print(f"Mean log-ratio: {log_ratio.mean():.4f}")
    print(f"Mean scaled log-ratio: {log_ratio_scaled.mean():.4f}")
    print(f"Peripheral vs central log-ratio (p): {p_peripheral:.4f}")


    # Table 1 — Meta vs WorldPop comparison metrics
    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""
    import pandas as pd
    tbl1 = pd.DataFrame([
        ("Number of quadkeys", valid.sum()),
        ("Spearman ρ", f"{r_spearman:.3f}"),
        ("KS statistic", f"{ks_stat:.3f}"),
        ("KS p-value", "< 0.001" if ks_pval < 0.001 else f"{ks_pval:.3f}"),
        ("Earth Mover's Distance", f"{emd:.3f}"),
        ("Gini (WorldPop)", f"{gini_wp:.3f}"),
        ("Gini (Meta baseline)", f"{gini_meta:.3f}"),
        ("ΔGini (Meta − WP)", f"{delta_gini:.3f}"),
        ("Mean log-ratio", f"{log_ratio.mean():.3f}"),
        ("Mean scaled log-ratio", f"{log_ratio_scaled.mean():.3f}")
    ], columns=["Metric", "Value"])

    tbl1.to_csv(out_dir / "Table1_meta_worldpop_metrics.csv", index=False)
    print(f"\n  Saved: {out_dir / 'Table1_meta_worldpop_metrics.csv'}")



def run_contextual_test(gdf_valid, context_col, label):
    """T-test: are residuals different in context vs not?"""
    if context_col not in gdf_valid.columns:
        return
    mask = gdf_valid[context_col].astype(bool)
    if mask.sum() < 3 or (~mask).sum() < 3:
        print(f"  {label}: insufficient samples, skipping")
        return
    res_in = gdf_valid[mask]["residual"].values
    res_out = gdf_valid[~mask]["residual"].values
    t, p = stats.ttest_ind(res_in, res_out)
    print(f"  {label}: mean residual (in)={res_in.mean():.4f}, (out)={res_out.mean():.4f}, p={p:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare Meta and WorldPop (post-harmonisation)")
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("-o", "--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    # p.add_argument("--normalize", action="store_true", default=True, help="Global-scaling: add scaled_log_ratio, scaled_relative (default: True)")
    # p.add_argument("--no-normalize", dest="normalize", action="store_false", help="Skip global-scaling normalization")
    p.add_argument("--informal", type=Path, default=None, help="GPKG/CSV with quadkey + informal (0/1)")
    p.add_argument("--rural", type=Path, default=None, help="GPKG/CSV with quadkey + rural (0/1)")
    p.add_argument("--nightlight", type=Path, default=None, help="GPKG/CSV with quadkey + nightlight value")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Run 01_harmonise_datasets.py first. Missing: {args.input}")
    out_dir = args.output_dir / "02"
    run_comparison(args.input, out_dir, args)


if __name__ == "__main__":
    main()
