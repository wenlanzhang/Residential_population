#!/usr/bin/env python3
"""
03a — Spearman correlation and regression (Residual ~ Poverty).

Usage:
  conda activate geo_env_LLM
  python scripts/03a_regression.py -i outputs/02_harmonised_with_residual.gpkg --project-crs EPSG:32737

Notes:
- Default dependent variable is 'scaled_log_ratio' (multiplicative local bias).
- If the chosen residual var is missing in the input GPKG, the script falls back to 'residual'.
- Outputs written to outputs/03a_regression/.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import poverty_utils  # your helper that loads & prepares the gdf

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "02" / "harmonised_with_residual.gpkg"
OUT_SUBDIR = "03a_regression"


def _stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))


def safe_standardize(arr):
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std < 1e-10:
        std = 1.0
    return (arr - mean) / std


def parse_args():
    p = argparse.ArgumentParser(description="03a — Spearman + regression")
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT, help="02_harmonised_with_residual.gpkg")
    p.add_argument("-o", "--output-dir", type=Path, default=PROJECT_ROOT / "outputs", help="outputs root")
    p.add_argument("--project-crs", type=str, default="EPSG:32737", help="CRS for distances (default UTM 37S)")
    p.add_argument("--residual-var", type=str, default="scaled_log_ratio",
                   choices=["residual", "log_ratio", "scaled_log_ratio", "scaled_relative"],
                   help="Which residual column to use as dependent variable.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input: {args.input}. Run step 02 first.")

    out_dir = args.output_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd
    # quick check for requested column
    sample = gpd.read_file(args.input, rows=1)
    residual_var = args.residual_var if args.residual_var in sample.columns else "residual"
    if residual_var != args.residual_var:
        print(f"Note: requested residual var '{args.residual_var}' not in input; falling back to '{residual_var}'")

    # load and prepare gdf (your helper should handle CRS, distance, population density columns, etc.)
    gdf = poverty_utils.load_and_prepare_gdf(args.input, args.project_crs, residual_col=residual_var)

    print("=" * 60)
    print("03a — Spearman + Regression")
    print("=" * 60)
    print(f"  Dependent variable: {residual_var}")
    print(f"  Valid quadkeys: {len(gdf)}")
    if "poverty_mean" in gdf.columns:
        print(f"  Poverty range: {gdf['poverty_mean'].min():.4f}–{gdf['poverty_mean'].max():.4f}")
    else:
        raise KeyError("poverty_mean missing in prepared gdf; ensure step 01 included poverty or re-run with poverty.")

    # pull arrays
    poverty = gdf["poverty_mean"].values
    dv = gdf[residual_var].values

    # standardize poverty for interpretable coefficients
    poverty_z = safe_standardize(poverty)

    # A. Spearman
    print("\n--- A. Spearman Correlation: Residual ~ Poverty ---")
    r_spear, p_spear = stats.spearmanr(poverty, dv)
    print(f"Spearman ρ = {r_spear:.4f}, p = {p_spear:.4f}")
    if residual_var in ["log_ratio", "scaled_log_ratio"]:
        print("Interpretation: ρ < 0 → Meta underrepresents poorer areas (multiplicative bias).")
    else:
        print("Interpretation: sign indicates direction of proportional bias (check metric).")

    # B. Regression
    print("\n--- B. Regression ---")
    print("\nModel 1: DV = β0 + β1 * Poverty_z (bivariate)")
    slope, intercept, r_val, p_val, se = linregress(poverty_z, dv)
    m1_poverty_coef, m1_poverty_se = slope, se
    m1_const_coef = intercept
    print(f"  β₀ = {intercept:.4f}")
    print(f"  β₁ = {slope:.4f} (SE = {se:.4f}), p = {p_val:.4f}")

    # Model 2: controlled
    print("\nModel 2: DV = β0 + β1*Poverty_z + β2*Distance_z + β3*LogPopDensity_z (controlled)")
    model2 = None
    model2_robust = None
    try:
        import statsmodels.api as sm
        # prepare predictors with safe standardization
        if "Distance" not in gdf.columns:
            raise KeyError("Distance column missing from gdf. poverty_utils.load_and_prepare_gdf should compute it.")
        if "PopulationDensity" not in gdf.columns:
            raise KeyError("PopulationDensity missing from gdf. Ensure it's prepared in poverty_utils.")

        X_df = pd.DataFrame({
            "Poverty": poverty_z,
            "Distance": safe_standardize(gdf["Distance"].values),
            "PopulationDensity": safe_standardize(np.log(gdf["PopulationDensity"].values + 1.0))
        })
        X = sm.add_constant(X_df)
        model2 = sm.OLS(dv, X).fit()
        model2_robust = sm.OLS(dv, X).fit(cov_type="HC3")

        # print tidy coefficients table
        print(model2.summary().tables[1].as_text())

        # multiplicative interpretation if DV is log ratio
        if residual_var in ["log_ratio", "scaled_log_ratio"]:
            try:
                mult_effect = np.exp(model2.params["Poverty"])
                print(f"\nexp(β₁) = {mult_effect:.3f} multiplicative change in Meta/WP per SD increase in poverty")
            except Exception:
                pass

        # save coefficients
        coef_df = pd.DataFrame({
            "coef": model2.params,
            "std_err": model2.bse,
            "std_err_HC3": model2_robust.bse,
            "t": model2.tvalues,
            "t_HC3": model2_robust.tvalues,
            "p": model2.pvalues,
            "p_HC3": model2_robust.pvalues,
        })
        coef_df.to_csv(out_dir / "regression_coefficients.csv")
        print(f"  Saved: {out_dir / 'regression_coefficients.csv'}")

        # C. Heteroskedasticity
        print("\n--- C. Heteroskedasticity ---")
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan, het_white
            bp_stat, bp_p, _, _ = het_breuschpagan(model2.resid, X)
            print(f"  Breusch-Pagan: LM={bp_stat:.4f}, p={bp_p:.4f}")
            w_stat, w_p, _, _ = het_white(model2.resid, X)
            print(f"  White: LM={w_stat:.4f}, p={w_p:.4f}")
            het_df = pd.DataFrame([
                {"Test": "Breusch-Pagan", "Statistic": bp_stat, "p_value": bp_p},
                {"Test": "White", "Statistic": w_stat, "p_value": w_p},
            ])
            het_df.to_csv(out_dir / "Table2b_heteroskedasticity.csv", index=False)
            print(f"  Saved: {out_dir / 'Table2b_heteroskedasticity.csv'}")
        except Exception as e:
            print(f"  Heteroskedasticity tests skipped or failed: {e}")

        # D. VIF
        print("\n--- D. Multicollinearity (VIF) ---")
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vif_results = []
            # VIF needs design matrix without const in position argument consistent with columns
            cols = X.columns.tolist()
            for i, col in enumerate(cols):
                if col == "const":
                    continue
                vif = variance_inflation_factor(X.values, cols.index(col))
                vif_results.append({"Variable": col, "VIF": vif})
            vif_df = pd.DataFrame(vif_results)
            vif_df.to_csv(out_dir / "Table2b_VIF.csv", index=False)
            print(vif_df.to_string(index=False))
            print(f"  Saved: {out_dir / 'Table2b_VIF.csv'}")
        except Exception as e:
            print(f"  VIF calculation skipped or failed: {e}")

        # E. Spatial autocorrelation (Moran's I on OLS residuals)
        print("\n--- E. Spatial Autocorrelation (Moran's I on residuals) ---")
        try:
            import libpysal
            from esda.moran import Moran
            gdf_proj = gdf.to_crs(args.project_crs)
            # create weights (queen contiguity); if fails, try k-nearest fallback
            try:
                w = libpysal.weights.Queen.from_dataframe(gdf_proj, use_index=True)
            except Exception:
                # fallback to kNN
                w = libpysal.weights.KNN.from_dataframe(gdf_proj, k=8)
            w.transform = "r"
            moran = Moran(model2.resid, w)
            print(f"  Moran's I = {moran.I:.4f}")
            print(f"  p-value (permutation) = {moran.p_sim:.4f}")
            moran_df = pd.DataFrame([
                {"Metric": "Moran's I", "Value": moran.I},
                {"Metric": "p-value (permutation)", "Value": moran.p_sim},
                {"Metric": "z-score", "Value": moran.z_sim},
            ])
            moran_df.to_csv(out_dir / "Table2c_Moran.csv", index=False)
            print(f"  Saved: {out_dir / 'Table2c_Moran.csv'}")
        except Exception as e:
            print(f"  Moran's I skipped or failed: {e}")

        # F. Local Moran (LISA) and Getis-Ord Gi*
        print("\n--- F. Local clustering (LISA, Getis-Ord Gi*) ---")
        try:
            from esda.moran import Moran_Local
            from esda.getisord import G_Local
            import geopandas as gpd
            # reuse gdf_proj and w if available, else recreate
            try:
                gdf_proj
            except NameError:
                gdf_proj = gdf.to_crs(args.project_crs)
            try:
                w
            except NameError:
                try:
                    w = libpysal.weights.Queen.from_dataframe(gdf_proj, use_index=True)
                    w.transform = "r"
                except Exception:
                    w = libpysal.weights.KNN.from_dataframe(gdf_proj, k=8)
                    w.transform = "r"

            li = Moran_Local(model2.resid, w)
            gi = G_Local(model2.resid, w)

            # save cluster results into gdf and export maps
            gdf_out = gdf.copy()
            gdf_out["lisa_q"] = li.q
            gdf_out["lisa_p"] = li.p_sim
            gdf_out["gi_star"] = gi.Gs
            gdf_out["gi_p"] = gi.p_sim

            # LISA cluster map
            fig, ax = plt.subplots(figsize=(8, 8))
            gdf_out.plot(ax=ax, column="lisa_q", categorical=True, legend=True,
                         cmap="RdYlBu_r", legend_kwds={"title": "LISA q"})
            ax.set_title("Local Moran's I (LISA) clusters")
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(out_dir / "03a_local_moran_map.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {out_dir / '03a_local_moran_map.png'}")

            # Gi* hotspot map
            fig, ax = plt.subplots(figsize=(8, 8))
            gdf_out.plot(ax=ax, column="gi_star", legend=True, cmap="RdBu_r", legend_kwds={"shrink": 0.6})
            ax.set_title("Getis-Ord Gi* (hotspots)")
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(out_dir / "03a_hotspot_map.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {out_dir / '03a_hotspot_map.png'}")

            # Save geo package with diagnostics if desired
            try:
                gdf_out.to_file(out_dir / "03a_regression_gdf.gpkg", driver="GPKG")
                print(f"  Saved: {out_dir / '03a_regression_gdf.gpkg'}")
            except Exception:
                pass

        except Exception as e:
            print(f"  Local clustering (LISA/Gi*) skipped or failed: {e}")

    except ImportError:
        print("  (Install statsmodels to run regressions and diagnostics: pip install statsmodels)")

    # write Table 2 formatted output
    print("\n--- Writing Table 2 (regression summary) ---")
    try:
        # minimal Model 1 entry + Model 2 detailed if available
        tbl2 = []
        tbl2.append(("Poverty (std)", f"{m1_poverty_coef:.3f}{_stars(p_val)} ({m1_poverty_se:.3f})", ""))
        tbl2.append(("Constant", f"{m1_const_coef:.3f}", ""))
        if model2 is not None:
            def _se_str(m, r, col):
                try:
                    base = f"{m.params[col]:.3f}{_stars(m.pvalues[col])} (SE={m.bse[col]:.3f}"
                    if r is not None and col in r.bse.index:
                        return base + f", HC3={r.bse[col]:.3f})"
                    return base + ")"
                except Exception:
                    return ""
            tbl2 = [
                ("Poverty (std)", f"{m1_poverty_coef:.3f}{_stars(p_val)} ({m1_poverty_se:.3f})",
                 _se_str(model2, model2_robust, "Poverty")),
                ("Distance (std)", "—", _se_str(model2, model2_robust, "Distance")),
                ("Population density (std)", "—", _se_str(model2, model2_robust, "PopulationDensity")),
                ("Constant", f"{m1_const_coef:.3f}***", f"{model2.params['const']:.3f}{_stars(model2.pvalues['const'])} ({model2.bse['const']:.3f})"),
                ("Observations", str(len(gdf)), str(len(gdf))),
            ]
        tbl2_df = pd.DataFrame(tbl2, columns=["Variable", "Model 1 (Bivariate)", "Model 2 (Controlled)"])
        tbl2_df.to_csv(out_dir / "Table2_regression.csv", index=False)
        print(f"  Saved: {out_dir / 'Table2_regression.csv'}")
    except Exception as e:
        print(f"  Writing Table 2 failed: {e}")

    print("\nDone.")
    print("=" * 60)


if __name__ == "__main__":
    main()
