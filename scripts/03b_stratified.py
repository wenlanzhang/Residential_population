#!/usr/bin/env python3
"""
03b — Stratified comparison and inequality amplification.

Residual by poverty stratum (Table 3), Gini by poverty quintile (Table 4).

Usage:
  python scripts/03b_stratified.py -i outputs/02_harmonised_with_residual.gpkg --project-crs EPSG:32737
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import poverty_utils

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "02" / "harmonised_with_residual.gpkg"
OUT_SUBDIR = "03b_stratified"


def gini_coefficient(x):
    """Gini coefficient (0 = equality, 1 = maximal inequality)."""
    x = np.asarray(x)
    x = x[~np.isnan(x) & (x >= 0)]
    if len(x) < 2:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def parse_args():
    p = argparse.ArgumentParser(description="03b — Stratified + inequality")
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("-o", "--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--project-crs", type=str, default="EPSG:32737", help="CRS for distance (default: UTM 37S Nairobi)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError("Provide -i path (output from script 02)")
    out_dir = args.output_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = poverty_utils.load_and_prepare_gdf(args.input, args.project_crs)
    print("=" * 60)
    print("03b — Stratified Comparison + Inequality Amplification")
    print("=" * 60)
    print(f"  Valid quadkeys: {len(gdf)}")

    # A. Interaction term: Residual ~ Poverty + PopulationDensity + Poverty×PopulationDensity
    print("\n--- A. Interaction: Poverty × PopulationDensity ---")
    try:
        import statsmodels.api as sm
        pov_s = (gdf["poverty_mean"] - gdf["poverty_mean"].mean()) / gdf["poverty_mean"].std()
        pop_s = (gdf["PopulationDensity"] - gdf["PopulationDensity"].mean()) / gdf["PopulationDensity"].std()
        X = pd.DataFrame({"Poverty": pov_s, "PopulationDensity": pop_s, "Poverty_x_PopDens": pov_s * pop_s})
        X = sm.add_constant(X)
        model_int = sm.OLS(gdf["residual"], X).fit()
        print(model_int.summary().tables[1].as_text())
        if "Poverty_x_PopDens" in model_int.params and model_int.pvalues["Poverty_x_PopDens"] < 0.05:
            print("  ✓ Significant interaction — inequality amplification varies by density")
        int_coef = pd.DataFrame({
            "coef": model_int.params, "std_err": model_int.bse,
            "p": model_int.pvalues,
        })
        int_coef.to_csv(out_dir / "Table_interaction.csv", index=True)
        print(f"  Saved: {out_dir / 'Table_interaction.csv'}")

        # Marginal effects: predicted residual across poverty levels (at median density)
        pov_vals = np.linspace(gdf["poverty_mean"].min(), gdf["poverty_mean"].max(), 50)
        pov_s_pred = (pov_vals - gdf["poverty_mean"].mean()) / gdf["poverty_mean"].std()
        pop_med_s = (gdf["PopulationDensity"].median() - gdf["PopulationDensity"].mean()) / gdf["PopulationDensity"].std()
        X_pred = sm.add_constant(pd.DataFrame({
            "Poverty": pov_s_pred,
            "PopulationDensity": np.full_like(pov_s_pred, pop_med_s),
            "Poverty_x_PopDens": pov_s_pred * pop_med_s,
        }))
        pred = model_int.predict(X_pred)
        se = np.sqrt(np.diag(X_pred @ model_int.cov_params() @ X_pred.T))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pov_vals, pred, "b-", lw=2, label="Predicted residual")
        ax.fill_between(pov_vals, pred - 1.96 * se, pred + 1.96 * se, alpha=0.3)
        ax.axhline(0, color="gray", ls="--")
        ax.set_xlabel("Poverty (MPI proportion)")
        ax.set_ylabel("Predicted residual")
        ax.set_title("Marginal effect: Residual across poverty (at median density)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "marginal_effects_poverty.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_dir / 'marginal_effects_poverty.png'}")
    except ImportError:
        print("  (Install statsmodels for interaction model)")
    except Exception as e:
        print(f"  Interaction model: {e}")

    # C. Stratified comparison
    print("\n--- C. Stratified Comparison ---")
    gdf["poverty_strata"] = pd.cut(
        gdf["poverty_mean"],
        bins=[-np.inf, gdf["poverty_mean"].quantile(0.25),
              gdf["poverty_mean"].quantile(0.75), np.inf],
        labels=["Low (bottom 25%)", "Medium", "High (top 25%)"],
    )
    strata_stats = gdf.groupby("poverty_strata", observed=False)["residual"].agg(["mean", "count"])
    print(strata_stats)

    order_s = ["Low (bottom 25%)", "Medium", "High (top 25%)"]
    tbl3 = pd.DataFrame([
        (s, strata_stats.loc[s, "mean"], int(strata_stats.loc[s, "count"])) for s in order_s
    ], columns=["Poverty Stratum", "Mean Residual", "N"])
    tbl3["Mean Residual"] = tbl3["Mean Residual"].map(lambda x: f"{x:.3f}")
    tbl3.to_csv(out_dir / "Table3_poverty_strata.csv", index=False)
    print(f"  Saved: {out_dir / 'Table3_poverty_strata.csv'}")

    h_stat, p_kw = stats.kruskal(
        gdf[gdf["poverty_strata"] == "Low (bottom 25%)"]["residual"],
        gdf[gdf["poverty_strata"] == "Medium"]["residual"],
        gdf[gdf["poverty_strata"] == "High (top 25%)"]["residual"],
    )
    print(f"Kruskal-Wallis H = {h_stat:.4f}, p = {p_kw:.4f}")

    # Boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["Low (bottom 25%)", "Medium", "High (top 25%)"]
    ax.boxplot(
        [gdf[gdf["poverty_strata"] == s]["residual"].values for s in order],
        labels=order,
        patch_artist=True,
    )
    ax.axhline(0, color="gray", ls="--")
    ax.set_ylabel("Residual: log(Meta) − log(WorldPop)")
    ax.set_title("Residual by poverty stratum")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_by_poverty_strata.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'residual_by_poverty_strata.png'}")

    # D. Inequality amplification
    print("\n--- D. Inequality Amplification Across Poverty Quintiles ---")
    gdf["pov_quintile"] = pd.qcut(
        gdf["poverty_mean"], q=5,
        labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    )
    q_order = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    results = []
    for q in q_order:
        sub = gdf[gdf["pov_quintile"] == q]
        gini_wp = gini_coefficient(sub["worldpop_count"])
        gini_meta = gini_coefficient(sub["meta_baseline"])
        results.append({
            "quintile": str(q),
            "n": len(sub),
            "gini_worldpop": gini_wp,
            "gini_meta": gini_meta,
            "delta_gini": gini_meta - gini_wp,
        })
    df_ineq = pd.DataFrame(results)
    print(df_ineq.to_string(index=False))

    q_labels = {"Q1 (low)": "Q1 (Lowest poverty)", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4", "Q5 (high)": "Q5 (Highest poverty)"}
    tbl4 = pd.DataFrame([
        (
            q_labels.get(r["quintile"], r["quintile"]),
            f"{r['gini_worldpop']:.3f}",
            f"{r['gini_meta']:.3f}",
            f"{r['delta_gini']:.3f}",
        )
        for r in results
    ], columns=["Poverty Quintile", "Gini (WorldPop)", "Gini (Meta)", "ΔGini (Meta − WP)"])
    tbl4.to_csv(out_dir / "Table4_gini_by_quintile.csv", index=False)
    print(f"\n  Saved: {out_dir / 'Table4_gini_by_quintile.csv'}")
    print("  If Meta Gini >> WorldPop Gini in high-poverty quintiles → inequality amplification")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df_ineq))
    w = 0.35
    ax.bar(x - w/2, df_ineq["gini_worldpop"], w, label="WorldPop", color="steelblue")
    ax.bar(x + w/2, df_ineq["gini_meta"], w, label="Meta", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(df_ineq["quintile"])
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Inequality (Gini) by poverty quintile")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gini_by_poverty_quintile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'gini_by_poverty_quintile.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
