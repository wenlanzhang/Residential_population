#!/usr/bin/env python3
"""
03c — Spatial regression (Spatial Lag and Spatial Error models).

Runs when Moran's I on OLS residuals is significant.
Uses spreg (PySAL): ML_Lag (SLM) and ML_Error (SEM).
Uses scaled_log_ratio as dependent variable (consistent with 03a).

Outputs:
  Table2d_spatial_regression.csv
  Table2e_Moran_SEM.csv
  Table_model_comparison.csv
  slm_residual_map.png
  sem_residual_map.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import poverty_utils

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "02" / "harmonised_with_residual.gpkg"
OUT_SUBDIR = "03c_spatial_regression"


def _stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))


def parse_args():
    p = argparse.ArgumentParser(description="03c — Spatial regression (SLM, SEM)")
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("-o", "--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--project-crs", type=str, default="EPSG:32737")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError("Provide -i path (output from script 02)")

    out_dir = args.output_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = poverty_utils.load_and_prepare_gdf(args.input, args.project_crs)

    print("=" * 60)
    print("03c — Spatial Regression (SLM, SEM)")
    print("=" * 60)
    print(f"  Valid quadkeys: {len(gdf)}")

    # --------------------------------------------------
    # 1️⃣ Prepare X (STANDARDIZED — consistent with 03a)
    # --------------------------------------------------
    X = gdf[["poverty_mean", "Distance", "PopulationDensity"]].copy()
    X.columns = ["Poverty", "Distance", "PopulationDensity"]

    # Standardize Poverty
    X["Poverty"] = (X["Poverty"] - X["Poverty"].mean()) / X["Poverty"].std()

    # Standardize Distance
    X["Distance"] = (X["Distance"] - X["Distance"].mean()) / X["Distance"].std()

    # Log + standardize population density
    X["PopulationDensity"] = np.log(X["PopulationDensity"] + 1)
    X["PopulationDensity"] = (
        (X["PopulationDensity"] - X["PopulationDensity"].mean())
        / X["PopulationDensity"].std()
    )

    x = X.values
    x_names = list(X.columns)

    # --------------------------------------------------
    # 2️⃣ Dependent variable (FIXED)
    # --------------------------------------------------
    y = gdf["scaled_log_ratio"].values.reshape(-1, 1)

    print(f"  Dependent variable: scaled_log_ratio")
    print(f"  Mean DV: {y.mean():.4f}")

    # --------------------------------------------------
    # 3️⃣ Spatial weights
    # --------------------------------------------------
    gdf_proj = gdf.to_crs(args.project_crs)
    from libpysal.weights import Queen
    w = Queen.from_dataframe(gdf_proj, use_index=True)
    w.transform = "r"

    from spreg import OLS, ML_Lag, ML_Error
    from spreg.diagnostics import likratiotest
    from esda.moran import Moran

    # --------------------------------------------------
    # 4️⃣ OLS (for model comparison only)
    # --------------------------------------------------
    ols_spreg = OLS(
        y, x, w=w,
        name_y="scaled_log_ratio",
        name_x=x_names,
        name_w="Queen",
        name_ds="quadkeys"
    )

    # --------------------------------------------------
    # 5️⃣ Spatial Lag Model
    # --------------------------------------------------
    print("\n--- Spatial Lag Model (SLM) ---")
    slm = ML_Lag(
        y, x, w,
        name_y="scaled_log_ratio",
        name_x=x_names,
        name_w="Queen",
        name_ds="quadkeys"
    )
    print(slm.summary)

    # --------------------------------------------------
    # 6️⃣ Spatial Error Model
    # --------------------------------------------------
    print("\n--- Spatial Error Model (SEM) ---")
    sem = ML_Error(
        y, x, w,
        name_y="scaled_log_ratio",
        name_x=x_names,
        name_w="Queen",
        name_ds="quadkeys"
    )
    print(sem.summary)

    # --------------------------------------------------
    # 7️⃣ Moran’s I on SEM filtered residuals
    # --------------------------------------------------
    print("\n--- Moran's I on SEM filtered residuals ---")
    moran_sem = Moran(np.asarray(sem.e_filtered).ravel(), w)
    print(f"  Moran's I = {moran_sem.I:.4f}")
    print(f"  p-value (permutation) = {moran_sem.p_sim:.4f}")

    # --------------------------------------------------
    # 8️⃣ Save coefficient table
    # --------------------------------------------------
    def _scalar(x):
        arr = np.asarray(x)
        return float(arr.flat[0]) if arr.size > 0 else np.nan

    rows = []
    for i, v in enumerate(["Constant", "Poverty", "Distance", "PopulationDensity"]):
        rows.append({
            "Variable": v,
            "SLM_coef": _scalar(slm.betas[i]),
            "SLM_SE": _scalar(slm.std_err[i]),
            "SLM_p": _scalar(slm.z_stat[i][1]),
            "SLM_sig": _stars(_scalar(slm.z_stat[i][1])),
            "SEM_coef": _scalar(sem.betas[i]),
            "SEM_SE": _scalar(sem.std_err[i]),
            "SEM_p": _scalar(sem.z_stat[i][1]),
            "SEM_sig": _stars(_scalar(sem.z_stat[i][1])),
        })

    rows.append({
        "Variable": "ρ (SLM) / λ (SEM)",
        "SLM_coef": _scalar(slm.rho),
        "SLM_SE": _scalar(slm.std_err[-1]),
        "SLM_p": _scalar(slm.z_stat[-1][1]),
        "SLM_sig": _stars(_scalar(slm.z_stat[-1][1])),
        "SEM_coef": _scalar(sem.lam),
        "SEM_SE": _scalar(sem.std_err[-1]),
        "SEM_p": _scalar(sem.z_stat[-1][1]),
        "SEM_sig": _stars(_scalar(sem.z_stat[-1][1])),
    })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_dir / "Table2d_spatial_regression.csv", index=False)

    # --------------------------------------------------
    # 9️⃣ Model comparison
    # --------------------------------------------------
    comp_df = pd.DataFrame([
        {"Model": "OLS", "AIC": ols_spreg.aic, "R2": ols_spreg.r2},
        {"Model": "SLM", "AIC": slm.aic, "R2": slm.pr2},
        {"Model": "SEM", "AIC": sem.aic, "R2": sem.pr2},
    ])
    comp_df.to_csv(out_dir / "Table_model_comparison.csv", index=False)

    # --------------------------------------------------
    # 🔟 Residual maps
    # --------------------------------------------------
    gdf_map = gdf.copy()
    gdf_map["slm_residual"] = np.asarray(slm.u).ravel()
    gdf_map["sem_residual"] = np.asarray(sem.u).ravel()

    def _diverging_map(gdf, col, title, fname):
        v = gdf[col].values
        lim = max(abs(np.nanmin(v)), abs(np.nanmax(v)), 1e-6)
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(
            ax=ax,
            column=col,
            legend=True,
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )
        ax.set_title(title)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    _diverging_map(gdf_map, "slm_residual",
                   "SLM residuals: Spatial Lag Model",
                   "slm_residual_map.png")

    _diverging_map(gdf_map, "sem_residual",
                   "SEM residuals: Spatial Error Model",
                   "sem_residual_map.png")

    print("=" * 60)
    print("Spatial regression complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
