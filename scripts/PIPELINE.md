# Analysis Pipeline

Run from project root. Scripts use default paths; `-i` can be omitted.

## Script structure

| Script | Purpose | Input (default) | Output |
|--------|---------|-----------------|--------|
| **01** | Harmonise + descriptive maps (data overview, bivariate WorldPop vs Meta) | WorldPop, Meta, Poverty (all have defaults) | `01/` |
| **02** | Compare Meta vs WorldPop, compute residuals | `01/harmonised_meta_worldpop.gpkg` | `02/` |
| **03a** | Spearman + regression, VIF, Moran's I | `02/harmonised_with_residual.gpkg` | `03a_regression/` |
| **03b** | Stratified + inequality amplification | `02/harmonised_with_residual.gpkg` | `03b_stratified/` |
| **03c** | Spatial regression (SLM, SEM) | `02/harmonised_with_residual.gpkg` | `03c_spatial_regression/` |
| **03d** | Bivariate map: Poverty × Residual (R) | `02/harmonised_with_residual.gpkg` | `03d_bivariate/` |

**Note:** 03a, 03b, and the R bivariate map require `poverty_mean` in the 02 output. Run 01 with poverty (default); use `--no-poverty` only if skipping poverty analysis.

## Run order

```bash
# 1. Harmonise (WorldPop + Meta + Poverty; all use default paths)
python scripts/01_harmonise_datasets.py

# 2. Compare Meta vs WorldPop (global total scaling: Meta scaled to match WorldPop totals)
python scripts/02_compare_meta_worldpop.py

# 3a. Regression analysis (--residual-var: scaled_log_ratio, scaled_relative, log_ratio, residual)
python scripts/03a_regression.py

# 3b. Stratified + inequality
python scripts/03b_stratified.py

# 3c. Spatial regression (SLM, SEM) — uses scaled_log_ratio; run when Moran's I on OLS residuals is significant
python scripts/03c_spatial_regression.py

# 3d. Bivariate map: Poverty × Residual (R)
Rscript scripts/03d_bivariate_map_poverty_residual.R   # --residual-var for alternatives
```

## Output organisation

All outputs are nested under their script number:

```
outputs/
├── 01/
│   ├── harmonised_meta_worldpop.gpkg
│   ├── 01_data_overview.png         # optional (--plot or Rscript 01_plot_descriptive.R)
│   ├── 01_bivariate_worldpop_meta.png
│   └── 01_bivariate_worldpop_meta_basemap.png   # optional (requires network for tiles)
├── 02/
│   ├── harmonised_with_residual.gpkg
│   ├── 02_distribution_histogram_kde.png
│   ├── 02_lorenz_curves.png
│   ├── 02_residual_raw.png
│   ├── 02_log_ratio.png
│   ├── 02_residual_relative.png
│   ├── 02_log_ratio_scaled.png
│   ├── 02_relative_scaled.png
│   └── Table1_meta_worldpop_metrics.csv
├── 03a_regression/
│   ├── Table2_regression.csv
│   ├── Table2b_VIF.csv
│   ├── Table2b_heteroskedasticity.csv
│   ├── Table2c_Moran.csv
│   ├── 03a_local_moran_map.png
│   ├── 03a_hotspot_map.png
│   ├── 03a_regression_gdf.gpkg
│   └── regression_coefficients.csv
├── 03b_stratified/
│   ├── Table_interaction.csv
│   ├── marginal_effects_poverty.png
│   ├── Table3_poverty_strata.csv
│   ├── Table4_gini_by_quintile.csv
│   ├── residual_by_poverty_strata.png
│   └── gini_by_poverty_quintile.png
├── 03c_spatial_regression/
│   ├── Table_model_comparison.csv    # OLS vs SLM vs SEM (AIC, R²)
│   ├── Table2d_spatial_regression.csv
│   ├── slm_residual_map.png
│   └── sem_residual_map.png
└── 03d_bivariate/
    ├── 03d_bivariate_poverty_residual.png       # Poverty × Residual
    └── 03d_bivariate_poverty_residual_basemap.png   # optional
```

## Optional

- **01_plot_descriptive.R** — Descriptive figures (data overview + bivariate, saved separately): `Rscript scripts/01_plot_descriptive.R` (or `python scripts/01_harmonise_datasets.py --plot`)
- **03d_bivariate_map_poverty_residual.R** — Poverty × Residual (from 02): `Rscript scripts/03d_bivariate_map_poverty_residual.R`
