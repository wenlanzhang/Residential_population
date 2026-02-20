# Residential Population Analysis

Analysis pipeline comparing **Meta** and **WorldPop** residential population estimates, with **poverty (MPI)** as an explanatory variable for digital representation bias.

## Overview

This project harmonises Meta Facebook baseline and WorldPop population rasters to a common quadkey grid, compares their spatial distributions, and investigates how poverty relates to residual bias (Meta underrepresentation relative to WorldPop). The pipeline includes:

- Harmonisation to the Meta quadkey grid
- Spatial pattern agreement (Spearman) and distribution similarity tests
- Inequality metrics (Gini, Lorenz curves)
- OLS regression with Poverty, Distance, and Log Population Density
- Spatial regression (SLM, SEM) when Moran's I indicates autocorrelation
- Bivariate maps (Poverty × Residual)

## Prerequisites

- **Python 3.9+** (with conda recommended: `conda activate geo_env_LLM`)
- **R 4.0+** (for descriptive plots and bivariate maps)

### Python packages

```bash
pip install -r requirements.txt
```

Main dependencies: geopandas, rasterio, rasterstats, pandas, numpy, scipy, matplotlib, statsmodels, libpysal, esda, spreg.

### R packages

```r
install.packages(c("sf", "ggplot2", "dplyr", "patchwork", "biscale", "cowplot"))
```

## Quick Start

Run from the project root. Scripts use default paths; `-i` can be omitted for standard runs.

```bash
# 1. Harmonise (WorldPop + Meta + Poverty)
python scripts/01_harmonise_datasets.py

# 2. Compare Meta vs WorldPop (residuals, Table 1 & 2)
python scripts/02_compare_meta_worldpop.py

# 3a. Regression (Spearman, OLS, VIF, Moran's I)
python scripts/03a_regression.py

# 3b. Stratified analysis + inequality
python scripts/03b_stratified.py

# 3c. Spatial regression (SLM, SEM)
python scripts/03c_spatial_regression.py

# 3d. Bivariate map: Poverty × Residual
Rscript scripts/03d_bivariate_map_poverty_residual.R
```

**Note:** Scripts 03a, 03b, and 03d require `poverty_mean` in the 02 output. Run 01 with poverty (default); use `--no-poverty` only if skipping poverty analysis.

## Pipeline

| Script | Purpose | Input (default) | Output |
|--------|---------|-----------------|--------|
| **01** | Harmonise + descriptive maps | WorldPop, Meta, Poverty rasters | `outputs/01/` |
| **02** | Compare Meta vs WorldPop, residuals | `01/harmonised_meta_worldpop.gpkg` | `outputs/02/` |
| **03a** | Spearman, OLS, VIF, Moran's I | `02/harmonised_with_residual.gpkg` | `outputs/03a_regression/` |
| **03b** | Stratified + inequality amplification | `02/harmonised_with_residual.gpkg` | `outputs/03b_stratified/` |
| **03c** | Spatial regression (SLM, SEM) | `02/harmonised_with_residual.gpkg` | `outputs/03c_spatial_regression/` |
| **03d** | Bivariate map: Poverty × Residual (R) | `02/harmonised_with_residual.gpkg` | `outputs/03d_bivariate/` |

## Outputs

```
outputs/
├── 01/
│   ├── harmonised_meta_worldpop.gpkg
│   ├── 01_data_overview.png
│   ├── 01_bivariate_worldpop_meta.png
│   └── 01_bivariate_worldpop_meta_basemap.png
├── 02/
│   ├── harmonised_with_residual.gpkg
│   ├── Table1_meta_worldpop_metrics.csv
│   ├── Table2_OLS_regression.csv          # OLS (Poverty, Distance, Log Pop Density)
│   ├── 02_distribution_histogram_kde.png
│   ├── 02_lorenz_curves.png
│   ├── 02_log_ratio_scaled.png
│   └── ...
├── 03a_regression/
│   ├── Table2_regression.csv
│   ├── Table2b_VIF.csv
│   ├── Table2b_heteroskedasticity.csv
│   ├── Table2c_Moran.csv
│   ├── 03a_local_moran_map.png
│   └── 03a_hotspot_map.png
├── 03b_stratified/
│   ├── Table_interaction.csv
│   ├── Table3_poverty_strata.csv
│   ├── Table4_gini_by_quintile.csv
│   └── ...
├── 03c_spatial_regression/
│   ├── Table_model_comparison.csv         # OLS vs SLM vs SEM (AIC, R²)
│   ├── Table_SLM_SEM_coefficients.csv     # SLM & SEM coefficients
│   ├── Table_AIC_comparison.csv           # Model | AIC
│   ├── Table2d_spatial_regression.csv
│   ├── slm_residual_map.png
│   └── sem_residual_map.png
└── 03d_bivariate/
    ├── 03d_bivariate_poverty_residual.png
    └── 03d_bivariate_poverty_residual_basemap.png
```

## Optional Scripts

- **01_plot_descriptive.R** — Descriptive figures (data overview + bivariate): `Rscript scripts/01_plot_descriptive.R` or `python scripts/01_harmonise_datasets.py --plot`
- **03d_bivariate_map_poverty_residual.R** — Poverty × Residual bivariate map: `Rscript scripts/03d_bivariate_map_poverty_residual.R`

## Data Defaults

Scripts use default paths (e.g. for Nairobi/Kenya). Override with CLI arguments:

```bash
python scripts/01_harmonise_datasets.py --worldpop /path/to.tif --meta /path/to.gpkg --poverty /path/to/poverty.tif
python scripts/01_harmonise_datasets.py --filter-by both --filter-min 50   # keep quadkeys where both Meta & WorldPop ≥ 50
```

## License

See project license file.
