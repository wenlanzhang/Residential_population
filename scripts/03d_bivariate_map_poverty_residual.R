#!/usr/bin/env Rscript
# 03d — Bivariate map: Poverty × Residual (Digital invisibility hotspots)
#
# Poverty (MPI): high = more poverty
# Residual = log(Meta) - log(WorldPop): negative = Meta underrepresents
# Key quadrant: High poverty (3) + Negative residual (1) = 3-1 = Digital invisibility
#
# Reads directly from script 02 output. No Python step needed.
# Usage: Rscript scripts/03d_bivariate_map_poverty_residual.R
#        Rscript scripts/03d_bivariate_map_poverty_residual.R -i outputs/02/harmonised_with_residual.gpkg

suppressPackageStartupMessages({
  library(sf)
  library(ggplot2)
  library(dplyr)
})

# Z-score classify for 3×3 bivariate: 1=low, 2=medium, 3=high
classify_z <- function(z, t = 0.5) {
  out <- rep(2L, length(z))
  out[z < -t] <- 1L
  out[z > t] <- 3L
  out
}

project_root <- getwd()
input_path <- file.path(project_root, "outputs", "02", "harmonised_with_residual.gpkg")
out_dir <- file.path(project_root, "outputs", "03d_bivariate")
output_path <- file.path(out_dir, "03d_bivariate_poverty_residual.png")
output_path_basemap <- file.path(out_dir, "03d_bivariate_poverty_residual_basemap.png")

# Residual metric: log_ratio, scaled_log_ratio, scaled_relative, or residual
residual_var <- "scaled_log_ratio"
args <- commandArgs(trailingOnly = TRUE)
for (i in seq_along(args)) {
  if (args[i] == "-i" && i < length(args)) {
    input_path <- args[i + 1]
  } else if (args[i] == "--residual-var" && i < length(args)) {
    residual_var <- args[i + 1]
  }
}

if (!file.exists(input_path)) {
  stop("Run script 02 first. Missing: ", input_path)
}

gdf <- st_read(input_path, quiet = TRUE)
if (!"poverty_mean" %in% names(gdf)) {
  stop("Input must include poverty_mean. Run script 01 with --poverty, then script 02.")
}
# Prefer scaled_log_ratio or scaled_relative; fallback to log_ratio or residual
if (!residual_var %in% names(gdf)) {
  residual_var <- if ("scaled_log_ratio" %in% names(gdf)) "scaled_log_ratio"
  else if ("scaled_relative" %in% names(gdf)) "scaled_relative"
  else if ("log_ratio" %in% names(gdf)) "log_ratio"
  else "residual"
}
if (!residual_var %in% names(gdf)) {
  stop("Input must include residual, log_ratio, scaled_log_ratio, or scaled_relative. Run script 02.")
}

# Legend labels for residual metric
residual_labels <- c(
  residual = "Residual: log(Meta) − log(WorldPop)",
  log_ratio = "Log-ratio: log(Meta) − log(WorldPop)",
  scaled_log_ratio = "Scaled log-ratio (z-score)",
  scaled_relative = "Scaled relative: (Meta−WP)/WP (z-score)"
)
residual_legend <- residual_labels[residual_var]

# Filter to valid quadkeys
gdf <- gdf %>%
  filter(!is.na(.data[[residual_var]]), !is.na(poverty_mean))
if ("poverty_n_pixels" %in% names(gdf)) {
  gdf <- gdf %>% filter(poverty_n_pixels > 0)
}

# 3×3 bivariate classification (z-score with threshold 0.5)
gdf <- gdf %>%
  mutate(
    poverty_z = (poverty_mean - mean(poverty_mean, na.rm = TRUE)) / sd(poverty_mean, na.rm = TRUE),
    residual_z = (.data[[residual_var]] - mean(.data[[residual_var]], na.rm = TRUE)) / sd(.data[[residual_var]], na.rm = TRUE),
    poverty_z = ifelse(is.na(poverty_z), 0, poverty_z),
    residual_z = ifelse(is.na(residual_z), 0, residual_z),
    pov_class = classify_z(poverty_z),
    res_class = classify_z(residual_z),
    bi_class = paste0(pov_class, "-", res_class)
  )

message("Valid quadkeys: ", nrow(gdf), "; residual metric: ", residual_var)

# Ensure all 9 classes exist for palette
bivariate_palette <- c(
  "1-1" = "#e8e8e8", "2-1" = "#e4acac", "3-1" = "#c85a5a",
  "1-2" = "#b8d6be", "2-2" = "#ad9ea5", "3-2" = "#985356",
  "1-3" = "#64acbe", "2-3" = "#627f8c", "3-3" = "#574249"
)
gdf$bi_class <- factor(gdf$bi_class, levels = names(bivariate_palette))

use_biscale <- requireNamespace("biscale", quietly = TRUE)
use_cowplot <- requireNamespace("cowplot", quietly = TRUE)
use_ggspatial <- requireNamespace("ggspatial", quietly = TRUE)
use_maptiles <- requireNamespace("maptiles", quietly = TRUE)

subtitle_str <- paste0("Poverty (rows) | ", residual_legend, " (cols). 3-1 = Digital invisibility hotspot")

# Base map (no basemap)
if (use_biscale && use_cowplot) {
  map <- ggplot(gdf) +
    geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.2, show.legend = FALSE) +
    biscale::bi_scale_fill(pal = "DkBlue", dim = 3) +
    biscale::bi_theme() +
    labs(
      title = "Bivariate: Poverty × Residual",
      subtitle = subtitle_str
    )
  legend <- biscale::bi_legend(
    pal = "DkBlue", dim = 3,
    xlab = "Higher Poverty ", ylab = paste0("Higher ", residual_legend, " (overrep) "),
    size = 8
  )
  p <- cowplot::ggdraw() +
    cowplot::draw_plot(map, 0, 0, 1, 1) +
    cowplot::draw_plot(legend, 0.02, 0.02, 0.22, 0.22)
} else {
  p <- ggplot(gdf) +
    geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.2) +
    scale_fill_manual(
      values = bivariate_palette,
      na.value = "grey90",
      drop = FALSE,
      name = paste0("Poverty | ", residual_legend)
    ) +
    theme_void() +
    theme(
      legend.position = c(0.02, 0.02),
      legend.justification = c(0, 0),
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    labs(title = "Bivariate: Poverty × Residual", subtitle = subtitle_str)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(output_path, p, width = 10, height = 8, dpi = 150, bg = "white")
message("Saved: ", output_path)

# With Sentinel basemap
basemap_layer <- NULL
if (use_maptiles && requireNamespace("terra", quietly = TRUE)) {
  tryCatch({
    s2_provider <- maptiles::create_provider(
      name = "Sentinel2-EOX",
      url = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/webmercator/{z}/{x}/{y}.jpeg",
      citation = "Sentinel-2 cloudless by EOX"
    )
    tiles <- maptiles::get_tiles(gdf, provider = s2_provider, zoom = 12, crop = TRUE, cachedir = tempdir())
    basemap_layer <- ggspatial::annotation_spatial(tiles, alpha = 0.9)
  }, error = function(e) {
    tryCatch({
      tiles <- maptiles::get_tiles(gdf, provider = "Esri.WorldImagery", zoom = 12, crop = TRUE, cachedir = tempdir())
      basemap_layer <<- ggspatial::annotation_spatial(tiles, alpha = 0.9)
    }, error = function(e2) NULL)
  })
}
if (is.null(basemap_layer) && use_ggspatial) {
  basemap_layer <- ggspatial::annotation_map_tile(type = "osm", zoom = 12, alpha = 0.8, cachedir = tempdir())
}

if (use_ggspatial && !is.null(basemap_layer)) {
  if (use_biscale && use_cowplot) {
    map_bm <- ggplot(gdf) +
      basemap_layer +
      geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.3, alpha = 0.5, show.legend = FALSE) +
      biscale::bi_scale_fill(pal = "DkBlue", dim = 3) +
      biscale::bi_theme() +
      coord_sf(expand = FALSE) +
      labs(title = "Bivariate: Poverty × Residual (Sentinel basemap)", subtitle = subtitle_str)
    legend_bm <- biscale::bi_legend(pal = "DkBlue", dim = 3, xlab = "Higher Poverty ", ylab = paste0("Higher ", residual_legend, " "), size = 8)
    p_bm <- cowplot::ggdraw() +
      cowplot::draw_plot(map_bm, 0, 0, 1, 1) +
      cowplot::draw_plot(legend_bm, 0.02, 0.02, 0.22, 0.22)
  } else {
    map_bm <- ggplot(gdf) +
      basemap_layer +
      geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.3, alpha = 0.5) +
      scale_fill_manual(values = bivariate_palette, na.value = "grey90", drop = FALSE, name = paste0("Poverty | ", residual_legend)) +
      theme_void() +
      theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0)) +
      coord_sf(expand = FALSE) +
      labs(title = "Bivariate: Poverty × Residual (basemap)", subtitle = subtitle_str)
    p_bm <- map_bm
  }
  ggsave(output_path_basemap, p_bm, width = 10, height = 8, dpi = 150, bg = "white")
  message("Saved: ", output_path_basemap)
}
