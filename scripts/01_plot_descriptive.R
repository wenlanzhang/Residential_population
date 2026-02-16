#!/usr/bin/env Rscript
# 01 — Descriptive figures: data overview + bivariate map (saved separately)
#
# Figure 1: Meta | WorldPop | Poverty (3-panel)
# Figure 2: Bivariate WorldPop vs Meta (z-score classification)
#
# Requires: harmonised data from 01_harmonise_datasets.py
# Usage: Rscript scripts/01_plot_descriptive.R
#        Rscript scripts/01_plot_descriptive.R -i outputs/01/harmonised_meta_worldpop.gpkg
#        Rscript scripts/01_plot_descriptive.R --threshold 0.5   # bivariate z-score threshold
#
# Optional: install.packages(c("patchwork", "biscale", "cowplot"))

suppressPackageStartupMessages({
  library(sf)
  library(ggplot2)
  library(dplyr)
})
use_patchwork <- requireNamespace("patchwork", quietly = TRUE)
if (use_patchwork) library(patchwork)

project_root <- getwd()
input_path <- file.path(project_root, "outputs", "01", "harmonised_meta_worldpop.gpkg")
out_dir <- file.path(project_root, "outputs", "01")
output_data_overview <- file.path(out_dir, "01_data_overview.png")
output_bivariate <- file.path(out_dir, "01_bivariate_worldpop_meta.png")
output_bivariate_basemap <- file.path(out_dir, "01_bivariate_worldpop_meta_basemap.png")
threshold <- 0.5

args <- commandArgs(trailingOnly = TRUE)
i <- 1
while (i <= length(args)) {
  if (args[i] == "-i" && i < length(args)) {
    input_path <- args[i + 1]
    out_dir <- dirname(input_path)
    output_data_overview <- file.path(out_dir, "01_data_overview.png")
    output_bivariate <- file.path(out_dir, "01_bivariate_worldpop_meta.png")
    output_bivariate_basemap <- file.path(out_dir, "01_bivariate_worldpop_meta_basemap.png")
    i <- i + 2
  } else if (args[i] == "--threshold" && i < length(args)) {
    threshold <- as.numeric(args[i + 1])
    if (is.na(threshold) || threshold <= 0) stop("--threshold must be positive (e.g. 0.5 or 1)")
    i <- i + 2
  } else {
    i <- i + 1
  }
}

if (!file.exists(input_path)) {
  stop("Run script 01 first. Missing: ", input_path)
}

gdf <- st_read(input_path, quiet = TRUE)

# -----------------------------------------------------------------------------
# Row 1: Data overview (Meta, WorldPop, Poverty)
# -----------------------------------------------------------------------------
theme_map <- function() {
  theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_text(size = 8, face = "plain"),
      legend.text = element_text(size = 7),
      plot.margin = margin(2, 2, 2, 2, "pt"),
      panel.border = element_rect(fill = NA, color = "grey70", linewidth = 0.5)
    )
}
guide_horizontal <- guide_colorbar(
  barwidth = unit(8, "lines"),
  barheight = unit(0.35, "cm"),
  title.position = "top"
)

p_meta <- ggplot(gdf) +
  geom_sf(aes(fill = meta_baseline), color = "white", linewidth = 0.12) +
  scale_fill_viridis_c(option = "plasma", na.value = "grey95", name = "Count",
    trans = "log1p", labels = scales::comma, guide = guide_horizontal) +
  labs(title = "Meta (midnight baseline)") +
  theme_map()

p_wp <- ggplot(gdf) +
  geom_sf(aes(fill = worldpop_count), color = "white", linewidth = 0.12) +
  scale_fill_viridis_c(option = "viridis", na.value = "grey95", name = "Count",
    trans = "log1p", labels = scales::comma, guide = guide_horizontal) +
  labs(title = "WorldPop (sum)") +
  theme_map()

has_poverty <- "poverty_mean" %in% names(gdf)
if (has_poverty) {
  p_pov <- ggplot(gdf) +
    geom_sf(aes(fill = poverty_mean), color = "white", linewidth = 0.12) +
    scale_fill_viridis_c(option = "viridis", na.value = "grey95", name = "MPI proportion",
      limits = c(0, NA), guide = guide_horizontal) +
    labs(title = "Poverty (MPI mean)") +
    theme_map()
  row1 <- p_meta + p_wp + p_pov
} else {
  row1 <- p_meta + p_wp
}

# -----------------------------------------------------------------------------
# Row 2: Bivariate WorldPop vs Meta
# -----------------------------------------------------------------------------
gdf_bi <- gdf %>% filter(worldpop_count > 0, meta_baseline > 0)
z_wp <- (gdf_bi$worldpop_count - mean(gdf_bi$worldpop_count)) / sd(gdf_bi$worldpop_count)
z_meta <- (gdf_bi$meta_baseline - mean(gdf_bi$meta_baseline)) / sd(gdf_bi$meta_baseline)
classify_z <- function(z, t) {
  dplyr::case_when(z < -t ~ 1L, z > t ~ 3L, TRUE ~ 2L)
}
gdf_bi <- gdf_bi %>%
  mutate(
    wp_class = classify_z(z_wp, threshold),
    meta_class = classify_z(z_meta, threshold),
    bi_class = paste0(wp_class, "-", meta_class)
  )

bivariate_palette <- c(
  "1-1" = "#e8e8e8", "2-1" = "#e4acac", "3-1" = "#c85a5a",
  "1-2" = "#b8d6be", "2-2" = "#ad9ea5", "3-2" = "#985356",
  "1-3" = "#64acbe", "2-3" = "#627f8c", "3-3" = "#574249"
)
gdf_bi$bi_class <- factor(gdf_bi$bi_class, levels = names(bivariate_palette))

use_biscale <- requireNamespace("biscale", quietly = TRUE)
use_cowplot <- requireNamespace("cowplot", quietly = TRUE)

if (use_biscale && use_cowplot) {
  map_bi <- ggplot(gdf_bi) +
    geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.2, show.legend = FALSE) +
    biscale::bi_scale_fill(pal = "DkBlue", dim = 3) +
    biscale::bi_theme() +
    labs(title = "Bivariate: WorldPop vs Meta (z-score)", subtitle = sprintf("Threshold ±%.1f", threshold))
  legend_bi <- biscale::bi_legend(pal = "DkBlue", dim = 3, xlab = "Higher WorldPop ", ylab = "Higher Meta ", size = 8)
  p_bi <- cowplot::ggdraw() +
    cowplot::draw_plot(map_bi, 0, 0, 1, 1) +
    cowplot::draw_plot(legend_bi, 0.15, 0.05, 0.22, 0.22)
} else if (use_biscale) {
  p_bi <- ggplot(gdf_bi) +
    geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.2) +
    biscale::bi_scale_fill(pal = "DkBlue", dim = 3) +
    biscale::bi_theme() +
    theme(legend.position = "right") +
    labs(title = "Bivariate: WorldPop vs Meta (z-score)", subtitle = sprintf("Threshold ±%.1f", threshold))
} else {
  p_bi <- ggplot(gdf_bi) +
    geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.2) +
    scale_fill_manual(values = bivariate_palette, na.value = "grey90", drop = FALSE,
      name = sprintf("Z-score (t=±%.1f)\nWorldPop | Meta", threshold)) +
    theme_void() +
    theme(
      legend.position = c(0.02, 0.02),
      legend.justification = c(0, 0),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 10)
    ) +
    labs(title = "Bivariate: WorldPop vs Meta", subtitle = sprintf("Threshold ±%.1f", threshold))
}

# -----------------------------------------------------------------------------
# Save separate figures
# -----------------------------------------------------------------------------
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Figure 1: Data overview (Meta, WorldPop, Poverty)
if (use_patchwork) {
  p_overview <- row1 + patchwork::plot_layout(ncol = if (has_poverty) 3 else 2) +
    patchwork::plot_annotation(title = "Harmonised datasets (quadkey grid)")
} else if (requireNamespace("gridExtra", quietly = TRUE)) {
  p_overview <- if (has_poverty) gridExtra::grid.arrange(p_meta, p_wp, p_pov, ncol = 3, top = "Harmonised datasets (quadkey grid)")
  else gridExtra::grid.arrange(p_meta, p_wp, ncol = 2, top = "Harmonised datasets (quadkey grid)")
} else {
  stop("Install patchwork or gridExtra: install.packages(c('patchwork','gridExtra'))")
}
if (use_patchwork) {
  ggsave(output_data_overview, p_overview, width = 12, height = 4, dpi = 200, bg = "white")
} else {
  png(output_data_overview, width = 12, height = 4, units = "in", res = 200, bg = "white")
  grid::grid.draw(p_overview)
  dev.off()
}
message("Saved: ", output_data_overview)

# Figure 2: Bivariate WorldPop vs Meta
ggsave(output_bivariate, p_bi, width = 10, height = 8, dpi = 150, bg = "white")
message("Saved: ", output_bivariate)

# Figure 3: Bivariate with Sentinel/satellite basemap
use_ggspatial <- requireNamespace("ggspatial", quietly = TRUE)
use_maptiles <- requireNamespace("maptiles", quietly = TRUE)
use_cowplot <- requireNamespace("cowplot", quietly = TRUE)
basemap_layer <- NULL
basemap_title_add <- " (with basemap)"
if (use_maptiles && requireNamespace("terra", quietly = TRUE)) {
  tryCatch({
    s2_provider <- maptiles::create_provider(
      name = "Sentinel2-EOX",
      url = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/webmercator/{z}/{x}/{y}.jpeg",
      citation = "Sentinel-2 cloudless by EOX (s2maps.eu)"
    )
    tiles_s2 <- maptiles::get_tiles(gdf_bi, provider = s2_provider, zoom = 12, crop = TRUE, cachedir = tempdir())
    basemap_layer <- ggspatial::annotation_spatial(tiles_s2, alpha = 0.9)
    basemap_title_add <- " (Sentinel-2 basemap)"
  }, error = function(e) {
    tryCatch({
      tiles_esri <- maptiles::get_tiles(gdf_bi, provider = "Esri.WorldImagery", zoom = 12, crop = TRUE, cachedir = tempdir())
      basemap_layer <<- ggspatial::annotation_spatial(tiles_esri, alpha = 0.9)
      basemap_title_add <<- " (satellite basemap)"
    }, error = function(e2) NULL)
  })
}
if (is.null(basemap_layer) && use_ggspatial) {
  basemap_layer <- ggspatial::annotation_map_tile(type = "osm", zoom = 12, alpha = 0.8, cachedir = tempdir())
  basemap_title_add <- " (with basemap)"
}
if (use_ggspatial && !is.null(basemap_layer)) {
  if (use_biscale && use_cowplot) {
    map_bm <- ggplot(gdf_bi) +
      basemap_layer +
      geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.3, alpha = 0.5, show.legend = FALSE) +
      biscale::bi_scale_fill(pal = "DkBlue", dim = 3) +
      biscale::bi_theme() +
      coord_sf(expand = FALSE) +
      labs(title = paste0("Bivariate: WorldPop vs Meta", basemap_title_add), subtitle = sprintf("Threshold ±%.1f", threshold))
    legend_bm <- biscale::bi_legend(pal = "DkBlue", dim = 3, xlab = "Higher WorldPop ", ylab = "Higher Meta ", size = 8)
    p_bm <- cowplot::ggdraw() +
      cowplot::draw_plot(map_bm, 0, 0, 1, 1) +
      cowplot::draw_plot(legend_bm, 0.05, 0.02, 0.22, 0.22)
  } else {
    map_bm <- ggplot(gdf_bi) +
      basemap_layer +
      geom_sf(aes(fill = bi_class), color = "white", linewidth = 0.3, alpha = 0.5) +
      scale_fill_manual(values = bivariate_palette, na.value = "grey90", drop = FALSE,
        name = sprintf("Z-score (t=±%.1f)\nWorldPop | Meta", threshold)) +
      theme_void() +
      theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0),
        plot.title = element_text(hjust = 0.5, face = "bold")) +
      coord_sf(expand = FALSE) +
      labs(title = paste0("Bivariate: WorldPop vs Meta", basemap_title_add), subtitle = sprintf("Threshold ±%.1f", threshold))
    p_bm <- map_bm
  }
  ggsave(output_bivariate_basemap, p_bm, width = 10, height = 8, dpi = 150, bg = "white")
  message("Saved: ", output_bivariate_basemap)
} else if (!use_ggspatial) {
  message("Install ggspatial for basemap version: install.packages('ggspatial')")
} else {
  message("Could not fetch basemap tiles (network?). Install maptiles: install.packages('maptiles')")
}
