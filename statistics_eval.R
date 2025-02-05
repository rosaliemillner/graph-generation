rm(list=ls())
#setwd("~/Desktop/MVA/ALTEGRAD/projet/stats")

library(ggplot2)
library(reshape2)
library(dplyr)

convert_to_numeric <- function(df) {
  char_cols <- sapply(df, is.character)
  df[char_cols] <- lapply(df[char_cols], as.numeric)
  return(df)
}

baseline_data8 <- read.csv("graph_8_properties_baseline1000.csv")
baseline_data8 <- baseline_data8[, -1]
baseline_data8 <- convert_to_numeric(baseline_data8)

modele_data8 <- read.csv("graph_8_properties_notremodele1000.csv")
modele_data8 <- modele_data8[, -1]
modele_data8 <- convert_to_numeric(modele_data8)

ground_truth_8 <- c(16,	92,	11.5,	251,	0.7590726017951965,	9,	3)


stats_baseline <- baseline_data8 %>%
  summarise(across(-1, list(mean = mean, sd = sd, min = min, max = max), na.rm = TRUE))

stats_modele <- modele_data8 %>%
  summarise(across(-1, list(mean = mean, sd = sd, min = min, max = max), na.rm = TRUE))

features <- colnames(baseline_data8)


# BOXPLOTS BOTH MODELS ######

for (i in seq_along(features)) {
  feature <- features[i]
  df_comparaison <- data.frame(
    Value = c(baseline_data8[[feature]], modele_data8[[feature]]),
    Dataset = rep(c("Baseline", "Modele"), each = 1000)
  )
  
  p <- ggplot(df_comparaison, aes(x = Dataset, y = Value, fill = Dataset)) +
    geom_boxplot() +
    geom_hline(yintercept = ground_truth_8[i], linetype = "dashed", color = "red") +
    ggtitle(paste("Boxplot comparatif de", feature)) +
    theme_minimal() +
    labs(y = feature) +
    theme(legend.position = "none")
  print(p)
}


# HISTOGRAM OUR MODEL ######

for (i in seq_along(features)) {
  feature <- features[i]
  df <- data.frame(
    Value = c(modele_data8[[feature]]),
    Dataset = "Modele"
  )
  
  p <- ggplot(df, aes(x = Value, fill = Dataset)) +
    geom_histogram(position = "dodge", bins = 20, alpha = 0.6) +
    geom_vline(xintercept = ground_truth_8[i], color = "red", linetype = "dashed", size = 1) +
    ggtitle(paste("Histogramme comparatif de", feature)) +
    theme_minimal() +
    labs(x = feature, y = "Frequency") +
    theme(legend.position = "none")
  print(p)
}