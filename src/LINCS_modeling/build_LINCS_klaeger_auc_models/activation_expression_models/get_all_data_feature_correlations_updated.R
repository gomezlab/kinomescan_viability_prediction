library(tidyverse)
library(here)
library(vroom)

data = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_auc.rds.gz'))


find_feature_correlations <- function(row_indexes = NA, all_data) {
  if (is.na(row_indexes)) {
    row_indexes = 1:dim(all_data)[1]
  }

  feature_cols <- grep("^act_|^exp_", colnames(all_data), value = TRUE)
  
  all_cor <- sapply(feature_cols, function(feature) {
    pearson_test <- cor.test(all_data[[feature]], all_data$auc, method = "pearson")
    spearman_test <- cor.test(all_data[[feature]], all_data$auc, method = "spearman")
    
    return(c(pearson_test$estimate, pearson_test$p.value, spearman_test$estimate, spearman_test$p.value))
  })

  all_cor <- t(all_cor)
  colnames(all_cor) <- c("pearson_r", "pearson_p", "spearman_r", "spearman_p")

  all_cor <- as.data.frame(all_cor)
  all_cor$feature <- rownames(all_cor)
  all_cor <- all_cor %>% 
    mutate(abs_pearson_r = abs(pearson_r), abs_spearman_r = abs(spearman_r)) %>% 
    arrange(desc(abs_pearson_r), desc(abs_spearman_r)) %>% 
    mutate(rank = 1:n()) %>%
    mutate(feature_type = case_when(
      str_detect(feature, "^act_") ~ "Activation",
      str_detect(feature, "^exp_") ~ "Expression",
      T ~ feature
    ))

  return(all_cor)	
}

feat_cors = find_feature_correlations(all_data = data)

write_csv(feat_cors, here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc_updated.csv'))