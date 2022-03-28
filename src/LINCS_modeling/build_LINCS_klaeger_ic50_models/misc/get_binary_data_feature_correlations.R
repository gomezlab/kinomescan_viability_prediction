library(tidyverse)
library(here)
library(vroom)

binary_data = read_rds(here('results/PRISM_LINCS_klaeger_binary_data_for_ml_auc.rds.gz'))

find_feature_correlations <- function(row_indexes = NA, all_data) {
  if (is.na(row_indexes)) {
    row_indexes = 1:dim(all_data)[1]
  }
  
  all_cor = cor(
    all_data %>% 
      pull(auc),
    
    all_data %>% 
      select(starts_with(c('exp')))
  ) %>%
    as.data.frame() %>%
    pivot_longer(everything(), names_to = "feature",values_to = "cor")
  
  
  all_correlations = all_cor %>% 
    mutate(abs_cor = abs(cor)) %>% 
    arrange(desc(abs_cor)) %>% 
    mutate(rank = 1:n()) %>%
    mutate(feature_type = case_when(
      str_detect(feature, "^exp_") ~ "Expression",
      T ~ feature
    ))
  
  return(all_correlations)	
}

binary_feat_cors = find_feature_correlations(all_data = binary_data)

variable_kinases = binary_data %>% 
	select(starts_with('act')) %>% 
	pivot_longer(starts_with('act'), names_to = "kinase", values_to = "binary_hit") %>% 
	select(kinase, binary_hit) %>% 
	mutate(binary_hit = as.numeric(binary_hit)) %>% 
	group_by(kinase) %>% 
	summarise(var = var(binary_hit)) %>% 
	filter(var > 0)

write_csv(binary_feat_cors, here('results/PRISM_LINCS_klaeger_binary_data_feature_correlations_auc.csv'))
write_csv(variable_kinases, here('results/PRISM_LINCS_klaeger_binary_data_variable_kinases_auc.csv'))
