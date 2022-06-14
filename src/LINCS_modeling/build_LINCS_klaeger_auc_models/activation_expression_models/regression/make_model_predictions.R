library(tidyverse)
library(here)
library(tidymodels)
library(conflicted)
conflict_prefer("slice", "dplyr")
conflict_prefer("filter", "dplyr")

not_tested_data = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/not_tested_data_xgboost.rds.gz'))
results = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/xgboost/final_xgboost_model.rds.gz'))

model_predictions = augment(results, not_tested_data %>% 
															mutate(ccle_name = NA, broad_id = NA)) %>% 
	select(-starts_with("act_"), -starts_with("exp_"))

sample_info = read_csv(here('data/CCLE_data/sample_info.csv.gz')) %>%
	mutate(cell_line_name_extra = paste0(cell_line_name, "\n",lineage_subtype, "\n",lineage_sub_subtype))

model_predictions_tidy = model_predictions %>%
	mutate(pred_auc = signif(.pred,3)) %>%
	left_join(sample_info %>% select(DepMap_ID, stripped_cell_line_name),
						by=c('depmap_id'='DepMap_ID')) %>% 
	select(-ccle_name, -.pred) %>% 
	write_csv(here('results/PRISM_LINCS_klaeger_models_auc/final_model_predictions_xgboost.csv'))