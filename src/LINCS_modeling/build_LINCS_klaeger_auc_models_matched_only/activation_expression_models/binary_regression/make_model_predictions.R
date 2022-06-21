library(tidyverse)
library(here)
library(tidymodels)
library(ROCR)
library(patchwork)
library(tictoc)
library(broom)
library(gghighlight)
library(Metrics)
library(vroom)
library(conflicted)
library(vip)
conflict_prefer("slice", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("vi", "vip")
conflict_prefer("explain", "fastshap")

data = vroom(here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.csv'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))
LINCS_klaeger_data_wide = read_csv(here('results/all_klaeger_LINCS_data_for_ml_long.csv')) %>% 
	select(-binary_hit) %>% 
	mutate(kinase = paste0("act_",kinase)) %>% 
	pivot_wider(names_from = kinase, values_from = relative_intensity) %>% 
	mutate_all(~replace(., is.na(.), 1)) 
CCLE_data = read_rds(here('data/full_CCLE_expression_set_for_ML.rds')) %>% 
	rename('depmap_id' = DepMap_ID)
PRISM_auc = read_csv(here('results/PRISM_auc_for_ml.csv'))
compound_match_list = read_csv(here('results/matching/LINCS_PRISM_klaeger_combined_drug_matches_final.csv')) %>% 
	mutate(drug = if_else(
		is.na(klaeger_name),
		LINCS_name, 
		klaeger_name
	)) 
results = read_rds(here('results/final_tuned_PRISM_LINCS_klaeger_auc_regression_model.rds'))

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 broad_id,
					 auc)
}
this_dataset = build_all_data_regression_viability_set(feature_correlations =  cors,
																											 num_features = 3000,
																											 all_data = data)

possible_drug_CCLE_combos = crossing(
	drug = unique(LINCS_klaeger_data_wide$drug),
	depmap_id = unique(CCLE_data$depmap_id)
) 

already_tested_combos = PRISM_auc %>%
	select(depmap_id,name) %>%
	rename("PRISM_name" = name) %>% 
	left_join(compound_match_list %>% 
							select(drug, PRISM_name), 
						by = 'PRISM_name') %>% 
	unique()

non_tested_combos = possible_drug_CCLE_combos %>%
	anti_join(already_tested_combos)

not_tested_data = non_tested_combos %>%
	left_join(LINCS_klaeger_data_wide %>% 
							select(drug,
										 any_of(names(this_dataset))),
						by = c('drug')) %>% 
	left_join(CCLE_data %>% 
							select(depmap_id,
										 any_of(names(this_dataset))),
						by = 'depmap_id')

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
	write_rds(here('results/final_auc_regression_model_predictions.rds'), compress = 'gz')