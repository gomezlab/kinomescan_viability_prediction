library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

data = vroom(here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.csv'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 auc,
					 broad_id)
}

this_dataset = build_all_data_regression_viability_set(feature_correlations =  cors,
																											 num_features = 5001,
																											 all_data = data) %>% 
	write_rds(here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.rds'))

set.seed(2222)
folds = vfold_cv(this_dataset, v = 10) %>% 
	write_rds(here('results/PRISM_LINCS_klaeger_folds_auc.rds'))
