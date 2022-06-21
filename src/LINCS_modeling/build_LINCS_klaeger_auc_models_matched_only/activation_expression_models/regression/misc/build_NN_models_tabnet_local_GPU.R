library(torch)
library(tidyverse)
library(here)
library(tabnet)
library(vroom)
library(finetune)
library(tidymodels)
library(tictoc)
library(doParallel)
library(vip)

data = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 auc,
					 broad_id)
}
tabnet_grid = read_rds(here('results/hyperparameter_grids/tabnet_grid.rds'))

args = data.frame(feature_num = c(100,200,300,400,500,1000,1500,2000,3000,4000,5000))

for(i in 1:length(args$feature_num)) {
	tic()	
	print(sprintf('Features: %02d',args$feature_num[i]))
	
	dir.create(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/', 
									sprintf('tabnet/results')), 
						 showWarnings = F, recursive = T)
	
	full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/tabnet/results', 
													sprintf('%dfeattest.rds.gz',args$feature_num)[i])
	
	
	this_dataset = build_all_data_regression_viability_set(feature_correlations =  cors,
																												 num_features = args$feature_num[i],
																												 all_data = data)
	set.seed(2222)
	folds = vfold_cv(this_dataset, v = 1)
	
	this_recipe = recipe(auc ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("auc"),
								new_role = "id variable")

tabnet_spec <- tabnet(epochs = 10, decision_width = tune(), attention_width = tune(),
											num_steps = tune(), penalty = 0.000001, virtual_batch_size = 512, momentum = 0.6,
											feature_reusage = 1.5, learn_rate = tune()) %>%
	set_engine("torch", verbose = TRUE) %>%
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(tabnet_spec) %>%
	add_recipe(this_recipe)

race_ctrl = control_race(
	save_pred = TRUE,
	parallel_over = "everything",
	verbose = TRUE
)

set.seed(2222)
fit <- this_wflow %>% 
	tune_race_anova(
		resamples = folds, 
		grid = tabnet_grid,
		control = race_ctrl
	) %>% 
	write_rds(full_output_file, compress = "gz")
toc()
}