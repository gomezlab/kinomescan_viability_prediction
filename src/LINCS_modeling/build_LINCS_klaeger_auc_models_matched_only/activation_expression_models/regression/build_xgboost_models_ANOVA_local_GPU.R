library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)
library(argparse)
library(xgboost)

args = data.frame(feature_num = c(5000))

this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/matched_only_models/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/matched_only_models/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

folds = read_rds(here('results/cv_folds/matched_only_models/PRISM_LINCS_klaeger_folds_auc.rds.gz'))

for(i in 1:length(args$feature_num)) {
	tic()	
print(sprintf('Features: %02d',args$feature_num[i]))

dir.create(here('results/PRISM_LINCS_klaeger_models_auc/matched_only_models/activation_expression/regression/', 
								sprintf('xgboost/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/matched_only_models/activation_expression/regression/xgboost/results', 
												sprintf('%dfeat.rds.gz',args$feature_num))

this_recipe = recipe(auc ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("auc"),
							auc_binary,
							new_role = "id variable") %>%
	step_select(depmap_id,
							ccle_name,
							auc,
							broad_id,
							any_of(cors$feature[1:args$feature_num[i]])) %>% 
	step_normalize(all_predictors())

xgb_spec <- boost_tree(
	trees = tune(), 
	tree_depth = tune()                  
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist") %>% 
	set_mode("regression")

xgb_grid = read_rds(here('results/hyperparameter_grids/xgb_grid.rds'))

this_wflow <-
	workflow() %>%
	add_model(xgb_spec) %>%
	add_recipe(this_recipe) 

race_ctrl = control_grid(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

results <- tune_grid(
	this_wflow,
	resamples = folds,
	grid = xgb_grid,
	control = race_ctrl
) %>% 
	collect_metrics() %>% 
	write_rds(full_output_file, compress = "gz")

toc()
}
