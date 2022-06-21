library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(recipeselectors)
library(doParallel)
library(patchwork)
library(ROCR)
library(vip)

dir.create(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/lasso_feature_selection', 
								sprintf('xgboost/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/lasso_feature_selection/xgboost/results', 
												sprintf('8269feat.csv'))

dataset_8269 = read_csv(
	here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/lasso_selected_data_kevin/lasso_select8269.csv.gz'))

this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_10000feat_auc.rds.gz'))
folds = read_rds(here(here('results/cv_folds/PRISM_LINCS_klaeger_folds_10000_auc.rds.gz')))
xgb_grid = read_rds(here('results/hyperparameter_grids/xgb_grid.rds'))

recipe_8269 = recipe(auc ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("auc"),
							new_role = "id variable") %>%
	step_select(depmap_id,
							ccle_name,
							auc,
							broad_id,
							any_of(names(dataset_8269)))

xgb_spec <- boost_tree(
	trees = tune(), 
	tree_depth = tune()
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist") %>% 
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(xgb_spec) %>%
	add_recipe(recipe_8269)

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
)

cv_metrics_regression = collect_metrics(results) %>% 
	write_csv(full_output_file)
