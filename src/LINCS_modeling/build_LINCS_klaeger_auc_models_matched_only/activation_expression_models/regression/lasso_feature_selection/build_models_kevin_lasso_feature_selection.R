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

dataset_2124 = read_csv(
	here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/lasso_selected_data_kevin/lasso_select2124.csv'))

dataset_8269 = read_csv(
	here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/lasso_selected_data_kevin/lasso_select8269.csv'))

dataset_9269 = read_csv(
	here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/lasso_selected_data_kevin/lasso_select9269.csv'))

this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_10000feat_auc.rds.gz'))
folds = read_rds(here(here('results/cv_folds/PRISM_LINCS_klaeger_folds_10000_auc.rds.gz')))
xgb_grid = read_rds(here('results/hyperparameter_grids/xgb_grid.rds'))

recipe_2124 = recipe(auc ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("auc"),
								new_role = "id variable") %>%
		step_select(depmap_id,
								ccle_name,
								auc,
								broad_id,
								any_of(names(dataset_2124)))

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

recipe_9269 = recipe(auc ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("auc"),
							new_role = "id variable") %>%
	step_select(depmap_id,
							ccle_name,
							auc,
							broad_id,
							any_of(names(dataset_9269)))

xgb_spec <- boost_tree(
	trees = tune(), 
	tree_depth = tune()
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist") %>% 
	set_mode("regression")

complete_workflowset = workflow_set(
	preproc = list(feat_2124 = recipe_2124,
								 feat_8269 = recipe_8269,
								 feat_9269 = recipe_9269),
	models = list(xgb = xgb_spec),
	cross = TRUE
)

complete_workflowset = complete_workflowset %>% 
	option_add(grid = xgb_grid, id = "feat_2124_xgb") %>%
	option_add(grid = xgb_grid, id = "feat_8269_xgb") %>%
	option_add(grid = xgb_grid, id = "feat_9269_xgb")

# this_wflow <-
# 	workflow() %>%
# 	add_model(xgb_spec) %>%
# 	add_recipe(recipe_2124)
# 
# race_ctrl = control_grid(
# 	save_pred = TRUE,
# 	parallel_over = "everything",
# 	verbose = TRUE
# )
# 
# results <- tune_grid(
# 	this_wflow,
# 	resamples = folds,
# 	grid = xgb_grid,
# 	control = race_ctrl
# )
# collect_metrics(results)
# 
# temp = results$.notes[[1]]
# temp$note[1]

race_ctrl = control_grid(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

set.seed(2222)
all_results = complete_workflowset %>% 
	workflow_map(
		"tune_grid",
		seed = 2222,
		resamples = folds,
		control = race_ctrl
	)

# temp = all_results$result
# autoplot(all_results))

cv_metrics_regression = collect_metrics(all_results)

# write_rds(
# 	all_results,
# 	here(
# 		'results/PRISM_klaeger_only_xgb_rf_NN_models_regression_results_lasso_feature_selection.rds.gz'),
# 	compress = "gz")

write_csv(
	cv_metrics_regression,
	here(
		'results/PRISM_LINCS_klaeger_xgb_models_regression_metrics_kevin_lasso_feature_selection.csv'))
