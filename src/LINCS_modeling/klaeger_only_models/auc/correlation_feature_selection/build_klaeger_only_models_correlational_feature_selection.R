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

this_dataset = read_rds(here('results/klaeger_only_models_auc/PRISM_klaeger_only_data_5000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

folds = read_rds(here(here('results/cv_folds/PRISM_klaeger_only_folds_auc.rds.gz')))

get_recipe = function(data, feature_number, feature_correlations) {
	this_recipe = recipe(auc_target ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("auc_target"),
								new_role = "id variable") %>%
		#		step_normalize(all_predictors()) %>%
		step_select(depmap_id,
								ccle_name,
								auc_target,
								broad_id,
								any_of(feature_correlations$feature[1:feature_number]))
	return(this_recipe)
}

recipe_100 = get_recipe(data = this_dataset, feature_number = 100, feature_correlations = cors)
recipe_200 = get_recipe(data = this_dataset, feature_number = 200, feature_correlations = cors)
recipe_300 = get_recipe(data = this_dataset, feature_number = 300, feature_correlations = cors)
recipe_400 = get_recipe(data = this_dataset, feature_number = 400, feature_correlations = cors)
recipe_500 = get_recipe(data = this_dataset, feature_number = 500, feature_correlations = cors)
recipe_1000 = get_recipe(data = this_dataset, feature_number = 1000, feature_correlations = cors)
recipe_1500 = get_recipe(data = this_dataset, feature_number = 1500, feature_correlations = cors)
recipe_2000 = get_recipe(data = this_dataset, feature_number = 2000, feature_correlations = cors)
recipe_3000 = get_recipe(data = this_dataset, feature_number = 3000, feature_correlations = cors)
recipe_4000 = get_recipe(data = this_dataset, feature_number = 4000, feature_correlations = cors)
recipe_5000 = get_recipe(data = this_dataset, feature_number = 5000, feature_correlations = cors)


xgb_spec <- boost_tree(
	trees = 168, 
	tree_depth = 5
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist") %>% 
	set_mode("regression")

# rf_spec <- rand_forest(
# 	trees = 2000
# ) %>% set_engine("ranger") %>%
# 	set_mode("regression")

keras_spec <- mlp(
	hidden_units = 1000, 
	penalty = 0.00005,
	epochs = 10                  
) %>% 
	set_engine("keras") %>% 
	set_mode("regression")

complete_workflowset = workflow_set(
	preproc = list(feat100 = recipe_100,
								 feat200 = recipe_200,
								 feat300 = recipe_300,
								 feat400 = recipe_400,
								 feat500 = recipe_500,
								 feat1000 = recipe_1000,
								 feat1500 = recipe_1500,
								 feat2000 = recipe_2000,
								 feat3000 = recipe_3000,
								 feat4000 = recipe_4000,
								 feat5000 = recipe_5000
	),
	models = list(
								xgb = xgb_spec,
								keras = keras_spec),
	cross = TRUE
)
#rf = rf_spec,
# this_wflow <-
# 	workflow() %>%
# 	add_model(keras_spec) %>%
# 	add_recipe(recipe_500)
# 
# race_ctrl = control_resamples(
# 	save_pred = TRUE,
# 	parallel_over = "everything",
# 	verbose = TRUE
# )
# 
# results <- fit_resamples(
# 	this_wflow,
# 	resamples = folds,
# 	control = race_ctrl
# )

# collect_metrics(results)

# temp = results$.notes[[1]]
# temp$note[1]

race_ctrl = control_resamples(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

all_results = complete_workflowset %>% 
	workflow_map(
		"fit_resamples",
		seed = 2222,
		resamples = folds,
		control = race_ctrl
	)

write_rds(
	all_results,
	here(
		'results/PRISM_klaeger_only_xgb_rf_NN_models_regression_results.rds.gz'),
	compress = "gz")

cv_metrics_regression = collect_metrics(all_results)

write_csv(
	cv_metrics_regression,
	here(
		'results/PRISM_klaeger_only_xgb_rf_NN_models_regression_metrics.csv'))
