library(tidyverse)
library(here)
library(tidymodels)
library(recipeselectors)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

doParallel::registerDoParallel()

all_data_filtered = read_csv(here('results/all_model_data_filtered.csv'))

this_dataset = full_data %>% 
	slice(1:100) %>% 
	select(1:100)

folds = vfold_cv(this_dataset, v = 10)

normal_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							ic50_binary,
							new_role = "id variable")

boruta_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							ic50_binary,
							new_role = "id variable") %>% 
	step_select_boruta(all_predictors(), outcome = "ic50")

infgain_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							ic50_binary,
							new_role = "id variable") %>% 
	step_select_infgain(all_predictors(), outcome = "ic50", top_p = 10, threshold = 0.9)

mrmr_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							ic50_binary,
							new_role = "id variable") %>% 
	step_select_mrmr(all_predictors(), outcome = "ic50", top_p = 10, threshold = 0.9)

carscore_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							ic50_binary,
							new_role = "id variable") %>% 
	step_select_carscore(all_predictors(), outcome = "ic50", top_p = 10, threshold = 0.9)


get_cv_metrics_feature_selection = function(query_recipe, query_folds) {
xgb_spec <- boost_tree(
	trees = 500, 
	tree_depth = 13, min_n = 29, 
	loss_reduction = 0.02280748,                     ## first three: model complexity
	sample_size = 0.8467527, mtry = 148,         ## randomness
	learn_rate = 0.01328023,                         ## step size
) %>% 
	set_engine("xgboost") %>% 
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(xgb_spec) %>%
	add_recipe(query_recipe)

ctrl <- control_resamples(save_pred = TRUE)

fit <-
	this_wflow %>%
	fit_resamples(folds, control = ctrl)

cv_metrics_regression = collect_metrics(fit)




}

selection_list_name = c('normal_recipe', 'boruta_recipe', 'infgain_recipe', 'mrmr_recipe', 'carscore_recipe')
selection_list = c(normal_recipe, boruta_recipe, infgain_recipe, mrmr_recipe, carscore_recipe)

all_data_regression_metrics = data.frame()
for (i in 1:length(selection_list)) {
	this_metrics = get_cv_metrics_feature_selection(query_recipe = selection_list[i], query_folds = folds) %>%
		mutate(selection_type = selection_list_name[i])
	all_data_regression_metrics = bind_rows(all_data_regression_metrics, this_metrics)
}
