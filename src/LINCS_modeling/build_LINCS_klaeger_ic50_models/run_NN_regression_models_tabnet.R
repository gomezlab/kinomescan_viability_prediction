library(tidyverse)
library(here)
library(tidymodels)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

doParallel::registerDoParallel()

all_data_filtered = read_csv(here('results/all_model_data_filtered.csv'))
all_data_feat_cors =  read_csv(here('results/all_filtered_data_feature_correlations.csv'))

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 ic50,
					 broad_id,
					 ic50)
}

features = 100
data = all_data_filtered

get_all_data_regression_cv_metrics = function(features, data) {
	this_dataset = build_all_data_regression_viability_set(feature_cor =  all_data_feat_cors,
																												 num_features = features,
																												 all_data = data)
	
	folds = vfold_cv(this_dataset, v = 10)
	
	this_recipe = recipe(ic50 ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("ic50"),
								new_role = "id variable") %>% 
		step_zv(all_predictors()) %>% 
		step_normalize(all_predictors())
	
	tabnet_spec <- tabnet(epochs = 100, batch_size = 16384, decision_width = 24, attention_width = 26,
												num_steps = 5, penalty = 0.000001, virtual_batch_size = 512, momentum = 0.6,
												feature_reusage = 1.5, learn_rate = 0.02) %>%
		set_engine("torch", verbose = TRUE) %>% 
		set_mode("regression") 
	
	this_wflow <-
		workflow() %>%
		add_model(tabnet_spec) %>%
		add_recipe(this_recipe)
	
	ctrl <- control_resamples(save_pred = TRUE)
	
	fit <-
		this_wflow %>% 
		fit_resamples(folds, control = ctrl)
	
	cv_metrics_regression = collect_metrics(fit)
	
	return(cv_metrics_regression)
}

feature_list = seq(1000, 10000, by = 1000)

all_data_regression_metrics = data.frame()
for (i in 1:length(feature_list)) {
	this_metrics = get_all_data_regression_cv_metrics(features = feature_list[i], data = all_data_filtered) %>%
		mutate(feature_number = feature_list[i])
	all_data_regression_metrics = bind_rows(all_data_regression_metrics, this_metrics)
}

write_csv(all_data_regression_metrics, here('results/klaeger_LINCS_NN_tabnet_regression_results.csv'))
