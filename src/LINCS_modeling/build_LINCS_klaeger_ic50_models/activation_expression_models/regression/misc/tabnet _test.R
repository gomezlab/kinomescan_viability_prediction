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

data = vroom(here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat_ic50.csv'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations_ic50.csv'))
build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 ic50,
					 broad_id)
}
args = data.frame(feature_num = c(500))
i = 1
this_dataset = build_all_data_regression_viability_set(feature_correlations =  cors,
																											 num_features = args$feature_num[i],
																											 all_data = data)

split = initial_split(this_dataset, prop = 3/4)
train = training(split)
test = testing(split)

this_recipe = recipe(ic50 ~ ., train) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							new_role = "id variable") %>%
	step_normalize(all_predictors())

tabnet_spec <- tabnet(epochs = 10, decision_width = 24, attention_width = 26,
											num_steps = 5, penalty = 0.000001, virtual_batch_size = 512, momentum = 0.6,
											feature_reusage = 1.5, learn_rate = 0.02) %>%
	set_engine("torch", verbose = TRUE) %>%
	set_mode("regression")

rf_spec = rand_forest() %>% 
	set_engine("ranger", num.threads = 16) %>% 
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(rf_spec) %>%
	add_recipe(this_recipe)

model = this_wflow %>% 
	fit(train)

preds <- test %>% 
	bind_cols(predict(model, test))

cor(preds$ic50, preds$.pred)
