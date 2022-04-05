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

this_dataset = read_rds(here('results/klaeger_only_models_auc/PRISM_klaeger_only_data_10000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

# lr_recipe = recipe(auc_target ~ ., this_dataset) %>%
# 	update_role(-starts_with("act_"),
# 							-starts_with("exp_"),
# 							-starts_with("auc"),
# 							new_role = "id variable") %>%
# 	step_select(depmap_id,
# 							ccle_name,
# 							auc_target,
# 							broad_id,
# 							any_of(cors$feature[1:5000])) %>%
# 	step_zv(all_predictors()) %>% 
# 	step_normalize(all_predictors())
# 
# lr_final_spec = linear_reg(penalty = 0.00001, mixture = 1) %>%
# 	set_engine("glmnet") %>% 
# 	set_mode("regression")
# 
# final_wflow <-
# 	workflow() %>%
# 	add_model(lr_final_spec) %>%
# 	add_recipe(lr_recipe)
# 
# final_results = 
# 	final_wflow %>% 
# 	fit(this_dataset) %>% 
# 	write_rds(here('results/klaeger_only_models_auc/5000feat_lasso_final_results.rds.gz'), compress = "gz")
# 
# all_importance = vi(final_results %>% extract_fit_parsnip()) %>%
# 	arrange(desc(Importance)) %>%
# 	filter(Importance > 0) %>% 
# 	mutate(rank = 1:n()) %>%
# 	write_csv(here('results/klaeger_only_models_auc/5000feat_lasso_selected_features.csv'))

lr_recipe = recipe(auc_target ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("auc"),
							new_role = "id variable") %>%
	step_zv(all_predictors()) %>% 
	step_normalize(all_predictors())

lr_final_spec = linear_reg(penalty = 0.000001, mixture = 1) %>%
	set_engine("glmnet") %>% 
	set_mode("regression")

final_wflow <-
	workflow() %>%
	add_model(lr_final_spec) %>%
	add_recipe(lr_recipe)

final_results = 
	final_wflow %>% 
	fit(this_dataset) %>% 
	write_rds(here('results/klaeger_only_models_auc/10000feat_lasso_final_results.rds.gz'), compress = "gz")

all_importance = vi(final_results %>% extract_fit_parsnip()) %>%
	arrange(desc(Importance)) %>%
	filter(Importance > 0) %>% 
	mutate(rank = 1:n()) %>%
	write_csv(here('results/klaeger_only_models_auc/10000feat_lasso_selected_features.csv'))
