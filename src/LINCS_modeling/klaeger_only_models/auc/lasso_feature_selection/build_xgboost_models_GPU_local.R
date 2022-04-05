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
set.seed(2222)

this_dataset = read_rds(here('results/klaeger_only_models_auc/PRISM_klaeger_only_data_5000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))
all_importance = read_csv(here('results/klaeger_only_models_auc/5000feat_lasso_selected_features.csv'))
folds = read_rds(here(here('results/cv_folds/PRISM_klaeger_only_folds_auc.rds.gz')))

tic()
parser <- ArgumentParser(description='Process input paramters')
parser$add_argument('--feature_num', default = 100, type="integer")

args = data.frame(feature_num = c(100,200,300,400,500,600,700,809))

for (feature_num in args$feature_num) {
print(sprintf('Features: %02d',feature_num))

dir.create(here('results/klaeger_only_models/activation_expression/regression/lasso_feature_selection', 
								sprintf('xgboost/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/klaeger_only_models/activation_expression/regression/lasso_feature_selection/xgboost/results', 
												sprintf('%dfeat.rds.gz',feature_num))

this_recipe = recipe(auc_target ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("auc_target"),
							new_role = "id variable") %>%
	step_select(depmap_id,
							ccle_name,
							auc_target,
							broad_id,
							any_of(all_importance$Variable[1:feature_num])) %>% 
	step_zv(all_predictors()) %>% 
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
	write_rds(full_output_file, compress = "gz")
}

toc()