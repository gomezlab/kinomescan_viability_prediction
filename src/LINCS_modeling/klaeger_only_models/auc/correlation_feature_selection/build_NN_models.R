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
library(keras)

tic()
parser <- ArgumentParser(description='Process input paramters')
parser$add_argument('--feature_num', default = 100, type="integer")

args = parser$parse_args()
args= data.frame(feature_num = 100)
print(sprintf('Features: %02d',args$feature_num))

dir.create(here('results/klaeger_only_models_auc/activation_expression/regression/', 
								sprintf('NN/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/klaeger_only_models_auc/activation_expression/regression/NN/results', 
												sprintf('%dfeat.rds.gz',args$feature_num))

this_dataset = read_rds(here('results/klaeger_only_models_auc/PRISM_klaeger_only_data_5000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

folds = read_rds(here(here('results/cv_folds/PRISM_klaeger_only_folds_auc.rds.gz')))

this_recipe = recipe(auc_target ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("auc_target"),
							new_role = "id variable") %>%
	step_select(depmap_id,
							ccle_name,
							auc_target,
							broad_id,
							any_of(cors$feature[1:args$feature_num])) %>%
	step_zv(all_predictors()) %>% 
	step_normalize(all_predictors())

keras_spec <- mlp(
	hidden_units = tune(), 
	penalty = tune()                  
) %>% 
	set_engine("keras", verbose = 0) %>% 
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(keras_spec) %>%
	add_recipe(this_recipe) 

keras_grid = read_rds(here('results/hyperparameter_grids/keras_grid.rds'))

race_ctrl = control_grid(
	save_pred = TRUE,
	parallel_over = "everything",
	verbose = TRUE
)

set.seed(2222)
results <- tune_grid(
	this_wflow,
	resamples = folds,
	grid = keras_grid,
	control = race_ctrl
) %>% 
	write_rds(full_output_file, compress = "gz")


toc()