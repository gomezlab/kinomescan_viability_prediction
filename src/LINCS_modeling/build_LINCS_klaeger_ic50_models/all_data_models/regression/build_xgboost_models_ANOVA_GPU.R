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
tic()
parser <- ArgumentParser(description='Process input paramters')
parser$add_argument('--feature_num', default = 100, type="integer")

args = parser$parse_args()
print(sprintf('Features: %02d',args$feature_num))

dir.create(here('results/PRISM_LINCS_klaeger_models/all_datasets/regression/', 
								sprintf('xgboost/results')), 
					 showWarnings = F, recursive = T)
dir.create(here('results/PRISM_LINCS_klaeger_models/all_datasets/regression/', 
								sprintf('xgboost/predictions')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models/all_datasets/regression/xgboost/results', 
												sprintf('%dfeat.rds.gz',args$feature_num))

pred_output_file = here('results/PRISM_LINCS_klaeger_models/all_datasets/regression/xgboost/predictions', 
												sprintf('%dfeat.rds.gz',args$feature_num))

this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat.rds'))
cors = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations.csv'))

folds = read_rds(here(results/'PRISM_LINCS_klaeger_all_multiomic_data_folds.rds'))

this_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("cnv_"),
							-starts_with("prot_"),
							-starts_with("dep_"),
							-starts_with("ic50"),
							ic50_binary,
							new_role = "id variable") %>%
	step_select(depmap_id,
							ccle_name,
							ic50,
							ic50_binary,
							broad_id,
							any_of(cors$feature[1:args$feature_num])) %>% 
	step_normalize(all_predictors())

xgb_spec <- boost_tree(
	trees = tune(), 
	tree_depth = tune()                  
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist") %>% 
	set_mode("regression")

xgb_grid = read_rds(here('results/hyperparameter_grids/xbg_grid.rds'))

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

write_rds(results$.predictions[[1]], pred_output_file)

toc()