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

tic()
parser <- ArgumentParser(description='Process input paramters')
parser$add_argument('--feature_num', default = 100, type="integer")

args = parser$parse_args()

print(sprintf('Features: %02d',args$feature_num))

dir.create(here('results/PRISM_LINCS_klaeger_models/activation_expression/regression/', 
								sprintf('lr/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models/activation_expression/regression/lr/results', 
												sprintf('%dfeat.rds.gz',args$feature_num))

this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat.rds'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations.csv'))

folds = read_rds(here('results/PRISM_LINCS_klaeger_folds.rds'))

this_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
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

lr_spec <- linear_reg(penalty = 0.1, mixture = 1) %>%
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(lr_spec) %>%
	add_recipe(this_recipe) 

race_ctrl = control_resamples(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

results <- fit_resamples(
	this_wflow,
	resamples = folds,
	control = race_ctrl
) %>% 
	write_rds(full_output_file, compress = "gz")

toc()
