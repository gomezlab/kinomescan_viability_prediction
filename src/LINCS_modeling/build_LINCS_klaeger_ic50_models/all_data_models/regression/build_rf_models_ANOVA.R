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

dir.create(here('results/PRISM_LINCS_klaeger_models/all_datasets/regression/', 
								sprintf('rand_forest/results')), 
					 showWarnings = F, recursive = T)
dir.create(here('results/PRISM_LINCS_klaeger_models/all_datasets/regression/', 
								sprintf('rand_forest/predictions')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models/all_datasets/regression/rand_forest/results', 
												sprintf('%dfeat.rds.gz',args$feature_num))

pred_output_file = here('results/PRISM_LINCS_klaeger_models/all_datasets/regression/rand_forest/predictions', 
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

rf_spec <- rand_forest(
	trees = tune()
) %>% set_engine("ranger", num.threads = 16) %>%
	set_mode("regression")

rf_param = rf_spec %>% 
	parameters() %>% 
	update(trees = trees(c(100, 2000)))

this_wflow <-
	workflow() %>%
	add_model(rf_spec) %>%
	add_recipe(this_recipe) 

rf_grid = rf_param %>% 
	grid_max_entropy(size = 15)

race_ctrl = control_race(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

results <- tune_race_anova(
	this_wflow,
	resamples = folds,
	grid = rf_grid,
	metrics = metric_set(rsq),
	control = race_ctrl
) %>% 
	write_rds(full_output_file, compress = "gz")

write_rds(results$.predictions[[1]], pred_output_file)

toc()