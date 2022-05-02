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

args = data.frame(feature_num = c(100,200,300,400,500,1000,1500,2000,3000,4000,5000))
this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat_ic50.rds.gz'))
cors = vroom(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations_ic50.csv'))
rf_grid = read_rds(here('results/hyperparameter_grids/rf_grid.rds'))
folds = read_rds(here('results/cv_folds/PRISM_LINCS_klaeger_all_multiomic_data_folds_ic50.rds.gz'))

tic()
for(i in 1:length(args$feature_num)) {
print(sprintf('Features: %02d',args$feature_num[i]))

dir.create(here('results/PRISM_LINCS_klaeger_models_ic50/all_datasets/regression/', 
								sprintf('rand_forest/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_ic50/all_datasets/regression/rand_forest/results', 
												sprintf('%dfeat.rds.gz',args$feature_num[i]))

this_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("cnv_"),
							-starts_with("prot_"),
							-starts_with("dep_"),
							-starts_with("ic50"),
							new_role = "id variable") %>%
	step_select(depmap_id,
							ccle_name,
							ic50,
							broad_id,
							any_of(cors$feature[1:args$feature_num[i]])) %>% 
	step_normalize(all_predictors())

rf_spec <- rand_forest(
	trees = tune()
) %>% set_engine("ranger", num.threads = 16) %>%
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(rf_spec) %>%
	add_recipe(this_recipe) 

race_ctrl = control_grid(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

results <- tune_grid(
	this_wflow,
	resamples = folds,
	grid = rf_grid,
	control = race_ctrl
) %>% 
	collect_metrics() %>% 
	write_rds(full_output_file, compress = "gz")

toc()
}