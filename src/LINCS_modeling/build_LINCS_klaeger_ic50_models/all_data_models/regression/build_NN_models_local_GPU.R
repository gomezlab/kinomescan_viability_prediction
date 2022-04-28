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

args = data.frame(feature_num = c(100,200,300,400,500,1000,1500,2000,3000,4000,5000))
this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat_ic50.rds'))
cors = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations_ic50.csv'))

folds = read_rds(here('results/PRISM_LINCS_klaeger_all_multiomic_data_folds_ic50.rds'))
keras_grid = read_rds(here('results/hyperparameter_grids/keras_grid.rds'))


for(i in 1:length(args$feature_num)) {
tic()	
print(sprintf('Features: %02d',args$feature_num[i]))

dir.create(here('results/PRISM_LINCS_klaeger_models_ic50/all_datasets/regression/', 
								sprintf('NN/results')), 
					 showWarnings = F, recursive = T)
dir.create(here('results/PRISM_LINCS_klaeger_models_ic50/all_datasets/regression/', 
								sprintf('NN/predictions')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_ic50/all_datasets/regression/NN/results', 
												sprintf('%dfeat.rds.gz',args$feature_num[i]))

pred_output_file = here('results/PRISM_LINCS_klaeger_models_ic50/all_datasets/regression/NN/predictions', 
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
	collect_metrics() %>% 
	write_rds(full_output_file, compress = "gz")
toc()
}

