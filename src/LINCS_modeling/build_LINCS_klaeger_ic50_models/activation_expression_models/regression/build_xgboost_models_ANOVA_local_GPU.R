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

args = data.frame(feature_num = c(6000,7000,8000,9000))
this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_data_for_ml_10000feat_ic50.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations_ic50.csv'))
folds = read_rds(here('results/cv_folds/PRISM_LINCS_klaeger_folds_10000_ic50.rds.gz'))

for(i in 1:length(args$feature_num)) {
	tic()	
print(sprintf('Features: %02d',args$feature_num[i]))

dir.create(here('results/PRISM_LINCS_klaeger_models_ic50/activation_expression/regression/', 
								sprintf('xgboost/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_ic50/activation_expression/regression/xgboost/results', 
												sprintf('%dfeat.rds',args$feature_num[i]))

this_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							ic50_binary,
							new_role = "id variable") %>%
	step_select(depmap_id,
							ccle_name,
							ic50,
							broad_id,
							any_of(cors$feature[1:args$feature_num[i]])) %>% 
	step_normalize(all_predictors())

xgb_spec <- boost_tree(
	trees = 168, 
	tree_depth = 5             
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist") %>% 
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(xgb_spec) %>%
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
	collect_metrics() %>% 
	write_rds(full_output_file, compress = "gz")
toc()
}
