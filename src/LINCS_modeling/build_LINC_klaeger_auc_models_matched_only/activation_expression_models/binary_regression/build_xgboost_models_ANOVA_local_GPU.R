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
library(conflicted)
conflict_prefer("slice", "dplyr")

args = data.frame(feature_num = c(1,100,200,300,400,500,1000,1500,2000,3000,4000,5000,6000,7000,8000))
this_dataset = read_rds(here('results/PRISM_LINCS_klaeger_binary_data_for_ml_5000feat_auc.rds.gz')) 
cors =  vroom(here('results/PRISM_LINCS_klaeger_binary_data_feature_correlations_auc.csv'))
variable_kinases = read_csv(here('results/PRISM_LINCS_klaeger_binary_data_variable_kinases_auc.csv'))

folds = read_rds(here('results/cv_folds/PRISM_LINCS_klaeger_binary_folds_10000_auc.rds.gz'))
xgb_grid = read_rds(here('results/hyperparameter_grids/xgb_grid.rds')) 

for(i in 1:length(args$feature_num)) {
	tic()	
print(sprintf('Features: %02d',args$feature_num[i]))

dir.create(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/binary_regression/', 
								sprintf('xgboost/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/binary_regression/xgboost/results', 
												sprintf('%dfeat.rds.gz',args$feature_num[i]))

this_recipe = recipe(auc ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("auc"),
							auc_binary,
							new_role = "id variable") %>% 
	step_select(depmap_id,
							ccle_name,
							auc,
							broad_id,
							any_of(c(cors$feature[1:args$feature_num[i]], variable_kinases$kinase))) %>% 
	step_dummy(starts_with("act_"), one_hot = TRUE)

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
		# grid = xgb_grid,
		control = race_ctrl
	) %>%
	write_rds(full_output_file, compress = "gz")
toc()
}
# 
# temp = results$.notes[[1]]
# temp$.notes[1]
# 
# temp2 = this_dataset$act_ABL1_nonphosphorylated
# temp3 = folds$splits[[1]]$data$ac