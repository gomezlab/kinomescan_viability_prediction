library(tidyverse)
library(here)
library(keras)
library(vroom)
library(tidymodels)
library(tfdatasets)
library(tictoc)
library(conflicted)
conflict_prefer("fit", "keras")
conflict_prefer("filter", "dplyr")
conflict_prefer("all_numeric", "tfdatasets")

compound_match_list = read_csv(here('results/matching/LINCS_PRISM_klaeger_combined_drug_matches_final.csv'))
full_dataset = read_rds(here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))
args = data.frame(feature_num = c(100,200,300,400,500,1000,1500,2000,3000,4000,5000))

hyper_grid = data.frame('dropout' = c(0.2,0.4,0.6,0.8),
												'units_1' = c(256,512,1024,2028),
												'units_2' = c(64,128,256,512),
												'units_3' = c(32,64,128,256),
												'units_4' = c(16,32,64,128),
												'learn_rate' = c(0.1, 0.01, 0.001, 0.0001))


for(i in 1:length(args$feature_num)) {
tic()	
	
dir.create(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/', 
									sprintf('keras/results')), 
						 showWarnings = F, recursive = T)
	
full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/keras/results', 
													sprintf('%dfeat.csv',args$feature_num[i]))

full_annotated_dataset = full_dataset %>% 
	rename('auc_target' = auc) %>% 
	select(depmap_id,
				 auc_target,
				 ccle_name,
				 broad_id,
				 any_of(cors$feature[1:args$feature_num[i]])) %>% 
	unique() %>% 
	drop_na()

split = initial_split(full_annotated_dataset)

#keras all data model
x_train = training(split) %>% 
	select(auc_target, starts_with(c("act", "exp_")))

spec = feature_spec(x_train, auc_target ~ .) %>%
	step_numeric_column(all_numeric()) %>%
	fit()

input <- layer_input_from_dataset(x_train %>% select(-auc_target))

#build_model = function(hyper_grid) {
main_output = input %>% 
	layer_dense_features(dense_features(spec)) %>%
	layer_dense(units = 2000, activation = 'relu') %>% 
	layer_dropout(rate = 0.4) %>% 
	layer_dense(units = 500, activation = 'relu') %>%
	layer_dropout(rate = 0.4) %>%
	layer_dense(units = 1, activation = 'linear')

model = keras_model(
	inputs = input,
	outputs = main_output
)

model %>% 
	compile(
		optimizer = optimizer_adam(),
		loss = 'mse',
		metrics= metric_root_mean_squared_error())

#return(model)

#}

print_dot_callback <- callback_lambda(
	on_epoch_end = function(epoch, logs) {
		if (epoch %% 80 == 0) cat("\n")
		cat(".")
	}
)  

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

history <- model %>% fit(
	x = x_train %>% select(-auc_target),
	y = x_train$auc_target,
	epochs = 10,
	validation_split = 0.2,
)

test_df = testing(split) %>% 
	select(auc_target, starts_with(c("act", "exp_")))
c(loss, rmse) %<-% (model %>% evaluate(test_df %>% select(-auc_target), test_df$auc_target))

test_predictions <- model %>% predict(test_df %>% select(-auc_target))

predictions = data.frame(predicted_auc = test_predictions) %>% 
	bind_cols(test_df)

rsq_val = cor(predictions$auc_target,
							predictions$predicted_auc)^2

this_metrics = data.frame('loss' = loss,
													'rmse' = rmse,
													'rsq' = rsq_val,
													'num_features' = args$feature_num[i]) %>%  
	write_csv(full_output_file)

toc()
}


