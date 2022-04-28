library(tidyverse)
library(here)
library(patchwork)
library(keras)
library(vroom)
library(tidymodels)
library(tfdatasets)
library(tictoc)
library(Metrics)
library(conflicted)
conflict_prefer("fit", "keras")
conflict_prefer("filter", "dplyr")
conflict_prefer("all_numeric", "tfdatasets")
conflict_prefer("rmse", "Metrics")

compound_match_list = read_csv(here('results/matching/LINCS_PRISM_klaeger_combined_drug_matches_final.csv'))
data = read_rds(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_data_for_ml_5000feat_ic50.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_data_feature_correlations_ic50.csv'))
args = data.frame(feature_num = c(2000))

i = 1
# for(i in 1:length(args$feature_num)) {
tic()	
	
dir.create(here('results/PRISM_LINCS_klaeger_models_ic50/activation_expression/regression/', 
									sprintf('keras/results')), 
						 showWarnings = F, recursive = T)
	
full_output_file = here('results/PRISM_LINCS_klaeger_models_ic50/activation_expression/regression/keras/results', 
													sprintf('%dfeat.csv',args$feature_num[i]))

this_dataset = data %>% 
	rename('ic50_target' = ic50) %>% 
	select(depmap_id,
				 ic50_target,
				 ccle_name,
				 broad_id,
				 any_of(cors$feature[1:args$feature_num[i]])) %>% 
	unique() %>% 
	drop_na()

spec = feature_spec(this_dataset %>% 
											select(ic50_target, starts_with(c("act", "exp_"))),
										ic50_target ~ .) %>%
	step_numeric_column(all_numeric()) %>%
	fit()

# split = initial_split(this_dataset)

set.seed(2222)
folds = vfold_cv(this_dataset, v = 10)


all_metrics = data.frame()
all_predictions = data.frame()
for (split_id in 1:10) {
#keras model
split = folds$splits[[split_id]]
x_train = training(split) %>% 
	select(ic50_target, starts_with(c("act", "exp_")))
x_val = testing(split) %>% 
	select(ic50_target, starts_with(c("act", "exp_")))

input <- layer_input_from_dataset(x_train %>% select(-ic50_target))

main_output = input %>% 
	layer_dense_features(dense_features(spec)) %>%
	layer_batch_normalization() %>% 
	layer_dense(units = 200, activation = 'relu') %>% 
	layer_dropout(rate = 0.2) %>% 
	layer_dense(units = 200, activation = 'relu') %>% 
	layer_dropout(rate = 0.2) %>% 
	layer_dense(units = 200, activation = 'relu') %>% 
	layer_dropout(rate = 0.2) %>% 
	layer_dense(units = 200, activation = 'relu') %>% 
	layer_dropout(rate = 0.2) %>% 
	layer_dense(units = 1, activation = 'linear')

model = keras_model(
	inputs = input,
	outputs = main_output
)

model %>% 
	compile(
		optimizer = optimizer_adam(learning_rate = 0.003),
		loss = 'mse',
		metrics= metric_root_mean_squared_error())

# print_dot_callback <- callback_lambda(
# 	on_epoch_end = function(epoch, logs) {
# 		if (epoch %% 80 == 0) cat("\n")
# 		cat(".")
# 	}
# )  

early_stop <- callback_early_stopping(monitor = "val_loss", 
																			patience = 25,
																			min_delta = 0.000006,
																			restore_best_weights = TRUE)
set.seed(2222)
history <- model %>% fit(
	x = x_train %>% select(-ic50_target),
	y = x_train$ic50_target,
	epochs = 500,
	batch_size = 512,
	validation_data = list(x_val %>% select(-ic50_target), x_val$ic50_target),
	callbacks = list(early_stop)
)

n_epochs = length(history$metrics$loss)

test_df = testing(split) %>% 
	select(ic50_target, starts_with(c("act", "exp_")))
c(loss, rmse) %<-% (model %>% evaluate(test_df %>% select(-ic50_target), test_df$ic50_target))

test_predictions <- model %>% predict(test_df %>% select(-ic50_target))

ids = testing(split) %>% 
	select(-starts_with(c("act_", "exp_")), -ic50_target)

predictions = data.frame(predicted_ic50 = test_predictions) %>% 
	bind_cols(test_df) %>% 
	select(-starts_with(c("act_", "exp_"))) %>% 
	bind_cols(ids) %>% 
	rename("ic50" = ic50_target) %>% 
	mutate("split_id" = split_id)

rsq_val = cor(predictions$ic50,
							predictions$predicted_ic50)^2

this_metrics = data.frame('loss' = loss,
													'rmse' = rmse,
													'rsq' = rsq_val,
													'num_features' = args$feature_num[i],
													'split_id' = split_id,
													'epochs' = n_epochs)

all_metrics = bind_rows(all_metrics, this_metrics)
all_predictions = bind_rows(all_predictions, predictions)

toc()

k_clear_session()
# }
}

write_csv(all_metrics, 
					here('results/PRISM_LINCS_klaeger_models_ic50/activation_expression/regression/keras/final_keras_cv_metrics.csv')
					)
write_rds(all_predictions, 
					here('results/PRISM_LINCS_klaeger_models_ic50/activation_expression/regression/keras/final_keras_cv_predictions.rds.gz'),
					compress = "gz"
)
