library(tidyverse)
library(here)
library(patchwork)
library(keras)
library(vroom)
library(tidymodels)
library(tfdatasets)
library(tictoc)
library(Metrics)
library(argparse)
library(conflicted)
conflict_prefer("fit", "keras")
conflict_prefer("filter", "dplyr")
conflict_prefer("all_numeric", "tfdatasets")
conflict_prefer("rmse", "Metrics")

#read in data
data = read_rds(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat_ic50.rds.gz'))
cors = vroom(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations_ic50.csv'))

tic()
parser <- ArgumentParser(description='Process input paramters')
parser$add_argument('--feature_num', default = 100, type="integer")

args = parser$parse_args()

print(sprintf('Features: %02d',args$feature_num))

dropout = c(0.2,0.4,0.6)
neurons = c(200,400,600,800,1000)
grid = data.frame(crossing(dropout, neurons))

# for(i in 1:length(args$feature_num)) {
tic()	

dir.create(here('results/PRISM_LINCS_klaeger_models_ic50/all_datasets/regression/', 
								sprintf('keras/results/six_layer')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_ic50/all_datasets/regression/keras/results/six_layer', 
												sprintf('%dfeat.csv',args$feature_num))

this_dataset = data %>% 
	rename('ic50_target' = ic50) %>% 
	select(depmap_id,
				 ic50_target,
				 ccle_name,
				 broad_id,
				 any_of(cors$feature[1:args$feature_num])) %>% 
	unique() %>% 
	drop_na()

spec = feature_spec(this_dataset %>% 
											select(ic50_target, starts_with(c("act", "exp_", "prot_", "cnv_", "dep_"))),
										ic50_target ~ .) %>%
	step_numeric_column(all_numeric()) %>%
	fit()

set.seed(2222)
folds = vfold_cv(this_dataset, v = 5)

all_tuning_summarised_metrics = data.frame()
for(j in 1:dim(grid)[1]) {
	
	all_metrics = data.frame()
	for (split_id in 1:5) {
		
		
		split = folds$splits[[split_id]]
		x_train = training(split) %>% 
			select(ic50_target, starts_with(c("act", "exp_", "prot_", "cnv_", "dep_")))
		x_val = testing(split) %>% 
			select(ic50_target, starts_with(c("act", "exp_", "prot_", "cnv_", "dep_")))
		
		input <- layer_input_from_dataset(x_train %>% select(-ic50_target))
		
		main_output = input %>% 
			layer_dense_features(dense_features(spec)) %>%
			layer_batch_normalization() %>%
			layer_dense(units = grid$neurons[j], activation = 'relu') %>% 
			layer_dropout(rate = grid$dropout[j]) %>% 
			layer_dense(units = grid$neurons[j], activation = 'relu') %>% 
			layer_dropout(rate = grid$dropout[j]) %>% 
			layer_dense(units = grid$neurons[j], activation = 'relu') %>% 
			layer_dropout(rate = grid$dropout[j]) %>% 
			layer_dense(units = grid$neurons[j], activation = 'relu') %>% 
			layer_dropout(rate = grid$dropout[j]) %>% 
			layer_dense(units = grid$neurons[j], activation = 'relu') %>% 
			layer_dropout(rate = grid$dropout[j]) %>% 
			layer_dense(units = grid$neurons[j], activation = 'relu') %>% 
			layer_dropout(rate = grid$dropout[j]) %>% 
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
			select(ic50_target, starts_with(c("act", "exp_", "prot_", "cnv_", "dep_")))
		c(loss, rmse) %<-% (model %>% evaluate(test_df %>% select(-ic50_target), test_df$ic50_target))
		
		test_predictions <- model %>% predict(test_df %>% select(-ic50_target))
		
		ids = testing(split) %>% 
			select(-starts_with(c("act", "exp_", "prot_", "cnv_", "dep_")), -ic50_target)
		
		predictions = data.frame(predicted_ic50 = test_predictions) %>% 
			bind_cols(test_df) %>% 
			select(-starts_with(c("act", "exp_", "prot_", "cnv_", "dep_"))) %>% 
			bind_cols(ids) %>% 
			rename("ic50" = ic50_target) %>% 
			mutate("split_id" = split_id)
		
		rsq_val = cor(predictions$ic50,
									predictions$predicted_ic50)^2
		
		this_metrics = data.frame('loss' = loss,
															'rmse' = rmse,
															'rsq' = rsq_val,
															'num_features' = args$feature_num,
															'split_id' = split_id,
															'epochs' = n_epochs,
															'dropout' = grid$dropout[j],
															'neurons' = grid$neurons[j])
		
		all_metrics = bind_rows(all_metrics, this_metrics)
		
		toc()
		
		k_clear_session()
	}
	
	this_tuning_summarised_metrics = all_metrics %>% 
		group_by(num_features, dropout, neurons) %>% 
		summarise(mean_rsq = mean(rsq),
							mean_rmse = mean(rmse),
							mean_loss = mean(loss),
							sd_rsq = sd(rsq),
							sd_rmse = sd(rmse),
							sd_loss = sd(loss),
							mean_epochs = mean(epochs))
	
	all_tuning_summarised_metrics = bind_rows(all_tuning_summarised_metrics, this_tuning_summarised_metrics)
	
}
write_csv(all_tuning_summarised_metrics, full_output_file)
# }