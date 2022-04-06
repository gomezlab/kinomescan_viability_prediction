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
data = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))
args = data.frame(feature_num = c(2000))

i = 1
# for(i in 1:length(args$feature_num)) {
tic()	
	
dir.create(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/', 
									sprintf('keras/results')), 
						 showWarnings = F, recursive = T)
	
full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/keras/results', 
													sprintf('%dfeat.csv',args$feature_num[i]))

this_dataset = data %>% 
	rename('auc_target' = auc) %>% 
	select(depmap_id,
				 auc_target,
				 ccle_name,
				 broad_id,
				 any_of(cors$feature[1:args$feature_num[i]])) %>% 
	unique() %>% 
	drop_na()

spec = feature_spec(this_dataset %>% 
											select(auc_target, starts_with(c("act", "exp_"))),
										auc_target ~ .) %>%
	step_numeric_column(all_numeric()) %>%
	fit()

# split = initial_split(this_dataset)

set.seed(2222)
folds = vfold_cv(this_dataset, v = 10)
split_id = 1
for (split_id in 1:10) {
#keras model
split = folds$splits[[split_id]]
x_train = training(split) %>% 
	select(auc_target, starts_with(c("act", "exp_")))

input <- layer_input_from_dataset(x_train %>% select(-auc_target))

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

history <- model %>% fit(
	x = x_train %>% select(-auc_target),
	y = x_train$auc_target,
	epochs = 500,
	batch_size = 512,
	#validation_split = 0.2,
	callbacks = list(early_stop)
)
k_clear_session()

test_df = testing(split) %>% 
	select(auc_target, starts_with(c("act", "exp_")))
c(loss, rmse) %<-% (model %>% evaluate(test_df %>% select(-auc_target), test_df$auc_target))

test_predictions <- model %>% predict(test_df %>% select(-auc_target))

ids = testing(split) %>% 
	select(-starts_with(c("act_", "exp_")), -auc_target)

predictions = data.frame(predicted_auc = test_predictions) %>% 
	bind_cols(test_df) %>% 
	select(-starts_with(c("act_", "exp_"))) %>% 
	bind_cols(ids) %>% 
	rename("auc" = auc_target)

rsq_val = cor(predictions$auc,
							predictions$predicted_auc)^2

this_metrics = data.frame('loss' = loss,
													'rmse' = rmse,
													'rsq' = rsq_val,
													'num_features' = args$feature_num[i])

toc()
# }
}

# comparing kinomescan vs klaeger results 
compound_match_list = read_csv(here('results/matching/LINCS_PRISM_klaeger_combined_drug_matches_no_overlap.csv'))

final_full_predictions = predictions %>% 
	left_join(compound_match_list, by = 'broad_id') %>% 
	mutate(origin = if_else(
		is.na(klaeger_name),
		"KINOMEscan",
		"Kinobeads"
	)) %>% 
	select(-klaeger_name, -LINCS_name) 

kinomescan_rsq = cor(
	final_full_predictions %>% 
		filter(origin == "KINOMEscan") %>% 
		pull(predicted_auc),
	final_full_predictions %>% 
		filter(origin == "KINOMEscan") %>% 
		pull(auc)
)^2

klaeger_rsq = cor(
	final_full_predictions %>% 
		filter(origin == "Kinobeads") %>% 
		pull(predicted_auc),
	final_full_predictions %>% 
		filter(origin == "Kinobeads") %>% 
		pull(auc)
)^2

kinomescan_rmse = rmse(
	final_full_predictions %>% 
		filter(origin == "KINOMEscan") %>% 
		pull(predicted_auc),
	final_full_predictions %>% 
		filter(origin == "KINOMEscan") %>% 
		pull(auc)
)

klaeger_rmse = rmse(
	final_full_predictions %>% 
		filter(origin == "Kinobeads") %>% 
		pull(predicted_auc),
	final_full_predictions %>% 
		filter(origin == "Kinobeads") %>% 
		pull(auc)
)

a = final_full_predictions %>%
	filter(origin == "KINOMEscan") %>% 
	ggplot(aes(x = predicted_auc, y = auc)) +
	geom_hex() +
	scale_fill_viridis_c() +
	labs(title = 
			 	paste0('KINOMEscan R\u00B2 = ',
			 				 round(
			 				 	kinomescan_rsq,
			 				 	2),
			 				 '/ RMSE = ',
			 				 round(
			 				 	kinomescan_rmse,
			 				 	2)
			 	), 
			 x = "Predicted AUC",
			 y = "Actual AUC") +
	geom_abline(intercept = 0, slope = 1, size = 0.3, colour = 'black', linetype = 3) +
	geom_smooth(colour = "red") +
	coord_cartesian(xlim = c(0.2,1), ylim= c(0,1)) +
	theme(
		panel.background = element_rect(fill = "transparent",colour = NA),
		panel.grid.minor = element_blank(), 
		panel.grid.major = element_blank(),
		plot.background = element_rect(fill = "transparent",colour = NA)
	)

b = final_full_predictions %>%
	filter(origin == "Kinobeads") %>% 
	ggplot(aes(x = predicted_auc, y = auc)) +
	geom_hex() +
	scale_fill_viridis_c() +
	labs(title = 
			 	paste0('Kinobeads R\u00B2 = ',
			 				 round(
			 				 	klaeger_rsq,
			 				 	2),
			 				 '/ RMSE = ',
			 				 round(
			 				 	klaeger_rmse,
			 				 	2)
			 	), 
			 x = "Predicted AUC",
			 y = "Actual AUC") +
	geom_abline(intercept = 0, slope = 1, size = 0.3, colour = 'black', linetype = 3) +
	geom_smooth(colour = "red") +
	coord_cartesian(xlim = c(0.2,1), ylim= c(0,1)) +
	theme(
		panel.background = element_rect(fill = "transparent",colour = NA),
		panel.grid.minor = element_blank(), 
		panel.grid.major = element_blank(),
		plot.background = element_rect(fill = "transparent",colour = NA)
	)
c = a+b
ggsave(here('figures/PRISM_LINCS_klaeger/keras_replicate_klaeger_kinomescan_comparison.png'), width = 12, height = 8)
