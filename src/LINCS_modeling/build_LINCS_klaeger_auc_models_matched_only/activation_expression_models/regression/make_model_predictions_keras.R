library(tidyverse)
library(here)
library(keras)
library(conflicted)
conflict_prefer("fit", "keras")
conflict_prefer("filter", "dplyr")

model = load_model_hdf5(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/keras/final_keras_model.h5'))

not_tested_data = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/not_tested_data.rds.gz'))


predictions <- model %>% 
	predict(not_tested_data %>% 
						select(starts_with(c("act_", "exp_")))) %>% 
	write_rds(here('results/PRISM_LINCS_klaeger_models_auc/final_model_predictions.rds'))


ids = not_tested_data %>% select(-starts_with(c("act_", "exp_")))

full_predictions = data.frame(predicted_auc = predictions) %>% 
	bind_cols(ids) %>% 
	write_csv(here('results/PRISM_LINCS_klaeger_models_auc/final_model_predictions_keras.csv'))