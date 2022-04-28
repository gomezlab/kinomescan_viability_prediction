library(tidyverse)
library(here)
library(vroom)
library(tictoc)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 10)
tic()
data = read_rds(here('results/PRISM_LINCS_klaeger_binary_data_for_ml_ic50.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_binary_data_feature_correlations_ic50.csv'))
variable_kinases = read_csv(here('results/PRISM_LINCS_klaeger_binary_data_variable_kinases_ic50.csv'))

feat5000_data = data %>% 
	select(any_of(c(cors$feature[1:4505], variable_kinases$kinase)),
				 depmap_id,
				 ccle_name,
				 ic50,
				 broad_id,
				 ic50_binary)

write_rds(feat5000_data, here('results/PRISM_LINCS_klaeger_binary_data_for_ml_5000feat_ic50.rds.gz'), compress = "gz")

feat10000_data = data %>% 
	select(any_of(c(cors$feature[1:9505], variable_kinases$kinase)),
				 depmap_id,
				 ccle_name,
				 ic50,
				 broad_id,
				 ic50_binary)

write_rds(feat10000_data, here('results/PRISM_LINCS_klaeger_binary_data_for_ml_10000feat_ic50.rds.gz'), compress = "gz")
toc()