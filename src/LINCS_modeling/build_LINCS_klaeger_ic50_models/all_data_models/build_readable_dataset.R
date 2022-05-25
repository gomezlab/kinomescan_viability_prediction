library(tidyverse)
library(here)
library(vroom)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 10)
data = read_rds(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_ic50.rds.gz'))
cors = vroom(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations_ic50.csv'))

feat5000_data = data %>% 
	select(any_of(cors$feature[1:5010]),
				 depmap_id,
				 ccle_name,
				 ic50,
				 broad_id,
				 ic50_binary)

write_rds(feat5000_data, here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat_ic50.rds.gz'), compress = "gz")

	