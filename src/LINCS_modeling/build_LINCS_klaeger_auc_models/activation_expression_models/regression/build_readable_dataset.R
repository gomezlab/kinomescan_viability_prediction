library(tidyverse)
library(here)
library(vroom)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 10)
data = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

feat5000_data = data %>% 
  select(any_of(cors$feature[1:5005]),
         depmap_id,
         ccle_name,
         auc,
         broad_id,
         auc_binary)

write_rds(feat5000_data, here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.rds.gz'), compress = "gz")
#write csv for python
#write_csv(feat5000_data, here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.csv'))


feat10000_data = data %>% 
	select(any_of(cors$feature[1:10005]),
				 depmap_id,
				 ccle_name,
				 auc,
				 broad_id,
				 auc_binary)

write_rds(feat10000_data, here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_for_ml_10000feat_auc.rds.gz'), compress = "gz")
