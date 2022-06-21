library(tidyverse)
library(here)
library(vroom)
library(tidymodels)


data = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat_auc.rds.gz'))

set.seed(2222)
folds = vfold_cv(data, v = 10) %>% 
	write_rds(here('results/cv_folds/PRISM_LINCS_klaeger_all_multiomic_data_folds_auc.rds.gz'), compress = "gz")
