library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

data = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/matched_only_models/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.rds.gz'))
data_10000 = read_rds(here('results/PRISM_LINCS_klaeger_models_auc/matched_only_models/PRISM_LINCS_klaeger_data_for_ml_10000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/matched_only_models/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

set.seed(2222)
folds = vfold_cv(data, v = 10) %>% 
	write_rds(here('results/cv_folds/matched_only_models/PRISM_LINCS_klaeger_folds_auc.rds.gz'), compress = "gz")

set.seed(2222)
folds = vfold_cv(data_10000, v = 10) %>% 
	write_rds(here('results/cv_folds/matched_only_models/PRISM_LINCS_klaeger_folds_10000_auc.rds.gz'), compress = "gz")