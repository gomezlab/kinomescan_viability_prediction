library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

data = read_rds(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_data_for_ml_5000feat_ic50.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_ic50/PRISM_LINCS_klaeger_data_feature_correlations_ic50.csv'))

set.seed(2222)
folds = vfold_cv(data, v = 10) %>% 
	write_rds(here('results/cv_folds/PRISM_LINCS_klaeger_folds_ic50.rds.gz'), compress = "gz")
