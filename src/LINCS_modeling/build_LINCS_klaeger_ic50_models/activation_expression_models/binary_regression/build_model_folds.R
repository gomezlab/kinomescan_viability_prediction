library(tidyverse)
library(here)
library(tidymodels)
library(finetune)
library(tictoc)

tic()
data = read_rds(here('results/PRISM_LINCS_klaeger_binary_data_for_ml_5000feat_ic50.rds.gz'))
data_10000 = read_rds(here('results/PRISM_LINCS_klaeger_binary_data_for_ml_10000feat_ic50.rds.gz'))

set.seed(2222)
folds = vfold_cv(data, v = 10) %>% 
	write_rds(here('results/cv_folds/PRISM_LINCS_klaeger_binary_folds_ic50.rds.gz'), compress = "gz")

folds = vfold_cv(data_10000, v = 10) %>% 
	write_rds(here('results/cv_folds/PRISM_LINCS_klaeger_binary_folds_10000_ic50.rds.gz'), compress = "gz")
toc()