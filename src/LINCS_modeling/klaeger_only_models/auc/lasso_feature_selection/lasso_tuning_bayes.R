library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)
library(vip)

this_dataset = read_rds(here('results/klaeger_only_models_auc/PRISM_klaeger_only_data_5000feat_auc.rds.gz'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_models_auc/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

folds = read_rds(here(here('results/cv_folds/PRISM_klaeger_only_folds_auc.rds.gz')))

lr_recipe = recipe(auc_target ~ ., this_dataset) %>%
  update_role(-starts_with("act_"),
              -starts_with("exp_"),
              -starts_with("auc"),
              new_role = "id variable") %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

lr_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>% 
  set_mode("regression")

set.seed(2222)
lr_grid = grid_max_entropy(penalty(), size = 30)

this_wflow <-
  workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(lr_recipe)

race_ctrl = control_bayes(
  save_pred = TRUE, 
  parallel_over = "everything",
  verbose = TRUE,
  no_improve = 10
)

set.seed(2222)
results = tune_bayes(
  this_wflow,
  resamples = folds,
  iter = 30,
  # grid = lr_grid,
  control = race_ctrl,
  initial = 6
) 

temp = collect_metrics(results)


