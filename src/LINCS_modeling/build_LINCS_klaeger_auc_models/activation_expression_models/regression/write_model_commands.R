library(tidyverse)
library(here)

feature_num = 100

writeLines(c("#!/bin/bash",
						 "#SBATCH -N 1",
						 "#SBATCH -n 1",
						 "#SBATCH -p volta-gpu",
						 "#SBATCH -t 1:00:00",
						 "#SBATCH --qos gpu_access",
						 "#SBATCH --mem=30G",
						 "#SBATCH --gres=gpu:1",
						 "#SBATCH --constraint='rhel8'",
						 "module load r/4.1.0",
						 "module load gcc/9.1.0",
						 "module load cuda/11.4",
						 sprintf("Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/activation_expression_models/regression/build_xgboost_models_ANOVA_GPU.R --feature_num %d", feature_num)
						 ), here("src/output.sh"))
