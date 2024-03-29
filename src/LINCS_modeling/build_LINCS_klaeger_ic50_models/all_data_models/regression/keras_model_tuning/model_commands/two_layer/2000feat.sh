#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p volta-gpu
#SBATCH -t 120:00:00
#SBATCH --qos gpu_access
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --constraint='rhel8'
#SBATCH --job-name=4i2000
module load r/4.1.0
module load gcc/9.1.0
module load cuda/11.4
Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/two_layer.R --feature_num 2000
