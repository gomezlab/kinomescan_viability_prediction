library(here)
library(tidyverse)

# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 		job_name = sprintf('iRF%d',feature_num)
# 
# 	command = sprintf('sbatch --job-name=%s --mem=90G -c 16 --time=96:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_rf_models_ANOVA.R --feature_num %d"', job_name, feature_num)
# 
# 		# print(command)
# 		system(command)
# 
# }
# 
# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 	job_name = sprintf('ic50lr%d',feature_num)
# 
# 	command = sprintf('sbatch --job-name=%s --mem=99G -c 16 --time=40:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_lr_models_ANOVA.R --feature_num %d"', job_name, feature_num)
# 
# 	# print(command)
# 	system(command)
# 
# }

# for (feature_num in c(300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 	job_name = sprintf('ixg%d',feature_num)
# 
# 	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=90G --time=48:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_xgboost_models_ANOVA_GPU.R --feature_num %d"', job_name, feature_num)
# 
# 	# print(command)
# 	system(command)
# 
# }


# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 
# 	job_name = sprintf('ic50NN%d',feature_num)
# 
# 	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=90G --time=40:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_NN_models.R --feature_num %d"', job_name, feature_num)
# 
# 	# print(command)
# 	system(command)
# 
# }

command = sprintf(' sbatch -N 1 -n 1 -p volta-gpu --constraint="rhel8" --job-name=%s --mem=20G --time=120:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu --wrap "module load r"') 

# for (feature_num in c(3000,4000,5000)) {
# 
# 	job_name = sprintf('1i%d',feature_num)
# 
# 	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --constraint="rhel8" --job-name=%s --mem=20G --time=120:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "module load r/4.1.0; module load gcc/9.1.0; module load cuda/11.4; Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/keras_model_tuning/one_layer.R --feature_num %d"', job_name, feature_num)
# 
# 	# print(command)
# 	system(command)
# 
# }


# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 	job_name = sprintf('2i%d',feature_num)
# 
# 	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --constraint="rhel8" --job-name=%s --mem=20G --time=120:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/keras_model_tuning/two_layer.R --feature_num %d"', job_name, feature_num)
# 
# 	# print(command)
# 	system(command)
# 
# }
# 
# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 	job_name = sprintf('4i%d',feature_num)
# 
# 	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --constraint="rhel8" --job-name=%s --mem=20G --time=120:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/keras_model_tuning/four_layer.R --feature_num %d"', job_name, feature_num)
# 
# 	# print(command)
# 	system(command)
# 
# }
# 
# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 	job_name = sprintf('6i%d',feature_num)
# 
# 	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --constraint="rhel8" --job-name=%s --mem=20G --time=120:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/keras_model_tuning/six_layer.R --feature_num %d"', job_name, feature_num)
# 
# 	# print(command)
# 	system(command)
# 
# }