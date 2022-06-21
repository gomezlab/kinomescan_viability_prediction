library(here)
library(tidyverse)

for (feature_num in c(1000,1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('1_%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=30G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/activation_expression_models/regression/keras_model_tuning/one_layer.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}


for (feature_num in c(1000,1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('2_%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=30G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/activation_expression_models/regression/keras_model_tuning/two_layer.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}

for (feature_num in c(1000,1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('4_%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=30G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/activation_expression_models/regression/keras_model_tuning/four_layer.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}

for (feature_num in c(1000,1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('6_%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=30G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/activation_expression_models/regression/keras_model_tuning/six_layer.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}