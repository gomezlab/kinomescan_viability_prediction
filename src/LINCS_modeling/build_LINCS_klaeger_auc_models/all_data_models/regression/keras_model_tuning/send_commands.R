library(here)
library(tidyverse)

for (feature_num in c(500,1000,1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('1a%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=50G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/one_layer.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}


for (feature_num in c(1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('2a%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=50G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/two_layer.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}

for (feature_num in c(1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('4a%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=50G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/four_layer.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}

for (feature_num in c(2000,3000,4000,5000)) {
	
	job_name = sprintf('6a%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=50G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/six_layer.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}