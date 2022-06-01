library(here)
library(tidyverse)

# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 		job_name = sprintf('iaRF%d',feature_num)
# 
# 	command = sprintf('sbatch --job-name=%s --mem=90G -c 16 --time=40:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/build_rf_models_ANOVA.R --feature_num %d"', job_name, feature_num)
# 
# 		# print(command)
# 		system(command)
# 
# }
# 
# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 	job_name = sprintf('ialr%d',feature_num)
# 
# 	command = sprintf('sbatch --job-name=%s --mem=99G -c 16 --time=40:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/build_lr_models_ANOVA.R --feature_num %d"', job_name, feature_num)
# 
# 	# print(command)
# 	system(command)
# 
# }

for (feature_num in c(1000,1500,2000,3000,4000,5000)) {

	job_name = sprintf('iaxg%d',feature_num)

	command = sprintf('sbatch -N 1 -n 1 -p dgx --job-name=%s --mem=90G --time=48:00:00 --qos gpu_access --gres=gpu:1 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/build_xgboost_models_ANOVA_GPU.R --feature_num %d"', job_name, feature_num)

	# print(command)
	system(command)

}
