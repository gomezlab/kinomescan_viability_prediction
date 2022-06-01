library(here)
library(tidyverse)

# for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
# 
# 		job_name = sprintf('iRF%d',feature_num)
# 
# 	command = sprintf('sbatch --job-name=%s --mem=90G -c 16 --time=40:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_rf_models_ANOVA.R --feature_num %d"', job_name, feature_num)
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

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {

	job_name = sprintf('ixg%d',feature_num)

	#command = sprintf('sbatch -N 1 -n 1 -p gpu --job-name=%s --mem=90G --time=48:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_xgboost_models_ANOVA_GPU.R --feature_num %d"', job_name, feature_num)
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --constraint=rhel8 --job-name=%s --mem=50G --time=48:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu --wrap=\"echo \' module load r/4.1.0; module load cuda/11.4; module load gcc/9.1.0; Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_xgboost_models_ANOVA_GPU.R --feature_num %d \' | bash\"', job_name, feature_num)
	
	# print(command)
	system(command)

}



