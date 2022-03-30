library(here)
library(tidyverse)

#,200,300,400,500,1000,1500,2000,3000,4000,5000

for (feature_num in c(200,300,400,500,1000,1500,2000,3000,4000,5000)) {

		job_name = sprintf('aucRF%d',feature_num)

	command = sprintf('sbatch --job-name=%s --mem=99G -c 16 --time=40:00:00 --wrap "Rscript src/LINCS_modeling/klaeger_only_models/auc/correlation_feature_selection/build_rf_models_ANOVA.R --feature_num %d"', job_name, feature_num)

		# print(command)
		system(command)

}

for (feature_num in c(200,300,400,500,1000,1500,2000,3000,4000,5000)) {

	job_name = sprintf('auclr%d',feature_num)

	command = sprintf('sbatch --job-name=%s --mem=99G -c 16 --time=40:00:00 --wrap "Rscript src/LINCS_modeling/klaeger_only_models/auc/correlation_feature_selection/build_lr_models_ANOVA.R --feature_num %d"', job_name, feature_num)

	# print(command)
	system(command)

}

for (feature_num in c(200,300,400,500,1000,1500,2000,3000,4000,5000)) {

	job_name = sprintf('aucxg%d',feature_num)

	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=90G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/klaeger_only_models/auc/correlation_feature_selection/build_xgboost_models_ANOVA_GPU.R --feature_num %d"', job_name, feature_num)

	# print(command)
	system(command)

}


for (feature_num in c(200,300,400,500,1000,1500,2000,3000,4000,5000)) {


	job_name = sprintf('aucNN%d',feature_num)

	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=90G --time=40:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/klaeger_only_models/auc/correlation_feature_selection/build_NN_models.R --feature_num %d"', job_name, feature_num)

	# print(command)
	system(command)

}