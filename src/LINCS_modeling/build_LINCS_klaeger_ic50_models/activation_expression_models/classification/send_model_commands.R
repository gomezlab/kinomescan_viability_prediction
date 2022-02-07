library(here)
library(tidyverse)

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {

		job_name = sprintf('RF_%d',feature_num)
		
		command = sprintf('sbatch --job-name=%s --mem=128G -c 8 --time=120:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/classification/build_rf_models_ANOVA.R --feature_num %d"', job_name, feature_num)
		
		# print(command)
		system(command)
	
}

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('xbg_%d',feature_num)
	
	command = sprintf('sbatch --job-name=%s --mem=128G -c 8 --time=120:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/classification/build_xgboost_models_ANOVA.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
	
	job_name = sprintf('svm_%d',feature_num)
	
	command = sprintf('sbatch --job-name=%s --mem=128G -c 8 --time=120:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/classification/build_svm_models_ANOVA.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}