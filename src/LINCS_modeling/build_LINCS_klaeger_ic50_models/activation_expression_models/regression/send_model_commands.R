library(here)
library(tidyverse)

for (feature_num in c(10)) {

		job_name = sprintf('RF_%d',feature_num)

		command = sprintf('sbatch --job-name=%s --mem=99G -c 16 --time=1:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_rf_models_ANOVA.R --feature_num %d"', job_name, feature_num)

		# print(command)
		system(command)

}

for (feature_num in c(10)) {
	
	job_name = sprintf('LR_%d',feature_num)
	
	command = sprintf('sbatch --job-name=%s --mem=99G -c 8 --time=1:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/regression/build_lr_models_ANOVA.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}