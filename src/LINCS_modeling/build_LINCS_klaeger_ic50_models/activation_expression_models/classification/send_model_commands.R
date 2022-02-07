library(here)
library(tidyverse)

for (feature_num in c(100)) {

		job_name = sprintf('RF_%d',feature_num)
		
		command = sprintf('sbatch --job-name=%s --mem=64G -c 16 --time=120:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/activation_expression_models/classification/build_rf_models_ANOVA.R --feature_num %d"', job_name, feature_num)
		
		# print(command)
		system(command)
	
}

