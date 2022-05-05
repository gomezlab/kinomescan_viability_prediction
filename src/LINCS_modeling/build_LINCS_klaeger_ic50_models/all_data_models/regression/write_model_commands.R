library(tidyverse)
library(here)

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) { 
	job_name = sprintf('1i%d',feature_num) 
	
	dir.create(here('src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/keras_model_tuning/model_commands/one_layer'), 
						 showWarnings = F, recursive = T)
	
	full_output_file = here('src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/keras_model_tuning/model_commands/one_layer', 
													sprintf('%dfeat.sh',feature_num))
	
	writeLines(c("#!/bin/bash",
							 "#SBATCH -N 1",
							 "#SBATCH -n 1",
							 "#SBATCH -p volta-gpu",
							 "#SBATCH -t 120:00:00",
							 "#SBATCH --qos gpu_access",
							 "#SBATCH --mem=30G",
							 "#SBATCH --gres=gpu:1",
							 "#SBATCH --constraint='rhel8'",
							 sprintf("#SBATCH --job-name=%s", job_name),
							 "module load r/4.1.0",
							 "module load gcc/9.1.0",
							 "module load cuda/11.4",
							 sprintf("Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/one_layer.R --feature_num %d", feature_num)
	), 
	here(full_output_file))
}

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) { 
	job_name = sprintf('4i%d',feature_num) 
	
	dir.create(here('src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/keras_model_tuning/model_commands/two_layer'), 
						 showWarnings = F, recursive = T)
	
	full_output_file = here('src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/keras_model_tuning/model_commands/two_layer', 
													sprintf('%dfeat.sh',feature_num))
	
	writeLines(c("#!/bin/bash",
							 "#SBATCH -N 1",
							 "#SBATCH -n 1",
							 "#SBATCH -p volta-gpu",
							 "#SBATCH -t 120:00:00",
							 "#SBATCH --qos gpu_access",
							 "#SBATCH --mem=30G",
							 "#SBATCH --gres=gpu:1",
							 "#SBATCH --constraint='rhel8'",
							 sprintf("#SBATCH --job-name=%s", job_name),
							 "module load r/4.1.0",
							 "module load gcc/9.1.0",
							 "module load cuda/11.4",
							 sprintf("Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/two_layer.R --feature_num %d", feature_num)
	), 
	here(full_output_file))
}

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) { 
	job_name = sprintf('1i%d',feature_num) 
	
	dir.create(here('src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/keras_model_tuning/model_commands/four_layer'), 
						 showWarnings = F, recursive = T)
	
	full_output_file = here('src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/keras_model_tuning/model_commands/four_layer', 
													sprintf('%dfeat.sh',feature_num))
	
	writeLines(c("#!/bin/bash",
							 "#SBATCH -N 1",
							 "#SBATCH -n 1",
							 "#SBATCH -p volta-gpu",
							 "#SBATCH -t 120:00:00",
							 "#SBATCH --qos gpu_access",
							 "#SBATCH --mem=30G",
							 "#SBATCH --gres=gpu:1",
							 "#SBATCH --constraint='rhel8'",
							 sprintf("#SBATCH --job-name=%s", job_name),
							 "module load r/4.1.0",
							 "module load gcc/9.1.0",
							 "module load cuda/11.4",
							 sprintf("Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/four_layer.R --feature_num %d", feature_num)
	), 
	here(full_output_file))
}

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) { 
	job_name = sprintf('6i%d',feature_num) 
	
	dir.create(here('src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/keras_model_tuning/model_commands/six_layer'), 
						 showWarnings = F, recursive = T)
	
	full_output_file = here('src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/regression/keras_model_tuning/model_commands/six_layer', 
													sprintf('%dfeat.sh',feature_num))
	
	writeLines(c("#!/bin/bash",
							 "#SBATCH -N 1",
							 "#SBATCH -n 1",
							 "#SBATCH -p volta-gpu",
							 "#SBATCH -t 120:00:00",
							 "#SBATCH --qos gpu_access",
							 "#SBATCH --mem=30G",
							 "#SBATCH --gres=gpu:1",
							 "#SBATCH --constraint='rhel8'",
							 sprintf("#SBATCH --job-name=%s", job_name),
							 "module load r/4.1.0",
							 "module load gcc/9.1.0",
							 "module load cuda/11.4",
							 sprintf("Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models/all_data_models/regression/keras_model_tuning/six_layer.R --feature_num %d", feature_num)
	), 
	here(full_output_file))
}