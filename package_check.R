#!/usr/bin/env Rscript

# to find all the library commands run:
#  grep -Rh library * | sort | uniq
# 
# then reformat the library calls to use p_load as below, plus dealing with the github only packages

if("pacman" %in% rownames(installed.packages()) == FALSE) {
	install.packages("pacman")
}

library(pacman)

p_load(Boruta)
p_load(EFS)
p_load(Metrics)
p_load(PharmacoGx)
p_load(ROCR)
p_load(argparse)
p_load(broom)
p_load(brulee)
p_load(conflicted)
p_load(cowplot)
p_load(data.table)
p_load(doParallel)
p_load(dr4pl)
p_load(fastshap)
p_load(finetune)
p_load(geneSynonym)
p_load(ggdendro)
p_load(gghighlight)
p_load(ggridges)
p_load(ggupset)
p_load(here)
p_load(hexbin)
p_load(janitor)
p_load(keras)
p_load(pROC)
p_load(pacman)
p_load(patchwork)
p_load(readxl)
p_load(recipeselectors)
p_load(reticulate)
p_load(tabnet)
p_load(tfdatasets)
p_load(tictoc)
p_load(tidyHeatmap)
p_load(tidymodels)
p_load(tidyverse)
p_load(torch)
p_load(umap)
p_load(vip)
p_load(vroom)
p_load(webchem)
p_load(wesanderson)
p_load(xgboost)

p_load_gh('mbergins/BerginskiRMisc')
p_load_gh('IDG-Kinase/DarkKinaseTools')
