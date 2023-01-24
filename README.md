# Modeling Responses to Kinase Inhibitors using Multi-Assay Kinome Inhibition States
This repository contains code for processing kinobeads and kinomescan data for cell line response prediction, as outlines in the paper: [insert link]

## Prerequesites 
This repository contains code written almost entirely in R, using Rstudio and 'Tidyverse' idioms. The file [`package_check.R`](package_check.R) describes all the R packages needed to run the code in a covenient "pacman" script. Running this script will install all the required packages in one go. 

## Repository Structure 
This repository is divided into three main folders:
* [`src`](src): source code for generating all results and figures
* [`data`](data): raw data used by the source code (included in zenodo)
* [`results`](results): results generated by source code (not included here)
* [`figures`](figures): figures generated by source code 

## Data Organization
The folder [`src/data_organization`](src/data_organization) contains code to process kinome profiling data from kinobeads and KINOMEscan assays, and link it to cell line responses and baseline multi-omics data. 

## Modeling 
The folder [`src/LINCS_modeling`](src/LINCS_modeling) contains code to build machine learning models using the combined dataset, predicting outcomes of [`IC50`](src/LINCS_modeling/build_LINCS_klaeger_ic50_models) and [`AUC`](src/LINCS_modeling/build_LINCS_klaeger_auc_models). This also includes code to process experimental data and validate model predictions. 

## Figure Building 
The majority of figures included in the paper are produced directly in the model-building and analysis code, however the code for some specific visualizations can be found in the folder [`src/LINCS_modeling/figure_building`](src/LINCS_modeling/figure_building)

## Figures
All the figures published in the paper can be found in the folder [`figures`](figures) and some specific figures generated as part of model building code can be found in [`figures/PRISM_LINCS_klaeger`](figures/PRISM_LINCS_klaeger)



