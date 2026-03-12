
# Multiscale And Multimodal Subtyping of Major Depressive Disorder: Structure-Function Coupling Underlies Cognitive Heterogeneity

This repository contains code (Python scripts within *source_code/* and Jupyter notebooks within *notebooks/*) for the analyses and figures in the paper "Multiscale And Multimodal Subtyping of Major Depressive Disorder: Structure-Function Coupling Underlies Cognitive Heterogeneity" by Moldovan et al. (2026; manuscript in preparation). The code is organized as follows:

- *notebooks/*: Jupyter notebooks for the analyses and figures in the paper. These notebooks are organized in a way that reflects the structure of the paper, with separate notebooks for each major analysis and figure.

- *source_code/*: Python scripts that contain functions used in the notebooks. These scripts are organized by analysis type (e.g., global clustering, module clustering) and include both main scripts for running the analyses and utility scripts for supporting functions.

## Blogpost
Contains a blogpost written in Dutch (.pdf and .html formats) that provides a non-technical summary of the paper and its findings, intended for a general audience. The blogpost is designed to be accessible to readers without a background in neuroscience or data science, and it highlights the key insights and implications of the research in an engaging and informative way. The blogpost is included in the repository to facilitate broader dissemination and understanding of the research findings among the general public, and to provide a resource for those interested in learning more about the study without needing to engage with the technical details of the analyses.

## Notebooks

### `00_initial_cohort_selection.ipynb`

Performs the initial cohort selection. It filters the dataset based on predefined inclusion and exclusion criteria (ICD-10 codes, demographics, cognitive measures, rfMRI measures, dMRI measures) to create refined cohorts for further study. The steps include loading the dataset, applying filters, and saving the resulting cohorts (together with depression subjects' IDs with suffixes for neuroimaging data extraction, see `02_workstation_pull.ipynb`) for subsequent analysis. Important to note that this is a preliminary step and further refinement may be necessary based on additional criteria or data quality checks (see subsequent notebooks such as `01_cohort_matching.ipynb` and `05_aggregate_dMRI.ipynb`).

### `01_cohort_matching.ipynb`

Performs propensity score matching to create balanced cohorts. It uses predefined covariates (age and sex) to match individuals with depression (F32 ICD-10 major depressive disorder) to control subjects without depression (and actually without any known ICD-10 codes), aiding comparability between groups. The steps include loading the initial cohort, calculating propensity scores, performing the matching process, and saving the combined cohort for further analysis. Additionally, subject IDs with suffixes for neuroimaging data extraction (see `02_workstation_pull.ipynb`) are extracted and saved for the control cohort. This step has not been performed in the previous notebook because those initial controls were not matched to the depression cohort. 

### `02_workstation_pull.ipynb`

Demonstrates how to pull neuroimaging (rfMRI and dMRI) data of preselected subjects (`00_initial_cohort_selection.ipynb`; `01_cohort_matching.ipynb`) from a remote workstation to your local machine using bash commands, rsync and sshpass (for automatic password authentication). The notebook includes instructions for setting up the necessary tools and provides example commands for pulling the data. It also emphasizes the importance of securely handling credentials and ensuring that the data transfer is performed in compliance with relevant data protection regulations. This step is crucial for subsequent analyses that require access to the neuroimaging data, such as those performed in `04_merge_aggregate_rfMRI.ipynb` and `05_aggregate_dMRI.ipynb`. Due to the sensitive nature of the data and the need for secure handling, this notebook does not include actual commands with specific paths or credentials, but rather provides a template and guidelines for performing the data transfer from a remote workstation that has the necessary UK Biobank data access and permissions.

### `03_atlas_integration.ipynb`

A step-by-step explanation of how to merge two integer-labelled NIfTI atlases — one cortical, one subcortical — into a single combined atlas. In our case this has been done for the Schaefer 1000 (Schaefer et al., 2018) and the Tian S4 (Tian et al., 2020) atlases. The process involves loading the atlases, optionally resampling the subcortical atlas to match the cortical grid, calculating label offsets to avoid collisions, merging the label arrays according to a precedence rule, and saving the resulting combined atlas along with an optional CSV label-mapping table. The notebook also handles edge cases such as shape/affine mismatches and missing resampling backends, ensuring robustness throughout the integration process. Optional dependencies for resampling are noted, and debug logging is included to provide insights into the merging process. 

This integrated atlas is used throughout (almost) all subsequent analyses, including the merging and aggregation of rfMRI timeseries in `04_merge_aggregate_rfMRI.ipynb`, the aggregation of dMRI measures in `05_aggregate_dMRI.ipynb`, and the global and module-level clustering analyses in `06_global_clustering.ipynb` and `09_module_clustering.ipynb`, respectively.

### `04_merge_aggregate_rfMRI.ipynb`

Merges preprocessed and preparcellated (into 1000 cortical ROIs following Schaefer 1000 atlas and into 54 subcortical ROIs following Tian 54 atlas) resting-state fMRI time series for all subjects of the depressed and control cohorts defined in 00_inital_cohort_selection.ipynb and 01_cohort_matching.ipynb. This merging results in the creation of a single timeseries file per subject containing the concatenated Schaefer 1000 + Tian 54 timeseries. Handles NaN values in the timeseries data according to specified strategies (e.g., interpolating missing values). Saves this information into CSV files. Determines missing subjects for which timeseries data is unavailable. Saves this information into a CSV file.Computes individual functional connectivity matrices using Pearson correlation. Averages these matrices across subjects to obtain a group-level connectivity matrix. Arithmetic averaging is used by default to reduce memory usage. Saves individual merged timeseries, individual connectivity matrices, and the group average matrix. Visualizes the average connectivity matrix.

### `05_aggregate_dMRI.ipynb`

Aggregates structural connectivity data into one averaged structural connectivity matrix from diffusion MRI for all subjects of the depressed and control cohorts separately defined in `00_inital_cohort_selection.ipynb` and `01_cohort_matching.ipynb`. Additionally excludes subjects with missing functional or structural data from the combined cohort file and saves it for future use. The notebook includes steps for loading ROI labels (from `04_merge_aggregate_rfMRI.ipynb`) for visualization, computing the average structural connectome, saving QC CSVs (missing files, NaN values), plotting and saving the average structural connectivity matrix figure, and updating the combined cohort file to exclude subjects with missing data.

### `06_global_clustering.ipynb`

Performs the global-level MDD subtyping using global functional and structural connectivity and structure-function coupling features derived from per-subject connectomes. It is intended to identify data-driven subtypes within the depression cohort via hierarchical clustering (Ward linkage) on node-wise connectivity-strength or coupling vectors; quantify clustering validity and stability (internal metrics + bootstrap stability); compare cluster-derived groups and the depression group to controls, and compare clusters between themselves via quantile regression (R / `quantreg`) with covariate adjustment and multiple comparisons correction; visualize covariate distributions and violinplots, cluster assignments; produce CSVs, TXT summary files and figures summarizing the analyses. The identified clusters are then used for the cognitive association analyses in `07_global_cognitive_associations.ipynb` and for the confirmatory analyses in `11_global_clustering_confirmatory.ipynb`.

### `07_global_cognitive_associations.ipynb`

Runs the analysis comparing cognitive performance between the depression cohort and the control reference cohort, and global-level connectivity/coupling-derived clusters within the depression cohort against each other and against controlsusing robust (median/MAD) z-scoring and quantile regression (R/`quantreg` via `rpy2`). Includes covariate adjustment, multiple comparisons correction, and visualizations (e.g., radarplots depicting the cognitive performance profiles of the clusters and the entire depression cohort relative to controls). Also includes within the quantile regression analyses a measure of robust explained dispersion (median-based R2) to quantify how much of the cognitive performance dispersion is explained by the clusters. This measure is calculated within the between-cluster comparison models. 

### `08_LRG_community_detection.ipynb`

This notebook implements the community detection analysis using the Laplacian Renormalization Group (LRG) method on the depression population (average) structural connectivity data. The analysis includes loading and preprocessing the data, applying the LRG algorithm to identify communities within the depression population structural connectivity matrix, reordering the matrix based on the identified communities, reordering all individual (from depression AND control cohorts) structural and functional matrices according to this modular structure, and visualizing the results (including producing Nifti masks for the MRIcroGL visuals). The notebook relies on helper functions defined in `LRG_modularization_utils.py` for key steps in the analysis pipeline.

Additionaly, the modules derived from this notebook are functionally - using the cortical Yeo 7-network (Yeo et al., 2011) and Cole-Anticevic subcortical (Ji et al., 2019) parcellations - and anatomically - using the Anatomical Automated Labeling SPM12 atlas (Tzourio-Mazoyer et al., 2002) - described using our inhouse compneuro-tools package (Tellaetxe-Elorriaga, 2025). For information on how to install this package and use it, see [documentation](https://github.com/itellaetxe/compneuro_tools).

Identified modules are then used as the basis for the module-level clustering and cognitive association analyses in `09_module_clustering.ipynb` and `10_module_cognitive_associations.ipynb`, respectively, and for the confirmatory analyses in `12_module_clustering_confirmatory.ipynb`.

### `09_module_clustering.ipynb`

Implements a comprehensive pipeline for identifying depression subtypes based on module-level brain connectivity patterns. The analysis uses module internal (within-module) and external (between-module) functional connectivity (FC), structural connectivity (SC), and structure-function coupling (SFC) to discover patient subgroups using hierarchical clustering (Ward's linkage). The notebook includes steps for data preparation, clustering, cluster validation and stability (internal metrics + bootstrap stability), statistical inference (quantile regression via rpy2/R for depression vs control, clusters vs control and cluster vs cluster while adjusting for covariates and correcting for multiple comparisons), and visualization (notably, violinplots and brainmaps, and Nifti masks for MRIcroGL visuals). It produces CSVs, TXT summary files and figures summarizing the analyses. The identified clusters are then used for the cognitive association analyses in `10_module_cognitive_associations.ipynb` and for the confirmatory analyses in `12_module_clustering_confirmatory.ipynb`.

### `10_module_cognitive_associations.ipynb`

Runs the analysis comparing cognitive performance between the depression cohort and the control reference cohort, and module-level connectivity/coupling-derived clusters within the depression cohort against each other and against controls using robust (median/MAD) z-scoring and quantile regression (R/`quantreg` via `rpy2`). Includes covariate adjustment, multiple comparisons correction, and visualizations (e.g., radarplots depicting the cognitive performance profiles of the clusters and the entire depression cohort relative to controls). Also includes within the quantile regression analyses a measure of robust explained dispersion (median-based R2) to quantify how much of the cognitive performance dispersion is explained by the clusters. This measure is calculated within the between-cluster comparison models.

### `11_global_clustering_confirmatory.ipynb`

Executes the **confirmatory pipeline** for global-level connectivity-derived depression subtypes. It validates the structure-function coupling cluster structure produced by `06_global_clustering.ipynb` by re-testing — via quantile regression — whether the identified subtypes (Cluster 0 and Cluster 1) differ from Controls and from each other in terms of global structural and functional connectivity (and again for already analysed structure-function coupling), after adjusting for age, sex, head motion and selected ICD-10 comorbidities.

Structure-function coupling-derived clusters are chosen for the global-level confirmatory pipeline because they are the most clinically relevant in this project as they explain the most robust R2 dispersion (see `07_global_cognitive_associations.ipynb`), although they are the least stable (see bootstrap diagnostics in `06_global_clustering.ipynb`). 

### `12_module_clustering_confirmatory.ipynb`

Executes the **confirmatory pipeline** for module-level connectivity-derived depression subtypes. It validates the external (between-module) structure-function coupling cluster structure produced by `09_module_clustering.ipynb` by re-testing — via quantile regression — whether the identified subtypes (Cluster 0 and Cluster 1) differ from Controls and from each other in terms of global structural and functional connectivity (and again for already analysed structure-function coupling), after adjusting for age, sex, head motion and selected ICD-10 comorbidities. Here too, violinplots are used to visualize the distribution of connectivity/coupling values across groups, and brainmaps (also Nifti masks for MRIcroGL visuals) are used to visualize the spatial distribution of connectivity/coupling differences between clusters.

External structure-function coupling-derived clusters are chosen for the module-level confirmatory pipeline because they are the most clinically relevant in this project as they explain the most robust R2 dispersion (see `10_module_cognitive_associations.ipynb`), although they are the least stable (see bootstrap diagnostics in `09_module_clustering.ipynb`). 

### `13_MRIcroGL_visuals.ipynb`
This notebook only describes where to find the scripts to reproduce the brainmap visualizations in MRIcroGL (Rorden, 2025).

## Source code

Organized as follows:
- `atlas/`: contains utility functions (and a CLI interface) for the atlas integration process in `03_atlas_integration.ipynb`.

- `clinical_associations/`: contains utility functions (and main scripts if you wish to skip the notebooks) for the cognitive association analyses in `07_global_cognitive_associations.ipynb` and `10_module_cognitive_associations.ipynb`.

- `clusters/`: contains utility functions (and main scripts if you wish to skip the notebooks) for the global-level and module-level clustering analyses in `06_global_clustering.ipynb` and `09_module_clustering.ipynb`, respectively, and for the confirmatory analyses in `11_global_clustering_confirmatory.ipynb` and `12_module_clustering_confirmatory.ipynb`.

- `cohort_definition/`: contains utility functions (and main scripts if you wish to skip the notebooks) for the initial cohort selection in `00_initial_cohort_selection.ipynb` and for the propensity score matching in `01_cohort_matching.ipynb`.

- `connectivity_matrices/`: contains utility functions (and main scripts if you wish to skip the notebooks) for the merging and aggregation of rfMRI timeseries in `04_merge_aggregate_rfMRI.ipynb` and for the aggregation of dMRI measures in `05_aggregate_dMRI.ipynb`.

- `LRG_modularization/`: contains utility functions for the community detection analysis in `08_LRG_community_detection.ipynb`.

- `MRIcroGL_visuals/`: contains scripts for reproducing the brainmaps based on Nifti masks from `08_LRG_community_detection.ipynb`, `09_module_clustering.ipynb`, and `12_module_clustering_confirmatory.ipynb`. These scripts are intended to be run in the MRIcroGL environment, and they rely on the Nifti masks generated in the respective notebooks to visualize the spatial distribution of connectivity/coupling differences between clusters. See [documentation](https://github.com/rordenlab/MRIcroGL) for instructions on how to use these scripts and general instructions about this program.

## Reproducibility
Due to the strict data access and usage policies of the UK Biobank, the actual data used in this project cannot be shared publicly. However, all code necessary to reproduce the analyses and figures in the paper is available in this repository. To run the analyses, you will need to have access to the UK Biobank data and follow the instructions provided in the notebooks for data preparation and analysis. The notebooks are designed to be run sequentially, with each notebook building on the results of the previous ones. If you have any questions or encounter any issues while running the code, please do not hesitate to reach out.

## Correspondence
For any questions or inquiries regarding the code or analyses, please contact the corresponding author, Florin Valentin Moldovan, at [florinmoldo22@gmail.com](mailto:florinmoldo22@gmail.com).

## References

Ji, J. L., Spronk, M., Kulkarni, K., Repovš, G., Anticevic, A., & Cole, M. W. (2019). Mapping the human brain’s cortical-subcortical functional network organization. NeuroImage, 185, 35–57. https://doi.org/10.1016/j.neuroimage.2018.10.006

Rorden, C. (2025). MRIcroGL: voxel-based visualization for neuroimaging. Nature Methods. https://doi.org/10.1038/s41592-025-02763-7

Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X.-N., Holmes, A. J., Eickhoff, S. B., & Yeo, B. T. T. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. Cerebral Cortex, 28(9), 3095–3114. https://doi.org/10.1093/cercor/bhx179

Tellaetxe-Elorriaga, I. (2025). compneuro_tools. Github. https://github.com/itellaetxe/compneuro_tools

Thomas Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D., Hollinshead, M., Roffman, J. L., Smoller, J. W., Zöllei, L., Polimeni, J. R., Fischl, B., Liu, H., & Buckner, R. L. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. Journal of Neurophysiology, 106(3), 1125–1165. https://doi.org/10.1152/jn.00338.2011

Tian, Y., Margulies, D. S., Breakspear, M., & Zalesky, A. (2020). Topographic organization of the human subcortex unveiled with functional connectivity gradients. Nature Neuroscience, 23(11), 1421–1432. https://doi.org/10.1038/s41593-020-00711-6

Tzourio-Mazoyer, N., Landeau, B., Papathanassiou, D., Crivello, F., Etard, O., Delcroix, N., Mazoyer, B., & Joliot, M. (2002). Automated Anatomical Labeling of Activations in SPM Using a Macroscopic Anatomical Parcellation of the MNI MRI Single-Subject Brain. NeuroImage, 15(1), 273–289. https://doi.org/10.1006/nimg.2001.0978
