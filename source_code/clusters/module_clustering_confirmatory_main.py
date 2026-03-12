"""
module: module_clustering_confirmatory_main.py
-------------------------------------------------
Top-level orchestration for module-level confirmatory analyses.

Purpose
-------
This script runs confirmatory tests that assess whether modules (from a
previous modularization step) show differences in connectivity features across
SFC-derived clusters and depression status. It coordinates:
 - R-based quantile regression (median, tau=0.5) with bootstrapped SEs for
     several group contrasts per module
 - multiple-testing correction (FDR-BH) across all tests within a modality
 - generation of annotated violin plots and module-level brainmaps

High-level steps
----------------
1. Load per-subject module connectivity features and covariates (CSV defined
     by `DATA`). The DataFrame must contain `eid`, `depression_status` (0/1),
     cluster label column (e.g., `sfc_external_cluster`) and module feature
     columns of the form `<module>_{direction}_{conn}` (e.g. `X9_0_external_functional`).
2. Initialize the R environment used by `rpy2` and confirm required R
     packages (quantreg, boot) are available.
3. Run `run_module_quantile_regression_pipeline(...)` which:
     - iterates modules × directions × connectivity modalities,
     - runs three quantile regression models per module to obtain p-values for
         four contrasts (depression vs control, cluster0 vs control, cluster1 vs
         control, cluster0 vs cluster1),
     - applies Benjamini–Hochberg FDR correction across all module tests per
         (connectivity, direction) family,
     - returns a `significance_map` usable for plotting and brainmap filtering.
4. Generate module-level violin plots annotated with FDR-corrected
     significance and save figures.
5. Generate module-level brainmaps (cluster medians and difference maps) for
     modules that pass the cluster0_vs_cluster1 FDR threshold.

Main expected inputs (configured near top of file)
--------------------------------------------
- `DATA` (pd.DataFrame loaded from CSV): per-subject features + covariates (from `module_clustering_main.py`).
- `colname_map` (optional CSV -> dict): mapping original -> sanitized names
    used when passing variable names to R.
- `COMMUNITY_LABELS_PATH`: text file of community labels used for brainmaps.
- `ATLAS_FILE`: NIfTI parcellation matching `community_labels` ordering.

Primary outputs
---------------
- Plain-text combined stdout log capturing both Python and R output (saved
    to `DEPRESSION_DIR/module_connectivity_output_sfc_external.txt`).
- Per-family FDR logs (text) written to `depressed_subjects_dir`.
- Annotated violin PNGs saved under `PLOTS_DIR` (filename template
    controlled by plotting utilities).
- Brainmap PNGs (and optional NIfTI masks) saved under `FIGURES_DIR`.

Runtime requirements and notes
-----------------------------
- Python environment with: pandas, numpy, rpy2, seaborn, matplotlib, nilearn,
    scikit-learn, and their dependencies.
- R installation with `quantreg` and `boot` packages available to rpy2.
- The pipeline writes a temporary CSV that R reads; running multiple
    instances in parallel without isolation may cause file contention.
- Bootstrapped SEs default to a high iteration count; set
    `QUANTILE_REGRESSION_BOOTSTRAP_R` lower for iterative development.

"""

import os
import sys
import io
from contextlib import contextmanager
import numpy as np
import pandas as pd

sys.path.append('source_code')

from clusters.module_clustering_confirmatory_utils import (
    setup_r_environment,
    run_module_quantile_regression_pipeline,
    plot_cluster_feature_brainmaps,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONNECTIVITY_TYPES = ('functional', 'structural', 'sfc')
DIRECTION_TYPES = ('internal', 'external')
ICD_10_COVARIATES = ['I10', 'Z864', 'F419']  # Hypertension, history of psychoactive substance abuse, Anxiety disorder

DEPRESSION_DIR = '.../F32_notask_STRCO_RSSCHA_RSTIA'

# Main data file originating from module_clustering_main.py that contains all module-level features and covariates for the confirmatory analysis.
DATA = pd.read_csv('.../module_connectivity_features_with_covariates.csv') 

PLOTS_DIR = '.../schaefer1000+tian54'
FIGURES_DIR = '.../schaefer1000+tian54'

# Motion metrics (column names in head_motion.csv)
fMRI_MOTION_METRIC = 'p24441_i2'  # Framewise displacement for fMRI
dMRI_MOTION_METRIC = 'p24453_i2'  # Mean relative head motion for dMRI

# Community labels and number of communities/modules from earlier LRG modularization
COMMUNITY_LABELS_PATH = '.../notebooks/CM.txt'
CM = np.loadtxt(COMMUNITY_LABELS_PATH)
mods = np.unique(CM)

# Path to Schaefer1000 + TianS4 combined atlas used for brainmap visualizations
ATLAS_FILE = '.../Schaefer1000_TianS4_combined.nii.gz'

# Bootstrap iterations for quantile regression (R) standard errors and confidence intervals
QUANTILE_REGRESSION_BOOTSTRAP_R = 1000

# Cluster column within DATA for which we want to confirmatory check associations with all 
# module-level features. This should be a column in the DATA CSV that contains cluster labels
# obtained from the exploratory clustering analysis ('module_clustering_main.py'). 
# In our case, 'sfc_external_cluster'
CLUSTER_COL = 'sfc_external_cluster'

# Mapping of original feature names to sanitized column names (if needed), see module_clustering_main and *_utils for details. 
# This CSV should contain at least 'original' and 'sanitized' columns. 
# The 'sanitized' names are used in the DataFrame and R scripts, while the 
# 'original' names are for reference and mapping back to modules. 
# The code will attempt to map module labels to sanitized column name prefixes 
# using this mapping, and will fall back to a safe transformation if no mapping is 
# found. This allows for flexible handling of feature names that may not 
# be valid Python identifiers or R variable names, while still maintaining 
# a connection to the original module labels.
colname_map_df = pd.read_csv(
    '.../module_connectivity_with_covariates_colname_map.csv'
)
if not {'original', 'sanitized'}.issubset(colname_map_df.columns):
    raise ValueError("colname_map CSV must contain 'original' and 'sanitized' columns")
colname_map = dict(zip(colname_map_df['original'], colname_map_df['sanitized']))

# ==============================================================================
# OUTPUT CAPTURE SETUP
# ==============================================================================
class Tee(io.TextIOBase):
    """File-like object that writes to both stdout and buffer."""

    def __init__(self, original, buffer):
        self.original = original
        self.buffer = buffer

    def write(self, s):
        self.original.write(s)
        self.buffer.write(s)
        return len(s)

    def flush(self):
        self.original.flush()
        self.buffer.flush()


@contextmanager
def capture_stdout_to_log(log_path):
    """Mirror stdout to a buffer and persist it to disk when the block ends."""
    original_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = Tee(original_stdout, buffer)
    try:
        yield buffer
    finally:
        sys.stdout = original_stdout
        with open(log_path, 'w') as log_file:
            log_file.write(buffer.getvalue())


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("=" * 80)
    print("Module-level Depression Subtyping Pipeline - Confirmatory Analysis")
    print("=" * 80)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Initialize R environment
    # ------------------------------------------------------------------
    print("\n[STEP 1/2] Initializing R environment...")
    setup_r_environment()

    # ------------------------------------------------------------------
    # STEP 2: Run module quantile regression pipeline and get significance map 
    # for brainmap visualization
    # ------------------------------------------------------------------
    print("\n[STEP 2/2] Running module quantile regression pipeline...")
    brainmap_significance = run_module_quantile_regression_pipeline(
        merged=DATA,
        final_df=DATA,
        mods=mods,
        cluster_col=CLUSTER_COL,
        R=QUANTILE_REGRESSION_BOOTSTRAP_R,
        conn_types=CONNECTIVITY_TYPES,
        dir_types=DIRECTION_TYPES,
        icd_covariates=ICD_10_COVARIATES,
        fMRI_MOTION_METRIC=fMRI_MOTION_METRIC,
        dMRI_MOTION_METRIC=dMRI_MOTION_METRIC,
        plots_dir=PLOTS_DIR,
        depressed_subjects_dir=DEPRESSION_DIR,
        colname_map=colname_map,
    )

    plot_cluster_feature_brainmaps(
        final_df=DATA,
        feature_df=DATA,
        cluster_col=CLUSTER_COL,
        atlas_img_path=ATLAS_FILE,
        community_labels=CM,
        fig_dir=FIGURES_DIR,
        modules=mods,
        minmax=True,
        significance_map=brainmap_significance,
        pvalue_threshold=0.05,
    )

    print("\nAnalysis complete!")
    print(f"Saved following outputs:")
    print(f" - Log of quantile regression results and diagnostics: {os.path.join(DEPRESSION_DIR, 'module_connectivity_output_sfc_external.txt')}")
    print(f" - FDR log: {os.path.join(DEPRESSION_DIR,'modular_*conn_type*_*dir_type*_connectivity_FDR_sfc_external.txt')}")
    print(f" - Brainmap figures directory: {FIGURES_DIR}")
    print(f" - Plots directory: {PLOTS_DIR}")


if __name__ == '__main__':
    log_path = os.path.join(DEPRESSION_DIR, 'module_connectivity_output_sfc_external.txt')
    with capture_stdout_to_log(log_path):
        main()