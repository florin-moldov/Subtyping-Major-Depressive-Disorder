"""Global-level depression subtyping and validation pipeline.

Purpose
-------
This script performs an end-to-end analysis of global connectivity/coupling features
derived from per-subject connectomes. It is intended to:
- identify data-driven subtypes within the depression cohort via hierarchical
  clustering on node-wise connectivity-strength / coupling vectors,
- quantify clustering validity and reproducibility (internal metrics +
  bootstrap stability),
- compare cluster-derived groups and the depression group to controls via
  quantile regression (R / `quantreg`) with covariate adjustment,
- visualize covariate distributions and cluster assignments, and
- produce CSVs and figures summarizing the analyses.

Expected inputs
---------------
- Per-subject connectivity data (organized under cohort folders):
  - Functional: `<eid>/i2/<eid>_connectivity.npy` (2D NumPy array, correlation
    or z-transformed matrix).
  - Structural: `<eid>/i2/connectome_streamline_count_10M.csv.gz` (square CSV of
    streamline counts per node).
- Cohort CSVs (combined and depression-only):
  - `COMBINED_COHORT_PATH` – CSV with per-subject covariates (must contain at
    least `eid`, `depression_status`, `p21003_i2` (age), `p31` (sex)).
  - `DEPRESSION_COHORT_PATH` – CSV for the depression cohort (same expected
    covariates; script will attempt to merge head-motion columns if missing).
- Head motion CSV: `HEAD_MOTION_PATH` (may be compressed). Expected to contain
  `p24441_i2` (fMRI) and/or `p24453_i2` (dMRI) per `eid`.
- Optional: pre-computed combined cluster CSVs for cross-modality agreement.

High-level steps (what the script does, in order)
-----------------------------------------------
1. Setup & I/O
  - Read configuration constants (paths, connectivity types, ICD covariates).
  - Prepare output directories under `VALIDATION_PLOTS_BASE_DIR` / `FIGURES_BASE_DIR`.

2. Load cohorts and head-motion
  - Read combined and depression cohort CSVs.
  - If head-motion columns (`p24441_i2` or `p24453_i2`) are missing, the
    script merges them from `HEAD_MOTION_PATH` and optionally writes the
    merged cohort CSV back to disk when `save_if_modified=True`.

3. Compute per-node connectivity strengths
  - For each subject in the target cohort (depressed subjects), load the
    appropriate connectivity matrix (functional or structural) and compute
    a per-node scalar "strength" (weighted degree normalized by n_nodes-1).
  - For Structure-Function Coupling (sfc), the script computes per-node
    Pearson correlations between FC and SC profiles (returned as SFC vectors).
  - Important to note: any NaN values in the resulting feature vectors (e.g., from constant rows in FC/SC for SFC) 
    are tracked and saved to a CSV for transparency. The clustering step will proceed with NaNs replaced by 
    zeros, but the NaN info can help identify subjects with potentially unreliable features.

4. Pre-clustering covariate association diagnostics (optional)
  - Compute per-node correlations (age, sex, motion) to identify strongly
    associated nodes prior to clustering. Produces CSVs and Manhattan-style
    plots. Tests for normality per-node to choose Pearson vs Spearman.

5. Hierarchical clustering (depression cohort)
  - Perform Ward linkage hierarchical clustering on subject-by-node strength
      feature matrix and cut the tree to produce two clusters (k=2).
  - Labels are normalized to 0/1 and dendrogram SVG is saved.

6. Clustering validation
  - Compute silhouette and Calinski-Harabasz scores for a range of k (2..20)
    and save CSVs and a validation figure.
  - Run bootstrap stability analysis (default 500 iterations):
    - resample subjects with replacement, recluster, align labels (flip if
      necessary), compute NMI and Jaccard, and tally per-subject stability.
    - Save per-subject stability CSV and a combined diagnostics figure.

7. Cross-modality agreement (if multiple modalities available)
  - If combined cluster CSVs exist for multiple connectivity types, the
    script computes pairwise agreement (proportion of identical labels),
    writes a wide-format assignments CSV, agreement matrix CSV, and plots
    (barplots, heatmap) summarizing cross-type consistency.

8. Covariate distribution plotting
  - Produce multi-panel figures summarizing Age, Sex, Head Motion, and
    ICD-10 comorbidity counts by Group and by Cluster. Performs
    Mann-Whitney and Chi-square tests, applies FDR correction, and
    annotates significance on plots. CSVs with raw and corrected p-values
    are written for transparency.
  - Also, a TXT summary table (`global_covariate_distribution_summary.txt`) is 
    generated that aggregates numeric summaries (n/mean/std/median/Q1/Q3 for 
    continuous variables and counts/percentages for binary variables) along with raw 
    and FDR-corrected p-values for each comparison.

9. Quantile (median) regression (R via rpy2)
  - Build a combined DataFrame for quantile regression with:
    - scalar connectivity summary per subject (mean of node strengths),
    - `Group` label ('Control' / 'Depression'), and
    - `Cluster` label ('Control', 'Cluster 0', 'Cluster 1').
  - Run four R quantile regression contrasts using `quantreg` and
    bootstrap standard errors (default R=1000):
    * Depression vs Control
    * Cluster 0 vs Control
    * Cluster 1 vs Control
    * Cluster 0 vs Cluster 1
  - Extract coefficient p-values, apply FDR correction, and save results.

10. Visualization of group comparisons
  - Create violin plots showing distributions for Control, Depression,
    Cluster 0 and Cluster 1 (colored consistently). Overlay FDR-corrected
    significance brackets corresponding to the contrasts above.

11. Outputs produced (examples)
  - CSVs:
    - `combined_cohort_F32_global_{conn_type}_connectivity_clusters.csv`
    - `global_{conn_type}_bootstrap_diagnostics.csv`
    - `global_{conn_type}_cluster_stability.csv`
    - per-node correlation CSVs and covariate-test CSVs under `out_dir`
  - Figures (SVG):
    - dendrogram: `{conn_type}_con/individual_avg_conn_dendrogram.svg`
    - bootstrap diagnostics combined SVG
    - covariate distribution multi-panel SVGs
    - violin plot: `{conn_type}_con/individual_avg_conn_clusters_violinplot.svg`
    - cross-modality agreement heatmap and barplots
  - Text tables:
    - `global_covariate_distribution_summary.txt` with numeric summaries and 
      test results for covariates.
  - Logs:
    - Console output is captured to `global_{conn_type}_connectivity_output.txt`.

Important notes and assumptions
-------------------------------
- The script expects consistent `eid` identifiers across the cohort CSVs,
  head-motion file, and per-subject connectivity folders.
- The quantile regression step requires a working R installation and the R
  package `quantreg`. The script attempts to install it
  via R if missing (may require network access and write permissions).
- Many steps write files to disk; ensure `VALIDATION_PLOTS_BASE_DIR`,
  `FIGURES_BASE_DIR`, and `COHORTS_DIR` are writable by the executing user.
- Defaults: bootstrap iterations = 500 (stability), quantile regression
  bootstrap R = 1000. These can be reduced for faster testing.

Adjust configuration constants near the top of the script to point to
local data paths if needed.
"""

import os
import sys
import io
from contextlib import contextmanager
import pandas as pd
import numpy as np
import sys
sys.path.append('source_code')  # Ensure source_code is in the Python path for imports

# Import all utility functions
from clusters.global_clustering_utils import (
    setup_r_environment,
    load_single_connectivity_matrix,
    compute_node_strength_from_matrix,
    compute_sfc_features_from_matrices,
    plot_dendrogram,
    perform_hierarchical_clustering,
    plot_bootstrap_diagnostics,
    ensure_2d_array,
    subject_scalar_summary,
    get_motion_columns,
    load_and_prepare_cohort_data,
    perform_clustering_validation,
    save_clustering_results,
    analyze_cross_modality_agreement,
    create_combined_dataframe,
    merge_combined_cohort_connectivity_clusters,
    determine_covariate_distributions,
    run_quantile_regression,
    apply_multiple_testing_correction,
    create_violin_plot_with_significance,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Connectivity types to analyze in a single run
CONNECTIVITY_TYPES = ['functional', 'structural', 'sfc']

# ICD-10 codes to include as covariates in quantile regression (comorbidity adjustment)
ICD_COVARIATES = ['I10', 'Z864', 'F419'] # these are the most common comorbidities in depression cohort, based on cohort_selection_main.py

# Progress reporting cadence (subjects)
PROGRESS_EVERY = int(os.getenv('GC_PROGRESS_EVERY', '50'))

# Data paths
DEPRESSION_DIR = '...data/UKB/F32_notask_STRCO_RSSCHA_RSTIA'
CONTROL_DIR = '...data/UKB/control_notask_STRCO_RSSCHA_RSTIA'
GENERAL_DIR = '...data/UKB/cohorts'
COMBINED_COHORT_PATH = '...data/UKB/cohorts/combined_cohort_F32.csv'
DEPRESSION_COHORT_PATH = '...data/UKB/cohorts/depression_cohort_F32.csv'
HEAD_MOTION_PATH = '...data/UKB/head_motion.csv.gz'

# Output directories
VALIDATION_PLOTS_BASE_DIR = '.../reports/plots/schaefer1000+tian54'
FIGURES_BASE_DIR = '.../reports/figures/schaefer1000+tian54'
COHORTS_DIR = '.../data/UKB/cohorts'

# Motion metrics (column names in head_motion.csv)
fMRI_MOTION_METRIC = 'p24441_i2'  # Framewise displacement for fMRI
dMRI_MOTION_METRIC = 'p24453_i2'  # Mean relative head motion for dMRI

# Dependent variable name in dataset for quantile regression
DEPENDENT_VAR = 'Connectivity'
# Column names for cluster and group labels in dataset for quantile regression
CLUSTER_COL = 'Cluster'
GROUP_COL = 'Group'

# Bootstrap iterations for quantile regression (R) standard errors and confidence intervals
QUANTILE_REGRESSION_BOOTSTRAP_R = 5000
# Bootstrap iterations for clustering stability analysis
CLUSTER_STABILITY_BOOTSTRAP_ITER = 5000

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

def execute_pipeline_for_conn_type(conn_type, cohort_data, validation_plots_dir,
                   figures_base_dir, motion_columns, primary_motion_metric, fdr_log_path,
                   icd_covariates, dependent_var, cluster_col, group_col):
  """Run the full pipeline for a specific connectivity modality."""

  combined_data = cohort_data['combined_data']
  depression_subject_ids = list(cohort_data['depression_subject_ids'])
  control_subject_ids = list(cohort_data['control_subject_ids'])

  # ------------------------------------------------------------------
  # STEP 3: Load connectivity data
  # ------------------------------------------------------------------
  print(f"\n[STEP 3/9] Computing {conn_type} connectivity measure data...")
  # ------------------------------------------------------------------
  # STEP 3a: Load connectivity matrices and compute per-node global features
  # ------------------------------------------------------------------
  # For 'functional' and 'structural' modalities we load single connectivity
  # matrices per subject and compute a node-wise strength vector.
  if conn_type in ('functional', 'structural'):
    dep_features = []   # list of per-subject node-strength vectors (depression)
    dep_ids = []        # corresponding subject IDs (keeps only subjects with data)
    total_dep = len(depression_subject_ids)
    for i, sid in enumerate(depression_subject_ids, start=1):
      # Attempt to load the subject's connectivity matrix; may return None if missing
      mat = load_single_connectivity_matrix(sid, DEPRESSION_DIR, conn_type)
      if mat is None:
        # Skip subjects without a valid matrix
        continue
      # Compute node strength (global feature) from the connectivity matrix
      dep_features.append(compute_node_strength_from_matrix(mat, conn_type=conn_type))
      dep_ids.append(sid)
      # Periodic progress reporting to keep the user informed on long runs
      if PROGRESS_EVERY > 0 and (i % PROGRESS_EVERY == 0 or i == total_dep):
        print(f"  Progress: {i}/{total_dep} {conn_type} depression subjects")

    # Repeat for control cohort
    ctrl_features = []  # list of per-subject node-strength vectors (control)
    ctrl_ids = []
    total_ctrl = len(control_subject_ids)
    for i, sid in enumerate(control_subject_ids, start=1):
      mat = load_single_connectivity_matrix(sid, CONTROL_DIR, conn_type)
      if mat is None:
        continue
      ctrl_features.append(compute_node_strength_from_matrix(mat, conn_type=conn_type))
      ctrl_ids.append(sid)
      if PROGRESS_EVERY > 0 and (i % PROGRESS_EVERY == 0 or i == total_ctrl):
        print(f"  Progress: {i}/{total_ctrl} {conn_type} control subjects")

    # If either group has no matrices, there's nothing to cluster for this modality
    if not dep_features or not ctrl_features:
      print("  No connectivity matrices found for this modality; skipping.")
      return

    # Replace the subject ID lists with the filtered lists (subjects with data)
    depression_subject_ids = dep_ids
    control_subject_ids = ctrl_ids

    # Basic sanity check: all node-strength vectors should have identical length
    n_nodes = dep_features[0].shape[0]
    if any(vec.shape[0] != n_nodes for vec in ctrl_features):
      raise ValueError(f"Node count mismatch dep={n_nodes} ctrl={ctrl_features[0].shape[0]}")

    print(f"  Loaded matrices for {len(dep_features)} depressed subjects")
    print(f"  Loaded matrices for {len(ctrl_features)} control subjects")

    # Expose for downstream steps: hierarchical clustering & summaries
    global_features_depression = dep_features
    global_features_control = ctrl_features

  # ------------------------------------------------------------------
  # STEP 3b: Structure-Function Coupling (SFC)
  # ------------------------------------------------------------------
  # For SFC we need both functional and structural matrices for the same subject.
  # We compute a coupling vector from paired FC/SC matrices.
  else:
    dep_features = []
    dep_ids = []
    total_dep = len(depression_subject_ids)
    sfc_nan_info_dep = pd.DataFrame({"eid": [], "percent_nan": []})  # To track % of NaNs in SFC vectors per depression subject
    for i, sid in enumerate(depression_subject_ids, start=1):
      # Load both FC and SC; skip subject if either modality is missing
      fc_mat = load_single_connectivity_matrix(sid, DEPRESSION_DIR, 'functional')
      sc_mat = load_single_connectivity_matrix(sid, DEPRESSION_DIR, 'structural')
      if fc_mat is None or sc_mat is None:
        continue
      # compute_sfc_features_from_matrices expects lists and returns
      # a list of per-subject vectors plus percent_nan. Unpack the
      # returned list and append the first (and only) vector so that
      # downstream code receives a 1D NumPy array (with .shape).
      sfc_list, percent_nan = compute_sfc_features_from_matrices([fc_mat], [sc_mat])
      sfc_vec = sfc_list[0]
      sfc_nan_info_dep = pd.concat([sfc_nan_info_dep, pd.DataFrame({"eid": [sid], "percent_nan": [percent_nan]})], ignore_index=True)
      dep_features.append(sfc_vec)
      dep_ids.append(sid)
      if PROGRESS_EVERY > 0 and (i % PROGRESS_EVERY == 0 or i == total_dep):
        print(f"  Progress: {i}/{total_dep} Structure-Function Coupling depression subjects")

    ctrl_features = []
    ctrl_ids = []
    total_ctrl = len(control_subject_ids)
    sfc_nan_info_ctrl = pd.DataFrame({"eid": [], "percent_nan": []})  # To track % of NaNs in SFC vectors per control subject
    for i, sid in enumerate(control_subject_ids, start=1):
      fc_mat = load_single_connectivity_matrix(sid, CONTROL_DIR, 'functional')
      sc_mat = load_single_connectivity_matrix(sid, CONTROL_DIR, 'structural')
      if fc_mat is None or sc_mat is None:
        continue
      sfc_list, percent_nan = compute_sfc_features_from_matrices([fc_mat], [sc_mat])
      sfc_vec = sfc_list[0]
      sfc_nan_info_ctrl = pd.concat([sfc_nan_info_ctrl, pd.DataFrame({"eid": [sid], "percent_nan": [percent_nan]})], ignore_index=True)
      ctrl_features.append(sfc_vec)
      ctrl_ids.append(sid)
      if PROGRESS_EVERY > 0 and (i % PROGRESS_EVERY == 0 or i == total_ctrl):
        print(f"  Progress: {i}/{total_ctrl} Structure-Function Coupling control subjects")

    # Concatenate and save NaN info for transparency; this can help identify if certain subjects have unreliable SFC estimates due to many NaN values (e.g., from constant FC/SC profiles)
    sfc_nan_info = pd.concat([sfc_nan_info_dep, sfc_nan_info_ctrl], ignore_index=True)
    sfc_nan_info.to_csv(os.path.join(GENERAL_DIR, 'sfc_vector_nan_info.csv'), index=False)

    # Need overlapping FC+SC subjects in both groups to compute SFC clusters
    if not dep_features or not ctrl_features:
      print("  Missing overlapping FC/SC subjects; skipping Structure-Function Coupling.")
      return

    # Update subject ID lists to the subset with both modalities present
    depression_subject_ids = dep_ids
    control_subject_ids = ctrl_ids

    # Ensure vector lengths match across groups
    n_nodes = dep_features[0].shape[0]
    if any(vec.shape[0] != n_nodes for vec in ctrl_features):
      raise ValueError("FC/SC node count mismatch for Structure-Function Coupling")

    print(f"  Loaded FC+SC matrices for {len(dep_features)} depressed subjects")
    print(f"  Loaded FC+SC matrices for {len(ctrl_features)} control subjects")

    # Expose for downstream steps: hierarchical clustering & summaries
    global_features_depression = dep_features
    global_features_control = ctrl_features

  # ------------------------------------------------------------------
  # STEP 4: Hierarchical clustering
  # ------------------------------------------------------------------
  print("\n[STEP 4/9] Hierarchical clustering (Ward linkage, k=2)...")
  X_mat = ensure_2d_array(np.asarray(global_features_depression))
  Z, clusters = perform_hierarchical_clustering(X_mat, n_clusters=2)

  print("  Cluster sizes:")
  print(f"    Cluster 0: {np.sum(clusters == 0)}")
  print(f"    Cluster 1: {np.sum(clusters == 1)}")

  plot_dendrogram(Z, conn_type, figures_base_dir)

  subject_scalar_depression = subject_scalar_summary(global_features_depression)
  subject_scalar_control = subject_scalar_summary(global_features_control)

  # ------------------------------------------------------------------
  # STEP 5: Clustering validation
  # ------------------------------------------------------------------
  print("\n[STEP 5/9] Clustering validation...")
  validation_results = perform_clustering_validation(X_mat, Z, clusters, conn_type, validation_plots_dir, CLUSTER_STABILITY_BOOTSTRAP_ITER)

  save_clustering_results(
    validation_results['stability_results'], clusters, subject_scalar_depression,
    conn_type, validation_plots_dir
  )

  plot_bootstrap_diagnostics(
    validation_results['stability_results'], clusters, conn_type,
    validation_plots_dir, analysis_level='global'
  )

  # ------------------------------------------------------------------
  # STEP 6: Cross-connectivity-type cluster agreement
  # ------------------------------------------------------------------
  print("\n[STEP 6/9] Cross-connectivity-type cluster agreement...")
  available_types = analyze_cross_modality_agreement(COHORTS_DIR, validation_plots_dir)

  # ------------------------------------------------------------------
  # STEP 7: Covariate distribution visualization
  # ------------------------------------------------------------------
  print("\n[STEP 7/9] Covariate distribution visualization...")
  combined_df = create_combined_dataframe(
    control_subject_ids, depression_subject_ids,
    subject_scalar_control, subject_scalar_depression,
    clusters, combined_data, conn_type=conn_type
  )

  determine_covariate_distributions(
    combined_df,
    available_types,
    conn_type,
    primary_motion_metric,
    validation_plots_dir,
    COHORTS_DIR,
    icd_covariates,
  )

  combined_df.to_csv(
    os.path.join(COHORTS_DIR, f'combined_cohort_F32_global_{conn_type}_connectivity_clusters.csv'),
    index=False
  )

  # ------------------------------------------------------------------
  # STEP 8: Quantile regression analysis (R)
  # ------------------------------------------------------------------
  print("\n[STEP 8/9] Quantile regression analysis...")
  motion_covariate_list = list(motion_columns.values())
  results = run_quantile_regression(
    combined_df,
    conn_type,
    icd_covariates,
    motion_covariates=motion_covariate_list,
    tau=0.5,
    R=QUANTILE_REGRESSION_BOOTSTRAP_R,
    dependent_var=dependent_var,
    cluster_col=cluster_col,
    group_col=group_col,
  )

  _, corrected_p_values = apply_multiple_testing_correction(
    p_values=list(results['p_values'].values()),
    test_methods=[
      'QR (median) depression vs control',
      'QR (median) cluster 0 vs control',
      'QR (median) cluster 1 vs control',
      'QR (median) cluster 0 vs cluster 1'
    ],
    variable_names=[f'Global {conn_type.capitalize()} Connectivity'] * 4,
    log_path=fdr_log_path
  )

  # ------------------------------------------------------------------
  # STEP 9: Violin plot with significance brackets
  # ------------------------------------------------------------------
  print("\n[STEP 9/9] Creating violin plot with significance brackets...")
  create_violin_plot_with_significance(
    subject_scalar_control,
    subject_scalar_depression,
    clusters,
    corrected_p_values,
    conn_type,
    os.path.join(validation_plots_dir, 'individual_avg_conn_clusters_violinplots.svg')
  )

  print(f"\n{'-'*80}")
  print(f"Completed {conn_type} connectivity analysis")
  print(f"{'-'*80}")


def run_pipeline_for_conn_type(conn_type, cohort_data):
  """Wrapper that sets up per-modality paths, logging, and execution."""

  validation_plots_dir = os.path.join(VALIDATION_PLOTS_BASE_DIR, f'{conn_type}_con')
  figures_dir = os.path.join(FIGURES_BASE_DIR, f'{conn_type}_con')
  os.makedirs(validation_plots_dir, exist_ok=True)
  os.makedirs(figures_dir, exist_ok=True)

  general_log_path = os.path.join(DEPRESSION_DIR, f'global_{conn_type}_connectivity_output.txt')
  fdr_log_path = os.path.join(DEPRESSION_DIR, f'global_{conn_type}_connectivity_FDR.txt')
  motion_columns = get_motion_columns(conn_type, fMRI_MOTION_METRIC, dMRI_MOTION_METRIC)
  primary_motion_metric = next(iter(motion_columns.values()))

  display_conn = "Structure-Function Coupling" if str(conn_type).lower() == "sfc" else str(conn_type).upper()
  print(f"\n{'='*80}")
  print(f"GLOBAL CONNECTIVITY PIPELINE — {display_conn}")
  print(f"{'='*80}\n")

  with capture_stdout_to_log(general_log_path):
    execute_pipeline_for_conn_type(
      conn_type=conn_type,
      cohort_data=cohort_data,
      validation_plots_dir=validation_plots_dir,
      figures_base_dir=FIGURES_BASE_DIR,
      motion_columns=motion_columns,
      primary_motion_metric=primary_motion_metric,
      fdr_log_path=fdr_log_path,
      icd_covariates=ICD_COVARIATES,
      dependent_var=DEPENDENT_VAR,
      cluster_col=CLUSTER_COL,
      group_col=GROUP_COL,
    )

  print(f"Saved run log to: {general_log_path}")
  print(f"Saved FDR correction log to: {fdr_log_path}\n")


def main():
  """Execute the complete global connectivity clustering pipeline for all modalities."""

  print(f"\n{'='*80}")
  print("Global-level Depression Subtyping Pipeline")
  display_modalities = [
    "Structure-Function Coupling" if str(ct).lower() == "sfc" else str(ct)
    for ct in CONNECTIVITY_TYPES
  ]
  print(f"Modalities queued: {', '.join(display_modalities)}")
  print(f"{'='*80}\n")

  print("[STEP 1/9] Initializing R environment once...")
  setup_r_environment()
  print("R packages loaded: quantreg")

  print("\n[STEP 2/9] Loading cohort data...")
  cohort_data = load_and_prepare_cohort_data(
    COMBINED_COHORT_PATH, DEPRESSION_COHORT_PATH, HEAD_MOTION_PATH, save_if_modified=True
  )
  print(f"  Depression subjects: {len(cohort_data['depression_subject_ids'])}")
  print(f"  Control subjects: {len(cohort_data['control_subject_ids'])}")

  for conn_type in CONNECTIVITY_TYPES:
    run_pipeline_for_conn_type(conn_type, cohort_data)

  merged_df = merge_combined_cohort_connectivity_clusters(COHORTS_DIR, conn_types=CONNECTIVITY_TYPES)
  merged_path = os.path.join(COHORTS_DIR, 'global_merged_connectivity_clusters.csv')
  merged_df.to_csv(merged_path, index=False)
  print(f"Saved merged connectivity clusters to: {merged_path}")

  print(f"\n{'='*80}")
  print("ALL MODALITIES COMPLETE")
  print(f"{'='*80}\n")


if __name__ == "__main__":
  main()

