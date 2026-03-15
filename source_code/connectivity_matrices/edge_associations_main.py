"""
Edge association pipeline for UKB depression subtyping project. This script performs the following steps:
1. Loads cohort data (combined, depression, head motion) and prepares it for analysis.
2. For each connectivity type (functional, structural):
   a. Loads edge feature matrices for depression and control subjects, converting connectivity matrices to edge vectors.
   b. Runs per-node association analyses between edge features and covariates (age, motion, sex) within each cohort, applying FDR correction for multiple comparisons.
   c. Summarizes and saves the number of significant edges associated with each covariate in both cohorts, along with correlation ranges and medians.
3. Saves results to text files and generates validation plots for significant associations.

Expected inputs
---------------
- CSV cohort files:
    - `COMBINED_COHORT_PATH` with columns including `eid`, `depression_status`,
        `p21003_i2` (age), `p31` (sex), and head motion columns (`p24441_i2`, `p24453_i2`).
    - `DEPRESSION_COHORT_PATH` with at least `eid` and (optionally) motion columns.
    - `HEAD_MOTION_PATH` with `eid`, `p24441_i2`, `p24453_i2` for merging when missing.
- Connectivity matrices on disk under:
    - `DEPRESSION_DIR/<eid>/i2/` and `CONTROL_DIR/<eid>/i2/`.
    - Functional: `<eid>_connectivity.npy`.
    - Structural: `connectome_streamline_count_10M.csv.gz`.
- Environment variable (optional): `GC_BATCH_SIZE` to control matrix loading batch size.

Outputs
-------
For each `conn_type` in `CONNECTIVITY_TYPES` the script writes:
- Per-covariate correlation maps (CSV) for depression and control cohorts to
    `VALIDATION_PLOTS_BASE_DIR/<conn_type>_con/`.
- Manhattan-style association plots (PNG) for each covariate and cohort to the
    same validation directory.
- Summary text files in `GENERAL_DIR` named
    `<conn_type>_<covariate>_edge_association.txt`, containing counts and
    correlation summary statistics for FDR-significant edges.

Side effects
------------
- If motion columns are missing from cohort files, merged versions are saved
    back to `COMBINED_COHORT_PATH` and `DEPRESSION_COHORT_PATH`.
"""

import os
import sys
import numpy as np

from edge_associations_utils import (
    get_motion_columns,
    build_edge_feature_matrix_from_connectivity,
    run_per_edge_associations,
    describe_significant_edges,
    load_and_prepare_cohort_data
)

# Connectivity types supported by this residual pipeline
CONNECTIVITY_TYPES = ['functional', 'structural']

# Data paths 
GENERAL_DIR = '.../data/UKB/cohorts'
DEPRESSION_DIR = '.../data/UKB/F32_notask_STRCO_RSSCHA_RSTIA'
CONTROL_DIR = '.../data/UKB/control_notask_STRCO_RSSCHA_RSTIA'
COMBINED_COHORT_PATH = '.../data/UKB/cohorts/combined_cohort_F32.csv'
DEPRESSION_COHORT_PATH = '.../data/UKB/cohorts/depression_cohort_F32.csv'
HEAD_MOTION_PATH = '.../data/UKB/head_motion.csv.gz'
VALIDATION_PLOTS_BASE_DIR = '.../reports/plots/schaefer1000+tian54'

BATCH_SIZE = int(os.getenv('GC_BATCH_SIZE', '25'))

EDGE_COVARIATES = ['age', 'motion', 'sex']

fMRI_MOTION_METRIC = 'p24441_i2'
dMRI_MOTION_METRIC = 'p24453_i2'


def execute_edge_associations_for_conn_type(conn_type, cohort_data, edge_covariates):
    combined_data = cohort_data['combined_data']
    depression_subject_ids = list(cohort_data['depression_subject_ids'])
    control_subject_ids = list(cohort_data['control_subject_ids'])

    validation_plots_dir = os.path.join(VALIDATION_PLOTS_BASE_DIR, f'{conn_type}_con')
    os.makedirs(validation_plots_dir, exist_ok=True)

    motion_columns = get_motion_columns(conn_type, fMRI_MOTION_METRIC, dMRI_MOTION_METRIC)
    primary_motion_metric = next(iter(motion_columns.values()))

    print(f"\n[STEP 2] Loading {conn_type} connectivity data...")
    X_dep, n_nodes, dep_ids = build_edge_feature_matrix_from_connectivity(
        depression_subject_ids,
        DEPRESSION_DIR,
        conn_type,
        batch_size=BATCH_SIZE,
        cache_dir=None, # makes it in /tmp/edge_cache by default, can set to a specific path if desired
        prefix=f"{conn_type}_dep_edges",
        dtype=np.float32,
    )
    X_ctrl, n_nodes_ctrl, ctrl_ids = build_edge_feature_matrix_from_connectivity(
        control_subject_ids,
        CONTROL_DIR,
        conn_type,
        batch_size=BATCH_SIZE,
        cache_dir=None, # makes it in /tmp/edge_cache by default, can set to a specific path if desired
        prefix=f"{conn_type}_ctrl_edges",
        dtype=np.float32,
    )

    if X_dep is None or X_ctrl is None:
        print("  No connectivity matrices found for this modality; skipping.")
        return

    if n_nodes != n_nodes_ctrl:
        raise ValueError(f"Edge count mismatch dep={n_nodes} ctrl={n_nodes_ctrl}")
    depression_subject_ids = dep_ids
    control_subject_ids = ctrl_ids

    print(f"  Loaded edge vectors for {X_dep.shape[0]} depressed subjects")
    print(f"  Loaded edge vectors for {X_ctrl.shape[0]} control subjects")

    print(f"\n[STEP 3] Edge association analysis...")
    results =run_per_edge_associations(
        X_dep, X_ctrl, combined_data, depression_subject_ids, control_subject_ids,
        conn_type, validation_plots_dir, covariates=edge_covariates,
        motion_metric=primary_motion_metric, motion_metrics=motion_columns,
        analysis_level='matrix_edges', connectivity_metric='edge weight'
    )

    print("Number of edges FDR significantly associated with covariates:")
    os.makedirs(GENERAL_DIR, exist_ok=True)
    for cov in edge_covariates:
        print(f"  Covariate: {cov}")
        if cov == 'motion':
            cov = list(motion_columns.keys()) # because only one motion metric stored per modality
            print(f"    Using motion metric: {cov}")

        control_corr_map = results[f'{cov}_ctrl_map']
        depression_corr_map = results[f'{cov}_dep_map']
        count_control, percentage_control, rmin_control, rmax_control, median_r_control = describe_significant_edges(control_corr_map)
        count_depressed, percentage_depressed, rmin_depressed, rmax_depressed, median_r_depressed = describe_significant_edges(depression_corr_map)

        # Save to txt file
        fname = f"{conn_type}_{cov}_edge_association.txt"
        out_path = os.path.join(GENERAL_DIR, fname)
        with open(out_path, "w") as txt:
            txt.write("conn_type;covariate;cohort;count;percentage;rmin;rmax;median\n")
            txt.write(
                f"{conn_type};{cov};Control;"
                f"{int(count_control)};{percentage_control:.2f};"
                f"{rmin_control:.3f};{rmax_control:.3f};{median_r_control:.3f}\n"
            )
            txt.write(
                f"{conn_type};{cov};Depression;"
                f"{int(count_depressed)};{percentage_depressed:.2f};"
                f"{rmin_depressed:.3f};{rmax_depressed:.3f};{median_r_depressed:.3f}\n"
            )

        print(f"    Control cohort: {count_control} significant edges ({percentage_control:.2f}%) / range: [{rmin_control:.3f}, {rmax_control:.3f}], median: {median_r_control:.3f}")
        print(f"    Depression cohort: {count_depressed} significant edges ({percentage_depressed:.2f}%) / range: [{rmin_depressed:.3f}, {rmax_depressed:.3f}], median: {median_r_depressed:.3f}")


def main():
    print(f"\n{'='*80}")
    print("EDGE ASSOCIATION PIPELINE")
    print(f"Modalities queued: {', '.join(CONNECTIVITY_TYPES)}")
    print(f"{'='*80}\n")

    print("[STEP 1] Loading cohort data...")
    cohort_data = load_and_prepare_cohort_data(
        COMBINED_COHORT_PATH, DEPRESSION_COHORT_PATH, HEAD_MOTION_PATH, save_if_modified=True
    )
    print(f"  Depression subjects: {len(cohort_data['depression_subject_ids'])}")
    print(f"  Control subjects: {len(cohort_data['control_subject_ids'])}")

    for conn_type in CONNECTIVITY_TYPES:
        execute_edge_associations_for_conn_type(conn_type, cohort_data, EDGE_COVARIATES)

    print(f"\n{'='*80}")
    print("EDGE ASSOCIATION PIPELINE COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    sys.exit(main())
