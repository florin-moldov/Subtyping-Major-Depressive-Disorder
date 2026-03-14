"""Main orchestration for module-level depression–cognition association tests.

Purpose
-------
Central runner that coordinates data loading, robust z-scoring, one-
sample and between-group quantile regression (via R/quantreg through
`rpy2`), multiple-testing correction, and visualization for module-level
analyses. The script delegates statistical and plotting work to
``modular_cog_associations_utils`` and is responsible for wiring file
paths, logs, and high-level pipeline sequencing.

High-level pipeline (implemented as labelled steps in `main()`)
---------------------------------------------------------------
1) Initialize R environment (loads R packages and configures rpy2
     sinks so R console output is captured in a designated R log file).
2) Load and preprocess data (calls `load_and_rename_cohort_data`).
3) Compute task-wise robust z-scores referenced to the control group
     and run one-sample median tests via quantile regression (R).
4) Aggregate task z-scores into composite domain z-scores and test
     domain-level effects (also via quantile regression + FDR).
5) Create cohort-level and cluster-specific visualizations (violin
     plots, radar overlays, connectivity–cognition scatter figures).
6) For each connectivity modality/direction, build depressed-only
     concatenated tables for between-cluster comparisons and run
     between-cluster quantile regression tests (with FDR correction).

Inputs (expected files & columns)
---------------------------------
- `DATA_FILE` (default: `data/UKB/cohorts/module_connectivity_features_with_covariates.csv`)
    - Must contain subject identifier column `eid`.
    - Must contain `depression_status` encoded as 0 (control) and 1 (depressed).
    - If cluster-aware analysis is used, the file must include cluster
        assignment columns named like `<conn_type>_<dir>_cluster` (e.g.
        `functional_internal_cluster`) whose values are numeric/strings
        that can be normalized to `Cluster 0` and `Cluster 1`.
- Optionally the combined cohort CSV `data/UKB/cohorts/combined_cohort_<DEPRESSION_CODES>.csv`
    (used to add missing cognitive task columns to the module-level file).

Main configurable constants (edited in-script)
----------------------------------------
- `DEPRESSION`: list of ICD10 codes used to define the depressed cohort
    (used for filename templating).
- `COVARIATES`: list of covariate column names supplied to quantile
    regression models.
- `DOMAINS`: mapping of cognitive domains to lists of task column names
    (used to compute composite domain z-scores).

Outputs and side-effects
------------------------
- Plot images saved under `PLOTS_DIR` (paths include modality and
    cluster details).
- Text logs written to `data/UKB/cohorts/`:
    - R console and analysis summary -> `modular_R_analysis_summary_<DEPRESSION_CODES>.txt`
    - Multiple testing summaries -> `modular_multiple_testing_summary_<DEPRESSION_CODES>.txt`
    - Robust z-score computation logs -> `modular_robust_z_scores_summary_<DEPRESSION_CODES>.txt`
    - Composite z-score logs -> `modular_composite_z_scores_summary_<DEPRESSION_CODES>.txt`
- CSV of depression cohort z-scores -> `data/UKB/cohorts/depression_cohort_z_scores_<DEPRESSION_CODES>.csv`
- Temporary concatenated between-cluster CSVs placed in `/tmp` during
    between-cluster testing (e.g. `<DEPRESSION_CODES>_<conn>_<dir>_cluster_zscores_concatenated.csv`).

Integrations and helper functions used
-------------------------------------
- `setup_r_environment` : prepares R side (packages, sink redirection).
- `load_and_rename_cohort_data` : canonicalizes and loads the module-level
    CSV; renames columns expected by downstream helpers.
- `calculate_robust_z_scores` : computes median/MAD referenced z-scores
    and writes a text log.
- `quantile_regression` : wrapper around R `quantreg::rq` for point
    tests and bootstrap-based inference; writes R output to the R log.
- `apply_multiple_testing_correction` : performs FDR/Bonferroni/etc.
    multiple-testing correction and appends results to the MT log.
- `calculate_composite_z_score` : aggregates task z-scores into domain
    composites (median/mean) and logs the operation.
- Plotting helpers: `plot_z_scores`, `plot_cognitive_distributions_violin`,
    `plot_conn_cognition_association`, and `register_radar_overlay_significance`.

Assumptions, performance notes, and cautions
-------------------------------------------
- R and the `quantreg` package are required at runtime; quantile
    regression with bootstrap (`R` parameter) can be CPU- and time-
    intensive (the script sets `R=1000` in a number of places).
- The script assumes consistent column naming for cognitive tasks and
    covariates; mismatches will raise `KeyError` or `FileNotFoundError`.
- Many helpers write human-readable logs rather than printing to
    stdout — inspect the files listed above for complete diagnostics.

Usage
-----
Run directly from the repository root:

```bash
python source_code/plots/modular_cog_associations_main.py
```

Modify the constants near `main()` (e.g., `DEPRESSION`, `COVARIATES`,
`DATA_FILE`) to change cohorts, covariates, or input file locations.
"""

import os
import pandas as pd
from module_cog_associations_utils import (
    setup_r_environment,
    quantile_regression,
    load_and_rename_cohort_data,
    plot_conn_cognition_association,
    apply_multiple_testing_correction,
    calculate_robust_z_scores,
    calculate_composite_z_score,
    plot_z_scores,
    plot_cognitive_distributions_violin,
    register_radar_overlay_significance,
)

def main():
    """
    Main workflow for modular clusters cognitive associations analysis.
    """
    print("=" * 80)
    print("Modular Clusters Cognitive Associations Analysis")
    print("=" * 80)
    
    # Configuration: high-level constants and file locations
    ASSOCIATION_with = "cognition"  # analysis target (cognition by default)
    COVARIATES = ["age_at_assessment", "sex", "I10", "Z864", "F419"]  # covariates for quantile regression models
    DEPRESSION = ["F32"]  # ICD10 code(s) used to define the depression cohort
    DEPRESSION_CODES = "_".join(DEPRESSION)  # filename-friendly code string
    PLOTS_DIR = ".../reports/plots"  # output directory for figures
    os.makedirs(PLOTS_DIR, exist_ok=True)
    OUTPUT_DIR = ".../data/UKB/cohorts"      # output directory for logs and CSV
    DATA_DIR   = ".../data/UKB/cohorts"   # directory for logs and CSV outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DATA_FILE = os.path.join(DATA_DIR, "module_connectivity_features_with_covariates.csv")  # main input table

    # Ensure cognitive task columns exist in the module-level data file;
    # if missing, merge them from the combined cohort CSV
    data_main = pd.read_csv(DATA_FILE)
    selected_task_columns = [
        'p20023_i2',
        'p23324_i2',
        'p6348_i2',
        'p6350_i2',
        'p4282_i2',
        'p399_i2_a2',
        'p21004_i2',
        'p20197_i2',
        'p6373_i2',
        'p20016_i2',
        'p26302_i2'
    ]
    missing = [c for c in selected_task_columns if c not in data_main.columns]
    if missing:
        print("Adding selected cognitive tasks to the data file...")
        combined_cohort_csv = os.path.join(
            DATA_DIR,
            f"combined_cohort_{DEPRESSION_CODES}.csv"
        )
        if not os.path.exists(combined_cohort_csv):
            # Fail early if the expected combined cohort file is missing
            raise FileNotFoundError(f"Combined cohort CSV not found: {combined_cohort_csv}")
        data_tasks = pd.read_csv(combined_cohort_csv)
        data_tasks = data_tasks[['eid'] + selected_task_columns]

        # Drop any overlapping task columns from the module file before merging
        overlap = set(data_main.columns).intersection(data_tasks.columns) - {'eid'}
        if overlap:
            data_main = data_main.drop(columns=list(overlap))

        # Left-merge to preserve module-level rows, then overwrite the file
        data_merged = pd.merge(data_main, data_tasks, on='eid', how='left')
        data_merged.to_csv(DATA_FILE, index=False)
        print(f"Updated data file with cognitive tasks: {DATA_FILE}")

    # Text output logs (separate files per type of helper)
    # R console output + overall R analysis summary
    R_LOG_PATH = os.path.join(OUTPUT_DIR, f"modular_R_analysis_summary_{DEPRESSION_CODES}.txt")
    # Multiple testing summaries (aggregated across tests)
    MT_LOG_PATH = os.path.join(OUTPUT_DIR, f"modular_multiple_testing_summary_{DEPRESSION_CODES}.txt")
    # Robust z-score computation log
    ROBUST_Z_LOG_PATH = os.path.join(OUTPUT_DIR, f"modular_robust_z_scores_summary_{DEPRESSION_CODES}.txt")
    # Composite (domain) z-score computation log
    COMPOSITE_Z_LOG_PATH = os.path.join(OUTPUT_DIR, f"modular_composite_z_scores_summary_{DEPRESSION_CODES}.txt")

    GROUP_COLUMN = 'depression_status'

    if ASSOCIATION_with == "cognition":
        DOMAINS = {
            'processing_speed': ['Snap_task_mean_reaction_time', 'Symbol_digit_substitution_task_correct', 'Trail_making_A_duration', 'Trail_making_B_duration'],
            'working_memory': ['Reverse_number_recall_task_span', 'Pairs_matching_task_errors_6_pairs'],
            'executive_function': ['Trail_making_B_duration', 'Tower_rearranging_task_correct', 'Snap_task_mean_reaction_time'],
            'associative_learning': ['Paired_associates_learning_task_correct'],
            'abstract_reasoning': ['Matrix_pattern_completion_correct', 'Fluid_intelligence_score'],
            'verbal_ability': ['Vocabulary_score']
        } 

    # Bootstrap iterations for quantile regression (R) standard errors and confidence intervals
    QUANTILE_REGRESSION_BOOTSTRAP_R = 5000
    
    # -------------------------------------------------------------------------
    # Step 1: Initializing R environment once
    # -------------------------------------------------------------------------
    print("\n[1/6] Initializing R environment...")
    # Prepare R (install/load packages, set sinks for R console output)
    setup_r_environment()
    
    # -------------------------------------------------------------------------
    # Step 2: Load and preprocess data
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading and preprocessing data...")
    
    # Load canonicalized dataset and apply any renaming/normalization
    # (helper handles expected column names and simple quality checks)
    data = load_and_rename_cohort_data(DATA_FILE)

    n_controls_total = int((data[GROUP_COLUMN] == 0).sum())
    n_depressed_total = int((data[GROUP_COLUMN] == 1).sum())
    n_fc_internal0_total = ((data['functional_internal_cluster'] == '0').sum())
    n_fc_internal1_total = ((data['functional_internal_cluster'] == '1').sum())
    n_sc_internal0_total = ((data['structural_internal_cluster'] == '0').sum())
    n_sc_internal1_total = ((data['structural_internal_cluster'] == '1').sum())
    n_sfc_internal0_total = ((data['sfc_internal_cluster'] == '0').sum())
    n_sfc_internal1_total = ((data['sfc_internal_cluster'] == '1').sum())
    print(f"  N controls: {n_controls_total}")
    print(f"  N depressed: {n_depressed_total}")
    print(f"  N functional internal cluster 0: {n_fc_internal0_total}")
    print(f"  N functional internal cluster 1: {n_fc_internal1_total}")
    print(f"  N structural internal cluster 0: {n_sc_internal0_total}")
    print(f"  N structural internal cluster 1: {n_sc_internal1_total}")
    print(f"  N sfc internal cluster 0: {n_sfc_internal0_total}")
    print(f"  N sfc internal cluster 1: {n_sfc_internal1_total}")
    
    # -------------------------------------------------------------------------
    # Step 3: Calculate task-wise robust z-scores (referenced to control cohort) for depression cohort, reverse where needed
    # and significance and apply multiple testing correction
    # -------------------------------------------------------------------------
    print("\n[3/6] Calculating task-wise z-scores and testing significance...")
    
    # Deduplicate variables used across domains before z-scoring (avoid repeated stats rows)
    unique_z_vars = sorted(set(var for domain_vars in DOMAINS.values() for var in domain_vars))
    # Compute median/MAD-based robust z-scores for the full cohort
    z_scored_data = calculate_robust_z_scores(
        data,
        vars=unique_z_vars,
        group_column=GROUP_COLUMN,
        control_value=0,
        depression_value=1,
        log_path=ROBUST_Z_LOG_PATH,
        log_context="Overall cohort; connectivity_type: N/A; cluster: N/A"
    )

    # Reverse z-scores for variables where higher is worse
    vars_to_reverse = [
        'Snap_task_mean_reaction_time',
        'Trail_making_A_duration',
        'Trail_making_B_duration',
        'Pairs_matching_task_errors_6_pairs'
    ]
    for var in vars_to_reverse:
        z_scored_data[f'{var}_z'] = -z_scored_data[f'{var}_z']

    # One-sample median tests for each task z-score (uses R quantile regression)
    z_scored_data.to_csv('/tmp/data.csv', index=False)
    p_values_vars_z = quantile_regression(
        tmp_csv_path='/tmp/data.csv',
        dependent_variables=[f'{var}_z' for var in unique_z_vars],
        covariates=COVARIATES,
        group_column=None,
        tau=0.5,
        R=QUANTILE_REGRESSION_BOOTSTRAP_R,
        test_against_zero=True,
        return_effects=False,
        r_output_log_path=R_LOG_PATH
    )

    # Multiple testing correction for variable-level one-sample tests (FDR)
    apply_multiple_testing_correction(
        p_values=list(p_values_vars_z.values()),
        variable_names=[f'{var}_z' for var in unique_z_vars],
        test_methods=['Quantile Regression'] * len(unique_z_vars),
        method='fdr_bh',
        alpha=0.05,
        log_path=MT_LOG_PATH,
        log_context="Overall cohort; connectivity_type: N/A; cluster: N/A; level: task-wise"
    )

    # -------------------------------------------------------------------------
    # Step 4: Calculate composite robust z-score and test
    # significance and apply multiple testing correction
    # -------------------------------------------------------------------------
    print("\n[4/6] Calculating composite z-scores and testing significance...")
    
    for domain, vars_list in DOMAINS.items():
        z_vars_domain = [f'{var}_z' for var in vars_list]
        z_scored_data = calculate_composite_z_score(
            z_scored_data,
            z_vars=z_vars_domain,
            output_column=f'{domain}',
            method='median',
            log_path=COMPOSITE_Z_LOG_PATH,
            log_context=f"Overall cohort; connectivity_type: N/A; cluster: N/A; domain: {domain}"
        )

    # One-sample comparisons for composite domain scores
    z_scored_data.to_csv('/tmp/data.csv', index=False)
    p_values_domains_z = quantile_regression(
        tmp_csv_path='/tmp/data.csv',
        dependent_variables=list(DOMAINS.keys()),
        covariates=COVARIATES,
        group_column=None,
        tau=0.5,
        R=QUANTILE_REGRESSION_BOOTSTRAP_R,
        test_against_zero=True,
        return_effects=False,
        r_output_log_path=R_LOG_PATH
    )

    # Multiple testing correction for domain-level one-sample tests (FDR)
    apply_multiple_testing_correction(
        p_values=list(p_values_domains_z.values()),
        variable_names=[f'{domain}' for domain in DOMAINS.keys()],
        test_methods=['Quantile Regression'] * len(DOMAINS),
        method='fdr_bh',
        alpha=0.05,
        log_path=MT_LOG_PATH,
        log_context="Overall cohort; connectivity_type: N/A; cluster: N/A; level: domain-wise"
    )

    # -------------------------------------------------------------------------
    # Step 5: Visualize z-scores (individual and composite)
    # -------------------------------------------------------------------------
    print("\n[5/6] Creating z-score visualizations...")

    task_z_vars = [f'{var}_z' for var in unique_z_vars]
    domain_vars = [f'{domain}' for domain in DOMAINS.keys()]

    # Visualize individual z-scores
    plot_z_scores(
        z_scored_data,
        z_vars=task_z_vars,
        association_type=ASSOCIATION_with,
        type_z_score='task',
        overall_title=(
            'Task Z-Scores (Depression Cohort; '
            f'N_control={n_controls_total}, N_depression={n_depressed_total})'
        ),
        save_path=f'{PLOTS_DIR}/{DEPRESSION_CODES}_{ASSOCIATION_with}_task_z_scores.png'
    )

    # Visualize composite domain z-scores
    plot_z_scores(
        z_scored_data,
        z_vars=domain_vars,
        association_type=ASSOCIATION_with,
        type_z_score='domain',
        overall_title=(
            'Domain Composite Z-Scores (Depression Cohort; '
            f'N_control={n_controls_total}, N_depression={n_depressed_total})'
        ),
        save_path=f'{PLOTS_DIR}/{DEPRESSION_CODES}_{ASSOCIATION_with}_domain_z_scores.png'
    )

    # Save z-scored data
    output_file = os.path.join(OUTPUT_DIR, f'depression_cohort_z_scores_{DEPRESSION_CODES}.csv')
    z_scored_data.to_csv(output_file, index=False)
    print(f"\nZ-scored data saved to: {output_file}")

    # -------------------------------------------------------------------------
    # Step 6: Cluster specific cognitive dysfunction profiles (from modular
    # connectivity clusters -> see `modular_connectivity.py`)
    # -------------------------------------------------------------------------
    print("\n[6/6] Determining cognitive dysfunction profiles for modular connectivity clusters...")

    def _normalize_cluster_label(value):
        if pd.isna(value):
            return value
        value_str = str(value).strip()
        if value_str.lower().startswith('cluster'):
            return value_str
        try:
            return f"Cluster {int(float(value_str))}"
        except (ValueError, TypeError):
            return value_str

    for conn_type in ['functional', 'structural', 'sfc']:
        for dir_type in ['internal', 'external']:
            cluster_col = f"{conn_type}_{dir_type}_cluster"
            if cluster_col not in data.columns:
                print(f"{cluster_col} not found in final_df; skipping {conn_type} {dir_type}.")
                continue

            print(f"\nAnalyzing connectivity type: {conn_type} and direction: {dir_type}")

            # Collect per-cluster z-scored (tasks + domains) datasets so we can
            # concatenate them for between-cluster comparisons within this
            # connectivity type + direction.
            cluster_between_parts = []

            for cluster_label in ['0', '1']:
                cluster_name = f"Cluster {cluster_label}"
                print("\n" + "-" * 80)
                print(f"Running steps 2–5 for {cluster_name} (depression) vs controls")
                print("-" * 80)

                # Subset to all controls plus depressed subjects in the
                # current connectivity cluster.
                cluster_subset = data[
                    (data[GROUP_COLUMN] == 0) |
                    ((data[GROUP_COLUMN] == 1) & (data[cluster_col] == cluster_label))
                ].copy()

                cluster_subset['Cluster'] = cluster_subset[cluster_col].apply(_normalize_cluster_label)
                cluster_subset['Connectivity_Type'] = conn_type
                cluster_subset['Direction'] = dir_type

                n_controls_cl = int((cluster_subset[GROUP_COLUMN] == 0).sum())
                n_depressed_cl = int((cluster_subset[GROUP_COLUMN] == 1).sum())
                print(f"  N controls: {n_controls_cl}")
                print(f"  N depressed in {cluster_name}: {n_depressed_cl}")

                # Step 3: task-wise robust z-scores, one-sample tests (againt 0)
                unique_z_vars_cluster = sorted(set(var for domain_vars in DOMAINS.values() for var in domain_vars))
                z_scored_cluster = calculate_robust_z_scores(
                    cluster_subset,
                    vars=unique_z_vars_cluster,
                    group_column=GROUP_COLUMN,
                    control_value=0,
                    depression_value=1,
                    log_path=ROBUST_Z_LOG_PATH,
                    log_context=f"Within-cluster; connectivity_type: {conn_type}; direction: {dir_type}; cluster: {cluster_name}",
                )

                for var in [
                    "Snap_task_mean_reaction_time",
                    "Trail_making_A_duration",
                    "Trail_making_B_duration",
                    "Pairs_matching_task_errors_6_pairs",
                ]:
                    z_scored_cluster[f"{var}_z"] = -z_scored_cluster[f"{var}_z"]

                z_scored_cluster.to_csv("/tmp/data.csv", index=False)
                p_vars_cl = quantile_regression(
                    tmp_csv_path="/tmp/data.csv",
                    dependent_variables=[f"{var}_z" for var in unique_z_vars_cluster],
                    covariates=COVARIATES,
                    group_column=None,
                    tau=0.5,
                    R=QUANTILE_REGRESSION_BOOTSTRAP_R,
                    test_against_zero=True,
                    return_effects=False,
                    r_output_log_path=R_LOG_PATH
                )

                apply_multiple_testing_correction(
                    p_values=list(p_vars_cl.values()),
                    variable_names=[f"{var}_z" for var in unique_z_vars_cluster],
                    test_methods=["Quantile Regression"] * len(unique_z_vars_cluster),
                    method="fdr_bh",
                    alpha=0.05,
                    log_path=MT_LOG_PATH,
                    log_context=f"Within-cluster; connectivity_type: {conn_type}; direction: {dir_type}; cluster: {cluster_name}; level: task-wise",
                )

                # Step 4: composite domain z-scores and one-sample tests
                for domain, vars_list in DOMAINS.items():
                    z_vars_dom = [f"{var}_z" for var in vars_list]
                    z_scored_cluster = calculate_composite_z_score(
                        z_scored_cluster,
                        z_vars=z_vars_dom,
                        output_column=f"{domain}",
                        method="median",
                        log_path=COMPOSITE_Z_LOG_PATH,
                        log_context=f"Within-cluster; connectivity_type: {conn_type}; direction: {dir_type}; cluster: {cluster_name}; domain: {domain}",
                    )

                z_scored_cluster.to_csv("/tmp/data.csv", index=False)
                p_dom_cl = quantile_regression(
                    tmp_csv_path="/tmp/data.csv",
                    dependent_variables=list(DOMAINS.keys()),
                    covariates=COVARIATES,
                    group_column=None,
                    tau=0.5,
                    R=QUANTILE_REGRESSION_BOOTSTRAP_R,
                    test_against_zero=True,
                    return_effects=False,
                    r_output_log_path=R_LOG_PATH
                )
                apply_multiple_testing_correction(
                    p_values=list(p_dom_cl.values()),
                    variable_names=[f"{domain}" for domain in DOMAINS.keys()],
                    test_methods=["Quantile Regression"] * len(DOMAINS),
                    method="fdr_bh",
                    alpha=0.05,
                    log_path=MT_LOG_PATH,
                    log_context=f"Within-cluster; connectivity_type: {conn_type}; direction: {dir_type}; cluster: {cluster_name}; level: domain-wise",
                )

                task_z_vars_cl = [f"{var}_z" for var in unique_z_vars_cluster]
                domain_vars_cl = [f"{domain}" for domain in DOMAINS.keys()]

                # Save depressed-only, z-scored rows for later between-cluster comparisons.
                # (Cluster labels already exist in the clustered cohort CSV.)
                between_cols = COVARIATES + ["Cluster", "Connectivity_Type", 
                                             "Direction", 
                                             f"X1_0_{dir_type}_{conn_type}",
                                             f"X2_0_{dir_type}_{conn_type}",
                                             f"X3_0_{dir_type}_{conn_type}",
                                             f"X4_0_{dir_type}_{conn_type}",
                                             f"X5_0_{dir_type}_{conn_type}",
                                             f"X6_0_{dir_type}_{conn_type}",
                                             f"X7_0_{dir_type}_{conn_type}",
                                             f"X8_0_{dir_type}_{conn_type}",
                                             f"X9_0_{dir_type}_{conn_type}"] + task_z_vars_cl + domain_vars_cl
                between_cols = [c for c in between_cols if c in z_scored_cluster.columns]
                cluster_between_parts.append(
                    z_scored_cluster.loc[z_scored_cluster[GROUP_COLUMN] == 1, between_cols].copy()
                )

                # Step 5: visualizations (task-wise and domain-level) for this cluster
                plot_z_scores(
                    z_scored_cluster,
                    z_vars=task_z_vars_cl,
                    association_type=ASSOCIATION_with,
                    type_z_score=f"task_{cluster_name.replace(' ', '_')}",
                    overall_title=(
                        "Task Z-Scores (Depression Cohort, "
                        f"{cluster_name}; N_control={n_controls_cl}, "
                        f"N_depression={n_depressed_cl})"
                    ),
                    save_path=f"{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/modular_{DEPRESSION_CODES}_{ASSOCIATION_with}_task_z_scores_{cluster_name.replace(' ', '_')}_{dir_type}.png",
                )

                plot_z_scores(
                    z_scored_cluster,
                    z_vars=domain_vars_cl,
                    association_type=ASSOCIATION_with,
                    type_z_score=f"domain_{cluster_name.replace(' ', '_')}",
                    overall_title=(
                        "Domain Composite Z-Scores (Depression Cohort, "
                        f"{cluster_name}; N_control={n_controls_cl}, "
                        f"N_depression={n_depressed_cl})"
                    ),
                    save_path=f"{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/modular_{DEPRESSION_CODES}_{ASSOCIATION_with}_domain_z_scores_{cluster_name.replace(' ', '_')}_{dir_type}.png",
                )
            # Concatenate per-cluster temporary datasets so between-cluster
            # comparisons can be run on a single table per connectivity type + direction.
            if len(cluster_between_parts) >= 2:
                between_df = pd.concat(cluster_between_parts, ignore_index=True)
                between_path = os.path.join(
                    "/tmp",
                    f"{DEPRESSION_CODES}_{conn_type}_{dir_type}_cluster_zscores_concatenated.csv",
                )
                between_df.to_csv(between_path, index=False)
                print(f"  Concatenated cluster z-score dataset saved to: {between_path}")

                # Visualize connectivity–cognition task associations for this connectivity type
                # Cluster specific 
                for mod in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']:
                    plot_conn_cognition_association(
                        data=between_df,
                        connectivity_var=f"{mod}_0_{dir_type}_{conn_type}",
                        cognitive_vars=[f'{var}_z' for var in unique_z_vars_cluster],
                        group_column='Cluster',
                        save_path=f'{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/modular_{mod}_{dir_type}_{DEPRESSION_CODES}_{ASSOCIATION_with}_task_connectivity.png',
                        overall_title=(
                        f'{mod} {conn_type.capitalize()} {dir_type.capitalize()} Connectivity vs Cognitive Z-Scores'
                        )
                    )

                # Cluster-only violin plots: task-wise z-scores and domain composites
                # Save separate PNGs per connectivity type/direction to the plots folder.
                try:
                    plot_cognitive_distributions_violin(
                        data=between_df,
                        variables=[f"{var}_z" for var in unique_z_vars_cluster],
                        group_column="Cluster",
                        plot_depression_clusters=True,
                        cluster_column="Cluster",
                        conn_type=conn_type,
                        dir_type=dir_type,
                        save_path=f'{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/modular_{DEPRESSION_CODES}_{ASSOCIATION_with}_task_clusters_violin_{dir_type}.png',
                        title=f'Task z-score distributions by Cluster ({conn_type} {dir_type})',
                    )
                except Exception as e:
                    print(f"  Warning: failed to create task cluster violin plot: {e}")

                try:
                    plot_cognitive_distributions_violin(
                        data=between_df,
                        variables=list(DOMAINS.keys()),
                        group_column="Cluster",
                        plot_depression_clusters=True,
                        cluster_column="Cluster",
                        conn_type=conn_type,
                        dir_type=dir_type,
                        save_path=f'{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/modular_{DEPRESSION_CODES}_{ASSOCIATION_with}_domain_clusters_violin_{dir_type}.png',
                        title=f'Domain composite distributions by Cluster ({conn_type} {dir_type})',
                    )
                except Exception as e:
                    print(f"  Warning: failed to create domain cluster violin plot: {e}")

                # Between-cluster quantile regression analyses for task-wise z-scores 
                # within this connectivity type (with FDR correction)
                p_vars_cls = quantile_regression(
                    tmp_csv_path=between_path,
                    dependent_variables=[f"{var}_z" for var in unique_z_vars_cluster],
                    covariates=COVARIATES,
                    group_column="Cluster",         
                    reference_group="Cluster 1",
                    comparison_groups=("Cluster 0",),
                    test_against_zero=False,
                    return_effects=False,
                    tau=0.5,
                    R=QUANTILE_REGRESSION_BOOTSTRAP_R,
                    r_output_log_path=R_LOG_PATH
                )

                reject_vars_cls, pvals_vars_cls = apply_multiple_testing_correction(
                    p_values=list(p_vars_cls.values()),
                    variable_names=[f"{var}_z" for var in unique_z_vars_cluster],
                    test_methods=["Quantile Regression"] * len(unique_z_vars_cluster),
                    method="fdr_bh",
                    alpha=0.05,
                    log_path=MT_LOG_PATH,
                    log_context=f"Between-cluster; connectivity_type: {conn_type}; direction: {dir_type}; clusters: Cluster 1 (ref) vs Cluster 0; level: task-wise",
                )

                register_radar_overlay_significance(
                    depression_codes=DEPRESSION_CODES,
                    association_type=ASSOCIATION_with,
                    base_kind="task",
                    conn_type=conn_type,
                    dir_type=dir_type,
                    variable_names=[f"{var}_z" for var in unique_z_vars_cluster],
                    pvals_corrected=pvals_vars_cls,
                    comparison_label="Cluster 0 vs Cluster 1 (between-cluster, FDR)",
                )

                # Visualize connectivity–cognition domain associations for this connectivity type
                # Cluster specific 
                for mod in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']:
                    plot_conn_cognition_association(
                        data=between_df,
                        connectivity_var=f"{mod}_0_{dir_type}_{conn_type}",
                        cognitive_vars=list(DOMAINS.keys()),
                        group_column='Cluster',
                        save_path=f'{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/modular_{mod}_{dir_type}_{DEPRESSION_CODES}_{ASSOCIATION_with}_domain_connectivity.png',
                        overall_title=(
                        f'{mod} {conn_type.capitalize()} {dir_type.capitalize()} Connectivity vs Cognitive Z-Scores'
                        )
                    )

                # Between-cluster quantile regression analyses for domain composite z-scores 
                # within this connectivity type (with FDR correction)
                p_dom_cls = quantile_regression(
                    tmp_csv_path=between_path,
                    dependent_variables=list(DOMAINS.keys()),
                    covariates=COVARIATES,
                    group_column="Cluster",         
                    reference_group="Cluster 1",
                    comparison_groups=("Cluster 0",),
                    test_against_zero=False,
                    return_effects=False,
                    tau=0.5,
                    R=QUANTILE_REGRESSION_BOOTSTRAP_R,
                    r_output_log_path=R_LOG_PATH
                )
                reject_dom_cls, pvals_dom_cls = apply_multiple_testing_correction(
                    p_values=list(p_dom_cls.values()),
                    variable_names=[f"{domain}" for domain in DOMAINS.keys()],
                    test_methods=["Quantile Regression"] * len(DOMAINS),
                    method="fdr_bh",
                    alpha=0.05,
                    log_path=MT_LOG_PATH,
                    log_context=f"Between-cluster; connectivity_type: {conn_type}; direction: {dir_type}; clusters: Cluster 1 (ref) vs Cluster 0; level: domain-wise",
                )

                register_radar_overlay_significance(
                    depression_codes=DEPRESSION_CODES,
                    association_type=ASSOCIATION_with,
                    base_kind="domain",
                    conn_type=conn_type,
                    dir_type=dir_type,
                    variable_names=[f"{domain}" for domain in DOMAINS.keys()],
                    pvals_corrected=pvals_dom_cls,
                    comparison_label="Cluster 0 vs Cluster 1 (between-cluster, FDR)",
                )

            else:
                print("  Skipping cluster dataset concatenation (missing one or more clusters).")
    
    print("=" * 80)
    print("\nAnalysis complete!")
    print(f"Saved following outputs:")
    print(f" - Plots directory: {PLOTS_DIR}")
    print(f" - Logs:\n{R_LOG_PATH}\n{MT_LOG_PATH}\n{ROBUST_Z_LOG_PATH}\n{COMPOSITE_Z_LOG_PATH}")


if __name__ == "__main__":
    main()