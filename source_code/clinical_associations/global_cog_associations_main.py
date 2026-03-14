"""Main orchestration script for global-level depression–cognition associations.

This script runs the end-to-end analysis used in the project to
compare cognitive performance between a depression cohort and a
control reference cohort, between clusters and the control reference cohort and 
between clusters themselves, using robust (median/MAD) z-scoring and
quantile regression (R/quantreg via rpy2). It calls helper routines in
``global_cog_associations_utils`` to perform the statistical work and
plotting; the main responsibilities here are orchestration, file
paths, logging configuration, and combining per-cluster datasets for
between-cluster tests.

Key behaviors and expectations
------------------------------
- Input: expects a combined cohort CSV at ``data/UKB/cohorts/combined_cohort_<DEPRESSION_CODES>.csv``
  that contains a column named ``depression_status`` encoded as numeric
  0 (control) and 1 (depressed). When clustered cohorts are present
  the script also reads per-modality clustered CSVs named like
  ``combined_cohort_<DEPRESSION_CODES>_global_<modality>_connectivity_clusters.csv``
  which must contain a ``Cluster`` column containing the strings
  ``Cluster 0`` and ``Cluster 1`` for depressed-subject cluster labels.
- Output: writes plots to ``reports/plots/*``, several human-readable
  text logs under ``data/UKB/cohorts/``, and a CSV of z-scored
  depression-cohort rows under ``data/UKB/cohorts/depression_cohort_z_scores_<DEPRESSION_CODES>.csv``.
- Analysis steps: (1) compute task-wise robust z-scores referenced to
  controls; (2) test one-sample median-vs-zero via quantile regression
  (R); (3) aggregate task z-scores into composite domain z-scores and
  test them; (4) visualize cohort-wide and cluster-specific profiles;
  (5) run between-cluster quantile regressions on concatenated per-
  modality depressed-only z-scored rows.

Design notes
------------
- The script avoids printing detailed statistical summaries to stdout;
  instead, the helper functions append text to log files so the full
  analysis record is captured on disk. R console output is directed to
  a designated R log file by the rpy2-invoked R blocks.
- This file focuses on orchestration and leaves numerical details (MAD
  handling, robust z-scoring, plotting details, quantile-regression
  R code) to the ``global_cog_associations_utils`` helpers.

"""

import os
import pandas as pd
from global_cog_associations_utils import (
    setup_r_environment,
    quantile_regression,
    load_and_rename_cohort_data,
    plot_conn_cognition_association,
    apply_multiple_testing_correction,
    calculate_robust_z_scores,
    calculate_composite_z_score,
    plot_z_scores,
    register_radar_overlay_significance,
    plot_cognitive_distributions_violin
)

def main():
    """
    Main workflow for depression associations analysis.
    """
    print("=" * 80)
    print("Depression Associations Analysis")
    print("=" * 80)

    # ----------------------
    # Configuration
    # ----------------------
    # `ASSOCIATION_with` indicates which association domain is being
    # evaluated; currently only 'cognition' is supported by this script.
    ASSOCIATION_with = "cognition"

    # Covariates passed to the quantile regression models. ICD-10 codes
    # (upper-case names) are included here when available in the cohort CSV.
    COVARIATES = ["age_at_assessment", "sex", "I10", "Z864", "F419"]

    # Depression cohort identifier(s) (ICD-10 codes). These are joined to
    # form filenames so multiple codes can be handled if needed.
    DEPRESSION = ["F32"]  # ICD10 code used for identifying depression cohort
    DEPRESSION_CODES = "_".join(DEPRESSION)

    # Output locations
    PLOTS_DIR = ".../reports/plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    DATA_DIR = ".../cohorts"
    OUTPUT_DIR = ".../cohorts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Expected input: a combined cohort CSV containing a numeric
    # `depression_status` column encoded as 0=control, 1=depressed.
    DATA_FILE = os.path.join(DATA_DIR, f"combined_cohort_{DEPRESSION_CODES}.csv")

    # Text output logs (separate files per type of helper)
    R_LOG_PATH = os.path.join(OUTPUT_DIR, f"R_analysis_summary_{DEPRESSION_CODES}.txt")
    MT_LOG_PATH = os.path.join(OUTPUT_DIR, f"multiple_testing_summary_{DEPRESSION_CODES}.txt")
    ROBUST_Z_LOG_PATH = os.path.join(OUTPUT_DIR, f"robust_z_scores_summary_{DEPRESSION_CODES}.txt")
    COMPOSITE_Z_LOG_PATH = os.path.join(OUTPUT_DIR, f"composite_z_scores_summary_{DEPRESSION_CODES}.txt")

    # The name of the column used to indicate control vs depression.
    # Expected encoding in input CSV: 0 -> control, 1 -> depression.
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
    setup_r_environment()
    
    # -------------------------------------------------------------------------
    # Step 2: Load and preprocess data
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading and preprocessing data...")

    # Load cohort CSV and standardize column names to the project's
    # canonical names (helper does renaming). The helper will raise or
    # return a DataFrame with expected columns when available.
    data = load_and_rename_cohort_data(DATA_FILE)

    # Count subjects by depression status (assumes 0/1 encoding)
    n_controls_total = int((data[GROUP_COLUMN] == 0).sum())
    n_depressed_total = int((data[GROUP_COLUMN] == 1).sum())

    # -------------------------------------------------------------------------
    # Step 3: Calculate task-wise robust z-scores (referenced to control cohort) 
    # for depression cohort, reverse where needed, test significance, 
    # and apply multiple testing correction
    # -------------------------------------------------------------------------
    print("\n[3/6] Calculating task-wise z-scores and testing significance...")
    # Determine the union of all cognitive variables used across domains so
    # we compute each z-score only once.
    unique_z_vars = sorted(set(var for domain_vars in DOMAINS.values() for var in domain_vars))

    # Quick diagnostic: plot raw distributions before z-scoring to help
    # identify gross distributional differences or data issues.
    plot_cognitive_distributions_violin(
        data=data,
        variables=unique_z_vars,
        group_column=GROUP_COLUMN,
        control_value=0,
        depression_value=1,
        save_path=os.path.join(PLOTS_DIR, "violin_raw_tasks.png"),
        title="Cognitive task distributions (raw): Control vs Depression",
    )
    # Compute robust z-scores referenced to the control group. The helper
    # uses control median and MAD; it appends a descriptive summary to
    # the provided log file and returns a DataFrame with new ``*_z`` cols
    # for depressed subjects (keeps original columns as well).
    z_scored_data = calculate_robust_z_scores(
        data,
        vars=unique_z_vars,
        group_column=GROUP_COLUMN,
        control_value=0,
        depression_value=1,
        log_path=ROBUST_Z_LOG_PATH,
        log_context="Overall cohort; connectivity_type: N/A; cluster: N/A"
    )

    # Reverse z-scores for variables where higher raw values indicate
    # worse performance so that larger z-scores always represent better
    # performance (consistent plotting/interpretation across tasks).
    vars_to_reverse = [
        'Snap_task_mean_reaction_time',
        'Trail_making_A_duration',
        'Trail_making_B_duration',
        'Pairs_matching_task_errors_6_pairs'
    ]
    for var in vars_to_reverse:
        z_scored_data[f'{var}_z'] = -z_scored_data[f'{var}_z']

    # Plot z-score distributions for the depression cohort to visualize
    # post-standardization behavior (this call optionally restricts to
    # depression-only rows for clarity).
    plot_cognitive_distributions_violin(
        data=z_scored_data,
        variables=[f"{var}_z" for var in unique_z_vars],
        group_column=GROUP_COLUMN,
        control_value=0,
        depression_value=1,
        plot_depression_only=True,
        save_path=os.path.join(PLOTS_DIR, "violin_task_zscores.png"),
        title="Cognitive task distributions (z-scores): Depression cohort",
    )

    # One-sample quantile regression tests: we write a small temporary CSV
    # that the R block reads via rpy2. `group_column=None` signals the
    # helper to run one-sample (intercept) tests vs zero for the z-scored
    # variables.
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

    # Multiple-testing correction across the variable-level tests.
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

    # Visualize composite domain z-scores for the depression cohort.
    plot_cognitive_distributions_violin(
        data=z_scored_data,
        variables=list(DOMAINS.keys()),
        group_column=GROUP_COLUMN,
        control_value=0,
        depression_value=1,
        plot_depression_only=True,
        save_path=os.path.join(PLOTS_DIR, "violin_domain_zscores.png"),
        title="Cognitive domain distributions (z-scores): Depression cohort",
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

    # Multiple testing correction for domain-level one-sample tests
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
        save_path=f'{PLOTS_DIR}/{DEPRESSION_CODES}_{ASSOCIATION_with}_task_z_scores.png',
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
        save_path=f'{PLOTS_DIR}/{DEPRESSION_CODES}_{ASSOCIATION_with}_domain_z_scores.png',
    )

    # Persist the depression-cohort z-scored table for downstream use and
    # inspection.
    output_file = os.path.join(OUTPUT_DIR, f'depression_cohort_z_scores_{DEPRESSION_CODES}.csv')
    z_scored_data.to_csv(output_file, index=False)
    print(f"\nZ-scored data saved to: {output_file}")

    # -------------------------------------------------------------------------
    # Step 6: Cluster specific cognitive dysfunction profiles (from global
    # connectivity clusters -> see `global_connectivity.py`)
    # -------------------------------------------------------------------------
    # Step 6: For each connectivity modality, if a clustered cohort CSV
    # exists, compute within-cluster z-scores and visualizations and
    # collect depressed-only rows to run between-cluster tests.
    print("\n[6/6] Determining cognitive dysfunction profiles for global connectivity clusters...")

    for conn_type in ['functional', 'structural', 'sfc']:
        print(f"\nAnalyzing connectivity type: {conn_type}")
        cluster_csv = os.path.join(DATA_DIR, f"combined_cohort_{DEPRESSION_CODES}_global_{conn_type}_connectivity_clusters.csv")
        if not os.path.exists(cluster_csv):
            print(f"  No clustered cohort file found at {cluster_csv}; skipping {conn_type}.")
            continue

        print("\nDetected clustered cohort file:", cluster_csv)
        cluster_data = load_and_rename_cohort_data(cluster_csv)

        # Collect per-cluster z-scored (tasks + domains) datasets so we can
        # concatenate them for between-cluster comparisons within this
        # connectivity type.
        cluster_between_parts = []

        for cluster_label in [0, 1]:
            cluster_name = f"Cluster {cluster_label}"
            print("\n" + "-" * 80)
            print(f"Running steps 2–5 for {cluster_name} (depression) vs controls")
            print("-" * 80)

            # Subset to all controls plus depressed subjects in the
            # current connectivity cluster. This keeps controls present in
            # every cluster-level analysis and compares the depressed
            # subjects from the given cluster to that shared control set.
            cluster_subset = cluster_data[
                (cluster_data[GROUP_COLUMN] == 0) |
                ((cluster_data[GROUP_COLUMN] == 1) & (cluster_data["Cluster"] == cluster_name))
            ].copy()

            n_controls_cl = int((cluster_subset[GROUP_COLUMN] == 0).sum())
            n_depressed_cl = int((cluster_subset[GROUP_COLUMN] == 1).sum())
            print(f"  N controls: {n_controls_cl}")
            print(f"  N depressed in {cluster_name}: {n_depressed_cl}")

            # Step 2: task-wise robust z-scores, one-sample tests (againt 0)
            unique_z_vars_cluster = sorted(set(var for domain_vars in DOMAINS.values() for var in domain_vars))
            z_scored_cluster = calculate_robust_z_scores(
                cluster_subset,
                vars=unique_z_vars_cluster,
                group_column=GROUP_COLUMN,
                control_value=0,
                depression_value=1,
                log_path=ROBUST_Z_LOG_PATH,
                log_context=f"Step 5 within-cluster; connectivity_type: {conn_type}; cluster: {cluster_name}",
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
                log_context=f"Step 5 within-cluster; connectivity_type: {conn_type}; cluster: {cluster_name}; level: task-wise",
            )

            # Step 3: composite domain z-scores and one-sample tests
            for domain, vars_list in DOMAINS.items():
                z_vars_dom = [f"{var}_z" for var in vars_list]
                z_scored_cluster = calculate_composite_z_score(
                    z_scored_cluster,
                    z_vars=z_vars_dom,
                    output_column=f"{domain}",
                    method="median",
                    log_path=COMPOSITE_Z_LOG_PATH,
                    log_context=f"Step 5 within-cluster; connectivity_type: {conn_type}; cluster: {cluster_name}; domain: {domain}",
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
                log_context=f"Step 5 within-cluster; connectivity_type: {conn_type}; cluster: {cluster_name}; level: domain-wise",
            )

            task_z_vars_cl = [f"{var}_z" for var in unique_z_vars_cluster]
            domain_vars_cl = [f"{domain}" for domain in DOMAINS.keys()]

            # Save depressed-only, z-scored rows for later between-cluster comparisons.
            # The clustered cohort CSV is expected to already contain the
            # string-valued `Cluster` column ("Cluster 0" / "Cluster 1").
            between_cols = COVARIATES + ["Cluster", "Connectivity_Type", "Connectivity"] + task_z_vars_cl + domain_vars_cl
            between_cols = [c for c in between_cols if c in z_scored_cluster.columns]
            cluster_between_parts.append(
                z_scored_cluster.loc[z_scored_cluster[GROUP_COLUMN] == 1, between_cols].copy()
            )

            # Step 4: visualizations (task-wise and domain-level) for this cluster
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
                save_path=f"{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/{DEPRESSION_CODES}_{ASSOCIATION_with}_task_z_scores_{cluster_name.replace(' ', '_')}.png",
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
                save_path=f"{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/{DEPRESSION_CODES}_{ASSOCIATION_with}_domain_z_scores_{cluster_name.replace(' ', '_')}.png",
            )

        # Concatenate per-cluster depressed-only datasets so between-cluster
        # quantile regressions can be run on a single table per connectivity
        # modality. This operation expects at least two cluster parts.
        if len(cluster_between_parts) >= 2:
            between_df = pd.concat(cluster_between_parts, ignore_index=True)
            between_path = os.path.join(
                "/tmp",
                f"{DEPRESSION_CODES}_{conn_type}_cluster_zscores_concatenated.csv",
            )
            between_df.to_csv(between_path, index=False)
            print(f"  Concatenated cluster z-score dataset saved to: {between_path}")

            plot_cognitive_distributions_violin(
                data=between_df,
                variables=task_z_vars_cl,
                group_column=GROUP_COLUMN,
                control_value=0,
                depression_value=1,
                plot_depression_clusters=True,
                cluster_column="Cluster",
                cluster_order=("Cluster 0", "Cluster 1"),
                conn_type=conn_type,
                save_path=(
                    f"{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/{DEPRESSION_CODES}_{ASSOCIATION_with}_task_z_scores_clusters_violin.png"
                ),
                title=(
                    f"Task Z-Scores by Cluster ({conn_type.capitalize()} connectivity)"
                ),
            )

            plot_cognitive_distributions_violin(
                data=between_df,
                variables=domain_vars_cl,
                group_column=GROUP_COLUMN,
                control_value=0,
                depression_value=1,
                plot_depression_clusters=True,
                cluster_column="Cluster",
                cluster_order=("Cluster 0", "Cluster 1"),
                conn_type=conn_type,
                save_path=(
                    f"{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/{DEPRESSION_CODES}_{ASSOCIATION_with}_domain_z_scores_clusters_violin.png"
                ),
                title=(
                    f"Domain Z-Scores by Cluster ({conn_type.capitalize()} connectivity)"
                ),
            )

            # Visualize connectivity–cognition task associations for this connectivity type
            # Cluster specific 
            plot_conn_cognition_association(
                data=between_df,
                connectivity_var='Connectivity',
                cognitive_vars=[f'{var}_z' for var in unique_z_vars_cluster],
                group_column='Cluster',
                save_path=f'{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/global_{DEPRESSION_CODES}_{ASSOCIATION_with}_task_connectivity.png',
                overall_title=(
                f'Global {conn_type.capitalize()} Connectivity vs Cognitive Z-Scores'
                )
            )

            # Between-cluster quantile regression analyses for task-wise z-scores
            # within this connectivity type. The `Cluster` column is used as
            # the group factor (reference = "Cluster 1"), and results are
            # passed through an FDR correction.
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
                log_context=f"Step 5 between-cluster; connectivity_type: {conn_type}; clusters: Cluster 1 (ref) vs Cluster 0; level: task-wise",
            )

            register_radar_overlay_significance(
                depression_codes=DEPRESSION_CODES,
                association_type=ASSOCIATION_with,
                base_kind="task",
                conn_type=conn_type,
                variable_names=[f"{var}_z" for var in unique_z_vars_cluster],
                pvals_corrected=pvals_vars_cls,
                comparison_label="Cluster 0 vs Cluster 1 (between-cluster, FDR)",
            )

            # Visualize connectivity–cognition domain associations for this connectivity type
            # Cluster specific 
            plot_conn_cognition_association(
                data=between_df,
                connectivity_var='Connectivity',
                cognitive_vars=list(DOMAINS.keys()),
                group_column='Cluster',
                save_path=f'{PLOTS_DIR}/schaefer1000+tian54/{conn_type}_con/global_{DEPRESSION_CODES}_{ASSOCIATION_with}_domain_connectivity.png',
                overall_title=(
                f'Global {conn_type.capitalize()} Connectivity vs Cognitive Z-Scores'
                )
            )

            # Between-cluster quantile regression analyses for domain composite
            # z-scores within this connectivity type (with FDR correction).
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
                log_context=f"Step 5 between-cluster; connectivity_type: {conn_type}; clusters: Cluster 1 (ref) vs Cluster 0; level: domain-wise",
            )

            register_radar_overlay_significance(
                depression_codes=DEPRESSION_CODES,
                association_type=ASSOCIATION_with,
                base_kind="domain",
                conn_type=conn_type,
                variable_names=[f"{domain}" for domain in DOMAINS.keys()],
                pvals_corrected=pvals_dom_cls,
                comparison_label="Cluster 0 vs Cluster 1 (between-cluster, FDR)",
            )

        else:
            # If one or both clusters are missing we skip between-cluster
            # tests for this modality.
            print("  Skipping cluster dataset concatenation (missing one or more clusters).")
    
    print("=" * 80)
    print("\nAnalysis complete!")
    print(f"Saved following outputs:")
    print(f" - Z-scored data: {output_file}")
    print(f" - Plots directory: {PLOTS_DIR}")
    print(f" - Logs:\n{R_LOG_PATH}\n{MT_LOG_PATH}\n{ROBUST_Z_LOG_PATH}\n{COMPOSITE_Z_LOG_PATH}")


if __name__ == "__main__":
    main()