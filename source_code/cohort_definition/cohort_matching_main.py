"""
Main script for propensity score matching of UK Biobank cohorts.

This script performs propensity score matching between depression and control cohorts
to create balanced groups for fair group comparisons.

Workflow:
1. Load depression and control cohorts
2. Perform propensity score matching on age and sex
3. Assess covariate balance before and after matching
4. Visualize propensity score distributions prior to matching
5. Combine matched cohorts for downstream analysis
6. Extract control subject IDs (with optional suffixes) from matched control cohort for neuroimaging data extraction (02_workstation_pull.ipynb)
7. Save matched control and combined cohorts

Uses utilities from cohort_matching_utils.py and cohort_selection_utils.py for all core functionality.
"""

import os
import pandas as pd
from cohort_matching_utils import (
    propensity_score_matching,
    assess_covariate_balance,
    plot_propensity_distributions,
    combine_matched_cohorts
)
from cohort_selection_utils import extract_subject_ids

def main():
    """
    Main workflow for propensity score matching.
    """
    print("=" * 80)
    print("Propensity Score Matching for UK Biobank Cohorts")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = '.../data/UKB/cohorts'
    OUTPUT_DIR = '.../data/UKB/cohorts'
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists
    PLOTS_DIR = '.../reports/plots'
    os.makedirs(PLOTS_DIR, exist_ok=True) # Ensure plots directory exists

    DEPRESSION = ['F32']  # Inclusion codes for depression cohort used in cohort selection (used for loading and naming depression files)
    DEPRESSION = "_".join(DEPRESSION)  # Joined string for filenames

    # Input files
    CONTROL_FILE = f'control_cohort.csv'
    TREATED_FILE = f'depression_cohort_{DEPRESSION}.csv'
    
    # Matching parameters
    COVARIATES = ['p21003_i2', 'p31']  # age, sex
    TREATMENT_INDICATOR = 'depression_status'
    MATCH_RATIO = 1
    CALIPER = 0.2
    RANDOM_STATE = 42

    # What suffix to add to extracted subject IDs (for neuroimaging data extraction)
    SUBJECT_ID_SUFFIX = '/i2'  # e.g., '/i2' if IDs need to match specific format
    
    # -------------------------------------------------------------------------
    # Step 1: Load cohorts
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading cohorts...")
    
    control_path = os.path.join(DATA_DIR, CONTROL_FILE)
    treated_path = os.path.join(DATA_DIR, TREATED_FILE)
    
    control_cohort = pd.read_csv(control_path)
    depression_cohort = pd.read_csv(treated_path)
    
    print(f"  Control cohort: {len(control_cohort)} subjects")
    print(f"  Depression cohort: {len(depression_cohort)} subjects")
    print(f"  Covariates: {COVARIATES}")
    
    # -------------------------------------------------------------------------
    # Step 2: Perform propensity score matching
    # -------------------------------------------------------------------------
    print("\n[2/7] Performing propensity score matching...")
    print(f"  Match ratio: 1:{MATCH_RATIO}")
    print(f"  Caliper: {CALIPER} SD")
    
    matched_controls, combined_before = propensity_score_matching(
        control_df=control_cohort,
        treated_df=depression_cohort,
        covariates=COVARIATES,
        treatment_indicator=TREATMENT_INDICATOR,
        match_ratio=MATCH_RATIO,
        caliper=CALIPER,
        random_state=RANDOM_STATE,
        replace=False,
        return_propensity_scores=True
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Assess covariate balance
    # -------------------------------------------------------------------------
    print("\n[3/7] Assessing covariate balance...")
    
    balance_stats = assess_covariate_balance(
        control_df=control_cohort,
        treated_df=depression_cohort,
        matched_control_df=matched_controls,
        covariates=COVARIATES
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Visualize propensity score distributions
    # -------------------------------------------------------------------------
    print("\n[4/7] Visualizing propensity score distributions...")
    
    plot_path = os.path.join(PLOTS_DIR, f'propensity_score_distributions_{DEPRESSION}.png')
    
    plot_propensity_distributions(
        propensity_scores=combined_before['propensity_score'],
        treatment=combined_before[TREATMENT_INDICATOR],
        save_path=plot_path
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Combine matched cohorts
    # -------------------------------------------------------------------------
    print("\n[5/7] Combining matched cohorts...")
    
    combined_cohort = combine_matched_cohorts(
        matched_control_df=matched_controls,
        treated_df=depression_cohort,
        treatment_column=TREATMENT_INDICATOR,
        drop_matching_cols=True
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Extract subject IDs for neuroimaging data extraction
    # -------------------------------------------------------------------------
    print("\n[6/7] Extracting subject IDs from matched cohorts...")
    extract_subject_ids(
        cohort_df=matched_controls,
        eid_column='control_eid',
        suffix=SUBJECT_ID_SUFFIX,
        out_path=os.path.join(OUTPUT_DIR, 'control_subject_ids.txt')
    )

    # -------------------------------------------------------------------------
    # Step 7: Save outputs
    # -------------------------------------------------------------------------
    print("\n[7/7] Saving outputs...")
    
    # Save matched control cohort
    matched_output = os.path.join(OUTPUT_DIR, f'matched_control_cohort_{DEPRESSION}.csv')
    matched_controls.to_csv(matched_output, index=False)
    print(f"  Matched controls saved: {matched_output}")
    print(f"    - {len(matched_controls)} subjects")
    print(f"    - {len(matched_controls.columns)} columns")
    
    # Save combined cohort
    combined_output = os.path.join(OUTPUT_DIR, f'combined_cohort_{DEPRESSION}.csv')
    combined_cohort.to_csv(combined_output, index=False)
    print(f"  Combined cohort saved: {combined_output}")
    print(f"    - {len(combined_cohort)} subjects")
    print(f"    - {len(combined_cohort.columns)} columns")
    
    # Save balance statistics
    balance_output = os.path.join(OUTPUT_DIR, f'balance_statistics_{DEPRESSION}.csv')
    balance_stats.to_csv(balance_output, index=False)
    print(f"  Balance statistics saved: {balance_output}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MATCHING SUMMARY")
    print("=" * 80)
    
    # Matching statistics
    matches_per_treated = matched_controls.groupby('treated_eid').size()
    print(f"\nMatching Performance:")
    print(f"  Treated subjects: {len(depression_cohort)}")
    print(f"  Matched controls: {len(matched_controls)}")
    print(f"  Achieved ratio: {len(matched_controls) / len(depression_cohort):.2f}:1")
    print(f"  Matches per treated subject:")
    print(f"    - Mean: {matches_per_treated.mean():.2f}")
    print(f"    - Min: {matches_per_treated.min()}")
    print(f"    - Max: {matches_per_treated.max()}")
    
    # Balance summary
    print(f"\nCovariate Balance:")
    well_balanced_before = sum(abs(balance_stats['smd_before']) < 0.1)
    well_balanced_after = sum(abs(balance_stats['smd_after']) < 0.1)
    print(f"  Before: {well_balanced_before}/{len(balance_stats)} well-balanced (|SMD| < 0.1)")
    print(f"  After: {well_balanced_after}/{len(balance_stats)} well-balanced (|SMD| < 0.1)")
    
    # Combined cohort composition
    print(f"\nCombined Cohort:")
    print(f"  Controls: {(combined_cohort[TREATMENT_INDICATOR] == 0).sum()}")
    print(f"  Treated: {(combined_cohort[TREATMENT_INDICATOR] == 1).sum()}")
    print(f"  Total: {len(combined_cohort)}")
    
    print("=" * 80)
    print("\nPropensity score matching complete!")


if __name__ == "__main__":
    main()
