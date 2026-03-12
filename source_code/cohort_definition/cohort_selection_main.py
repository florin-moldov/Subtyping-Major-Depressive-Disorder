"""
Main script for cohort definition and selection from UK Biobank data.

This script orchestrates the complete workflow for creating depression and control cohorts:
1. Load UK Biobank datasets (diagnosis, cognitive, rs- and tfMRI, dMRI, demographics)
2. Extract individuals based on ICD-10 codes
3. Select which datasets must overlap (required) and which are optional merges
4. Create depression and control cohorts based on specified criteria
5. Extract depressed subject IDs from depression cohort for later extraction of neuroimaging data (02_workstation_pull.ipynb)
6. Determine and visualize comorbidities in the depression cohort (and saves most prevalent ones)
7. Export cohorts and comorbidities to CSV files

Key features:
- Required vs Optional datasets: Users can select which datasets must have overlapping individuals
    using `REQUIRED_DATASETS` and optionally merge others with `OPTIONAL_DATASETS`. Merging with 
    optional datasets is done via left joins to retain all individuals from the required datasets.
- Inclusion and exclusion criteria: Depression cohort includes individuals with specified ICD-10 codes,
    while the control cohort excludes individuals with any depression diagnosis. Or controls can be defined as those
    without any ICD-10 codes.
- Creates separate CSV files for depression and control cohorts with appropriate labels.

Uses utilities from cohort_selection_utils.py for all core functionality.
"""

import os
from cohort_selection_utils import (
    extract_subject_ids,
    load_ukb_datasets,
    standardize_eid_columns,
    create_cohort,
    count_codes_in_cohort,
    plot_comorbidity_distribution,
    build_comorbidity_indicator_matrix,
    save_cohort,
)


def main():
    
    # -------------------------------------------------------------------------
    # User configuration
    # -------------------------------------------------------------------------
    DATA_DIR = '.../data/UKB'
    OUTPUT_DIR = '.../data/UKB/cohorts'
    

    CONTROL_CLEAN = True  # If True, control cohort excludes any individuals with any ICD-10 codes for diagnosis source "icd10" 
    if not CONTROL_CLEAN:
        CONTROL = ['F32', 'F33']  # Excluding these ICD-10 codes for control cohort
    else:
        CONTROL = None  # Control cohort excludes any individuals with any ICD-10 codes

    DEPRESSION_inc = ['F32']  # Inclusion codes for depression
    DEPRESSION_CODES = "_".join(DEPRESSION_inc)  # Joined string for filenames
    DEPRESSION_exc = ['F33']  # Exclusion codes for depression

    DEPRESSION_OUTPUT = os.path.join(OUTPUT_DIR, f'depression_cohort_{DEPRESSION_CODES}.csv')
    CONTROL_OUTPUT = os.path.join(OUTPUT_DIR, f'control_cohort.csv')

    # Select which datasets must have overlapping individuals (required)
    # Valid names: ["cognitive", "taskfmri", "restingfmri", "dMRI", "demographics"]
    REQUIRED_DATASETS = ["cognitive", "restingfmri", "dMRI", "demographics"]
    # Optionally add datasets without reducing the cohort (left merge)
    OPTIONAL_DATASETS = []

    # Suffix to add to extracted subject IDs (for neuroimaging data extraction)
    SUBJECT_ID_SUFFIX = '/i2'  # e.g., '/i2' if IDs need to match specific format

    # -------------------------------------------------------------------------
    # Step 1: Load all UK Biobank datasets
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading UK Biobank datasets...")
    columns_cognitive = [
            "eid", "p399_i2_a1", "p399_i2_a2", "p4282_i2", "p20018_i2",
            "p20197_i2", "p23324_i2", "p6348_i2", "p20023_i2",
            "p21004_i2", "p6350_i2", "p20016_i2", "p6373_i2", 
            "p26302_i2",
        ]
    columns_demographics = ["eid", "p31", "p21003_i2"]

    datasets = load_ukb_datasets(data_dir=DATA_DIR, columns_cognitive=columns_cognitive, columns_demographics=columns_demographics, convert_sex=True)
    
    # -------------------------------------------------------------------------
    # Step 2: Standardize eid columns to string type
    # -------------------------------------------------------------------------
    print("\n[2/7] Standardizing eid columns...")
    datasets = standardize_eid_columns(datasets, target_type='str')
    
    # -------------------------------------------------------------------------
    # Step 3: Create depression cohort
    # -------------------------------------------------------------------------
    print("\n[3/7] Creating depression cohort...")
    
    depression_cohort = create_cohort(
        diagnosis_df_icd10=datasets['diagnosis_ICD10'],
        cognitive_df=datasets['cognitive'],
        taskfmri_df=datasets['taskfmri'],
        restingfmri_df=datasets['restingfmri'],
        dmri_df=datasets['dMRI'],
        demographics_df=datasets['demographics'],
        icd10_codes=DEPRESSION_inc,
        cohort_type='depression',
        exclude_icd10_codes=DEPRESSION_exc,
        required_datasets=REQUIRED_DATASETS,
        optional_datasets=OPTIONAL_DATASETS,
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Extract subject IDs from depression cohort for later extraction of 
    # neuroimaging data (02_workstation_pull.ipynb)
    # -------------------------------------------------------------------------
    print("\n[4/7] Extracting subject IDs from depression cohort...")
    extract_subject_ids(
        cohort_df=depression_cohort,
        eid_column='eid',
        suffix=SUBJECT_ID_SUFFIX,
        out_path=os.path.join(OUTPUT_DIR, 'depression_subject_ids.txt')
    )

    # -------------------------------------------------------------------------
    # Step 5: Create control cohort
    # -------------------------------------------------------------------------
    print("\n[5/7] Creating control cohort...")
    
    control_cohort = create_cohort(
        diagnosis_df_icd10=datasets['diagnosis_ICD10'],
        cognitive_df=datasets['cognitive'],
        taskfmri_df=datasets['taskfmri'],
        restingfmri_df=datasets['restingfmri'],
        dmri_df=datasets['dMRI'],
        demographics_df=datasets['demographics'],
        cohort_type='control',
        icd10_codes=CONTROL,
        exclude_icd10_codes=None,
        required_datasets=REQUIRED_DATASETS,
        optional_datasets=OPTIONAL_DATASETS,
        control_no_icd10=CONTROL_CLEAN,
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Cohort characteristics and visualization
    # -------------------------------------------------------------------------
    print("\n[6/7] Analyzing cohort characteristics...")

    # Analyze comorbidities in depression cohort
    print("  - Analyzing comorbidities in depression cohort...")
    comorbidity_distribution = count_codes_in_cohort(
        cohort_df=depression_cohort,
        codes_column_cohort='codes',
        coding_filepath=os.path.join(DATA_DIR, 'coding19.tsv'),
        output_path=os.path.join(OUTPUT_DIR, f'icd10_code_distribution_depression_cohort.csv')
    )
    print(f"    Found {len(comorbidity_distribution)} unique ICD-10 codes in depression cohort.")

    # Plot comorbidity distribution
    plot_output_path = os.path.join(OUTPUT_DIR, f'comorbidity_distribution_depression_cohort.png')
    plot_comorbidity_distribution(
        code_distribution=comorbidity_distribution,
        proportion_threshold= 0.10,
        title= "ICD-10 Code Distribution",
        figsize= (20, 20),
        wrap_width= 40, 
        y_label_rotation= 15.0, 
        y_spacing= 3.0,
        output_path=plot_output_path,
    )
    print(f"Comorbidity distribution plot saved to: {plot_output_path}")

    # Generate and save a dataframe with for each subject in the depression cohort whether they have each of the top N comorbid ICD-10 codes
    depression_cohort = build_comorbidity_indicator_matrix(cohort_df = depression_cohort,
                                   eid_column = 'eid',
                                   codes_column_cohort = 'codes',
                                   proportion_threshold = 0.10,
                                   exclude_codes = DEPRESSION_inc)
    # -------------------------------------------------------------------------
    # Step 7: Save cohorts to CSV files
    # -------------------------------------------------------------------------
    print("\n[7/7] Saving cohorts...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save depression cohort
    save_cohort(depression_cohort, DEPRESSION_OUTPUT)
    
    # Save control cohort
    save_cohort(control_cohort, CONTROL_OUTPUT)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COHORT SUMMARY")
    print("=" * 80)
    print(f"Depression cohort: {len(depression_cohort)} individuals")
    print(f"  - Saved to: {DEPRESSION_OUTPUT}")
    print(f"  - Columns: {list(depression_cohort.columns)}")
    print(f"  - Subject IDs saved to: {os.path.join(OUTPUT_DIR, 'depression_subject_ids.txt')}")
    print(f"Control cohort: {len(control_cohort)} individuals")
    print(f"  - Saved to: {CONTROL_OUTPUT}")
    print(f"  - Columns: {list(control_cohort.columns)}")
    print()
    print("=" * 80)
    print("\nCohort definition complete!")


if __name__ == "__main__":
    main()
