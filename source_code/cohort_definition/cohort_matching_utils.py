"""
cohort_matching_utils
----------------------

Utilities for propensity-score based cohort matching and diagnostics.

This module provides functions to:
- Estimate propensity scores using a preprocessed logistic regression
    pipeline with cross-validation.
- Perform caliper-constrained nearest-neighbour propensity-score matching.
- Assess covariate balance using standardized mean differences (SMD).
- Visualize propensity score distributions and common-support diagnostics.
- Combine matched control and treated cohorts for downstream analysis.
- Extract control subject IDs for neuroimaging data extraction.

Conventions and expectations
- Input tables are pandas ``DataFrame`` objects. Most helpers assume a
    participant identifier column named ``'eid'`` by default; this can be
    overridden by function arguments where applicable.
- Covariates passed to matching helpers must exist in both control and
    treated DataFrames. Categorical covariates are autodetected by dtype
    and are one-hot encoded; numeric covariates are standardized.
- Many helpers print diagnostic summaries, may emit ``warnings.warn`` for
    dropped rows / unmatched units, and call ``matplotlib.pyplot.show()`` to
    display plots. Callers running in non-interactive environments should
    capture figures or pass ``save_path`` where available.

Design notes
- All changes in this file are applied to copies of input DataFrames when
    possible to avoid unexpected mutation of caller objects. Returned DataFrames
    are new objects containing selected/merged columns described in each
    function's docstring.

Main exported functions
- ``propensity_score_matching``: core matching routine
- ``assess_covariate_balance``: compute SMDs before/after matching
- ``plot_propensity_distributions``: visualization of propensity diagnostics
- ``combine_matched_cohorts``: produce single cohort with treatment flag

Architecture
------------
The script is organized into functional sections:

- **Utility Functions**

- **Data preparation and propensity score estimation/matching**

- **Covariate balance assessment pre- and post-matching**

- **Visualization**

- **Final cohort assembly**

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import gaussian_kde
import warnings
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _caliper_matching(
    control_ps: np.ndarray,
    treated_ps: np.ndarray,
    caliper: float,
    match_ratio: int,
    replace: bool,
    random_state: int
) -> np.ndarray:
    """
    Perform nearest neighbor matching with caliper constraint.
    
    For each treated subject, finds the closest control subjects within the caliper
    distance (in logit propensity score space).
    
    Parameters
    ----------
    control_ps : np.ndarray
        Propensity scores for control subjects
    treated_ps : np.ndarray
        Propensity scores for treated subjects
    caliper : float
        Maximum allowed distance in logit propensity score space
    match_ratio : int
        Number of controls to match per treated subject
    replace : bool
        Whether to sample controls with replacement
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Indices of matched control subjects
    
    Notes
    -----
    - Distance is calculated in logit space: |logit(PS_control) - logit(PS_treated)|
    - If no matches within caliper, a warning is issued and that treated subject is skipped
    - Without replacement, each control can only be matched once
    
    Additional details
    ------------------
    - The returned array contains indices referring to positions in the input
        ``control_ps`` array. Its length may be smaller than ``len(treated_ps) * match_ratio``
        if some treated units have no eligible controls within the caliper.
    - When ``replace=True`` the same control index may appear multiple times in
        the output; when ``replace=False`` indices are unique.
    - ``random_state`` is used to seed numpy for deterministic behaviour but
        the current matching selection is deterministic based on distance ordering.
    """
    np.random.seed(random_state)
    
    matched_indices = []
    used_controls = np.array([], dtype=int)
    unmatched_count = 0
    
    for treated_score in treated_ps:
        # Calculate distances in logit space
        logit_control = np.log(control_ps / (1 - control_ps))
        logit_treated = np.log(treated_score / (1 - treated_score))
        distances = np.abs(logit_control - logit_treated)
        
        # Find controls within caliper
        within_caliper = np.where(distances <= caliper)[0]
        
        # Filter out already used controls if without replacement
        if not replace and len(used_controls) > 0:
            within_caliper = np.setdiff1d(within_caliper, used_controls)
        
        if len(within_caliper) == 0:
            unmatched_count += 1
            continue
        
        # Select closest matches up to match_ratio
        n_matches = min(match_ratio, len(within_caliper))
        distances_within = distances[within_caliper]
        sort_order = np.argsort(distances_within)[:n_matches]
        closest_indices = within_caliper[sort_order]
        
        matched_indices.extend(closest_indices.tolist())
        
        if not replace:
            used_controls = np.concatenate([used_controls, closest_indices])
    
    if unmatched_count > 0:
        warnings.warn(
            f"Could not find matches within caliper for {unmatched_count} treated subjects "
            f"({100*unmatched_count/len(treated_ps):.1f}%). Consider increasing caliper."
        )
    
    return np.array(matched_indices, dtype=int)

def extract_subject_ids(
    cohort_df: pd.DataFrame,
    eid_column: str = 'eid',
    suffix: Union[str, List[str]] = '',
    out_path: str = None
) -> List[str]:
    """
    Extract subject IDs from a cohort DataFrame and append specified suffix (or suffixes).
    
    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort DataFrame containing subject IDs
    eid_column : str, optional
        Name of the column containing subject IDs (default: 'eid')
    suffix : str or List[str], optional
        Suffix or list of suffixes to append to each subject ID (default: ''). 
        Format examples: '/i2', ['/i2', '/i3']
    
    Returns
    -------
    None
        The function writes the extracted subject ID lines to ``out_path`` if
        provided and prints a short summary. It does not return the list of
        subject IDs. Callers that need the list should build it themselves
        or read back the file written to ``out_path``.
    
    Examples
    --------
    >>> subject_ids = extract_subject_ids(depression_cohort, eid_column='eid', suffix='/i2', out_path='data/UKB/depression_subject_ids.txt')
    """
    if eid_column not in cohort_df.columns:
        raise ValueError(f"Column '{eid_column}' not found in cohort DataFrame")
    
    subject_ids = cohort_df[eid_column].astype(str).tolist()
    if suffix:
        if isinstance(suffix, str):
            subject_ids = [f"{sid}{suffix}" for sid in subject_ids]
        elif isinstance(suffix, list):
            subject_ids = [f"{sid}{suf}" for sid in subject_ids for suf in suffix]
        else:
            raise ValueError("Suffix must be a string or a list of strings")
    print(f"Extracted {len(subject_ids)} subject IDs from cohort")

    if out_path:
        with open(out_path, 'w') as f:
            for sid in subject_ids:
                f.write(f"{sid}\n")
        print(f"Subject IDs saved to {out_path}")

# ==============================================================================
# DATA PREPARATION AND PROPENSITY SCORE ESTIMATION / MATCHING
# ==============================================================================
def propensity_score_matching(
    control_df: pd.DataFrame,
    treated_df: pd.DataFrame,
    covariates: List[str],
    eid_column: str = "eid",
    treatment_indicator: str = "condition",
    match_ratio: int = 1,
    caliper: float = 0.2,
    random_state: int = 42,
    replace: bool = False,
    return_propensity_scores: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform propensity score matching between control and treated cohorts.

    High-level steps
    ----------------
    1. Combine control and treated cohorts and preserve original eids/indexes.
    2. Preprocess covariates (standardize numerics, one-hot encode categoricals).
    3. Estimate propensity scores using logistic regression CV.
    4. Perform caliper-based propensity score matching.

    Parameters
    ----------
    control_df : pd.DataFrame
        Control cohort (not treated).
    treated_df : pd.DataFrame
        Treated cohort.
    covariates : List[str]
        Covariate column names to include in the propensity model. Columns
        must be present in both `control_df` and `treated_df`.
    eid_column : str, optional
        Name of the column containing participant IDs (default: "eid").
    treatment_indicator : str, optional
        Name used internally for the treatment indicator when combining data
        (default: "condition" to match the function signature).
    match_ratio : int, optional
        Number of controls to match per treated subject (default: 1).
    caliper : float, optional
        Caliper width (multiples of the SD of logit(PS)). Smaller values are
        stricter and may lead to unmatched treated subjects (default: 0.2).
    random_state : int, optional
        Seed for reproducible behaviour (default: 42).
    replace : bool, optional
        Whether matching is performed with replacement (default: False).
    return_propensity_scores : bool, optional
        If True, return a tuple ``(final_matched, combined_clean)`` where the
        second element contains the combined dataset with estimated propensity
        scores prior to matching.

    Returns
    -------
    final_matched : pd.DataFrame
        DataFrame of matched controls. Columns include at minimum:
        - ``control_eid``: identifier for the matched control (from original control DF)
        - ``treated_eid``: identifier for the treated subject the control was matched to
        - ``propensity_score``: estimated propensity score used for matching
        plus the original columns from the control cohort (merged in).

    combined_clean : pd.DataFrame, optional
        When ``return_propensity_scores=True``, the function also returns the
        combined (control+treated) DataFrame used to fit the propensity model
        with an added ``propensity_score`` column.

    Raises
    ------
    ValueError
        - If ``match_ratio < 1`` or ``caliper <= 0``.
        - If the ``eid_column`` is missing from either input DataFrame.
        - If any of the ``covariates`` are missing in either DataFrame.

    Notes
    -----
    - Missing rows in covariates are dropped with a warning; use consistent
      preprocessing to avoid excessive row loss.
    - The function creates temporary columns (e.g. ``'_original_index'``,
      ``'_original_eid'``, ``'propensity_score'``) during processing; most are
      removed from the final returned DataFrame. The combined_clean object
      (when returned) preserves ``propensity_score``.
    - If some treated units cannot be matched within the caliper they are
      skipped and a warning is issued; the achieved match ratio may therefore
      be lower than the requested ``match_ratio``.
    - The routine prints concise diagnostics and may raise warnings for
      model convergence or dropped observations.

    Examples
    --------
    >>> # Basic 1:1 matching on age and sex
    >>> matched_controls = propensity_score_matching(
    ...     control_df=control_cohort,
    ...     treated_df=depression_cohort,
    ...     covariates=['p21003_i2', 'p31']
    ... )

    >>> # Request propensity scores for inspection
    >>> matched, combined = propensity_score_matching(
    ...     control_df=control_cohort,
    ...     treated_df=depression_cohort,
    ...     covariates=['age', 'sex'],
    ...     return_propensity_scores=True
    ... )
    """
    # Validate inputs
    if match_ratio < 1:
        raise ValueError("match_ratio must be >= 1")
    if caliper <= 0:
        raise ValueError("caliper must be > 0")
    
    # Check for eid column
    if eid_column not in control_df.columns:
        raise ValueError(f"eid_column '{eid_column}' not found in control_df")
    if eid_column not in treated_df.columns:
        raise ValueError(f"eid_column '{eid_column}' not found in treated_df")
    
    # Check for missing covariates
    missing_control = set(covariates) - set(control_df.columns)
    missing_treated = set(covariates) - set(treated_df.columns)
    if missing_control:
        raise ValueError(f"Covariates missing in control_df: {missing_control}")
    if missing_treated:
        raise ValueError(f"Covariates missing in treated_df: {missing_treated}")
    
    # Create copies to avoid modifying original dataframes
    control = control_df.copy()
    treated = treated_df.copy()
    control_original = control_df.copy()
    
    # Add treatment indicator
    control[treatment_indicator] = 0
    treated[treatment_indicator] = 1
    
    # Store original indices and eids
    control['_original_index'] = control.index
    treated['_original_index'] = treated.index
    control['_original_eid'] = control[eid_column]
    treated['_original_eid'] = treated[eid_column]
    
    # Combine datasets
    combined = pd.concat([control, treated], axis=0, ignore_index=True)
    
    # Handle missing values
    initial_size = len(combined)
    combined_clean = combined[covariates + [treatment_indicator, '_original_index', '_original_eid']].dropna()
    dropped = initial_size - len(combined_clean)
    
    if dropped > 0:
        warnings.warn(
            f"Dropped {dropped} rows ({dropped/initial_size*100:.1f}%) due to missing values in covariates"
        )
    
    # Separate features and treatment
    X = combined_clean[covariates]
    y = combined_clean[treatment_indicator]
    
    # Estimate propensity scores using logistic regression with preprocessing
    print("Estimating propensity scores...")
    
    # Automatically detect numerical and categorical columns
    numerical_columns_selector = make_column_selector(dtype_exclude=object)
    categorical_columns_selector = make_column_selector(dtype_include=object)
    
    numerical_columns = numerical_columns_selector(X)
    categorical_columns = categorical_columns_selector(X)
    
    print(f"  Numerical covariates: {numerical_columns}")
    print(f"  Categorical covariates: {categorical_columns}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ]
    )
    
    # Create model pipeline with cross-validated logistic regression
    model = make_pipeline(
        preprocessor,
        LogisticRegressionCV(
            cv=10,
            penalty='l2',
            Cs=10,
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        )
    )
    
    model.fit(X, y)
    propensity_scores = model.predict_proba(X)[:, 1]
    
    # Add propensity scores to dataframe
    combined_clean['propensity_score'] = propensity_scores
    
    # Split back into control and treated
    control_scored = combined_clean[combined_clean[treatment_indicator] == 0].copy()
    treated_scored = combined_clean[combined_clean[treatment_indicator] == 1].copy()
    
    print(f"\nCohort sizes:")
    print(f"  Control: {len(control_scored)} subjects")
    print(f"  Treated: {len(treated_scored)} subjects")
    print(f"\nPropensity score ranges:")
    print(f"  Control: [{control_scored['propensity_score'].min():.3f}, {control_scored['propensity_score'].max():.3f}]")
    print(f"  Treated: [{treated_scored['propensity_score'].min():.3f}, {treated_scored['propensity_score'].max():.3f}]")
    
    # Calculate caliper in logit score units
    logit_ps = np.log(propensity_scores / (1 - propensity_scores))
    ps_caliper = caliper * np.std(logit_ps)
    
    print(f"\nCaliper: {ps_caliper:.4f} (logit scale, {caliper} × SD)")
    
    # Perform matching
    print(f"\nPerforming caliper matching (ratio 1:{match_ratio})...")
    
    matched_control_indices = _caliper_matching(
        control_scored['propensity_score'].values,
        treated_scored['propensity_score'].values,
        caliper=ps_caliper,
        match_ratio=match_ratio,
        replace=replace,
        random_state=random_state
    )
    
    # Create matched control dataframe
    matched_controls = control_scored.iloc[matched_control_indices].copy()
    
    # Add which treated subject each control was matched to
    treated_eids = treated_scored['_original_eid'].values
    matched_treated_eids = np.repeat(treated_eids, match_ratio)[:len(matched_controls)]
    matched_controls['treated_eid'] = matched_treated_eids
    
    # Rename control eid column for clarity
    matched_controls['control_eid'] = matched_controls['_original_eid']
    
    # Remove temporary columns
    columns_to_drop = [treatment_indicator, '_original_index', '_original_eid']
    matched_controls = matched_controls.drop(columns=columns_to_drop)
    
    # Merge with original control data to get all columns
    control_full = control_original.copy()
    control_full = control_full.drop(columns=covariates)
    
    final_matched = matched_controls.merge(
        control_full,
        left_on='control_eid',
        right_on=eid_column,
        how='left',
        suffixes=('_matched', '_original')
    )
    
    # Drop duplicate eid column if exists
    if eid_column in final_matched.columns and eid_column != 'control_eid':
        final_matched = final_matched.drop(columns=[eid_column])
    
    # Reorder columns: control_eid, treated_eid, propensity_score first
    priority_cols = ['control_eid', 'treated_eid', 'propensity_score']
    existing_priority = [col for col in priority_cols if col in final_matched.columns]
    other_cols = [col for col in final_matched.columns if col not in existing_priority]
    final_matched = final_matched[existing_priority + other_cols]
    
    print(f"\nMatching results:")
    print(f"  Matched {len(final_matched)} controls to {len(treated_scored)} treated subjects")
    print(f"  Achieved ratio: {len(final_matched) / len(treated_scored):.2f}:1")
    print(f"  Output columns: {len(final_matched.columns)}")
    
    if return_propensity_scores:
        return final_matched, combined_clean
    else:
        return final_matched

# ==============================================================================
# COVARIATE BALANCE ASSESSMENT PRE- AND POST-MATCHING
# ==============================================================================
def assess_covariate_balance(
    control_df: pd.DataFrame,
    treated_df: pd.DataFrame,
    matched_control_df: pd.DataFrame,
    covariates: List[str],
) -> pd.DataFrame:
    """
    Assess covariate balance before and after matching using standardized mean differences (SMD).
    
    SMD is a scale-free measure of balance that compares the difference in means
    (or proportions for categorical variables) standardized by the pooled standard deviation.
    
    Parameters
    ----------
    control_df : pd.DataFrame
        Original unmatched control cohort
    treated_df : pd.DataFrame
        Treated cohort
    matched_control_df : pd.DataFrame
        Matched control cohort after PSM
    covariates : List[str]
        List of covariates to assess
    
    Returns
    -------
    pd.DataFrame
        Balance statistics with columns:
        - 'covariate': Covariate name (for categorical: 'var_name=category')
        - 'smd_before': SMD before matching
        - 'smd_after': SMD after matching
        - 'improvement': Percentage improvement in balance
        - 'variable_type': 'continuous' or 'categorical'
    
    Notes
    -----
    **SMD Interpretation:**
    - |SMD| < 0.1: Well-balanced
    - |SMD| 0.1-0.2: Acceptable balance
    - |SMD| > 0.2: Imbalanced
    
    **Continuous variables:**
        SMD = (mean_treated - mean_control) / pooled_std
    
    **Categorical variables:**
        SMD calculated for each category as:
        SMD = (prop_treated - prop_control) / pooled_std_proportion

    Balance Assessment Criteria (Austin, 2009):
        |SMD| < 0.1  → Negligible imbalance (well-balanced)
        |SMD| < 0.2  → Small imbalance (acceptable in practice)
        |SMD| ≥ 0.2  → Meaningful imbalance (may introduce confounding)
    
    References
    ----------
    Austin, P. C. (2009). Balance diagnostics for comparing the distribution of 
        baseline covariates between treatment groups in propensity-score matched samples. 
        Statistics in Medicine, 28(25), 3083-3107.
    
    Normand, S. L. T., et al. (2001). Validating recommendations for coronary angiography 
        following acute myocardial infarction in the elderly: a matched analysis using 
        propensity scores. Journal of Clinical Epidemiology, 54(4), 387-398.
    
    Stuart, E. A. (2010). Matching methods for causal inference: A review and a look forward. 
        Statistical Science, 25(1), 1-21.
    
    Examples
    --------
    >>> balance = assess_covariate_balance(
    ...     control_df=control_cohort,
    ...     treated_df=depression_cohort,
    ...     matched_control_df=matched_controls,
    ...     covariates=['p21003_i2', 'p31']
    ... )
    >>> print(balance)

    Raises
    ------
    KeyError
        If any covariate in ``covariates`` is missing from the input DataFrames.

    Notes
    -----
    - Categorical detection uses pandas dtype checks on ``control_df``. If a
        categorical variable is encoded as integers, cast it to ``object`` or
        ``category`` prior to calling this function.
    - Missing values may affect means and proportions; callers should ensure
        consistent preprocessing across datasets before computing balance.
    """
    balance_stats = []
    
    for cov in covariates:
        # Check if variable is categorical
        is_categorical = (
            pd.api.types.is_object_dtype(control_df[cov]) or
            pd.api.types.is_string_dtype(control_df[cov]) or
            pd.api.types.is_categorical_dtype(control_df[cov])
        )
        
        if is_categorical:
            # Handle categorical variables
            all_categories = set(treated_df[cov].unique()) | \
                           set(control_df[cov].unique()) | \
                           set(matched_control_df[cov].unique())
            all_categories = sorted([c for c in all_categories if pd.notna(c)])
            
            # Calculate SMD for each category
            for category in all_categories:
                # Calculate proportions
                p_treated = (treated_df[cov] == category).mean()
                p_control_before = (control_df[cov] == category).mean()
                p_control_after = (matched_control_df[cov] == category).mean()
                
                # Calculate pooled standard deviations for proportions
                std_treated = np.sqrt(p_treated * (1 - p_treated))
                std_control_before = np.sqrt(p_control_before * (1 - p_control_before))
                std_control_after = np.sqrt(p_control_after * (1 - p_control_after))
                
                pooled_std_before = np.sqrt((std_treated**2 + std_control_before**2) / 2)
                pooled_std_after = np.sqrt((std_treated**2 + std_control_after**2) / 2)
                
                # Calculate SMD
                smd_before = (p_treated - p_control_before) / pooled_std_before if pooled_std_before > 0 else 0
                smd_after = (p_treated - p_control_after) / pooled_std_after if pooled_std_after > 0 else 0
                
                # Calculate improvement
                improvement = (abs(smd_before) - abs(smd_after)) / abs(smd_before) * 100 if smd_before != 0 else 0
                
                balance_stats.append({
                    'covariate': f'{cov}={category}',
                    'smd_before': smd_before,
                    'smd_after': smd_after,
                    'improvement': improvement,
                    'variable_type': 'categorical'
                })
        
        else:
            # Handle continuous variables
            treated_mean = treated_df[cov].mean()
            control_mean_before = control_df[cov].mean()
            control_mean_after = matched_control_df[cov].mean()
            
            # Calculate pooled standard deviations
            treated_std = treated_df[cov].std()
            control_std_before = control_df[cov].std()
            control_std_after = matched_control_df[cov].std()
            
            pooled_std_before = np.sqrt((treated_std**2 + control_std_before**2) / 2)
            pooled_std_after = np.sqrt((treated_std**2 + control_std_after**2) / 2)
            
            # Calculate SMD
            smd_before = (treated_mean - control_mean_before) / pooled_std_before if pooled_std_before > 0 else 0
            smd_after = (treated_mean - control_mean_after) / pooled_std_after if pooled_std_after > 0 else 0
            
            # Calculate improvement
            improvement = (abs(smd_before) - abs(smd_after)) / abs(smd_before) * 100 if smd_before != 0 else 0
            
            balance_stats.append({
                'covariate': cov,
                'smd_before': smd_before,
                'smd_after': smd_after,
                'improvement': improvement,
                'variable_type': 'continuous'
            })
    
    balance_df = pd.DataFrame(balance_stats)
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print("COVARIATE BALANCE ASSESSMENT")
    print("=" * 80)
    print(f"{'Covariate':<30} {'Type':<12} {'SMD Before':<12} {'SMD After':<12} {'Improv.':<10}")
    print("-" * 80)
    
    for _, row in balance_df.iterrows():
        var_type = row['variable_type'][:4]
        print(f"{row['covariate']:<30} {var_type:<12} {row['smd_before']:>11.4f} "
              f"{row['smd_after']:>11.4f} {row['improvement']:>8.1f}%")
    
    print("=" * 80)
    
    # Summary statistics
    well_balanced_before = sum(abs(balance_df['smd_before']) < 0.1)
    well_balanced_after = sum(abs(balance_df['smd_after']) < 0.1)
    acceptable_before = sum(abs(balance_df['smd_before']) < 0.2)
    acceptable_after = sum(abs(balance_df['smd_after']) < 0.2)
    
    print(f"\nBefore Matching:")
    print(f"  Well-balanced (|SMD| < 0.1): {well_balanced_before} / {len(balance_df)}")
    print(f"  Acceptable (|SMD| < 0.2): {acceptable_before} / {len(balance_df)}")
    print(f"\nAfter Matching:")
    print(f"  Well-balanced (|SMD| < 0.1): {well_balanced_after} / {len(balance_df)}")
    print(f"  Acceptable (|SMD| < 0.2): {acceptable_after} / {len(balance_df)}")
    
    # Separate stats by variable type
    continuous_vars = balance_df[balance_df['variable_type'] == 'continuous']
    categorical_vars = balance_df[balance_df['variable_type'] == 'categorical']
    
    if len(continuous_vars) > 0:
        print(f"\nContinuous variables ({len(continuous_vars)}):")
        print(f"  Mean |SMD| before: {abs(continuous_vars['smd_before']).mean():.4f}")
        print(f"  Mean |SMD| after: {abs(continuous_vars['smd_after']).mean():.4f}")
    
    if len(categorical_vars) > 0:
        print(f"\nCategorical variables ({len(categorical_vars)}):")
        print(f"  Mean |SMD| before: {abs(categorical_vars['smd_before']).mean():.4f}")
        print(f"  Mean |SMD| after: {abs(categorical_vars['smd_after']).mean():.4f}")
    
    print("=" * 80)
    
    return balance_df

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_propensity_distributions(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot propensity score distributions for treated and control groups.
    
    Creates a 4-panel visualization:
    1. Overlapping histograms
    2. Kernel density estimates
    3. Mirror histogram (back-to-back) for common support assessment
    4. Box plots
    
    Parameters
    ----------
    propensity_scores : np.ndarray
        Propensity scores (probabilities between 0 and 1)
    treatment : np.ndarray
        Binary treatment indicator (1=treated, 0=control)
    figsize : Tuple[int, int], optional
        Figure size (width, height) (default: (14, 10))
    save_path : str, optional
        Path to save figure. If None, figure is not saved (default: None)
    
    Examples
    --------
    >>> # Plot propensity distributions
    >>> plot_propensity_distributions(
    ...     propensity_scores=combined['propensity_score'],
    ...     treatment=combined['treatment'],
    ...     save_path='reports/plots/propensity_distributions.svg'
    ... )
    
    Notes
    -----
    - Common support region is the overlap between treated and control distributions
    - Extreme propensity scores (< 0.1 or > 0.9) may indicate poor overlap
    
    Additional notes
    ----------------
    - ``propensity_scores`` and ``treatment`` must be one-dimensional arrays
        of the same length. The function will coerce inputs to numpy arrays.
    - The function calls ``plt.show()`` to display the figure and will save
        to ``save_path`` if provided. In headless environments capture the
        returned figure or provide ``save_path`` to persist results.
    - KDE plots are only drawn when there are at least two observations in
        the group; small group sizes will omit KDEs but histograms/boxplots
        remain informative.
    """
    treatment = np.array(treatment)
    propensity_scores = np.array(propensity_scores)
    
    treated_ps = propensity_scores[treatment == 1]
    control_ps = propensity_scores[treatment == 0]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Overlapping Histograms
    axes[0, 0].hist(treated_ps, bins=50, alpha=0.6, label='Treated', color='#e74c3c', density=True)
    axes[0, 0].hist(control_ps, bins=50, alpha=0.6, label='Control', color='#3498db', density=True)
    axes[0, 0].set_xlabel('Propensity Score', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title('Distribution of Propensity Scores (Histogram)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Kernel Density Plot
    axes[0, 1].hist(treated_ps, bins=50, alpha=0.3, label='Treated', color='#e74c3c', density=True)
    axes[0, 1].hist(control_ps, bins=50, alpha=0.3, label='Control', color='#3498db', density=True)
    
    x_range = np.linspace(0, 1, 200)
    if len(treated_ps) > 1:
        kde_treated = gaussian_kde(treated_ps)
        axes[0, 1].plot(x_range, kde_treated(x_range), color='#e74c3c', linewidth=2, label='Treated (KDE)')
    if len(control_ps) > 1:
        kde_control = gaussian_kde(control_ps)
        axes[0, 1].plot(x_range, kde_control(x_range), color='#3498db', linewidth=2, label='Control (KDE)')
    
    axes[0, 1].set_xlabel('Propensity Score', fontsize=11)
    axes[0, 1].set_ylabel('Density', fontsize=11)
    axes[0, 1].set_title('Distribution with Kernel Density Estimate', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Mirror Histogram (back-to-back)
    bins = np.linspace(0, 1, 51)
    treated_hist, _ = np.histogram(treated_ps, bins=bins, density=True)
    control_hist, _ = np.histogram(control_ps, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    axes[1, 0].barh(bin_centers, treated_hist, height=0.018, alpha=0.7, label='Treated', color='#e74c3c')
    axes[1, 0].barh(bin_centers, -control_hist, height=0.018, alpha=0.7, label='Control', color='#3498db')
    axes[1, 0].set_ylabel('Propensity Score', fontsize=11)
    axes[1, 0].set_xlabel('Density', fontsize=11)
    axes[1, 0].set_title('Mirror Histogram (Common Support Assessment)', fontsize=12, fontweight='bold')
    axes[1, 0].axvline(x=0, color='black', linewidth=0.8)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Box Plot
    data_for_box = [control_ps, treated_ps]
    bp = axes[1, 1].boxplot(data_for_box, labels=['Control', 'Treated'],
                            patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    
    axes[1, 1].set_ylabel('Propensity Score', fontsize=11)
    axes[1, 1].set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    # Add summary statistics
    common_min = max(treated_ps.min(), control_ps.min())
    common_max = min(treated_ps.max(), control_ps.max())
    
    stats_text = f"""Treated: n={len(treated_ps)}, mean={treated_ps.mean():.3f}, range={np.min(treated_ps):.3f}, {np.max(treated_ps):.3f}
    Control: n={len(control_ps)}, mean={control_ps.mean():.3f}, range={np.min(control_ps):.3f}, {np.max(control_ps):.3f}
    Common Support: [{common_min:.3f}, {common_max:.3f}]"""
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    # Print detailed diagnostics
    print("\n" + "=" * 80)
    print("PROPENSITY SCORE DIAGNOSTICS")
    print("=" * 80)
    
    print(f"\nTreated Group (n={len(treated_ps)}):")
    print(f"  Range: [{treated_ps.min():.4f}, {treated_ps.max():.4f}]")
    print(f"  Mean: {treated_ps.mean():.4f}")
    print(f"  Median: {np.median(treated_ps):.4f}")
    print(f"  Std: {treated_ps.std():.4f}")
    
    print(f"\nControl Group (n={len(control_ps)}):")
    print(f"  Range: [{control_ps.min():.4f}, {control_ps.max():.4f}]")
    print(f"  Mean: {control_ps.mean():.4f}")
    print(f"  Median: {np.median(control_ps):.4f}")
    print(f"  Std: {control_ps.std():.4f}")
    
    # Common support region
    print(f"\nCommon Support Region: [{common_min:.4f}, {common_max:.4f}]")
    
    # Check for extreme propensity scores
    extreme_low = np.sum(propensity_scores < 0.1)
    extreme_high = np.sum(propensity_scores > 0.9)
    print(f"\nExtreme Scores:")
    print(f"  < 0.1: {extreme_low} ({100*extreme_low/len(propensity_scores):.1f}%)")
    print(f"  > 0.9: {extreme_high} ({100*extreme_high/len(propensity_scores):.1f}%)")
    
    # Overlap assessment
    treated_in_support = np.sum((treated_ps >= common_min) & (treated_ps <= common_max))
    control_in_support = np.sum((control_ps >= common_min) & (control_ps <= common_max))
    print(f"\nUnits in Common Support:")
    print(f"  Treated: {treated_in_support}/{len(treated_ps)} ({100*treated_in_support/len(treated_ps):.1f}%)")
    print(f"  Control: {control_in_support}/{len(control_ps)} ({100*control_in_support/len(control_ps):.1f}%)")
    print("=" * 80)


# ==============================================================================
# FINAL COHORT ASSEMBLY
# ==============================================================================
def combine_matched_cohorts(
    matched_control_df: pd.DataFrame,
    treated_df: pd.DataFrame,
    treatment_column: str = "condition",
    control_eid_column: str = "control_eid",
    treated_eid_column: str = "eid",
    drop_matching_cols: bool = True
) -> pd.DataFrame:
    """
    Combine matched control and treated cohorts into a single DataFrame.
    
    Parameters
    ----------
    matched_control_df : pd.DataFrame
        Matched control cohort from propensity_score_matching()
    treated_df : pd.DataFrame
        Original treated cohort
    treatment_column : str, optional
        Name for the treatment indicator column (default: "condition")
    control_eid_column : str, optional
        Name of the EID column in matched controls (default: "control_eid")
    treated_eid_column : str, optional
        Name of the EID column in treated cohort (default: "eid")
    drop_matching_cols : bool, optional
        Whether to drop matching-specific columns (treated_eid, propensity_score) (default: True)
    
    Returns
    -------
    pd.DataFrame
        Combined cohort with treatment indicator column
    
    Examples
    --------
    >>> combined = combine_matched_cohorts(
    ...     matched_control_df=matched_controls,
    ...     treated_df=depression_cohort,
    ...     treatment_column='depression_status'
    ... )
    >>> print(combined['depression_status'].value_counts())
    
    
    Raises
    ------
    ValueError
            If the input DataFrames do not contain the expected identifier columns
            and the function cannot align matched controls with treated subjects.

    Notes
    -----
    - ``matched_control_df`` is expected to contain a column specified by
        ``control_eid_column`` (default: ``'control_eid'``) that links back to
        the original control subject identifiers. If that column is missing the
        function will still attempt to concatenate but the linkage may be lost.
    - The function prints a summary of the combined cohort and does not write
        output to disk.
    """
    # Create copies
    matched_controls = matched_control_df.copy()
    treated = treated_df.copy()
    
    # Add treatment indicators
    matched_controls[treatment_column] = 0
    treated[treatment_column] = 1
    
    # Rename control_eid to standard eid
    if control_eid_column in matched_controls.columns:
        matched_controls = matched_controls.rename(columns={control_eid_column: treated_eid_column})
    
    # Drop matching-specific columns if requested
    if drop_matching_cols:
        cols_to_drop = ['treated_eid', 'propensity_score']
        for col in cols_to_drop:
            if col in matched_controls.columns:
                matched_controls = matched_controls.drop(columns=[col])
    
    # Combine cohorts
    combined = pd.concat([matched_controls, treated], axis=0, ignore_index=True)
    
    print(f"\nCombined cohort created:")
    print(f"  Controls: {(combined[treatment_column] == 0).sum()}")
    print(f"  Treated: {(combined[treatment_column] == 1).sum()}")
    print(f"  Total: {len(combined)}")
    print(f"  Columns: {len(combined.columns)}")
    
    return combined
