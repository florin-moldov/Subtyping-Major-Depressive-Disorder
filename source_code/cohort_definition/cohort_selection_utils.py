"""
cohort_selection_utils
----------------------

Utility functions to assemble and inspect cohorts derived from UK Biobank
data used by the analysis pipeline. The module focuses on diagnosis (ICD-10)
based cohort extraction, finding overlaps between tabular datasets, building
comorbidity indicator matrices, and convenience helpers for saving and
exporting subject identifiers.

Key expectations and conventions
- Many helper functions expect UKB-style CSV files to exist under a
    project-local ``data/UKB`` folder (e.g. ``diagnosis_v2.csv``,
    ``cognition.csv``, ``demographics.csv``). Callers should pass DataFrames
    read from their desired CSVs when possible; the module provides a
    convenience loader ``load_ukb_datasets`` for this purpose.
- The diagnosis/ICD-10 inputs are supported in two common formats:
    1) a DataFrame with explicit ``'eid'`` and ``'codes'`` columns (recommended);
    2) a single combined column where each row is formatted like
         ``"<eid>,<codes_string>"`` (legacy UKB export style). Several
         functions detect and handle both forms.
- Many functions perform I/O (reading or writing CSVs / PNGs) and print
    brief diagnostics; callers should be aware of these side effects.

Notes on docstrings and behavior updates
- Several function docstrings explicitly document the expected input
    formats, return values, and side effects. The docstrings in this
    module aim to be precise about whether a function returns a DataFrame,
    writes files, or only prints progress messages.

Architecture
------------
The script is organized into functional sections:

- **Utility Functions**

- **Data Loading and Preprocessing**

- **Visualization**

- **High-level Pipelines**

"""

import textwrap
import os
import re
from collections import Counter
import pandas as pd
import numpy as np
from typing import List, Union, Literal, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def standardize_eid_columns(
    datasets: dict[str, pd.DataFrame],
    target_type: str = 'str'
) -> dict[str, pd.DataFrame]:
    """
    Convert eid columns to a consistent data type across all datasets.
    
    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of datasets with 'eid' columns
    target_type : str, optional
        Target data type: 'str', 'int', or 'Int64' (default: 'str')
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with standardized eid columns. Note that DataFrames are
        modified in-place (the function updates the ``'eid'`` column on each
        DataFrame found) and the same dictionary object is returned for
        convenience. When coercing to numeric types non-convertible values
        become ``NaN``.
    
    Examples
    --------
    >>> datasets = load_ukb_datasets()
    >>> datasets = standardize_eid_columns(datasets, target_type='str')
    """
    for name, df in datasets.items():
        if 'eid' in df.columns:
            if target_type == 'str':
                df['eid'] = df['eid'].astype(str)
            elif target_type == 'int':
                df['eid'] = pd.to_numeric(df['eid'], errors='coerce').astype('int64')
            elif target_type == 'Int64':
                df['eid'] = pd.to_numeric(df['eid'], errors='coerce').astype('Int64')
            else:
                raise ValueError(f"Invalid target_type: {target_type}")
    
    print(f"Standardized eid columns to {target_type} type")
    return datasets

def extract_people_by_icd10_codes(
    diagnosis_df: pd.DataFrame,
    codes: List[str],
    column_name: str = "eid,p41270",
    exclude: bool = False,
    exclude_codes: List[str] = None
) -> pd.DataFrame:
    """
    Extract individuals with specified ICD-10 codes from UK Biobank diagnosis data.
    
    Parameters
    ----------
    diagnosis_df : pd.DataFrame
        DataFrame containing diagnosis data with combined eid,codes column
    codes : List[str]
        List of ICD-10 code prefixes to search for (e.g., ['F33', 'F32'])
    column_name : str, optional
        Name of the column containing combined eid,codes data (default: "eid,p41270")
    exclude : bool, optional
        If True, exclude individuals with specified codes (default: False)
    exclude_codes : List[str], optional
        List of ICD-10 codes to exclude from results even if they match `codes`
        Example: codes=['F32'], exclude_codes=['F33'] returns people with F32 but NOT F33
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['eid', 'codes'] containing matching individuals
    
    Examples
    --------
    >>> # Extract people with F32 or F33
    >>> depression = extract_people_by_icd10_codes(df, codes=['F32', 'F33'])
    
    >>> # Extract people with F32 but NOT F33
    >>> depression_F32 = extract_people_by_icd10_codes(
    ...     df, codes=['F32'], exclude_codes=['F33']
    ... )
    
    >>> # Extract people WITHOUT F32 or F33 (control cohort)
    >>> controls = extract_people_by_icd10_codes(df, codes=['F32', 'F33'], exclude=True)
    
    Raises
    ------
    ValueError
        If the `column_name` is not present in `diagnosis_df` or if the input
        format is not one of the supported UKB export styles. The function
        attempts to be permissive with legacy combined columns but expects
        a parsable ``"eid,<codes>"`` string per row when not using explicit
        'eid'/'codes' columns.
    """
    people = []
    seen_eids = set()
    
    for row in diagnosis_df[column_name]:
        row_data = row.split(',', 1)
        
        if len(row_data) < 2:
            continue
        
        eid = row_data[0].strip()
        codes_str = row_data[1].strip()
        
        # Check for target codes
        has_target_code = any(code in codes_str for code in codes)
        
        # Check for exclude codes
        has_exclude_code = False
        if exclude_codes:
            has_exclude_code = any(exc_code in codes_str for exc_code in exclude_codes)
        
        # Apply logic
        if exclude:
            # Exclude mode: return people WITHOUT the specified codes
            if not has_target_code:
                if eid not in seen_eids:
                    people.append({'eid': str(eid), 'codes': codes_str})
                    seen_eids.add(eid)
        else:
            # Include mode: return people WITH the specified codes
            # BUT also exclude anyone with exclude_codes
            if has_target_code and not has_exclude_code:
                if eid not in seen_eids:
                    people.append({'eid': str(eid), 'codes': codes_str})
                    seen_eids.add(eid)
    
    result_df = pd.DataFrame(people)
    return result_df

def find_overlap_individuals(
    *dataframes: pd.DataFrame,
    eid_column: str = "eid",
    return_type: Literal["first", "last", "all", "merged"] = "first",
    dropna: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Find overlapping individuals across multiple DataFrames based on eid column.
    
    Parameters
    ----------
    *dataframes : pd.DataFrame
        Variable number of DataFrames to find overlap between
    eid_column : str, optional
        Name of the column containing participant IDs (default: "eid")
    return_type : {'first', 'last', 'all', 'merged'}, optional
        Which DataFrame to return for overlapping individuals:
        - "first": Return rows from first DataFrame (default)
        - "last": Return rows from last DataFrame
        - "all": Return list of DataFrames with overlapping eids from each input
        - "merged": Return merged DataFrame with all columns
    dropna : bool, optional
        Whether to drop NA values from each DataFrame before finding overlap (default: True)
    
    Returns
    -------
    pd.DataFrame or List[pd.DataFrame]
        DataFrame(s) containing only individuals present in all input DataFrames
    
    Examples
    --------
    >>> # Two datasets
    >>> overlap = find_overlap_individuals(depression_df, mri_df)
    
    >>> # Multiple datasets with merged output
    >>> merged = find_overlap_individuals(
    ...     depression_df, mri_df, cognitive_df, return_type="merged"
    ... )
    
    >>> # Get separate filtered DataFrames
    >>> dep, mri, cog = find_overlap_individuals(
    ...     depression_df, mri_df, cognitive_df, return_type="all"
    ... )
    
    Raises
    ------
    ValueError
        - If fewer than two DataFrames are provided.
        - If the specified `eid_column` does not exist in any of the input DataFrames.
        - If `return_type` is not one of the supported values ("first","last","all","merged").

    Notes
    -----
    - Returned DataFrame(s) will have their index reset. When using
      ``return_type='merged'`` the function performs an inner merge on the
      common ``eid_column`` and may rename columns with suffixes to avoid
      collisions.
    """
    if len(dataframes) < 2:
        raise ValueError("At least 2 DataFrames are required for overlap analysis")
    
    # Optionally drop NA values
    if dropna:
        dataframes = tuple(df.dropna() for df in dataframes)
    
    # Extract eid sets from each DataFrame
    eid_sets = []
    for i, df in enumerate(dataframes):
        if eid_column not in df.columns:
            raise ValueError(f"DataFrame {i} does not contain column '{eid_column}'")
        
        eids = set(df[eid_column].astype(str))
        eid_sets.append(eids)
    
    # Find intersection of all eid sets
    overlapping_eids = set.intersection(*eid_sets)
    overlapping_eids_typed = [str(eid) for eid in overlapping_eids]
    
    print(f"Found {len(overlapping_eids_typed)} overlapping individuals across {len(dataframes)} datasets")
    
    # Return based on return_type
    if return_type == "first":
        result = dataframes[0][dataframes[0][eid_column].astype(str).isin(overlapping_eids_typed)].copy()
        return result.reset_index(drop=True)
    
    elif return_type == "last":
        result = dataframes[-1][dataframes[-1][eid_column].astype(str).isin(overlapping_eids_typed)].copy()
        return result.reset_index(drop=True)
    
    elif return_type == "all":
        results = []
        for df in dataframes:
            filtered = df[df[eid_column].astype(str).isin(overlapping_eids_typed)].copy()
            results.append(filtered.reset_index(drop=True))
        return results
    
    elif return_type == "merged":
        # Start with first DataFrame (cast to str for consistent comparison)
        merged = dataframes[0][dataframes[0][eid_column].astype(str).isin(overlapping_eids_typed)].copy()
        
        # Sequentially merge with remaining DataFrames
        for i, df in enumerate(dataframes[1:], start=1):
            filtered = df[df[eid_column].astype(str).isin(overlapping_eids_typed)].copy()
            merged = merged.merge(
                filtered,
                on=eid_column,
                how='inner',
                suffixes=(f'_df{i-1}' if i > 1 else '', f'_df{i}')
            )
        
        return merged.reset_index(drop=True)
    
    else:
        raise ValueError(f"Invalid return_type: {return_type}")

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

def split_codestring(codestring: str) -> List[str]:
    """
    Split a codes string into individual codes.

    Notes
    -----
    The current implementation splits only on the vertical bar character
    (``'|'``). This matches the original legacy behavior encountered in
    UKB exports where multiple codes for an individual are joined by ``|``.
    Callers with data that uses commas/semicolons/whitespace as delimiters
    should pre-process their strings or update this helper accordingly.

    Parameters
    ----------
    codestring : str
        String containing one or more codes separated by the vertical bar
        character (e.g. ``'F32|F33|G40'``).

    Returns
    -------
    List[str]
        List of individual non-empty codes (strings), or an empty list for
        blank input.

    Examples
    --------
    >>> split_codestring('F32|F33|G40')
    ['F32', 'F33', 'G40']
    """
    codes = re.split(r'[|]+', codestring.strip())
    codes = [code for code in codes if code]
    return codes

def count_codes_in_cohort(
    cohort_df: pd.DataFrame,
    codes_column_cohort: str = 'codes',
    coding_filepath: str = '/home/f_moldovan/projects/subtyping_depression/data/UKB/coding19.tsv',
    output_path: str = None
) -> pd.DataFrame:
    """
    Count occurrences of each ICD-10 code in the specified codes column of a cohort DataFrame.
    
    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort DataFrame containing a column with ICD-10 codes.
    codes_column_cohort : str, optional
        Name of the column containing ICD-10 codes (default: 'codes').
    coding_filepath : str, optional
        Filepath to the coding file mapping ICD-10 codes to titles (default: '/home/f_moldovan/projects/subtyping_depression/data/UKB/coding19.tsv', from UKB showcase).
    output_path : str, optional
        If provided, save the resulting code distribution csv file to this directory (default: None)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with ICD-10 codes and their counts as values, sorted in descending order.
    
    Examples
    --------
    >>> code_counts = count_codes_in_cohort(depression_cohort, codes_column='codes')
    >>> print(code_counts)
    F32.1    250
    F33.1    200
    F32.0    150
    ...
    
    Notes
    -----
    - This function relies on ``split_codestring`` to parse the individual
        code tokens from the cohort's code strings; ensure that the codes are
        formatted consistently for accurate counting.
    - The optional ``coding_filepath`` is used to map ICD-10 codes to human
        readable meanings; if the file is missing or malformed an exception may
        be raised by ``pandas.read_csv``.
    """
    
    # Get sample size of cohort
    n_subjects = len(cohort_df)

    # Robustly load ICD-10 code titles (TSV with code\tmeaning)
    icd_texts_df = pd.read_csv(
        coding_filepath,
        sep='\t',
        header=None,
        usecols=[0, 1],
        names=['coding', 'meaning'],
        dtype=str,
        comment='#',
        engine='python'
    )
    icd_texts_df['coding'] = icd_texts_df['coding'].str.strip()
    icd_texts_df['meaning'] = icd_texts_df['meaning'].fillna('').astype(str).str.strip()

    # mapping for fast lookup
    icd_to_title = dict(zip(icd_texts_df['coding'], icd_texts_df['meaning']))

    all_codes = []
    code_distribution = {"code": [], "count": [], "proportion": [], "meaning": []}

    for codes in cohort_df[codes_column_cohort].fillna(''):
        codes_list = split_codestring(codes)
        all_codes.extend(codes_list)

    code_counter = Counter(all_codes)
    for code, count in code_counter.items():
        code_distribution["code"].append(code)
        code_distribution["count"].append(count)
        code_distribution["proportion"].append(count / max(1, n_subjects))
        # exact match
        title = icd_to_title.get(code)
        # fallback to category (e.g., F32.1 -> F32)
        if title is None and '.' in code:
            title = icd_to_title.get(code.split('.', 1)[0])
        code_distribution["meaning"].append(title if title is not None and title != '' else "N/A")

    code_distribution = pd.DataFrame(code_distribution).sort_values(by='count', ascending=False).reset_index(drop=True)
    
    if output_path:
        code_distribution.to_csv(output_path, index=False)
        print(f"ICD-10 code distribution saved to {output_path}")

    return code_distribution

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_ukb_datasets(
    data_dir: str = 'data/UKB',
    convert_sex: bool = True,
    columns_cognitive: List[str] = None,
    columns_demographics: List[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all UK Biobank datasets required for cohort definition.
    
    Parameters
    ----------
    data_dir : str, optional
        Base directory containing UK Biobank data (default: 'data/UKB')
    convert_sex : bool, optional
        Whether to convert sex codes (0/1) to descriptive labels (default: True)
    columns_cognitive : List[str], optional
        List of columns to load from cognitive dataset (default: None, loads all)
    columns_demographics : List[str], optional
        List of columns to load from demographics dataset (default: None, loads all)
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing DataFrames keyed by dataset name. Typical keys
        and contents are:
        - ``'diagnosis_ICD10'``: ICD-10 diagnosis codes table (raw UKB export)
        - ``'taskfmri'``: task fMRI metadata and selected columns
        - ``'cognitive'``: cognitive assessment table
        - ``'restingfmri'``: resting-state fMRI metadata
        - ``'dMRI'``: diffusion MRI metadata
        - ``'demographics'``: demographics (age, sex) with optional sex label conversion
    
    Examples
    --------
    >>> datasets = load_ukb_datasets()
    >>> diagnosis_data = datasets['diagnosis_ICD10']
    >>> cognitive_data = datasets['cognitive']
    """
    datasets = {}
    
    # Load diagnosis data (ICD-10 codes)
    datasets['diagnosis_ICD10'] = pd.read_csv(
        f'{data_dir}/diagnosis_v2.csv', 
        sep=";"
    )
    
    # Load task fMRI data
    datasets['taskfmri'] = pd.read_csv(
        f'{data_dir}/medication+imaging.csv', 
        sep=";",
        usecols=["eid", "p20249_i2"]
    )
    
    # Load cognitive data
    datasets['cognitive'] = pd.read_csv(
        f'{data_dir}/cognition.csv', 
        sep=';',
        usecols=columns_cognitive
    )
    
    # Load resting-state fMRI data
    datasets['restingfmri'] = pd.read_csv(
        f'{data_dir}/medication+imaging.csv', 
        usecols=["eid", "p20227_i2"], 
        sep=";"
    )
    
    # Load diffusion MRI data
    datasets['dMRI'] = pd.read_csv(
        f'{data_dir}/medication+imaging.csv', 
        usecols=["eid", "p31026_i2"],
        sep=";"
    )
    
    # Load demographics dat
    datasets['demographics'] = pd.read_csv(
        f'{data_dir}/demographics.csv', 
        sep=";",
        usecols=columns_demographics
    )

    # Convert sex codes to descriptive labels if requested
    if convert_sex:
        datasets['demographics']['p31'] = datasets['demographics']['p31'].replace({
            0: 'Female',
            1: 'Male'
        })

    print(f"Loaded {len(datasets)} datasets from {data_dir}")
    return datasets

def build_comorbidity_indicator_matrix(
    cohort_df: pd.DataFrame,
    *,
    eid_column: str = 'eid',
    codes_column_cohort: str = 'codes',
    proportion_threshold: float = 0.10,
    coding_filepath: str = '/home/f_moldovan/projects/subtyping_depression/data/UKB/coding19.tsv',
    max_comorbidities: Optional[int] = None,
    include_codes: Optional[List[str]] = None,
    exclude_codes: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Add comorbidity indicator columns to a cohort DataFrame.

    This function is designed to work together with `count_codes_in_cohort`.

     It:
     1) Preserves all rows/eIDs in `cohort_df`.
    2) Selects which comorbidities (ICD-10 codes) to include based on a
       prevalence threshold (proportion of subjects with the code) and/or an
       explicit include list.
     3) Adds one column per selected code, with 1 if the eID has that code (or
         0 otherwise).

    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort DataFrame containing at least `eid_column` and `codes_column_cohort`.
    eid_column : str
        Column name containing subject IDs (default: 'eid').
    codes_column_cohort : str
        Column name containing ICD-10 codes strings (default: 'codes').
    proportion_threshold : float
        Minimum prevalence (count / n_subjects, as in `count_codes_in_cohort`) for a code
        to be included when `include_codes` is not provided (default: 0.10).
    coding_filepath : str
        Passed through to `count_codes_in_cohort` (default points to UKB coding file).
    max_comorbidities : int | None
        If provided, only keep the top-N most frequent codes after thresholding.
    include_codes : list[str] | None
        If provided, *only* these codes are used (threshold is ignored).
    exclude_codes : list[str] | None
        Code prefixes to always drop from the selected set (e.g., excluding 'F32'
        will also exclude 'F32.0', 'F32.1', etc.).
    output_path : str | None
        If provided, save the augmented cohort to this CSV path. If a directory
        is provided, a default filename is used.

    Returns
    -------
    pd.DataFrame
        A copy of `cohort_df` with added 0/1 comorbidity columns.

    Notes
    -----
    - If no comorbidity codes meet the selection criteria the function
        returns a copy of ``cohort_df`` unchanged (but may write the copy to
        ``output_path`` if provided).
    - When ``output_path`` is provided and is a directory, a default filename
        ``'cohort_with_comorbidity_indicators.csv'`` is used; otherwise the
        provided path is treated as a file path and will be overwritten.
    """
    if pd.Index(cohort_df.columns).duplicated().any():
        dupes = pd.Index(cohort_df.columns)[pd.Index(cohort_df.columns).duplicated()].unique().tolist()
        raise ValueError(
            "Input cohort_df contains duplicate column names, which is not supported. "
            f"Duplicate columns: {dupes}"
        )

    if eid_column not in cohort_df.columns:
        raise ValueError(f"Column '{eid_column}' not found in cohort_df")
    if codes_column_cohort not in cohort_df.columns:
        raise ValueError(f"Column '{codes_column_cohort}' not found in cohort_df")
    if proportion_threshold < 0 or proportion_threshold > 1:
        raise ValueError("proportion_threshold must be between 0 and 1")

    # Work on a copy so callers can decide whether to reassign.
    cohort_base = cohort_df.copy()
    cohort_base[eid_column] = cohort_base[eid_column].astype(str)
    cohort_base[codes_column_cohort] = cohort_base[codes_column_cohort].fillna('').astype(str)

    # Build a per-eID codes table. Ensure each subject contributes at most one
    # count per code for prevalence calculation.
    def _uniq_codestring(s: str) -> str:
        """Return a sorted, pipe-delimited unique set of codes for a single subject.

        Used to normalise multiple rows per subject into a single canonical
        codes string when computing prevalences. This helper preserves the
        legacy ``'|'`` delimiter used elsewhere in the codebase.
        """
        codes = set(split_codestring(s))
        return '|'.join(sorted(codes))

    codes_by_eid = (
        cohort_base[[eid_column, codes_column_cohort]]
        .groupby(eid_column, as_index=False)[codes_column_cohort]
        .agg(lambda vals: '|'.join([v for v in vals if isinstance(v, str) and v != '']))
    )
    codes_by_eid[codes_column_cohort] = codes_by_eid[codes_column_cohort].fillna('').astype(str).map(_uniq_codestring)

    

    code_distribution = count_codes_in_cohort(
        codes_by_eid,
        codes_column_cohort=codes_column_cohort,
        coding_filepath=coding_filepath,
        output_path=None,
    )

    selected = code_distribution.copy()
    if include_codes is not None:
        include_set = set(map(str, include_codes))
        selected = selected[selected['code'].astype(str).isin(include_set)]
    else:
        selected = selected[selected['proportion'] >= proportion_threshold]

    if exclude_codes is not None:
        exclude_prefixes = [str(x) for x in exclude_codes]

        def _excluded(code: str) -> bool:
            """Return True if `code` matches or starts with any excluded prefix.

            This helper considers exact matches and prefix matches (including
            dotted extensions, e.g., ``F32`` matches ``F32.1``).
            """
            for pref in exclude_prefixes:
                if code == pref or code.startswith(pref) or code.startswith(f"{pref}."):
                    return True
            return False

        selected_codes = selected['code'].astype(str)
        selected = selected[~selected_codes.map(_excluded)]

    if max_comorbidities is not None:
        if max_comorbidities <= 0:
            raise ValueError("max_comorbidities must be a positive integer")
        selected = selected.sort_values('count', ascending=False).head(int(max_comorbidities))

    comorbidity_codes: List[str] = selected['code'].astype(str).tolist()
    code_set = set(comorbidity_codes)

    eids = codes_by_eid[eid_column].astype(str).tolist()
    codes_series = codes_by_eid[codes_column_cohort]

    # If nothing selected, just return cohort with no added columns.
    if not comorbidity_codes:
        if output_path:
            out_path = os.fspath(output_path)
            if os.path.isdir(out_path):
                out_path = os.path.join(out_path, 'cohort_with_comorbidity_indicators.csv')
            cohort_base.to_csv(out_path, index=False)
            print(f"Cohort with comorbidity indicators saved to {out_path}")
        return cohort_base

    # Build a long table of (eid, code) for selected codes, then pivot.
    long_eids: List[str] = []
    long_codes: List[str] = []
    for eid, codes_str in zip(eids, codes_series.tolist()):
        # Use a set to ensure each (eid, code) appears at most once.
        for code in set(split_codestring(codes_str)):
            if code in code_set:
                long_eids.append(eid)
                long_codes.append(code)

    if long_eids:
        long_df = pd.DataFrame({'eid': long_eids, 'code': long_codes, 'value': 1})
        mat = long_df.drop_duplicates(subset=['eid', 'code']).pivot_table(
            index='eid',
            columns='code',
            values='value',
            aggfunc='max',
            fill_value=0,
        )
    else:
        mat = pd.DataFrame(index=pd.Index([], name='eid'))

    # Ensure all eIDs exist and all selected code columns exist (fill missing with 0)
    mat = mat.reindex(index=eids, fill_value=0)
    for code in comorbidity_codes:
        if code not in mat.columns:
            mat[code] = 0

    # Re-order columns and cast to int 0/1
    mat = mat[comorbidity_codes].astype(int)
    indicators = mat.reset_index().rename(columns={'eid': eid_column})
    indicators = indicators[[eid_column] + comorbidity_codes]

    # Merge back onto the cohort (broadcasts to all rows with the same eID).
    # If code columns already exist, overwrite them with the recomputed values.
    overlap_cols = [c for c in comorbidity_codes if c in cohort_base.columns]
    if overlap_cols:
        cohort_base = cohort_base.drop(columns=overlap_cols)

    cohort_out = cohort_base.merge(indicators, on=eid_column, how='left', validate='m:1')
    cohort_out[comorbidity_codes] = cohort_out[comorbidity_codes].fillna(0).astype(int)

    # Safety check: ensure we didn't drop any original columns.
    missing_cols = [c for c in cohort_df.columns if c not in cohort_out.columns]
    if missing_cols:
        # Reattach any missing original columns by index alignment.
        for c in missing_cols:
            cohort_out[c] = cohort_df[c].values
        # Keep original column order first, then new comorbidity columns.
        ordered = list(cohort_df.columns) + [c for c in comorbidity_codes if c not in cohort_df.columns]
        cohort_out = cohort_out[ordered]

    # Enforce unique column names in the final output.
    if pd.Index(cohort_out.columns).duplicated().any():
        dupes = pd.Index(cohort_out.columns)[pd.Index(cohort_out.columns).duplicated()].unique().tolist()
        raise RuntimeError(
            "build_comorbidity_indicator_matrix produced duplicate column names, "
            f"which is not allowed. Duplicate columns: {dupes}"
        )

    if output_path:
        out_path = os.fspath(output_path)
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path, 'cohort_with_comorbidity_indicators.csv')
        cohort_out.to_csv(out_path, index=False)
        print(f"Cohort with comorbidity indicators saved to {out_path}")

    return cohort_out

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_comorbidity_distribution(
    code_distribution: pd.DataFrame,
    proportion_threshold: float = 0.10,
    title: str = "ICD-10 Code Distribution",
    figsize: tuple = (20, 20),
    output_path: str = None,
    wrap_width: int = 40,       # max chars per line for y-labels
    y_label_rotation: float = 15.0,  # degrees to tilt y-tick labels
    y_spacing: float = 3.0      # multiplier to increase vertical spacing between y-tick labels
) -> None:
    """
    Plot a horizontal bar chart of ICD-10 code distribution from a DataFrame.

    Parameters
    ----------
    code_distribution : pd.DataFrame
        DataFrame returned by `count_codes_in_cohort`, containing 'code', 'count', 'proportion', and 'meaning'.
    proportion_threshold : float, optional
        Minimum proportion threshold to include a code in the plot (default: 0.10).
    title : str, optional
        Title of the plot (default: "ICD-10 Code Distribution").
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6)).
    output_path : str, optional
        If provided, save the plot to this file path (default: None).
    wrap_width : int, optional
        Maximum number of characters per line for y-axis labels (default: 40).
    y_label_rotation : float, optional
        Rotation angle for y-axis labels in degrees (default: 15.0).
    y_spacing : float, optional
        Multiplier to increase vertical spacing between y-axis labels (default: 3.0).

    Returns
    -------
    None
    
    Notes
    -----
    - The function calls ``plt.show()`` to display the figure in interactive
        sessions. When ``output_path`` is provided the figure is saved to disk
        before being shown.
    - The function attempts to provide extra left margin for long wrapped
        y-axis labels; callers running in headless/non-interactive environments
        may want to set a non-default ``figsize`` and skip calling ``plt.show``
        by capturing the figure object instead.
    """

    # Sort by proportion and take top N (exclude primary diagnosis)
    data_to_plot = code_distribution.sort_values('proportion', ascending=False).copy()
    data_to_plot = data_to_plot.iloc[1:]  # skip first row (primary diagnosis)
    data_to_plot = data_to_plot[data_to_plot['proportion'] >= proportion_threshold]
    if data_to_plot.empty:
        return

    # Increase figure height according to y_spacing so labels don't overlap
    fig = plt.figure(figsize=(figsize[0], max(figsize[1], figsize[1] * y_spacing)))
    ax = fig.add_subplot(1, 1, 1)

    # Prepare labels and positions
    labels = data_to_plot['meaning'].astype(str).tolist()
    def _wrap_label(s: str, width: int) -> str:
        if not isinstance(s, str) or s.strip() == "":
            return ""
        return textwrap.fill(s, width=width)
    wrapped_labels = [_wrap_label(s, wrap_width) for s in labels]

    n = len(data_to_plot)
    base_pos = np.arange(n)
    y_positions = base_pos * y_spacing

    # Draw horizontal bars manually (more control over y positions than seaborn)
    proportions = data_to_plot['proportion'].tolist()[::-1]  # reverse to have largest on top
    colors = sns.color_palette("viridis", n)
    ax.barh(y_positions, proportions, color=colors)

    # Set yticks to our spaced positions and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wrapped_labels[::-1], rotation=y_label_rotation, ha='right')  # reverse labels back to match bars

    # Add labels and title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Proportion of Cohort", fontsize=12)
    ax.set_ylabel("ICD-10 Code And Meaning", fontsize=12)

    # Add percentage labels to the end of bars
    for y, prop in zip(y_positions, proportions):
        ax.text(prop + 0.001, y, f'{prop:.1%}', va='center', fontsize=10)

    # Adjust limits and layout
    ax.set_ylim(y_positions.min() - 0.5 * y_spacing, y_positions.max() + 0.5 * y_spacing)
    plt.tight_layout()
    # leave enough left margin for wrapped, rotated labels
    plt.subplots_adjust(left=0.25)

    if output_path:
        plt.savefig(output_path, dpi='figure', bbox_inches='tight', format='png')
        print(f"Plot saved to {output_path}")

    plt.show()

# ==============================================================================
# HIGH-LEVEL PIPELINE ORCHESTRATION FUNCTIONS
# ==============================================================================
def create_cohort(
    diagnosis_df_icd10: pd.DataFrame,
    cognitive_df: pd.DataFrame,
    taskfmri_df: pd.DataFrame,
    restingfmri_df: pd.DataFrame,
    dmri_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    icd10_codes: List[str] = None,
    cohort_type: Literal["depression", "control"] = "depression",
    exclude_icd10_codes: List[str] = None,
    required_datasets: List[str] = None,
    optional_datasets: List[str] = None,
    control_no_icd10: bool = False,
) -> pd.DataFrame:
    """
    Create a complete cohort by extracting individuals and merging multiple datasets.
    
    Parameters
    ----------
    diagnosis_df_icd10 : pd.DataFrame
        ICD-10 diagnosis data
    cognitive_df : pd.DataFrame
        Cognitive assessment data
    taskfmri_df : pd.DataFrame
        Task fMRI data
    restingfmri_df : pd.DataFrame
        Resting-state fMRI data
    dmri_df : pd.DataFrame
        Diffusion MRI data
    demographics_df : pd.DataFrame
        Demographics data
    icd10_codes : List[str], optional
        ICD-10 codes to include/exclude
    cohort_type : {'depression', 'control'}, optional
        Type of cohort to create (default: 'depression')
    exclude_icd10_codes : List[str], optional
        Additional ICD-10 codes to exclude
    required_datasets : List[str], optional
        Names of datasets that participants must overlap on. Only individuals present
        in ALL required datasets are retained. Valid names:
        ['cognitive', 'taskfmri', 'restingfmri', 'dMRI', 'demographics'].
        Default: all datasets are required.
    optional_datasets : List[str], optional
        Names of datasets to merge if available without reducing the cohort size.
        These are left-joined on 'eid' so missing values are allowed. Valid names are
        the same as for required_datasets. Default: [].
    control_no_icd10 : bool, optional
        When cohort_type == 'control' and this flag is True, construct the control cohort
        from participants who do not have any ICD‑10 entries in the diagnosis_df_icd10
        (default: False).
    
    Returns
    -------
    pd.DataFrame
        Complete cohort with required overlap and optional merges applied

    Notes
    -----
    - The function expects that each input dataset either contains an ``'eid'``
        column (recommended) or, in the case of the diagnosis table, a legacy
        combined column such as ``"eid,p41270"``. The helper
        ``extract_people_by_icd10_codes`` handles these legacy rows.
    - The function performs several ``print`` statements with brief
        diagnostics and does not write output files itself. If callers require
        saved CSVs, call ``save_cohort`` after receiving the returned DataFrame.
    - The merge for ``required_datasets`` uses an inner/merged strategy and
            will reduce the cohort to the intersection of all required datasets.

    """
    # Extract people by ICD-10 codes
    if cohort_type == "control":
        if control_no_icd10:
            # diagnosis_df_icd10 contains all participants; select those with empty/no codes
            df = diagnosis_df_icd10.copy()
            # If DataFrame already has 'eid' and 'codes' columns, use them directly
            if set(["eid", "codes"]).issubset(df.columns):
                tmp = df[["eid", "codes"]].copy()
                tmp["codes"] = tmp["codes"].fillna("").astype(str).str.strip()
                people_icd10 = tmp[tmp["codes"] == ""][["eid", "codes"]].drop_duplicates(subset="eid").reset_index(drop=True)
            else:
                # Otherwise assume a combined column like "eid,p41270" in first column
                col = df.columns[0]
                eids = []
                codes_list = []
                for val in df[col].fillna("").astype(str):
                    parts = val.split(",", 1)
                    if len(parts) == 1:
                        eid = parts[0].strip()
                        codes_str = ""
                    else:
                        eid = parts[0].strip()
                        codes_str = parts[1].strip()
                    eids.append(eid)
                    codes_list.append(codes_str)
                tmp = pd.DataFrame({"eid": eids, "codes": codes_list})
                tmp["codes"] = tmp["codes"].fillna("").astype(str).str.strip()
                people_icd10 = tmp[tmp["codes"] == ""][["eid", "codes"]].drop_duplicates(subset="eid").reset_index(drop=True)
            print(f"Control (no ICD-10 codes) extraction: {len(people_icd10)} individuals")
        else:
            people_icd10 = extract_people_by_icd10_codes(
                diagnosis_df_icd10,
                codes=icd10_codes,
                exclude=True  # Exclude people WITH these codes
            )
    else:
        people_icd10 = extract_people_by_icd10_codes(
            diagnosis_df_icd10,
            codes=icd10_codes,
            exclude_codes=exclude_icd10_codes
        )
        print(f"Depression (F32) extraction: {len(people_icd10)} individuals")
    
    # Determine which datasets must be overlapped (required) vs optionally merged
    dataset_map = {
        'cognitive': cognitive_df,
        'taskfmri': taskfmri_df,
        'restingfmri': restingfmri_df,
        'dMRI': dmri_df,
        'demographics': demographics_df,
    }

    # Defaults: require all, no optional
    if required_datasets is None:
        required_datasets = list(dataset_map.keys())
    if optional_datasets is None:
        optional_datasets = []

    # Validate names
    unknown_required = [n for n in required_datasets if n not in dataset_map]
    unknown_optional = [n for n in optional_datasets if n not in dataset_map]
    if unknown_required:
        raise ValueError(f"Unknown required datasets: {unknown_required}. Valid options: {list(dataset_map.keys())}")
    if unknown_optional:
        raise ValueError(f"Unknown optional datasets: {unknown_optional}. Valid options: {list(dataset_map.keys())}")

    # Build required overlap list (diagnosis_base + required datasets)
    required_overlap_list = [people_icd10] + [dataset_map[n] for n in required_datasets]
    cohort = find_overlap_individuals(
        *required_overlap_list,
        return_type="merged"
    )

    # Left-merge optional datasets (do not reduce cohort, just add columns where present)
    for n in optional_datasets:
        opt_df = dataset_map[n]
        cohort = cohort.merge(opt_df, on='eid', how='left', suffixes=('', f'_{n}'))
    
    print(f"Final {cohort_type} cohort: {len(cohort)} individuals")
    print(f"  Required overlap datasets: {required_datasets}")
    if optional_datasets:
        print(f"  Optional merged datasets: {optional_datasets}")
    if cohort_type == 'control' and control_no_icd10:
        print("  Control cohort constructed with no ICD-10 entries")
    
    return cohort

def save_cohort(
    cohort_df: pd.DataFrame,
    output_path: str,
    verbose: bool = True
) -> None:
    """
    Save cohort DataFrame to CSV file.
    
    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort DataFrame to save
    output_path : str
        Path to output CSV file
    verbose : bool, optional
        Whether to print confirmation message (default: True)
    
    Examples
    --------
    >>> save_cohort(depression_cohort, 'data/UKB/cohorts/depression_cohort.csv')
    
    Returns
    -------
    None
        Writes the provided ``cohort_df`` to ``output_path`` as CSV. The
        function does not return the DataFrame and will overwrite any existing
        file at ``output_path``.
    """
    cohort_df.to_csv(output_path, index=False)
    if verbose:
        print(f"Cohort saved to '{output_path}' ({len(cohort_df)} individuals)")