"""Utility functions for robust depression–cognition association analyses.

This module provides the core statistical, plotting, and logging
building blocks used by `modular_cog_associations_main.py` to implement a
median-based depression–cognition analysis pipeline focusing on module-level
connectivity derived depression subtypes.

Main responsibilities
---------------------

* **Cohort loading and renaming**

    - :func:`load_and_rename_cohort_data` loads a combined control +
        depression cohort CSV and standardizes key cognitive and covariate
        column names into the human-readable names used throughout the
        project (e.g. mapping raw UK Biobank field IDs to descriptive
        labels such as ``age_at_assessment`` or
        ``Snap_task_mean_reaction_time``).

* **Robust task-wise and domain-wise z-scores**

    - :func:`calculate_robust_z_scores` computes robust, control-
        referenced z-scores for specified cognitive variables using the
        control group median and median absolute deviation (MAD) as
        reference. It returns the depression cohort rows with new ``*_z``
        columns and appends a detailed summary of control medians, MADs,
        and depression-group z-score distributions to an optional log file
        (including a caller-specified ``log_context`` string, which
        typically encodes connectivity type and cluster).

    - :func:`calculate_composite_z_score` aggregates sets of task-wise
        z-scores into composite cognitive domains (e.g. processing speed,
        working memory) using either the mean or the median across
        constituent tasks. It writes a concise description of the domain,
        included variables, and composite distribution (median, IQR, range)
        to an optional log file.

* **Quantile regression and R integration**

    - :func:`quantile_regression` wraps R's :mod:`quantreg` package via
        :mod:`rpy2` to perform two types of analyses on data supplied
        through a temporary CSV:

            1. **Cluster-contrast mode** (``test_against_zero=False``): fits a
                 separate quantile regression for each dependent variable of the
                 form ``DV ~ group_column + covariates`` at a specified
                 quantile (typically ``tau=0.5`` for the median). It supports
                 arbitrary group factor (``Cluster 1`` vs ``Cluster 0``) and 
                 returns p-values and effect estimates for the requested comparison levels.

            2. **One-sample mode** (``test_against_zero=True``): fits
                 ``DV ~ covariates`` or ``DV ~ 1`` for already z-scored
                 variables and tests whether the median differs from 0.

    - All R-side console output (model summaries, warnings, custom
        ``cat`` calls) is directed into a caller-specified log file via R's
        ``sink`` mechanism; nothing is printed to the Python terminal. When
        present in the data, the ``Connectivity_Type`` and ``Cluster``
        columns are also logged, so each R block clearly indicates the
        modality and cluster context of the analysis.

* **Multiple testing support**

    - :func:`apply_multiple_testing_correction` provides FDR
        (Benjamini–Hochberg) and Bonferroni corrections for a list of
        p-values. Instead of printing to stdout, it builds a formatted
        table (variable name, test type, raw p-value, adjusted p-value,
        significance marker) and appends it to an optional text log,
        prefixed with a free-form ``log_context`` string that usually
        encodes the level of analysis (task-wise vs domain-wise) and the
        corresponding connectivity/cluster configuration.

* **Visualization utilities**

    - :func:`plot_cognitive_distributions_violin` generates violin plots of
        cognitive variable distributions across groups (e.g. depression vs control or connectivity clusters).

    - :func:`plot_conn_cognition_association` creates scatter plots with median-based regression lines to visualize relationships between 
        connectivity features and cognitive scores, optionally faceted by cluster.

    - :func:`plot_z_scores` generates radar-style (spider) plots of
        z-score profiles for either tasks or domains. It uses a stable color
        mapping to ensure that clusters are colored consistently across analyses according to which clustering approach 
        they originated from and supports flexible figure saving and overlaying of multiple cluster profiles into a 
        single radar plot for comparison. Also shows significance annotations for each variable based on 
        between-cluster FDR-corrected comparisons.

    - Several internal helpers and a global registry
        (``_RADAR_OVERLAY_STORE``) support building overlaid radar charts
        that combine multiple polygons (e.g. overall cohort vs connectivity
        clusters) into a single figure.

Architecture
------------
The script is organized into functional sections:

- **Utility Functions**

- **R Package Management**

- **Data Loading and Preprocessing**

- **Quantile Regression**

- **Visualization**

Important implementation notes
------------------------------
- Several functions are stateful: `plot_z_scores` will register per-call
    summaries in the module-level registries (``_RADAR_OVERLAY_STORE``,
    ``_RADAR_OVERLAY_PVALS``, etc.). These registries are used by
    ``_render_connectivity_overlay`` to build overlaid radar figures and
    therefore rely on callers to use consistent `save_path` naming
    conventions (for example filenames containing ``Cluster_0`` or
    ``schaefer1000+tian54``). The registries are not thread-safe.
- Several routines invoke R via :mod:`rpy2` (notably
    :func:`quantile_regression`). These functions set rpy2 conversion
    rules inside the function scope; callers running multiple R-enabled
    operations in the same interpreter should be aware that rpy2's
    conversion context is touched.
- Some operations are computationally intensive by default (for
    example, bootstrapping in ``quantile_regression`` uses a large
    default ``R``), so developers may want to pass smaller values when
    testing.
- Many functions write files (SVGs, CSVs, and log files). Functions
    that accept a ``save_path`` or ``log_path`` will create parent
    directories as needed and may print saved paths for convenience.
    File-write failures are generally caught to avoid breaking
    long-running batch analyses.
"""
 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats import multitest
import rpy2.robjects as ro  # assumes you have rpy2 installed and R installed and configured to version >= 4.0
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from typing import Dict, List, Tuple, Optional, Literal, Sequence, Union

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
# Global registry to assemble overlaid radar charts across calls
# Key: (depression_codes, association_type, base_kind['task'|'domain'])
_RADAR_OVERLAY_STORE: Dict[Tuple[str, str, str], Dict[str, Dict]] = {}
_RADAR_OVERLAY_PVALS: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]] = {}
_RADAR_OVERLAY_PVALS_META: Dict[Tuple[str, str, str], Dict[str, str]] = {}
_RADAR_OVERLAY_SE_OVERALL: Dict[Tuple[str, str, str], Dict[str, float]] = {}
_RADAR_OVERLAY_SE_CLUSTER: Dict[Tuple[str, str, str], Dict[str, Dict[str, Dict[str, float]]]] = {}

# Return a mapping from cluster label -> hex color for the given modality/direction
def _cluster_colors_for_modality(conn_type: str, dir_type: str) -> Dict[str, str]:
    """Return a cluster color mapping for a specified connectivity modality.

    This helper normalizes the provided ``conn_type`` and ``dir_type`` strings
    (lowercasing and stripping whitespace) and looks up a small built-in
    palette keyed by the tuple ``(conn_type, dir_type)``. The returned mapping
    uses string cluster labels (``'0'`` and ``'1'``) as keys and hexadecimal
    color codes as values. The result is directly suitable to pass as a
    seaborn/matplotlib ``palette`` argument.

    Parameters
    ----------
    conn_type : str
        Connectivity type. Expected values (case-insensitive) are
        ``'functional'``, ``'structural'`` or ``'sfc'``. If a value not
        present in the built-in palette is passed, a default color mapping
        will be returned.
    dir_type : str
        Direction string. Expected values (case-insensitive) are
        ``'internal'`` or ``'external'``. Non-matching values will fall back
        to the default mapping.

    Returns
    -------
    Dict[str, str]
        Mapping from cluster label strings ``'0'`` and ``'1'`` to hex color
        codes (e.g. ``{"0": "#380075", "1": "#b2df8a"}``).

    Notes
    -----
    - The function is intentionally permissive: unknown input values do not
      raise an error but instead return a sensible default palette. This
      makes plotting robust to minor upstream naming variations but can hide
      misspellings; callers that need strict validation should check inputs
      before calling.
    """
    key = (str(conn_type).lower().strip(), str(dir_type).lower().strip())
    palette = {
        ("functional", "internal"): {"0": "#380075", "1": "#b2df8a"},
        ("functional", "external"): {"0": "#e7298a", "1": "#66a61e"},
        ("structural", "internal"): {"0": "#ffd92f", "1": "#bc80bd"},
        ("structural", "external"): {"0": "#f1b6da", "1": "#a6d854"},
        ("sfc", "internal"): {"0": "#cab2d6", "1": "#b3de69"},
        ("sfc", "external"): {"0": "#0026ff", "1": "#fd7600"},
    }
    return palette.get(key, {"0": "#1f77b4", "1": "#ff7f0e"})


# Parse a modality and direction (e.g., 'functional','internal') from a string label
def _parse_modality_from_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a connectivity modality and direction from an arbitrary label.

    This is a lightweight substring-based parser used to extract a guessed
    connectivity type and direction from free-form labels (for example,
    filenames or figure titles).

    Parameters
    ----------
    label : str
        Input string to parse. The function performs a case-insensitive
        substring search for the tokens ``'functional'``, ``'structural'``,
        ``'sfc'``, and the directions ``'internal'`` / ``'external'``.

    Returns
    -------
    (Optional[str], Optional[str])
        Tuple ``(conn, direction)`` where each element is either the matched
        string (lowercase) or ``None`` if no match was found.
    """
    lower = str(label).lower()
    conn = "functional" if "functional" in lower else "structural" if "structural" in lower else "sfc" if "sfc" in lower else None
    direction = "internal" if "internal" in lower else "external" if "external" in lower else None
    return conn, direction


# Heuristically infer modality and direction from a filepath or string
def _infer_modality_from_path(path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Infer connectivity modality and direction heuristically from a file path.

    The function examines the provided ``path`` (typically a plot file path)
    and applies simple heuristics to guess the connectivity type and
    direction:

    - Preference is given to the parent directory name (e.g. ``functional_con``)
      but the filename itself is also inspected for tokens.
    - Filenames ending with ``_internal.svg`` or containing ``internal`` are
      classified as ``'internal'``; likewise for ``'external'``.

    Parameters
    ----------
    path : Optional[str]
        File path or filename to inspect. If ``None`` or empty, ``(None,
        None)`` is returned.

    Returns
    -------
    (Optional[str], Optional[str])
        Tuple ``(conn, direction)`` where ``conn`` is one of
        ``'functional'``, ``'structural'``, ``'sfc'`` or ``None`` if not
        inferrable, and ``direction`` is one of ``'internal'`` or
        ``'external'`` or ``None``.
    """
    if not path:
        return None, None
    base_name = os.path.basename(path).lower()
    conn_dir = os.path.basename(os.path.dirname(path)).lower()

    conn = None
    if conn_dir.startswith("functional") or "functional" in base_name:
        conn = "functional"
    elif conn_dir.startswith("structural") or "structural" in base_name:
        conn = "structural"
    elif conn_dir.startswith("sfc") or "sfc" in base_name:
        conn = "sfc"

    direction = None
    if base_name.endswith("_internal.svg") or "internal" in base_name:
        direction = "internal"
    elif base_name.endswith("_external.svg") or "external" in base_name:
        direction = "external"

    return conn, direction

# Safely append a text block to a log file; swallow IO errors to avoid crashing
def _append_to_text_log(log_path: Optional[str], block: str) -> None:
    """Append a block of text to a text log file, creating parent directories.

    This helper writes the provided ``block`` to ``log_path`` using UTF-8
    encoding and ensures the final newline is present. If ``log_path`` is
    ``None``, no action is taken. All IO exceptions are suppressed because
    logging should not interrupt downstream analyses.

    Parameters
    ----------
    log_path : Optional[str]
        Path to the text log file. If ``None``, the function is a no-op.
    block : str
        Text to append to the log file. A trailing newline will be added if
        not already present.
    """
    if not log_path:
        return
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(block)
            if not block.endswith("\n"):
                f.write("\n")
    except Exception:
        # Logging should never break the main analysis
        pass

# Build an R formula block string that encodes ICD covariates as factors
def build_icd_factor_block(icd_covariates: Sequence[str], data_frame_name: str = "data") -> str:
    """
    Build an R snippet that factors ICD-10 covariate columns if they exist in the R dataframe.

    Parameters
    ----------
    icd_covariates : Sequence[str]
        Iterable of ICD-10 covariate column names (e.g. ["I10", "Z864"]).
    data_frame_name : str, optional
        Name of the R data.frame variable (default: "data").

    Returns
    -------
    str
        R code block (multi-line string) like:
        if ("I10" %in% colnames(data)) data[["I10"]] <- factor(data[["I10"]])
        if ("Z864" %in% colnames(data)) data[["Z864"]] <- factor(data[["Z864"]])
    """
    lines = []
    for cov in icd_covariates or []:
        # ensure cov is a simple string and escape embedded quotes
        cov_str = str(cov)
        cov_safe = cov_str.replace('"', '\\"').replace("'", "\\'")
        # use data[["colname"]] form to avoid parsing issues with $ and non-syntactic names
        lines.append(f'if ("{cov_safe}" %in% colnames({data_frame_name})) {data_frame_name}[["{cov_safe}"]] <- factor({data_frame_name}[["{cov_safe}"]])')
    return "\n".join(lines)

# Apply multiple-testing correction (e.g., FDR) and write a brief summary to log
def apply_multiple_testing_correction(
    p_values: List[float],
    variable_names: List[str],
    test_methods: List[str],
    method: Literal['fdr_bh', 'bonferroni'] = 'fdr_bh',
    alpha: float = 0.05,
    log_path: Optional[str] = None,
    log_context: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction to p-values.
    
    Parameters
    ----------
    p_values : List[float]
        List of uncorrected p-values
    variable_names : List[str]
        List of variable names
    test_methods : List[str]
        List of test methods used
    method : {'fdr_bh', 'bonferroni'}, optional
        Correction method:
        - 'fdr_bh': False Discovery Rate (Benjamini-Hochberg) - recommended
        - 'bonferroni': Bonferroni correction - more conservative
        (default: 'fdr_bh')
    alpha : float, optional
        Family-wise error rate or false discovery rate threshold (default: 0.05)
    log_path : str or None, optional
        Path to log file for writing the correction summary table. If None, no log is written.
    log_context : str, optional
        Additional context to include in the log summary (e.g., "Model 1: Depression vs Control").
    
    Returns
    -------
    reject : np.ndarray
        Boolean array indicating which hypotheses are rejected
    pvals_corrected : np.ndarray
        Array of corrected p-values
    
    Examples
    --------
    >>> reject, corrected = apply_multiple_testing_correction(
    ...     p_values=[0.01, 0.05, 0.10],
    ...     variable_names=['var1', 'var2', 'var3'],
    ...     test_methods=['Quantile Regression', 'Quantile Regression', 'Quantile Regression']
    ... )
    
        Notes
        -----
        - FDR controls the expected proportion of false positives among
            rejections.
        - Bonferroni controls the family-wise error rate (FWER).
        - The input lists ``p_values``, ``variable_names`` and ``test_methods``
            must have the same length; otherwise results will be misaligned.
            This function does not currently coerce or validate differing lengths
            beyond relying on the supplied list iteration.
    """
    if method == 'fdr_bh':
        reject, pvals_corrected = multitest.fdrcorrection(p_values, alpha=alpha)
        correction_name = 'FDR (Benjamini-Hochberg)'
    elif method == 'bonferroni':
        reject, pvals_corrected, _, _ = multitest.multipletests(
            p_values, alpha=alpha, method='bonferroni'
        )
        correction_name = 'Bonferroni'
    else:
        raise ValueError(f"Unknown method: {method}")

    lines: List[str] = []
    lines.append("\n" + "=" * 80)
    lines.append(f"MULTIPLE TESTING CORRECTION: {correction_name}")
    if log_context:
        lines.append(f"Context: {log_context}")
    lines.append("=" * 80)
    lines.append(f"\n{'Variable':<40} {'Test':<20} {'p (raw)':<12} {'p (adj)':<12} {'Sig.':<5}")
    lines.append("-" * 90)

    for i, var in enumerate(variable_names):
        sig_marker = "***" if pvals_corrected[i] < 0.001 else \
                     "**" if pvals_corrected[i] < 0.01 else \
                     "*" if pvals_corrected[i] < 0.05 else "n.s."

        lines.append(
            f"{var:<40} {test_methods[i]:<20} {p_values[i]:<12.6f} "
            f"{pvals_corrected[i]:<12.6f} {sig_marker:<5}"
        )

    lines.append("-" * 90)
    lines.append(f"Significant results: {reject.sum()} / {len(p_values)}")
    lines.append("=" * 80)

    _append_to_text_log(log_path, "\n".join(lines))

    return reject, pvals_corrected

# Format a p-value as a human-friendly string (e.g., '<0.001' or '0.012')
def format_pvalue(p: float) -> str:
    """
    Format p-value for display with significance stars.
    
    Parameters
    ----------
    p : float
        P-value to format
    
    Returns
    -------
    str
        Formatted p-value string with significance annotation
    
    Examples
    --------
    >>> format_pvalue(0.0001)
    'p < 0.001***'
    >>> format_pvalue(0.045)
    'p = 0.045*'
    >>> format_pvalue(0.12)
    'p = 0.120 n.s.'
    """
    if p < 0.001:
        return "p < 0.001***"
    elif p < 0.01:
        return f"p = {p:.3f}**"
    elif p < 0.05:
        return f"p = {p:.3f}*"
    else:
        return f"p = {p:.3f} n.s."

# ==============================================================================
# R PACKAGE MANAGEMENT
# ==============================================================================
# Ensure an R package is installed (used by the rpy2 setup helpers)
def install_r_package_if_missing(package_name):
    """Install an R package if it is not already installed.

    This helper executes a small R snippet via `rpy2` that checks for the
    package and runs `install.packages()` if necessary. It may require
    network access and appropriate permissions in the R environment.

    Parameters
    ----------
    package_name : str
        Name of the R package to install (e.g., 'quantreg', 'multcomp').

    Returns
    -------
    None

    Notes
    -----
    If R or `rpy2` is not available/configured correctly this function
    will raise exceptions from the underlying R runtime. The caller
    should handle these exceptions if a graceful fallback is desired.
    """
    # Execute a small R snippet via rpy2 that checks for package
    # availability and installs it if missing. This requires network
    # access and write permissions for R's library path.
    ro.r(f'''
        if (!require("{package_name}", quietly = TRUE)) {{
            install.packages("{package_name}", repos = "https://cloud.r-project.org")
        }}
    ''')

# Configure rpy2 / R environment and redirect R console output into a log
def setup_r_environment():
    """Initialize the R environment and import commonly used R packages.

    This function ensures `rpy2` pandas conversion is active within the
    function context, installs missing R packages (`quantreg`)
    if necessary, and returns imported R package objects for use by other
    functions (notably `run_quantile_regression`).

    Returns
    -------
    dict
        Mapping with keys: 'base', 'utils', 'stats', 'quantreg'.

    Raises
    ------
    Exception
        Exceptions from `importr` or the R runtime are propagated to the
        caller (e.g., if R is not installed).
    """
    # Ensure rpy2 conversion rules (especially pandas<->R data.frame)
    # are available in the current context/thread by using a localconverter.
    # This avoids ContextVar-related NotImplementedError when importr() is
    # called from notebook kernels or non-main threads.
    with localconverter(default_converter + pandas2ri.converter):
        base = importr('base')
        utils = importr('utils')
        stats = importr('stats')

        install_r_package_if_missing('quantreg')

        quantreg = importr('quantreg')

    return {
        'base': base,
        'utils': utils,
        'stats': stats,
        'quantreg': quantreg,
    }

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
# Load the module-level cohort CSV and normalize column names used across the pipeline
def load_and_rename_cohort_data(
    file_path: str,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Load cohort data and rename columns to human-readable names.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing cohort data
    column_mapping : Dict[str, str], optional
        Dictionary mapping original column names to new names
        If None, uses default UK Biobank cognitive variable mapping
    
    Returns
    -------
    pd.DataFrame
        Loaded and renamed DataFrame
    
    Examples
    --------
    >>> data = load_and_rename_cohort_data('data/UKB/cohorts/combined_cohort.csv')
    >>> print(data.columns)
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Default UK Biobank cognitive variable mapping
    if column_mapping is None:
        column_mapping = {
            'p21003_i2': 'age_at_assessment',
            'p31': 'sex',
            'p20023_i2': 'Snap_task_mean_reaction_time',
            'p4282_i2': 'Reverse_number_recall_task_span',
            'p6348_i2': 'Trail_making_A_duration',
            'p6350_i2': 'Trail_making_B_duration',
            'p23324_i2': 'Symbol_digit_substitution_task_correct',
            'p21004_i2': 'Tower_rearranging_task_correct',
            'p20197_i2': 'Paired_associates_learning_task_correct',
            'p20018_i2': 'Prospective_memory_task_score',
            'p399_i2_a1': 'Pairs_matching_task_errors_3_pairs',
            'p399_i2_a2': 'Pairs_matching_task_errors_6_pairs',
            'p20016_i2': 'Fluid_intelligence_score',
            'p6373_i2': 'Matrix_pattern_completion_correct',
            'p26302_i2': 'Vocabulary_score'
        }
    
    # Rename columns
    data = data.rename(columns=column_mapping)
    
    return data

# Compute robust (median/MAD) z-scores referenced to controls; logs progress
def calculate_robust_z_scores(
    data: pd.DataFrame,
    vars: List[str],
    group_column: str = 'depression_status',
    control_value: int = 0,
    depression_value: int = 1,
    log_path: Optional[str] = None,
    log_context: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate robust z-scores for cognitive variables in the depression cohort.

    Z-scores are referenced to the control cohort distribution using the
    median and median absolute deviation (MAD):

        z = 0.6745 * (X_depression - μ_control) / MAD_control

    where:

    - ``X_depression``: individual score in the depression group
    - ``μ_control``: median of the control group
    - ``MAD_control``: median absolute deviation of the control group
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing both control and depression groups.
    vars : List[str]
        List of cognitive variable names to calculate z-scores for.
    group_column : str, optional
        Column name for grouping (default: 'depression_status')
    control_value : int, optional
        Value representing control group (default: 0)
    depression_value : int, optional
        Value representing depression group (default: 1)
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - Depression cohort rows only
        - Z-score columns (suffixed with ``'_z'``)
        - Reference statistics (control median and MAD) printed to console
    
    Examples
    --------
    >>> z_scored_data = calculate_robust_z_scores(
    ...     data,
    ...     vars=['Fluid_intelligence_score', 'Vocabulary_score']
    ... )
    >>> print(z_scored_data[['Fluid_intelligence_score', 'Fluid_intelligence_score_z']].head())
    
    Notes
    -----
    - Negative z-scores indicate performance below the control median.
    - Positive z-scores indicate performance above the control median.
    - Missing values are preserved (NaN → NaN in the corresponding z-score).
    - If the control group's MAD for a variable is zero, the division will
      produce ``inf`` or ``NaN`` values in the corresponding z-score column;
      callers should inspect or clean such values if desired.

    Interpretation Guide:
        z = -2.0: Performance 2 MAD below control median (2.3rd percentile)
        z = -1.0: Performance 1 MAD below control median (15.9th percentile)
        z =  0.0: Performance at control median (50th percentile)
        z = +1.0: Performance 1 MAD above control median (84.1st percentile)
        z = +2.0: Performance 2 MAD above control median (97.7th percentile)
    """
    # Separate control and depression cohorts
    control_data = data[data[group_column] == control_value].copy()
    depression_data = data[data[group_column] == depression_value].copy()

    lines: List[str] = []
    lines.append("\n" + "=" * 80)
    lines.append("Z-SCORE CALCULATION")
    if log_context:
        lines.append(f"Context: {log_context}")
    lines.append("=" * 80)
    lines.append(f"\nReference group: Control (n = {len(control_data)})")
    lines.append(f"Target group: Depression (n = {len(depression_data)})")
    lines.append("\nControl Group Statistics (Reference):")
    lines.append("-" * 80)
    lines.append(f"{'Variable':<45} {'Median':<12} {'MAD':<12}")
    lines.append("-" * 80)

    for var in vars:
        if var not in data.columns:
            lines.append(f"Warning: Variable '{var}' not found in DataFrame")
            continue
        
        # Calculate control median and MAD
        control_median = control_data[var].median()
        control_mad = np.median(np.abs(control_data[var] - control_median))
        # Guard against zero MAD which would produce infinite/NaN z-scores.
        # Replace exact-zero or non-finite MAD with a tiny epsilon so that
        # z-scores remain numerically stable while still indicating near-
        # zero dispersion in the control reference distribution.
        if not np.isfinite(control_mad) or control_mad == 0:
            control_mad = 1e-6
        
        lines.append(f"{var:<45} {control_median:<12.3f} {control_mad:<12.3f}")
        
        # Calculate z-scores for depression cohort
        depression_data[f'{var}_z'] = (
            0.6745 * (depression_data[var] - control_median) / control_mad
        )
    
    lines.append("-" * 80)

    # Summary statistics for z-scores
    lines.append("\nDepression Group Z-Score Summary (median-based):")
    lines.append("-" * 80)
    lines.append(f"{'Variable':<45} {'Median Z':<12} {'IQR Z':<12} {'Min Z':<12} {'Max Z':<12}")
    lines.append("-" * 80)

    for var in vars:
        if var not in data.columns:
            continue
        
        z_col = f'{var}_z'
        z_median = depression_data[z_col].median()
        q25 = depression_data[z_col].quantile(0.25)
        q75 = depression_data[z_col].quantile(0.75)
        z_iqr = q75 - q25
        z_min = depression_data[z_col].min()
        z_max = depression_data[z_col].max()

        lines.append(
            f"{var:<45} {z_median:<12.3f} {z_iqr:<12.3f} {z_min:<12.3f} {z_max:<12.3f}"
        )

    lines.append("-" * 80)
    lines.append("=" * 80)

    _append_to_text_log(log_path, "\n".join(lines))

    return depression_data


# Aggregate task-level z-scores into a domain composite (median/mean)
def calculate_composite_z_score(
    z_scored_data: pd.DataFrame,
    z_vars: List[str],
    output_column: str = 'composite_cognitive_z',
    method: Literal['mean', 'median'] = 'median',
    log_path: Optional[str] = None,
    log_context: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate a composite cognitive z-score by aggregating z-scores.
    
    Parameters
    ----------
    z_scored_data : pd.DataFrame
        DataFrame with individual z-score columns (e.g., from
        :func:`calculate_robust_z_scores`).
    z_vars : List[str]
        List of z-score column names to average (e.g., ['Reasoning_z', 'Memory_z'])
    output_column : str, optional
        Name for the composite z-score column (default:
        ``'composite_cognitive_z'``).
    method : {'mean', 'median'}, optional
        Aggregation method (default: 'median')
        - 'mean': Average z-scores (sensitive to outliers)
        - 'median': Median z-scores (robust to outliers)
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added composite z-score column
    
    Examples
    --------
    >>> # First calculate individual z-scores
    >>> z_data = calculate_robust_z_scores(data, vars=['Fluid_intelligence_score'])
    >>> # Then calculate composite
    >>> z_data = calculate_composite_z_score(
    ...     z_data,
    ...     z_vars=['Fluid_intelligence_score_z', 'Vocabulary_score_z']
    ... )
    >>> print(z_data['composite_cognitive_z'].describe())

    Notes
    -----
    - Composite z-scores provide a global cognitive performance index.
    - The mean is familiar but sensitive to extreme values.
    - The median is more robust for skewed or heavy-tailed distributions.
    - Missing values are handled by ignoring them (``skipna=True``).
    """
    if method == 'mean':
        z_scored_data[output_column] = z_scored_data[z_vars].mean(axis=1)
    elif method == 'median':
        z_scored_data[output_column] = z_scored_data[z_vars].median(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean' or 'median'.")

    lines: List[str] = []
    lines.append("\n" + "=" * 80)
    lines.append(f"COMPOSITE Z-SCORE ({method.upper()})")
    if log_context:
        lines.append(f"Context: {log_context}")
    lines.append("=" * 80)
    lines.append(f"Combined {len(z_vars)} domains:")
    for var in z_vars:
        lines.append(f"  - {var}")

    lines.append("\nComposite z-score statistics (median-focused):")
    median_val = z_scored_data[output_column].median()
    q25 = z_scored_data[output_column].quantile(0.25)
    q75 = z_scored_data[output_column].quantile(0.75)
    iqr = q75 - q25
    lines.append(f"  Median: {median_val:.3f}")
    lines.append(f"  IQR: {iqr:.3f} (Q1={q25:.3f}, Q3={q75:.3f})")
    lines.append(
        f"  Range: [{z_scored_data[output_column].min():.3f}, "
        f"{z_scored_data[output_column].max():.3f}]"
    )
    lines.append("=" * 80)

    _append_to_text_log(log_path, "\n".join(lines))

    return z_scored_data

# ==============================================================================
# QUANTILE REGRESSION
# ==============================================================================
# Wrapper that runs quantile regression in R (quantreg::rq) and returns p-values/effects
def quantile_regression(
    tmp_csv_path: str = "/tmp/combined_data.csv",
    dependent_variables: Sequence[str] = ("Connectivity",),
    covariates: Sequence[str] = ("age_at_assessment", "sex"),
    group_column: str = "Group",
    reference_group: str = "Control",
    comparison_groups: Optional[Sequence[str]] = None,
    tau: float = 0.5,
    R: int = 10000,
    test_against_zero: bool = False,
    return_effects: bool = False,
    r_output_log_path: Optional[str] = None,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, List[float]]]]:
    """Run quantile regression in R for one or more dependent variables.

    This function expects a CSV at ``tmp_csv_path`` with columns at least:

    - One or more numeric dependent variables (given in ``dependent_variables``)
    - Either:
        - ``group_column`` (factor-like): values including "Depression" and
          "Control" for between-group comparisons, or
        - No group column, for one-sample tests against zero (e.g. median-based
          z-scores referenced to the control cohort).
    - Optionally, covariates (numeric or categorical) specified in
      ``covariates`` (used in both modes when provided).

    Two modes are supported:

    1. Group contrast (default, ``test_against_zero=False``)
         For each dependent variable, fits a separate quantile regression model
         of the form::

                 DV ~ group_column + covariates

         at quantile ``tau``, using bootstrapped standard errors and percentile
         confidence intervals via ``boot.rq``.

         By default this behaves like the original implementation: it compares
         "Depression" against "Control" with "Control" as the reference.
         You can instead compare arbitrary groups by specifying
         ``reference_group`` and ``comparison_groups`` (e.g. Cluster 0/1 vs
         Depression).

    2. One-sample test against zero (``test_against_zero=True``)
         Intended for already z-scored (median-based) dependent variables that
         are referenced to the control cohort (control median = 0). For each
         dependent variable, it fits a quantile regression model::

                 DV ~ covariates    # or DV ~ 1 if no covariates are given

         and tests whether the estimated median (intercept) differs from 0.
         It returns p-values for the intercept term per dependent variable.

    Parameters
    ----------
    tmp_csv_path : str, optional
        Path to the temporary CSV used as input to R (default:
        "/tmp/combined_data.csv").
    dependent_variables : Sequence[str], optional
        Names of the dependent variable columns (default: ("Connectivity",)).
    covariates : Sequence[str], optional
        Covariate column names to include in every model (default:
        ("age_at_assessment", "sex")).
    group_column : str, optional
        Name of the group column (default: "Group"). Only used in
        group-contrast mode.
    reference_group : str, optional
        Reference level used in group-contrast mode (default: "Control").
        The returned coefficients/p-values correspond to each comparison group
        relative to this reference.
    comparison_groups : Optional[Sequence[str]], optional
        Which group levels to compare against ``reference_group``. If None,
        defaults to ("Depression",) when ``reference_group == "Control"``;
        otherwise uses all non-reference levels present in the data.
        For cluster-vs-cluster comparisons, simply set ``group_column`` to your
        cluster label column (e.g. "Cluster") and set ``reference_group`` /
        ``comparison_groups`` accordingly.
    tau : float, optional
        Quantile to estimate (default: 0.5 for the median).
    R : int, optional
        Number of bootstrap replications used both in ``summary.rq``
        (for bootstrap standard errors) and in the explicit ``boot.rq``
        call (for percentile confidence intervals). Defaults to 10000.
    test_against_zero : bool, optional
        If ``False`` (default), perform a between-group comparison
        (Depression vs Control) via ``group_column``. If ``True``, ignore
        any group information and perform a one-sample test of whether the
        median of each dependent variable differs from zero (suitable for
        robust z-scored variables).
    return_effects : bool, optional
        If ``True``, additionally return effect sizes, standard errors, and
        subject-level predicted values for each dependent variable.

        Returns
        -------
        Dict[str, float]
                If ``return_effects`` is False (default), returns a mapping from a
                canonical key to the p-value. In group-contrast mode the keys are of
                the form ``"<DV>::<GroupLevel>"`` (for example,
                ``"Fluid_intelligence_score::Depression"``); in one-sample mode the
                keys are the dependent variable names and map to the intercept p-
                values.

        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
                If ``return_effects`` is True, the function additionally returns
                ``effects`` and ``se_effects`` mappings with the same key scheme
                described above. The function may in future also return per-subject
                predicted values, but that is not guaranteed in all code paths.

        Side effects
        -----------
        - Writes detailed R console output to ``r_output_log_path`` (created if
            necessary) using R's ``sink`` mechanism.
        - During execution the R objects ``quantreg_p_values``, ``quantreg_"
            "effects`` and ``quantreg_se`` are created in the R global
            environment and then read back into Python. These names are internal
            implementation details and callers should not rely on their presence.

        Notes
        -----
        - The function relies on an available R runtime and the R package
            ``quantreg`` (installed via :func:`install_r_package_if_missing` if
            absent). Errors from the R runtime will propagate as exceptions.
        - Large values of ``R`` (bootstrap replications) can be time-consuming; a
            smaller value is recommended for interactive testing.
    """

    # Normalize inputs
    if isinstance(dependent_variables, str):  # allow single string for convenience
        dependent_vars = [dependent_variables]
    else:
        dependent_vars = list(dependent_variables)

    covars = list(covariates)
    if not test_against_zero and group_column is None:
        raise ValueError("group_column must be provided when test_against_zero is False")

    # RHS for one-sample models: only covariates; if none, use intercept-only model
    one_sample_rhs = " + ".join(covars) if covars else "1"

    if r_output_log_path is None:
        r_output_log_path = "/tmp/quantile_regression_R_output.txt"
    try:
        log_dir = os.path.dirname(r_output_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # If we cannot create directories, R will still attempt to write and
        # will error inside its tryCatch blocks; keep going.
        pass

    dep_vars_r = "c(" + ", ".join(f'"{dv}"' for dv in dependent_vars) + ")"
    covar_names_r = (
        "c(" + ", ".join(f'"{c}"' for c in covars) + ")" if covars else "c()"
    )

    # Factor sex by default if present among covariates
    factor_sex_block = "data$sex <- factor(data$sex)" if "sex" in covars else ""

    # Factor ICD_10 codes if present among covariates
    icd_covars = [c for c in covars if c[0].isupper() and c[1:].isdigit()]
    factor_ICD_block = build_icd_factor_block(icd_covariates=icd_covars)

    # Center age covariate inside R if requested via covariates.
    # This keeps the public Python API simple (users pass 'age_at_assessment')
    # while ensuring the regression intercept corresponds to a typical age.
    age_center_block = (
        'if ("age_at_assessment" %in% colnames(data)) {\n'
        '  data$age_at_assessment <- data$age_at_assessment - '\
        'median(data$age_at_assessment, na.rm = TRUE)\n'
        '}'
        if "age_at_assessment" in covars
        else ""
    )

    # Prepare R code for group-contrast vs one-sample modes
    comparison_groups_r = (
        "NULL"
        if comparison_groups is None
        else "c(" + ", ".join(f'\"{g}\"' for g in comparison_groups) + ")"
    )

    # rpy2 uses a contextvar to hold conversion rules.
    # In some notebook environments the context may be missing, causing
    # NotImplementedError at runtime. Always establish a local conversion
    # context for the duration of the R interaction.
    with localconverter(default_converter):
        if not test_against_zero:
            # Between-cluster comparisons
            # The following rpy2 call executes a multi-line R script that
            # reads the temporary CSV, prepares the group factor, fits
            # quantile regression models, performs bootstrap CIs, computes robust R2, and
            # assigns named result vectors into R's global environment so
            # Python can extract p-values and effects after the call.
            ro.r(
                f'''
            # ---- R console logging (stdout + messages) ----
            log_path <- "{r_output_log_path}"
            tryCatch({{
                sink(file = log_path, append = TRUE)
                sink(file = log_path, append = TRUE, type = "message")
            }}, error = function(e) {{
                cat(paste0("[sink error] ", conditionMessage(e), "\n"),
                    file = log_path, append = TRUE)
            }})
            on.exit({{
                tryCatch(sink(type = "message"), error = function(e) {{}})
                tryCatch(sink(), error = function(e) {{}})
                cat("===== /quantile_regression =====\n")
            }}, add = TRUE)

            cat("\n===== quantile_regression (group-contrast) =====\n")
            cat(paste0("tmp_csv_path: {tmp_csv_path}\n"))
            cat(paste0("tau: {tau}, R: {R}\n"))

            library(quantreg)
            # Load and prepare data
            data <- read.csv("{tmp_csv_path}")

            # If available, log connectivity type and cluster context
            if ("Connectivity_Type" %in% colnames(data)) {{
                conn_types <- unique(as.character(data$Connectivity_Type))
                cat("Connectivity_Type in data: ",
                    paste(conn_types, collapse = ", "), "\n")
            }}
            if ("Cluster" %in% colnames(data)) {{
                clusters <- unique(as.character(data$Cluster))
                cat("Cluster levels in data: ",
                    paste(clusters, collapse = ", "), "\n")
            }}

            # Handle numeric or coded group values: 1 = Depression, 0 = Control
            # Only apply this mapping for depression-status-like columns.
            if ( "{group_column}" %in% colnames(data)) {{
                if (is.numeric(data${group_column})) {{
                    uvals <- unique(na.omit(data${group_column}))
                    is_dep_status_col <- ("{group_column}" == "Group") || grepl("depress|status|case", "{group_column}", ignore.case = TRUE)
                    if (is_dep_status_col && all(uvals %in% c(0, 1))) {{
                        data${group_column} <- ifelse(data${group_column} == 1, "Depression",
                                           ifelse(data${group_column} == 0, "Control", NA))
                    }}
                }}
                data${group_column} <- factor(data${group_column})
            }} else {{
                stop("Group column not found in data but test_against_zero is False; " ,
                     "either provide a Group column or set test_against_zero=True.")
            }}

            # Use the provided group column directly for modeling.
            model_group_col <- "{group_column}"
            {factor_sex_block}
            {factor_ICD_block}
            {age_center_block}

            dep_vars <- {dep_vars_r}
            covar_names <- {covar_names_r}
            p_values <- c()
            effects <- c()
            se_values <- c()

            # Restrict to requested groups and set reference
            reference_group <- "{reference_group}"
            comparison_groups <- {comparison_groups_r}
            group_levels <- levels(data[[model_group_col]])
            if (is.null(comparison_groups)) {{
                comparison_groups <- setdiff(group_levels, reference_group)
            }}
            keep_groups <- unique(c(reference_group, comparison_groups))
            missing_groups <- setdiff(keep_groups, group_levels)
            if (length(missing_groups) > 0) {{
                stop(paste0(
                    "Requested group levels not present in data: ",
                    paste(missing_groups, collapse = ", "),
                    ". Available levels: ",
                    paste(group_levels, collapse = ", ")
                ))
            }}

            data_sub <- data[data[[model_group_col]] %in% keep_groups, ]
            data_sub[[model_group_col]] <- droplevels(data_sub[[model_group_col]])
            data_sub[[model_group_col]] <- relevel(data_sub[[model_group_col]], ref = reference_group)

            for (g in keep_groups) {{
                print(paste("N", g, ":", sum(data_sub[[model_group_col]] == g)))
            }}
            # Function to compute a robust R^2 based on MAD, since traditional R^2 is not meaningful for quantile regression residuals
            robust_R2 <- function(y, yhat) {{

                y <- as.numeric(y)
                yhat <- as.numeric(yhat)

                stopifnot(length(y) == length(yhat))

                ok <- is.finite(y) & is.finite(yhat)
                y <- y[ok]
                yhat <- yhat[ok]

                if (length(y) < 2) return(NA_real_)

                s_res <- mad(y - yhat, center = 0, constant = 1)
                s_tot <- mad(y, center = median(y), constant = 1)

                if (!is.finite(s_tot) || s_tot == 0) return(NA_real_)

                R2 <- 1 - (s_res / s_tot)^2

                # numerical safety clamp
                R2 <- max(min(R2, 1), -Inf)
                return(R2)
            }}  
            for (dv in dep_vars) {{
                cat("\n==================================================\n")
                cat("QUANTILE REGRESSION FOR DEPENDENT VARIABLE:", dv, "\n")
                cat("==================================================\n")

                # Build formula DV ~ group + covariates
                rhs_terms <- c(model_group_col, covar_names)
                rhs_terms <- rhs_terms[!is.na(rhs_terms) & nzchar(rhs_terms)]
                formula_str <- paste(dv, "~", paste(rhs_terms, collapse = " + "))
                cat("Formula:", formula_str, "\n")
                fml <- as.formula(formula_str)

                X <- model.matrix(fml, data = data_sub)
                y <- data_sub[[dv]]

                model <- rq(fml, data = data_sub, tau = {tau})
                R2_MAD <- robust_R2(y, fitted(model))
                cat(sprintf("R2_MAD (median-based): %.2f\n", R2_MAD))
                model_summary <- summary.rq(model, se = "boot", R = {R})
                print(model_summary)

                set.seed(123)
                boot_out <- boot.rq(x = X, y = y, tau = {tau}, R = {R})
                if (is.null(colnames(boot_out$B))) {{
                    colnames(boot_out$B) <- colnames(X)
                }}
                cis <- apply(boot_out$B, 2, quantile, probs = c(0.025, 0.975))
                print("95% bootstrap CIs for all coefficients:")
                print(cis)

                coef_table <- model_summary$coefficients

                for (cg in comparison_groups) {{
                    term_name <- paste0(model_group_col, cg)
                    if (!(term_name %in% rownames(coef_table))) {{
                        cat("Warning: term '", term_name, "' not found in model for ", dv, "\n", sep = "")
                        next
                    }}

                    p_raw <- coef_table[term_name, "Pr(>|t|)"]
                    p_clamped <- ifelse(p_raw < 2.2e-16, 2.2e-16, p_raw)
                    coef_group <- coef_table[term_name, "Value"]
                    se_group <- coef_table[term_name, "Std. Error"]
                    p_str <- ifelse(p_clamped <= 2.2e-16, "< 2.2e-16", sprintf("%.2e", p_clamped))

                    key <- paste(dv, cg, sep = "::")
                    p_values[key] <- p_clamped
                    effects[key] <- coef_group
                    se_values[key] <- se_group
                }}
            }}

            assign("quantreg_p_values", p_values, envir = .GlobalEnv)
            assign("quantreg_effects", effects, envir = .GlobalEnv)
            assign("quantreg_se", se_values, envir = .GlobalEnv)
            '''
            )
        else:
            # One-sample test against zero for (modified) z-scored variables
            ro.r(
                f'''
            # ---- R console logging (stdout + messages) ----
            log_path <- "{r_output_log_path}"
            tryCatch({{
                sink(file = log_path, append = TRUE)
                sink(file = log_path, append = TRUE, type = "message")
            }}, error = function(e) {{
                cat(paste0("[sink error] ", conditionMessage(e), "\n"),
                    file = log_path, append = TRUE)
            }})
            on.exit({{
                tryCatch(sink(type = "message"), error = function(e) {{}})
                tryCatch(sink(), error = function(e) {{}})
                cat("===== /quantile_regression =====\n")
            }}, add = TRUE)

            cat("\n===== quantile_regression (one-sample vs 0) =====\n")
            cat(paste0("tmp_csv_path: {tmp_csv_path}\n"))
            cat(paste0("tau: {tau}, R: {R}\n"))

            library(quantreg)

            # Load data (z-scored DVs)
            data <- read.csv("{tmp_csv_path}")

            # If available, log connectivity type and cluster context
            if ("Connectivity_Type" %in% colnames(data)) {{
                conn_types <- unique(as.character(data$Connectivity_Type))
                cat("Connectivity_Type in data: ",
                    paste(conn_types, collapse = ", "), "\n")
            }}
            if ("Cluster" %in% colnames(data)) {{
                clusters <- unique(as.character(data$Cluster))
                cat("Cluster levels in data: ",
                    paste(clusters, collapse = ", "), "\n")
            }}

            {factor_sex_block}
            {factor_ICD_block}
            {age_center_block}

            dep_vars <- {dep_vars_r}
            covar_names <- {covar_names_r}
            p_values <- c()
            effects <- c()
            se_values <- c()

            for (dv in dep_vars) {{
                cat("\n==================================================\n")
                cat("ONE-SAMPLE QUANTILE REGRESSION FOR DEPENDENT VARIABLE:", dv, "\n")
                cat("(testing median vs 0 for z-scored variable)\n")
                cat("==================================================\n")
                
                # Model: DV ~ covariates (or ~ 1 if none); test intercept vs 0
                formula_str <- paste(dv, "~", "{one_sample_rhs}")
                cat("Formula:", formula_str, "\n")
                fml <- as.formula(formula_str)

                X <- model.matrix(fml, data = data)
                y <- data[[dv]]

                model <- rq(fml, data = data, tau = {tau})
                model_summary <- summary.rq(model, se = "boot", R = {R})
                print(model_summary)

                set.seed(123)
                boot_out <- boot.rq(x = X, y = y, tau = {tau}, R = {R})
                cis <- apply(boot_out$B, 2, quantile, probs = c(0.025, 0.975))
                print("95% bootstrap CIs for all coefficients:")
                print(cis)

                coef_table <- model_summary$coefficients
                if (!("(Intercept)" %in% rownames(coef_table))) {{
                    cat("Warning: term '(Intercept)' not found in model for", dv, "\n")
                }} else {{
                    p_raw <- coef_table["(Intercept)", "Pr(>|t|)"]
                    p_clamped <- ifelse(p_raw < 2.2e-16, 2.2e-16, p_raw)

                    coef_int <- coef_table["(Intercept)", "Value"]
                    se_int <- coef_table["(Intercept)", "Std. Error"]
                    p_str <- ifelse(p_clamped <= 2.2e-16, "< 2.2e-16", sprintf("%.2e", p_clamped))

                    p_values[dv] <- p_clamped
                    effects[dv] <- coef_int
                    se_values[dv] <- se_int
                }}
            }}

            assign("quantreg_p_values", p_values, envir = .GlobalEnv)
            assign("quantreg_effects", effects, envir = .GlobalEnv)
            assign("quantreg_se", se_values, envir = .GlobalEnv)
            '''
            )

    # Extract p-values back into Python as a named vector
    with localconverter(default_converter):
        r_p_values = ro.r("quantreg_p_values")
        r_names = list(ro.r("names(quantreg_p_values)"))

    p_values: Dict[str, float] = {}
    for i, name in enumerate(r_names):
        p_values[str(name)] = float(r_p_values[i])

    # Optionally extract effect estimates (group coefficient or intercept)
    effects: Dict[str, float] = {}
    se_effects: Dict[str, float] = {}
    try:
        with localconverter(default_converter):
            r_effects = ro.r("quantreg_effects")
            r_effect_names = list(ro.r("names(quantreg_effects)"))
            for i, name in enumerate(r_effect_names):
                effects[str(name)] = float(r_effects[i])
            r_se = ro.r("quantreg_se")
            r_se_names = list(ro.r("names(quantreg_se)") )
            for i, name in enumerate(r_se_names):
                se_effects[str(name)] = float(r_se[i])
    except Exception:
        effects = {}
        se_effects = {}

    if return_effects:
        return p_values, effects, se_effects

    return p_values

# ==============================================================================
# VISUALIZATION
# ==============================================================================
# Plot violin distributions per group/cluster for cognitive variables and save SVG
def plot_cognitive_distributions_violin(
    data: pd.DataFrame,
    variables: Sequence[str],
    group_column: str = "depression_status",
    control_value: int = 0,
    depression_value: int = 1,
    control_label: str = "Control",
    depression_label: str = "Depression",
    control_color: str = "#2ca02c",
    depression_color: str = "#6a3d9a",
    plot_depression_only: bool = False,
    plot_depression_clusters: bool = False,
    cluster_column: str = "Cluster",
    cluster_order: Optional[Sequence[str]] = None,
    conn_type: Optional[str] = None,
    dir_type: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 150,
    title: Optional[str] = None,
) -> None:
    """Plot control vs. depression distributions for cognitive variables.

    Creates a multi-panel figure with one subplot per variable. Each subplot
    displays violin(s) of the distribution depending on the plotting mode:

    - Default: two violins per subplot (control on the left, depression on the
      right).
    - ``plot_depression_only=True``: single depression violin per subplot.
    - ``plot_depression_clusters=True``: depression data split by cluster
      labels given in ``cluster_column`` (cluster-specific violins).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all observations and the variables to plot.
    variables : Sequence[str]
        Sequence of cognitive variable column names to visualize. Variables
        not present in ``data`` are skipped and the corresponding subplot
        shows a "Missing" annotation.
    group_column : str, optional
        Column name identifying group membership (default: ``'depression_status'``).
    control_value : int, optional
        Value in ``group_column`` identifying control rows (default: ``0``).
    depression_value : int, optional
        Value in ``group_column`` identifying depression rows (default: ``1``).
    control_label : str, optional
        Label string used for control group in legends and x-axis (default:
        ``'Control'``).
    depression_label : str, optional
        Label string used for depression group (default: ``'Depression'``).
    control_color : str, optional
        Hex color or named matplotlib color for the control violin (default:
        ``'#2ca02c'``).
    depression_color : str, optional
        Hex color or named matplotlib color for the depression violin
        (default: ``'#6a3d9a'``).
    plot_depression_only : bool, optional
        If True, only plot the depression group's violin (ignores control
        values). Useful when control data are not available (e.g., cohort-
        specific z-scores).
    plot_depression_clusters : bool, optional
        If True, plot depression data split by ``cluster_column`` values.
        This mode overrides ``plot_depression_only`` and constructs the plot
        using cluster-specific colors resolved via
        :func:`_cluster_colors_for_modality` when ``conn_type`` is provided.
    cluster_column : str, optional
        Column name containing cluster labels when ``plot_depression_clusters``
        is True. Cluster labels are converted to strings for palette lookup.
    cluster_order : Sequence[str], optional
        Optional order for cluster labels on the x-axis. If None, labels are
        sorted lexicographically.
    conn_type : str, optional
        Optional connectivity type hint passed to
        :func:`_cluster_colors_for_modality` to derive modality-specific
        cluster colors (e.g., ``'functional'``, ``'structural'``, ``'sfc'``).
    dir_type : str, optional
        Optional direction hint (``'internal'`` or ``'external'``) used when
        resolving cluster colors from ``conn_type``.
    save_path : str, optional
        If provided, the figure is saved to this path. Parent directories are
        created as needed.
    figsize : Tuple[int, int], optional
        Figure size in inches (default: ``(16, 10)``). Height is adjusted
        automatically if there are many variables.
    dpi : int, optional
        Resolution in DPI for saved figures (default: ``150``).
    title : str, optional
        Overall figure title. If None, no suptitle is set.

    Returns
    -------
    None

    Notes
    -----
    - Cluster labels are treated as strings when constructing palettes; call
      sites that use integer labels should convert them to strings first or
      rely on the default color when no mapping exists.
    - Missing variables do not raise; the corresponding subplot will display
      "Missing: <var>". This function closes the figure after optionally
      saving it and does not return the matplotlib Figure object.
    """
    if not variables:
        raise ValueError("No cognitive variables provided for violin plotting.")

    n_vars = len(variables)
    n_cols = 3 if n_vars >= 3 else n_vars
    n_rows = int(math.ceil(n_vars / n_cols))

    # Increase height per row to prevent y-axis label overlap
    adjusted_figsize = (figsize[0], figsize[1] * n_rows / 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=adjusted_figsize, dpi=dpi, squeeze=False)
    axes_flat = axes.flatten()

    def _pretty_label(label: str) -> str:
        return str(label).replace("_", " ")

    if plot_depression_clusters:
        control_data = pd.DataFrame()
        depression_data = data.copy()
    else:
        control_data = data[data[group_column] == control_value]
        depression_data = data[data[group_column] == depression_value]

    for idx, var in enumerate(variables):
        ax = axes_flat[idx]
        if var not in data.columns:
            ax.text(0.5, 0.5, f"Missing: {var}", ha="center", va="center")
            ax.set_axis_off()
            continue

        if plot_depression_clusters:
            depression_vals = depression_data[var].dropna().to_numpy()
            if depression_vals.size == 0:
                ax.text(0.5, 0.5, f"No data: {var}", ha="center", va="center")
                ax.set_axis_off()
                continue
        else:
            control_vals = control_data[var].dropna().to_numpy()
            depression_vals = depression_data[var].dropna().to_numpy()
            if control_vals.size == 0 and depression_vals.size == 0:
                ax.text(0.5, 0.5, f"No data: {var}", ha="center", va="center")
                ax.set_axis_off()
                continue

        if plot_depression_clusters:
            if cluster_column not in depression_data.columns:
                ax.text(0.5, 0.5, f"Missing: {cluster_column}", ha="center", va="center")
                ax.set_axis_off()
                continue

            cluster_vals = depression_data[cluster_column].dropna().astype(str)
            if cluster_vals.empty:
                ax.text(0.5, 0.5, f"No clusters: {var}", ha="center", va="center")
                ax.set_axis_off()
                continue

            plot_df = pd.DataFrame({
                "Cluster": cluster_vals,
                var: depression_data.loc[cluster_vals.index, var].to_numpy(),
            })

            if cluster_order is None:
                order = sorted(plot_df["Cluster"].unique())
            else:
                order = [str(v) for v in cluster_order]

            # Determine modality and direction. Priority order for direction:
            # 1) explicit `dir_type` argument,
            # 2) parsed direction from `conn_type` label,
            # 3) infer from a single-valued `Direction` column in the data,
            # 4) default to 'internal'.
            conn, parsed_direction = _parse_modality_from_label(conn_type or "")

            # If conn not found in conn_type string, try to infer from data
            if not conn and "Connectivity_Type" in data.columns:
                unique_conn = pd.Series(data["Connectivity_Type"]).dropna().unique()
                if len(unique_conn) == 1:
                    conn = str(unique_conn[0]).lower()

            # Decide final direction
            if dir_type:
                dir_final = dir_type
            elif parsed_direction:
                dir_final = parsed_direction
            elif "Direction" in data.columns:
                unique_dirs = pd.Series(data["Direction"]).dropna().unique()
                dir_final = str(unique_dirs[0]) if len(unique_dirs) == 1 else "internal"
            else:
                dir_final = "internal"

            if not conn:
                cluster_colors = {"0": depression_color, "1": depression_color}
            else:
                cluster_colors = _cluster_colors_for_modality(conn, dir_final) or {}

            palette = {}
            for label in order:
                key = label.replace("Cluster", "").strip()
                palette[label] = cluster_colors.get(key, depression_color)

        elif plot_depression_only:
            plot_df = pd.DataFrame({
                "Group": [depression_label] * len(depression_vals),
                var: depression_vals,
            })
            order = [depression_label]
            palette = {depression_label: depression_color}
        else:
            plot_df = pd.DataFrame({
                "Group": ([control_label] * len(control_vals)) + ([depression_label] * len(depression_vals)),
                var: np.concatenate([control_vals, depression_vals]) if control_vals.size and depression_vals.size
                else (control_vals if control_vals.size else depression_vals),
            })
            order = [control_label, depression_label]
            palette = {control_label: control_color, depression_label: depression_color}

        sns.violinplot(
            data=plot_df,
            x="Cluster" if plot_depression_clusters else "Group",
            y=var,
            order=order,
            palette=palette,
            ax=ax,
        )

        ax.set_title(_pretty_label(var), fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    for ax in axes_flat[n_vars:]:
        ax.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=14)

    # Increase spacing to prevent y-axis label overlap
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None, h_pad=3.0, w_pad=2.0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# Create summary violin/box/point plots for z-scores (task or domain level)
def plot_z_scores(
    z_scored_data: pd.DataFrame,
    z_vars: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 20),
    association_type: str = 'cognition',
    type_z_score: str = 'task',
    overall_title: str = 'Z-Scores (Depression Cohort)'
) -> None:
    """Visualize z-score profiles as a radar (spider) chart and optionally
    register overlay information for connectivity-specific comparisons.

    This function computes sample medians and bootstrap standard errors for
    the supplied z-score variables, plots a radar polygon (median values
    mapped to radial coordinates) and, when ``save_path`` is provided and the
    filename follows the project's naming conventions, records results in the
    module-level overlay registries used to build combined overall-vs-cluster
    radar figures.

    Parameters
    ----------
    z_scored_data : pd.DataFrame
        DataFrame containing z-score columns (e.g., generated by
        :func:`calculate_robust_z_scores`). Rows correspond to subjects.
    z_vars : List[str]
        List of z-score column names to plot (must contain at least 3
        variables with finite medians).
    save_path : str, optional
        Path where the individual radar SVG will be saved. If provided,
        the function may also register the computed medians and SEs in the
        overlay registry for later assembly of combined plots. If ``None``,
        the plot is displayed in memory and not registered.
    figsize : Tuple[int, int], optional
        Figure size in inches (default: ``(20, 20)``).
    association_type : str, optional
        High-level label describing the association type (default:
        ``'cognition'``). Used when composing overlay registry keys.
    type_z_score : str, optional
        Descriptor for the z-score plot (e.g. ``'task'``, ``'domain'``,
        ``'Cluster_0'``). This string is inspected to determine whether the
        call corresponds to an overall or cluster-specific figure.
    overall_title : str, optional
        Title displayed at the top of the radar plot.

    Returns
    -------
    None

    Side effects
    -----------
    - When ``save_path`` is provided and the filename follows the
      repository's naming conventions (containing cluster and modality
      tokens), the function records medians and bootstrap SEs into module-
      level registries (``_RADAR_OVERLAY_STORE``, ``_RADAR_OVERLAY_SE_OVERALL``,
      ``_RADAR_OVERLAY_SE_CLUSTER``) and may trigger the overlay assembly
      routine ``_render_connectivity_overlay`` if all required pieces are
      available.

    Notes
    -----
    - At least 3 variables with finite medians are required; otherwise the
      function raises ``ValueError``.
    - Bootstrap standard errors are estimated with a fixed RNG seed for
      reproducibility (5000 resamples by default); this can be adjusted in
      the code if desired.
        - The function maps medians to radial coordinates using an asymmetric
            nominal median range ``[-0.70, 0.40]`` (expanded only when data/SE
            exceed these bounds) and visualizes median ± SE bands when available.
    """

    def _pretty_label(v: str) -> str:
        base = v[:-2] if v.endswith("_z") else v
        return base.replace("_", " ")

    # -----------------------------
    # Bootstrap SE for medians (resample with replacement)
    # -----------------------------
    def _bootstrap_median_se(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan
        if values.size == 1:
            return 0.0
        n = values.size
        medians = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = rng.choice(values, size=n, replace=True)
            medians[i] = np.nanmedian(sample)
        return float(np.nanstd(medians, ddof=1)) if n_boot > 1 else 0.0

    rng = np.random.default_rng(123)
    n_boot = 5000
    se_map: Dict[str, float] = {}
    for var in z_vars:
        if var not in z_scored_data.columns:
            se_map[var] = np.nan
            continue
        se_map[var] = _bootstrap_median_se(z_scored_data[var].to_numpy(), n_boot=n_boot, rng=rng)

    # -----------------------------
    # Compute per-variable sample medians only (ignore model-based values)
    # -----------------------------
    med_signed = {}
    for var in z_vars:
        if var not in z_scored_data.columns:
            vals = np.asarray([], dtype=float)
        else:
            vals = np.asarray(z_scored_data[var].dropna(), dtype=float)

        if vals is None or len(vals) == 0:
            med_signed[var] = np.nan
            continue

        # Sample median (from observed data)
        med_signed[var] = float(np.nanmedian(vals))

    # Order variables alphabetically by their pretty labels so that
    # axis positions are fixed and independent of effect size.
    finite_vars = [v for v in z_vars if np.isfinite(med_signed.get(v, np.nan))]
    var_order = sorted(finite_vars, key=lambda v: _pretty_label(v).lower())
    if len(var_order) < 3:
        raise ValueError(
            "Radar chart requires at least 3 variables with finite medians; "
            f"got {len(var_order)}."
        )

    # Signed medians in axis order
    med = np.asarray([med_signed[v] for v in var_order], dtype=float)

    # -----------------------------
    # Build polar coordinates
    # -----------------------------
    n = len(var_order)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    # -----------------------------
    # Determine radial mapping using nominal fixed bounds but allow grow-to-fit
    # -----------------------------
    fixed_min = -0.70
    fixed_max = 0.40
    data_max = float(np.nanmax(np.abs(med))) if med.size > 0 else 0.0
    se_vals_for_range = []
    for var_name in var_order:
        se_val = se_map.get(var_name)
        try:
            se_float = float(se_val)
        except (TypeError, ValueError):
            se_float = 0.0
        if not np.isfinite(se_float) or se_float < 0:
            se_float = 0.0
        se_vals_for_range.append(se_float)
    se_vals_for_range = np.asarray(se_vals_for_range, dtype=float)
    med_plus_se = np.abs(med) + se_vals_for_range
    med_se_max = np.nanmax(med_plus_se) if med_plus_se.size else np.nan
    range_candidate = data_max
    if np.isfinite(med_se_max):
        range_candidate = max(range_candidate, med_se_max)

    # Asymmetric bounds in median space (respect fixed_min/fixed_max by default,
    # but expand if data +/- SE exceed them).
    lower_candidate = np.nanmin(med - se_vals_for_range) if med.size else np.nan
    upper_candidate = np.nanmax(med + se_vals_for_range) if med.size else np.nan
    lower_med = float(min(fixed_min, lower_candidate)) if np.isfinite(lower_candidate) else float(fixed_min)
    upper_med = float(max(fixed_max, upper_candidate)) if np.isfinite(upper_candidate) else float(fixed_max)
    med_span = upper_med - lower_med
    if not np.isfinite(med_span) or med_span <= 0:
        med_span = max(fixed_max - fixed_min, 1.0)
        lower_med = fixed_min
        upper_med = fixed_min + med_span

    def _to_radial(arr_signed: np.ndarray) -> np.ndarray:
        return (np.asarray(arr_signed, dtype=float) - lower_med) / med_span

    def _to_radial_scalar(value: float) -> float:
        return float((float(value) - lower_med) / med_span)

    def _to_median_scalar(radius: float) -> float:
        return float(lower_med + float(radius) * med_span)

    med_closed = _to_radial(np.concatenate([med, med[:1]]))

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Reference ring at zero-median level (smooth circle) - thicker solid line
    r_zero = _to_radial_scalar(0.0)
    theta_dense = np.linspace(0.0, 2 * np.pi, 720)
    ax.plot(theta_dense, np.full_like(theta_dense, r_zero), color="black", lw=3.0, ls="-", alpha=0.95)

    # Choose polygon/marker color based on whether this is the overall cohort
    conn_guess, dir_guess = _infer_modality_from_path(save_path)
    if conn_guess is None or dir_guess is None:
        conn_from_label, dir_from_label = _parse_modality_from_label(type_z_score)
        conn_guess = conn_guess or conn_from_label
        dir_guess = dir_guess or dir_from_label
    cluster_colors = _cluster_colors_for_modality(conn_guess or "functional", dir_guess or "internal")
    if ("Cluster_0" in type_z_score) or ("Cluster 0" in type_z_score):
        profile_color = cluster_colors["0"]
        profile_line_style = "-"
    elif ("Cluster_1" in type_z_score) or ("Cluster 1" in type_z_score):
        profile_color = cluster_colors["1"]
        profile_line_style = "-"
    else:
        profile_color = "black"
        profile_line_style = ":"

    # Outline-only polygon and markers (no fill)
    ax.plot(angles_closed, med_closed, color=profile_color, lw=2.5, ls=profile_line_style, zorder=3)
    ax.scatter(angles, _to_radial(med), marker="o", s=120, facecolor="white", edgecolor=profile_color, linewidth=1.5, zorder=4)

    se_vals = []
    for var_name in var_order:
        se_val = se_map.get(var_name)
        try:
            se_float = float(se_val)
        except (TypeError, ValueError):
            se_float = np.nan
        if not np.isfinite(se_float) or se_float < 0:
            se_float = np.nan
        se_vals.append(se_float)
    se_vals = np.asarray(se_vals, dtype=float)
    if np.isfinite(se_vals).any():
        r_low = _to_radial(med - se_vals)
        r_high = _to_radial(med + se_vals)
        r_low = np.clip(r_low, 0.0, 1.0)
        r_high = np.clip(r_high, 0.0, 1.0)
        r_low = np.minimum(r_low, r_high)
        r_low_closed = np.concatenate([r_low, r_low[:1]])
        r_high_closed = np.concatenate([r_high, r_high[:1]])
        r_low_masked = np.ma.masked_invalid(r_low_closed)
        r_high_masked = np.ma.masked_invalid(r_high_closed)
        ax.fill_between(
            angles_closed,
            r_low_masked,
            r_high_masked,
            color=profile_color,
            alpha=0.30,
            zorder=1,
            linewidth=0,
        )

    # Add a small legend explaining the profile and the zero-median ring
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.patches import Patch as _Patch
    legend_handles = [
        _Line2D([], [], color=profile_color, lw=2.5, ls=profile_line_style, label=type_z_score.replace("_", " ")),
        _Line2D([], [], color="black", lw=3.0, label="Zero median (control median)")
    ]
    if np.isfinite(se_vals).any():
        legend_handles.append(_Patch(facecolor=profile_color, alpha=0.30, label="Median ± SE band"))
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    # -----------------------------
    # Labels & Axis Formatting
    # -----------------------------
    xticklabels = [_pretty_label(v) for v in var_order]
    ax.set_xticks(angles)
    ax.set_xticklabels(xticklabels, fontsize=16)
    ax.tick_params(axis="x", pad=20)

    # Radial ticks: fixed range and rounded steps
    tick_step = 0.10
    tick_meds = np.round(
        np.arange(fixed_min, fixed_max + tick_step * 0.5, tick_step),
        2,
    )
    tick_radii = _to_radial(tick_meds)
    ax.set_yticks(tick_radii)
    # Use a FuncFormatter so tick labels display median-space values
    from matplotlib.ticker import FuncFormatter

    def _radius_to_signed_label(r, pos=None):
        signed = _to_median_scalar(r)
        if abs(signed) < 1e-8:
            return "0.00"
        return f"{signed:.2f}"

    ax.yaxis.set_major_formatter(FuncFormatter(_radius_to_signed_label))
    ax.tick_params(axis="y", labelsize=12)

    # Set radial limits to normalized [0, 1] radial range
    ax.set_ylim(0.0, 1.0)

    # Remove radial axis label as requested
    ax.set_ylabel("")

    ax.set_title(
        overall_title,
        fontsize=18,
        fontweight="bold",
        pad=30,
    )
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
        print(f"\nFigure saved to: {save_path}")

    plt.close(fig)

    # ------------------------------------------------------------------
    # Connectivity-type overlays: overall vs Cluster 0 vs Cluster 1
    # ------------------------------------------------------------------
    # We use the calls to plot_z_scores for the overall cohort and for
    # each connectivity-specific cluster subset to assemble overlaid
    # radar figures for functional, structural, and sfc connectivity.
    if save_path is None:
        return

    base_name = os.path.basename(save_path)
    # Determine base kind ('task' or 'domain') from type_z_score
    base_kind = "task" if type_z_score.startswith("task") else (
        "domain" if type_z_score.startswith("domain") else None
    )
    if base_kind is None:
        return

    # Derive depression_codes from filename prefix (e.g. 'F32_...' or 'modular_F32_...')
    depression_codes = "unknown"
    if "_" in base_name:
        first_token, remainder = base_name.split("_", 1)
        if first_token == "modular" and "_" in remainder:
            depression_codes = remainder.split("_", 1)[0]
        else:
            depression_codes = first_token

    key = (depression_codes, association_type, base_kind)
    store = _RADAR_OVERLAY_STORE.setdefault(
        key,
        {"overall": {}, "functional": {}, "structural": {}, "sfc": {}, "_paths": {}}
    )

    # Map current medians into a name->value dict (signed medians)
    med_signed_map = {v: float(med_signed.get(v, np.nan)) for v in z_vars}

    # Detect overall vs cluster-specific vs subset calls from save_path
    is_cluster_call = "_Cluster_0" in base_name or "_Cluster_1" in base_name
    is_subset_call = "identical_labels" in base_name or "different_labels" in base_name

    # Overall call: base figure in PLOTS_DIR (no cluster / subset suffix)
    if (not is_cluster_call) and (not is_subset_call):
        store["overall"] = med_signed_map
        _RADAR_OVERLAY_SE_OVERALL[key] = {k: float(v) for k, v in se_map.items() if np.isfinite(v)}
        return

    # Cluster-specific calls: paths live under .../schaefer1000+tian54/{conn_type}_con/
    if not is_cluster_call:
        return

    conn_dir = os.path.basename(os.path.dirname(save_path))  # e.g. 'functional_con'
    if conn_dir.startswith("functional"):
        conn_type = "functional"
    elif conn_dir.startswith("structural"):
        conn_type = "structural"
    elif conn_dir.startswith("sfc"):
        conn_type = "sfc"
    else:
        return

    cluster_label = None
    if "Cluster_0" in base_name:
        cluster_label = "Cluster 0"
    elif "Cluster_1" in base_name:
        cluster_label = "Cluster 1"
    if cluster_label is None:
        return

    dir_type = None
    if base_name.endswith("_internal.svg"):
        dir_type = "internal"
    elif base_name.endswith("_external.svg"):
        dir_type = "external"
    if dir_type is None:
        return

    conn_store = store.setdefault(conn_type, {})
    conn_store[cluster_label] = med_signed_map
    store.setdefault("_paths", {})[conn_type] = save_path
    _RADAR_OVERLAY_SE_CLUSTER.setdefault(key, {}).setdefault(conn_type, {}).setdefault(dir_type, {})[cluster_label] = {
        k: float(v) for k, v in se_map.items() if np.isfinite(v)
    }

    # If we do not yet have overall + both clusters, defer overlay
    if not store["overall"] or "Cluster 0" not in conn_store or "Cluster 1" not in conn_store:
        return

    _render_connectivity_overlay(key, conn_type, dir_type)


# Record corrected p-values and flags into an in-memory registry for radar overlays
def register_radar_overlay_significance(
    depression_codes: str,
    association_type: str,
    base_kind: str,
    conn_type: str,
    dir_type: str,
    variable_names: Sequence[str],
    pvals_corrected: Sequence[float],
    comparison_label: str,
) -> None:
    """Register between-cluster corrected p-values for overlay annotations.

    This function stores FDR-corrected p-values for a connectivity/type
    combination in the module-level overlay registry so that when the
    overlay figure for that connectivity type is assembled the per-variable
    significance markers can be displayed.

    Parameters
    ----------
    depression_codes : str
        Identifier string extracted from the filename prefix (for example,
        ``'F32'`` or ``'modular_F32'``). See ``plot_z_scores`` for how this
        is derived when saving files.
    association_type : str
        High-level association label (e.g., ``'cognition'``).
    base_kind : str
        Either ``'task'`` or ``'domain'`` indicating the plotting base.
    conn_type : str
        Connectivity type (``'functional'``, ``'structural'``, or ``'sfc'``).
    dir_type : str
        Direction string (``'internal'`` or ``'external'``).
    variable_names : Sequence[str]
        Sequence of variable names (ordered as in the radar plot) corresponding
        to the supplied ``pvals_corrected``.
    pvals_corrected : Sequence[float]
        Sequence of corrected p-values (same length and order as
        ``variable_names``).
    comparison_label : str
        Short label describing the comparison used (e.g. ``'Cluster 0 vs 1'``).

    Returns
    -------
    None

    Side effects
    -----------
    Stores the provided p-values and metadata into ``_RADAR_OVERLAY_PVALS`` and
    ``_RADAR_OVERLAY_PVALS_META`` and attempts to trigger the overlay
    renderer ``_render_connectivity_overlay`` for the given key.
    """
    key = (str(depression_codes), str(association_type), str(base_kind))
    conn = str(conn_type)
    dir = str(dir_type)

    pval_map: Dict[str, float] = {}
    for name, pval in zip(variable_names, pvals_corrected):
        try:
            pval_map[str(name)] = float(pval)
        except Exception:
            continue

    _RADAR_OVERLAY_PVALS.setdefault(key, {})[conn] = pval_map
    _RADAR_OVERLAY_PVALS_META.setdefault(key, {})[conn] = str(comparison_label)

    try:
        _render_connectivity_overlay(key, conn, dir)
    except Exception:
        pass


# Render a stored connectivity overlay (used internally when assembling radar plots)
def _render_connectivity_overlay(key: Tuple[str, str, str], conn_type: str, dir_type: str) -> None:
    """Render overlaid radar plot for a connectivity type with significance.

    This routine reads stored median profiles and bootstrap SEs from the
    module-level registries (``_RADAR_OVERLAY_STORE``,
    ``_RADAR_OVERLAY_SE_OVERALL``, ``_RADAR_OVERLAY_SE_CLUSTER``) and, when
    the registry contains the overall profile and both cluster profiles for
    the requested connectivity type and direction, assembles and saves an
    overlaid SVG figure comparing Overall vs Cluster 0 vs Cluster 1.

    Parameters
    ----------
    key : Tuple[str, str, str]
        Registry key tuple ``(depression_codes, association_type, base_kind)``
        used when the original individual radar figures were written.
    conn_type : str
        Connectivity type (``'functional'``, ``'structural'`` or ``'sfc'``).
    dir_type : str
        Direction (``'internal'`` or ``'external'``) used to select the
        correct cluster SE entries.

    Returns
    -------
    None

    Notes
    -----
    - The function performs many early returns if required registry entries
      or file-path conventions are not met; it is therefore safe to call
      repeatedly as figures are generated incrementally.
    - The overlay output filename and save location are derived from the
      previously stored individual cluster figure paths (stored under
      ``_RADAR_OVERLAY_STORE[key]['_paths'][conn_type]``); if these are
      missing the overlay is not produced.
    """
    if key not in _RADAR_OVERLAY_STORE:
        return

    store = _RADAR_OVERLAY_STORE.get(key, {})
    conn_store = store.get(conn_type, {})
    if not store.get("overall") or "Cluster 0" not in conn_store or "Cluster 1" not in conn_store:
        return

    depression_codes, association_type, base_kind = key

    def _pretty_label(v: str) -> str:
        base = v[:-2] if v.endswith("_z") else v
        return base.replace("_", " ")

    var_names = sorted(store["overall"].keys(), key=lambda v: _pretty_label(v).lower())
    n = len(var_names)
    if n < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    def _profile_closed(profile_map: Dict[str, float]) -> np.ndarray:
        arr = np.asarray([profile_map.get(v, np.nan) for v in var_names], dtype=float)
        return np.concatenate([arr, arr[:1]])

    # Extract signed profiles (may contain NaNs)
    overall_signed = _profile_closed(store["overall"])  # signed medians
    c0_signed = _profile_closed(conn_store.get("Cluster 0", {}))
    c1_signed = _profile_closed(conn_store.get("Cluster 1", {}))

    # Use fixed nominal tick range, but allow expansion if data/SE exceed bounds.
    # Mapping is asymmetric (fixed_min..fixed_max), not centered on zero.
    fixed_min = -0.70
    fixed_max = 0.40
    overall_se = _RADAR_OVERLAY_SE_OVERALL.get(key, {})
    cluster_se = _RADAR_OVERLAY_SE_CLUSTER.get(key, {}).get(conn_type, {}).get(dir_type, {})

    def _max_abs_with_se(profile: np.ndarray, se_map: Dict[str, float]) -> float:
        prof = np.asarray(profile, dtype=float)
        se_vals = []
        for var_name in var_names:
            se_val = se_map.get(var_name)
            try:
                se_float = float(se_val)
            except (TypeError, ValueError):
                se_float = 0.0
            if not np.isfinite(se_float) or se_float < 0:
                se_float = 0.0
            se_vals.append(se_float)
        se_vals = np.asarray(se_vals, dtype=float)
        target_len = len(var_names)
        if prof.size < target_len:
            pad = np.full(target_len - prof.size, np.nan)
            prof = np.concatenate([prof, pad])
        elif prof.size > target_len:
            prof = prof[:target_len]
        vals = np.abs(prof) + se_vals
        return float(np.nanmax(vals)) if vals.size else np.nan

    def _profile_min_max_with_se(profile: np.ndarray, se_map: Dict[str, float]) -> Tuple[float, float]:
        prof = np.asarray(profile, dtype=float)
        se_vals = []
        for var_name in var_names:
            se_val = se_map.get(var_name)
            try:
                se_float = float(se_val)
            except (TypeError, ValueError):
                se_float = 0.0
            if not np.isfinite(se_float) or se_float < 0:
                se_float = 0.0
            se_vals.append(se_float)
        se_vals = np.asarray(se_vals, dtype=float)
        target_len = len(var_names)
        if prof.size < target_len:
            pad = np.full(target_len - prof.size, np.nan)
            prof = np.concatenate([prof, pad])
        elif prof.size > target_len:
            prof = prof[:target_len]
        low = np.nanmin(prof - se_vals) if prof.size else np.nan
        high = np.nanmax(prof + se_vals) if prof.size else np.nan
        return float(low), float(high)

    # Compute lower/upper bounds across all profiles (including SEs)
    base_low = np.nanmin(np.concatenate([overall_signed, c0_signed, c1_signed]))
    base_high = np.nanmax(np.concatenate([overall_signed, c0_signed, c1_signed]))
    lower_med = fixed_min if not np.isfinite(base_low) else min(fixed_min, float(base_low))
    upper_med = fixed_max if not np.isfinite(base_high) else max(fixed_max, float(base_high))

    if overall_se:
        low, high = _profile_min_max_with_se(overall_signed, overall_se)
        if np.isfinite(low):
            lower_med = min(lower_med, low)
        if np.isfinite(high):
            upper_med = max(upper_med, high)
    if cluster_se:
        low, high = _profile_min_max_with_se(c0_signed, cluster_se.get("Cluster 0", {}))
        if np.isfinite(low):
            lower_med = min(lower_med, low)
        if np.isfinite(high):
            upper_med = max(upper_med, high)
        low, high = _profile_min_max_with_se(c1_signed, cluster_se.get("Cluster 1", {}))
        if np.isfinite(low):
            lower_med = min(lower_med, low)
        if np.isfinite(high):
            upper_med = max(upper_med, high)

    med_span = upper_med - lower_med
    if not np.isfinite(med_span) or med_span <= 0:
        med_span = max(fixed_max - fixed_min, 1.0)
        lower_med = fixed_min
        upper_med = fixed_min + med_span

    def _to_radial(arr_signed: np.ndarray) -> np.ndarray:
        return (np.asarray(arr_signed, dtype=float) - lower_med) / med_span

    def _to_radial_scalar(value: float) -> float:
        return float((float(value) - lower_med) / med_span)

    def _to_median_scalar(radius: float) -> float:
        return float(lower_med + float(radius) * med_span)

    overall_closed = _to_radial(overall_signed)
    c0_closed = _to_radial(c0_signed)
    c1_closed = _to_radial(c1_signed)

    pval_map = _RADAR_OVERLAY_PVALS.get(key, {}).get(conn_type, {})
    comparison_label = _RADAR_OVERLAY_PVALS_META.get(key, {}).get(conn_type)

    def _sig_marker(p: float) -> str:
        if p is None or not np.isfinite(p):
            return r"$\mathit{ns}$"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return r"$\mathit{ns}$"

    xticklabels = []
    for v in var_names:
        marker = _sig_marker(pval_map.get(v, np.nan))
        xticklabels.append(f"{_pretty_label(v)} {marker}")

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, polar=True)

    # Reference ring at zero-median level (smooth circle) - thicker solid line
    r_zero = _to_radial_scalar(0.0)
    theta_dense = np.linspace(0.0, 2 * np.pi, 720)
    ax.plot(theta_dense, np.full_like(theta_dense, r_zero), color="black", lw=3.0, ls="-", alpha=0.95)

    cluster_colors = _cluster_colors_for_modality(conn_type, dir_type)
    colors = ["black", cluster_colors["0"], cluster_colors["1"]]
    labels = ["Overall Depression", "Cluster 0", "Cluster 1"]
    polys = [overall_closed, c0_closed, c1_closed]
    alphas = [0.45, 0.35, 0.35]
    lws = [2.5, 2.0, 2.0]

    line_styles = [":", "-", "-"]
    for poly, col, lab, a, lw_, ls_ in zip(polys, colors, labels, alphas, lws, line_styles):
        # outline only (no filled polygon)
        ax.plot(angles_closed, poly, color=col, lw=lw_, ls=ls_, zorder=3)
        ax.scatter(angles, poly[:-1], marker='o', s=80, facecolor='white', edgecolor=col, linewidth=1.5, zorder=4)

    def _build_se_band(se_map: Dict[str, float], profile_vals: np.ndarray, color: str, alpha: float) -> None:
        se_vals = []
        for var_name in var_names:
            se_val = se_map.get(var_name)
            try:
                se_float = float(se_val)
            except (TypeError, ValueError):
                se_float = np.nan
            if not np.isfinite(se_float) or se_float < 0:
                se_float = np.nan
            se_vals.append(se_float)
        se_vals = np.asarray(se_vals, dtype=float)
        profile_arr = np.asarray(profile_vals, dtype=float)
        target_len = len(var_names)
        if profile_arr.size < target_len:
            pad = np.full(target_len - profile_arr.size, np.nan)
            profile_arr = np.concatenate([profile_arr, pad])
        elif profile_arr.size > target_len:
            profile_arr = profile_arr[:target_len]
        r_low = _to_radial(profile_arr - se_vals)
        r_high = _to_radial(profile_arr + se_vals)
        r_low = np.clip(r_low, 0.0, 1.0)
        r_high = np.clip(r_high, 0.0, 1.0)
        r_low = np.minimum(r_low, r_high)
        r_low_closed = np.concatenate([r_low, r_low[:1]])
        r_high_closed = np.concatenate([r_high, r_high[:1]])
        r_low_masked = np.ma.masked_invalid(r_low_closed)
        r_high_masked = np.ma.masked_invalid(r_high_closed)
        ax.fill_between(
            angles_closed,
            r_low_masked,
            r_high_masked,
            color=color,
            alpha=alpha,
            zorder=1,
            linewidth=0,
        )

    if overall_se:
        _build_se_band(overall_se, overall_signed, "black", 0.30)
    if cluster_se:
        c0_se = cluster_se.get("Cluster 0", {})
        c1_se = cluster_se.get("Cluster 1", {})
        if c0_se:
            _build_se_band(c0_se, c0_signed, colors[1], 0.30)
        if c1_se:
            _build_se_band(c1_se, c1_signed, colors[2], 0.30)

    ax.set_xticks(angles)
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.tick_params(axis="x", pad=20)

    # Radial ticks: fixed range and rounded steps
    tick_step = 0.10
    tick_meds = np.round(
        np.arange(fixed_min, fixed_max + tick_step * 0.5, tick_step),
        2,
    )
    tick_radii = _to_radial(tick_meds)
    ax.set_yticks(tick_radii)

    # Use a FuncFormatter so tick labels display signed median values (radius - med_range)
    from matplotlib.ticker import FuncFormatter

    def _radius_to_signed_label(r, pos=None):
        signed = _to_median_scalar(r)
        if abs(signed) < 1e-8:
            return "0.00"
        return f"{signed:.2f}"

    ax.yaxis.set_major_formatter(FuncFormatter(_radius_to_signed_label))
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0.0, 1.0)

    if conn_type == "functional":
        title_conn = "functional"
    elif conn_type == "structural":
        title_conn = "structural"
    else:
        title_conn = "sfc"
    ax.set_title(
        f"Z-Score Radar ({base_kind}, {title_conn} connectivity): Overall vs Clusters",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    from matplotlib.patches import Patch as _Patch
    from matplotlib.lines import Line2D as _Line2D

    legend_handles = [
        _Line2D([], [], color="black", lw=2.5, ls=":", label="Overall Depression"),
        _Patch(color=colors[1], label=labels[1]),
        _Patch(color=colors[2], label=labels[2]),
    ]
    if overall_se:
        legend_handles.append(_Patch(facecolor="black", alpha=0.20, label="Overall median ± SE"))
    if cluster_se:
        if cluster_se.get("Cluster 0"):
            legend_handles.append(_Patch(facecolor=colors[1], alpha=0.18, label="Cluster 0 median ± SE"))
        if cluster_se.get("Cluster 1"):
            legend_handles.append(_Patch(facecolor=colors[2], alpha=0.18, label="Cluster 1 median ± SE"))
    # Add zero-line legend entry (explain zero-median ring)
    legend_handles.append(_Line2D([], [], color="black", lw=3.0, label="Zero median (control median)"))
    if comparison_label:
        legend_handles.append(_Line2D([], [], color="none", label=f"Significance: {comparison_label}"))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.05, 1.02), frameon=True)

    if base_kind == "task":
        out_name = f"{depression_codes}_{association_type}_task_z_scores_{conn_type}_{dir_type}_overall_vs_clusters.svg"
    else:
        out_name = f"{depression_codes}_{association_type}_domain_z_scores_{conn_type}_{dir_type}_overall_vs_clusters.svg"

    save_path = store.get("_paths", {}).get(conn_type)
    if not save_path:
        return

    cluster_con_dir = os.path.dirname(save_path)
    cluster_base_dir = os.path.dirname(cluster_con_dir)
    plots_dir = os.path.dirname(cluster_base_dir)

    out_path = os.path.join(plots_dir, out_name)
    fig.subplots_adjust(right=0.80)
    try:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", format="svg")
        print(f"Overlaid connectivity radar figure saved to: {out_path}")
    except Exception as exc:
        print(f"Failed to save overlaid connectivity radar figure {out_path}: {exc}")
    plt.close(fig)

# Plot scatter/regression figures between connectivity metrics and cognitive z-scores
def plot_conn_cognition_association(
    data: pd.DataFrame,
    connectivity_var: str,
    cognitive_vars: Union[str, List[str]],
    save_path: Optional[str] = None,
    group_column: Optional[str] = None,
    overall_title: Optional[str] = None,
) -> None:
    """Scatter plots of connectivity vs. cognitive variables with median
    (quantile) regression lines and optional grouping.

    For each cognitive variable, this function plots the connectivity feature
    (min-max scaled to [0, 1]) on the x-axis and the cognitive score on the
    y-axis. When ``group_column`` is provided the function colors points by
    group and fits separate median (50th percentile) quantile regression
    lines per group; it also fits a semi-transparent overall median fit across
    all points. If quantile regression fails, a simple least-squares line is
    used as a fallback.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the connectivity variable, cognitive variables,
        and optional grouping column.
    connectivity_var : str
        Name of the connectivity variable (x-axis). Values are min-max
        scaled internally to the unit interval before plotting.
    cognitive_vars : str or List[str]
        Single cognitive variable name or a list of names to plot (y-axis).
    save_path : str, optional
        If provided, the entire figure grid is saved to this path.
    group_column : str, optional
        Column name used for coloring and group-specific regression lines.
        Group labels are looked up in ``_cluster_colors_for_modality`` when
        they match known cluster strings; otherwise default colors are used.
    overall_title : str, optional
        Title for the full figure. If None, an automatic title based on the
        connectivity variable and cognitive variables is used.

    Returns
    -------
    None

    Notes
    -----
    - The function uses :class:`statsmodels.QuantReg` to estimate median
      regression lines and falls back to ``numpy.polyfit`` if the solver
      fails.
    - Connectivity values are min-max scaled across all rows prior to
      plotting so that fitted slopes are comparable across different
      connectivity features.
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # Normalize cognitive_vars to list
    if isinstance(cognitive_vars, str):
        cognitive_vars = [cognitive_vars]
    
    # Validate all variables exist
    if connectivity_var not in data.columns:
        print(f"Warning: Connectivity variable '{connectivity_var}' not found in data")
        return
    
    missing_cogs = [v for v in cognitive_vars if v not in data.columns]
    if missing_cogs:
        print(f"Warning: Missing cognitive variables: {missing_cogs}")
        cognitive_vars = [v for v in cognitive_vars if v in data.columns]
    
    if not cognitive_vars:
        print("No valid cognitive variables to plot")
        return
    
    conn_guess, dir_guess = _parse_modality_from_label(connectivity_var)
    cluster_colors = _cluster_colors_for_modality(conn_guess or "functional", dir_guess or "internal")
    group_colors = {
        'Control': 'green',
        'Depression': 'purple',
        'Cluster 0': cluster_colors["0"],
        'Cluster 1': cluster_colors["1"],
    }
    
    # Determine subplot layout
    n_plots = len(cognitive_vars)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Ensure axes is always 2D array for consistency
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, n_cols)
    
    # Min-max scale connectivity variable to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_all_values = data[connectivity_var].values.reshape(-1, 1)
    x_all_scaled = scaler.fit_transform(x_all_values).flatten()
    x_all = pd.Series(x_all_scaled, index=data.index)
    
    for idx, cognitive_var in enumerate(cognitive_vars):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        y_all = data[cognitive_var]
        
        if group_column and group_column in data.columns:
            # Plot points grouped by color
            groups = sorted(data[group_column].unique())
            for group in groups:
                mask = data[group_column] == group
                x_group = x_all[mask]
                y_group = data.loc[mask, cognitive_var]
                color = group_colors.get(str(group), 'gray')
                ax.scatter(x_group, y_group, alpha=0.6, edgecolor='k', linewidth=0.5,
                          color=color, label=str(group), s=40)
            
            # Draw overall median (50th percentile) regression line (all points) - more transparent
            valid_mask = np.isfinite(x_all) & np.isfinite(y_all)
            x_valid = x_all[valid_mask].values
            y_valid = y_all[valid_mask].values
            if len(x_valid) >= 2:
                x_vals_all = np.array([x_valid.min(), x_valid.max()])
                try:
                    X_all = sm.add_constant(x_valid, has_constant='add')
                    model_all = sm.QuantReg(y_valid, X_all)
                    res_all = model_all.fit(q=0.5, max_iter=10000, p_tol=1e-8)
                    intercept_all, slope_all = res_all.params
                    y_vals_all = intercept_all + slope_all * x_vals_all
                except Exception:
                    slope_all, intercept_all = np.polyfit(x_valid, y_valid, 1)
                    y_vals_all = intercept_all + slope_all * x_vals_all
                ax.plot(x_vals_all, y_vals_all, color='black', lw=2, linestyle='--', 
                       alpha=0.4, label=f'Overall (median): y={slope_all:.3f}x+{intercept_all:.3f}', zorder=10)
            
            # Draw median (50th percentile) regression lines for each group
            for group in groups:
                mask = data[group_column] == group
                x_group = x_all[mask]
                y_group = data.loc[mask, cognitive_var]
                valid = np.isfinite(x_group) & np.isfinite(y_group)
                x_g = x_group[valid].values
                y_g = y_group[valid].values
                
                if len(x_g) >= 2:
                    x_vals_g = np.array([x_g.min(), x_g.max()])
                    try:
                        X_g = sm.add_constant(x_g, has_constant='add')
                        model_g = sm.QuantReg(y_g, X_g)
                        res_g = model_g.fit(q=0.5, max_iter=10000, p_tol=1e-8)
                        intercept_g, slope_g = res_g.params
                        y_vals_g = intercept_g + slope_g * x_vals_g
                    except Exception:
                        slope_g, intercept_g = np.polyfit(x_g, y_g, 1)
                        y_vals_g = intercept_g + slope_g * x_vals_g
                    color = group_colors.get(str(group), 'gray')
                    ax.plot(x_vals_g, y_vals_g, color=color, lw=2,
                           label=f'{group} (median): y={slope_g:.3f}x+{intercept_g:.3f}', zorder=9)
        else:
            # All points same color, single median (50th percentile) regression line
            ax.scatter(x_all, y_all, alpha=0.6, edgecolor='k', color='steelblue', s=40)
            valid_mask = np.isfinite(x_all) & np.isfinite(y_all)
            x_valid = x_all[valid_mask].values
            y_valid = y_all[valid_mask].values
            if len(x_valid) >= 2:
                x_vals = np.array([x_valid.min(), x_valid.max()])
                try:
                    X_vals = sm.add_constant(x_valid, has_constant='add')
                    model = sm.QuantReg(y_valid, X_vals)
                    res = model.fit(q=0.5, max_iter=10000, p_tol=1e-8)
                    intercept, slope = res.params
                    y_vals = intercept + slope * x_vals
                except Exception:
                    slope, intercept = np.polyfit(x_valid, y_valid, 1)
                    y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color='red', lw=2, 
                       label=f'Median fit: y={slope:.3f}x+{intercept:.3f}')
        
        ax.set_xlabel(connectivity_var.replace("_", " ").title() + " (scaled [0,1])", fontsize=10)
        ax.set_ylabel(cognitive_var.replace("_", " ").title(), fontsize=10)
        ax.set_title(f"{cognitive_var.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    if overall_title:
        fig.suptitle(overall_title, fontsize=14, fontweight='bold', y=1.00)
    else:
        fig.suptitle(f"{connectivity_var.replace('_', ' ').title()} vs Cognitive Variables", 
                    fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot grid saved to: {save_path}")
    
    plt.close()