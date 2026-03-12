"""
Module-level clustering utilities for depression subtyping analysis.

This script provides a comprehensive suite of tools for analyzing modular (community-based)
brain connectivity patterns in depressed and control cohorts. The analysis workflow supports:

1. **Data Processing**: Computing module-level connectivity metrics from subject-level
   functional (FC) and structural (SC) connectivity matrices, including structure-function
   coupling (SFC) for each brain module/community.

2. **Hierarchical Clustering**: Identifying depression subtypes (clusters) based on module
   connectivity profiles using Ward's linkage method. Clustering is performed separately
   for each connectivity type (functional, structural, SFC) and direction (internal vs.
   external module connectivity).

3. **Cluster Validation**: Bootstrap-based stability analysis, silhouette scoring,
   Calinski-Harabasz scoring, and cross-modality agreement quantification to assess
   cluster quality and reproducibility.

4. **Statistical Inference**: Quantile regression models (via rpy2/R) to compare
   connectivity distributions between:
   - Depression vs. Control
   - Cluster 0 vs. Control
   - Cluster 1 vs. Control
   - Cluster 0 vs. Cluster 1
   Multiple testing correction (FDR-BH) applied to all p-values.

5. **Visualization**: Multi-panel plots including:
   - Covariate distributions (age, sex, motion, comorbidities) across groups/clusters
   - Module violin plots with significance annotations
   - Brain maps (NIfTI overlays) showing cluster-specific module profiles
   - Correlation heatmaps and validation metric curves

Architecture
------------
The script is organized into functional sections:

- **Utility Functions**

- **R Package Management**

- **Data Loading and Preprocessing**

- **Clustering and Validation**

- **Quantile Regression**

- **Visualization**

- **High-level Pipelines**

Dependencies
------------
- NumPy, pandas, matplotlib, seaborn: Standard data/plotting libraries
- nibabel, nilearn: Neuroimaging data handling and visualization
- scipy, scikit-learn, statsmodels: Statistical analysis and machine learning
- rpy2: R interface for quantile regression (requires R with quantreg package)

Notes
-----
- Module labels (community assignments) must be provided as input from prior network
  partitioning (e.g., Louvain, Leiden, or anatomical atlas-based communities).
- Clustering operates only on depressed subjects; controls are included for comparisons.
- All p-values undergo FDR-BH correction to control false discovery rate.
- Motion metrics are modality-specific (rfMRI motion for FC, dMRI motion for SC).
"""
import os
import itertools
import warnings
import math
from matplotlib.patches import Patch
from typing import Dict, List, Tuple, Optional, Literal
from statsmodels.stats import multitest
import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from nilearn import plotting
from nilearn import image
import re
import textwrap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, normalized_mutual_info_score
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats as sp_stats
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _display_conn_type(conn_type: str) -> str:
    """Convert connectivity type to title-case display string.
    
    Parameters
    ----------
    conn_type : str
        Connectivity type identifier ('sfc').
    
    Returns
    -------
    str
        Display-friendly string: 'Functional', 'Structural', or 
        'Structure-Function Coupling'.
    """
    # Special case: 'sfc' -> 'Structure-Function Coupling'
    # Otherwise: capitalize first letter (e.g., 'functional' -> 'Functional')
    return "Structure-Function Coupling" if str(conn_type).lower() == "sfc" else str(conn_type).capitalize()

def _parse_modality_from_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract connectivity type and direction from a column or label string.
    
    Parameters
    ----------
    label : str
        Label string potentially containing keywords like 'functional', 
        'structural', 'sfc', 'internal', 'external'.
    
    Returns
    -------
    conn : str or None
        Connectivity type: 'functional', 'structural', 'sfc', or None.
    direction : str or None
        Direction type: 'internal', 'external', or None.
    """
    # Convert label to lowercase for case-insensitive matching
    lower = str(label).lower()
    
    # Check for connectivity type keywords in priority order
    conn = "functional" if "functional" in lower else "structural" if "structural" in lower else "sfc" if "sfc" in lower else None
    
    # Check for direction type keywords
    direction = "internal" if "internal" in lower else "external" if "external" in lower else None
    
    return conn, direction

def _display_conn_type_text(conn_type: str) -> str:
    """Convert connectivity type to lowercase display text.
    
    Parameters
    ----------
    conn_type : str
        Connectivity type identifier ('sfc').
    
    Returns
    -------
    str
        Lowercase display string: 'functional', 'structural', or 
        'structure-function coupling'.
    """
    return "structure-function coupling" if str(conn_type).lower() == "sfc" else str(conn_type)

def get_motion_columns(conn_type, fMRI_MOTION_METRIC, dMRI_MOTION_METRIC):
  """Return a mapping of motion covariate labels to cohort columns for a modality.
  Parameters
  ----------
  conn_type : str
      Connectivity type ('functional', 'structural', 'sfc').
  fMRI_MOTION_METRIC : str
      Column name for fMRI motion metric.
  dMRI_MOTION_METRIC : str
      Column name for dMRI motion metric.

  Returns
  -------
  dict
      Mapping of motion covariate labels to cohort columns.
  """
  # Define motion metric mappings for each connectivity type
  # Functional connectivity uses fMRI motion (e.g., framewise displacement)
  # Structural connectivity uses dMRI motion (e.g., head motion during diffusion scan)
  # SFC requires both since it couples functional and structural data
  MOTION_METRICS = {
  'functional': {'motion_fmri': fMRI_MOTION_METRIC},
  'structural': {'motion_dmri': dMRI_MOTION_METRIC},
  'sfc': {
    'motion_fmri': fMRI_MOTION_METRIC,
    'motion_dmri': dMRI_MOTION_METRIC,
  },
}
  # Look up the motion columns for the specified connectivity type
  mapping = MOTION_METRICS.get(conn_type)
  
  # If connectivity type is unrecognized, fall back to default dMRI motion column
  if mapping is None:
    return {'motion': 'p24453_i2'}
  return mapping

def _cluster_colors_for_modality(conn_type: str, dir_type: str) -> Dict[str, str]:
    """Get cluster color palette for a specific connectivity type and direction.
    
    Provides distinct color schemes for each (connectivity, direction) pair to
    enhance visual distinction in multi-panel plots.
    
    Parameters
    ----------
    conn_type : str
        Connectivity type: 'functional', 'structural', or 'sfc'.
    dir_type : str
        Direction type: 'internal' or 'external'.
    
    Returns
    -------
    dict
        Color mapping: {'0': hex_color, '1': hex_color} for Cluster 0 and 1.
    """
    # Normalize inputs to lowercase and strip whitespace for robust matching
    key = (str(conn_type).lower().strip(), str(dir_type).lower().strip())
    
    # Define unique color pairs for each (connectivity_type, direction) combination
    # This ensures clusters are visually distinguishable across different modalities
    palette = {
        ("functional", "internal"): {"0": "#380075", "1": "#b2df8a"},  # Purple/Green
        ("functional", "external"): {"0": "#e7298a", "1": "#66a61e"},  # Magenta/Olive
        ("structural", "internal"): {"0": "#ffd92f", "1": "#bc80bd"},  # Yellow/Lavender
        ("structural", "external"): {"0": "#f1b6da", "1": "#a6d854"},  # Pink/Lime
        ("sfc", "internal"): {"0": "#cab2d6", "1": "#b3de69"},        # Light Purple/Chartreuse
        ("sfc", "external"): {"0": "#0026ff", "1": "#fd7600"},        # Blue/Orange
    }
    
    # Return color mapping for this modality; fall back to default if not found
    return palette.get(key, {"0": "#6a3d9a", "1": "#b2df8a"})

def _append_to_text_log(log_path: Optional[str], block: str) -> None:
    """Append a block of text to a log file, creating parent dirs if needed.

    Parameters
    ----------
    log_path : str or None
        Absolute path to log file. If None, no action is taken.
    block : str
        Text block to append to log file.

    Notes
    -----
    If ``log_path`` is None, this is a no-op. Creates parent directories
    automatically. Silently ignores errors to prevent breaking main analysis.
    """
    # If no log path provided, skip logging (allows optional logging)
    if not log_path:
        return
    try:
        # Extract directory path from the log file path
        log_dir = os.path.dirname(log_path)
        
        # Create parent directories if they don't exist
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Open log file in append mode and write the text block
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(block)
            
            # Ensure the log ends with a newline for proper formatting
            if not block.endswith("\n"):
                f.write("\n")
    except Exception:
        # Logging failures should never crash the main analysis
        # Silently ignore errors (e.g., permission issues, disk full)
        pass

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
    - FDR controls the expected proportion of false positives among rejections
    - Bonferroni controls the family-wise error rate (FWER)
    - FDR is less conservative and has more power than Bonferroni
    """
    # Apply the chosen multiple testing correction method
    if method == 'fdr_bh':
        # False Discovery Rate (Benjamini-Hochberg): less conservative, recommended for most cases
        reject, pvals_corrected = multitest.fdrcorrection(p_values, alpha=alpha)
        correction_name = 'FDR (Benjamini-Hochberg)'
    elif method == 'bonferroni':
        # Bonferroni correction: more conservative, controls family-wise error rate
        reject, pvals_corrected, _, _ = multitest.multipletests(
            p_values, alpha=alpha, method='bonferroni'
        )
        correction_name = 'Bonferroni'
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build a formatted text summary table of the correction results
    lines: List[str] = []
    lines.append("\n" + "=" * 80)
    lines.append(f"MULTIPLE TESTING CORRECTION: {correction_name}")
    if log_context:
        lines.append(f"Context: {log_context}")
    lines.append("=" * 80)
    
    # Table header with column names
    lines.append(f"\n{'Variable':<40} {'Test':<20} {'p (raw)':<12} {'p (adj)':<12} {'Sig.':<5}")
    lines.append("-" * 90)

    # Add a row for each test with raw and corrected p-values
    for i, var in enumerate(variable_names):
        # Assign significance markers based on corrected p-value thresholds
        sig_marker = "***" if pvals_corrected[i] < 0.001 else \
                     "**" if pvals_corrected[i] < 0.01 else \
                     "*" if pvals_corrected[i] < 0.05 else "n.s."

        lines.append(
            f"{var:<40} {test_methods[i]:<20} {p_values[i]:<12.6f} "
            f"{pvals_corrected[i]:<12.6f} {sig_marker:<5}"
        )

    # Summary footer showing count of significant results
    lines.append("-" * 90)
    lines.append(f"Significant results: {reject.sum()} / {len(p_values)}")
    lines.append("=" * 80)

    # Write the summary table to the log file if specified
    _append_to_text_log(log_path, "\n".join(lines))

    return reject, pvals_corrected

def _select_module_columns(feature_df: pd.DataFrame, conn_type: str, dir_type: str) -> list:
    """Select feature columns matching connectivity type and direction.

    Robust column selector that matches tokens like 'functional' and 'internal' as whole tokens
    separated by underscores/dashes/dots or string boundaries, in either order.
    Falls back to simple substring matching if no regex matches found.

    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame containing module feature columns.
    conn_type : str
        Connectivity type to match: 'functional', 'structural', or 'sfc'.
    dir_type : str
        Direction type to match: 'internal' or 'external'.

    Returns
    -------
    list of str
        Column names matching the connectivity type and direction.

    Raises
    ------
    AssertionError
        If no columns match the specified connectivity and direction.
    """
    # Convert all column names to strings for consistent processing
    cols = feature_df.columns.astype(str)
    
    # Define regex patterns to match connectivity/direction as whole tokens
    # Tokens can be separated by underscores, dashes, dots, spaces, or string boundaries
    sep = r'(?:^|[_\-\.\s])'         # allowed token start (start of string or separator)
    end = r'(?:$|[_\-\.\s])'         # allowed token end (end of string or separator)
    
    # Create patterns matching both possible orders: conn_type...dir_type OR dir_type...conn_type
    pat_a_b = rf'{sep}{re.escape(conn_type)}{end}.*{sep}{re.escape(dir_type)}{end}'
    pat_b_a = rf'{sep}{re.escape(dir_type)}{end}.*{sep}{re.escape(conn_type)}{end}'
    
    # Find all columns matching either pattern (order-independent)
    mask = cols.str.contains(pat_a_b, regex=True) | cols.str.contains(pat_b_a, regex=True)
    selected = cols[mask].tolist()

    # If regex matching fails, fall back to simple substring matching for backward compatibility
    if not selected:
        # Check if both conn_type and dir_type appear anywhere in the column name
        selected = [c for c in cols if (conn_type in c and dir_type in c)]

    # Final validation: ensure at least one column was found
    if not selected:
        raise AssertionError(
            f"No columns found for connectivity='{conn_type}' and direction='{dir_type}'. "
            f"Tried patterns: '{pat_a_b}' and '{pat_b_a}'."
        )
    return selected

def _is_number(val) -> bool:
    """Check if a value can be interpreted as a finite float.
    
    Parameters
    ----------
    val : any
        Value to test (can be str, int, float, or other type).
    
    Returns
    -------
    bool
        True if val can be converted to a finite float, False otherwise.
    
    Notes
    -----
    Returns False for None, NaN, infinity, and non-numeric types.
    """
    try:
        # Check for None explicitly (can't convert to float)
        if val is None:
            return False
        
        # Attempt to convert value to float (will raise exception if not possible)
        float(str(val))
        return True
    except (ValueError, TypeError):
        # Conversion failed: value is not numeric
        return False
    
def get_subject_ids_by_status(
    data: pd.DataFrame,
    group_col: str = 'depression_status',
    depressed_value: int = 1,
    control_value: int = 0,
) -> Tuple[List[str], List[str]]:
    """Extract and return sorted depressed and control subject IDs.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'eid' column and group status column.
    group_col : str, default='depression_status'
        Name of column containing group status (0=control, 1=depressed).
    depressed_value : int, default=1
        Value in group_col indicating depressed subjects.
    control_value : int, default=0
        Value in group_col indicating control subjects.

    Returns
    -------
    depressed_ids : list of str
        Sorted list of depressed subject IDs (as strings).
    control_ids : list of str
        Sorted list of control subject IDs (as strings).
    """
    # Extract subject IDs where group status matches depressed value
    depressed_ids = data.loc[data[group_col] == depressed_value, 'eid'].astype(str).tolist()
    
    # Extract subject IDs where group status matches control value
    control_ids = data.loc[data[group_col] == control_value, 'eid'].astype(str).tolist()
    
    # Sort both lists for consistent ordering across analyses
    depressed_ids.sort()
    control_ids.sort()
    
    return depressed_ids, control_ids

# ==============================================================================
# R PACKAGE MANAGEMENT
# ==============================================================================
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
        Saves figures and a TXT summary table to `out_dir`.

    Notes
    -----
    If R or `rpy2` is not available/configured correctly this function
    will raise exceptions from the underlying R runtime. The caller
    should handle these exceptions if a graceful fallback is desired.
    """
    # Execute R code through rpy2 interface:
    # - require() attempts to load the package, returns FALSE if not available
    # - quietly=TRUE suppresses loading messages
    # - If package not found, install.packages() downloads and installs from CRAN
    # - repos argument specifies the CRAN mirror to use
    ro.r(f'''
        if (!require("{package_name}", quietly = TRUE)) {{
            install.packages("{package_name}", repos = "https://cloud.r-project.org")
        }}
    ''')


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
    # Use localconverter to ensure rpy2 can convert pandas DataFrames to/from R data.frames
    # This is required when calling importr() from notebook kernels or multithreaded contexts
    # to avoid ContextVar-related NotImplementedError exceptions
    with localconverter(default_converter + pandas2ri.converter):
        # Import core R packages that are always available
        base = importr('base')        # R base package (fundamental R functions)
        utils = importr('utils')      # R utils package (utility functions)
        stats = importr('stats')      # R stats package (statistical functions)

        # Install quantile regression and multiple comparison packages if not present
        install_r_package_if_missing('quantreg')   # For quantile regression models

        # Import the statistical packages for use in quantile regression analysis
        quantreg = importr('quantreg')   # Quantile regression functionality

    # Return dictionary of imported R package objects for downstream use
    return {
        'base': base,
        'utils': utils,
        'stats': stats,
        'quantreg': quantreg,
    }

def _sanitize_colnames(cols):
    """Sanitize column names to be safe Python/R identifiers.

    Transforms column names to follow identifier rules and ensures uniqueness.

    Parameters
    ----------
    cols : list-like
        Original column names to sanitize.

    Returns
    -------
    sanitized_cols : list
        List of sanitized column names (safe identifiers).
    mapping : dict
        Mapping from original column names to sanitized names.

    Notes
    -----
    Sanitization rules:
    - Non-alphanumeric characters replaced with underscore
    - Leading digits prefixed with 'X'
    - Multiple underscores collapsed
    - Uniqueness ensured by appending numeric suffixes when needed
    """
    out = []           # List of sanitized column names
    mapping = {}       # Mapping from original to sanitized names
    seen = {}          # Counter for each base name to ensure uniqueness
    
    for c in cols:
        # Replace any non-alphanumeric character (except underscore) with underscore
        s = re.sub(r'[^0-9A-Za-z_]', '_', str(c))
        
        # Collapse multiple consecutive underscores into single underscore
        s = re.sub(r'_+', '_', s).strip('_')
        
        # If name starts with a digit, prefix with 'X' (R/Python identifier rule)
        if re.match(r'^[0-9]', s):
            s = 'X' + s
        
        # Handle empty strings (e.g., if original name was all special chars)
        if s == '':
            s = 'col'
        
        # Ensure uniqueness by appending counter if name already seen
        base = s
        i = seen.get(base, 0)
        if i > 0:
            s = f"{base}_{i}"  # Append _1, _2, _3, etc. for duplicates
        seen[base] = i + 1
        
        # Add sanitized name to output list and update mapping
        out.append(s)
        mapping[c] = s
    
    return out, mapping

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_combined_cohort_data(cohort_path: str) -> pd.DataFrame:
    """Load combined cohort CSV file for module clustering analysis.
    
    Parameters
    ----------
    cohort_path : str
        Absolute path to CSV file containing module connectivity features,
        subject IDs, and covariates.
    
    Returns
    -------
    pd.DataFrame
        Cohort data with columns for eid, depression_status, module features,
        and covariates.
    """
    # Simple wrapper around pandas read_csv for consistent loading
    # Returns DataFrame with subject-level module connectivity metrics and covariates
    return pd.read_csv(cohort_path)

def compute_module_connectivity(FC_matrix: np.ndarray,SC_matrix: np.ndarray, community_labels: list) -> dict:
    """
    Compute internal and external connectivity for each module.

    Parameters
    ----------
    FC_matrix : np.ndarray
        The functional connectivity matrix (N x N).
    SC_matrix : np.ndarray
        The structural connectivity matrix (N x N).
    community_labels : list
        List of community labels for each region (length N).

    Returns
    -------    
    module_FC_internal_conn_values : dict
        A dictionary with module labels as keys and internal functional connectivity values.
    module_FC_external_conn_values : dict
        A dictionary with module labels as keys and external functional connectivity values.
    module_SC_internal_conn_values : dict
        A dictionary with module labels as keys and internal structural connectivity values.
    module_SC_external_conn_values : dict
        A dictionary with module labels as keys and external structural connectivity values.
    module_FC_internal_connectivity : dict
        A dictionary with module labels as keys and mean absolute internal functional connectivity values.
    module_FC_external_connectivity : dict
        A dictionary with module labels as keys and mean absolute external functional connectivity values.
    module_SC_internal_connectivity : dict
        A dictionary with module labels as keys and mean absolute internal structural connectivity values.
    module_SC_external_connectivity : dict
        A dictionary with module labels as keys and mean absolute external structural connectivity values.

    """
    # Get list of all unique module/community identifiers from the labels
    unique_modules = np.unique(community_labels)
    
    # Initialize dictionaries to store connectivity values and summary statistics
    # for each module across both FC (functional) and SC (structural) modalities
    module_FC_internal_conn_values = {}   # Full FC matrices within each module
    module_FC_external_conn_values = {}   # Full FC matrices from module to others
    module_SC_internal_conn_values = {}   # Full SC matrices within each module
    module_SC_external_conn_values = {}   # Full SC matrices from module to others
    module_FC_internal_connectivity = {}  # Mean abs FC within module (summary statistic)
    module_FC_external_connectivity = {}  # Mean abs FC to other modules (summary statistic)
    module_SC_internal_connectivity = {}  # Mean SC within module (summary statistic)
    module_SC_external_connectivity = {}  # Mean SC to other modules (summary statistic)

    # Loop through each module to compute its internal and external connectivity
    for module in unique_modules:
        # Get indices of all ROIs belonging to current module
        module_indices = np.where(community_labels == module)[0]
        
        # Get indices of all ROIs NOT belonging to current module
        other_indices = np.where(community_labels != module)[0]

        # === Functional Connectivity (FC) Internal ===
        # Extract submatrix of FC connections within the module (module ROIs × module ROIs)
        FC_internal_conn_values = FC_matrix[np.ix_(module_indices, module_indices)]
        module_FC_internal_conn_values[module] = FC_internal_conn_values
        
        # Compute summary: mean of absolute FC values within module
        FC_internal_connectivity = np.mean(np.abs(FC_internal_conn_values))
        module_FC_internal_connectivity[module] = FC_internal_connectivity
        
        # === Functional Connectivity (FC) External ===
        # Extract submatrix of FC connections from module to other modules (module ROIs × other ROIs)
        FC_external_conn_values = FC_matrix[np.ix_(module_indices, other_indices)]
        module_FC_external_conn_values[module] = FC_external_conn_values
        
        # Compute summary: mean of absolute FC values from module to others
        FC_external_connectivity = np.mean(np.abs(FC_external_conn_values))
        module_FC_external_connectivity[module] = FC_external_connectivity
        
        # === Structural Connectivity (SC) Internal ===
        # Extract submatrix of SC connections within the module (module ROIs × module ROIs)
        SC_internal_conn_values = SC_matrix[np.ix_(module_indices, module_indices)]
        module_SC_internal_conn_values[module] = SC_internal_conn_values
        
        # Compute summary: mean of SC values within module (no absolute value for SC)
        SC_internal_connectivity = np.mean(SC_internal_conn_values)
        module_SC_internal_connectivity[module] = SC_internal_connectivity
        
        # === Structural Connectivity (SC) External ===
        # Extract submatrix of SC connections from module to other modules (module ROIs × other ROIs)
        SC_external_conn_values = SC_matrix[np.ix_(module_indices, other_indices)]
        module_SC_external_conn_values[module] = SC_external_conn_values
        
        # Compute summary: mean of SC values from module to others
        SC_external_connectivity = np.mean(SC_external_conn_values)
        module_SC_external_connectivity[module] = SC_external_connectivity

    # Return all connectivity matrices and summary statistics
    # Full matrices allow for detailed SFC computation later
    # Summary statistics provide direct features for clustering
    return module_FC_internal_conn_values, module_FC_external_conn_values, module_FC_internal_connectivity, module_FC_external_connectivity, module_SC_internal_conn_values, module_SC_external_conn_values, module_SC_internal_connectivity, module_SC_external_connectivity

def compute_module_sfc(
    module_FC_internal_conn_values: dict,
    module_FC_external_conn_values: dict,
    module_SC_internal_conn_values: dict,
    module_SC_external_conn_values: dict,
    sfc_warning_log_dir: Optional[str] = None,
    subject_id: Optional[str] = None,
    cohort_label: Optional[str] = None,
) -> dict:
    """
    Compute internal and external structure-function coupling (SFC) at the module level.

    This function computes per-module SFC by performing row-wise Pearson
    correlations between structural (SC) and functional (FC) connectivity
    patterns. It returns two dictionaries mapping module labels to aggregated
    SFC summary statistics (mean of per-node correlations) for internal and
    external directions respectively.

    Notes
    -----
    - `sfc_warning_log_dir` (optional): if provided, the function will capture
        and suppress `scipy.stats.ConstantInputWarning` instances that arise when
        attempting to compute Pearson correlations on constant vectors. Captured
        warnings are recorded into two CSV files inside this directory:
            - `sfc_internal_warnings.csv`
            - `sfc_external_warnings.csv`
        Both files are opened in append mode so multiple subjects (or cohorts)
        that use the same directory will be concatenated into the same logs.

    - `subject_id` / `cohort_label` (optional): when logging is enabled these
        values are included with each warning record to identify the subject and
        cohort that produced the warning.

    - The function also counts the number of per-node correlations attempted and
        the number that resulted in NaN (e.g., due to constant inputs). For each
        subject, a per-subject summary row is appended to the internal/external
        CSVs with fields `n_nodes`, `n_nan`, and `nan_fraction` (n_nan / n_nodes).

    Parameters
    ----------
    module_FC_internal_conn_values : dict
        A dictionary with module labels as keys and functional internal connectivity values.
    module_FC_external_conn_values : dict
        A dictionary with module labels as keys and functional external connectivity values.
    module_SC_internal_conn_values : dict
        A dictionary with module labels as keys and structural internal connectivity values.
    module_SC_external_conn_values : dict
        A dictionary with module labels as keys and structural external connectivity values.
    sfc_warning_log_dir : str or None, optional
        Directory path where per-subject SFC warning CSVs will be appended. If
        None, no files are written (default: None).
    subject_id : str or None, optional
        Subject identifier to include in warning/log records (default: None).
    cohort_label : str or None, optional
        Cohort label (e.g., 'F32' or 'control') to include in warning/log records (default: None).

    Returns
    -------
    tuple of two dicts
        `(module_internal_sfc, module_external_sfc)` where each dict maps
        module_label -> aggregated SFC (float) or `np.nan` when undefined.
    """

    def _rowwise_pearsonr(a: np.ndarray, b: np.ndarray, warnings_log: Optional[list] = None,
                         module_label: Optional[str] = None, direction: Optional[str] = None) -> np.ndarray:
        """
        Compute Pearson r for each row between `a` and `b` (row-wise correlation).

        This helper function computes a Pearson correlation coefficient for each
        row pair `a[i, :]` and `b[i, :]`. Behavior additions:

        - `warnings_log` (optional list): if provided, this list will be
          extended with structured dictionaries describing any
          `ConstantInputWarning` captured during the call. Each record includes
          `subject_id`, `cohort`, `module`, `row_index`, `direction`, and
          `warning_message`.
        - `module_label` and `direction`: descriptive metadata attached to any
          captured warning records.

        Parameters
        ----------
        a, b : np.ndarray
            Two 2D arrays with identical shape `(n_rows, n_cols)`.
        warnings_log : list or None
            Mutable list used to collect warning records (default: None).
        module_label : str or None
            Module identifier to include in warning records (default: None).
        direction : str or None
            Either `'internal'` or `'external'` to describe the comparison
            direction (default: None).

        Returns
        -------
        np.ndarray
            1D array of length `n_rows` with Pearson r values. Entries are
            `np.nan` when correlation is undefined (e.g., constant input).
        """
        # Handle empty arrays early (return empty result)
        if a.size == 0 or b.size == 0:
            return np.array([], dtype=float)

        # Ensure inputs are float arrays for numerical stability
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        
        # Validate that both matrices have the same shape
        if a.shape != b.shape:
            raise ValueError(f"Shapes must match for rowwise correlation: {a.shape} vs {b.shape}")

        # Compute Pearson correlation coefficient for each row independently
        # This correlates SC and FC patterns across ROIs within/outside the module
        # scipy.pearsonr handles edge cases (constant values, NaNs) gracefully
        n_rows = a.shape[0]
        r_list = np.empty(n_rows, dtype=float)
        r_list.fill(np.nan)  # Initialize with NaN (undefined correlations)
        
        for i in range(n_rows):
            # Extract row i from both matrices (representing connectivity pattern for one node)
            ai = a[i, :]
            bi = b[i, :]
            
            # Create mask for finite values (exclude NaN/Inf from both arrays)
            mask = np.isfinite(ai) & np.isfinite(bi)
            
            # Need at least 2 valid points to compute correlation
            if np.count_nonzero(mask) < 2:
                r_list[i] = np.nan
                continue
            
            try:
                # Capture warnings emitted by scipy.stats.pearsonr (e.g., ConstantInputWarning)
                # Use warnings.catch_warnings(record=True) so warnings are recorded and not printed
                with warnings.catch_warnings(record=True) as w:
                    # Always record warnings so we can inspect them
                    warnings.simplefilter("always")
                    r_val, _ = sp_stats.pearsonr(ai[mask], bi[mask])

                    # Inspect recorded warnings and log ConstantInputWarning occurrences
                    if warnings_log is not None and len(w) > 0:
                        for warn in w:
                            # Only capture ConstantInputWarning to mirror existing logging elsewhere
                            try:
                                from scipy.stats import ConstantInputWarning
                                is_const_warn = issubclass(warn.category, ConstantInputWarning)
                            except Exception:
                                is_const_warn = False

                            if is_const_warn:
                                # Build a structured warning record for downstream analysis.
                                # Include module-specific node counts so warning rows
                                # and per-module summary rows share the same fields.
                                rec = {
                                    'subject_id': subject_id,
                                    'cohort': cohort_label,
                                    'module': module_label,
                                    'row_index': int(i),
                                    'direction': direction,
                                    'warning_message': str(warn.message),
                                }
                                # Number of columns used for this row correlation
                                try:
                                    ncols = int(ai.size)
                                except Exception:
                                    ncols = None

                                if str(direction) == 'internal':
                                    rec['n_nodes'] = ncols
                                    # This row produced a NaN correlation
                                    rec['n_nan'] = 1
                                    try:
                                        rec['nan_fraction'] = float(rec['n_nan']) / float(rec['n_nodes']) if rec['n_nodes'] else None
                                    except Exception:
                                        rec['nan_fraction'] = None
                                else:
                                    # external direction: count other nodes
                                    rec['n_other_nodes'] = ncols
                                    rec['n_nan'] = 1
                                    try:
                                        rec['nan_fraction'] = float(rec['n_nan']) / float(rec['n_other_nodes']) if rec['n_other_nodes'] else None
                                    except Exception:
                                        rec['nan_fraction'] = None

                                warnings_log.append(rec)
            except Exception:
                # If correlation fails (e.g., numerical issues), mark as NaN
                r_val = np.nan
            
            # pearsonr may return NaN for constant arrays; preserve that
            if not np.isfinite(r_val):
                r_list[i] = np.nan
            else:
                r_list[i] = float(r_val)
        
        return r_list

    # Initialize dictionaries to store SFC values for each module
    module_internal_sfc = {}  # SFC within each module (intra-module coupling)
    module_external_sfc = {}  # SFC between module and rest of brain (inter-module coupling)

    # Prepare containers to collect ConstantInputWarning occurrences for internal/external SFC
    internal_warnings = []
    external_warnings = []


    # Loop through all modules to compute their structure-function coupling
    for module in module_FC_internal_conn_values.keys():
        # Retrieve connectivity matrices for this module (may be None if missing)
        FC_int = module_FC_internal_conn_values.get(module)  # FC within module
        SC_int = module_SC_internal_conn_values.get(module)  # SC within module
        FC_ext = module_FC_external_conn_values.get(module)  # FC to other modules
        SC_ext = module_SC_external_conn_values.get(module)  # SC to other modules

        # === INTERNAL SFC COMPUTATION ===
        # Compute how well structural connectivity predicts functional connectivity
        # within the module (intra-module structure-function coupling)
        if FC_int is None or SC_int is None or np.prod(FC_int.shape) == 0 or np.prod(SC_int.shape) == 0:
            # If matrices are missing or empty, SFC cannot be computed
            module_internal_sfc[module] = np.nan
        else:
            try:
                # Compute row-wise correlation: for each node in the module,
                # correlate its SC pattern with its FC pattern (both within-module)
                # Pass `internal_warnings` to collect any ConstantInputWarning occurrences
                r_int = _rowwise_pearsonr(
                    SC_int,
                    FC_int,
                    warnings_log=internal_warnings,
                    module_label=module,
                    direction='internal',
                )  # per-node r within module
                # Build per-module internal summary and append below
                if r_int is not None:
                    n_module_nodes = int(r_int.size)  # number of nodes within this module
                    # Create and append a per-module summary record for internal SFC
                    try:
                        module_n_nan = int(np.count_nonzero(np.isnan(r_int)))
                        module_n_nodes = n_module_nodes
                        module_nan_frac = (
                            float(module_n_nan) / float(module_n_nodes)
                            if module_n_nodes > 0 else np.nan
                        )
                    except Exception:
                        module_n_nan = 0
                        module_n_nodes = n_module_nodes
                        module_nan_frac = np.nan

                    internal_module_summary = {
                        'subject_id': subject_id,
                        'cohort': cohort_label,
                        'module': module,
                        'row_index': None,
                        'direction': 'internal',
                        'warning_message': None,
                        'n_nodes': int(module_n_nodes),
                        'n_nan': int(module_n_nan),
                        'nan_fraction': module_nan_frac,
                    }
                    internal_warnings.append(internal_module_summary)
                # Aggregate per-node correlations: use mean of correlation values
                # (NaN values are ignored via nanmean)
                if r_int.size == 0 or np.all(np.isnan(r_int)):
                    module_internal_sfc[module] = np.nan
                else:
                    # Module-level internal SFC is the mean per-node correlation
                    module_internal_sfc[module] = float(np.nanmean(r_int))
            except Exception:
                # If any error occurs during computation, mark as NaN
                module_internal_sfc[module] = np.nan

        # === EXTERNAL SFC COMPUTATION ===
        # Compute how well structural connectivity predicts functional connectivity
        # between the module and other modules (inter-module structure-function coupling)
        if FC_ext is None or SC_ext is None or np.prod(FC_ext.shape) == 0 or np.prod(SC_ext.shape) == 0:
            # If matrices are missing or empty, SFC cannot be computed
            module_external_sfc[module] = np.nan
        else:
            try:
                # Compute row-wise correlation: for each node in the module,
                # correlate its SC pattern with its FC pattern (both to external nodes)
                # Pass `external_warnings` to collect any ConstantInputWarning occurrences
                r_ext = _rowwise_pearsonr(
                    SC_ext,
                    FC_ext,
                    warnings_log=external_warnings,
                    module_label=module,
                    direction='external',
                )
                # Build per-module external summary and append below
                if r_ext is not None:
                    # Number of nodes outside this module (columns for external comparison)
                    n_other_nodes = int(SC_ext.shape[1]) if hasattr(SC_ext, 'shape') and SC_ext.size > 0 else 0
                    # Create and append a per-module summary record for external SFC
                    try:
                        module_ext_n_nan = int(np.count_nonzero(np.isnan(r_ext)))
                        module_ext_n_other = n_other_nodes
                        module_ext_nan_frac = (
                            float(module_ext_n_nan) / float(module_ext_n_other)
                            if module_ext_n_other > 0 else np.nan
                        )
                    except Exception:
                        module_ext_n_nan = 0
                        module_ext_n_other = n_other_nodes
                        module_ext_nan_frac = np.nan

                    external_module_summary = {
                        'subject_id': subject_id,
                        'cohort': cohort_label,
                        'module': module,
                        'row_index': None,
                        'direction': 'external',
                        'warning_message': None,
                        'n_other_nodes': int(module_ext_n_other),
                        'n_nan': int(module_ext_n_nan),
                        'nan_fraction': module_ext_nan_frac,
                    }
                    external_warnings.append(external_module_summary)
                # Aggregate per-node correlations
                if r_ext.size == 0 or np.all(np.isnan(r_ext)):
                    module_external_sfc[module] = np.nan
                else:
                    # Module-level external SFC is the mean per-node correlation
                    module_external_sfc[module] = float(np.nanmean(r_ext))
            except Exception:
                # If any error occurs during computation, mark as NaN
                module_external_sfc[module] = np.nan

    # If a log directory was provided, append the captured warning records to CSV files
    # Consolidate per-row warning entries and per-module summaries into a single
    # row per (subject_id, cohort, module, direction) to avoid duplicate NaN rows.
    if sfc_warning_log_dir is not None:
        try:
            os.makedirs(sfc_warning_log_dir, exist_ok=True)

            def _collapse_warning_records(records: list, external: bool = False) -> pd.DataFrame:
                """Collapse per-row warning records and module summaries into one row per module."""
                if not records:
                    return pd.DataFrame()

                df = pd.DataFrame(records)

                # Normalize columns so groupby works even if some keys are missing
                for c in ['subject_id', 'cohort', 'module', 'direction', 'row_index', 'warning_message']:
                    if c not in df.columns:
                        df[c] = None

                denom_col = 'n_other_nodes' if external else 'n_nodes'
                if denom_col not in df.columns:
                    df[denom_col] = None

                if 'n_nan' not in df.columns:
                    df['n_nan'] = 0

                collapsed_rows = []
                grp = df.groupby(['subject_id', 'cohort', 'module', 'direction'], dropna=False)
                for key, g in grp:
                    subj_id, cohort, module, direction = key

                    row_indices = [str(int(x)) for x in g['row_index'].dropna().tolist() if str(x).strip() != '']
                    messages = [str(x) for x in g['warning_message'].dropna().tolist() if str(x).strip() != '']

                    denom_vals = [int(x) for x in g[denom_col].dropna().unique().tolist() if str(x) != 'None']
                    denom = denom_vals[0] if denom_vals else (int(g[denom_col].dropna().iloc[0]) if not g[denom_col].dropna().empty else None)

                    # Prefer explicit per-module summary row n_nan when present (row_index is null)
                    has_summary_row = g['row_index'].isnull().any()
                    if has_summary_row:
                        summary_vals = [int(x) for x in g.loc[g['row_index'].isnull(), 'n_nan'].dropna().unique().tolist()]
                        total_n_nan = int(summary_vals[0]) if summary_vals else len(row_indices)
                    else:
                        n_nan_vals = [int(x) for x in g['n_nan'].dropna().astype(int).tolist()]
                        total_n_nan = int(np.sum(n_nan_vals)) if n_nan_vals else len(row_indices)

                    try:
                        nan_fraction = float(total_n_nan) / float(denom) if denom and denom > 0 else np.nan
                    except Exception:
                        nan_fraction = np.nan

                    collapsed_rows.append({
                        'subject_id': subj_id,
                        'cohort': cohort,
                        'module': module,
                        'row_index': ';'.join(row_indices) if row_indices else None,
                        'direction': direction,
                        'warning_message': '; '.join(messages) if messages else None,
                        denom_col: int(denom) if denom is not None else None,
                        'n_nan': int(total_n_nan),
                        'nan_fraction': nan_fraction,
                    })

                return pd.DataFrame(collapsed_rows)

            # Internal warnings CSV
            internal_path = os.path.join(sfc_warning_log_dir, 'sfc_internal_warnings.csv')
            if internal_warnings:
                internal_df = _collapse_warning_records(internal_warnings, external=False)
                header = not os.path.exists(internal_path)
                cols = ['subject_id', 'cohort', 'module', 'row_index', 'direction', 'warning_message', 'n_nodes', 'n_nan', 'nan_fraction']
                internal_df.to_csv(internal_path, mode='a', index=False, header=header, columns=cols)

            # External warnings CSV
            external_path = os.path.join(sfc_warning_log_dir, 'sfc_external_warnings.csv')
            if external_warnings:
                external_df = _collapse_warning_records(external_warnings, external=True)
                header = not os.path.exists(external_path)
                cols_ext = ['subject_id', 'cohort', 'module', 'row_index', 'direction', 'warning_message', 'n_other_nodes', 'n_nan', 'nan_fraction']
                external_df.to_csv(external_path, mode='a', index=False, header=header, columns=cols_ext)
        except Exception:
            # Logging failures should not interrupt SFC computation
            pass

    return module_internal_sfc, module_external_sfc

def merge_covariates_for_module(
    final_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    covariate_columns: List[str],
    general_dir: str,
    output_basename: str = 'module_connectivity_features_with_covariates',
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Merge covariates into module features and sanitize column names.

    Parameters
    ----------
    final_df : pd.DataFrame
        DataFrame with module connectivity features and 'eid' column.
    cohort_df : pd.DataFrame
        DataFrame with covariate data and 'eid' column.
    covariate_columns : list of str
        Column names to extract from cohort_df (must include 'eid').
    general_dir : str
        Output directory for saving merged data and column mapping.
    output_basename : str, default='module_connectivity_features_with_covariates'
        Base name for output CSV files.

    Returns
    -------
    merged : pd.DataFrame
        Merged DataFrame with sanitized column names.
    colname_map : dict
        Mapping from original to sanitized column names.

    Notes
    -----
    Saves two CSV files:
    - {output_basename}.csv: merged data with sanitized columns
    - {output_basename}_colname_map.csv: mapping of original to sanitized names
    """
    # Create independent copies to avoid modifying original DataFrames
    final_df = final_df.copy()
    cohort_df = cohort_df.copy()
    
    # Ensure 'eid' column is string type in both DataFrames for consistent merging
    final_df['eid'] = final_df['eid'].astype(str)
    cohort_df['eid'] = cohort_df['eid'].astype(str)
    
    # Extract only the covariate columns we need from the cohort DataFrame
    covariates_df = cohort_df[covariate_columns].copy()
    
    # Merge module features with covariates on subject ID ('eid')
    # Left join ensures all subjects from final_df are retained
    merged = pd.merge(final_df, covariates_df, on='eid', how='left')

    # Sanitize column names to be safe for R and avoid special characters
    # This prevents issues when passing DataFrames to R via rpy2
    sanitized_cols, colname_map = _sanitize_colnames(merged.columns.tolist())
    merged.columns = sanitized_cols

    # Save the column name mapping for reference (original -> sanitized)
    map_df = pd.DataFrame(list(colname_map.items()), columns=['original', 'sanitized'])
    map_df.to_csv(os.path.join(general_dir, f'{output_basename}_colname_map.csv'), index=False)
    
    # Save the merged DataFrame with sanitized column names
    merged.to_csv(os.path.join(general_dir, f'{output_basename}.csv'), index=False)

    return merged, colname_map

def build_module_connectivity_dataframe(
    depressed_subjects_dir: str,
    control_subjects_dir: str,
    depressed_subject_ids: List[str],
    control_subject_ids: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a wide-format DataFrame of module connectivity features from subject CSVs.

    Parameters
    ----------
    depressed_subjects_dir : str
        Parent directory containing depressed subject folders ({eid}/i2/).
    control_subjects_dir : str
        Parent directory containing control subject folders ({eid}/i2/).
    depressed_subject_ids : list of str
        List of depressed subject IDs (eids).
    control_subject_ids : list of str
        List of control subject IDs (eids).

    Returns
    -------
    final_df : pd.DataFrame
        Wide-format DataFrame with columns: 'eid', 'depression_status', and
        module features formatted as '{module}_{direction}_{connectivity_type}'.
    modules : list of str
        Sorted list of unique module identifiers found in the data.

    Notes
    -----
    Expects three CSV files per subject in {eid}/i2/ directory:
    - functional_module_connectivity.csv
    - structural_module_connectivity.csv
    - module_sfc.csv
    Skips subjects missing any of these files.
    """
    # Initialize list to collect all subject rows
    all_rows = []
    
    # Process both cohorts (depressed and control) in a single loop
    # Each iteration processes one cohort with its specific directory and subject list
    for cohort, subjects_dir, subject_ids, depression_status in [
        ('F32', depressed_subjects_dir, depressed_subject_ids, 1),      # Depressed cohort
        ('control', control_subjects_dir, control_subject_ids, 0),     # Control cohort
    ]:
        print(f"Processing cohort {cohort} for combined module connectivity data...")
        
        # Loop through each subject ID in this cohort
        for subject_id in subject_ids:
            # Construct path to subject's connectivity data (in i2 subfolder)
            subject_dir = os.path.join(subjects_dir, subject_id, 'i2')
            
            # Define paths to the three required CSV files for this subject
            func_file = os.path.join(subject_dir, 'functional_module_connectivity.csv')
            struct_file = os.path.join(subject_dir, 'structural_module_connectivity.csv')
            sfc_file = os.path.join(subject_dir, 'module_sfc.csv')
            
            # Skip this subject if any required file is missing
            if not (os.path.exists(func_file) and os.path.exists(struct_file) and os.path.exists(sfc_file)):
                continue
            
            # Load the three CSV files containing module-level metrics
            func_df = pd.read_csv(func_file)      # Functional connectivity per module
            struct_df = pd.read_csv(struct_file)  # Structural connectivity per module
            sfc_df = pd.read_csv(sfc_file)        # Structure-function coupling per module

            # Initialize dictionary to store this subject's features
            row = {}
            
            # Extract functional connectivity features for each module
            # Each module gets internal and external connectivity columns
            for _, r in func_df.iterrows():
                mod = str(r['Module'])  # Module identifier (e.g., '0', '1', etc.)
                row[f"{mod}_internal_functional"] = r.get('Internal_Connectivity')
                row[f"{mod}_external_functional"] = r.get('External_Connectivity')
            
            # Extract structural connectivity features for each module
            for _, r in struct_df.iterrows():
                mod = str(r['Module'])
                row[f"{mod}_internal_structural"] = r.get('Internal_Connectivity')
                row[f"{mod}_external_structural"] = r.get('External_Connectivity')
            
            # Extract structure-function coupling features for each module
            for _, r in sfc_df.iterrows():
                mod = str(r['Module'])
                row[f"{mod}_internal_sfc"] = r.get('Internal_SFC')
                row[f"{mod}_external_sfc"] = r.get('External_SFC')

            # Add subject metadata to the row
            row['eid'] = subject_id                    # Subject ID
            row['depression_status'] = depression_status  # 0=control, 1=depressed
            all_rows.append(row)

    # Convert list of subject rows to DataFrame
    final_df = pd.DataFrame(all_rows)
    
    # Get all feature columns (excluding metadata columns)
    cols = [c for c in final_df.columns if c not in ['eid', 'depression_status']]
    
    # Extract unique module identifiers from column names
    # Sort modules numerically if possible, otherwise alphabetically
    mods = sorted(
        {col.split('_', 1)[0] for col in cols},  # Extract module ID (first part before '_')
        key=lambda v: (float(v) if _is_number(v) else v),  # Numeric sorting when possible
    )
    
    # Create ordered list of columns with consistent pattern:
    # For each module: internal_functional, external_functional, internal_structural, 
    # external_structural, internal_sfc, external_sfc
    ordered_cols = []
    for m in mods:
        ordered_cols.extend(
            [
                f"{m}_internal_functional",
                f"{m}_external_functional",
                f"{m}_internal_structural",
                f"{m}_external_structural",
                f"{m}_internal_sfc",
                f"{m}_external_sfc",
            ]
        )
    
    # Keep only columns that actually exist in the DataFrame
    # (handles cases where some modules might be missing certain features)
    ordered_cols = [c for c in ordered_cols if c in final_df.columns]
    
    # Reorder DataFrame: metadata columns first, then ordered feature columns
    final_df = final_df[['eid', 'depression_status'] + ordered_cols]
    
    return final_df, mods

# ==============================================================================
# CLUSTERING AND VALIDATION
# ==============================================================================
def bootstrap_clustering_stability(X, original_labels, n_boot=500, seed=12345):
    """Compute bootstrap-based clustering stability metrics.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_subjects, n_features)
        Feature matrix
    original_labels : np.ndarray, shape (n_subjects,)
        Original cluster labels (0-indexed)
    n_boot : int, default=500
        Number of bootstrap iterations
    seed : int, default=12345
        Random seed for reproducibility
    
    Returns
    -------
    dict with keys:
        - per_subject_stability : np.ndarray, per-subject stability proportion
        - jaccard_cluster0 : list, Jaccard similarities for cluster 0
        - jaccard_cluster1 : list, Jaccard similarities for cluster 1
        - nmi_list : list, NMI values per bootstrap
        - sample_counts : np.ndarray, how many times each subject was sampled
        - match_counts : np.ndarray, how many times each subject matched original label
    
    Notes
    -----
    Uses subject resampling with replacement. Labels are aligned (flipped if needed)
    to maximize agreement with original labels.
    """
    # Initialize random number generator with fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    n_subjects = X.shape[0]
    
    # Initialize arrays to track per-subject sampling and cluster assignment consistency
    sample_counts = np.zeros(n_subjects, dtype=int)  # How many times each subject was sampled
    match_counts = np.zeros(n_subjects, dtype=int)   # How many times cluster matched original
    
    # Initialize lists to store cluster similarity metrics across bootstrap iterations
    jaccard_cluster0 = []  # Jaccard similarity for cluster 0 (subject overlap)
    jaccard_cluster1 = []  # Jaccard similarity for cluster 1 (subject overlap)
    nmi_list = []          # Normalized Mutual Information scores
    
    def _jaccard(a, b):
        """Compute Jaccard similarity between two sets (intersection/union)."""
        # If both sets are empty, similarity is undefined
        if not a and not b:
            return np.nan
        # Count subjects in both sets (intersection)
        inter = len(a & b)
        # Count subjects in at least one set (union)
        union = len(a | b)
        # Jaccard index: proportion of union that is intersection
        return inter / union if union > 0 else np.nan
    
    # Main bootstrap loop: repeat clustering on resampled data n_boot times
    for b in range(n_boot):
        # Resample subjects with replacement (bootstrap sample)
        # Same subject can appear multiple times in bootstrap sample
        indices_b = rng.integers(0, n_subjects, size=n_subjects)
        X_b = X[indices_b]  # Extract feature matrix for resampled subjects
        
        # Perform hierarchical clustering on bootstrap sample
        Z_b = linkage(X_b, method='ward')  # Ward's method minimizes within-cluster variance
        labels_b = fcluster(Z_b, t=2, criterion='maxclust') - 1  # Force 2 clusters (0-indexed)
        
        # Skip this bootstrap iteration if clustering failed to produce 2 clusters
        if len(np.unique(labels_b)) < 2:
            nmi_list.append(np.nan)
            # Still count these subjects as sampled (even though clustering failed)
            for subj_idx in indices_b:
                sample_counts[subj_idx] += 1
            jaccard_cluster0.append(np.nan)
            jaccard_cluster1.append(np.nan)
            continue
        
        # === LABEL ALIGNMENT ===
        # Cluster labels (0/1) are arbitrary; align bootstrap labels to maximize
        # agreement with original labels (prevents label-switching bias)
        orig_sub_labels = original_labels[indices_b]  # Original labels for sampled subjects
        
        # Compute agreement with original labels (as-is)
        agree = np.mean(labels_b == orig_sub_labels)
        
        # Compute agreement if we flip labels (0->1, 1->0)
        agree_flipped = np.mean((1 - labels_b) == orig_sub_labels)
        
        # If flipping improves agreement, flip the bootstrap labels
        if agree_flipped > agree:
            labels_b = 1 - labels_b
        
        # === UPDATE STABILITY COUNTS ===
        # For each subject in the bootstrap sample, check if cluster matches original
        for pos, subj_idx in enumerate(indices_b):
            sample_counts[subj_idx] += 1  # Increment sampling count
            # If bootstrap cluster matches original cluster, increment match count
            if labels_b[pos] == original_labels[subj_idx]:
                match_counts[subj_idx] += 1
        
        # === COMPUTE JACCARD SIMILARITIES ===
        # Jaccard measures overlap between original and bootstrap cluster membership
        # Use unique indices to avoid counting duplicate subjects in bootstrap sample
        boot_cluster0 = set(np.unique(indices_b[labels_b == 0]))  # Bootstrap cluster 0 members
        boot_cluster1 = set(np.unique(indices_b[labels_b == 1]))  # Bootstrap cluster 1 members
        orig_cluster0 = set(np.where(original_labels == 0)[0])    # Original cluster 0 members
        orig_cluster1 = set(np.where(original_labels == 1)[0])    # Original cluster 1 members
        
        jaccard_cluster0.append(_jaccard(orig_cluster0, boot_cluster0))
        jaccard_cluster1.append(_jaccard(orig_cluster1, boot_cluster1))
        
        # === COMPUTE NORMALIZED MUTUAL INFORMATION (NMI) ===
        # NMI quantifies agreement between original and bootstrap cluster assignments
        # Range [0, 1]: 0=no agreement, 1=perfect agreement
        # Normalized by entropy to handle different cluster sizes
        try:
            nmi_val = float(normalized_mutual_info_score(orig_sub_labels, labels_b))
        except Exception:
            # If NMI computation fails (e.g., degenerate clustering), mark as NaN
            nmi_val = np.nan
        nmi_list.append(nmi_val)
    
    # === COMPUTE PER-SUBJECT STABILITY SCORES ===
    # Stability = proportion of times subject was assigned to its original cluster
    # across all bootstrap iterations where it was sampled
    stability_scores = np.full(n_subjects, np.nan)  # Initialize with NaN
    mask = sample_counts > 0  # Only compute for subjects that were sampled at least once
    
    # Stability score = match_count / sample_count (proportion of consistent assignments)
    stability_scores[mask] = match_counts[mask] / sample_counts[mask]
    
    # Return dictionary of all stability metrics
    return {
        'per_subject_stability': stability_scores,  # Per-subject stability proportion [0-1]
        'jaccard_cluster0': jaccard_cluster0,       # Cluster 0 Jaccard similarity per iteration
        'jaccard_cluster1': jaccard_cluster1,       # Cluster 1 Jaccard similarity per iteration
        'nmi_list': nmi_list,                       # NMI scores per bootstrap iteration
        'sample_counts': sample_counts,             # How many times each subject was sampled
        'match_counts': match_counts,               # How many times cluster matched original
        # Cluster label order for consistent plotting
        'cluster_label_order': ['0', '1'],
    }

def analyze_cross_modality_agreement(cohorts_dir, validation_plots_dir):
    """Analyze and visualize cluster agreement across connectivity modalities and directions.
    
    For depressed subjects, computes pairwise agreement (proportion in same cluster)
    across all available clustering solutions (functional/structural/SFC × internal/external).
    Generates:
    - CSV with per-subject cluster assignments across modalities
    - CSV with pairwise agreement proportions
    - CSV with cluster distribution by modality
    - Heatmap of cross-modality agreement
    - Bar plots showing cluster proportions per modality
    - Multi-panel bar plots overlaying subjects with identical labels across modality subsets
    
    Parameters
    ----------
    cohorts_dir : str
        Directory containing module_connectivity_features_with_clusters.csv.
    validation_plots_dir : str
        Output directory for saving plots and CSV summaries.
    
    Returns
    -------
    list of str
        List of available cluster column names that were analyzed.
    
    Notes
    -----
    - Only analyzes depressed subjects (depression_status == 1).
    - Cluster labels are automatically aligned (flipped if needed) to maximize agreement.
    - Requires at least 2 different clustering solutions to compute agreement.
    """
    # Construct path to CSV file containing cluster assignments for all modalities
    clusters_path = os.path.join(cohorts_dir, 'module_connectivity_features_with_clusters.csv')
    
    # Check if cluster file exists; return empty list if not found
    if not os.path.exists(clusters_path):
        print(f"  Missing modular cluster file: {clusters_path}")
        return []

    # Load cluster assignments and ensure subject IDs are strings
    clusters_df = pd.read_csv(clusters_path)
    clusters_df['eid'] = clusters_df['eid'].astype(str)
    
    # Filter to only depressed subjects (depression_status == 1)
    # Agreement analysis focuses on heterogeneity within the depressed group
    clusters_df = clusters_df[clusters_df['depression_status'] == 1].copy()

    # Define all possible cluster column names across modalities and directions
    cluster_cols = [
        'functional_internal_cluster', 'functional_external_cluster',
        'structural_internal_cluster', 'structural_external_cluster',
        'sfc_internal_cluster', 'sfc_external_cluster',
    ]
    
    # Identify which cluster columns are actually present in the data
    available_cols = [c for c in cluster_cols if c in clusters_df.columns]
    
    # Need at least 2 clustering solutions to compute pairwise agreement
    if len(available_cols) < 2:
        print("  Not enough modular cluster columns available for agreement analysis")
        return []

    print(f"  Found modular cluster columns: {', '.join(available_cols)}")
    print("  Computing pairwise agreement...")

    # Extract cluster assignments for all available modalities (one row per subject)
    assignments_df = clusters_df[['eid'] + available_cols].copy()
    
    # Save per-subject cluster assignments across all modalities to CSV
    assign_path = os.path.join(
        validation_plots_dir,
        'modular_connectivity_cluster_assignments_depression_across_types.csv',
    )
    assignments_df.to_csv(assign_path, index=False)
    print(f"  Saved per-subject assignments across types to: {assign_path}")

    def _to_numeric(series):
        """Convert cluster labels to numeric values, coercing errors to NaN."""
        return pd.to_numeric(series.astype(str), errors='coerce')

    def _normalize_cluster_label(value):
        """Normalize cluster labels to '0' or '1' strings, handling various formats."""
        # Return NaN for missing values
        if pd.isna(value):
            return np.nan
        
        # Convert to string and strip whitespace
        value_str = str(value).strip()
        
        # Try to extract numeric cluster ID using regex (handles 'Cluster 0', '0', etc.)
        m = re.search(r"(\d+)", value_str)
        if m:
            return m.group(1)
        
        # Try to convert to integer (handles float representations like '0.0')
        try:
            return str(int(float(value_str)))
        except Exception:
            # If all parsing fails, return NaN
            return np.nan

    # Initialize lists and matrices to store pairwise agreement results
    pair_results = []  # List of dictionaries with agreement statistics
    
    # Create agreement matrix (all modality pairs): initialize with NaN
    agreement_matrix = {ct: {ct2: np.nan for ct2 in available_cols} for ct in available_cols}
    
    # Diagonal elements are 1.0 (each modality perfectly agrees with itself)
    for ct in available_cols:
        agreement_matrix[ct][ct] = 1.0

    # Compute pairwise agreement for all unique pairs of modalities
    for i, col1 in enumerate(available_cols):
        # Only compute upper triangle (avoid duplicate comparisons)
        for col2 in available_cols[i + 1:]:
            # Extract subjects with valid cluster labels in both modalities
            merged = assignments_df[['eid', col1, col2]].dropna()
            
            # Skip if no subjects have both cluster assignments
            if merged.empty:
                print(f"    No overlapping subjects for {col1} vs {col2}; skipping.")
                continue

            # Convert cluster labels to numeric (0/1) and handle NaN
            c1 = _to_numeric(merged[col1]).to_numpy()
            c2 = _to_numeric(merged[col2]).to_numpy()
            
            # Filter to subjects with valid (non-NaN) cluster labels in both modalities
            valid = (~np.isnan(c1)) & (~np.isnan(c2))
            c1, c2 = c1[valid], c2[valid]

            # Skip if no valid cluster pairs after filtering
            if c1.size == 0:
                print(f"    {col1} vs {col2}: no valid cluster labels; skipping.")
                continue

            # Compute proportion of subjects in same cluster (as-is)
            agree = np.mean(c1 == c2)
            
            # Compute proportion if we flip one set of labels (0↔1)
            # This handles arbitrary cluster label assignments across modalities
            agree_flipped = np.mean(c1 == 1 - c2)
            
            # Use maximum agreement (accounts for label switching)
            prop_same = max(agree, agree_flipped)

            # Store pairwise agreement result
            pair_results.append({
                'type_a': col1,
                'type_b': col2,
                'proportion_same_cluster': prop_same,
                'n_subjects': int(c1.size),
            })

            # Fill agreement matrix symmetrically (upper and lower triangles)
            agreement_matrix[col1][col2] = prop_same
            agreement_matrix[col2][col1] = prop_same

            print(f"    {col1} vs {col2}: proportion same cluster = {prop_same:.3f} (n = {c1.size})")

    # Save pairwise agreement results to CSV
    agreement_path = os.path.join(
        validation_plots_dir,
        'modular_connectivity_cluster_agreement_across_types.csv',
    )
    pd.DataFrame(pair_results).to_csv(agreement_path, index=False)
    print(f"  Saved pairwise agreement summary to: {agreement_path}")

    # === COMPUTE CLUSTER SIZE DISTRIBUTION FOR EACH MODALITY ===
    # This shows how many subjects are in Cluster 0 vs Cluster 1 for each modality
    dist_rows = []
    for col in available_cols:
        # Normalize cluster labels to '0' or '1' strings
        normalized = assignments_df[col].apply(_normalize_cluster_label)
        
        # Count subjects in each cluster
        counts = normalized.value_counts()
        n_c0 = int(counts.get('0', 0))  # Number in Cluster 0
        n_c1 = int(counts.get('1', 0))  # Number in Cluster 1
        n_total = n_c0 + n_c1           # Total subjects with valid cluster labels
        
        # Compute proportions for each cluster
        prop_c0 = n_c0 / n_total if n_total > 0 else np.nan
        prop_c1 = n_c1 / n_total if n_total > 0 else np.nan
        
        # Store distribution statistics for this modality
        dist_rows.append({
            'connectivity_type': col,
            'n_total': n_total,
            'n_cluster_0': n_c0,
            'n_cluster_1': n_c1,
            'prop_cluster_0': prop_c0,
            'prop_cluster_1': prop_c1,
        })

    # Convert distribution statistics to DataFrame and save to CSV
    dist_df = pd.DataFrame(dist_rows)
    dist_csv_path = os.path.join(
        validation_plots_dir,
        'modular_connectivity_cluster_distribution_by_type.csv',
    )
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"  Saved cluster distribution table to: {dist_csv_path}")

    # === CREATE CLUSTER DISTRIBUTION BARPLOT ===
    # Shows proportion of subjects in Cluster 0 vs Cluster 1 for each modality
    plt.figure(figsize=(8, 5))
    
    # Reshape data from wide to long format for plotting
    plot_df = dist_df.melt(
        id_vars='connectivity_type',
        value_vars=['prop_cluster_0', 'prop_cluster_1'],
        var_name='cluster',
        value_name='proportion',
    )
    
    # Rename cluster labels for display
    plot_df['cluster'] = plot_df['cluster'].map({
        'prop_cluster_0': 'Cluster 0',
        'prop_cluster_1': 'Cluster 1',
    })
    
    # Set up x-axis positions for grouped bars
    order_types = list(plot_df['connectivity_type'].unique())
    x = np.arange(len(order_types))
    width = 0.35  # Width of each bar
    
    # Plot bars for each modality using modality-specific colors
    for i, ct in enumerate(order_types):
        # Extract connectivity type and direction from column name
        conn, direction = _parse_modality_from_label(ct)
        
        # Get color scheme for this modality
        colors = _cluster_colors_for_modality(conn or "functional", direction or "internal")
        
        # Extract cluster proportions for this modality
        c0 = float(plot_df[(plot_df['connectivity_type'] == ct) & (plot_df['cluster'] == 'Cluster 0')]['proportion'].iloc[0])
        c1 = float(plot_df[(plot_df['connectivity_type'] == ct) & (plot_df['cluster'] == 'Cluster 1')]['proportion'].iloc[0])
        
        # Plot bars (Cluster 0 on left, Cluster 1 on right)
        plt.bar(x[i] - width / 2, c0, width, color=colors["0"], alpha=0.9)
        plt.bar(x[i] + width / 2, c1, width, color=colors["1"], alpha=0.9)
    # Format barplot axes and labels
    plt.xticks(x, order_types, rotation=30, ha='right')
    plt.ylim(0, 1)  # Proportions range from 0 to 1
    plt.ylabel('Proportion of depressed subjects')
    plt.xlabel('Modality / direction')
    plt.title('Modular cluster membership distribution by type')
    plt.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Cluster')
    
    # Save distribution barplot to file
    dist_fig_path = os.path.join(
        validation_plots_dir,
        'modular_connectivity_cluster_distribution_by_type_barplot.png',
    )
    plt.savefig(dist_fig_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved cluster distribution barplot to: {dist_fig_path}")

    # === CREATE AGREEMENT HEATMAP ===
    # Shows pairwise cluster agreement between all modality combinations
    
    # Convert agreement matrix (dict of dicts) to DataFrame for heatmap
    mat_df = pd.DataFrame(agreement_matrix)
    mat_df = mat_df.reindex(index=available_cols, columns=available_cols)
    
    # Create heatmap with annotated values
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat_df, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='Blues')
    plt.title('Proportion of subjects in the same cluster\nacross modular cluster types')
    plt.ylabel('Modality / direction')
    plt.xlabel('Modality / direction')
    plt.tight_layout()
    
    # Save agreement heatmap to file
    agree_fig_path = os.path.join(
        validation_plots_dir,
        'modular_connectivity_cluster_agreement_across_types_heatmap.png',
    )
    plt.savefig(agree_fig_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved agreement heatmap to: {agree_fig_path}")

    # === IDENTICAL-LABEL OVERLAY ANALYSIS ===
    # This section analyzes subjects who have identical cluster labels across
    # multiple modalities (e.g., same cluster in functional AND structural)
    
    if assignments_df is not None:
        # Get cluster columns that are available in the assignments DataFrame
        cluster_cols = [c for c in assignments_df.columns if c in available_cols]

        # Need at least 2 modalities to find subjects with identical labels across them
        if len(cluster_cols) >= 2:
            def _build_identical_distribution(modalities):
                """Build distribution of subjects with identical vs different labels across modalities.
                
                Returns DataFrame with total counts and identical-label counts for each cluster.
                """
                # Need at least 2 modalities to compute "identical across modalities"
                if len(modalities) < 2:
                    return None
                
                # Extract column names for the specified modalities
                combo_cols = [ct for ct in modalities]
                
                # Check if all required columns exist in the data
                if any(col not in assignments_df.columns for col in combo_cols):
                    return None

                # Keep only subjects with valid cluster labels in ALL specified modalities
                subset = assignments_df.dropna(subset=combo_cols).copy()
                if subset.empty:
                    return None

                # Normalize cluster labels to '0' or '1' for all modalities
                for col in combo_cols:
                    subset[col] = subset[col].apply(_normalize_cluster_label)
                
                # Remove subjects with any NaN labels after normalization
                subset = subset.dropna(subset=combo_cols)
                if subset.empty:
                    return None

                # Identify subjects with IDENTICAL cluster labels across all modalities
                # nunique(axis=1) == 1 means all values in that row are the same
                identical_mask = subset[combo_cols].nunique(axis=1) == 1
                identical_subset = subset.loc[identical_mask].copy()

                # Build distribution table comparing total vs identical-label subjects
                rows = []
                for ct, col in zip(modalities, combo_cols):
                    # Count subjects in each cluster across ALL subjects
                    counts_total = subset[col].value_counts()
                    
                    # Count subjects in each cluster among IDENTICAL-label subjects only
                    counts_ident = identical_subset[col].value_counts()
                    
                    # Extract counts for Cluster 0 and Cluster 1
                    n_c0_total = int(counts_total.get('0', 0))
                    n_c1_total = int(counts_total.get('1', 0))
                    n_c0_ident = int(counts_ident.get('0', 0))
                    n_c1_ident = int(counts_ident.get('1', 0))
                    
                    # Create rows for each cluster (one for Cluster 0, one for Cluster 1)
                    rows.append({
                        'connectivity_type': ct,
                        'cluster': 'Cluster 0',
                        'n_total': n_c0_total,        # All subjects in Cluster 0
                        'n_identical': n_c0_ident,     # Subjects with identical labels across modalities
                    })
                    rows.append({
                        'connectivity_type': ct,
                        'cluster': 'Cluster 1',
                        'n_total': n_c1_total,        # All subjects in Cluster 1
                        'n_identical': n_c1_ident,     # Subjects with identical labels across modalities
                    })

                return pd.DataFrame(rows)

            def _plot_overlay(ax, modalities, title):
                """Create overlaid barplot showing total vs identical-label subject counts.
                
                Plots two sets of bars:
                - Light bars: all subjects in each cluster
                - Dark bars: subjects with identical labels across all specified modalities
                """
                # Check if all required modalities are available in the data
                missing = [ct for ct in modalities if ct not in assignments_df.columns]
                if missing:
                    # Display error message if modalities are missing
                    ax.text(0.5, 0.5, f"Missing modality data: {', '.join(missing)}",
                            ha='center', va='center')
                    ax.set_axis_off()
                    return

                # Build distribution data (total vs identical-label subjects)
                data = _build_identical_distribution(modalities)
                if data is None or data.empty:
                    # Display message if no subjects have identical labels
                    ax.text(0.5, 0.5, 'No overlapping subjects with identical labels',
                            ha='center', va='center')
                    ax.set_axis_off()
                    return

                # Filter to modalities that are actually available
                ordered_modalities = [ct for ct in modalities if ct in available_cols]
                if not ordered_modalities:
                    ax.text(0.5, 0.5, 'Modalities not available', ha='center', va='center')
                    ax.set_axis_off()
                    return

                # Set up bar positions
                x = np.arange(len(ordered_modalities))
                width = 0.35  # Bar width

                def _counts(cluster_label, field):
                    """Extract counts for a specific cluster and field across all modalities."""
                    vals = []
                    for ct in ordered_modalities:
                        # Filter data to this modality and cluster
                        subset = data[
                            (data['connectivity_type'] == ct) &
                            (data['cluster'] == cluster_label)
                        ]
                        if subset.empty:
                            vals.append(0)
                        else:
                            vals.append(int(subset[field].iloc[0]))
                    return np.asarray(vals, dtype=int)

                # Extract counts for both clusters (total and identical)
                totals0 = _counts('Cluster 0', 'n_total')       # All Cluster 0 subjects
                totals1 = _counts('Cluster 1', 'n_total')       # All Cluster 1 subjects
                ident0 = _counts('Cluster 0', 'n_identical')    # Identical-label Cluster 0
                ident1 = _counts('Cluster 1', 'n_identical')    # Identical-label Cluster 1

                # Plot overlaid bars for each modality
                for i, ct in enumerate(ordered_modalities):
                    # Get modality-specific colors
                    conn, direction = _parse_modality_from_label(ct)
                    colors = _cluster_colors_for_modality(conn or "functional", direction or "internal")
                    
                    # Plot light bars (all subjects) - wider and semi-transparent
                    ax.bar(x[i] - width / 2, totals0[i], width, color=colors["0"], alpha=0.3)
                    ax.bar(x[i] + width / 2, totals1[i], width, color=colors["1"], alpha=0.3)
                    
                    # Plot dark bars (identical-label subjects) - narrower and opaque, overlaid on top
                    ax.bar(x[i] - width / 2, ident0[i], width * 0.6, color=colors["0"], alpha=0.9)
                    ax.bar(x[i] + width / 2, ident1[i], width * 0.6, color=colors["1"], alpha=0.9)

                # Format plot axes and labels
                ax.set_xticks(x)
                ax.set_xticklabels(ordered_modalities, rotation=30, ha='right')
                ax.set_ylabel('Subjects')
                ax.set_xlabel('')
                ax.set_title(textwrap.fill(title, width=40))  # Wrap long titles
                ax.grid(alpha=0.3, axis='y')
                
                # Add legend explaining the two bar types
                legend_handles = [
                    Patch(facecolor='black', alpha=0.3, label='All subjects'),
                    Patch(facecolor='black', alpha=0.9, label='Identical label subjects'),
                ]
                ax.legend(handles=legend_handles, fontsize=7)

            # === SAVE IDENTICAL-LABEL DISTRIBUTION TO CSV ===
            # Compute distribution for all available modalities combined
            modalities_for_csv = [ct for ct in available_cols if ct in assignments_df.columns]
            csv_df = _build_identical_distribution(modalities_for_csv)
            
            if csv_df is not None and not csv_df.empty:
                # Save CSV with total vs identical-label counts for each modality
                dist_ident_csv_path = os.path.join(
                    validation_plots_dir,
                    'modular_connectivity_cluster_distribution_with_identical_across_types.csv',
                )
                csv_df.to_csv(dist_ident_csv_path, index=False)
                print(
                    "  Saved distribution with identical-label subset to: "
                    f"{dist_ident_csv_path}"
                )
            else:
                # No subjects have complete labels across all modalities
                print(
                    "  No subjects with complete labels across available modalities; "
                    "skipping identical-label distribution CSV."
                )

            # === CREATE MULTI-PANEL OVERLAY PLOTS ===
            # Generate one panel for each combination of modalities (all sizes from k=2 to k=max)
            # This shows which subjects are consistent across different modality subsets
            
            combo_specs = []  # List of (modality_list, title) tuples
            
            # Iterate from largest combination (all modalities) down to pairs
            for k in range(len(available_cols), 1, -1):
                # Generate all k-sized combinations of modalities
                for mods in itertools.combinations(available_cols, k):
                    title = " + ".join(mods)  # Panel title shows which modalities are combined
                    combo_specs.append((list(mods), title))

            # Set up multi-panel figure layout
            n_panels = max(1, len(combo_specs))
            ncols = 3  # 3 columns of panels
            nrows = int(math.ceil(n_panels / ncols))  # Calculate required rows
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.8))
            axes = np.atleast_2d(axes)  # Ensure axes is always 2D array

            # Plot each modality combination in its own panel
            for ax, (mods, title) in zip(axes.flat, combo_specs):
                _plot_overlay(ax, mods, title)

            # Turn off unused panels (if number of combinations doesn't fill the grid)
            for ax in axes.flat[len(combo_specs):]:
                ax.set_axis_off()

            # Adjust spacing between panels
            fig.subplots_adjust(hspace=1.6, wspace=0.25)

            # Add column headers at the top of each column
            for col_idx in range(ncols):
                ax_col = axes[0, col_idx]  # Top panel in this column
                bbox = ax_col.get_position()  # Get panel position
                x_center = (bbox.x0 + bbox.x1) / 2.0  # Center of panel
                
                # Add text header above the panel
                fig.text(
                    x_center,
                    0.98,
                    'Modality / direction',
                    ha='center',
                    va='top',
                    fontsize=10,
                    fontweight='bold',
                )
            
            # Save the multi-panel figure
            ident_fig_path = os.path.join(
                validation_plots_dir,
                'modular_connectivity_cluster_distribution_with_identical_across_types_barplot.png',
            )
            fig.savefig(ident_fig_path, dpi=300, bbox_inches='tight', format='png')
            plt.close(fig)
            print(f"  Saved multi-panel identical-label overlay plot to: {ident_fig_path}")
        else:
            # Not enough modalities to compute identical-label analysis
            print("  Not enough modalities with cluster labels for identical-label overlay.")

    # Return list of available cluster column names for downstream use
    return available_cols

def perform_module_hierarchical_clustering(
    final_df: pd.DataFrame,
    plots_dir: str,
    figures_dir: str,
    conn_types: Tuple[str, ...] = ('functional', 'structural', 'sfc'),
    dir_types: Tuple[str, ...] = ('internal', 'external'),
    n_clusters: int = 2,
    bootstrap_iter: int = 500,
) -> pd.DataFrame:
    """Perform hierarchical clustering on depressed subjects for each modality and direction.
    
    Applies Ward's linkage clustering to module connectivity features separately for each
    (connectivity type, direction) combination. For each clustering solution:
    - Generates dendrogram visualization
    - Computes bootstrap stability metrics
    - Creates diagnostic plots
    - Adds cluster labels to final_df
    
    Parameters
    ----------
    final_df : pd.DataFrame
        DataFrame with 'eid', 'depression_status', and module feature columns.
        Must contain columns matching pattern '{module}_{dir_type}_{conn_type}'.
    plots_dir : str
        Directory for saving bootstrap diagnostic plots.
    figures_dir : str
        Directory for saving dendrograms.
    conn_types : tuple of str, default=('functional', 'structural', 'sfc')
        Connectivity types to cluster.
    dir_types : tuple of str, default=('internal', 'external')
        Direction types to cluster.
    n_clusters : int, default=2
        Number of clusters to extract from dendrogram.
    bootstrap_iter : int, default=500
        Number of bootstrap iterations for stability analysis.
    
    Returns
    -------
    pd.DataFrame
        Updated final_df with new columns: '{conn_type}_{dir_type}_cluster' containing
        cluster labels ('0', '1', or 'Control').
    
    Notes
    -----
    - Only depressed subjects (depression_status == 1) are clustered.
    - Control subjects receive cluster label 'Control'.
    - Bootstrap stability quantifies how consistently each subject is assigned to
      the same cluster across resampled datasets.
    """
    # Extract feature matrix for depressed subjects only (exclude control subjects)
    # Controls are not clustered; they serve as a comparison group
    feature_df = final_df.loc[final_df['depression_status'] == 1].drop(
        columns=['eid', 'depression_status']  # Remove non-feature columns
    ).copy()

    # Perform clustering separately for each (connectivity_type, direction) combination
    # This allows detection of modality-specific subtypes
    for conn_type in conn_types:
        for dir_type in dir_types:
            # Select feature columns matching this connectivity type and direction
            # E.g., for functional + internal: all columns like '{module}_internal_functional'
            selected_cols = _select_module_columns(feature_df, conn_type, dir_type)
            
            # Skip if no matching columns found (e.g., missing data for this modality)
            if not selected_cols:
                print(f"Skipping clustering for {conn_type} {dir_type}: no columns found.")
                continue
            
            # Extract feature matrix (subjects × modules)
            X = feature_df[selected_cols].values
            
            # Need at least 2 subjects to perform clustering
            if X.shape[0] < 2:
                print(f"Skipping clustering for {conn_type} {dir_type}: not enough subjects.")
                continue

            # === HIERARCHICAL CLUSTERING ===
            # Perform Ward's linkage clustering (minimizes within-cluster variance)
            Z = linkage(X, method='ward')

            # === GENERATE DENDROGRAM ===
            # Visualize hierarchical clustering tree structure
            plt.figure(figsize=(10, 6))
            dendrogram(Z)
            plt.title(
                f'Hierarchical Clustering Dendrogram (Ward Linkage) - '
                f'{_display_conn_type(conn_type)} {dir_type.capitalize()} Connectivity'
            )
            plt.ylabel('Distance')  # Y-axis shows linkage distance (dissimilarity)
            plt.tight_layout()
            
            # Save dendrogram to connectivity-specific subdirectory
            os.makedirs(os.path.join(figures_dir, f'{conn_type}_con'), exist_ok=True)
            plt.savefig(
                os.path.join(
                    figures_dir,
                    f'{conn_type}_con',
                    f'{dir_type}_connectivity_dendrogram.png',
                ),
                dpi=300,
                bbox_inches='tight',
                format='png',
            )
            plt.close()

            # === EXTRACT CLUSTER LABELS ===
            # Cut dendrogram to get n_clusters (default=2) cluster assignments
            # fcluster returns 1-indexed labels; subtract 1 to get 0-indexed
            cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
            
            # Add cluster labels to final_df
            # Depressed subjects: assign cluster label ('0' or '1' as strings)
            final_df.loc[final_df['depression_status'] == 1, f'{conn_type}_{dir_type}_cluster'] = (
                cluster_labels.astype(str)
            )
            
            # Control subjects: label as 'Control' (not clustered)
            final_df.loc[final_df['depression_status'] == 0, f'{conn_type}_{dir_type}_cluster'] = 'Control'

            # === BOOTSTRAP STABILITY ANALYSIS ===
            # Assess how reproducible the cluster assignments are via resampling
            stability_results = bootstrap_clustering_stability(X, cluster_labels, n_boot=bootstrap_iter)
            
            # Generate diagnostic plots showing stability metrics
            plot_bootstrap_diagnostics(
                stability_results,
                cluster_labels,
                conn_type,
                os.path.join(plots_dir, f'{conn_type}_con'),  # Save to modality-specific dir
                analysis_level='modular',
                dir_type=dir_type,
            )

    # Return DataFrame with added cluster label columns
    return final_df


def compute_clustering_validation(
    final_df: pd.DataFrame,
    conn_types: Tuple[str, ...] = ('functional', 'structural', 'sfc'),
    dir_types: Tuple[str, ...] = ('internal', 'external'),
    k_min: int = 2,
    k_max: int = 20,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Compute silhouette and Calinski-Harabasz scores across cluster counts.

    For each (connectivity_type, direction_type) combination, performs hierarchical
    clustering with k clusters for k in [k_min, k_max] and computes validation metrics.

    Parameters
    ----------
    final_df : pd.DataFrame
        DataFrame with 'eid', 'depression_status', and module feature columns.
    conn_types : tuple of str, default=('functional', 'structural', 'sfc')
        Connectivity types to evaluate.
    dir_types : tuple of str, default=('internal', 'external')
        Direction types to evaluate.
    k_min : int, default=2
        Minimum number of clusters to test.
    k_max : int, default=20
        Maximum number of clusters to test.

    Returns
    -------
    validation_df : pd.DataFrame
        DataFrame with columns: connectivity_type, direction_type, num_clusters,
        silhouette_score, calinski_harabasz_score.
    summary : list of dict
        Summary with best k values for each metric and modality.

    Notes
    -----
    - Only depressed subjects (depression_status == 1) are clustered.
    - Uses Ward's linkage method for hierarchical clustering.
    - Silhouette score: measures cluster cohesion and separation (higher is better).
    - Calinski-Harabasz score: ratio of between-cluster to within-cluster variance (higher is better).
    """
    # === PREPARE FEATURE MATRIX ===
    # Identify columns to drop: subject ID, diagnosis, and any existing cluster labels
    drop_cols = ['eid', 'depression_status'] + [
        f'{conn}_{dir_type}_cluster'
        for conn in conn_types
        for dir_type in dir_types
    ]
    
    # Extract depressed subjects only (controls are not clustered)
    # Drop non-feature columns to get pure connectivity feature matrix
    feature_df = final_df.loc[final_df['depression_status'] == 1].drop(
        columns=[c for c in drop_cols if c in final_df.columns]
    ).copy()
    
    # Store validation metrics for all configurations tested
    validation_results = []
    # Store best cluster numbers for each (connectivity, direction) combination
    summary = []

    # === ITERATE OVER MODALITIES ===
    # Test cluster validation across all connectivity types and directions
    for conn_type in conn_types:
        for dir_type in dir_types:
            # Select feature columns matching this connectivity type and direction
            selected_cols = _select_module_columns(feature_df, conn_type, dir_type)
            
            # Skip if no matching columns found
            if not selected_cols:
                continue
            
            # Extract feature matrix (subjects × modules)
            X = feature_df[selected_cols].values
            
            # Need at least 2 subjects to perform clustering
            if X.shape[0] < 2:
                continue
            
            # === TEST MULTIPLE CLUSTER NUMBERS ===
            # For each k from k_min to k_max, compute validation metrics
            # This helps determine the optimal number of clusters
            for k in range(k_min, k_max + 1):
                # Perform Ward's linkage hierarchical clustering
                Z = linkage(X, method='ward')
                
                # Cut dendrogram to get k cluster assignments
                cluster_labels = fcluster(Z, t=k, criterion='maxclust')
                
                # === COMPUTE VALIDATION METRICS ===
                # Silhouette score: measures how similar subjects are to their own cluster
                # vs. other clusters. Range [-1, 1], higher is better.
                sil_score = silhouette_score(X, cluster_labels)
                
                # Calinski-Harabasz score: ratio of between-cluster to within-cluster variance
                # Higher score indicates better-defined clusters
                ch_score = calinski_harabasz_score(X, cluster_labels)
                
                # Store results for this configuration
                validation_results.append({
                    'connectivity_type': conn_type,
                    'direction_type': dir_type,
                    'num_clusters': k,
                    'silhouette_score': sil_score,
                    'calinski_harabasz_score': ch_score,
                })

            # === FIND BEST CLUSTER NUMBERS ===
            # For this modality, identify k values that maximize each validation metric
            # Filter results to this specific (connectivity_type, direction_type)
            modality_results = [
                r for r in validation_results 
                if r['connectivity_type'] == conn_type and r['direction_type'] == dir_type
            ]
            
            # Find k with highest silhouette score
            best_k_sil = max(modality_results, key=lambda x: x['silhouette_score'])
            
            # Find k with highest Calinski-Harabasz score
            best_k_ch = max(modality_results, key=lambda x: x['calinski_harabasz_score'])
            
            # Store summary of best configurations for this modality
            summary.append({
                'connectivity_type': conn_type,
                'direction_type': dir_type,
                'best_k_silhouette': best_k_sil['num_clusters'],
                'best_silhouette_score': best_k_sil['silhouette_score'],
                'best_k_calinski_harabasz': best_k_ch['num_clusters'],
                'best_calinski_harabasz_score': best_k_ch['calinski_harabasz_score'],
            })

    # Return full validation results as DataFrame + summary of best k values
    return pd.DataFrame(validation_results), summary

# ==============================================================================
# QUANTILE REGRESSION
# ==============================================================================
def run_quantile_regression(combined_df, conn_type, icd_covariates,
                            motion_covariates=None, tau=0.5, R=10000, 
                            dependent_var='Connectivity', cluster_col='Cluster',
                            group_col='Group'):
    """Run quantile regression models in R and extract p-values.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined dataframe with all relevant variables
    conn_type : str
        Connectivity type ('functional', 'structural', or Structure-Function Coupling)
    motion_covariates : list of str, optional
        Column names for head motion covariates to include. Defaults to the
        modality-specific column used elsewhere if not provided.
    tau : float, default=0.5
        Quantile to estimate (e.g., 0.5 for median)
    R : int, default=10000
        Number of bootstrap replications for standard errors
    icd_covariates : list of str
        List of ICD-10 code covariate column names (comorbidities)
    dependent_var : str, default='Connectivity'
        Name of the dependent variable column in combined_df
    cluster_col : str, default='Cluster'
        Name of the cluster label column in combined_df
    group_col : str, default='Group'
        Name of the group label column in combined_df
    
    Returns
    -------
    dict
        Dictionary with p-values for each contrast
    """
    # === SETUP MOTION COVARIATES ===
    # If motion covariates not specified, use modality-specific defaults
    motion_covariates = list(motion_covariates or [])
    if not motion_covariates:
        # Functional connectivity: use fMRI head motion covariate (p24441_i2)
        if str(conn_type).lower().startswith('functional'):
            motion_covariates = ['p24441_i2']
        # Structural/SFC: use dMRI head motion covariate (p24453_i2)
        else:
            motion_covariates = ['p24453_i2']

    # === HANDLE MISSING COMORBIDITY DATA ===
    # Controls do not have ICD-10 codes in the dataset
    # Fill missing values with 0 (no diagnosis) to ensure consistent 0/1 coding
    # This prevents R from encountering NA values in categorical predictors
    for cov in icd_covariates or []:
        if cov in combined_df.columns:
            # Convert to numeric and replace NaN with 0
            combined_df[cov] = pd.to_numeric(combined_df[cov], errors='coerce').fillna(0)

    # === EXPORT DATA TO CSV FOR R ===
    # Save DataFrame to temporary file for R to read
    combined_df.to_csv('/tmp/combined_data.csv', index=False)
    
    # === EXECUTE QUANTILE REGRESSION IN R ===
    # Use localconverter to enable automatic pandas<->R dataframe conversion
    with localconverter(default_converter + pandas2ri.converter):
        # Transfer Python variables to R global environment
        # These variables will be accessible inside the R script block below
        
        # Pass connectivity type to R (e.g., 'functional', 'structural', 'sfc')
        ro.globalenv['conn_type'] = ro.StrVector([str(conn_type)])
        
        # Pass dependent variable name (column to model, e.g., 'Connectivity')
        ro.globalenv['dependent_var'] = ro.StrVector([str(dependent_var)])
        
        # Pass group column name (distinguishes controls vs. depressed subjects)
        ro.globalenv['group_col'] = ro.StrVector([str(group_col)])
        
        # Pass cluster column name (distinguishes Cluster 0 vs. Cluster 1 vs. Control)
        ro.globalenv['cluster_col'] = ro.StrVector([str(cluster_col)])
        
        # Pass ICD-10 comorbidity covariate column names (list of strings)
        ro.globalenv['icd_covariates'] = ro.StrVector([str(cov) for cov in icd_covariates])
        
        # Pass motion covariate column names (head motion regressors)
        ro.globalenv['motion_cols'] = ro.StrVector([str(col) for col in motion_covariates])
        
        # Pass quantile to estimate (tau=0.5 for median regression)
        ro.globalenv['tau'] = tau
        
        # Pass number of bootstrap replications for standard errors
        ro.globalenv['R'] = R

        # === R SCRIPT FOR QUANTILE REGRESSION ===
        # This R code block performs three quantile regression models with bootstrapped SEs:
        #   Model 1: Depression vs Control
        #   Model 2: Cluster 0 vs Control AND Cluster 1 vs Control
        #   Model 3: Cluster 0 vs Cluster 1
        ro.r(r'''
        # Load required R packages
        library(quantreg)  # For quantile regression (rq function)
        library(multcomp)  # For multiple comparisons (not used directly here)
        
        # Set random seed for reproducible bootstrap sampling
        set.seed(123)
             
        # Extract connectivity type as single string
        conn_type <- as.character(conn_type[[1]])
        
        # Load data from CSV file exported from Python
        data <- read.csv("/tmp/combined_data.csv")
        
        # Validate that required columns exist in dataset
        if (!(group_col %in% colnames(data))) {
            stop(paste0("group_col not found in data: ", group_col))
        }
        if (!(cluster_col %in% colnames(data))) {
            stop(paste0("cluster_col not found in data: ", cluster_col))
        }
             
        # === CONVERT VARIABLES TO FACTORS ===
        # Sex (p31) should be categorical (0=female, 1=male)
        data$p31 <- factor(data$p31)
             
        # === CONVERT GROUP LABELS TO TEXT ===
        # If group column contains numeric 0/1, convert to "Control"/"Depression"
        # This makes model output more interpretable
        if (1 %in% unique(data[[group_col]]) & 0 %in% unique(data[[group_col]])) {
            data[[group_col]] <- ifelse(data[[group_col]] == 1, "Depression", "Control")
        }
        # Convert to factor for regression modeling
        data[[group_col]] <- factor(data[[group_col]])

        # === CONVERT CLUSTER LABELS TO TEXT ===
        # Cluster column may contain mix of numeric (0, 1) and string ("Control")
        # Map numeric 0/1 to "Cluster 0"/"Cluster 1", preserve "Control" as-is
        vals <- data[[cluster_col]]
        
        # Detect which entries are numeric 0 or 1 (handles factors/numerics/character-coded numbers)
        # suppressWarnings prevents coercion warnings for non-numeric values like "Control"
        is_num01 <- suppressWarnings(!is.na(as.numeric(as.character(vals))) & as.numeric(as.character(vals)) %in% c(0,1))
        
        # Convert all to character, then remap numeric 0/1 to cluster labels
        vals_mapped <- as.character(vals)
        vals_mapped[is_num01] <- ifelse(as.numeric(as.character(vals[is_num01])) == 0, "Cluster 0", "Cluster 1")
        
        # Convert to factor for regression modeling
        data[[cluster_col]] <- factor(vals_mapped)

        # === ENSURE ICD COMORBIDITY COVARIATES ARE BINARY FACTORS ===
        # ICD codes should be 0 (no diagnosis) or 1 (diagnosis present)
        # Controls have 0s; depressed subjects may have 1s
        for (cov in icd_covariates) {
            if (cov %in% colnames(data)) {
                # Replace any remaining NAs with 0 (no diagnosis)
                data[[cov]][is.na(data[[cov]])] <- 0
                # Convert to factor (R will create dummy variables in model matrix)
                data[[cov]] <- factor(data[[cov]])
            }
        }
             
        # === CENTER AGE COVARIATE ===
        # Centering improves interpretability: intercept represents outcome at median age
        # rather than at age=0 (which is nonsensical)
        age_median <- median(data$p21003_i2, na.rm = TRUE)
        data$age_centered <- data$p21003_i2 - age_median
        
        # === CENTER MOTION COVARIATES ===
        # Head motion is a critical confound in neuroimaging
        # Centering motion covariates around median improves numerical stability
        motion_cols <- as.character(motion_cols)
        
        # Only include motion columns that actually exist in the dataset
        motion_cols <- motion_cols[motion_cols %in% colnames(data)]
        
        # Warn if no motion covariates found (though analysis continues)
        if (length(motion_cols) == 0) {
            warning("No motion covariates found in dataset for quantile regression; proceeding without motion covariates")
        }
        
        # Create centered versions of each motion covariate
        motion_terms <- c()  # Will store names of centered motion columns
        for (col in motion_cols) {
            # Create new column name (e.g., "p24441_i2_centered")
            centered_name <- paste0(col, "_centered")
            # Center around median
            med_val <- median(data[[col]], na.rm = TRUE)
            data[[centered_name]] <- data[[col]] - med_val
            # Add to list of motion terms to include in regression formula
            motion_terms <- c(motion_terms, centered_name)
        }

        # === FILTER ICD COVARIATES TO THOSE IN DATASET ===
        # Only include comorbidity covariates that actually exist in the data
        icd_terms <- as.character(icd_covariates)
        icd_terms <- icd_terms[icd_terms %in% colnames(data)]
             
        # =====================================================================
        # MODEL 1: DEPRESSION vs CONTROL
        # =====================================================================
        # Test whether depressed subjects differ from controls on connectivity
        # Covariates: age, sex, motion, comorbidities
        print("========== MODEL 1: Depression vs Control ==========")
        print(paste0("dependent_var: ", dependent_var))
        
        # === SUBSET DATA ===
        # Include only Depression and Control subjects (exclude cluster labels temporarily)
        data_dep_ctrl <- data[data[[group_col]] %in% c("Depression", "Control"), ]
        
        # Drop unused factor levels (e.g., if data originally had other groups)
        data_dep_ctrl[[group_col]] <- droplevels(factor(data_dep_ctrl[[group_col]]))
        
        # Set "Control" as reference level so Depression coefficient represents Depression vs Control
        data_dep_ctrl[[group_col]] <- relevel(data_dep_ctrl[[group_col]], ref = "Control")

        # === BUILD MODEL FORMULA ===
        # Right-hand side predictors: group, age, sex, motion, comorbidities
        rhs_dep_ctrl <- c(group_col, "age_centered", "p31", motion_terms, icd_terms)
             
        # Wrap variable names in backticks to handle non-syntactic names
        # (e.g., column names starting with digits like p24441_i2)
        rhs_dep_ctrl_safe <- paste0("`", rhs_dep_ctrl, "`")
        fml_dep_ctrl <- as.formula(paste0("`", dependent_var, "` ~ ", paste(rhs_dep_ctrl_safe, collapse = " + ")))

        # === CONSTRUCT MODEL MATRIX ===
        # Extract design matrix (X) and outcome vector (y) for bootstrap
        X_dep_ctrl <- model.matrix(fml_dep_ctrl, data = data_dep_ctrl)
        y_dep_ctrl <- data_dep_ctrl[[dependent_var]]

        # === FIT QUANTILE REGRESSION MODEL ===
        # rq() estimates the conditional quantile (default tau=0.5 for median)
        model_dep_ctrl <- rq(fml_dep_ctrl, data = data_dep_ctrl, tau = tau)
        
        # Compute bootstrapped standard errors (R bootstrap replications)
        summary_dep_ctrl <- summary.rq(model_dep_ctrl, se = "boot", R = R)
        print(summary_dep_ctrl)
        
        # === COMPUTE BOOTSTRAP CONFIDENCE INTERVALS ===
        # boot.rq() performs paired (x,y) bootstrap to estimate coefficient distribution
        boot_out_dep_ctrl <- boot.rq(x = X_dep_ctrl, y = y_dep_ctrl, tau = tau, R = R)
        
        # Extract 95% CI from bootstrap distribution (2.5th and 97.5th percentiles)
        cis_dep_ctrl <- apply(boot_out_dep_ctrl$B, 2, quantile, probs = c(0.025, 0.975))
        print("95% CI for coefficients:")
        print(cis_dep_ctrl)
        
        # === EXTRACT P-VALUE FOR DEPRESSION vs CONTROL ===
        # Get coefficient table from summary
        coef_table <- summary_dep_ctrl$coefficients
        coef_rownames <- rownames(coef_table)
        
        # Remove backticks from rownames for matching
        coef_rownames_clean <- gsub("`", "", coef_rownames)
        
        # Helper function to extract p-value for a given term
        get_p <- function(term_name) {
            # Find index of matching coefficient
            idx <- which(coef_rownames_clean == term_name)
            if (length(idx) == 0) {
                warning(paste0("Coefficient not found for term: ", term_name))
                return(NA_real_)
            }
            # Extract p-value from coefficient table
            pval <- coef_table[idx[1], "Pr(>|t|)"]
            return(pval)
        }

        # Extract p-value for Depression vs Control comparison
        p_depression_vs_control <- get_p(paste0(group_col, "Depression"))
        
        # Handle numerical underflow: R reports p-values < 2.2e-16 as exact 0
        # Clamp to minimum representable p-value to avoid downstream issues
        p_depression_vs_control <- ifelse(is.na(p_depression_vs_control), NA_real_, ifelse(p_depression_vs_control < 2.2e-16, 2.2e-16, p_depression_vs_control))
        
        # =====================================================================
        # MODEL 2: CLUSTER 0 vs CONTROL AND CLUSTER 1 vs CONTROL
        # =====================================================================
        # Test whether each depression cluster differs from controls
        # This model has three groups: Control (reference), Cluster 0, Cluster 1
        print("========== MODEL 2: Clusters vs Control ==========")
        print(paste0("dependent_var: ", dependent_var))
        
        # === SUBSET DATA ===
        # Include all three groups: Cluster 0, Cluster 1, Control
        data_clusters_ctrl <- data[data[[cluster_col]] %in% c("Cluster 0", "Cluster 1", "Control"), ]
        
        # Drop unused factor levels
        data_clusters_ctrl[[cluster_col]] <- droplevels(factor(data_clusters_ctrl[[cluster_col]]))
        
        # Set "Control" as reference level
        # Cluster 0 and Cluster 1 coefficients will represent differences from Control
        data_clusters_ctrl[[cluster_col]] <- relevel(data_clusters_ctrl[[cluster_col]], ref = "Control")

        # === BUILD MODEL FORMULA ===
        # Right-hand side: cluster labels, age, sex, motion, comorbidities
        rhs_clusters_ctrl <- c(cluster_col, "age_centered", "p31", motion_terms, icd_terms)
        
        # Wrap variable names in backticks for safety
        rhs_clusters_ctrl_safe <- paste0("`", rhs_clusters_ctrl, "`")
        fml_clusters_ctrl <- as.formula(paste0("`", dependent_var, "` ~ ", paste(rhs_clusters_ctrl_safe, collapse = " + ")))

        # === CONSTRUCT MODEL MATRIX ===
        X_clusters_ctrl <- model.matrix(fml_clusters_ctrl, data = data_clusters_ctrl)
        y_clusters_ctrl <- data_clusters_ctrl[[dependent_var]]

        # === FIT QUANTILE REGRESSION MODEL ===
        model_all <- rq(fml_clusters_ctrl, data = data_clusters_ctrl, tau = tau)
        
        # Compute bootstrapped standard errors
        summary_clusters_ctrl <- summary.rq(model_all, se = "boot", R = R)
        print(summary_clusters_ctrl)
        
        # === COMPUTE BOOTSTRAP CONFIDENCE INTERVALS ===
        boot_out_clusters <- boot.rq(x = X_clusters_ctrl, y = y_clusters_ctrl, tau = tau, R = R)
        
        # Extract 95% CIs for all coefficients
        cis_coef <- apply(boot_out_clusters$B, 2, quantile, probs = c(0.025, 0.975))
        print("95% CI for coefficients:")
        print(cis_coef)
        
        # === EXTRACT P-VALUES FOR CLUSTER COMPARISONS ===
        # Get coefficient table
        coef_table <- summary_clusters_ctrl$coefficients
        coef_rownames <- rownames(coef_table)
        coef_rownames_clean <- gsub("`", "", coef_rownames)
        
        # Helper function to extract p-values (same as Model 1)
        get_p <- function(term_name) {
            idx <- which(coef_rownames_clean == term_name)
            if (length(idx) == 0) {
                warning(paste0("Coefficient not found for term: ", term_name))
                return(NA_real_)
            }
            pval <- coef_table[idx[1], "Pr(>|t|)"]
            return(pval)
        }

        # Extract p-value for Cluster 0 vs Control
        p_cluster0_vs_control <- get_p(paste0(cluster_col, "Cluster 0"))
        p_cluster0_vs_control <- ifelse(is.na(p_cluster0_vs_control), NA_real_, ifelse(p_cluster0_vs_control < 2.2e-16, 2.2e-16, p_cluster0_vs_control))
        
        # Extract p-value for Cluster 1 vs Control
        p_cluster1_vs_control <- get_p(paste0(cluster_col, "Cluster 1"))
        p_cluster1_vs_control <- ifelse(is.na(p_cluster1_vs_control), NA_real_, ifelse(p_cluster1_vs_control < 2.2e-16, 2.2e-16, p_cluster1_vs_control))

        # =====================================================================
        # MODEL 3: CLUSTER 0 vs CLUSTER 1
        # =====================================================================
        # Direct comparison between the two depression clusters
        # Tests whether clusters differ significantly on connectivity
        print("========== MODEL 3: Cluster 0 vs Cluster 1 ==========")
        print(paste0("dependent_var: ", dependent_var))
        
        # === SUBSET DATA ===
        # Include only subjects in Cluster 0 or Cluster 1 (exclude controls)
        data_cluster0_1 <- data[data[[cluster_col]] %in% c("Cluster 0", "Cluster 1"), ]
        
        # Drop unused factor levels (i.e., "Control")
        data_cluster0_1[[cluster_col]] <- droplevels(factor(data_cluster0_1[[cluster_col]]))
        
        # Set "Cluster 1" as reference level
        # Cluster 0 coefficient will represent Cluster 0 vs Cluster 1 difference
        data_cluster0_1[[cluster_col]] <- relevel(data_cluster0_1[[cluster_col]], ref = "Cluster 1")
        
        # === BUILD MODEL FORMULA ===
        # Same covariates as Models 1 and 2
        rhs_cluster0_1 <- c(cluster_col, "age_centered", "p31", motion_terms, icd_terms)
        
        # Wrap variable names in backticks
        rhs_cluster0_1_safe <- paste0("`", rhs_cluster0_1, "`")
        fml_cluster0_1 <- as.formula(paste0("`", dependent_var, "` ~ ", paste(rhs_cluster0_1_safe, collapse = " + ")))
        
        # === CONSTRUCT MODEL MATRIX ===
        X_cluster0_1 <- model.matrix(fml_cluster0_1, data = data_cluster0_1)
        y_cluster0_1 <- data_cluster0_1[[dependent_var]]
             
        # === FIT QUANTILE REGRESSION MODEL ===
        model_cluster0_1 <- rq(fml_cluster0_1, data = data_cluster0_1, tau = tau)
        
        # Compute bootstrapped standard errors
        summary_cluster0_1 <- summary.rq(model_cluster0_1, se = "boot", R = R)
        print(summary_cluster0_1)
        
        # === COMPUTE BOOTSTRAP CONFIDENCE INTERVALS ===
        boot_out_cluster0_1 <- boot.rq(x = X_cluster0_1, y = y_cluster0_1, tau = tau, R = R)
        
        # Extract 95% CIs for all coefficients
        cis_cluster0_1 <- apply(boot_out_cluster0_1$B, 2, quantile, probs = c(0.025, 0.975))
        print("95% CI for coefficients:")
        print(cis_cluster0_1)
             
        # === EXTRACT P-VALUE FOR CLUSTER 0 vs CLUSTER 1 ===
        # Get coefficient table
        coef_table <- summary_cluster0_1$coefficients
        coef_rownames <- rownames(coef_table)
        coef_rownames_clean <- gsub("`", "", coef_rownames)
        
        # Helper function to extract p-values (same as previous models)
        get_p <- function(term_name) {
            idx <- which(coef_rownames_clean == term_name)
            if (length(idx) == 0) {
                warning(paste0("Coefficient not found for term: ", term_name))
                return(NA_real_)
            }
            pval <- coef_table[idx[1], "Pr(>|t|)"]
            return(pval)
        }

        # Extract p-value for Cluster 0 vs Cluster 1 comparison
        p_cluster0_vs_cluster1 <- get_p(paste0(cluster_col, "Cluster 0"))
        p_cluster0_vs_cluster1 <- ifelse(is.na(p_cluster0_vs_cluster1), NA_real_, ifelse(p_cluster0_vs_cluster1 < 2.2e-16, 2.2e-16, p_cluster0_vs_cluster1))
             
        # === COLLECT ALL P-VALUES ===
        # Create named vector with all four comparisons
        p_values <- c(
            depression_vs_control = p_depression_vs_control,   # Model 1
            cluster0_vs_control = p_cluster0_vs_control,       # Model 2
            cluster1_vs_control = p_cluster1_vs_control,       # Model 2
            cluster0_vs_cluster1 = p_cluster0_vs_cluster1      # Model 3
        )
        
        # Store p-values in R global environment for extraction by Python
        assign("quantreg_p_values", p_values, envir = .GlobalEnv)
             
    ''')  # End R script block

    # === EXTRACT RESULTS FROM R TO PYTHON ===
    # Transfer p-values from R global environment back to Python
    with localconverter(default_converter):
        # Get p-values vector from R
        r_p_values = ro.r("quantreg_p_values")
        
        # Get names of p-values (e.g., "depression_vs_control", "cluster0_vs_control", etc.)
        r_names = list(ro.r("names(quantreg_p_values)"))
        
        # Attempt to extract dependent variable name if stored as attribute
        # (Not currently set in R script, so will likely return NA)
        r_dep = ro.r('if (!is.null(attr(quantreg_p_values, "dependent_var"))) as.character(attr(quantreg_p_values, "dependent_var")) else NA_character_')
    
    # === EXTRACT DEPENDENT VARIABLE NAME ===
    # Parse dependent variable name from R or fall back to Python parameter
    try:
        dep_list = list(r_dep)
        # Use R-provided name if available, otherwise use Python parameter value
        dependent_var_name = str(dep_list[0]) if dep_list and dep_list[0] is not ro.rinterface.NA_Character else str(dependent_var)
    except Exception:
        # If extraction fails, use Python parameter value
        dependent_var_name = str(dependent_var)

    # === CONVERT R P-VALUES TO PYTHON DICTIONARY ===
    p_values: Dict[str, float] = {}
    
    # Validate that names and values have same length
    if len(r_names) != len(list(r_p_values)):
        raise RuntimeError("Mismatch between p-value names and values returned from R")

    # Extract each p-value from R vector
    for i, name in enumerate(r_names):
        val = list(r_p_values)[i]
        # Convert R NA to Python NaN, otherwise convert to float
        p_values[str(name)] = float(val) if val is not ro.rinterface.NA_Real else float('nan')

    # Return dependent variable name and all p-values
    return {
        'dependent_var': dependent_var_name,
        'p_values': p_values
    }

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_bootstrap_diagnostics(stability_results, clusters, conn_type, out_dir, analysis_level='global', dir_type = None):
    """Create combined bootstrap stability diagnostics figure.
    
    Parameters
    ----------
    stability_results : dict
        Output from bootstrap_clustering_stability
    clusters : np.ndarray
        Cluster labels (0-indexed)
    conn_type : str
        Connectivity type ('functional', 'structural', Structure-Function Coupling)
    dir_type : str, default=None
        Direction of connectivity ('internal', 'external', or None)
        If None, not included in titles.
    out_dir : str
        Output directory
    analysis_level : str, default='global'
        Analysis level for filename (e.g., 'global', 'modular')
    
    Returns
    -------
    None
        Saves figure to out_dir
    """
    # === EXTRACT STABILITY METRICS FROM RESULTS ===
    # Per-subject stability: proportion of bootstrap iterations where each subject
    # was assigned to the same cluster as in the original clustering
    stability = stability_results['per_subject_stability']
    
    # NMI (Normalized Mutual Information): measures overall agreement between
    # bootstrap clustering and original clustering across all subjects
    nmi_arr = np.asarray(stability_results['nmi_list'], dtype=float)
    
    # Jaccard similarity for Cluster 0: proportion of Cluster 0 members correctly
    # reassigned to Cluster 0 in each bootstrap iteration
    jacc0_arr = np.asarray(stability_results['jaccard_cluster0'], dtype=float)
    
    # Jaccard similarity for Cluster 1: proportion of Cluster 1 members correctly
    # reassigned to Cluster 1 in each bootstrap iteration
    jacc1_arr = np.asarray(stability_results['jaccard_cluster1'], dtype=float)
    
    # === CREATE FIGURE WITH GRID LAYOUT ===
    # 4 rows, 2 columns: top row spans both columns (per-subject barplot)
    # Row 2: NMI timeseries (left) and histogram (right)
    # Row 3: Jaccard histograms for Cluster 0 (left) and Cluster 1 (right)
    # Row 4: Stability by cluster boxplot (spans both columns)
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[2.5, 1.2, 1.2, 1.0], 
                          hspace=0.45, wspace=0.3)
    
    # === DETERMINE TITLE SUFFIX ===
    # Include direction type in titles if provided
    if dir_type is None:
        dir_suffix = ''
    else:
        dir_suffix = f' ({dir_type})'

    # Get modality-specific cluster colors for consistent visualization
    cluster_colors = _cluster_colors_for_modality(conn_type, dir_type)

    # === PANEL 1: PER-SUBJECT STABILITY BARPLOT ===
    # Top row (spans both columns): shows how stable each subject's cluster assignment is
    # Subjects are sorted by stability (most stable on left, least stable on right)
    ax_bar = fig.add_subplot(gs[0, :])
    
    # Find indices of subjects with valid (non-NaN) stability scores
    valid_idx = np.where(~np.isnan(stability))[0]
    
    if valid_idx.size > 0:
        # Extract valid stability scores and cluster labels
        st_valid = stability[valid_idx]
        cl_valid = clusters[valid_idx].astype(int)
        
        # Sort subjects by stability (descending order)
        order = np.argsort(st_valid)[::-1]
        st_sorted = st_valid[order]
        cl_sorted = cl_valid[order]
        
        # Color bars by cluster membership (Cluster 0 or Cluster 1)
        colors = np.array([cluster_colors["0"], cluster_colors["1"]])[cl_sorted]
        
        # Create barplot: one bar per subject, colored by cluster
        ax_bar.bar(np.arange(len(st_sorted)), st_sorted, color=colors, 
                  edgecolor='k', linewidth=0.2)
        
        # Configure axes and labels
        ax_bar.set_ylabel('Per-subject stability')
        ax_bar.set_xlabel('Subjects (sorted by stability)')
        ax_bar.set_title(f'Per-subject cluster stability (bootstrap) — {conn_type} — {dir_suffix}')
        ax_bar.set_ylim(0, 1.02)  # Stability ranges from 0 to 1
        
        # Add legend showing cluster colors
        handles = [mpatches.Patch(color=cluster_colors["0"], label='Cluster 0'), 
              mpatches.Patch(color=cluster_colors["1"], label='Cluster 1')]
        ax_bar.legend(handles=handles, loc='upper right')
    
    # === PANEL 2A: NMI TIMESERIES ===
    # Left panel of row 2: shows NMI (overall clustering agreement) across bootstrap iterations
    # High NMI indicates bootstrap clustering closely matches original clustering
    ax_nmi_ts = fig.add_subplot(gs[1, 0])
    
    if np.any(~np.isnan(nmi_arr)):
        # Plot NMI values across bootstrap iterations
        ax_nmi_ts.plot(np.arange(len(nmi_arr)), nmi_arr, marker='o', 
                      linestyle='-', alpha=0.6)
        
        # Add horizontal line showing mean NMI across all bootstrap iterations
        ax_nmi_ts.axhline(np.nanmean(nmi_arr), color='red', linestyle='--', 
                         label=f"Mean = {np.nanmean(nmi_arr):.3f}")
        
        ax_nmi_ts.set_xlabel('Bootstrap iteration')
        ax_nmi_ts.set_ylabel('NMI')  # NMI = Normalized Mutual Information
        ax_nmi_ts.set_title(f'NMI across bootstraps — {conn_type} — {dir_suffix}')
        ax_nmi_ts.legend()
    
    # === PANEL 2B: NMI HISTOGRAM ===
    # Right panel of row 2: distribution of NMI values across bootstrap iterations
    ax_nmi_hist = fig.add_subplot(gs[1, 1])
    
    if np.any(~np.isnan(nmi_arr)):
        # Create histogram with kernel density estimate overlay
        sns.histplot(nmi_arr[~np.isnan(nmi_arr)], bins=30, kde=True, 
                    ax=ax_nmi_hist, color='gray')
        ax_nmi_hist.set_xlabel('NMI')
        ax_nmi_hist.set_title('NMI Distribution')
    
    # === PANEL 3A: JACCARD SIMILARITY HISTOGRAM FOR CLUSTER 0 ===
    # Left panel of row 3: distribution of Jaccard similarity for Cluster 0
    # Jaccard = proportion of original Cluster 0 members reassigned to Cluster 0 in bootstrap
    ax_j0 = fig.add_subplot(gs[2, 0])
    
    if np.any(~np.isnan(jacc0_arr)):
        # Create histogram colored by Cluster 0 color, with KDE overlay
        sns.histplot(jacc0_arr[~np.isnan(jacc0_arr)], bins=30, kde=True, 
                color=cluster_colors["0"], ax=ax_j0)
        ax_j0.set_xlabel('Jaccard similarity')
        ax_j0.set_title(f'Bootstrap Jaccard — Cluster 0 ({conn_type}) — {dir_suffix}')
        ax_j0.grid(alpha=0.3, axis='y')  # Add subtle horizontal gridlines
    
    # === PANEL 3B: JACCARD SIMILARITY HISTOGRAM FOR CLUSTER 1 ===
    # Right panel of row 3: distribution of Jaccard similarity for Cluster 1
    ax_j1 = fig.add_subplot(gs[2, 1])
    
    if np.any(~np.isnan(jacc1_arr)):
        # Create histogram colored by Cluster 1 color, with KDE overlay
        sns.histplot(jacc1_arr[~np.isnan(jacc1_arr)], bins=30, kde=True, 
                color=cluster_colors["1"], ax=ax_j1)
        ax_j1.set_xlabel('Jaccard similarity')
        ax_j1.set_title(f'Bootstrap Jaccard — Cluster 1 ({conn_type}) — {dir_suffix}')
        ax_j1.grid(alpha=0.3, axis='y')  # Add subtle horizontal gridlines
    
    # === PANEL 4: STABILITY BY CLUSTER BOXPLOT ===
    # Bottom row (spans both columns): compare stability distributions between clusters
    # Shows whether one cluster is more stable than the other
    ax_box = fig.add_subplot(gs[3, :])
    
    # Create DataFrame with cluster labels and stability scores for seaborn boxplot
    cluster_df = pd.DataFrame({
        'Cluster': clusters,
        'cluster_stability': stability
    })
    
    # Only create boxplot if at least 2 distinct clusters exist
    if cluster_df['Cluster'].nunique(dropna=True) >= 2:
        # Respect cluster label order from stability results if provided
        # (ensures consistent ordering across plots)
        order_list = stability_results.get('cluster_label_order', ['0', '1'])
        
        # Create boxplot with cluster-specific colors
        sns.boxplot(data=cluster_df.assign(Cluster_str=cluster_df['Cluster'].astype(str)),
               x='Cluster_str', y='cluster_stability', hue='Cluster_str',
               order=order_list, dodge=False, palette=cluster_colors, ax=ax_box)
        
        ax_box.set_xlabel('Cluster')
        ax_box.set_ylabel('Per-subject stability')
        ax_box.set_title(f'Stability by cluster — {conn_type}')
        ax_box.grid(alpha=0.3, axis='y')  # Add subtle horizontal gridlines
        
        # Remove redundant legend (cluster info already in x-axis labels)
        try:
            leg = ax_box.get_legend()
            if leg is not None:
                leg.remove()
        except Exception:
            pass
    
    # === SAVE FIGURE TO FILE ===
    # Construct output filename including analysis level, connectivity type, and direction
    out_path = os.path.join(out_dir, 
                           f'{analysis_level}_{conn_type}_{dir_suffix}_bootstrap_diagnostics_combined.png')
    
    # Save figure as high-resolution PNG
    fig.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
    
    # Close figure to free memory
    plt.close(fig)
    
    print(f"Saved bootstrap diagnostics to: {out_path}")

def plot_module_violin_across_clusters(
    final_df,
    feature_df,
    plots_dir,
    modules=None,
    conn_types=('functional', 'structural', 'sfc'),
    dir_types=('internal', 'external'),
    minmax=True,
    figsize_per_row=(12, 3),
    palette_name='Set2',
    save_png=True,
    significance_map=None,
    comparison_groups=None,
    output_name=None,
):
    """Plot module-wise distributions (violin) across clusters for each connectivity type.

    For each connectivity type, creates one figure per direction where each module
    occupies a column showing violin plots across group/cluster categories.

    Parameters
    ----------
    final_df : pd.DataFrame
        DataFrame containing 'eid', 'depression_status', and cluster columns
        named '{conn_type}_{dir_type}_cluster'.
    feature_df : pd.DataFrame
        DataFrame containing feature columns like '<module>_{dir_type}_{conn_type}'.
        If contains 'eid', will be aligned to final_df; otherwise assumes same row order.
    plots_dir : str
        Directory where figures will be saved.
    modules : list or None, default=None
        Ordered list of module labels to plot. If None, inferred from feature_df.
    conn_types : tuple, default=('functional', 'structural', 'sfc')
        Connectivity types to visualize.
    dir_types : tuple, default=('internal', 'external')
        Directions to visualize (subset of ('internal','external')).
    minmax : bool, default=True
        If True, min-max scale each feature to [0, 1] across subjects before plotting.
    figsize_per_row : tuple, default=(12, 3)
        (width, height) per row to scale the figure size.
    palette_name : str, default='Set2'
        Seaborn palette name for cluster colors (not used if modality-specific colors applied).
    save_png : bool, default=True
        If True, save the figures to plots_dir.
    significance_map : dict, optional
        Mapping {(conn_type, dir_type, module): {comparison_key: p-value}}
        providing corrected p-values for group comparisons to annotate.
    comparison_groups : dict, optional
        Ordered mapping of comparison keys to (group_a, group_b) tuples for
        significance bracket annotation. Default comparisons:
        'depression_vs_control', 'cluster0_vs_control', 'cluster1_vs_control',
        'cluster0_vs_cluster1'.
    output_name : str, optional
        Custom relative path template (formatted with conn, dir, module)
        for saved figures. Falls back to default path when None.

    Notes
    -----
    - Groups displayed: Control, Depression, Cluster 0, Cluster 1 (where applicable).
    - Significance brackets drawn above violins for comparisons with p < 0.05.
    - Each connectivity type gets separate figures for internal/external directions.
    """

    # === CREATE OUTPUT DIRECTORY ===
    os.makedirs(plots_dir, exist_ok=True)

    # === DEFINE FIXED GROUP ORDER FOR PLOTTING ===
    # Order determines x-axis arrangement in violin plots:
    # Control (baseline) -> Depression (all depressed) -> Cluster 0 -> Cluster 1
    group_order = ['Control', 'Depression', 'Cluster 0', 'Cluster 1']

    # === SETUP DIRECTION TYPES ===
    # Convert to tuple, default to both internal and external connectivity
    dir_types = tuple(dir_types) if dir_types else ('internal', 'external')
    
    # === SETUP SIGNIFICANCE MAP AND COMPARISON GROUPS ===
    # significance_map: optional dict mapping (conn, dir, module) -> comparison_name -> p-value
    # comparison_groups: defines which group pairs to compare
    significance_map = significance_map or {}
    comparison_groups = comparison_groups or {
        'depression_vs_control': ('Depression', 'Control'),       # Main effect
        'cluster0_vs_control': ('Cluster 0', 'Control'),         # Cluster-specific
        'cluster1_vs_control': ('Cluster 1', 'Control'),         # Cluster-specific  
        'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),      # Inter-cluster
    }

    # === HELPER FUNCTION: CREATE SIGNIFICANCE LOOKUP KEY ===
    # Constructs tuple key for looking up p-values in significance_map
    def _sig_key(conn_label, dir_label, module_label):
        return (str(conn_label), str(dir_label), str(module_label))

    # === HELPER FUNCTION: FORMAT P-VALUE AS SIGNIFICANCE STARS ===
    # Converts numeric p-value to standard notation: *** (p<0.001), ** (p<0.01), * (p<0.05), ns
    def _format_sig_label(value):
        # If already a string, return as-is
        if isinstance(value, str):
            return value
        
        # Try to convert to float
        try:
            pval = float(value)
        except (TypeError, ValueError):
            return 'ns'
        
        # Check for NaN or infinity
        if not np.isfinite(pval):
            return 'ns'
        
        # Apply significance thresholds
        if pval < 0.001:
            return '***'
        if pval < 0.01:
            return '**'
        if pval < 0.05:
            return '*'
        return 'ns'

    # === HELPER FUNCTION: DRAW SIGNIFICANCE BRACKET ===
    # Draws bracket connecting two x-positions at specified y-height with significance label
    def _draw_bracket(ax, x1, x2, y, height, label):
        # Draw bracket shape: vertical-horizontal-horizontal-vertical
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color='black', linewidth=1.0)
        # Add label centered above bracket
        ax.text((x1 + x2) / 2.0, y + height * 1.15, label, ha='center', va='bottom', fontsize=9)

    # === HELPER FUNCTION: DRAW SIGNIFICANCE BRACKET OUTSIDE AXES ===
    # Alternative bracket drawing method using axes-fraction coordinates
    # Draws bracket above the plot area to avoid overlap with data
    def _draw_bracket_outside(ax, idx_a, idx_b, y_frac, height_frac, label):
        """
        Draw a bracket and label outside the top of the axes using axes-fraction coordinates
        so it doesn't overlap the plotted data. `idx_a` and `idx_b` are category indices
        (0-based) corresponding to `ax.get_xticks()` positions.
        y_frac and height_frac are in axes-fraction coordinates (0..1 range).
        """
        # === GET X-AXIS POSITIONS IN DATA COORDINATES ===
        try:
            # Get tick positions from axes
            xt = ax.get_xticks()
            
            # If tick positions cover our indices, use actual tick positions
            if len(xt) > max(idx_a, idx_b):
                x1_data = xt[idx_a]
                x2_data = xt[idx_b]
            else:
                # Fall back to index values if ticks don't cover range
                x1_data = float(idx_a)
                x2_data = float(idx_b)
        except Exception:
            # If tick extraction fails, use indices directly
            x1_data = float(idx_a)
            x2_data = float(idx_b)

        # === CONVERT DATA COORDINATES TO AXES-FRACTION COORDINATES ===
        # Axes-fraction coordinates: (0,0) = bottom-left, (1,1) = top-right of axes
        try:
            # Transform data coordinates to display (pixel) coordinates
            p1_disp = ax.transData.transform((x1_data, 0.0))
            p2_disp = ax.transData.transform((x2_data, 0.0))
            
            # Transform display coordinates to axes-fraction coordinates
            p1_axes = ax.transAxes.inverted().transform(p1_disp)
            p2_axes = ax.transAxes.inverted().transform(p2_disp)
            
            # Extract x-coordinates in axes-fraction space
            x1_frac = p1_axes[0]
            x2_frac = p2_axes[0]
        except Exception:
            # Fall back to simple fraction calculation if transformation fails
            # Center each category at (index + 0.5) / total_categories
            x1_frac = (idx_a + 0.5) / max(1.0, len(ax.get_xticks()))
            x2_frac = (idx_b + 0.5) / max(1.0, len(ax.get_xticks()))

        # === DRAW BRACKET IN AXES-FRACTION COORDINATES ===
        # Plot bracket shape: vertical-horizontal-horizontal-vertical
        # Using transAxes ensures bracket stays above plot regardless of data range
        ax.plot([x1_frac, x1_frac, x2_frac, x2_frac], [y_frac, y_frac + height_frac, y_frac + height_frac, y_frac],
                transform=ax.transAxes, color='black', linewidth=1.0)
        
        # Add significance label centered above bracket
        ax.text((x1_frac + x2_frac) / 2.0, y_frac + height_frac * 1.15, label,
                ha='center', va='bottom', fontsize=9, transform=ax.transAxes)

    # === HELPER FUNCTION: ANNOTATE AXIS WITH SIGNIFICANCE BRACKETS ===
    # Adds significance brackets to a violin plot panel based on FDR-corrected p-values
    def _annotate_axis(ax, df_plot, present_groups, conn_label, dir_label, module_label):
        # Skip annotation if no data or no groups present
        if df_plot.empty or not present_groups:
            return
        
        # === LOOKUP SIGNIFICANCE VALUES FOR THIS MODULE ===
        # Get dict of comparison_name -> p_value for this (conn, dir, module)
        comp_dict = significance_map.get(_sig_key(conn_label, dir_label, module_label))
        
        # Skip if no significance data available for this module
        if not comp_dict:
            return
        
        # === DETERMINE Y-AXIS DATA RANGE ===
        # If global min-max scaling was applied, data range is [0, 1]
        if minmax:
            y_min = 0.0
            y_max = 1.0
        else:
            # Use actual data min/max from plotted values
            y_min = float(df_plot['value'].min())
            y_max = float(df_plot['value'].max())
        
        # Calculate data range for positioning brackets
        data_range = y_max - y_min
        
        # Handle edge case of zero range (all values identical)
        if not np.isfinite(data_range) or data_range <= 0:
            data_range = 1.0
        
        # Step size for vertical spacing between stacked brackets
        # 12% of data range, minimum 0.05
        step = max(data_range * 0.12, 0.05)
        
        # === COLLECT VALID COMPARISONS TO ANNOTATE ===
        valid_comps = []  # List of (idx_a, idx_b, label) tuples
        
        for comp_name, groups in comparison_groups.items():
            # Skip if no p-value available for this comparison
            if comp_name not in comp_dict:
                continue
            
            # Skip if both groups in comparison are not present in this plot
            if not all(g in present_groups for g in groups):
                continue
            
            # Get x-axis indices for the two groups being compared
            idx_a = present_groups.index(groups[0])
            idx_b = present_groups.index(groups[1])
            
            # Skip if comparing a group to itself (shouldn't happen)
            if idx_a == idx_b:
                continue
            
            # Format p-value as significance label (*, **, ***, or ns)
            label = _format_sig_label(comp_dict[comp_name])
            
            # Add to list of comparisons to draw
            valid_comps.append((idx_a, idx_b, label))
        
        # If no valid comparisons, skip annotation
        if not valid_comps:
            return
        
        # === DRAW SIGNIFICANCE BRACKETS ===
        # Bracket height (vertical span)
        height = step * 0.35
        
        # Draw brackets stacked vertically above the data range
        # Start at y_max and stack upward with spacing = step
        for i, (idx_a, idx_b, label) in enumerate(valid_comps, start=1):
            # Base y-position for this bracket (higher for each successive bracket)
            y_base = y_max + step * i
            
            # Draw bracket connecting groups at idx_a and idx_b
            _draw_bracket(ax, idx_a, idx_b, y_base, height, label)

        # === EXPAND Y-AXIS TO ACCOMMODATE BRACKETS ===
        # Add extra space above data to fit all stacked brackets
        extra_stack = len(valid_comps) + 2.0  # +2 for padding
        new_top = y_max + step * extra_stack
        ax.set_ylim(y_min, new_top)

        # === SET Y-AXIS TICKS FOR MIN-MAX SCALED DATA ===
        # When data is scaled to [0,1], show ticks at 0, 0.25, 0.5, 0.75, 1.0
        if minmax:
            try:
                ax.set_yticks(np.linspace(0.0, 1.0, 5))
            except Exception:
                pass

    # === MAIN LOOP: ITERATE OVER CONNECTIVITY TYPES ===
    # Create separate figure for each connectivity modality (functional, structural, SFC)
    for conn in conn_types:
        # === CONSTRUCT COLUMN NAME SUFFIXES ===
        # Internal connectivity: e.g., "_internal_functional"
        # External connectivity: e.g., "_external_functional"
        suf_int = f"_internal_{conn}"
        suf_ext = f"_external_{conn}"

        # === INFER MODULES FROM FEATURE COLUMNS ===
        # If modules list not provided, infer from feature_df column names
        if modules is None:
            # Find all columns ending with internal or external suffix for this conn type
            cols = [c for c in feature_df.columns if c.endswith(suf_int) or c.endswith(suf_ext)]
            
            # Extract module names (prefix before first underscore)
            # e.g., "1_internal_functional" -> "1"
            modules_local = sorted({c.split('_', 1)[0] for c in cols}, key=lambda x: x)
        else:
            # Use provided module list
            modules_local = list(modules)

        # Skip if no modules found for this connectivity type
        if len(modules_local) == 0:
            print(f"No modules found for {conn}; skipping.")
            continue

        # === CONSTRUCT ORDERED FEATURE COLUMN NAMES ===
        # Expected column names for each module
        feat_int = [f"{m}{suf_int}" for m in modules_local]  # Internal connectivity columns
        feat_ext = [f"{m}{suf_ext}" for m in modules_local]  # External connectivity columns
        
        # === FILTER TO EXISTING COLUMNS ===
        # Keep only modules that have at least one column (internal or external) in feature_df
        modules_final = []  # Final list of modules to plot
        int_cols = []       # Corresponding internal column names (or None if missing)
        ext_cols = []       # Corresponding external column names (or None if missing)
        
        for m, a, b in zip(modules_local, feat_int, feat_ext):
            # Include module if either internal or external column exists
            if a in feature_df.columns or b in feature_df.columns:
                modules_final.append(m)
                int_cols.append(a if a in feature_df.columns else None)
                ext_cols.append(b if b in feature_df.columns else None)
        
        # Skip if no valid feature columns found
        if not modules_final:
            print(f"No ordered features for {conn}; skipping.")
            continue

        # === PREPARE DATA: MERGE CLUSTER LABELS WITH FEATURES ===
        # Align final_df (cluster labels) with feature_df (connectivity values)
        # Keep depression_status to distinguish Control vs Depression groups
        
        if 'eid' in final_df.columns and 'eid' in feature_df.columns:
            # === MERGE BY SUBJECT ID ===
            # Select cluster columns for this connectivity type
            merged = final_df[['eid', 'depression_status', f'{conn}_internal_cluster', f'{conn}_external_cluster']].merge(
                feature_df[['eid'] + [c for c in feature_df.columns if c in int_cols + ext_cols]],
                on='eid', how='left'
            )
        else:
            # === ALIGN BY INDEX ===
            # If no 'eid' column, assume DataFrames are already aligned by row order
            merged = final_df[['depression_status', f'{conn}_internal_cluster', f'{conn}_external_cluster']].copy().reset_index(drop=True)
            
            # Copy feature columns from feature_df to merged DataFrame
            for c in set([c for c in int_cols + ext_cols if c]):
                merged[c] = feature_df[c].values if c in feature_df.columns else np.nan
            
            # Add eid column (use existing or create RangeIndex)
            merged.insert(0, 'eid', final_df.get('eid', pd.RangeIndex(len(final_df))))

        # === OPTIONAL: APPLY GLOBAL MIN-MAX SCALING ===
        # Scale all feature columns to [0, 1] range for visualization consistency
        if minmax:
            # Identify feature columns (exclude metadata columns)
            feature_cols_z = [
                c
                for c in merged.columns
                if c not in ('eid', 'depression_status', f'{conn}_internal_cluster', f'{conn}_external_cluster')
            ]
            
            if feature_cols_z:
                # Fill missing values with column medians before scaling
                scaled = merged[feature_cols_z].fillna(merged[feature_cols_z].median()).astype(float)
                
                # Apply min-max scaling to [0, 1]
                scaler = MinMaxScaler(feature_range=(0.0, 1.0))
                scaled_vals = scaler.fit_transform(scaled.values.reshape(-1, 1)).reshape(scaled.values.shape)
                
                # Replace original values with scaled values
                merged[feature_cols_z] = scaled_vals

        # === IDENTIFY AVAILABLE DIRECTION TYPES ===
        # Check which direction types have cluster columns in the data
        available_dir_types = []
        
        for dir_type in dir_types:
            cluster_col_name = f'{conn}_{dir_type}_cluster'
            
            # Skip if cluster column doesn't exist for this (conn, dir) combination
            if cluster_col_name not in merged.columns:
                print(f"{cluster_col_name} not in final_df; skipping {conn} {dir_type}.")
                continue
            
            # Add to list of available directions
            available_dir_types.append(dir_type)

        # Skip if no valid direction types found
        if not available_dir_types:
            continue

        # === CONFIGURE FIGURE DIMENSIONS ===
        # Figure size per direction (internal/external) separately
        ncols = len(modules_final)  # Number of subplot columns (one per module)
        base_w, base_h = figsize_per_row  # Base width and height from parameters

        # === MAP DIRECTION TYPES TO COLUMN LISTS ===
        # Maps 'internal' -> int_cols, 'external' -> ext_cols
        dir_to_cols = {
            'internal': int_cols,
            'external': ext_cols,
        }

        # === HELPER FUNCTION: BUILD GROUP DATAFRAME FOR VIOLIN PLOT ===
        # Transforms data into long-form DataFrame with 'group' and 'value' columns
        # Groups: 'Control', 'Depression', 'Cluster 0', 'Cluster 1'
        def _build_group_df(col_name, cluster_col_name):
            # Return empty DataFrame if column doesn't exist
            if col_name is None or col_name not in merged.columns or cluster_col_name not in merged.columns:
                return pd.DataFrame(columns=['group', 'value'])
            
            # Extract relevant columns: feature values, depression status, cluster labels
            vals = merged[[col_name, 'depression_status', cluster_col_name]].copy()
            
            # Drop rows with missing feature values
            vals = vals.dropna(subset=[col_name])
            
            # Build list of (group, value) dictionaries
            rows = []
            
            for _, r in vals.iterrows():
                v = r[col_name]  # Feature value (connectivity)
                dep = r['depression_status']  # 0=control, 1=depressed
                cl = str(r[cluster_col_name]) if pd.notna(r[cluster_col_name]) else None
                
                # === ASSIGN GROUPS ===
                if dep == 0:
                    # Control subjects: single group
                    rows.append({'group': 'Control', 'value': v})
                elif dep == 1:
                    # Depressed subjects: appear in both 'Depression' and cluster-specific groups
                    rows.append({'group': 'Depression', 'value': v})  # All depressed subjects
                    
                    # Also assign to cluster-specific group if clustered
                    if cl == '0':
                        rows.append({'group': 'Cluster 0', 'value': v})
                    elif cl == '1':
                        rows.append({'group': 'Cluster 1', 'value': v})
            
            # Return empty DataFrame if no valid rows
            if not rows:
                return pd.DataFrame(columns=['group', 'value'])
            
            # Convert to DataFrame for seaborn plotting
            return pd.DataFrame(rows)

        # === LOOP OVER DIRECTION TYPES ===
        # Create separate figure for each direction (internal/external)
        for dir_idx, dir_type in enumerate(available_dir_types):
            # === GET MODALITY-SPECIFIC CLUSTER COLORS ===
            cluster_colors = _cluster_colors_for_modality(conn, dir_type)
            
            # === DEFINE GROUP COLOR PALETTE ===
            # Consistent colors across all plots
            group_colors = {
                'Control': "#2ca02c",          # Green
                'Depression': "#6a3d9a",       # Purple
                'Cluster 0': cluster_colors['0'],  # Modality-specific
                'Cluster 1': cluster_colors['1'],  # Modality-specific
            }
            
            # === CREATE FIGURE WITH SUBPLOTS ===
            # One subplot per module (arranged horizontally)
            fig_w = base_w
            fig_h = base_h * 1.6  # Increased height for significance brackets
            
            fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(fig_w, fig_h), sharex='col', sharey='row')
            
            # Ensure axes is always an array (even for single subplot)
            if ncols == 1:
                axes = np.array([axes], dtype=object)
            else:
                axes = np.array(axes, dtype=object)

            # === GET COLUMN LIST FOR THIS DIRECTION ===
            # Either int_cols or ext_cols depending on dir_type
            col_list = dir_to_cols.get(dir_type, [])
            # === LOOP OVER MODULES ===
            # Create one subplot (violin plot) per module
            for mod_idx, m in enumerate(modules_final):
                ax = axes[mod_idx]  # Current subplot axes
                
                # Get column name for this module and direction
                col_name = col_list[mod_idx] if mod_idx < len(col_list) else None
                
                # Cluster column name for this connectivity type and direction
                cluster_col = f'{conn}_{dir_type}_cluster'

                # === HANDLE MISSING DATA CASE ===
                if col_name is None:
                    # Display "Missing" text if column doesn't exist
                    ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
                    ax.set_title(f"Module {m}")
                    ax.set_xlabel('')
                    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
                    
                    # Rotate x-tick labels for readability
                    for tick in ax.get_xticklabels():
                        tick.set_ha('right')
                    
                    # Only leftmost subplot gets y-axis label
                    if mod_idx == 0:
                        ax.set_ylabel(f"{dir_type.capitalize()} connectivity")
                    else:
                        ax.set_ylabel('')
                    
                    # Add gridlines for reference
                    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.6, linewidth=0.8)
                    continue

                # === BUILD DATA FOR VIOLIN PLOT ===
                # Transform data into long-form with 'group' and 'value' columns
                df_dir = _build_group_df(col_name, cluster_col)
                
                if not df_dir.empty:
                    # === CREATE VIOLIN PLOT ===
                    # Filter to groups that are actually present in the data
                    present_groups = [g for g in group_order if g in df_dir['group'].unique()]
                    
                    # Create violin plot with seaborn
                    sns.violinplot(
                        x='group',
                        y='value',
                        data=df_dir,
                        order=present_groups,  # Ensures consistent ordering
                        palette=[group_colors[g] for g in present_groups],  # Apply color scheme
                        cut=0,  # Don't extend violins beyond data range
                        ax=ax,
                    )
                    
                    # === ADD SIGNIFICANCE ANNOTATIONS ===
                    # Draw brackets with FDR-corrected p-values if available
                    _annotate_axis(ax, df_dir, present_groups, conn, dir_type, m)
                else:
                    # Display "No data" if no valid data points
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                
                # === FORMAT SUBPLOT ===
                ax.set_title(f"Module {m}")
                ax.set_xlabel('')  # No x-axis label (group names are self-explanatory)
                ax.tick_params(axis='x', labelrotation=30, labelsize=8)
                
                # Rotate x-tick labels for readability
                for tick in ax.get_xticklabels():
                    tick.set_ha('right')
                
                # Only leftmost subplot gets y-axis label
                if mod_idx == 0:
                    ax.set_ylabel(f"{dir_type.capitalize()} connectivity")
                else:
                    ax.set_ylabel('')
                
                # Add gridlines for reference
                ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.6, linewidth=0.8)

            # === ADD DIRECTION TYPE ANNOTATION ===
            # Add text label above leftmost subplot indicating direction (Internal/External)
            ax_row = axes[0]
            ax_row.annotate(
                dir_type.capitalize(),
                xy=(0.5, 1.10),  # Position slightly above top of axes
                xycoords='axes fraction',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
            )

            # === SET FIGURE TITLE ===
            # Overall title describing connectivity type and direction
            fig_title = f"Modular group {_display_conn_type_text(conn)} distributions ({dir_type})"
            
            # Wrap long title to multiple lines (max 50 chars per line)
            fig.suptitle(textwrap.fill(fig_title, width=50), fontsize=12, fontweight='bold', y=0.99)
            
            # Adjust layout to prevent overlap
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])


            # === CONSTRUCT OUTPUT FILENAME ===
            # Create slug from module list (sanitized for filesystem)
            modules_slug = "__".join(str(m) for m in modules_final) or 'modules'
            modules_slug = re.sub(r'[^0-9A-Za-z]+', '_', modules_slug).strip('_') or modules_slug
            
            # Sanitize direction type for filename
            dir_slug = re.sub(r'[^0-9A-Za-z]+', '_', str(dir_type)).strip('_') or str(dir_type)

            # === DETERMINE OUTPUT PATH ===
            if output_name:
                # Use provided filename template with format substitution
                rel_path = output_name.format(conn=conn, dir=dir_slug, module=modules_slug)
                
                # If template doesn't include {dir} placeholder, append dir_slug manually
                if '{dir}' not in output_name:
                    base, ext = os.path.splitext(rel_path)
                    
                    if ext:
                        # Has extension: insert dir_slug before extension
                        rel_path = f"{base}_{dir_slug}{ext}"
                    else:
                        # No extension: append dir_slug and .png if save_png is True
                        rel_path = f"{rel_path}_{dir_slug}.png" if save_png else f"{rel_path}_{dir_slug}"
                
                out_path = os.path.join(plots_dir, rel_path)
            else:
                # Use default filename format
                out_path = os.path.join(plots_dir, f'{conn}_con', f"F32_{conn}_{dir_slug}_module_violin_by_cluster.png")

            # === SAVE FIGURE ===
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # Save as PNG if requested
            if save_png:
                fig.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
            
            # Close figure to free memory
            plt.close(fig)
            
            print(f"Saved {out_path}")

def determine_covariate_distributions(
    combined_df,
    available_types,
    motion_metric,
    out_dir,
    cohorts_dir,
    icd_covariates=None,
    group_col='depression_status',
    conn_types=None,
    dir_types=None,
):
    """Plot covariate distributions by group and cluster with statistical tests.
    
    Creates multi-row covariate distribution plots showing age, sex, head motion,
    and ICD-10 comorbidities across depression/control groups and cluster assignments.
    Performs statistical tests (Mann-Whitney U, Chi-square) with FDR correction.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined dataframe with columns: eid, depression_status (or group_col),
        p21003_i2 (age), p31 (sex), motion metrics, connectivity values, and
        cluster labels.
    available_types : list of str
        Available connectivity types (e.g., ['functional', 'structural', 'sfc']).
    motion_metric : str
        Default motion column name (e.g., 'p24441_i2' or 'p24453_i2').
    out_dir : str
        Output directory for saving plots and summary tables.
    cohorts_dir : str
        Directory containing cluster CSV files for loading additional modalities.
    icd_covariates : list of str, optional
        List of ICD-10 code covariate column names (comorbidities).
    group_col : str, default='depression_status'
        Column name for group status (0=control, 1=depression).
    conn_types : list of str, optional
        Connectivity types to include in row configs. Default: ['functional', 'structural', 'sfc'].
    dir_types : list of str, optional
        Direction types to include. Default: ['internal', 'external'].
    
    Returns
    -------
    None
        Saves PNG figure and TXT/CSV summary tables to out_dir.

    Notes
    -----
    - Creates 3×N multi-row plot (N = number of covariate columns).
    - First row shows distributions by Depression vs Control.
    - Subsequent rows show distributions by cluster for each (conn_type, dir_type).
    - Statistical tests: Mann-Whitney U for continuous variables, Chi-square for categorical.
    - All p-values undergo FDR-BH correction.
    - Significance brackets annotated on plots (*, **, ***).
    """
    def _summarize_covariate_values(df, x_col, cov_name, cov_col, groups):
        # === HELPER: GENERATE NUMERIC SUMMARIES FOR EACH GROUP ===
        # Compute descriptive statistics (mean, std, median, quartiles, counts)
        # for a covariate across all groups (Control, Depression, Cluster 0, Cluster 1)
        rows = []  # Store one row per group
        for grp in groups:
            # Filter dataframe to current group
            grp_df = df.loc[df[x_col] == grp]
            if cov_col not in grp_df.columns:
                continue
            values = grp_df[cov_col].dropna()
            if values.empty:
                summary = {
                    'n': 0,
                    'mean': np.nan,
                    'std': np.nan,
                    'median': np.nan,
                    'q1': np.nan,
                    'q3': np.nan,
                    'count_1': np.nan,
                    'prop_1': np.nan,
                }
            else:
                if pd.api.types.is_numeric_dtype(values):
                    summary = {
                        'n': int(values.shape[0]),
                        'mean': float(values.mean()),
                        'std': float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                        'median': float(values.median()),
                        'q1': float(values.quantile(0.25)),
                        'q3': float(values.quantile(0.75)),
                        # If the numeric column only contains 0/1, report counts/props for females (0)
                        'count_1': float(values.eq(0).sum()) if values.isin([0, 1]).all() else np.nan,
                        'prop_1': float(values.eq(0).mean()) if values.isin([0, 1]).all() else np.nan,
                    }
                else:
                    # For categorical sex variables, count females regardless of encoding
                    counts = values.value_counts(dropna=True)
                    # Try multiple encodings: 0, 'Female', 'female', 'F', 'f'
                    count_female = 0.0
                    for female_val in [0, 'Female', 'female', 'F', 'f']:
                        if female_val in counts.index:
                            count_female = float(counts.get(female_val, 0.0))
                            break
                    prop_female = (count_female / float(values.shape[0])) if values.shape[0] > 0 else np.nan
                    summary = {
                        'n': int(values.shape[0]),
                        'mean': np.nan,
                        'std': np.nan,
                        'median': np.nan,
                        'q1': np.nan,
                        'q3': np.nan,
                        'count_1': count_female,
                        'prop_1': prop_female,
                    }

            rows.append({
                'covariate': cov_name,
                'covariate_column': cov_col,
                'group_label': grp,
                **summary,
            })
        return rows

    # === INITIALIZE COVARIATE COLUMNS AND COLOR PALETTE ===
    # Normalize ICD covariates list (convert None to empty list)
    icd_covariates = list(icd_covariates or [])
    # Define column headers for multi-panel grid: Age, Sex, Motion, plus ICD comorbidities
    covariate_columns = ["Age", "Sex", "Head Motion"] + [str(c) for c in icd_covariates]
    # Base color mapping for Control vs Depression groups; cluster colors added dynamically
    base_category_colors = {
        'Control': "#2ca02c",      # Green for healthy controls
        'Depression': "#6a3d9a",   # Purple for depressed subjects
    }

    # === MOTION METRIC LOOKUP TABLE ===
    # Default motion column names for each connectivity type
    # fMRI motion used for functional and SFC; dMRI motion used for structural
    motion_lookup = {
        'functional': 'p24441_i2',
        'structural': 'p24453_i2',
        'sfc': 'p24441_i2',
    }

    def _ensure_group_column(df):
        # === HELPER: ENSURE CANONICAL 'Group' COLUMN EXISTS ===
        # Convert numeric depression_status (0/1) to categorical Group labels (Control/Depression)
        # If 'Group' already exists, preserve it; otherwise create from group_col
        if 'Group' in df.columns:
            return df  # Already has Group column, return as-is
        if group_col in df.columns:
            df = df.copy()  # Avoid modifying original dataframe
            # Map 0 -> Control, 1 -> Depression; preserve other values as strings
            df['Group'] = df[group_col].map({0: 'Control', 1: 'Depression'}).fillna(df[group_col].astype(str))
            return df
        return df  # No group column found, return unchanged

    def _normalize_cluster_label(value):
        # === HELPER: NORMALIZE CLUSTER LABELS TO CONSISTENT FORMAT ===
        # Convert various cluster label formats to standardized strings:
        # 'Cluster 0' -> 'Cluster 0', '0' -> 'Cluster 0', 'control' -> 'Control'
        if pd.isna(value):
            return value  # Preserve NaN values
        value_str = str(value).strip()
        # If already prefixed with 'Cluster', clean up underscores and return
        if value_str.lower().startswith('cluster'):
            return value_str.replace('_', ' ')  # 'Cluster_0' -> 'Cluster 0'
        # Map 'control' string to capitalized 'Control'
        if value_str.lower() == 'control':
            return 'Control'
        # Try converting numeric values to 'Cluster N' format
        try:
            return f"Cluster {int(float(value_str))}"  # '0' -> 'Cluster 0'
        except (ValueError, TypeError):
            return value_str  # If conversion fails, return original

    def _wrap_title(text, width=28):
        # === HELPER: WRAP LONG TEXT FOR PLOT TITLES ===
        # Insert line breaks to prevent excessively wide axis titles
        return textwrap.fill(str(text), width=width)

    def _format_sig_label(pval, label_prefix=None):
        # === HELPER: CONVERT P-VALUE TO SIGNIFICANCE STARS ===
        # Generate asterisk notation: *** (p<0.001), ** (p<0.01), * (p<0.05), ns (not significant)
        try:
            pval = float(pval)  # Convert to float for comparison
        except (TypeError, ValueError):
            label = 'ns'  # If conversion fails, mark as not significant
        else:
            # Apply standard significance thresholds
            if not np.isfinite(pval):
                label = 'ns'  # NaN or Inf -> not significant
            elif pval < 0.001:
                label = '***'  # Highly significant
            elif pval < 0.01:
                label = '**'   # Very significant
            elif pval < 0.05:
                label = '*'    # Significant
            else:
                label = 'ns'   # Not significant
        # Optionally prepend custom label (e.g., 'FDR ***')
        if label_prefix:
            return f"{label_prefix}: {label}"
        return label

    def _draw_bracket(ax, x1, x2, y, height, label):
        # === HELPER: DRAW SIGNIFICANCE BRACKET ON AXIS ===
        # Draw horizontal line with vertical caps connecting two groups, with label above
        # x1, x2: x-positions of the two groups being compared
        # y: base y-position for bracket, height: vertical extent of bracket
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color='black', linewidth=1.0)
        # Center text label above bracket
        ax.text((x1 + x2) / 2.0, y + height * 1.05, label, ha='center', va='bottom', fontsize=8)

    def _annotate_axis(ax, present_groups, comp_dicts, y_max=None):
        # === HELPER: ANNOTATE AXIS WITH SIGNIFICANCE BRACKETS ===
        # Stack multiple significance brackets vertically above plot
        # Each bracket shows comparison label and significance level (stars)
        
        # Skip annotation if no groups are present in the plot
        if not present_groups:
            return
        
        # Determine y-axis limits for positioning brackets
        if y_max is None or not np.isfinite(y_max):
            y_min, y_max = ax.get_ylim()  # Use current axis limits
        else:
            y_min = 0.0  # Assume zero baseline if max provided
        
        # Calculate vertical spacing for stacked brackets (12% of data range per bracket)
        data_range = y_max - y_min
        if not np.isfinite(data_range) or data_range <= 0:
            data_range = 1.0  # Fallback to unit range
        step = data_range * 0.12  # Vertical increment for each bracket
        current_y = y_max + step * 0.25  # Start position above data
        
        def _is_sig(pval):
            # Check if p-value meets significance threshold (p < 0.05)
            try:
                pval = float(pval)
            except (TypeError, ValueError):
                return False
            return np.isfinite(pval) and pval < 0.05
        
        # === DRAW BRACKETS FOR SIGNIFICANT COMPARISONS ===
        # Loop through all provided comparisons and draw brackets for significant tests only
        for comp in comp_dicts:
            label_prefix = comp.get('label_prefix')  # E.g., 'FDR' for FDR-corrected
            comp_map = comp.get('comparisons', {})  # Dict of comparison_name -> (group1, group2)
            for comp_name, groups in comp_map.items():
                # Skip if either group is not present in current plot
                if not all(g in present_groups for g in groups):
                    continue
                # Get p-value for this comparison
                pval = comp.get('pvals', {}).get(comp_name)
                if not _is_sig(pval):
                    continue  # Skip non-significant comparisons
                # Format significance label (stars)
                label = _format_sig_label(pval, label_prefix=label_prefix)
                # Get x-positions for the two groups
                idx_a = present_groups.index(groups[0])
                idx_b = present_groups.index(groups[1])
                if idx_a == idx_b:
                    continue  # Can't compare group to itself
                height = step * 0.30  # Bracket height
                # Draw bracket connecting the two groups
                _draw_bracket(ax, idx_a, idx_b, current_y, height, label)
                current_y += step  # Move up for next bracket
        
        # Expand y-axis limit to accommodate all brackets
        ax.set_ylim(y_min, current_y + step * 0.30)

    def _mannwhitney_test(df, x_col, group_a, group_b, value_col):
        # === HELPER: MANN-WHITNEY U TEST FOR TWO INDEPENDENT GROUPS ===
        # Non-parametric test for comparing continuous distributions between two groups
        # Null hypothesis: two groups have same distribution (no location shift)
        if value_col not in df.columns:
            return np.nan, np.nan  # Column doesn't exist
        # Extract values for each group, dropping missing values
        vals_a = df.loc[df[x_col] == group_a, value_col].dropna()
        vals_b = df.loc[df[x_col] == group_b, value_col].dropna()
        if vals_a.empty or vals_b.empty:
            return np.nan, np.nan  # Can't test if either group has no data
        try:
            # Run two-sided Mann-Whitney U test
            stat, pval = sp_stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            return float(stat), float(pval)
        except Exception:
            return np.nan, np.nan  # Test failed (e.g., all values identical)

    def _chi2_test(df, x_col, group_a, group_b, value_col):
        # === HELPER: CHI-SQUARE TEST FOR CATEGORICAL VARIABLES ===
        # Test for independence between grouping variable and categorical covariate
        # Null hypothesis: no association between group and covariate distribution
        if value_col not in df.columns:
            return np.nan, np.nan  # Column doesn't exist
        # Filter to only the two groups being compared
        df_sub = df[df[x_col].isin([group_a, group_b])].copy()
        if df_sub.empty:
            return np.nan, np.nan  # No data for these groups
        # Build contingency table: rows = groups, columns = covariate values
        table = pd.crosstab(df_sub[x_col], df_sub[value_col])
        # Require at least 2x2 table with non-zero margins for valid test
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan, np.nan  # Degenerate table
        if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
            return np.nan, np.nan  # Zero marginals break test
        try:
            # Run chi-square test of independence
            stat, pval, _, _ = sp_stats.chi2_contingency(table.values)
            return float(stat), float(pval)
        except Exception:
            return np.nan, np.nan  # Test failed (e.g., expected frequencies too low)

    def _compute_covariate_tests(row_cfg):
        # === HELPER: ASSEMBLE AND RUN COVARIATE STATISTICAL TESTS ===
        # For a given row configuration (Group or Cluster), run all relevant
        # statistical tests (Mann-Whitney U for continuous, Chi-square for categorical)
        # and apply FDR-BH correction across all tests
        tests = []  # List of tuples: (row_title, covariate_name, comparison_name, pval, stat, test_type)
        
        # === DEFINE COMPARISON GROUPS ===
        # For 'Group' row: compare Depression vs Control
        comparisons_group = {
            'depression_vs_control': ('Depression', 'Control'),
        }
        # For 'Cluster' rows: compare each cluster to control and clusters to each other
        comparisons_cluster = {
            'cluster0_vs_control': ('Cluster 0', 'Control'),
            'cluster1_vs_control': ('Cluster 1', 'Control'),
            'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
        }

        # Extract row configuration details
        df = row_cfg['df']  # Dataframe for this row
        x_col = row_cfg['x_col']  # Grouping column ('Group' or 'Cluster')
        row_title = row_cfg['title']  # Title for this row
        # Select appropriate comparison map based on grouping variable
        comp_map = comparisons_group if x_col == 'Group' else comparisons_cluster

        # === TEST 1: AGE (MANN-WHITNEY U) ===
        # Test for age differences between groups using non-parametric test
        if 'p21003_i2' in df.columns:
            for comp_name, (g1, g2) in comp_map.items():
                stat, pval = _mannwhitney_test(df, x_col, g1, g2, 'p21003_i2')
                tests.append((row_title, 'Age', comp_name, pval, stat, 'Mann-Whitney U'))

        # === TEST 2: HEAD MOTION (MANN-WHITNEY U) ===
        # Test for head motion differences between groups
        # Motion can be single column or list of columns (fMRI + dMRI)
        motion_col = row_cfg.get('motion_col')
        if isinstance(motion_col, (list, tuple)):
            # Multiple motion metrics (e.g., fMRI and dMRI motion)
            for col in motion_col:
                if col not in df.columns:
                    continue  # Skip missing columns
                cov_key = f"Head Motion ({col})"  # Unique key per metric
                for comp_name, (g1, g2) in comp_map.items():
                        stat, pval = _mannwhitney_test(df, x_col, g1, g2, col)
                        tests.append((row_title, cov_key, comp_name, pval, stat, 'Mann-Whitney U'))
        elif isinstance(motion_col, str):
            # Single motion metric
            if motion_col in df.columns:
                for comp_name, (g1, g2) in comp_map.items():
                    stat, pval = _mannwhitney_test(df, x_col, g1, g2, motion_col)
                    tests.append((row_title, 'Head Motion', comp_name, pval, stat, 'Mann-Whitney U'))

        # === TEST 3: SEX (CHI-SQUARE) ===
        # Test for sex distribution differences between groups (categorical test)
        if 'p31' in df.columns:
            for comp_name, (g1, g2) in comp_map.items():
                stat, pval = _chi2_test(df, x_col, g1, g2, 'p31')
                tests.append((row_title, 'Sex', comp_name, pval, stat, 'Chi-square'))

        # === TEST 4: ICD-10 COMORBIDITIES (CHI-SQUARE) ===
        # Test for ICD-10 comorbidity distribution differences between clusters
        # Only tested for Cluster rows (inter-cluster comparisons)
        if x_col == 'Cluster' and 'Cluster' in df.columns:
            # Filter to only depressed subjects with cluster assignments
            df_icd = df.copy()
            df_icd = df_icd[df_icd['Cluster'].isin(['Cluster 0', 'Cluster 1'])]
            # ICD tests only compare Cluster 0 vs Cluster 1 (not vs Control)
            icd_comp_map = {
                'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
            }
            for cov in icd_covariates:
                if cov not in df_icd.columns:
                    continue  # Skip missing ICD columns
                cov_key = str(cov)
                for comp_name, (g1, g2) in icd_comp_map.items():
                    stat, pval = _chi2_test(df_icd, 'Cluster', g1, g2, cov)
                    tests.append((row_title, cov_key, comp_name, pval, stat, 'Chi-square'))

        # === HANDLE EMPTY TEST LIST ===
        if not tests:
            return {}, None  # No tests to run

        # === APPLY FDR-BH CORRECTION ===
        # Extract raw p-values and metadata
        raw_pvals = np.array([t[3] for t in tests], dtype=float)
        test_methods = [t[2] for t in tests]  # Comparison names
        variable_names = [f"{t[0]}::{t[1]}" for t in tests]  # Row::Covariate labels

        # Initialize corrected p-values array
        corrected_pvals = np.full_like(raw_pvals, np.nan, dtype=float)
        valid = np.isfinite(raw_pvals)  # Only correct valid p-values
        if np.any(valid):
            # Apply FDR-BH correction to all valid p-values together
            _, corr_vals = apply_multiple_testing_correction(
                p_values=raw_pvals[valid].tolist(),
                test_methods=[t for t, v in zip(test_methods, valid) if v],
                variable_names=[v for v, ok in zip(variable_names, valid) if ok],
            )
            corrected_pvals[valid] = corr_vals

        # === BUILD OUTPUT STRUCTURES ===
        # sig_map: nested dict for plotting {covariate: {comparison: fdr_pval}}
        sig_map = {}
        # log_rows: list of dicts for CSV export
        log_rows = []
        for (row_title, cov_key, comp_name, pval_raw, stat_val, test_name), pval_corr in zip(tests, corrected_pvals):
            # Add to significance map for downstream plotting
            sig_map.setdefault(cov_key, {})[comp_name] = pval_corr
            # Add to log table
            log_rows.append({
                'row_title': row_title,
                'covariate': cov_key,
                'comparison': comp_name,
                'test_name': test_name,
                'test_statistic': stat_val,
                'p_value_raw': pval_raw,
                'p_value_fdr': pval_corr,
            })
        log_df = pd.DataFrame(log_rows)
        return sig_map, log_df

    def _plot_base_covariates(row_axes, df, x_col, motion_col, sig_map, category_colors):
        # === HELPER: PLOT AGE, SEX, AND MOTION COVARIATES ===
        # Create violin plots for continuous variables (age, motion) and count plot for sex
        # Annotate with FDR-corrected significance brackets
        
        # === MOTION LABEL MAPPING ===
        # Map internal motion column names to human-readable labels for plot legends
        motion_label_map = {
            'p24441_i2': 'rfMRI-derived',  # Functional MRI motion
            'p24453_i2': 'dMRI-derived',   # Diffusion MRI motion
        }
        
        # === PREPARE GROUP ORDERING AND COLOR PALETTE ===
        # Define canonical order for groups (controls first, then clusters)
        order = ['Control', 'Cluster 0', 'Cluster 1'] if x_col == 'Cluster' else ['Control', 'Depression']
        # Filter to only present groups in data
        present = [o for o in order if o in df.get(x_col, pd.Series(dtype=str)).unique()]
        # Palette as a list ordered by present groups for x-based violin plots
        # Explicitly ensure Control is always green, not cluster-colored
        palette_list = []
        for k in present:
            if k == 'Control':
                palette_list.append("#2ca02c")  # Always use green for Control
            else:
                palette_list.append(category_colors.get(k, 'grey'))
        # === PANEL 1: AGE VIOLIN PLOT ===
        # Show distribution of age across groups with significance annotations
        if 'p21003_i2' in df.columns:
            # Use x-axis positioning so violins are side-by-side
            sns.violinplot(data=df, x=x_col, y='p21003_i2', order=present,
                           palette=palette_list, ax=row_axes[0])
            row_axes[0].set_xlabel('')
            row_axes[0].set_ylabel('')  # Clear y-label (set globally in figure)
            
            # === ANNOTATE WITH SIGNIFICANCE BRACKETS ===
            # Add brackets showing FDR-corrected p-values for group comparisons
            age_pvals = sig_map.get('Age', {})  # Get FDR p-values from test results
            comp_dicts = [{'label_prefix': None, 'comparisons': {
                'depression_vs_control': ('Depression', 'Control'),
                'cluster0_vs_control': ('Cluster 0', 'Control'),
                'cluster1_vs_control': ('Cluster 1', 'Control'),
                'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
            }, 'pvals': age_pvals}]
            _annotate_axis(row_axes[0], present, comp_dicts)
        else:
            row_axes[0].set_visible(False)  # Hide panel if age not available

        # === PANEL 2: SEX COUNT PLOT ===
        # Show counts of male/female across groups (categorical variable)
        hue_col = 'p31'  # Sex column (typically 0=female, 1=male)
        if hue_col in df.columns:
            # Get unique sex values present in data
            hue_vals = sorted(pd.Series(df[hue_col]).dropna().unique())
            # For Group comparisons (Control vs Depression), use cohort colors for sex bars
            # For Cluster comparisons, use cluster-specific colors
            if x_col == 'Group':
                # Use Control green and Depression purple for sex value bars
                sex_palette = ["#2ca02c", "#6a3d9a"][:len(hue_vals)]
            else:
                # Use cluster colors for sex value bars when comparing clusters
                sex_palette = []
                for i in range(len(hue_vals)):
                    if i == 0:
                        sex_palette.append(category_colors.get('Cluster 0', 'grey'))
                    elif i == 1:
                        sex_palette.append(category_colors.get('Cluster 1', 'grey'))
                    else:
                        sex_palette.append('grey')
            # Create count plot showing sex distribution across groups
            sns.countplot(data=df, x=x_col, hue=hue_col, order=present, palette=sex_palette, ax=row_axes[1])
            # Keep legend to indicate sex values (important for categorical interpretation)
        else:
            row_axes[1].set_visible(False)  # Hide if sex not available
        row_axes[1].set_title('')  # Clear title (set globally)
        row_axes[1].set_ylabel('')  # Clear y-label
        
        # === ANNOTATE WITH SIGNIFICANCE BRACKETS ===
        sex_pvals = sig_map.get('Sex', {})
        if sex_pvals and present:
            # Get max count for y-axis scaling
            counts = pd.crosstab(df[x_col], df[hue_col]) if hue_col in df.columns else pd.DataFrame()
            y_max = float(counts.values.max()) if not counts.empty else None
            comp_dicts = [{'label_prefix': None, 'comparisons': {
                'depression_vs_control': ('Depression', 'Control'),
                'cluster0_vs_control': ('Cluster 0', 'Control'),
                'cluster1_vs_control': ('Cluster 1', 'Control'),
                'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
            }, 'pvals': sex_pvals}]
            _annotate_axis(row_axes[1], present, comp_dicts, y_max=y_max)

        if isinstance(motion_col, (list, tuple)):
            motion_cols = [c for c in motion_col if c in df.columns]
            if motion_cols:
                motion_df = df[[x_col] + motion_cols].melt(
                    id_vars=x_col,
                    value_vars=motion_cols,
                    var_name='motion_metric',
                    value_name='motion_value',
                )
                motion_df['motion_metric'] = motion_df['motion_metric'].map(
                    lambda v: motion_label_map.get(v, v)
                )
                hue_order = [motion_label_map.get(c, c) for c in motion_cols]
                sns.violinplot(
                    data=motion_df,
                    x=x_col,
                    y='motion_value',
                    hue='motion_metric',
                    order=present,
                    hue_order=hue_order,
                    split=len(hue_order) == 2,
                    palette=['#4C72B0', '#DD8452'][: len(hue_order)],
                    legend=False,
                    ax=row_axes[2],
                )
                row_axes[2].legend(
                    title='Head Motion',
                    loc='upper right',
                    frameon=True,
                    fontsize=7,
                    title_fontsize=8,
                    borderaxespad=0.2,
                    handlelength=1.2,
                    labelspacing=0.3,
                )
                row_axes[2].set_title('')
                row_axes[2].set_ylabel('')
                comp_dicts = []
                for metric in motion_cols:
                    metric_key = f"Head Motion ({metric})"
                    pvals = sig_map.get(metric_key, {})
                    if not pvals:
                        continue
                    label_prefix = motion_label_map.get(metric, metric)
                    comp_dicts.append({
                        'label_prefix': label_prefix,
                        'comparisons': {
                            'depression_vs_control': ('Depression', 'Control'),
                            'cluster0_vs_control': ('Cluster 0', 'Control'),
                            'cluster1_vs_control': ('Cluster 1', 'Control'),
                            'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
                        },
                        'pvals': pvals,
                    })
                if comp_dicts:
                    _annotate_axis(row_axes[2], present, comp_dicts)
            else:
                row_axes[2].set_visible(False)
        elif motion_col in df.columns:
            # Single-column motion plot: use x-axis positioning for side-by-side violins
            sns.violinplot(data=df, x=x_col, y=motion_col, order=present,
                           palette=palette_list, ax=row_axes[2])
            row_axes[2].set_xlabel('')
            row_axes[2].set_title('')
            row_axes[2].set_ylabel('')
            motion_pvals = sig_map.get('Head Motion', {})
            comp_dicts = [{'label_prefix': None, 'comparisons': {
                'depression_vs_control': ('Depression', 'Control'),
                'cluster0_vs_control': ('Cluster 0', 'Control'),
                'cluster1_vs_control': ('Cluster 1', 'Control'),
                'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
            }, 'pvals': motion_pvals}]
            _annotate_axis(row_axes[2], present, comp_dicts)
        else:
            row_axes[2].set_visible(False)

    def _plot_icd(row_axes, df, x_col, sig_map, category_colors):
        if not icd_covariates:
            for ax in row_axes[3:]:
                ax.set_visible(False)
            return

        df_dep = df[df.get('Group') == 'Depression'].copy() if 'Group' in df.columns else df.copy()
        if x_col == 'Cluster' and 'Cluster' in df_dep.columns:
            df_dep = df_dep[df_dep['Cluster'] != 'Control']

        cluster_order = ['Cluster 0', 'Cluster 1']
        present_clusters = [c for c in cluster_order if c in df_dep.get('Cluster', pd.Series(dtype=str)).unique()]

        rows = []
        for cov in icd_covariates:
            if cov not in df_dep.columns:
                continue
            overall = int(df_dep[cov].eq(1).sum())
            rows.append({'ICD': str(cov), 'label': 'Depression', 'count': overall})
            for cl in present_clusters:
                cl_count = int(df_dep[df_dep.get('Cluster') == cl][cov].eq(1).sum())
                rows.append({'ICD': str(cov), 'label': cl, 'count': cl_count})

        if not rows:
            for ax in row_axes[3:]:
                ax.set_visible(False)
            return

        plot_df = pd.DataFrame(rows)
        icd_ax = row_axes[3]

        if x_col == 'Group':
            dep_df = plot_df[plot_df['label'] == 'Depression']
            if dep_df.empty:
                icd_ax.set_visible(False)
            else:
                sns.barplot(data=dep_df, x='ICD', y='count', color=category_colors.get('Depression', '#6a3d9a'), ax=icd_ax)
                icd_ax.set_title('')
                icd_ax.set_xlabel('ICD')
                icd_ax.set_ylabel('')
                icd_ax.tick_params(axis='x', rotation=45)
            for ax in row_axes[4:]:
                ax.set_visible(False)
            return

        palette = {'Depression': category_colors.get('Depression', '#6a3d9a')}
        for cl in present_clusters:
            palette[cl] = category_colors.get(cl, 'grey')

        sns.barplot(data=plot_df, x='ICD', y='count', hue='label', palette=palette, ax=icd_ax)
        icd_ax.set_title('')
        icd_ax.set_xlabel('ICD')
        icd_ax.set_ylabel('')
        icd_ax.tick_params(axis='x', rotation=45)
        if icd_ax.legend_:
            icd_ax.legend_.set_title('Label')
        icd_y_max = float(plot_df['count'].max()) if not plot_df.empty else None
        comp_dicts = []
        for cov in icd_covariates:
            cov_key = str(cov)
            pvals = sig_map.get(cov_key, {})
            if not pvals:
                continue
            comp_dicts.append({
                'label_prefix': cov_key,
                'comparisons': {
                    'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
                },
                'pvals': pvals,
            })
        if comp_dicts:
            present_groups = ['Cluster 0', 'Cluster 1']
            _annotate_axis(icd_ax, present_groups, comp_dicts, y_max=icd_y_max)
        for ax in row_axes[4:]:
            ax.set_visible(False)

    # Ensure a canonical `Group` column exists on the combined dataframe
    combined_df = _ensure_group_column(combined_df)
    row_configs = [{
        'title': 'By Group',
        'df': combined_df,
        'x_col': 'Group',
        'motion_col': ['p24441_i2', 'p24453_i2'],
        'approach': None,
        'cluster_colors': None,
    }]

    # Connectivity/direction types to iterate when building cluster row configs
    conn_types = list(conn_types or ['functional', 'structural', 'sfc'])
    dir_types = list(dir_types or ['internal', 'external'])

    for ct in conn_types:
        display_conn = _display_conn_type(ct)
        for dt in dir_types:
            cl_col = f"{ct}_{dt}_cluster"
            if cl_col not in combined_df.columns:
                continue

            df_cluster = combined_df.copy()
            df_cluster['Cluster'] = df_cluster[cl_col].apply(_normalize_cluster_label)
            title_suffix = f" {dt}" if dt else ""

            if ct == 'sfc':
                row_configs.append({
                    'title': f"By Cluster ({display_conn}{title_suffix})",
                    'df': df_cluster,
                    'x_col': 'Cluster',
                    'motion_col': ['p24441_i2', 'p24453_i2'],
                    'approach': f"{display_conn}{title_suffix}".strip(),
                    'cluster_colors': _cluster_colors_for_modality(ct, dt),
                })
            else:
                motion_col = motion_lookup.get(ct, motion_metric)
                row_configs.append({
                    'title': f"By Cluster ({display_conn}{title_suffix})",
                    'df': df_cluster,
                    'x_col': 'Cluster',
                    'motion_col': motion_col,
                    'approach': f"{display_conn}{title_suffix}".strip(),
                    'cluster_colors': _cluster_colors_for_modality(ct, dt),
                })

    for ct in sorted(set(available_types or [])):
        if not cohorts_dir:
            continue
        cluster_file = os.path.join(cohorts_dir, f'combined_cohort_F32_global_{ct}_connectivity_clusters.csv')
        if os.path.exists(cluster_file):
            df_ct = pd.read_csv(cluster_file)
            df_ct = _ensure_group_column(df_ct)
        else:
            continue
        row_configs.append({
            'title': f"By Cluster ({_display_conn_type(ct)})",
            'df': df_ct,
            'x_col': 'Cluster',
            'motion_col': motion_lookup.get(ct, motion_metric),
            'approach': f"{_display_conn_type(ct)}".strip(),
            'cluster_colors': _cluster_colors_for_modality(ct, 'internal'),
        })

    # Create the figure grid: one row per row_config and one column per covariate
    n_rows = len(row_configs)
    n_cols = len(covariate_columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)

    # Main plotting: for each row configuration compute tests, draw panels, collect summaries
    print("  Creating multi-row covariate plots...")
    summary_rows = []
    test_rows = []
    for row_idx, cfg in enumerate(row_configs):
        row_axes = axes[row_idx]
        row_sig, log_df = _compute_covariate_tests(cfg)
        if log_df is not None and not log_df.empty:
            row_slug = re.sub(r'[^0-9A-Za-z]+', '_', str(cfg.get('title', 'row'))).strip('_')
            log_path = os.path.join(out_dir, f"covariate_tests_{row_slug}.csv")
            log_df.to_csv(log_path, index=False)
            log_df = log_df.copy()
            log_df['row_title'] = cfg.get('title')
            test_rows.append(log_df)
        cluster_colors = cfg.get('cluster_colors') or _cluster_colors_for_modality('functional', 'internal')
        category_colors = dict(base_category_colors)
        category_colors.update({
            'Cluster 0': cluster_colors['0'],
            'Cluster 1': cluster_colors['1'],
        })
        _plot_base_covariates(row_axes, cfg['df'], cfg['x_col'], cfg['motion_col'], row_sig, category_colors)
        _plot_icd(row_axes, cfg['df'], cfg['x_col'], row_sig, category_colors)

        df_cfg = cfg['df']
        x_col = cfg['x_col']
        groups = list(pd.Series(df_cfg.get(x_col, pd.Series(dtype=str))).dropna().unique())
        covariate_specs = []
        if 'p21003_i2' in df_cfg.columns:
            covariate_specs.append(('Age', 'p21003_i2'))

        motion_col = cfg.get('motion_col')
        if isinstance(motion_col, (list, tuple)):
            for col in motion_col:
                if col in df_cfg.columns:
                    covariate_specs.append((f"Head Motion ({col})", col))
        elif isinstance(motion_col, str) and motion_col in df_cfg.columns:
            covariate_specs.append(('Head Motion', motion_col))

        if 'p31' in df_cfg.columns:
            covariate_specs.append(('Sex', 'p31'))

        for cov in icd_covariates:
            if cov in df_cfg.columns:
                covariate_specs.append((str(cov), str(cov)))

        for cov_name, cov_col in covariate_specs:
            rows = _summarize_covariate_values(df_cfg, x_col, cov_name, cov_col, groups)
            for row in rows:
                row['row_title'] = cfg.get('title')
                row['x_col'] = x_col
            summary_rows.extend(rows)

    # Shared column headers to reduce duplicate titles
    for col_idx, col_name in enumerate(covariate_columns):
        ax = axes[0, col_idx]
        if col_name == 'Age':
            title = 'Age (years)'
        elif col_name == 'Sex':
            title = 'Sex (counts)'
        elif col_name == 'Head Motion':
            title = 'Mean Relative Head Motion (mm)'
        elif col_name not in ('Age', 'Sex', 'Head Motion'):
            title = 'ICD-10 Comorbidity (counts)'
        else:
            title = col_name
        ax.set_title(_wrap_title(title), fontsize=10, fontweight='bold')

    # Row-level labels for clustering approach (skip first row)
    for row_idx, cfg in enumerate(row_configs):
        approach = cfg.get('approach')
        if not approach:
            continue
        ax = axes[row_idx, 0]
        ax.annotate(
            approach,
            xy=(0.0, 1.12),
            xycoords='axes fraction',
            ha='left',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    fig.subplots_adjust(hspace=0.4 if n_rows > 1 else 0.3)
    out_name = 'modular_covariates_by_group_clusters_all.png'

    fig.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved covariate distribution plots")

    # If numeric summaries were collected, merge with per-row test outputs and write a table
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if test_rows:
            tests_df = pd.concat(test_rows, ignore_index=True)
            tests_df = tests_df.rename(columns={
                'covariate': 'covariate_name',
            })
            summary_df = summary_df.merge(
                tests_df,
                how='left',
                left_on=['row_title', 'covariate'],
                right_on=['row_title', 'covariate_name'],
            )
            summary_df = summary_df.drop(columns=[c for c in ['covariate_name'] if c in summary_df.columns])

        table_path = os.path.join(out_dir, 'modular_covariate_distribution_summary.txt')
        # Write the merged summary as a human-readable fixed-width table
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(summary_df.to_string(index=False))
            f.write("\n")
        print(f"  Saved covariate distribution summary table to: {table_path}")

def plot_cluster_feature_brainmaps(
    final_df,
    feature_df,
    atlas_img_path,
    community_labels,
    fig_dir,
    modules=None,
    alpha=0.85,
    background_white=True,
    output_basename_prefix='F32',
    minmax=True,
    significance_map=None,
    pvalue_threshold=0.05,
    save_niftis=True,
    nifti_dir=None,
    nifti_suffix='.nii.gz',
):
    """Plot cluster-level module profile differences as brain maps.

    For each connectivity type and direction:
        - optionally min-max scale module features to [0, 1] across subjects (per feature)
        - compute cluster-wise median profiles for Cluster 0 and Cluster 1
        - plot Cluster 0 and Cluster 1 side-by-side and (Cluster 0 - Cluster 1) maps
        - include only modules with significant cluster0_vs_cluster1 corrected p-values

    Parameters
    ----------
    final_df : pd.DataFrame
        Must contain 'eid' and cluster columns named '{conn_type}_{dir_type}_cluster'.
    feature_df : pd.DataFrame
        Must contain per-subject features with columns like:
        '<module>_internal_{connectivity_type}', '<module>_external_{connectivity_type}'.
        If it contains 'eid', it will be aligned to final_df by eid.
    atlas_img_path : str
        Path to a parcellation atlas NIfTI where unique non-zero integers identify ROIs.
    community_labels : array-like
        Length-N array mapping ROI index -> module/community label.
        ROI index corresponds to ordering of atlas region codes (sorted unique > 0).
    fig_dir : str
        Directory to save output figures.
    modules : list or None, default=None
        Optional module ordering (strings or numbers). If None, inferred from feature columns.
    alpha : float, default=0.85
        Overlay transparency for plot_stat_map (0=transparent, 1=opaque).
    background_white : bool, default=True
        If True, use white background (black_bg=False). If False, black background.
    output_basename_prefix : str, default='F32'
        Filename prefix for saved figures.
    minmax : bool, default=True
        If True, min-max scale features to [0, 1] before computing cluster medians.
    significance_map : dict, optional
        Mapping {(conn_type, dir_type, module): p_value} for cluster0_vs_cluster1 comparisons.
        When provided, only significant modules (p <= pvalue_threshold) are plotted.
    pvalue_threshold : float, default=0.05
        Significance threshold for filtering modules when significance_map is provided.
    save_niftis : bool, default=True
        If True, save NIfTI masks in addition to PNG figures.
    nifti_dir : str or None, default=None
        Directory for saving NIfTI files. If None, uses fig_dir/nifti_masks/.
    nifti_suffix : str, default='.nii.gz'
        File extension for saved NIfTI files.

    Notes
    -----
    Creates three types of brain maps per (connectivity_type, direction) pair:
    - Side-by-side Cluster 0 and Cluster 1 median profiles
    - Difference map (Cluster 0 - Cluster 1)
    Uses fixed slice coordinates (center of mass) for consistency across all maps.
    """

    # === CREATE OUTPUT DIRECTORIES ===
    # Ensure figure directory exists
    os.makedirs(fig_dir, exist_ok=True)
    # Create NIfTI output directory if saving volumetric masks
    if save_niftis:
        nifti_dir = nifti_dir or os.path.join(fig_dir, 'nifti_masks')
        os.makedirs(nifti_dir, exist_ok=True)

    # === LOAD ATLAS AND EXTRACT REGION CODES ===
    # Load parcellation atlas NIfTI (each unique integer = one ROI)
    atlas_img = image.load_img(atlas_img_path)
    atlas_data = atlas_img.get_fdata()  # 3D array of region codes

    # Convert community labels to numpy array (one module assignment per ROI)
    labels = np.asarray(community_labels)
    # Extract unique non-zero region codes from atlas (sorted unique integers)
    region_codes = np.unique(atlas_data)
    region_codes = region_codes[region_codes > 0]  # Exclude background (0)

    # === VALIDATE ATLAS-LABEL ALIGNMENT ===
    # Ensure number of ROIs matches number of community labels
    if region_codes.size != labels.size:
        print(
            f"[WARN] Atlas unique region count ({region_codes.size}) != community_labels length ({labels.size}). "
            "Mapping may be approximate."
        )
        # Handle mismatch by truncating or padding
        if region_codes.size > labels.size:
            region_codes = region_codes[: labels.size]  # Use only first N codes
        else:
            # Pad region codes with new unique codes to match all ROI indices
            extra = np.arange(region_codes.max() + 1, region_codes.max() + 1 + (labels.size - region_codes.size))
            region_codes = np.concatenate([region_codes, extra])

    # === CREATE ROI INDEX TO ATLAS CODE MAPPING ===
    # Map each ROI index (0-based) to its corresponding atlas integer code
    roi_to_code = dict(zip(range(labels.size), region_codes[: labels.size]))

    # === CHOOSE FIXED SLICE COORDINATES FOR CONSISTENT VISUALIZATION ===
    # Use center of mass of all non-zero atlas voxels as reference point
    # This ensures all brain maps (across clusters and directions) show same anatomical slices
    nonzero = np.argwhere(atlas_data > 0)  # Get all non-background voxel coordinates
    if nonzero.size > 0:
        center_vox = nonzero.mean(axis=0)  # Compute centroid in voxel space
        # Transform voxel coordinates to world (MNI) coordinates
        center_world = nib.affines.apply_affine(atlas_img.affine, center_vox)
        cut_coords = tuple(center_world.tolist())  # x, y, z in mm
    else:
        cut_coords = None  # Fallback if atlas is empty

    # === DEFINE CONNECTIVITY TYPES AND DIRECTIONS TO PROCESS ===
    conn_types = ['functional', 'structural', 'sfc']  # All modalities
    directions = ['internal', 'external']  # Within-module vs between-module

    def _build_map(values_vec, module_to_pos, labels_arr, atlas_vals, roi_code_map):
        # === HELPER: MAP MODULE VALUES TO ATLAS SPACE ===
        # Convert vector of module-level values to full 3D brain volume
        # Each ROI gets assigned the value of its module
        map_vals = np.zeros_like(atlas_vals, dtype=float)  # Initialize empty brain volume
        for roi_idx in range(labels_arr.size):  # Loop over all ROIs
            comm = str(labels_arr[roi_idx])  # Get module assignment for this ROI
            pos = module_to_pos.get(comm)  # Get position in values vector
            if pos is None:
                continue  # Skip if module not in feature set
            code = roi_code_map.get(roi_idx)  # Get atlas integer code for this ROI
            if code is None:
                continue  # Skip if ROI index unmapped
            # Assign module value to all voxels with this region code
            map_vals[atlas_vals == code] = float(values_vec[pos])
        return map_vals

    def _save_nifti(map_data, out_path):
        # === HELPER: SAVE 3D MAP AS NIFTI FILE ===
        if not save_niftis:
            return  # Skip if NIfTI saving disabled
        # Create NIfTI image with same affine and header as atlas
        img = nib.Nifti1Image(map_data, atlas_img.affine, header=atlas_img.header.copy())
        try:
            nib.save(img, out_path)
            print(f"Saved {out_path}")
        except Exception as exc:
            print(f"[WARN] Failed to save NIfTI {out_path}: {exc}")

    def _plot_map(values_vec, title, cmap, vmin, vmax, out_path, colorbar_label=None, nifti_out_path=None):
        # === HELPER: CREATE SINGLE BRAIN MAP FIGURE ===
        # Plot one module profile as orthogonal brain slices with colorbar
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5))
        fig.subplots_adjust(left=0.02, right=0.90, bottom=0.02, top=0.90, wspace=0.02, hspace=0.06)
        # Build 3D brain volume from module values
        map_data = _build_map(values_vec, module_to_pos, labels, atlas_data, roi_to_code)
        # Create NIfTI image
        img = nib.Nifti1Image(map_data, atlas_img.affine, header=atlas_img.header.copy())
        # Plot orthogonal slices (axial, sagittal, coronal)
        plotting.plot_stat_map(
            img,
            display_mode='ortho',  # Show all three standard views
            black_bg=not background_white,  # Background color
            colorbar=False,  # Add colorbar separately for custom positioning
            cmap=cmap,  # Color map (e.g., YlOrBr for single maps, RdBu_r for differences)
            vmin=vmin,  # Min value for color scale
            vmax=vmax,  # Max value for color scale
            cut_coords=cut_coords,  # Fixed slice coordinates for consistency
            transparency=alpha,  # Overlay opacity
            title=textwrap.fill(title, width=45),  # Wrap long titles
            axes=ax,
        )
        ax.set_axis_off()  # Remove axis borders

        # === ADD CUSTOM COLORBAR ===
        # Create scalar mappable for colorbar
        sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])  # Required for matplotlib compatibility
        # Add colorbar axis to right side of figure
        cax = fig.add_axes([0.915, 0.12, 0.02, 0.76])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cax)
        if colorbar_label:
            cbar.set_label(colorbar_label)  # Add colorbar label
        try:
            cb_ticks = list(np.linspace(vmin, vmax, 5))  # 5 evenly spaced ticks
            cbar.set_ticks(cb_ticks)
        except Exception:
            pass  # Use default ticks if custom ticks fail

        # === SAVE FIGURE ===
        fig.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)
        print(f"Saved {out_path}")
        # Optionally save NIfTI volume
        if nifti_out_path:
            _save_nifti(map_data, nifti_out_path)

    def _plot_pair(values_vec_a, values_vec_b, title_a, title_b, cmap, vmin, vmax, out_path,
                   colorbar_label=None, fig_title=None, nifti_out_a=None, nifti_out_b=None):
        # === HELPER: CREATE SIDE-BY-SIDE BRAIN MAP FIGURE ===
        # Plot two module profiles (e.g., Cluster 0 vs Cluster 1) as adjacent orthogonal brain slices
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5))  # Two panels side-by-side
        fig.subplots_adjust(left=0.02, right=0.90, bottom=0.02, top=0.86, wspace=0.02, hspace=0.06)
        if fig_title:
            fig.suptitle(fig_title, fontsize=12, fontweight='bold')  # Overall figure title

        # === BUILD BRAIN VOLUMES FOR BOTH PROFILES ===
        map_data_a = _build_map(values_vec_a, module_to_pos, labels, atlas_data, roi_to_code)
        map_data_b = _build_map(values_vec_b, module_to_pos, labels, atlas_data, roi_to_code)
        
        # === PLOT BOTH BRAIN MAPS WITH SAME COLORMAP/SCALE ===
        for ax, map_data, title in zip(axes, [map_data_a, map_data_b], [title_a, title_b]):
            # Create NIfTI image for this profile
            img = nib.Nifti1Image(map_data, atlas_img.affine, header=atlas_img.header.copy())
            # Plot orthogonal slices
            plotting.plot_stat_map(
                img,
                display_mode='ortho',  # Axial + sagittal + coronal views
                black_bg=not background_white,
                colorbar=False,  # Use shared colorbar instead
                cmap=cmap,  # Same colormap for both (ensures comparability)
                vmin=vmin,  # Same color scale min
                vmax=vmax,  # Same color scale max
                cut_coords=cut_coords,  # Same anatomical slices for both
                transparency=alpha,
                title=title,
                axes=ax,
            )
            ax.set_axis_off()

        # === ADD SHARED COLORBAR FOR BOTH PANELS ===
        sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.915, 0.12, 0.02, 0.76])  # Colorbar on right side
        cbar = fig.colorbar(sm, cax=cax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        try:
            cb_ticks = list(np.linspace(vmin, vmax, 5))
            cbar.set_ticks(cb_ticks)
        except Exception:
            pass

        # === SAVE FIGURE ===
        fig.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)
        print(f"Saved {out_path}")
        # Optionally save NIfTI volumes for both profiles
        if nifti_out_a:
            _save_nifti(map_data_a, nifti_out_a)
        if nifti_out_b:
            _save_nifti(map_data_b, nifti_out_b)

    # === INITIALIZE SIGNIFICANCE MAP ===
    significance_map = significance_map or {}  # Empty dict if not provided

    # === MAIN LOOP: ITERATE OVER CONNECTIVITY TYPES ===
    for conn_type in conn_types:
        # === DEFINE COLORMAPS ===
        cmap_single = 'YlOrBr'  # Yellow-Orange-Brown for single cluster maps
        cmap_diff = 'RdBu_r'  # Red-Blue diverging for difference maps
        
        # === BUILD FEATURE LISTS PER DIRECTION ===
        # Collect features for internal and external separately, then pool for scaling
        # This matches the scaling strategy in plot_module_violin_across_clusters
        dir_to_features = {}  # Maps direction -> list of feature column names
        pooled_features = []  # All features for this conn_type (for global min-max scaling)
        
        for dir_type in directions:
            # === IDENTIFY RELEVANT FEATURE COLUMNS ===
            # Pattern: '<module>_internal_functional', '<module>_external_structural', etc.
            pattern = f"_{dir_type}_{conn_type}"
            sel_cols = [c for c in feature_df.columns if pattern in c]
            if not sel_cols:
                continue  # No features for this direction

            # === INFER MODULE LIST ===
            if modules is None:
                # Extract module IDs from column names (part before first underscore)
                modules_local = sorted({c.split('_', 1)[0] for c in sel_cols}, key=lambda x: x)
            else:
                modules_local = list(modules)  # Use provided module list

            # === FILTER TO SIGNIFICANT MODULES ONLY ===
            # If significance_map provided, keep only modules with p <= threshold
            if significance_map:
                sig_modules = []  # Modules passing significance threshold
                for m in modules_local:
                    # Look up FDR-corrected p-value for this module
                    pval = significance_map.get((conn_type, dir_type, str(m)))
                    if pval is None:
                        continue  # No test result for this module
                    try:
                        pval_float = float(pval)
                    except (TypeError, ValueError):
                        continue  # Invalid p-value
                    # Keep if p-value is significant
                    if np.isfinite(pval_float) and pval_float <= pvalue_threshold:
                        sig_modules.append(m)
                
                # Report dropped modules
                dropped = sorted(set(modules_local) - set(sig_modules))
                if dropped:
                    print(
                        f"[INFO] Dropping {len(dropped)} modules for {conn_type} {dir_type} due to non-significance: "
                        f"{dropped[:10]}{'...' if len(dropped)>10 else ''}"
                    )
                modules_local = sig_modules  # Update to only significant modules

            # === BUILD ORDERED FEATURE COLUMN LIST ===
            ordered_features = [f"{m}_{dir_type}_{conn_type}" for m in modules_local]
            ordered_features = [c for c in ordered_features if c in feature_df.columns]
            if not ordered_features:
                continue  # No valid features after filtering

            # Store for this direction and add to pooled list
            dir_to_features[dir_type] = ordered_features
            pooled_features.extend(ordered_features)

        # === DEDUPLICATE POOLED FEATURES ===
        pooled_features = list(dict.fromkeys(pooled_features))  # Remove duplicates, preserve order
        if not pooled_features:
            print(f"No features for {conn_type}; skipping.")
            continue  # Skip this connectivity type if no features

        # === FIT GLOBAL MIN-MAX SCALER FOR THIS CONNECTIVITY TYPE ===
        # Scale all features (internal + external) together to [0, 1] range
        pooled_scaler = None
        if minmax:
            # Extract all pooled feature values, fill missing with median
            pooled_vals = feature_df[pooled_features].copy()
            pooled_vals = pooled_vals.fillna(pooled_vals.median()).astype(float)
            # Fit scaler on flattened values (treat all features as one distribution)
            pooled_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            pooled_scaler.fit(pooled_vals.values.reshape(-1, 1))  # Flatten to 1D for global fit

        # === PROCESS EACH DIRECTION (INTERNAL, EXTERNAL) ===
        for dir_type in directions:
            ordered_features = dir_to_features.get(dir_type)
            if not ordered_features:
                print(f"No features for {conn_type} {dir_type}; skipping.")
                continue

            # === PREPARE FEATURE MATRIX WITH OPTIONAL MIN-MAX SCALING ===
            # Fill missing values with median before scaling
            X = feature_df[ordered_features].copy().fillna(feature_df[ordered_features].median())
            if minmax:
                # Apply global scaler fitted on pooled features
                X_vals = X.astype(float)
                X_scaled = pooled_scaler.transform(X_vals.values.reshape(-1, 1)).reshape(X_vals.values.shape)
                X_z = pd.DataFrame(X_scaled, columns=ordered_features, index=feature_df.index)
            else:
                X_z = X.reset_index(drop=True)  # No scaling

            # === VERIFY CLUSTER COLUMN EXISTS ===
            cluster_col = f"{conn_type}_{dir_type}_cluster"  # e.g., 'functional_internal_cluster'
            if cluster_col not in final_df.columns:
                print(f"{cluster_col} not found in final_df; skipping {conn_type} {dir_type}.")
                continue

            # === ALIGN FEATURES WITH CLUSTER LABELS ===
            if 'eid' in final_df.columns and 'eid' in feature_df.columns:
                X_z = X_z.set_index(feature_df['eid']).reindex(final_df['eid']).reset_index(drop=True)
            else:
                X_z = X_z.reset_index(drop=True)

            # sanity check: aligned feature matrix must have same number of rows as final_df
            if X_z.shape[0] != final_df.shape[0]:
                raise ValueError(f"Row-misalignment after reindexing: X_z has {X_z.shape[0]} rows but final_df has {final_df.shape[0]} rows. Check 'eid' alignment between feature_df and final_df.")

            cluster_vals = final_df[cluster_col].dropna().astype(str)
            if cluster_vals.empty:
                print(f"No clusters found for {conn_type} {dir_type}; skipping.")
                continue

            # normalize cluster labels to simple numeric strings ('0','1',...)
            def _normalize_cluster_label(s):
                if pd.isna(s):
                    return s
                s = str(s).strip()
                m = re.search(r"(\d+)", s)
                if m:
                    return m.group(1)
                try:
                    return str(int(float(s)))
                except Exception:
                    return s

            cluster_series = final_df[cluster_col].apply(_normalize_cluster_label).astype(object)

            # enforce Cluster 0 and Cluster 1 presence
            mask_c0 = (cluster_series == '0').values
            mask_c1 = (cluster_series == '1').values
            if mask_c0.sum() == 0 or mask_c1.sum() == 0:
                print(f"Missing Cluster 0/1 for {conn_type} {dir_type}; skipping.")
                continue

            # cluster-wise medians for cluster 0 and 1
            med_c0 = X_z.loc[mask_c0, :].median(axis=0).values
            med_c1 = X_z.loc[mask_c1, :].median(axis=0).values
            diff_vec = med_c0 - med_c1

            # module mapping
            n_modules = len(ordered_features)
            module_ids = [ordered_features[i].split('_', 1)[0] for i in range(0, len(ordered_features))]
            module_to_pos = {str(m): i for i, m in enumerate(module_ids)}

            # warn if some atlas community labels do not map to any module position
            label_keys = {str(l) for l in labels}
            missing_labels = label_keys - set(module_to_pos.keys())
            if missing_labels:
                # do not fail, but log a warning with counts
                print(f"[WARN] {len(missing_labels)} atlas community labels have no matching module in selected modules for {conn_type} {dir_type} (examples: {list(missing_labels)[:5]})")

            out_dir_conn = os.path.join(fig_dir, f'{conn_type}_con')
            os.makedirs(out_dir_conn, exist_ok=True)

            # Cluster-specific maps (min-max scaling already applied to features)
            vmin_single, vmax_single = 0.0, 1.0
            out_path_pair = os.path.join(
                out_dir_conn,
                f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster0_vs_cluster1_brainmaps.png",
            )
            nifti_out_c0 = None
            nifti_out_c1 = None
            if save_niftis:
                nifti_out_c0 = os.path.join(
                    nifti_dir,
                    f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster0_mask{nifti_suffix}",
                )
                nifti_out_c1 = os.path.join(
                    nifti_dir,
                    f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster1_mask{nifti_suffix}",
                )
            _plot_pair(
                med_c0,
                med_c1,
                f"Cluster 0 — {dir_type.capitalize()}",
                f"Cluster 1 — {dir_type.capitalize()}",
                cmap_single,
                vmin_single,
                vmax_single,
                out_path_pair,
                colorbar_label="Median value",
                fig_title=f"Median cluster {_display_conn_type_text(conn_type)} profiles across modules",
                nifti_out_a=nifti_out_c0,
                nifti_out_b=nifti_out_c1,
            )

            # Difference map (Cluster 0 - Cluster 1), symmetric around zero
            finite_vals = diff_vec[np.isfinite(diff_vec)]
            if finite_vals.size > 0:
                abs_max = float(np.round(np.max(np.abs(finite_vals)), 2))
            else:
                abs_max = 1.0
            if abs_max == 0.0:
                abs_max = 0.5
            vmin_diff, vmax_diff = float(-abs_max), float(abs_max)

            out_path_diff = os.path.join(out_dir_conn, f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster0_minus_cluster1_brainmap.png")
            nifti_out_diff = None
            if save_niftis:
                nifti_out_diff = os.path.join(
                    nifti_dir,
                    f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster0_minus_cluster1_mask{nifti_suffix}",
                )
            _plot_map(
                diff_vec,
                f"Difference between median cluster {_display_conn_type_text(conn_type)} profiles across modules",
                cmap_diff,
                vmin_diff,
                vmax_diff,
                out_path_diff,
                colorbar_label="Difference value",
                nifti_out_path=nifti_out_diff,
            )

def plot_module_metric_distributions(
    final_df: pd.DataFrame,
    mods: List[str],
    plots_dir: str,
) -> None:
    """Plot per-module histograms for each metric type by cohort.

    Parameters
    ----------
    final_df : pd.DataFrame
        DataFrame with 'depression_status' and module feature columns.
    mods : list of str
        List of module identifiers to plot.
    plots_dir : str
        Output directory for saving histogram figures.

    Notes
    -----
    Creates separate multi-panel histogram figures for each metric type:
    - internal_functional, external_functional
    - internal_structural, external_structural
    - internal_sfc, external_sfc
    Generates one figure per cohort (F32=depressed, control) per metric type.
    """
    # === MAIN LOOP: ITERATE OVER COHORTS ===
    # Create separate histogram figures for depressed (F32) and control groups
    for cohort_label, depression_status in [('F32', 1), ('control', 0)]:
        print(f"Processing cohort {cohort_label} for module connectivity distributions...")
        
        # === ITERATE OVER CONNECTIVITY METRIC TYPES ===
        # For each combination of direction (internal/external) and modality (functional/structural/SFC)
        for metric_type in [
            'internal_functional',
            'external_functional',
            'internal_structural',
            'external_structural',
            'internal_sfc',
            'external_sfc',
        ]:
            # === CREATE MULTI-PANEL HISTOGRAM FIGURE ===
            plt.figure(figsize=(15, 10))  # Large figure for multi-panel grid
            num_modules = len(mods)
            n_cols = 4  # 4 columns of histograms
            n_rows = (num_modules + n_cols - 1) // n_cols  # Calculate required rows

            # === PLOT ONE HISTOGRAM PER MODULE ===
            for i, mod in enumerate(mods):
                # Build column name: '<module>_<metric_type>' (e.g., '0_internal_functional')
                col_name = f"{mod}_{metric_type}"
                if col_name not in final_df.columns:
                    continue  # Skip if this module-metric combo doesn't exist
                
                # Create subplot for this module
                plt.subplot(n_rows, n_cols, i + 1)
                # Plot histogram with KDE overlay for this cohort
                sns.histplot(
                    final_df.loc[final_df['depression_status'] == depression_status, col_name],
                    bins=30,  # 30 histogram bins
                    kde=True,  # Add kernel density estimate overlay
                )
                plt.title(f'Module {mod}')  # Module identifier as panel title
                plt.xlabel('Connectivity Value')
                plt.ylabel('Frequency')

            # === FINALIZE AND SAVE FIGURE ===
            plt.tight_layout()  # Adjust spacing to prevent overlap
            plt.suptitle(
                f'Distribution of {metric_type.replace("_", " ").title()} Across Modules',
                y=1.02,  # Position title above panels
                fontsize=16,
            )
            out_path = os.path.join(plots_dir, f'{cohort_label}_{metric_type}_distributions.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
            plt.close()


def plot_module_correlation_matrices(final_df: pd.DataFrame, plots_dir: str) -> None:
    """Plot FDR-significant Spearman correlation matrices by cohort.

    Parameters
    ----------
    final_df : pd.DataFrame
        DataFrame with 'eid', 'depression_status', and module feature columns.
    plots_dir : str
        Output directory for saving correlation heatmaps.

    Notes
    -----
    - Computes Spearman correlations across all module features separately
      for depressed (F32) and control cohorts.
    - Applies FDR-BH correction to all pairwise correlations.
    - Only displays correlations that pass FDR significance threshold (p < 0.05).
    - Skips cohorts with insufficient data or no significant correlations.
    """
    # === MAIN LOOP: ITERATE OVER COHORTS ===
    # Create separate correlation matrices for depressed (F32) and control groups
    for cohort_label, depression_status in [('F32', 1), ('control', 0)]:
        print(f"Processing cohort {cohort_label} for module connectivity correlation matrix...")
        
        # === FILTER TO COHORT AND EXTRACT FEATURE COLUMNS ===
        # Remove metadata columns (eid, depression_status) to leave only connectivity features
        feature_df = final_df.loc[final_df['depression_status'] == depression_status].drop(
            columns=['eid', 'depression_status']
        ).copy()
        
        # === REORDER COLUMNS BY MODULE AND CONNECTIVITY TYPE ===
        # Group columns by connectivity type (functional, structural, SFC) for easier interpretation
        try:
            # Extract all feature column names
            cols_all = feature_df.columns.tolist()
            # Infer module IDs from column names (part before first underscore)
            mods_for_order = sorted(
                {c.split('_', 1)[0] for c in cols_all},
                key=lambda v: (float(v) if _is_number(v) else v),  # Sort numerically if possible
            )
            # Rebuild column order: for each conn_type, list all modules' internal then external features
            new_order = []
            for conn in ['functional', 'structural', 'sfc']:
                for m in mods_for_order:
                    # Add internal_{conn} column for this module if it exists
                    for suf in [f"{m}_internal_{conn}", f"{m}_external_{conn}"]:
                        if suf in feature_df.columns:
                            new_order.append(suf)
            if new_order:
                feature_df = feature_df.reindex(columns=new_order)  # Apply new column order
        except Exception:
            pass  # If reordering fails, use original order

        # === REMOVE ROWS WITH MISSING DATA ===
        # Correlation requires complete cases (no NaNs)
        feature_df_clean = feature_df.dropna(axis=0, how='any')
        if feature_df_clean.shape[0] < 3:
            print(f"Skipping correlation matrix for {cohort_label}: not enough complete cases.")
            continue  # Need at least 3 subjects for meaningful correlation

        # === COMPUTE SPEARMAN CORRELATION MATRIX ===
        # Spearman is non-parametric and robust to outliers
        corr_vals, pval_vals = sp_stats.spearmanr(feature_df_clean, axis=0)
        # Convert arrays to DataFrames with feature names as row/column labels
        corr_matrix = pd.DataFrame(corr_vals, index=feature_df_clean.columns, columns=feature_df_clean.columns)
        pval_matrix = pd.DataFrame(pval_vals, index=feature_df_clean.columns, columns=feature_df_clean.columns)

        # === APPLY FDR-BH CORRECTION TO P-VALUES ===
        # Extract upper triangle (avoid diagonal and duplicate pairs)
        tri_idx = np.triu_indices_from(pval_matrix.values, k=1)
        pvals = pval_matrix.values[tri_idx]  # All pairwise p-values
        valid = np.isfinite(pvals)  # Filter to valid p-values
        sig_mask = np.zeros_like(pval_matrix.values, dtype=bool)  # Boolean mask for significance
        if np.any(valid):
            # Run FDR-BH correction on valid p-values
            reject, _, _, _ = multitest.multipletests(pvals[valid], alpha=0.05, method='fdr_bh')
            sig = np.zeros_like(pvals, dtype=bool)
            sig[valid] = reject  # Mark rejected (significant) tests
            # Fill upper triangle of mask
            sig_mask[tri_idx] = sig
            # Mirror to lower triangle (correlation matrix is symmetric)
            sig_mask = sig_mask | sig_mask.T

        # === CHECK IF ANY CORRELATIONS ARE SIGNIFICANT ===
        if not sig_mask.any():
            print(f"No FDR-significant Spearman correlations for {cohort_label}; skipping heatmap.")
            continue

        # === CREATE HEATMAP SHOWING ONLY SIGNIFICANT CORRELATIONS ===
        # Mask out non-significant correlations (set to NaN)
        corr_sig = corr_matrix.where(sig_mask)
        plt.figure(figsize=(20, 16))  # Large figure for detailed correlation matrix
        sns.heatmap(
            corr_sig,
            cmap='coolwarm',  # Red-blue diverging colormap
            center=0,  # Center colormap at zero correlation
            annot=False,  # No text annotations (too many features)
            fmt=".2f",
            cbar_kws={"shrink": 0.8},  # Slightly smaller colorbar
            square=True,  # Force square cells
            linewidths=0.5,  # Grid lines between cells
            mask=~sig_mask,  # Hide non-significant correlations
        )
        plt.title(f'FDR-significant Spearman Correlations - {cohort_label}', fontsize=18)
        out_path = os.path.join(plots_dir, f'{cohort_label}_module_connectivity_correlation_matrix.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()

def plot_clustering_validation_metrics(
    validation_df: pd.DataFrame,
    plots_dir: str,
) -> None:
    """Plot silhouette and Calinski-Harabasz validation curves per modality and direction.

    Parameters
    ----------
    validation_df : pd.DataFrame
        DataFrame with columns: 'connectivity_type', 'direction_type',
        'num_clusters', 'silhouette_score', 'calinski_harabasz_score'.
    plots_dir : str
        Output directory for saving validation metric plots.

    Notes
    -----
    Creates two-panel line plots showing how silhouette and Calinski-Harabasz
    scores vary with the number of clusters (k). Separate figures generated
    for each (connectivity_type, direction_type) combination.
    """
    # === MAIN LOOP: ITERATE OVER CONNECTIVITY TYPES AND DIRECTIONS ===
    # Create separate validation plots for each modality-direction combination
    for conn_type in ['functional', 'structural', 'sfc']:
        for dir_type in ['internal', 'external']:
            # === FILTER VALIDATION RESULTS TO CURRENT MODALITY/DIRECTION ===
            subset = validation_df[
                (validation_df['connectivity_type'] == conn_type)
                & (validation_df['direction_type'] == dir_type)
            ]
            if subset.empty:
                continue  # Skip if no validation data for this combination
            
            # === CREATE TWO-PANEL LINE PLOT ===
            # Left panel: Silhouette score, Right panel: Calinski-Harabasz score
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # === LEFT PANEL: SILHOUETTE SCORE ===
            # Silhouette measures cluster cohesion and separation (-1 to 1, higher is better)
            axes[0].plot(subset['num_clusters'], subset['silhouette_score'], marker='o', label='Silhouette Score')
            
            # === RIGHT PANEL: CALINSKI-HARABASZ SCORE ===
            # CH score is ratio of between-cluster to within-cluster variance (higher is better)
            axes[1].plot(subset['num_clusters'], subset['calinski_harabasz_score'], marker='s', label='Calinski-Harabasz Score')
            
            # === FORMAT AXES ===
            axes[1].set_xlabel('Number of Clusters (k)')  # Only right panel gets x-label
            axes[0].set_ylabel('Score')
            axes[1].set_ylabel('Score')
            
            # === SET X-AXIS TICKS TO INTEGER K VALUES ===
            # Ensure x-axis shows only tested cluster numbers (e.g., 2, 3, 4, 5)
            try:
                xticks = subset['num_clusters'].astype(int).tolist()
                axes[0].set_xticks(xticks)
                axes[1].set_xticks(xticks)
                axes[0].set_xticklabels([str(int(x)) for x in xticks])
                axes[1].set_xticklabels([str(int(x)) for x in xticks])
            except Exception:
                pass  # Use default ticks if conversion fails
            
            # === ADD LEGENDS AND GRID ===
            axes[0].legend()
            axes[1].legend()
            axes[0].grid()  # Add gridlines for easier reading
            axes[1].grid()
            
            # === ADD OVERALL TITLE ===
            fig.suptitle(
                f'Clustering Validation Metrics - {_display_conn_type(conn_type)} {dir_type.capitalize()} Connectivity',
                fontsize=16,
            )
            
            # === SAVE FIGURE ===
            out_path = os.path.join(plots_dir, f'{conn_type}_con', f'{dir_type}_connectivity_clustering_validation.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
            plt.close()

# ==============================================================================
# HIGH-LEVEL PIPELINE ORCHESTRATION FUNCTIONS
# ==============================================================================
def compute_and_save_module_connectivity(
    subjects_dir: str,
    subject_ids: List[str],
    community_labels: np.ndarray,
    progress_every: int = 50,
    sfc_warning_log_dir: Optional[str] = None,
    cohort_label: Optional[str] = None,
) -> int:
    """Compute and save module connectivity and SFC CSV files for each subject.
    
    For each subject with available FC and SC matrices, computes:
    - Internal and external functional connectivity per module
    - Internal and external structural connectivity per module
    - Internal and external structure-function coupling per module
    
    Saves three CSV files per subject:
    - functional_module_connectivity.csv
    - structural_module_connectivity.csv
    - module_sfc.csv
    
    Parameters
    ----------
    subjects_dir : str
        Parent directory containing subject folders (organized as {eid}/i2/).
    subject_ids : list of str
        List of subject IDs (eids) to process.
    community_labels : np.ndarray
        Array of community/module labels for each ROI (length = number of ROIs).
    progress_every : int, default=50
        Print progress update every N subjects (0 to disable).
    
    Returns
    -------
    int
        Number of subjects successfully processed.
    
    Notes
    -----
    - Expects connectivity matrices named:
      {eid}_functional_connectivity_matrix_reordered.npy
      {eid}_structural_connectivity_matrix_reordered.npy
    - Skips subjects missing either matrix.
    - SFC is computed via rowwise Pearson correlation between SC and FC.
    - `sfc_warning_log_dir`: directory where per-subject SFC warning CSVs will
        be appended (internal and external logs). If None, no logs are written.
    - `cohort_label`: short label for the cohort (e.g., 'F32' or 'control')
        which will be included in the per-subject log records when logging is enabled.
    """
    # === PROCESS EACH SUBJECT ===
    # Loop through all subject IDs and compute module-level connectivity metrics
    processed = 0  # Counter for successful subjects
    total = len(subject_ids)
    
    for i, subject_id in enumerate(subject_ids, start=1):
        # === LOCATE SUBJECT'S CONNECTIVITY MATRICES ===
        # Expected directory structure: subjects_dir/subject_id/i2/
        subject_dir = os.path.join(subjects_dir, subject_id, 'i2')
        
        # Paths to functional and structural connectivity matrices
        # Matrices should be reordered to match atlas parcellation
        func_path = os.path.join(subject_dir, f'{subject_id}_functional_connectivity_matrix_reordered.npy')
        struct_path = os.path.join(subject_dir, f'{subject_id}_structural_connectivity_matrix_reordered.npy')
        
        # Skip subject if either matrix is missing
        if not os.path.exists(func_path) or not os.path.exists(struct_path):
            continue

        # === LOAD CONNECTIVITY MATRICES ===
        FC = np.load(func_path)  # Functional connectivity (correlation/covariance matrix)
        SC = np.load(struct_path)  # Structural connectivity (streamline/FA matrix)

        # === COMPUTE MODULE-LEVEL CONNECTIVITY ===
        # Extract internal (within-module) and external (between-module) connectivity
        # for both functional and structural modalities
        (
            module_FC_internal_conn_values,   # Raw FC values for internal edges
            module_FC_external_conn_values,   # Raw FC values for external edges  
            module_FC_internal_connectivity,  # Aggregated internal FC per module
            module_FC_external_connectivity,  # Aggregated external FC per module
            module_SC_internal_conn_values,   # Raw SC values for internal edges
            module_SC_external_conn_values,   # Raw SC values for external edges
            module_SC_internal_connectivity,  # Aggregated internal SC per module
            module_SC_external_connectivity,  # Aggregated external SC per module
        ) = compute_module_connectivity(FC, SC, community_labels)

        # === SAVE FUNCTIONAL AND STRUCTURAL MODULE CONNECTIVITY ===
        # For each connectivity type, write CSV with module-level summary statistics
        for conn_type, internal_conn, external_conn in [
            ('functional', module_FC_internal_connectivity, module_FC_external_connectivity),
            ('structural', module_SC_internal_connectivity, module_SC_external_connectivity),
        ]:
            # Create CSV file for this connectivity type
            output_file = os.path.join(subject_dir, f'{conn_type}_module_connectivity.csv')
            
            with open(output_file, 'w') as f:
                # Write header
                f.write('Module,Internal_Connectivity,External_Connectivity\n')
                
                # Write one row per module with internal/external connectivity values
                for module in internal_conn.keys():
                    f.write(f'{module},{internal_conn[module]},{external_conn[module]}\n')

        # === COMPUTE STRUCTURE-FUNCTION COUPLING (SFC) ===
        # SFC measures correspondence between SC and FC at module level
        # Computed via Pearson correlation between SC and FC edge weights
        module_internal_sfc, module_external_sfc = compute_module_sfc(
            module_FC_internal_conn_values,  # FC internal edge values
            module_FC_external_conn_values,  # FC external edge values
            module_SC_internal_conn_values,  # SC internal edge values
            module_SC_external_conn_values,  # SC external edge values
            sfc_warning_log_dir=sfc_warning_log_dir,
            subject_id=subject_id,
            cohort_label=cohort_label,
        )
        
        # === SAVE SFC MODULE CONNECTIVITY ===
        output_file_sfc = os.path.join(subject_dir, 'module_sfc.csv')
        
        with open(output_file_sfc, 'w') as f:
            # Write header
            f.write('Module,Internal_SFC,External_SFC\n')
            
            # Write one row per module with internal/external SFC values
            for module in module_internal_sfc.keys():
                f.write(f'{module},{module_internal_sfc[module]},{module_external_sfc[module]}\n')

        # === UPDATE PROGRESS ===
        processed += 1  # Increment successful subject counter
        
        # Print progress message at specified intervals
        if progress_every > 0 and (i % progress_every == 0 or i == total):
            print(f"  Progress: {i}/{total} subjects processed")
    
    # Return total number of subjects successfully processed
    return processed

def run_module_quantile_regression_pipeline(
    merged: pd.DataFrame,
    final_df: pd.DataFrame,
    mods: List[str],
    R: int,
    conn_types: Tuple[str, ...],
    dir_types: Tuple[str, ...],
    icd_covariates: List[str],
    fMRI_MOTION_METRIC: str,
    dMRI_MOTION_METRIC: str,
    plots_dir: str,
    depressed_subjects_dir: str,
    colname_map: Dict[str, str],
) -> Dict[Tuple[str, str, str], float]:
    """Run quantile regression for all modules and generate annotated violin plots.
    
    For each (connectivity type, direction, module) combination:
    1. Runs quantile regression (median) with 4 group comparisons:
       - Depression vs. Control
       - Cluster 0 vs. Control
       - Cluster 1 vs. Control
       - Cluster 0 vs. Cluster 1
    2. Applies FDR-BH correction across all tests within each (conn_type, dir_type)
    3. Generates violin plots with significance annotations
    4. Saves FDR correction logs
    
    Parameters
    ----------
    merged : pd.DataFrame
        Combined DataFrame with sanitized column names, covariates, cluster labels.
    final_df : pd.DataFrame
        Original DataFrame with unsanitized column names for plotting.
    mods : list of str
        Ordered list of module identifiers.
    R : int
        Number of bootstrap samples for quantile regression SEs and confidence intervals.
    conn_types : tuple of str
        Connectivity types to analyze ('functional', 'structural', 'sfc').
    dir_types : tuple of str
        Direction types to analyze ('internal', 'external').
    icd_covariates : list of str
        ICD-10 comorbidity column names to include as covariates.
    fMRI_MOTION_METRIC : str
        Column name for fMRI head motion metric.
    dMRI_MOTION_METRIC : str
        Column name for dMRI head motion metric.
    plots_dir : str
        Output directory for violin plots.
    depressed_subjects_dir : str
        Directory for saving FDR correction logs.
    colname_map : dict
        Mapping from original to sanitized column names.
    
    Returns
    -------
    dict
        Brainmap significance map: {(conn_type, dir_type, module): p_value_cluster0_vs_cluster1}
        for use in brain visualization filtering.
    
    Notes
    -----
    - Quantile regression uses R's quantreg package via rpy2.
    - All p-values within each (conn_type, dir_type) are FDR-corrected together.
    - Violin plots show min-max scaled connectivity distributions.
    - Only modules with significant cluster0_vs_cluster1 comparisons are highlighted
      in subsequent brain maps.
    """
    # === DEFINE GROUP COLUMN AND COMPARISON SEQUENCE ===
    # group_col: column distinguishing depressed (1) vs. control (0) subjects
    group_col = 'depression_status'
    
    # comparison_sequence: ordered list of group comparisons to test
    # Each tuple: (comparison_key, descriptive_label)
    # These comparisons test: main effect, cluster-specific effects, inter-cluster difference
    comparison_sequence = [
        ('depression_vs_control', 'QR (median) depression vs control'),      # Main effect
        ('cluster0_vs_control', 'QR (median) cluster 0 vs control'),         # Cluster 0 specificity
        ('cluster1_vs_control', 'QR (median) cluster 1 vs control'),         # Cluster 1 specificity
        ('cluster0_vs_cluster1', 'QR (median) cluster 0 vs cluster 1'),      # Inter-cluster difference
    ]

    # === INITIALIZE RESULTS STORAGE ===
    # brainmap_significance: stores cluster0_vs_cluster1 p-values for brain visualization
    # Only modules with significant inter-cluster differences will be highlighted
    brainmap_significance = {}
    
    # safe_to_orig: reverse mapping from sanitized column names back to original names
    # Needed for plotting with original (unsanitized) column names from final_df
    safe_to_orig = {v: k for k, v in colname_map.items()}

    # === ITERATE OVER CONNECTIVITY TYPES ===
    # For each connectivity modality (functional, structural, SFC), run full regression pipeline
    for conn_type in conn_types:
        # === GET MODALITY-SPECIFIC MOTION COVARIATES ===
        # Functional connectivity uses fMRI motion; structural/SFC use dMRI motion
        motion_dict = get_motion_columns(conn_type, fMRI_MOTION_METRIC, dMRI_MOTION_METRIC)
        motion_columns = list(motion_dict.values())
        
        # === INITIALIZE SIGNIFICANCE MAP FOR THIS MODALITY ===
        # significance_payload: stores FDR-corrected p-values for all modules in this modality
        # Key: (conn_type, dir_type, module) -> dict of comparison_name -> corrected_p_value
        significance_payload = {}

        # === ITERATE OVER DIRECTION TYPES ===
        # For each direction (internal/external connectivity), run regressions and apply FDR correction
        for dir_type in dir_types:
            # === SETUP FDR CORRECTION LOG FILE ===
            # Log file will contain all raw and corrected p-values for this (conn_type, dir_type)
            fdr_log_path = os.path.join(
                depressed_subjects_dir,
                f'modular_{conn_type}_{dir_type}_connectivity_FDR.txt',
            )
            
            # Extract comparison keys and labels from comparison_sequence
            comparison_keys = [key for key, _ in comparison_sequence]
            test_labels = [label for _, label in comparison_sequence]

            # === INITIALIZE COLLECTORS FOR FDR CORRECTION ===
            # We'll collect all raw p-values across all modules, then apply FDR correction once
            # This ensures family-wise error control across all modules in this (conn_type, dir_type)
            ordered_raw_p_all = []       # All raw p-values (flattened)
            test_labels_all = []         # Corresponding test labels
            variable_names_all = []      # Corresponding dependent variable names
            module_keys_all = []         # Corresponding module identifiers
            module_depvar_safe = {}      # Maps module_key -> sanitized dependent variable name

            # === ITERATE OVER MODULES ===
            # Run quantile regression for each module separately
            for m in mods:
                module_key = str(m)  # Module identifier (e.g., '1', '2', ...)
                
                # === CONSTRUCT VARIABLE NAMES ===
                # Dependent variable: module connectivity (e.g., '1_internal_functional')
                dependent_var = f'{m}_{dir_type}_{conn_type}'
                
                # Cluster column: cluster assignments for this modality (e.g., 'functional_internal_cluster')
                cluster_col = f'{conn_type}_{dir_type}_cluster'

                # === MAP TO SANITIZED COLUMN NAMES ===
                # merged DataFrame has sanitized column names (R-safe, no special characters)
                # Use colname_map to translate from original to sanitized names
                dependent_var_safe = colname_map.get(dependent_var, dependent_var)
                cluster_col_safe = colname_map.get(cluster_col, cluster_col)
                motion_columns_safe = [colname_map.get(c, c) for c in motion_columns]

                # === RUN QUANTILE REGRESSION IN R ===
                # Fits 3 models:
                #   Model 1: Depression vs Control
                #   Model 2: Cluster 0 vs Control, Cluster 1 vs Control
                #   Model 3: Cluster 0 vs Cluster 1
                # Returns dict with 'dependent_var' and 'p_values' (4 comparisons)
                results = run_quantile_regression(
                    merged,                   # DataFrame with sanitized column names
                    conn_type,                # Connectivity type (for motion covariate selection)
                    icd_covariates,           # ICD-10 comorbidity covariates
                    motion_columns_safe,      # Sanitized motion covariate column names
                    tau=0.5,                  # Median regression (50th percentile)
                    R=R,                      # Bootstrap replications for standard errors
                    dependent_var=dependent_var_safe,
                    cluster_col=cluster_col_safe,
                    group_col=group_col,
                )

                # === EXTRACT RAW P-VALUES ===
                # Extract p-values for all 4 comparisons in order
                ordered_raw_p = [results['p_values'].get(key, np.nan) for key in comparison_keys]
                
                # Append to collectors for FDR correction across all modules
                ordered_raw_p_all.extend(ordered_raw_p)
                test_labels_all.extend(test_labels)
                variable_names_all.extend([results.get('dependent_var', dependent_var_safe)] * len(comparison_keys))
                module_keys_all.extend([module_key] * len(comparison_keys))
                
                # Store sanitized dependent variable name for this module
                module_depvar_safe[module_key] = results.get('dependent_var', dependent_var_safe)

            # === APPLY FDR CORRECTION ACROSS ALL MODULES ===
            # Use Benjamini-Hochberg FDR correction to control false discovery rate
            # Corrects all p-values (modules × comparisons) within this (conn_type, dir_type) together
            _, corrected_p_values = apply_multiple_testing_correction(
                p_values=ordered_raw_p_all,      # All raw p-values (flattened list)
                test_methods=test_labels_all,    # Test labels (for logging)
                variable_names=variable_names_all,  # Variable names (for logging)
                log_path=fdr_log_path,           # Output log file
            )

            # === ORGANIZE CORRECTED P-VALUES BY MODULE ===
            # Reconstruct per-module dictionaries with corrected p-values
            corrected_idx = 0  # Index into corrected_p_values list
            
            for module_key in [str(m) for m in mods]:
                # Create dict mapping comparison_key -> corrected_p_value for this module
                comparison_corrected = {}
                
                for key in comparison_keys:
                    # Extract corrected p-value for this comparison
                    comparison_corrected[key] = (
                        corrected_p_values[corrected_idx]
                        if corrected_idx < len(corrected_p_values)
                        else np.nan
                    )
                    corrected_idx += 1

                # Store corrected p-values in significance payload
                # Key: (conn_type, dir_type, module_key) -> dict of comparison -> p_value
                significance_payload[(conn_type, dir_type, module_key)] = comparison_corrected
                
                # === MAP BACK TO ORIGINAL COLUMN NAMES FOR BRAINMAP ===
                # brainmap_significance uses original (unsanitized) module names
                # Extract original dependent variable name from sanitized name
                dependent_var_safe = module_depvar_safe.get(module_key, module_key)
                dependent_var_original = safe_to_orig.get(dependent_var_safe, dependent_var_safe)
                
                # Extract module identifier from original dependent variable name
                # E.g., '1_internal_functional' -> '1'
                module_from_dep = str(dependent_var_original).split('_', 1)[0]
                
                # Store cluster0_vs_cluster1 p-value for brain visualization filtering
                # Only modules with significant inter-cluster differences will be highlighted
                brainmap_significance[(conn_type, dir_type, module_from_dep)] = comparison_corrected.get(
                    'cluster0_vs_cluster1', np.nan
                )

        # === GENERATE ANNOTATED VIOLIN PLOTS ===
        # Create multi-panel violin plots showing module connectivity distributions
        # across Control, Depression, Cluster 0, and Cluster 1 groups
        # Plots are annotated with FDR-corrected significance stars
        plot_module_violin_across_clusters(
            final_df=final_df,          # Original DataFrame with unsanitized column names
            feature_df=final_df,        # Same DataFrame (contains both cluster labels and features)
            plots_dir=plots_dir,        # Output directory for plots
            minmax=True,                # Apply min-max scaling to [0, 1] for visualization
            modules=list(mods),         # Ordered list of module identifiers
            conn_types=(conn_type,),    # Single connectivity type for this iteration
            dir_types=dir_types,        # Both internal and external directions
            significance_map=significance_payload,  # FDR-corrected p-values for annotation
            output_name="{conn}_con/F32_{conn}_{dir}_module_violin.png",  # Filename template
        )

    # === RETURN BRAINMAP SIGNIFICANCE MAP ===
    # Return dict mapping (conn_type, dir_type, module) -> cluster0_vs_cluster1_p_value
    # Used downstream for filtering brain visualizations to show only significant modules
    return brainmap_significance