"""
module_clustering_confirmatory_utils
-----------------------------------

Utilities for module-level confirmatory analyses of connectivity features
against SFC-derived clustering. This module contains helpers to run
quantile-regression tests (via R/quantreg through rpy2), apply multiple-testing
correction, and produce publication-ready visualizations (violin panels and
brainmap figures) for module-wise connectivity profiles.

Primary responsibilities
 - run_quantile_regression: call R's quantreg to fit median quantile
     regression models for group contrasts (Depression vs Control, Cluster0 vs
     Control, Cluster1 vs Control, Cluster0 vs Cluster1) and return p-values.
 - run_module_quantile_regression_pipeline: orchestrate regression runs across
     modules, collect raw p-values, apply Benjamini-Hochberg FDR correction,
     and prepare a significance map for plotting and brainmap filtering.
 - plot_module_violin_across_clusters: create violin-panel figures for each
     module showing distributions across Control/Depression/Cluster groups and
     annotate significance based on corrected p-values.
 - plot_cluster_feature_brainmaps: render module-level median profiles and
     difference maps as NIfTI/PNG brain images using a provided atlas and
     community label mapping.

Expected inputs
 - DataFrames:
     - `merged` / `final_df`: subject-level DataFrame containing at minimum
         `eid`, `depression_status` (0/1), and cluster label columns (e.g.
         `sfc_external_cluster` or modality-specific `{conn}_{dir}_cluster`).
     - `feature_df`: DataFrame with per-subject module features named like
         `<module>_{dir}_{conn}` (e.g., `X9_0_external_functional`). If `eid` is
         present in both DataFrames, the code will merge by `eid`.
 - `colname_map` (optional): mapping original -> sanitized column names used
     to translate feature names to R-safe identifiers when necessary.
 - R environment: R must be installed and the `quantreg` (and optionally
     `boot`) R packages must be available. The code uses `rpy2` to invoke R.

Primary outputs
 - Per-module regression results: dictionaries mapping comparison keys
     (`depression_vs_control`, `cluster0_vs_control`, `cluster1_vs_control`,
     `cluster0_vs_cluster1`) to p-values.
 - `significance_map`: mapping of `(conn_type, dir_type, module)` -> corrected
     p-value dictionaries used by plotting helpers to annotate significance.
 - PNG figures and optional NIfTI masks saved to `plots_dir`/`fig_dir` paths.

Notes, assumptions, and runtime considerations
 - Quantile regression is performed in R and writes/reads a temporary CSV.
     The environment must provide working R and rpy2 binding. Bootstrapped
     standard errors can be computationally expensive (controlled by `R`).
 - The plotting helpers perform optional global min-max scaling for
     visualization consistency; ensure this is the desired behavior.
 - The module contains robust handling for several module-label naming
     conventions (e.g., `X1_0`, `1_0`, `1.0`) when matching significance results
     back to features.

Architecture
------------
The script is organized into functional sections:

- **Utility Functions**

- **R Package Management**

- **Quantile Regression**

- **Visualization**

- **High-level Pipeline**

"""

import os
from typing import Dict, List, Tuple, Optional, Literal
from statsmodels.stats import multitest
import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib.colors import Normalize
from matplotlib import cm
from nilearn import plotting
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from nilearn import image
import re
import textwrap
import seaborn as sns
import rpy2.robjects as ro
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
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

def _module_label_to_safe_prefix(module_label, conn_type: str, colname_map: Optional[Dict[str, str]] = None) -> str:
    raw = str(module_label)
    # If no colname_map provided, fall back to the simple transformation
    if not colname_map:
        if raw.startswith("X"):
            return raw
        return "X" + raw.replace(".", "_")

    lookup_key = f"{raw}_internal_{conn_type}"
    safe = colname_map.get(lookup_key)
    if isinstance(safe, str) and "_internal_" in safe:
        return safe.split("_internal_", 1)[0]
    if raw.startswith("X"):
        return raw
    return "X" + raw.replace(".", "_")

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
def plot_module_violin_across_clusters(
    final_df,
    feature_df,
    plots_dir,
    modules=None,
    cluster_col = 'sfc_external_cluster',
    conn_types=('functional', 'structural', 'sfc'),
    dir_types=('internal', 'external'),
    minmax=True,
    figsize_per_row=(12, 3),
    save_png=True,
    significance_map=None,
    comparison_groups=None,
    output_name=None,
    colname_map: Optional[Dict[str, str]] = None,
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
    cluster_col : str, default='sfc_external_cluster'
        Column name in final_df that contains cluster labels to use for grouping.
    conn_types : tuple, default=('functional', 'structural', 'sfc')
        Connectivity types to visualize.
    dir_types : tuple, default=('internal', 'external')
        Directions to visualize (subset of ('internal','external')).
    minmax : bool, default=True
        If True, min-max scale each feature to [0, 1] across subjects before plotting.
    figsize_per_row : tuple, default=(12, 3)
        (width, height) per row to scale the figure size.
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
    colname_map : dict, optional
        Mapping from raw column names to display names for plot labels. Keys should match the raw column names in feature_df, and values are the desired display names.

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
        # Try several candidate module keys to be robust to naming differences
        def _find_comp_dict(conn_l, dir_l, mod_l):
            # exact match first
            candidates = [mod_l]
            if isinstance(mod_l, str):
                # strip leading 'X' (e.g., 'X1_0' -> '1_0')
                stripped = re.sub(r'^X', '', mod_l)
                if stripped not in candidates:
                    candidates.append(stripped)
                # replace underscores with dots (e.g., '1_0' -> '1.0')
                dotted = stripped.replace('_', '.')
                if dotted not in candidates:
                    candidates.append(dotted)
                # try just the numeric prefix
                prefix = stripped.split('_', 1)[0]
                if prefix not in candidates:
                    candidates.append(prefix)
                # extract numeric substring if present
                m = re.search(r"(\d+(?:\.\d+)?)", mod_l)
                if m:
                    if m.group(1) not in candidates:
                        candidates.append(m.group(1))

            for cand in candidates:
                key = _sig_key(conn_l, dir_l, cand)
                if key in significance_map:
                    return significance_map[key]
            return None

        comp_dict = _find_comp_dict(conn_label, dir_label, module_label)
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
            # Use provided module list, but ensure labels are safe for column name construction
            modules_local = list(modules)
            modules_local = [_module_label_to_safe_prefix(m, conn, colname_map) for m in modules_local]

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
            merged = final_df[['eid', 'depression_status', f'{cluster_col}']].merge(
                feature_df[['eid'] + [c for c in feature_df.columns if c in int_cols + ext_cols]],
                on='eid', how='left'
            )
        else:
            # === ALIGN BY INDEX ===
            # If no 'eid' column, assume DataFrames are already aligned by row order
            merged = final_df[['depression_status', f'{cluster_col}']].copy().reset_index(drop=True)
            
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
                if c not in ('eid', 'depression_status', f'{cluster_col}')
            ]
            
            if feature_cols_z:
                # Fill missing values with column medians before scaling
                scaled = merged[feature_cols_z].fillna(merged[feature_cols_z].median()).astype(float)
                
                # Apply min-max scaling to [0, 1]
                scaler = MinMaxScaler(feature_range=(0.0, 1.0))
                scaled_vals = scaler.fit_transform(scaled.values.reshape(-1, 1)).reshape(scaled.values.shape)
                
                # Replace original values with scaled values
                merged[feature_cols_z] = scaled_vals

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
        for dir_idx, dir_type in enumerate(dir_types):

            # === DEFINE GROUP COLOR PALETTE ===
            # Consistent colors across all plots
            group_colors = {
                'Control': "#2ca02c",          
                'Depression': "#6a3d9a",      
                'Cluster 0': "#0026ff",
                'Cluster 1': "#fd7600",
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
                
                # Specify cluster column name for grouping (same for all modules)
                cluster_col_name = cluster_col 

                # === HANDLE MISSING DATA CASE ===
                if col_name is None:
                    # Display "Missing" text if column doesn't exist
                    ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
                    display_m = (m.lstrip('X').replace('_', '.') if isinstance(m, str) and m.startswith('X') else str(m))
                    ax.set_title(f"Module {display_m}")
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
                df_dir = _build_group_df(col_name, cluster_col_name)
                
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
                # Display module labels in a human-friendly way: strip leading 'X' and
                # convert underscores to dots (e.g., 'X1_0' -> '1.0') when present.
                display_m = (m.lstrip('X').replace('_', '.') if isinstance(m, str) and m.startswith('X') else str(m))
                ax.set_title(f"Module {display_m}")
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
            out_dir = os.path.join(plots_dir, 'sfc_con')
            if output_name:
                # Use provided filename template with format substitution
                rel_path = output_name.format(conn=conn, dir=dir_slug, module=modules_slug)
                rel_path = os.path.basename(rel_path)
                # If template doesn't include {dir} placeholder, append dir_slug manually
                if '{dir}' not in output_name:
                    base, ext = os.path.splitext(rel_path)
                    
                    if ext:
                        # Has extension: insert dir_slug before extension
                        rel_path = f"{base}_{dir_slug}{ext}"
                    else:
                        # No extension: append dir_slug and .png if save_png is True
                        rel_path = f"{rel_path}_{dir_slug}.png" if save_png else f"{rel_path}_{dir_slug}"
                base, ext = os.path.splitext(rel_path)
                if not base.endswith('sfc_external'):
                    base = f"{base}_sfc_external"
                rel_path = f"{base}{ext}" if ext else base
                out_path = os.path.join(out_dir, rel_path)
            else:
                # Use default filename format
                out_path = os.path.join(out_dir, f"F32_{conn}_{dir_slug}_module_violin_by_cluster_sfc_external.png")

            # === SAVE FIGURE ===
            # Ensure output directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Save as PNG if requested
            if save_png:
                fig.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
            
            # Close figure to free memory
            plt.close(fig)
            
            print(f"Saved {out_path}")

def plot_cluster_feature_brainmaps(
    final_df,
    feature_df,
    cluster_col,
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
    cluster_col : str
        Column name in final_df that contains cluster labels to use for grouping (e.g., 'sfc_external_cluster').
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
                modules_prefixes = list(modules_local)
            else:
                modules_local = list(modules)  # Use provided module list
                modules_prefixes = [_module_label_to_safe_prefix(m, conn_type) for m in modules_local]

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
            ordered_features = [f"{m}_{dir_type}_{conn_type}" for m in modules_prefixes]
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
            cluster_col = "sfc_external_cluster" # Cluster labels are based on sfc_external features
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
            module_labels_for_map = []
            for m, p in zip(modules_local, modules_prefixes):
                col_name = f"{p}_{dir_type}_{conn_type}"
                if col_name in ordered_features:
                    module_labels_for_map.append(str(m))
            module_to_pos = {label: i for i, label in enumerate(module_labels_for_map)}

            # warn if some atlas community labels do not map to any module position
            label_keys = {str(l) for l in labels}
            missing_labels = label_keys - set(module_to_pos.keys())
            if missing_labels:
                # do not fail, but log a warning with counts
                print(f"[WARN] {len(missing_labels)} atlas community labels have no matching module in selected modules for {conn_type} {dir_type} (examples: {list(missing_labels)[:5]})")

            out_dir_conn = os.path.join(fig_dir, f'sfc_con')
            os.makedirs(out_dir_conn, exist_ok=True)

            # Cluster-specific maps (min-max scaling already applied to features)
            vmin_single, vmax_single = 0.0, 1.0
            out_path_pair = os.path.join(
                out_dir_conn,
                f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster0_vs_cluster1_brainmaps_sfc_external.png",
            )
            nifti_out_c0 = None
            nifti_out_c1 = None
            if save_niftis:
                nifti_out_c0 = os.path.join(
                    nifti_dir,
                    f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster0_mask_sfc_external{nifti_suffix}",
                )
                nifti_out_c1 = os.path.join(
                    nifti_dir,
                    f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster1_mask_sfc_external{nifti_suffix}",
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

            out_path_diff = os.path.join(out_dir_conn, f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster0_minus_cluster1_brainmap_sfc_external.png")
            nifti_out_diff = None
            if save_niftis:
                nifti_out_diff = os.path.join(
                    nifti_dir,
                    f"{output_basename_prefix}_{conn_type}_{dir_type}_cluster0_minus_cluster1_mask_sfc_external{nifti_suffix}",
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

# ==============================================================================
# HIGH-LEVEL PIPELINE ORCHESTRATION FUNCTIONS
# ==============================================================================
def run_module_quantile_regression_pipeline(
    merged: pd.DataFrame,
    final_df: pd.DataFrame,
    mods: List[str],
    cluster_col: str,
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
    cluster_col : str
        Column name in merged that contains cluster labels (e.g., 'sfc_external_cluster').
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
                f'modular_{conn_type}_{dir_type}_connectivity_FDR_sfc_external.txt',
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
                
                # Cluster column: cluster assignments for this modality
                cluster_col = cluster_col  # Already defined as 'sfc_external_cluster' since clusters are based on sfc_external features

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
            cluster_col=cluster_col,    # Column name for cluster labels (e.g., 'sfc_external_cluster')
            conn_types=(conn_type,),    # Single connectivity type for this iteration
            dir_types=dir_types,        # Both internal and external directions
            significance_map=significance_payload,  # FDR-corrected p-values for annotation
            output_name="{conn}_con/F32_{conn}_{dir}_module_violin.png",  # Filename template
            colname_map=colname_map,
        )

    # === RETURN BRAINMAP SIGNIFICANCE MAP ===
    # Return dict mapping (conn_type, dir_type, module) -> cluster0_vs_cluster1_p_value
    # Used downstream for filtering brain visualizations to show only significant modules
    return brainmap_significance