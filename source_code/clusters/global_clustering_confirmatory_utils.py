"""Utility functions for global-level structure-function coupling subtypes characterization and comparison analysis.

This module contains functions used to perform quantile regression, multiple testing correction, and visualization for 
the global-level structure-function coupling subtypes analysis. It also includes helper functions for 
managing R package dependencies and formatting outputs.

Architecture
------------
The script is organized into functional sections:

- **Utility Functions**

- **R Package Management**

- **Quantile Regression**

- **Visualization**

Requirements
------------
- Python packages: numpy, pandas, matplotlib, seaborn, scipy, sklearn,
    statsmodels, rpy2
- R (system installation) with package: `quantreg` (the
    module will attempt to install this via R if missing)

Side effects
------------
- Several functions write output files (CSV, TXT, PNG) to provided output
    directories. `run_quantile_regression` writes a temporary CSV to
    `/tmp/combined_data.csv` and invokes R; it also creates R objects in
    the R global environment during execution.

Notes
------------
- This module relies heavily on NumPy/Pandas for data handling and
  rpy2 to call R for certain statistical tests (quantile regression).

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional, Literal
from statsmodels.stats import multitest

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _display_conn_type(conn_type: str) -> str:
    return "Structure-Function Coupling" if str(conn_type).lower() == "sfc" else str(conn_type).capitalize()

def _append_to_text_log(log_path: Optional[str], block: str) -> None:
    """Append a block of text to a log file, creating parent dirs if needed.

    This helper is intentionally fault tolerant: failures to create
    directories or write to the file are swallowed to avoid interrupting
    long-running analyses. The function writes UTF-8 text and ensures the
    appended block ends with a newline.

    If ``log_path`` is None, this is a no-op.
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
    if not (len(p_values) == len(variable_names) == len(test_methods)):
        raise ValueError("Lengths of p_values, variable_names, and test_methods must match")
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

  MOTION_METRICS = {
  'functional': {'motion_fmri': fMRI_MOTION_METRIC},
  'structural': {'motion_dmri': dMRI_MOTION_METRIC},
  'sfc': {
    'motion_fmri': fMRI_MOTION_METRIC,
    'motion_dmri': dMRI_MOTION_METRIC,
  },
}
  mapping = MOTION_METRICS.get(conn_type)
  if mapping is None:
    return {'motion': 'p24453_i2'}
  return mapping

# ==============================================================================
# R PACKAGE MANAGEMENT
# ==============================================================================
def install_r_package_if_missing(package_name):
    """Install R package only if not already installed.
    
    Parameters
    ----------
    package_name : str
        Name of the R package to install (e.g., 'quantreg', 'multcomp')
    
    Returns
    -------
    None
    """
    ro.r(f'''
        if (!require("{package_name}", quietly = TRUE)) {{
            install.packages("{package_name}", repos = "https://cloud.r-project.org")
        }}
    ''')


def setup_r_environment():
    """Initialize R environment with required packages.
    
    Returns
    -------
    dict
        Dictionary containing imported R modules: 'base', 'utils', 'stats', 
        'quantreg'
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
    motion_covariates = list(motion_covariates or [])
    if not motion_covariates:
        if str(conn_type).lower().startswith('functional'):
            motion_covariates = ['p24441_i2']
        else:
            motion_covariates = ['p24453_i2']

    # Controls do not have ICD-10 codes; ensure missing values are treated as 0.
    # This also ensures R sees consistent 0/1 columns instead of NA.
    for cov in icd_covariates or []:
        if cov in combined_df.columns:
            combined_df[cov] = pd.to_numeric(combined_df[cov], errors='coerce').fillna(0)

    # Save data for R
    # Write a temporary CSV that the R block will read via rpy2. This
    # path is reused across calls ('/tmp/combined_data.csv') so it will be
    # overwritten — callers should avoid concurrent R launches that share
    # the same temp path.
    combined_df.to_csv('/tmp/combined_data.csv', index=False)
    
    # Run R quantile regression inside a localconverter context so rpy2
    # conversion rules (pandas<->R) are available in this execution context.
    # Within this `with` block we prepare a few global variables in R's
    # environment (using `ro.globalenv`) so the embedded R script can
    # reference them directly instead of building a long string via Python.
    with localconverter(default_converter + pandas2ri.converter):
        # Pass connectivity type to R
        ro.globalenv['conn_type'] = ro.StrVector([str(conn_type)])
        # Pass dependent variable name to R
        ro.globalenv['dependent_var'] = ro.StrVector([str(dependent_var)])
        # Pass group and cluster column names to R
        ro.globalenv['group_col'] = ro.StrVector([str(group_col)])
        ro.globalenv['cluster_col'] = ro.StrVector([str(cluster_col)])
        # Pass ICD covariates to R
        ro.globalenv['icd_covariates'] = ro.StrVector([str(cov) for cov in icd_covariates])
        # Pass motion covariates, tau, and R
        ro.globalenv['motion_cols'] = ro.StrVector([str(col) for col in motion_covariates])
        ro.globalenv['tau'] = tau
        ro.globalenv['R'] = R

        # Run R quantile regression
        ro.r(r'''
        library(quantreg)
        library(multcomp)
        
        set.seed(123)
             
        conn_type <- as.character(conn_type[[1]])
        data <- read.csv("/tmp/combined_data.csv")
        if (!(group_col %in% colnames(data))) {
            stop(paste0("group_col not found in data: ", group_col))
        }
        if (!(cluster_col %in% colnames(data))) {
            stop(paste0("cluster_col not found in data: ", cluster_col))
        }
             
        data$p31 <- factor(data$p31)
             
        # Ensure group column has string values "Control" and "Depression" if original are numeric (0,1)
        if (1 %in% unique(data[[group_col]]) & 0 %in% unique(data[[group_col]])) {
            data[[group_col]] <- ifelse(data[[group_col]] == 1, "Depression", "Control")
        }
        data[[group_col]] <- factor(data[[group_col]])

        # Ensure cluster column has string values "Control", "Cluster 0", "Cluster 1" if original are numeric
        # and mixed with string (0,1,"Control")
        # Only map numeric 0/1 to cluster labels, preserve existing text labels like "Control"
        vals <- data[[cluster_col]]
        # detect numeric entries equal to 0 or 1 (handles factors/numeric/character-coded numbers)
        is_num01 <- suppressWarnings(!is.na(as.numeric(as.character(vals))) & as.numeric(as.character(vals)) %in% c(0,1))
        vals_mapped <- as.character(vals)
        vals_mapped[is_num01] <- ifelse(as.numeric(as.character(vals[is_num01])) == 0, "Cluster 0", "Cluster 1")
        data[[cluster_col]] <- factor(vals_mapped)

        # Ensure ICD covariates are factors with 0/1 coding. (0 = no diagnosis, 1 = diagnosis)
        # Controls have 0s; depressed subjects may have 1s.
        for (cov in icd_covariates) {
            if (cov %in% colnames(data)) {
                data[[cov]][is.na(data[[cov]])] <- 0
                data[[cov]] <- factor(data[[cov]])
            }
        }
             
        age_median <- median(data$p21003_i2, na.rm = TRUE)
        data$age_centered <- data$p21003_i2 - age_median
        
        motion_cols <- as.character(motion_cols)
        motion_cols <- motion_cols[motion_cols %in% colnames(data)]
        if (length(motion_cols) == 0) {
            warning("No motion covariates found in dataset for quantile regression; proceeding without motion covariates")
        }
        motion_terms <- c()
        for (col in motion_cols) {
            centered_name <- paste0(col, "_centered")
            med_val <- median(data[[col]], na.rm = TRUE)
            data[[centered_name]] <- data[[col]] - med_val
            motion_terms <- c(motion_terms, centered_name)
        }

        # Only include ICD covariates that exist in the dataset.
        icd_terms <- as.character(icd_covariates)
        icd_terms <- icd_terms[icd_terms %in% colnames(data)]
             
        # Model 1: Depression vs Control
        print("========== MODEL 1: Depression vs Control ==========")
        print(paste0("dependent_var: ", dependent_var))
        data_dep_ctrl <- data[data[[group_col]] %in% c("Depression", "Control"), ]
        data_dep_ctrl[[group_col]] <- droplevels(factor(data_dep_ctrl[[group_col]]))
        data_dep_ctrl[[group_col]] <- relevel(data_dep_ctrl[[group_col]], ref = "Control")

        rhs_dep_ctrl <- c(group_col, "age_centered", "p31", motion_terms, icd_terms)
             
        # Safely wrap variable names in backticks so non-syntactic names (e.g. starting with digits)
        # do not cause parsing errors when constructing formulas.
        rhs_dep_ctrl_safe <- paste0("`", rhs_dep_ctrl, "`")
        fml_dep_ctrl <- as.formula(paste0("`", dependent_var, "` ~ ", paste(rhs_dep_ctrl_safe, collapse = " + ")))

        X_dep_ctrl <- model.matrix(fml_dep_ctrl, data = data_dep_ctrl)
        y_dep_ctrl <- data_dep_ctrl[[dependent_var]]

        model_dep_ctrl <- rq(fml_dep_ctrl, data = data_dep_ctrl, tau = tau)
        summary_dep_ctrl <- summary.rq(model_dep_ctrl, se = "boot", R = R)
        print(summary_dep_ctrl)
        
        boot_out_dep_ctrl <- boot.rq(x = X_dep_ctrl, y = y_dep_ctrl, tau = tau, R = R)
        cis_dep_ctrl <- apply(boot_out_dep_ctrl$B, 2, quantile, probs = c(0.025, 0.975))
        print("95% CI for coefficients:")
        print(cis_dep_ctrl)
        
        coef_table <- summary_dep_ctrl$coefficients
        coef_rownames <- rownames(coef_table)
        coef_rownames_clean <- gsub("`", "", coef_rownames)
        get_p <- function(term_name) {
            idx <- which(coef_rownames_clean == term_name)
            if (length(idx) == 0) {
                warning(paste0("Coefficient not found for term: ", term_name))
                return(NA_real_)
            }
            pval <- coef_table[idx[1], "Pr(>|t|)"]
            return(pval)
        }

        p_depression_vs_control <- get_p(paste0(group_col, "Depression"))
        p_depression_vs_control <- ifelse(is.na(p_depression_vs_control), NA_real_, ifelse(p_depression_vs_control < 2.2e-16, 2.2e-16, p_depression_vs_control))
        
        # Model 2: Clusters vs Control
        print("========== MODEL 2: Clusters vs Control ==========")
        print(paste0("dependent_var: ", dependent_var))
        data_clusters_ctrl <- data[data[[cluster_col]] %in% c("Cluster 0", "Cluster 1", "Control"), ]
        data_clusters_ctrl[[cluster_col]] <- droplevels(factor(data_clusters_ctrl[[cluster_col]]))
        data_clusters_ctrl[[cluster_col]] <- relevel(data_clusters_ctrl[[cluster_col]], ref = "Control")

        rhs_clusters_ctrl <- c(cluster_col, "age_centered", "p31", motion_terms, icd_terms)
        rhs_clusters_ctrl_safe <- paste0("`", rhs_clusters_ctrl, "`")
        fml_clusters_ctrl <- as.formula(paste0("`", dependent_var, "` ~ ", paste(rhs_clusters_ctrl_safe, collapse = " + ")))

        X_clusters_ctrl <- model.matrix(fml_clusters_ctrl, data = data_clusters_ctrl)
        y_clusters_ctrl <- data_clusters_ctrl[[dependent_var]]

        model_all <- rq(fml_clusters_ctrl, data = data_clusters_ctrl, tau = tau)
        summary_clusters_ctrl <- summary.rq(model_all, se = "boot", R = R)
        print(summary_clusters_ctrl)
        
        
        boot_out_clusters <- boot.rq(x = X_clusters_ctrl, y = y_clusters_ctrl, tau = tau, R = R)
        cis_coef <- apply(boot_out_clusters$B, 2, quantile, probs = c(0.025, 0.975))
        print("95% CI for coefficients:")
        print(cis_coef)
        
        coef_table <- summary_clusters_ctrl$coefficients
        coef_rownames <- rownames(coef_table)
        coef_rownames_clean <- gsub("`", "", coef_rownames)
        get_p <- function(term_name) {
            idx <- which(coef_rownames_clean == term_name)
            if (length(idx) == 0) {
                warning(paste0("Coefficient not found for term: ", term_name))
                return(NA_real_)
            }
            pval <- coef_table[idx[1], "Pr(>|t|)"]
            return(pval)
        }

        p_cluster0_vs_control <- get_p(paste0(cluster_col, "Cluster 0"))
        p_cluster0_vs_control <- ifelse(is.na(p_cluster0_vs_control), NA_real_, ifelse(p_cluster0_vs_control < 2.2e-16, 2.2e-16, p_cluster0_vs_control))
        
        p_cluster1_vs_control <- get_p(paste0(cluster_col, "Cluster 1"))
        p_cluster1_vs_control <- ifelse(is.na(p_cluster1_vs_control), NA_real_, ifelse(p_cluster1_vs_control < 2.2e-16, 2.2e-16, p_cluster1_vs_control))

        # Model 3: Cluster 0 versus Cluster 1
        print("========== MODEL 3: Cluster 0 vs Cluster 1 ==========")
        print(paste0("dependent_var: ", dependent_var))
        data_cluster0_1 <- data[data[[cluster_col]] %in% c("Cluster 0", "Cluster 1"), ]
        data_cluster0_1[[cluster_col]] <- droplevels(factor(data_cluster0_1[[cluster_col]]))
        data_cluster0_1[[cluster_col]] <- relevel(data_cluster0_1[[cluster_col]], ref = "Cluster 1")
        rhs_cluster0_1 <- c(cluster_col, "age_centered", "p31", motion_terms, icd_terms)
        rhs_cluster0_1_safe <- paste0("`", rhs_cluster0_1, "`")
        fml_cluster0_1 <- as.formula(paste0("`", dependent_var, "` ~ ", paste(rhs_cluster0_1_safe, collapse = " + ")))
        
        X_cluster0_1 <- model.matrix(fml_cluster0_1, data = data_cluster0_1)
        y_cluster0_1 <- data_cluster0_1[[dependent_var]]
             
        model_cluster0_1 <- rq(fml_cluster0_1, data = data_cluster0_1, tau = tau)
        summary_cluster0_1 <- summary.rq(model_cluster0_1, se = "boot", R = R)
        print(summary_cluster0_1)
        
        boot_out_cluster0_1 <- boot.rq(x = X_cluster0_1, y = y_cluster0_1, tau = tau, R = R)
        cis_cluster0_1 <- apply(boot_out_cluster0_1$B, 2, quantile, probs = c(0.025, 0.975))
        print("95% CI for coefficients:")
        print(cis_cluster0_1)
             
        coef_table <- summary_cluster0_1$coefficients
        coef_rownames <- rownames(coef_table)
        coef_rownames_clean <- gsub("`", "", coef_rownames)
        get_p <- function(term_name) {
            idx <- which(coef_rownames_clean == term_name)
            if (length(idx) == 0) {
                warning(paste0("Coefficient not found for term: ", term_name))
                return(NA_real_)
            }
            pval <- coef_table[idx[1], "Pr(>|t|)"]
            return(pval)
        }

        p_cluster0_vs_cluster1 <- get_p(paste0(cluster_col, "Cluster 0"))
        p_cluster0_vs_cluster1 <- ifelse(is.na(p_cluster0_vs_cluster1), NA_real_, ifelse(p_cluster0_vs_cluster1 < 2.2e-16, 2.2e-16, p_cluster0_vs_cluster1))
             
        # Collect p-values
        p_values <- c(
            depression_vs_control = p_depression_vs_control,
            cluster0_vs_control = p_cluster0_vs_control,
            cluster1_vs_control = p_cluster1_vs_control,
            cluster0_vs_cluster1 = p_cluster0_vs_cluster1
        )
        
        assign("quantreg_p_values", p_values, envir = .GlobalEnv)
             
    ''')

    # Extract p-values back into Python as a named vector. The R block
    # assigns `quantreg_p_values` into R's global environment; here we
    # pull it back and make sure names align with values.
    with localconverter(default_converter):
        r_p_values = ro.r("quantreg_p_values")
        r_names = list(ro.r("names(quantreg_p_values)"))
        r_dep = ro.r('if (!is.null(attr(quantreg_p_values, "dependent_var"))) as.character(attr(quantreg_p_values, "dependent_var")) else NA_character_')
    
    # Extract dependent variable name
    try:
        dep_list = list(r_dep)
        dependent_var_name = str(dep_list[0]) if dep_list and dep_list[0] is not ro.rinterface.NA_Character else str(dependent_var)
    except Exception:
        dependent_var_name = str(dependent_var)

    # Build a plain Python dict mapping the R-returned names to floats.
    p_values: Dict[str, float] = {}
    if len(r_names) != len(list(r_p_values)):
        # Defensive check: ensure R returned the same number of names
        # and values; if not, fail early to avoid silent misalignment.
        raise RuntimeError("Mismatch between p-value names and values returned from R")

    for i, name in enumerate(r_names):
        val = list(r_p_values)[i]
        p_values[str(name)] = float(val) if val is not ro.rinterface.NA_Real else float('nan')

    return {
        'dependent_var': dependent_var_name,
        'p_values': p_values
    }

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def create_violin_plot_with_significance(subject_scalar_control, subject_scalar_depression,
                                         clusters, corrected_p_values, conn_type, out_file):
    """Create violin plot with significance brackets.
    
    Parameters
    ----------
    subject_scalar_control : np.ndarray
        Control connectivity values
    subject_scalar_depression : np.ndarray
        Depression connectivity values
    clusters : np.ndarray
        Cluster labels
    corrected_p_values : tuple
        FDR-corrected p-values
    conn_type : str
        Connectivity type ('functional', 'structural', or 'sfc')
    out_file : str
        Output file path
    
    Returns
    -------
    None
    """
    ctrl_vals = np.asarray(subject_scalar_control, dtype=float)
    dep_vals = np.asarray(subject_scalar_depression, dtype=float)
    all_vals = np.concatenate([ctrl_vals, dep_vals])
    mask = np.isfinite(all_vals)
    if np.any(mask):
        # Replace missing subject-level values with the global median to
        # avoid NaNs in the plotting pipeline (which would produce empty
        # or truncated violins). Then fit a single MinMaxScaler on the
        # combined flattened data so that control and depression values
        # are scaled to the same [0,1] range, preserving relative
        # distributional shape while making panels visually comparable.
        global_median = float(np.median(all_vals[mask]))
        ctrl_fill = np.where(np.isfinite(ctrl_vals), ctrl_vals, global_median).astype(float)
        dep_fill = np.where(np.isfinite(dep_vals), dep_vals, global_median).astype(float)
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        flattened = np.concatenate([ctrl_fill, dep_fill]).reshape(-1, 1)
        scaler.fit(flattened)
        ctrl_vals = scaler.transform(ctrl_fill.reshape(-1, 1)).ravel()
        dep_vals = scaler.transform(dep_fill.reshape(-1, 1)).ravel()

    cluster0_data = dep_vals[clusters == 0]
    cluster1_data = dep_vals[clusters == 1]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot violins at positions 0, 1, 2, 3
    ctrl_df = pd.DataFrame({'Connectivity': ctrl_vals, 'Position': 0})
    dep_df = pd.DataFrame({'Connectivity': dep_vals, 'Position': 1})
    c0_df = pd.DataFrame({'Connectivity': cluster0_data, 'Position': 2})
    c1_df = pd.DataFrame({'Connectivity': cluster1_data, 'Position': 3})
    
    sns.violinplot(data=ctrl_df, x='Position', y='Connectivity', ax=ax, color='#2ca02c')
    sns.violinplot(data=dep_df, x='Position', y='Connectivity', ax=ax, color='#6a3d9a')
    sns.violinplot(data=c0_df, x='Position', y='Connectivity', ax=ax, color="#17becf")
    sns.violinplot(data=c1_df, x='Position', y='Connectivity', ax=ax, color="#7f7f7f")
    
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Control', 'Depression', 'Cluster 0', 'Cluster 1'])
    ax.set_title(f'Global {_display_conn_type(conn_type)} Connectivity by Group')
    ax.set_xlabel('Group')
    ax.set_ylabel(f"Global {_display_conn_type(conn_type)} Connectivity")
    ax.grid(alpha=0.3, axis='y')
    
    # Significance stars as horizontal brackets between groups
    def _p_label(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return r'$\mathit{ns}$'

    # Base y-level for brackets (slightly above current max)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min if y_max > y_min else 1.0
    bar_height = 0.02 * y_range
    y_current = y_max + 0.02 * y_range

    def _add_sig_bar(x1, x2, p):
        nonlocal y_current
        y = y_current
        # horizontal line between x1 and x2
        ax.plot([x1, x2], [y, y], lw=1.5, c='black')
        # small vertical ticks at the ends
        ax.plot([x1, x1], [y, y - bar_height], lw=1.0, c='black')
        ax.plot([x2, x2], [y, y - bar_height], lw=1.0, c='black')
        # significance label (stars or ns) centered above the bar
        label = _p_label(float(p))
        ax.text((x1 + x2) / 2.0, y + 0.3 * bar_height, label,
                ha='center', va='bottom', fontsize=11)
        y_current += 0.10 * y_range

    # x positions for the four violins
    x_ctrl, x_dep, x_c0, x_c1 = 0, 1, 2, 3

    # corrected_p_values may be either the p-value array itself or a tuple
    # (reject, pvals). Normalize to a 1D array of p-values.
    if isinstance(corrected_p_values, (tuple, list)) and len(corrected_p_values) == 2:
        p_fdr = corrected_p_values[1]
    else:
        p_fdr = corrected_p_values
    p_fdr = np.atleast_1d(p_fdr).astype(float)

    # Depression vs Control
    if p_fdr.size > 0:
        _add_sig_bar(x_ctrl, x_dep, p_fdr[0])
    # Cluster 0 vs Control
    if p_fdr.size > 1:
        _add_sig_bar(x_ctrl, x_c0, p_fdr[1])
    # Cluster 1 vs Control
    if p_fdr.size > 2:
        _add_sig_bar(x_ctrl, x_c1, p_fdr[2])
    # Cluster 0 vs Cluster 1
    if p_fdr.size > 3:
        _add_sig_bar(x_c0, x_c1, p_fdr[3])

    # extend y-limits to make sure brackets are visible
    ax.set_ylim(y_min, y_current + 0.05 * y_range)
    ax.set_xlim(-0.6, 3.6)
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print("  Saved violin plot with significance brackets")