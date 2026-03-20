"""Utility functions for global-level depression subtyping analysis.

This module contains functions used to perform clustering, validation,
visualization, and statistical testing on global connectivity-derived features
for depression and control cohorts. Functions are organized broadly by
purpose: data loading, R integration, statistics, plotting, clustering,
and cross-modality agreement analyses.

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

Requirements
------------
- R (system installation) with package: `quantreg` (the
    module will attempt to install it via R if missing)

Side effects
------------
- Several functions write output files (CSV, TXT, SVG) to provided output
    directories. `run_quantile_regression` writes a temporary CSV to
    `/tmp/combined_data.csv` and invokes R; it also creates R objects in
    the R global environment during execution.

Notes
------------
- This module relies heavily on NumPy/Pandas for data handling and
  rpy2 to call R for certain statistical tests (quantile regression).
- Many helper functions expect a specific on-disk layout for per-subject
  connectivity matrices: a per-subject folder named by subject id
  containing an `i2` subfolder with connectivity files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import textwrap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, normalized_mutual_info_score
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats as sp_stats
import warnings
from statsmodels.stats import multitest
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional, Literal

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _append_to_text_log(log_path: Optional[str], block: str) -> None:
    """Append a block of text to a log file, creating parent dirs if needed.

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

def _display_conn_type(conn_type: str) -> str:
    """Return a human-readable connectivity type string.

    Parameters
    ----------
    conn_type : str
        Input connectivity type token (e.g., 'functional', 'structural', 'sfc').

    Returns
    -------
    str
        Display-friendly label. For example, 'sfc' -> 'Structure-Function Coupling',
        otherwise capitalizes the provided string.
    """
    return "Structure-Function Coupling" if str(conn_type).lower() == "sfc" else str(conn_type).capitalize()

def _cluster_colors_for_conn_type(conn_type: str) -> Dict[str, str]:
    """Return a color mapping for cluster labels for a given connectivity type.

    Parameters
    ----------
    conn_type : str
        Connectivity type identifier. Common values: 'functional', 'structural', 'sfc'.

    Returns
    -------
    Dict[str, str]
        Mapping from cluster label string ('0' and '1') to hex color codes.
    """
    key = str(conn_type).lower().strip()
    if key in ("functional",):
        return {"0": "#ccebc5", "1": "#fb9a99"}
    if key in ("structural",):
        return {"0": "#d62728", "1": "#8c564b"}
    if key in ("sfc", "structure-function coupling", "structure function coupling"):
        return {"0": "#17becf", "1": "#7f7f7f"}
    return {"0": "#1f77b4", "1": "#ff7f0e"}

def ensure_2d_array(feats):
    """Ensure the input is a 2D NumPy array.

    Parameters
    ----------
    feats : array-like
        Input features. If 1D, this function will reshape to (-1, 1). If
        already 2D it is returned as an ndarray.

    Returns
    -------
    np.ndarray
        2D NumPy array.
    """
    arr = np.asarray(feats)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def _stack_matrices(mats):
    """Stack a list of 2D matrices into a 3D NumPy array when appropriate.

    Parameters
    ----------
    mats : list or np.ndarray or np.memmap
        Collection of per-subject connectivity matrices. If a `np.memmap` is
        provided it is returned unchanged. If `mats` is a list of 2D NumPy
        arrays the function returns a 3D array with shape (n_subjects, n_nodes, n_nodes).

    Returns
    -------
    np.ndarray or np.memmap
        Stacked array or the original memmap.
    """
    if isinstance(mats, np.memmap):
        return mats
    if isinstance(mats, list):
        if len(mats) == 0:
            return np.asarray(mats)
        if isinstance(mats[0], np.ndarray) and mats[0].ndim == 2:
            return np.stack(mats, axis=0)
    return np.asarray(mats)

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
def load_single_connectivity_matrix(subject_id, cohort_dir, conn_type='functional'):
    """Load a single subject connectivity matrix for a given subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier (string); function assumes a subdirectory named
        `<subject_id>/i2` under `cohort_dir`.
    cohort_dir : str
        Base directory containing per-subject folders.
    conn_type : {'functional', 'structural'}, default 'functional'
        Which connectivity modality to load. For 'functional' a NumPy `.npy`
        file named `<eid>_connectivity.npy` is expected. For 'structural' a
        compressed CSV named `connectome_streamline_count_10M.csv.gz` is
        expected.

    Returns
    -------
    np.ndarray or None
        2D connectivity matrix as a float array if the file exists; returns
        ``None`` and emits a warning (printed) when the expected file is
        missing.
    """
    # Per-subject data layout: <cohort_dir>/<subject_id>/i2/<files>
    # The `i2` subfolder is used across the pipeline and expected by
    # downstream consumers; this helper encapsulates that file convention.
    subject_dir = os.path.join(cohort_dir, subject_id, 'i2')
    if conn_type == 'functional':
        file_path = os.path.join(subject_dir, f'{subject_id}_connectivity.npy')
        if os.path.isfile(file_path):
            data = np.load(file_path)
            np.fill_diagonal(data, 0)
            return data.astype(float)
        print(f"Warning: {file_path} not found for subject {subject_id}")
        return None

    if conn_type == 'structural':
        file_path = os.path.join(subject_dir, 'connectome_streamline_count_10M.csv.gz')
        if os.path.isfile(file_path):
            # For structural connectomes stored as CSV (possibly gzipped)
            # read via pandas for robust parsing, convert to NumPy and
            # zero the diagonal similarly.
            data = pd.read_csv(file_path, compression='infer', header=None).to_numpy()
            np.fill_diagonal(data, 0)
            return data.astype(float)
        print(f"Warning: {file_path} not found for subject {subject_id}")
        return None

    raise ValueError("conn_type must be 'functional' or 'structural' for matrix loading")

def compute_node_strength_from_matrix(mat, conn_type='functional'):
    """Compute per-node normalized (weighted) degree / strength from a connectivity matrix.

    Parameters
    ----------
    mat : array-like, shape (n_nodes, n_nodes)
        Square connectivity matrix.
    conn_type : {'functional','structural'}, default 'functional'
        If 'functional' the absolute value of edges is used when computing
        node strength; for 'structural' signed values are used as-is.

    Returns
    -------
    np.ndarray
        1D array of node strengths normalized by (n_nodes - 1).
    """
    data = np.asarray(mat, dtype=float).copy()
    if data.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {data.shape}")
    np.fill_diagonal(data, 0)
    if conn_type == 'functional':
        np.abs(data, out=data)
    weighted_degrees = np.sum(data, axis=1)
    n_nodes = data.shape[0]
    return weighted_degrees / (n_nodes - 1)

def compute_sfc_features_from_matrices(fc_mats, sc_mats):
    """Compute per-node Structure-Function Coupling (SFC) vectors for subjects.

    For each subject this function computes the Pearson correlation between
    the SC and FC connectivity profiles for each node (excluding the node's
    self-connection). If fewer than two valid paired entries exist for a
    node the value is set to NaN and later converted to 0.0 in the output.

    Parameters
    ----------
    fc_mats : sequence of 2D arrays
        Functional connectivity matrices per subject.
    sc_mats : sequence of 2D arrays
        Structural connectivity matrices per subject.

    Returns
    -------
    list of np.ndarray
        Per-subject 1D SFC vectors (length = n_nodes). NaNs are converted to
        numerical zeros prior to returning.
    """
    # Stack per-subject matrices into 3D arrays of shape
    # (n_subjects, n_nodes, n_nodes). The helper returns memmaps
    # unchanged which allows memory-efficient workflows when needed.
    fc_arr = _stack_matrices(fc_mats)
    sc_arr = _stack_matrices(sc_mats)
    if fc_arr.shape != sc_arr.shape:
        raise ValueError(f"FC and SC shape mismatch: {fc_arr.shape} vs {sc_arr.shape}")

    sfc_features = []
    for subj_idx in range(fc_arr.shape[0]):
        FC = fc_arr[subj_idx].copy()
        SC = sc_arr[subj_idx].copy()
        np.fill_diagonal(FC, 0)
        np.fill_diagonal(SC, 0)
        n_nodes = FC.shape[0]
        sfc_vec = np.full(n_nodes, np.nan, dtype=float)
        nan_counter = 0
        # Iterate nodes: compute pearsonr across the node's connectivity
        # vectors. We require at least two finite paired observations to
        # compute a correlation; otherwise record NaN (converted to 0
        # before returning to keep downstream matrices numeric).
        for i in range(n_nodes):
            fc_vec = FC[i, :]
            sc_vec = SC[i, :]
            mask = np.isfinite(fc_vec) & np.isfinite(sc_vec)
            if np.count_nonzero(mask) < 2:
                sfc_vec[i] = np.nan
                continue
            try:
                # Suppress SciPy's ConstantInputWarning when inputs are constant
                # (pearsonr raises a warning in that case). We still catch
                # exceptions and record NaN for problematic pairs.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", sp_stats.ConstantInputWarning)
                    r, _ = sp_stats.pearsonr(sc_vec[mask], fc_vec[mask])
                    nan_counter += 1 if not np.isfinite(r) else 0
            except Exception:
                r = np.nan
                nan_counter += 1
            sfc_vec[i] = r if np.isfinite(r) else np.nan
        # percent_nan reports the proportion of nodes (for this subject)
        # for which the SFC correlation was invalid. Note this variable
        # is overwritten for each subject and the function returns the
        # last subject's percent_nan value alongside the list of vectors
        # (preserving prior behavior of the codebase).
        percent_nan = (nan_counter / n_nodes) * 100

        # Convert NaNs to numeric zeros for downstream matrix operations;
        # this keeps array dtypes numeric and avoids propagation of NaNs
        # into clustering and distance computations.
        sfc_vec = np.nan_to_num(sfc_vec, nan=0.0)
        sfc_features.append(sfc_vec)
    # Return a list of per-subject SFC vectors and the last computed
    # percent_nan summary (behavior preserved from earlier versions).
    return sfc_features, percent_nan

def subject_scalar_summary(features, summary_func=np.nanmedian):
    """Convert per-subject vector features to scalar summaries.

    Parameters
    ----------
    features : list of array-like
        List of feature vectors (one per subject).
    summary_func : callable, default `np.nanmedian`
        Function to summarize each vector (e.g., `np.nanmean`, `np.nanmedian`).

    Returns
    -------
    np.ndarray
        1D array of scalar summaries, one per subject.
    """
    arrs = [np.asarray(f) for f in features]
    out = []
    for a in arrs:
        if a.ndim == 0:
            out.append(float(a))
        elif a.size == 1:
            out.append(float(a.ravel()[0]))
        else:
            out.append(float(summary_func(a)))
    return np.asarray(out, dtype=float)

# ==============================================================================
# CLUSTERING AND VALIDATION
# ==============================================================================
def perform_hierarchical_clustering(X, n_clusters=2):
    """Perform Ward linkage hierarchical clustering and return flat clusters.

    Parameters
    ----------
    X : np.ndarray, shape (n_subjects, n_features)
        Feature matrix used for clustering.
    n_clusters : int, default 2
        Number of clusters to form when cutting the dendrogram.

    Returns
    -------
    Z : np.ndarray
        Linkage matrix (SciPy format).
    clusters : np.ndarray, shape (n_subjects,)
        Integer cluster labels, zero-indexed (0..n_clusters-1).
    """
    # Compute hierarchical clustering using Ward linkage. The linkage
    # matrix `Z` encodes the tree structure, and `fcluster` cuts the
    # tree into `n_clusters` flat clusters. SciPy's `fcluster` returns
    # 1-indexed labels, so we subtract 1 for zero-indexed labels.
    Z = linkage(X, method='ward')
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = clusters - 1  # Remap from 1-indexed to 0-indexed
    return Z, clusters


def compute_silhouette_scores(X, Z, k_range=range(2, 21)):
    """Compute silhouette scores across a range of cluster counts.

    Note: silhouette is undefined when only a single unique label exists for
    a given k; in that case this function uses NaN for the silhouette value.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix used for computing pairwise distances.
    Z : np.ndarray
        Linkage matrix (unused directly for silhouette computation but used
        to produce cluster labels via `fcluster`).
    k_range : iterable, default=range(2, 21)
        Values of k to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `k` and `silhouette_score`.
    """
    results = []
    for k in k_range:
        labels_k = fcluster(Z, t=k, criterion='maxclust')
        n_unique = len(np.unique(labels_k))
        if n_unique < 2:
            sil_val = np.nan
        else:
            try:
                sil_val = float(silhouette_score(X, labels_k, metric='euclidean'))
            except Exception:
                sil_val = np.nan
        results.append((k, sil_val))
    
    return pd.DataFrame(results, columns=['k', 'silhouette_score'])


def compute_calinski_harabasz_scores(X, Z, k_range=range(2, 21)):
    """Compute Calinski-Harabasz scores across a range of cluster counts.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    Z : np.ndarray
        Linkage matrix used to derive labels for each tested k.
    k_range : iterable, default range(2, 21)
        Range of k values to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `k` and `calinski_harabasz`.
    """
    results = []
    for k in k_range:
        labels_k = fcluster(Z, t=k, criterion='maxclust')
        if len(np.unique(labels_k)) < 2:
            ch_val = np.nan
        else:
            try:
                ch_val = float(calinski_harabasz_score(X, labels_k))
            except Exception:
                ch_val = np.nan
        results.append((k, ch_val))
    
    return pd.DataFrame(results, columns=['k', 'calinski_harabasz'])


def bootstrap_clustering_stability(X, original_labels, n_boot, seed=12345):
    """Compute bootstrap-based clustering stability metrics.

    This routine performs `n_boot` bootstrap resamples (sampling subjects
    with replacement), reclusters each bootstrap sample (k=2), and compares
    the bootstrap-derived labels to the original labels. Labels from the
    bootstrap are aligned (flipped) when doing so improves agreement.

    Parameters
    ----------
    X : np.ndarray, shape (n_subjects, n_features)
        Feature matrix used for clustering.
    original_labels : np.ndarray, shape (n_subjects,)
        Original cluster labels (expected numeric 0/1 encoding).
    n_boot : int
        Number of bootstrap iterations to perform.
    seed : int, default 12345
        Seed for the random number generator for reproducibility.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - `per_subject_stability` : np.ndarray, length n_subjects; proportion
          of times each subject retained the same label when sampled.
        - `jaccard_cluster0` : list of floats; Jaccard similarities (orig vs bootstrap)
          for cluster 0 per bootstrap iteration.
        - `jaccard_cluster1` : list of floats; Jaccard similarities for cluster 1.
        - `nmi_list` : list of floats; normalized mutual information values per
          bootstrap.
        - `sample_counts` : np.ndarray, how many times each subject was sampled.
        - `match_counts` : np.ndarray, how many times each subject matched its
          original label when sampled.
        - `cluster_label_order` : suggested plotting order for cluster labels.

    Notes
    -----
    - Runtime scales with `n_boot` and the clustering cost; large values of
      `n_boot` will increase execution time substantially.
    - When a bootstrap sample yields <2 unique labels, NMI and Jaccard for
      that iteration will be recorded as NaN.
    """
    # Reproducible bootstrap: use NumPy's Generator for random sampling
    rng = np.random.default_rng(seed)
    n_subjects = X.shape[0]
    
    sample_counts = np.zeros(n_subjects, dtype=int)
    match_counts = np.zeros(n_subjects, dtype=int)
    jaccard_cluster0 = []
    jaccard_cluster1 = []
    nmi_list = []
    
    def _jaccard(a, b):
        if not a and not b:
            return np.nan
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else np.nan
    
    for b in range(n_boot):
        indices_b = rng.integers(0, n_subjects, size=n_subjects)
        X_b = X[indices_b]
        
        Z_b = linkage(X_b, method='ward')
        labels_b = fcluster(Z_b, t=2, criterion='maxclust') - 1
        
        # If the bootstrap sample yields fewer than two unique labels we
        # cannot compute meaningful cluster agreement metrics; record
        # NaNs for this iteration and continue.
        if len(np.unique(labels_b)) < 2:
            nmi_list.append(np.nan)
            for subj_idx in indices_b:
                sample_counts[subj_idx] += 1
            jaccard_cluster0.append(np.nan)
            jaccard_cluster1.append(np.nan)
            continue
        
        # Attempt to align cluster labels between the bootstrap sample
        # and the original labels. Some clustering algorithms flip label
        # encoding arbitrarily; we choose the orientation that maximizes
        # agreement with the original labels.
        orig_sub_labels = original_labels[indices_b]
        agree = np.mean(labels_b == orig_sub_labels)
        agree_flipped = np.mean((1 - labels_b) == orig_sub_labels)
        if agree_flipped > agree:
            labels_b = 1 - labels_b
        
        # Update subject-level sampling and match counts. Note that
        # `indices_b` contains repeats because bootstrapping samples with
        # replacement; we increment the per-subject counters for each
        # time a subject was included in the bootstrap sample.
        for pos, subj_idx in enumerate(indices_b):
            sample_counts[subj_idx] += 1
            if labels_b[pos] == original_labels[subj_idx]:
                match_counts[subj_idx] += 1
        
        # Jaccard similarities
        boot_cluster0 = set(np.unique(indices_b[labels_b == 0]))
        boot_cluster1 = set(np.unique(indices_b[labels_b == 1]))
        orig_cluster0 = set(np.where(original_labels == 0)[0])
        orig_cluster1 = set(np.where(original_labels == 1)[0])
        jaccard_cluster0.append(_jaccard(orig_cluster0, boot_cluster0))
        jaccard_cluster1.append(_jaccard(orig_cluster1, boot_cluster1))
        
        # NMI
        try:
            nmi_val = float(normalized_mutual_info_score(orig_sub_labels, labels_b))
        except Exception:
            nmi_val = np.nan
        nmi_list.append(nmi_val)
    
    # Compute per-subject stability
    stability_scores = np.full(n_subjects, np.nan)
    mask = sample_counts > 0
    stability_scores[mask] = match_counts[mask] / sample_counts[mask]
    
    return {
        'per_subject_stability': stability_scores,
        'jaccard_cluster0': jaccard_cluster0,
        'jaccard_cluster1': jaccard_cluster1,
        'nmi_list': nmi_list,
        'sample_counts': sample_counts,
        'match_counts': match_counts,
        # Suggest a plotting order for clusters: ensure Cluster 0 is first
        'cluster_label_order': ['0', '1'],
    }

def analyze_cross_modality_agreement(cohorts_dir, validation_plots_dir):
    """Analyze cluster agreement across available connectivity modalities."""
    # This function compares cluster assignments for depressed subjects
    # across multiple connectivity modalities (functional, structural,
    # and SFC). It expects per-modality CSVs named using the project
    # convention and containing a `Cluster` column with string labels
    # like "Cluster 0" / "Cluster 1" for depressed subjects.
    all_conn_types = ['functional', 'structural', 'sfc']
    cluster_files = {
        ct: os.path.join(cohorts_dir, f'combined_cohort_F32_global_{ct}_connectivity_clusters.csv')
        for ct in all_conn_types
    }

    available_types = [ct for ct, path in cluster_files.items() if os.path.exists(path)]

    if len(available_types) >= 2:
        display_map = {ct: _display_conn_type(ct) for ct in available_types}
        print(f"  Found cluster files for: {', '.join(display_map.values())}")
        print("  Computing pairwise agreement...")

        type_to_df: Dict[str, pd.DataFrame] = {}
        for ct in available_types:
            # Read each per-modality clustered cohort and extract the
            # depressed-subject assignments. The 'Group' column is used to
            # filter for depressed subjects only; this ensures comparisons
            # reflect cluster membership among the clinical cohort.
            df_ct = pd.read_csv(cluster_files[ct])
            df_ct = df_ct[df_ct['Group'] == 'Depression'][['eid', 'Cluster']].copy()
            df_ct['eid'] = df_ct['eid'].astype(str)
            type_to_df[ct] = df_ct

        assignments_df = None
        for ct, df_ct in type_to_df.items():
            tmp = df_ct.rename(columns={'Cluster': f'Cluster_{ct}'})
            if assignments_df is None:
                assignments_df = tmp
            else:
                assignments_df = assignments_df.merge(tmp, on='eid', how='outer')

        if assignments_df is not None:
            assign_path = os.path.join(
                validation_plots_dir,
                'global_connectivity_cluster_assignments_depression_across_types.csv',
            )
            assignments_df.to_csv(assign_path, index=False)
            print(f"  Saved per-subject assignments across types to: {assign_path}")

        pair_results = []
        agreement_matrix = {ct: {ct2: np.nan for ct2 in available_types} for ct in available_types}
        for ct in available_types:
            agreement_matrix[ct][ct] = 1.0

        # For every unordered pair of available connectivity types compute
        # the proportion of depressed subjects that are assigned to the
        # same cluster across modalities. We merge per-modality tables on
        # `eid` and then compare the `Cluster` assignments.
        for i, ct1 in enumerate(available_types):
            for ct2 in available_types[i + 1:]:
                df1 = type_to_df[ct1]
                df2 = type_to_df[ct2]
                merged = df1.merge(df2, on='eid', suffixes=('_' + ct1, '_' + ct2))

                # If there are no overlapping depressed subjects between
                # the two modalities, there's nothing to compare.
                if merged.empty:
                    print(f"    No overlapping subjects for {display_map.get(ct1, ct1)} vs {display_map.get(ct2, ct2)}; skipping.")
                    continue

                # Map canonical text labels ('Cluster 0'/'Cluster 1') to
                # numeric codes to allow direct equality and flipping
                # comparisons. We only map the expected text labels; any
                # unexpected values will map to NaN and be excluded from
                # the comparison by the `valid` mask below.
                map_clusters = {'Cluster 0': 0, 'Cluster 1': 1}
                c1 = merged[f'Cluster_{ct1}'].map(map_clusters).to_numpy()
                c2 = merged[f'Cluster_{ct2}'].map(map_clusters).to_numpy()
                valid = (~np.isnan(c1)) & (~np.isnan(c2))
                c1, c2 = c1[valid], c2[valid]

                # If after mapping there are no valid paired labels skip.
                if c1.size == 0:
                    print(f"    {display_map.get(ct1, ct1)} vs {display_map.get(ct2, ct2)}: no valid cluster labels; skipping.")
                    continue

                # Compute agreement both in the direct encoding and after
                # flipping the second label (to handle label sign ambiguity
                # across independent clustering runs). Use the larger of the
                # two proportions as the robust agreement statistic.
                agree = np.mean(c1 == c2)
                agree_flipped = np.mean(c1 == 1 - c2)
                prop_same = max(agree, agree_flipped)

                pair_results.append({
                    'type_a': ct1,
                    'type_b': ct2,
                    'proportion_same_cluster': prop_same,
                    'n_subjects': int(c1.size),
                })

                # Fill the symmetric agreement matrix used for the heatmap.
                agreement_matrix[ct1][ct2] = prop_same
                agreement_matrix[ct2][ct1] = prop_same

                print(f"    {display_map.get(ct1, ct1)} vs {display_map.get(ct2, ct2)}: proportion same cluster = {prop_same:.3f} (n = {c1.size})")

        agreement_path = os.path.join(
            validation_plots_dir,
            'global_connectivity_cluster_agreement_across_types.csv',
        )
        pd.DataFrame(pair_results).to_csv(agreement_path, index=False)
        print(f"  Saved pairwise agreement summary to: {agreement_path}")

        # Summarize cluster label distributions per connectivity type and
        # save a CSV with counts and proportions. This table feeds the
        # small barplot below which visualizes the fraction of depressed
        # subjects assigned to each cluster for each modality.
        dist_rows = []
        for ct in available_types:
            df_ct = type_to_df[ct]
            counts = df_ct['Cluster'].value_counts()
            n_c0 = int(counts.get('Cluster 0', 0))
            n_c1 = int(counts.get('Cluster 1', 0))
            n_total = n_c0 + n_c1
            prop_c0 = n_c0 / n_total if n_total > 0 else np.nan
            prop_c1 = n_c1 / n_total if n_total > 0 else np.nan
            dist_rows.append({
                'connectivity_type': ct,
                'n_total': n_total,
                'n_cluster_0': n_c0,
                'n_cluster_1': n_c1,
                'prop_cluster_0': prop_c0,
                'prop_cluster_1': prop_c1,
            })

        dist_df = pd.DataFrame(dist_rows)
        dist_csv_path = os.path.join(
            validation_plots_dir,
            'global_connectivity_cluster_distribution_by_type.csv',
        )
        dist_df.to_csv(dist_csv_path, index=False)
        print(f"  Saved cluster distribution table to: {dist_csv_path}")

        # Create a grouped barplot showing proportions per modality.
        # We defensively handle missing entries by filling with NaN so
        # bars with missing data are simply not drawn.
        plt.figure(figsize=(6, 5))
        order_ct = [ct for ct in available_types]
        x = np.arange(len(order_ct))
        width = 0.35
        prop0 = [float(dist_df.loc[dist_df['connectivity_type'] == ct, 'prop_cluster_0'].iloc[0])
                 if not dist_df.loc[dist_df['connectivity_type'] == ct].empty else np.nan
                 for ct in order_ct]
        prop1 = [float(dist_df.loc[dist_df['connectivity_type'] == ct, 'prop_cluster_1'].iloc[0])
                 if not dist_df.loc[dist_df['connectivity_type'] == ct].empty else np.nan
                 for ct in order_ct]
        for i, ct in enumerate(order_ct):
            colors = _cluster_colors_for_conn_type(ct)
            plt.bar(x[i] - width / 2, prop0[i], width, color=colors["0"], alpha=0.9)
            plt.bar(x[i] + width / 2, prop1[i], width, color=colors["1"], alpha=0.9)
        plt.xticks(x, [display_map.get(ct, ct) for ct in order_ct])
        # The legend here is illustrative: the script uses 'All subjects'
        # and 'Identical label subjects' elsewhere; keep simple legend for
        # consistency across figures.
        from matplotlib.patches import Patch as _Patch
        legend_handles = [
            _Patch(facecolor='black', alpha=0.3, label='All subjects'),
            _Patch(facecolor='black', alpha=0.9, label='Identical label subjects'),
        ]
        plt.legend(handles=legend_handles, fontsize=8)
        plt.ylim(0, 1)
        plt.ylabel('Proportion of depressed subjects')
        plt.xlabel('Connectivity type')
        plt.title('Cluster membership distribution by type')
        plt.grid(alpha=0.3, axis='y')
        plt.legend(title='Cluster')
        dist_fig_path = os.path.join(
            validation_plots_dir,
            'global_connectivity_cluster_distribution_by_type_barplot.svg',
        )
        plt.savefig(dist_fig_path, dpi=300, bbox_inches='tight', format='svg')
        plt.close()
        print(f"  Saved cluster distribution barplot to: {dist_fig_path}")

        # Convert the agreement dict to a DataFrame ordered by
        # `available_types` and produce a heatmap for visualizing pairwise
        # agreement across modalities.
        mat_df = pd.DataFrame(agreement_matrix)
        mat_df = mat_df.reindex(index=available_types, columns=available_types)
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(mat_df, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='Blues')
        plt.title('Proportion of subjects in the same cluster\nacross connectivity types')
        plt.ylabel('Connectivity type')
        plt.xlabel('Connectivity type')
        ax.set_xticklabels([display_map.get(ct, ct) for ct in mat_df.columns], rotation=30, ha='right')
        ax.set_yticklabels([display_map.get(ct, ct) for ct in mat_df.index], rotation=0)
        plt.tight_layout()
        agree_fig_path = os.path.join(
            validation_plots_dir,
            'global_connectivity_cluster_agreement_across_types_heatmap.svg',
        )
        plt.savefig(agree_fig_path, dpi=300, bbox_inches='tight', format='svg')
        plt.close()
        print(f"  Saved agreement heatmap to: {agree_fig_path}")

        # If we have per-modality assignments (`assignments_df`) then
        # compute whether some subjects have identical cluster labels
        # across modalities and create overlay plots that show how many
        # subjects are identically labelled across modality combinations.
        # This is useful to inspect cross-modality consistency within the
        # depressed cohort.
        if assignments_df is not None:
            cluster_cols = [c for c in assignments_df.columns if c.startswith('Cluster_')]

            # Require at least two modalities with labels to build overlays.
            if len(cluster_cols) >= 2:
                # Build a small tidy table summarizing, for each modality
                # and each cluster label, the total number of subjects with
                # that label and the number of subjects whose label is
                # identical across the requested set of modalities.
                def _build_identical_distribution(modalities):
                    # Modalities: list of modality tokens, e.g. ['functional','sfc']
                    if len(modalities) < 2:
                        return None
                    combo_cols = [f'Cluster_{ct}' for ct in modalities]
                    # If any requested column is missing we cannot compute
                    # the identical-label subset for that combination.
                    if any(col not in assignments_df.columns for col in combo_cols):
                        return None

                    # Keep only subjects that have non-missing labels for all
                    # modalities in the combo; drop rows with NA in any of
                    # the combo columns so counts reflect complete cases.
                    subset = assignments_df.dropna(subset=combo_cols).copy()
                    if subset.empty:
                        return None

                    # Identify rows where all combo columns have the same
                    # label (nunique == 1). These are the `identical` subset.
                    identical_mask = subset[combo_cols].nunique(axis=1) == 1
                    identical_subset = subset.loc[identical_mask].copy()

                    # Build per-modality rows with counts for total and
                    # identical-labelled subjects for Cluster 0 and 1.
                    rows = []
                    for ct, col in zip(modalities, combo_cols):
                        counts_total = subset[col].value_counts()
                        counts_ident = identical_subset[col].value_counts()
                        for cl in ['Cluster 0', 'Cluster 1']:
                            rows.append({
                                'connectivity_type': ct,
                                'cluster': cl,
                                'n_total': int(counts_total.get(cl, 0)),
                                'n_identical': int(counts_ident.get(cl, 0)),
                            })

                    return pd.DataFrame(rows)

                # Plotting helper: draw a grouped bar chart overlaying the
                # total counts and the subset that are identically labelled
                # across the provided modalities. The function handles
                # missing modalities and empty results defensively, writing
                # a placeholder text on the axis when necessary.
                def _plot_overlay(ax, modalities, title):
                    # If some requested modalities are not present in the
                    # assignments table, render a message and skip plotting.
                    missing = [ct for ct in modalities if f'Cluster_{ct}' not in assignments_df.columns]
                    if missing:
                        ax.text(0.5, 0.5, f"Missing modality data: {', '.join(missing)}",
                                ha='center', va='center')
                        ax.set_axis_off()
                        return

                    # Build the aggregated data for this modality combination.
                    data = _build_identical_distribution(modalities)
                    if data is None or data.empty:
                        ax.text(0.5, 0.5, 'No overlapping subjects with identical labels',
                                ha='center', va='center')
                        ax.set_axis_off()
                        return

                    # Keep only modalities that are both requested and
                    # available, and define plotting positions.
                    ordered_modalities = [ct for ct in modalities if ct in available_types]
                    if not ordered_modalities:
                        ax.text(0.5, 0.5, 'Modalities not available', ha='center', va='center')
                        ax.set_axis_off()
                        return

                    x = np.arange(len(ordered_modalities))
                    width = 0.35

                    # Helper to extract per-modality vectors for plotting.
                    def _counts(cluster_label, field):
                        vals = []
                        for ct in ordered_modalities:
                            subset = data[
                                (data['connectivity_type'] == ct) &
                                (data['cluster'] == cluster_label)
                            ]
                            if subset.empty:
                                vals.append(0)
                            else:
                                vals.append(int(subset[field].iloc[0]))
                        return np.asarray(vals, dtype=int)

                    totals0 = _counts('Cluster 0', 'n_total')
                    totals1 = _counts('Cluster 1', 'n_total')
                    ident0 = _counts('Cluster 0', 'n_identical')
                    ident1 = _counts('Cluster 1', 'n_identical')

                    # Draw semi-transparent bars for totals and shorter
                    # opaque bars in front for the identical-label subset.
                    for i, ct in enumerate(ordered_modalities):
                        colors = _cluster_colors_for_conn_type(ct)
                        ax.bar(x[i] - width / 2, totals0[i], width, color=colors["0"], alpha=0.3)
                        ax.bar(x[i] + width / 2, totals1[i], width, color=colors["1"], alpha=0.3)
                        ax.bar(x[i] - width / 2, ident0[i], width * 0.6, color=colors["0"], alpha=0.9)
                        ax.bar(x[i] + width / 2, ident1[i], width * 0.6, color=colors["1"], alpha=0.9)

                    ax.set_xticks(x)
                    ax.set_xticklabels([display_map.get(ct, ct) for ct in ordered_modalities])
                    ax.set_ylabel('Subjects')
                    ax.set_xlabel('Connectivity type')
                    ax.set_title(title)
                    ax.grid(alpha=0.3, axis='y')
                    from matplotlib.patches import Patch as _Patch
                    legend_handles = [
                        _Patch(facecolor='black', alpha=0.3, label='All subjects'),
                        _Patch(facecolor='black', alpha=0.9, label='Identical label subjects'),
                    ]
                    ax.legend(handles=legend_handles, fontsize=7)

                # If at least two modalities are present, write a CSV with
                # the identical-label distributions (helps with external
                # inspection) and produce a multi-panel figure showing the
                # overlay for several useful modality combinations.
                modalities_for_csv = [ct for ct in available_types if f'Cluster_{ct}' in assignments_df.columns]
                csv_df = _build_identical_distribution(modalities_for_csv)
                if csv_df is not None and not csv_df.empty:
                    dist_ident_csv_path = os.path.join(
                        validation_plots_dir,
                        'global_connectivity_cluster_distribution_with_identical_across_types.csv',
                    )
                    csv_df.to_csv(dist_ident_csv_path, index=False)
                    print(
                        "  Saved distribution with identical-label subset to: "
                        f"{dist_ident_csv_path}"
                    )
                else:
                    print(
                        "  No subjects with complete labels across available modalities; "
                        "skipping identical-label distribution CSV."
                    )

                # Build a 2x2 grid of overlays for common modality
                # combinations: all three, and each pairwise combo. This
                # provides quick visual checks of where identical labels
                # concentrate across modalities.
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                combo_specs = [
                    (
                        ['functional', 'structural', 'sfc'],
                        'All modalities (Functional, Structural, Structure-Function Coupling)'
                    ),
                    (['functional', 'structural'], 'Functional + Structural'),
                    (['functional', 'sfc'], 'Functional + Structure-Function Coupling'),
                    (['structural', 'sfc'], 'Structural + Structure-Function Coupling'),
                ]

                for ax, (mods, title) in zip(axes.flat, combo_specs):
                    _plot_overlay(ax, mods, title)

                fig.subplots_adjust(hspace=0.35, wspace=0.25)
                ident_fig_path = os.path.join(
                    validation_plots_dir,
                    'global_connectivity_cluster_distribution_with_identical_across_types_barplot.svg',
                )
                fig.savefig(ident_fig_path, dpi=300, bbox_inches='tight', format='svg')
                plt.close(fig)
                print(f"  Saved multi-panel identical-label overlay plot to: {ident_fig_path}")
            else:
                print("  Not enough modalities with cluster labels for identical-label overlay.")
    else:
        print("  Not enough connectivity types available for agreement analysis")

    return available_types

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
def plot_dendrogram(linkage_matrix, conn_type, out_dir):
    """Plot and save a hierarchical clustering dendrogram (Ward linkage).

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix as returned by `scipy.cluster.hierarchy.linkage`.
    conn_type : str
        Connectivity type used for figure title and filename.
    out_dir : str
        Directory where the SVG will be written. The function will create
        directories as needed when saving.

    Returns
    -------
    None
        The dendrogram SVG is written to ``<out_dir>/<conn_type>_con/individual_avg_conn_dendrogram.svg``.
    """
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
    plt.ylabel('Distance')
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f'{conn_type}_con/individual_avg_conn_dendrogram.svg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='svg')
    plt.close()
    print(f"Saved dendrogram to: {out_path}")

def plot_bootstrap_diagnostics(stability_results, clusters, conn_type, out_dir, analysis_level='global', dir_type = None):
    """Create and save a multi-panel bootstrap stability diagnostics figure.

    Parameters
    ----------
    stability_results : dict
        Output from `bootstrap_clustering_stability` (see that function's
        docstring for the returned keys and types).
    clusters : np.ndarray
        Integer cluster labels aligned to subjects used to compute
        `stability_results`.
    conn_type : str
        Connectivity type (e.g., 'functional', 'structural', 'sfc'). Used to
        select plotting colors and for titles/filenames.
    out_dir : str
        Directory where the combined diagnostics SVG will be written.
    analysis_level : str, default 'global'
        Prefix used for the saved filename (e.g., 'global' or 'modular').
    dir_type : str or None, default None
        Optional direction label to include in titles (for directional
        connectivities); if None no suffix is appended.

    Returns
    -------
    None
        Saves a figure named ``{analysis_level}_{conn_type}_{dir_suffix}_bootstrap_diagnostics_combined.svg``
        under `out_dir` and prints the saved path.
    """
    # Unpack bootstrap diagnostics and create human-friendly arrays for
    # plotting. Many plotting branches are guarded to handle cases where
    # diagnostic vectors contain NaNs (e.g., small sample sizes).
    stability = stability_results['per_subject_stability']
    nmi_arr = np.asarray(stability_results['nmi_list'], dtype=float)
    jacc0_arr = np.asarray(stability_results['jaccard_cluster0'], dtype=float)
    jacc1_arr = np.asarray(stability_results['jaccard_cluster1'], dtype=float)
    
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[2.5, 1.2, 1.2, 1.0], 
                          hspace=0.45, wspace=0.3)
    
    # Determine title suffix based on dir_type
    if dir_type is None:
        dir_suffix = ''
    else:
        dir_suffix = f' ({dir_type})'

    cluster_colors = _cluster_colors_for_conn_type(conn_type)

    # Per-subject stability barplot: when stability contains NaNs
    # (subjects never sampled) only plot the valid subset to avoid
    # rendering artifacts.
    ax_bar = fig.add_subplot(gs[0, :])
    valid_idx = np.where(~np.isnan(stability))[0]
    if valid_idx.size > 0:
        st_valid = stability[valid_idx]
        cl_valid = clusters[valid_idx].astype(int)
        order = np.argsort(st_valid)[::-1]
        st_sorted = st_valid[order]
        cl_sorted = cl_valid[order]
        colors = np.array([cluster_colors["0"], cluster_colors["1"]])[cl_sorted]
        ax_bar.bar(np.arange(len(st_sorted)), st_sorted, color=colors, 
                  edgecolor='k', linewidth=0.2)
        ax_bar.set_ylabel('Per-subject stability')
        ax_bar.set_xlabel('Subjects (sorted by stability)')
        ax_bar.set_title(f'Per-subject cluster stability (bootstrap) — {conn_type} — {dir_suffix}')
        ax_bar.set_ylim(0, 1.02)
        handles = [mpatches.Patch(color=cluster_colors["0"], label='Cluster 0'), 
              mpatches.Patch(color=cluster_colors["1"], label='Cluster 1')]
        ax_bar.legend(handles=handles, loc='upper right')
    
    # NMI timeseries: plotting is defensive — only render panels when
    # non-NaN data exist.
    ax_nmi_ts = fig.add_subplot(gs[1, 0])
    if np.any(~np.isnan(nmi_arr)):
        ax_nmi_ts.plot(np.arange(len(nmi_arr)), nmi_arr, marker='o', 
                      linestyle='-', alpha=0.6)
        ax_nmi_ts.axhline(np.nanmean(nmi_arr), color='red', linestyle='--', 
                         label=f"Mean = {np.nanmean(nmi_arr):.3f}")
        ax_nmi_ts.set_xlabel('Bootstrap iteration')
        ax_nmi_ts.set_ylabel('NMI')
        ax_nmi_ts.set_title(f'NMI across bootstraps — {conn_type} — {dir_suffix}')
        ax_nmi_ts.legend()
    
    # NMI histogram
    ax_nmi_hist = fig.add_subplot(gs[1, 1])
    if np.any(~np.isnan(nmi_arr)):
        sns.histplot(nmi_arr[~np.isnan(nmi_arr)], bins=30, kde=True, 
                    ax=ax_nmi_hist, color='gray')
        ax_nmi_hist.set_xlabel('NMI')
        ax_nmi_hist.set_title('NMI Distribution')
    
    # Jaccard histograms: plot per-cluster Jaccard similarities from
    # bootstrap iterations using connectivity-type-specific colors.
    ax_j0 = fig.add_subplot(gs[2, 0])
    if np.any(~np.isnan(jacc0_arr)):
        sns.histplot(jacc0_arr[~np.isnan(jacc0_arr)], bins=30, kde=True, 
                color=cluster_colors["0"], ax=ax_j0)
        ax_j0.set_xlabel('Jaccard similarity')
        ax_j0.set_title(f'Bootstrap Jaccard — Cluster 0 ({conn_type}) — {dir_suffix}')
        ax_j0.grid(alpha=0.3, axis='y')
    
    ax_j1 = fig.add_subplot(gs[2, 1])
    if np.any(~np.isnan(jacc1_arr)):
        sns.histplot(jacc1_arr[~np.isnan(jacc1_arr)], bins=30, kde=True, 
                color=cluster_colors["1"], ax=ax_j1)
        ax_j1.set_xlabel('Jaccard similarity')
        ax_j1.set_title(f'Bootstrap Jaccard — Cluster 1 ({conn_type}) — {dir_suffix}')
        ax_j1.grid(alpha=0.3, axis='y')
    
    # Stability by cluster boxplot: the boxplot groups by cluster;
    # require at least two unique cluster labels to produce a meaningful
    # comparative plot.
    ax_box = fig.add_subplot(gs[3, :])
    cluster_df = pd.DataFrame({
        'Cluster': clusters,
        'cluster_stability': stability
    })
    if cluster_df['Cluster'].nunique(dropna=True) >= 2:
        # Respect suggested label order from stability results if provided
        order_list = stability_results.get('cluster_label_order', ['0', '1'])
        sns.boxplot(data=cluster_df.assign(Cluster_str=cluster_df['Cluster'].astype(str)),
               x='Cluster_str', y='cluster_stability', hue='Cluster_str',
               order=order_list, dodge=False, palette=cluster_colors, ax=ax_box)
        ax_box.set_xlabel('Cluster')
        ax_box.set_ylabel('Per-subject stability')
        ax_box.set_title(f'Stability by cluster — {conn_type}')
        ax_box.grid(alpha=0.3, axis='y')
        try:
            leg = ax_box.get_legend()
            if leg is not None:
                leg.remove()
        except Exception:
            pass
    
    out_path = os.path.join(out_dir, 
                           f'{analysis_level}_{conn_type}_{dir_suffix}_bootstrap_diagnostics_combined.svg')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', format='svg')
    plt.close(fig)
    print(f"Saved bootstrap diagnostics to: {out_path}")

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
    
    cluster_colors = _cluster_colors_for_conn_type(conn_type)
    sns.violinplot(data=ctrl_df, x='Position', y='Connectivity', ax=ax, color="#2ca02c")
    sns.violinplot(data=dep_df, x='Position', y='Connectivity', ax=ax, color="#6a3d9a")
    sns.violinplot(data=c0_df, x='Position', y='Connectivity', ax=ax, color=cluster_colors["0"])
    sns.violinplot(data=c1_df, x='Position', y='Connectivity', ax=ax, color=cluster_colors["1"])
    
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
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight', format='svg')
    plt.close()
    print("  Saved violin plot with significance brackets")

def determine_covariate_distributions(
    combined_df,
    available_types,
    conn_type,
    motion_metric,
    out_dir,
    cohorts_dir,
    icd_covariates=None,
    group_col='Group',
):
    """Determine and test covariate distributions by group and cluster with FDR comparisons.

    This function produces a multi-row, multi-column figure summarizing
    demographic and motion covariates as well as ICD-10 comorbidity counts.
    For each row configuration (e.g., by Group, by Cluster, and by other
    connectivity modalities) the function runs per-comparison tests
    (Mann-Whitney U for continuous variables, Chi-square for categorical),
    applies FDR correction via `apply_multiple_testing_correction`, logs
    results to CSV, and annotates significant comparisons on the plots.

    This function also generates a TXT summary table (`global_covariate_distribution_summary.txt`) 
    that aggregates numeric summaries (n/mean/std/median/Q1/Q3 for continuous variables 
    and counts/percentages for binary variables) along with raw and FDR-corrected 
    p-values for each comparison.

    Expected columns in `combined_df` (if present):
    - `eid`, `Group`, `Cluster`, `p21003_i2` (age), `p31` (sex),
      `p24441_i2`, `p24453_i2` (motion metrics), plus any ICD covariate
      columns provided in `icd_covariates`.

    Side effects
    ------------
    - Writes per-row test CSVs to `out_dir` (if tests are performed).
    - Writes several SVGs summarizing covariate distributions.
    - Writes a TXT table (`global_covariate_distribution_summary.txt`) with
        numeric covariate summaries (n/mean/std/median/Q1/Q3 and binary counts)
        and merged raw/FDR test results.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined cohort dataframe with demographic, cluster, and motion columns.
    available_types : list-like
        Connectivity modalities available on disk (used to include cross-type rows).
    conn_type : str
        Primary connectivity type used for labeling (e.g., 'functional').
    motion_metric : str
        Default motion metric column name (used when modality-specific name is missing).
    out_dir : str
        Directory where CSVs and SVGs will be saved.
    cohorts_dir : str
        Directory containing per-modality combined cohort CSVs (used when
        building additional rows for other modalities).
    icd_covariates : list of str, optional
        Column names for ICD-10 comorbidity indicators to include.
    group_col : str, default 'Group'
        Column name to use as the group indicator when `Group` is not present.

    Returns
    -------
    None
        Saves figures, CSVs and TXT summary table to `out_dir` and prints saved paths.
    """
    # Normalize the ICD covariates argument to a list for downstream loops
    icd_covariates = list(icd_covariates or [])
    covariate_columns = ["Age", "Sex", "Head Motion"] + [str(c) for c in icd_covariates]
    def _category_colors_for_row(row_conn_type: str) -> Dict[str, str]:
        """
        Return a mapping of category label to color for a plotting row.

        Parameters
        ----------
        row_conn_type : str
            Connectivity type used to select cluster colors.

        Returns
        -------
        dict
            Mapping of labels ('Control','Cluster 0','Cluster 1','Depression') to colors.
        """
        # Get cluster-specific colors for the provided connectivity type
        cluster_colors = _cluster_colors_for_conn_type(row_conn_type)
        # Map canonical labels to colors so plotting is consistent across rows
        return {
            'Control': "#2ca02c",
            'Cluster 0': cluster_colors["0"],
            'Cluster 1': cluster_colors["1"],
            'Depression': "#6a3d9a",
        }

    # Map a connectivity type to its default motion column name
    motion_lookup = {
        'functional': 'p24441_i2',
        'structural': 'p24453_i2',
        'sfc': 'p24441_i2',
    }

    def _ensure_group_column(df):
        """
        Ensure the returned DataFrame has a `Group` column.

        If `Group` is missing but `group_col` exists this maps 0/1 to
        'Control'/'Depression' and returns a copy. Otherwise returns the input.
        """
        # Ensure a `Group` column exists. If not present but an alternate
        # `group_col` is provided (e.g., 0/1 values) map those to labels.
        if 'Group' in df.columns:
            return df
        if group_col in df.columns:
            df = df.copy()
            # Map binary/group codes to readable labels and preserve non-binary as str
            df['Group'] = df[group_col].map({0: 'Control', 1: 'Depression'}).fillna(df[group_col].astype(str))
            return df
        return df

    def _normalize_cluster_label(value):
        """
        Normalize cluster label strings into a consistent format.

        Examples
        --------
        - 'cluster_0' -> 'Cluster 0'
        - 0 -> 'Cluster 0'
        - 'Control' -> 'Control'
        """
        # Normalize cluster labels to a consistent human-readable form
        if pd.isna(value):
            return value
        value_str = str(value).strip()
        # If already has the word 'cluster' keep it but normalize underscores
        if value_str.lower().startswith('cluster'):
            return value_str.replace('_', ' ')
        # Map explicit 'control' string to canonical 'Control'
        if value_str.lower() == 'control':
            return 'Control'
        # Try to coerce numeric labels like 0 or '0' into 'Cluster 0'
        try:
            return f"Cluster {int(float(value_str))}"
        except (ValueError, TypeError):
            # Fall back to returning the original string
            return value_str

    def _wrap_title(text, width=28):
        """
        Wrap a title string to a maximum line width for plotting.

        Parameters
        ----------
        text : str
            Title text to wrap.
        width : int
            Maximum line width.

        Returns
        -------
        str
            Wrapped title text.
        """
        # Short utility to wrap long titles for axis headers
        return textwrap.fill(str(text), width=width)

    def _format_sig_label(pval, label_prefix=None):
        """
        Format a p-value into a short significance label (stars or ns).

        Parameters
        ----------
        pval : float
            p-value to format.
        label_prefix : str, optional
            Optional prefix to include before the label (e.g., name of covariate).

        Returns
        -------
        str
            Formatted significance label.
        """
        # Convert pvalue to float if possible and map to star labels
        try:
            pval = float(pval)
        except (TypeError, ValueError):
            label = 'ns'
        else:
            if not np.isfinite(pval):
                label = 'ns'
            elif pval < 0.001:
                label = '***'
            elif pval < 0.01:
                label = '**'
            elif pval < 0.05:
                label = '*'
            else:
                label = 'ns'
        # Prepend an optional prefix (e.g., covariate name) to the sig label
        if label_prefix:
            return f"{label_prefix}: {label}"
        return label

    def _draw_bracket(ax, x1, x2, y, height, label):
        """
        Draw a significance bracket with a label on the given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw on.
        x1, x2 : int or float
            X positions for bracket ends.
        y : float
            Baseline y coordinate.
        height : float
            Vertical height of the bracket.
        label : str
            Text label to draw above the bracket.
        """
        # Draw a simple significance bracket between two x positions
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color='black', linewidth=1.0)
        ax.text((x1 + x2) / 2.0, y + height * 1.05, label, ha='center', va='bottom', fontsize=8)

    def _annotate_axis(ax, present_groups, comp_dicts, y_max=None):
        """
        Annotate an axis with significance brackets based on comparison dicts.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to annotate.
        present_groups : list[str]
            Ordered list of present group labels corresponding to x positions.
        comp_dicts : list[dict]
            List of comparison dictionaries containing 'comparisons' and 'pvals'.
        y_max : float, optional
            Maximum y-value to use as a baseline for annotation scaling.
        """
        # Annotate an axis with significance brackets for all requested comparisons
        if not present_groups:
            return
        # Determine baseline y-range to place brackets above the plotted data
        if y_max is None or not np.isfinite(y_max):
            y_min, y_max = ax.get_ylim()
        else:
            # When a user-specified y_max is provided, use 0 as the bottom
            y_min = 0.0
        data_range = y_max - y_min
        if not np.isfinite(data_range) or data_range <= 0:
            data_range = 1.0
        step = data_range * 0.12
        current_y = y_max + step * 0.25

        def _is_sig(pval):
            try:
                pval = float(pval)
            except (TypeError, ValueError):
                return False
            return np.isfinite(pval) and pval < 0.05

        # Loop through each comparison block and draw brackets for significant ones
        for comp in comp_dicts:
            label_prefix = comp.get('label_prefix')
            comp_map = comp.get('comparisons', {})
            for comp_name, groups in comp_map.items():
                if not all(g in present_groups for g in groups):
                    continue
                pval = comp.get('pvals', {}).get(comp_name)
                if not _is_sig(pval):
                    continue
                label = _format_sig_label(pval, label_prefix=label_prefix)
                idx_a = present_groups.index(groups[0])
                idx_b = present_groups.index(groups[1])
                if idx_a == idx_b:
                    continue
                height = step * 0.30
                _draw_bracket(ax, idx_a, idx_b, current_y, height, label)
                current_y += step
        ax.set_ylim(y_min, current_y + step * 0.30)

    def _mannwhitney_test(df, x_col, group_a, group_b, value_col):
        """
        Run a two-sided Mann-Whitney U test between two groups for a value column.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the data.
        x_col : str
            Column name for the grouping variable.
        group_a, group_b : str
            Names of the groups to compare.
        value_col : str
            Column name for the values to test.

        Returns
        -------
        tuple
            (statistic, p-value) or (np.nan, np.nan) on failure or missing data.
        """
        # Run Mann-Whitney U between two groups for a numeric column.
        if value_col not in df.columns:
            return np.nan, np.nan
        vals_a = df.loc[df[x_col] == group_a, value_col].dropna()
        vals_b = df.loc[df[x_col] == group_b, value_col].dropna()
        if vals_a.empty or vals_b.empty:
            return np.nan, np.nan
        try:
            stat, pval = sp_stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            return float(stat), float(pval)
        except Exception:
            # Return NaNs on test failure rather than raising here
            return np.nan, np.nan

    def _chi2_test(df, x_col, group_a, group_b, value_col):
        """
        Run a Chi-square test of independence for a 2xN contingency table.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the data.
        x_col : str
            Column name for the grouping variable.
        group_a, group_b : str
            Names of the groups to compare.
        value_col : str
            Column name for the values to test.

        Returns
        -------
        tuple
            (statistic, p-value) or (np.nan, np.nan) on failure or missing data.
        """
        # Run chi-square on contingency table between two groups for a categorical column
        if value_col not in df.columns:
            return np.nan, np.nan
        df_sub = df[df[x_col].isin([group_a, group_b])].copy()
        if df_sub.empty:
            return np.nan, np.nan
        table = pd.crosstab(df_sub[x_col], df_sub[value_col])
        # Require at least 2x2 table with non-zero margins
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan, np.nan
        if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
            return np.nan, np.nan
        try:
            stat, pval, _, _ = sp_stats.chi2_contingency(table.values)
            return float(stat), float(pval)
        except Exception:
            return np.nan, np.nan

    def _compute_covariate_tests(row_cfg):
        """
        Compute statistical tests for a covariate row configuration.

        Parameters
        ----------
        row_cfg : dict
        Configuration containing keys 'df', 'x_col', and other plotting params.

        Returns
        -------
        tuple
            (sig_map, log_df) where `sig_map` maps covariates to corrected p-values
            per comparison and `log_df` is a DataFrame with raw and corrected p-values.
        """
        # Build a list of (row_title, covariate, comparison, raw_pval, stat, test_name)
        tests = []
        comparisons_group = {
            'depression_vs_control': ('Depression', 'Control'),
        }
        comparisons_cluster = {
            'cluster0_vs_control': ('Cluster 0', 'Control'),
            'cluster1_vs_control': ('Cluster 1', 'Control'),
            'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
        }

        # Local copies of config for readability
        df = row_cfg['df']
        x_col = row_cfg['x_col']
        row_title = row_cfg['title']
        # Choose comparisons depending on whether we group by 'Group' or 'Cluster'
        comp_map = comparisons_group if x_col == 'Group' else comparisons_cluster

        # Test age differences (non-parametric) between the requested groups
        if 'p21003_i2' in df.columns:
            for comp_name, (g1, g2) in comp_map.items():
                stat, pval = _mannwhitney_test(df, x_col, g1, g2, 'p21003_i2')
                tests.append((row_title, 'Age', comp_name, pval, stat, 'Mann-Whitney U'))

        # Handle motion metrics: can be a single column or a list of columns
        motion_col = row_cfg.get('motion_col')
        if isinstance(motion_col, (list, tuple)):
            for col in motion_col:
                if col not in df.columns:
                    continue
                cov_key = f"Head Motion ({col})"
                for comp_name, (g1, g2) in comp_map.items():
                    stat, pval = _mannwhitney_test(df, x_col, g1, g2, col)
                    tests.append((row_title, cov_key, comp_name, pval, stat, 'Mann-Whitney U'))
        elif isinstance(motion_col, str):
            if motion_col in df.columns:
                for comp_name, (g1, g2) in comp_map.items():
                    stat, pval = _mannwhitney_test(df, x_col, g1, g2, motion_col)
                    tests.append((row_title, 'Head Motion', comp_name, pval, stat, 'Mann-Whitney U'))

        # Test sex (categorical) differences using chi-square
        if 'p31' in df.columns:
            for comp_name, (g1, g2) in comp_map.items():
                stat, pval = _chi2_test(df, x_col, g1, g2, 'p31')
                tests.append((row_title, 'Sex', comp_name, pval, stat, 'Chi-square'))

        # When comparing clusters, also test ICD covariates between Cluster 0 and 1
        if x_col == 'Cluster' and 'Cluster' in df.columns:
            df_icd = df.copy()
            df_icd = df_icd[df_icd['Cluster'].isin(['Cluster 0', 'Cluster 1'])]
            icd_comp_map = {
                'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
            }
            for cov in icd_covariates:
                if cov not in df_icd.columns:
                    continue
                cov_key = str(cov)
                for comp_name, (g1, g2) in icd_comp_map.items():
                    stat, pval = _chi2_test(df_icd, 'Cluster', g1, g2, cov)
                    tests.append((row_title, cov_key, comp_name, pval, stat, 'Chi-square'))

        # If no tests were added, return empty results
        if not tests:
            return {}, None

        # Extract raw p-values and run multiple testing correction
        raw_pvals = np.array([t[3] for t in tests], dtype=float)
        test_methods = [t[2] for t in tests]
        variable_names = [f"{t[0]}::{t[1]}" for t in tests]

        corrected_pvals = np.full_like(raw_pvals, np.nan, dtype=float)
        valid = np.isfinite(raw_pvals)
        if np.any(valid):
            _, corr_vals = apply_multiple_testing_correction(
                p_values=raw_pvals[valid].tolist(),
                test_methods=[t for t, v in zip(test_methods, valid) if v],
                variable_names=[v for v, ok in zip(variable_names, valid) if ok],
            )
            corrected_pvals[valid] = corr_vals

        # Build a mapping of covariate -> comparison -> corrected_pval
        sig_map = {}
        log_rows = []
        for (row_title, cov_key, comp_name, pval_raw, stat_val, test_name), pval_corr in zip(tests, corrected_pvals):
            sig_map.setdefault(cov_key, {})[comp_name] = pval_corr
            log_rows.append({
                'row_title': row_title,
                'covariate': cov_key,
                'comparison': comp_name,
                'test_name': test_name,
                'test_statistic': stat_val,
                'p_value_raw': pval_raw,
                'p_value_fdr': pval_corr,
            })
        # Return map of corrected p-values and a log DataFrame for later saving
        log_df = pd.DataFrame(log_rows)
        return sig_map, log_df

    def _plot_base_covariates(row_axes, df, x_col, motion_col, sig_map, category_colors):
        """
        Render base covariate plots (age, sex, motion) into provided axes.

        Parameters
        ----------
        row_axes : sequence of matplotlib.axes.Axes
            Axes to draw on (expected subplots for age, sex, motion).
        df : pd.DataFrame
            Dataframe containing plotting variables.
        x_col : str
            Column used for x-axis grouping ('Group' or 'Cluster').
        motion_col : str or sequence
            Motion column(s) to plot.
        sig_map : dict
            Mapping from covariate to comparison p-values for annotation.
        category_colors : dict
            Mapping of category label to color.
        """

        # Map internal motion column names to human-friendly labels
        motion_label_map = {
            'p24441_i2': 'rfMRI-derived',
            'p24453_i2': 'dMRI-derived',
        }
        # Prepare the canonical ordering of groups we want to show in plots
        order = ['Control', 'Cluster 0', 'Cluster 1'] if x_col == 'Cluster' else ['Control', 'Depression']
        present = [o for o in order if o in df.get(x_col, pd.Series(dtype=str)).unique()]
        # Palette as a list ordered by present groups for x-based violin plots
        # Explicitly ensure Control is always green, not cluster-colored
        palette_list = []
        for k in present:
            if k == 'Control':
                palette_list.append("#2ca02c")  # Always use green for Control
            else:
                palette_list.append(category_colors.get(k, 'grey'))
        # Age violin: use x-axis positioning so violins are side-by-side
        if 'p21003_i2' in df.columns:
            sns.violinplot(data=df, x=x_col, y='p21003_i2', order=present,
                           palette=palette_list, ax=row_axes[0])
            row_axes[0].set_xlabel('')
            row_axes[0].set_ylabel('')
            age_pvals = sig_map.get('Age', {})
            comp_dicts = [{'label_prefix': None, 'comparisons': {
                'depression_vs_control': ('Depression', 'Control'),
                'cluster0_vs_control': ('Cluster 0', 'Control'),
                'cluster1_vs_control': ('Cluster 1', 'Control'),
                'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
            }, 'pvals': age_pvals}]
            _annotate_axis(row_axes[0], present, comp_dicts)
        else:
            row_axes[0].set_visible(False)

        # Sex countplot: show counts per sex value across groups (uses countplot)
        hue_col = 'p31'
        if hue_col in df.columns:
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
            sns.countplot(data=df, x=x_col, hue=hue_col, order=present, palette=sex_palette, ax=row_axes[1])
            # The countplot adds a legend; preserve it (we remove only violin legends earlier)
        else:
            row_axes[1].set_visible(False)
        row_axes[1].set_title('')
        row_axes[1].set_ylabel('')
        sex_pvals = sig_map.get('Sex', {})
        if sex_pvals and present:
            counts = pd.crosstab(df[x_col], df[hue_col]) if hue_col in df.columns else pd.DataFrame()
            y_max = float(counts.values.max()) if not counts.empty else None
            comp_dicts = [{'label_prefix': None, 'comparisons': {
                'depression_vs_control': ('Depression', 'Control'),
                'cluster0_vs_control': ('Cluster 0', 'Control'),
                'cluster1_vs_control': ('Cluster 1', 'Control'),
                'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
            }, 'pvals': sex_pvals}]
            _annotate_axis(row_axes[1], present, comp_dicts, y_max=y_max)

        # Motion plotting: support either a list of motion metrics (melted plot)
        # or a single-column violin.
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
                    # Suppress seaborn legend here; we will add a custom one below
                    legend=False,
                    ax=row_axes[2],
                )
                # Add a formatted legend for head motion metrics
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
            # Use same approach as age violin - create palette list from category_colors
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
        """
        Plot ICD-10 comorbidity bar plots into ICD-dedicated axes.

        Parameters
        ----------
        row_axes : sequence of matplotlib.axes.Axes
            Axes for ICD-related plots.
        df : pd.DataFrame
            Input data (filtered to Depression when appropriate).
        x_col : str
            Grouping variable name.
        sig_map : dict
            Map of ICD covariates to corrected p-values for annotation.
        category_colors : dict
            Color mapping for categories.
        """
        # If no ICD covariates requested, hide ICD axes and return early
        if not icd_covariates:
            for ax in row_axes[3:]:
                ax.set_visible(False)
            return

        # Restrict ICD plots to the Depression group (or entire df if Group not present)
        df_dep = df[df.get('Group') == 'Depression'].copy() if 'Group' in df.columns else df.copy()
        if x_col == 'Cluster' and 'Cluster' in df_dep.columns:
            df_dep = df_dep[df_dep['Cluster'] != 'Control']

        cluster_order = ['Cluster 0', 'Cluster 1']
        present_clusters = [c for c in cluster_order if c in df_dep.get('Cluster', pd.Series(dtype=str)).unique()]

        # Build a small table with counts per ICD and label for plotting
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

        # If plotting by Group, show only Depression counts as a simple barplot
        if x_col == 'Group':
            dep_df = plot_df[plot_df['label'] == 'Depression']
            if dep_df.empty:
                icd_ax.set_visible(False)
            else:
                sns.barplot(data=dep_df, x='ICD', y='count', color=category_colors.get('Depression', "#6a3d9a"), ax=icd_ax)
                icd_ax.set_title('')
                icd_ax.set_xlabel('ICD')
                icd_ax.set_ylabel('')
                icd_ax.tick_params(axis='x', rotation=45)
            for ax in row_axes[4:]:
                ax.set_visible(False)
            return

        palette = {'Depression': category_colors.get('Depression', "#6a3d9a")}
        for cl in present_clusters:
            palette[cl] = category_colors.get(cl, 'grey')

        # When comparing clusters, use a grouped barplot with the provided palette
        sns.barplot(data=plot_df, x='ICD', y='count', hue='label', palette=palette, ax=icd_ax)
        icd_ax.set_title('')
        icd_ax.set_xlabel('ICD')
        icd_ax.set_ylabel('')
        icd_ax.tick_params(axis='x', rotation=45)
        # Keep a titled legend for ICD plots so readers know which bar corresponds to which label
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

    def _summarize_covariate_values(df, x_col, cov_name, cov_col, groups):
        # Produce numeric summaries for each group for the given covariate
        rows = []
        for grp in groups:
            grp_df = df.loc[df[x_col] == grp]
            if cov_col not in grp_df.columns:
                continue
            values = grp_df[cov_col].dropna()
            # If no observations in this group, fill numerics with NaN/zeros
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
                # For numeric covariates compute n/mean/std/median/IQR
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

    combined_df = _ensure_group_column(combined_df)
    # Build a list of plotting row configurations (by Group, by Cluster, and cross-modality clusters)
    row_configs = [{
        'title': 'By Group',
        'df': combined_df,
        'x_col': 'Group',
        'motion_col': ['p24441_i2', 'p24453_i2'],
        'approach': None,
        'conn_type': str(conn_type).lower(),
    }]

    display_conn = _display_conn_type(conn_type)
    # Prepare a cluster-normalized copy used for cluster-based rows
    df_cluster = combined_df.copy()
    df_cluster['Cluster'] = df_cluster['Cluster'].apply(_normalize_cluster_label)
    if str(conn_type).lower() == 'sfc':
        row_configs.append({
            'title': f"By Cluster ({display_conn})",
            'df': df_cluster,
            'x_col': 'Cluster',
            'motion_col': ['p24441_i2', 'p24453_i2'],
            'approach': display_conn,
            'conn_type': str(conn_type).lower(),
        })
    else:
        row_configs.append({
            'title': f"By Cluster ({display_conn})",
            'df': df_cluster,
            'x_col': 'Cluster',
            'motion_col': motion_lookup.get(conn_type, motion_metric),
            'approach': display_conn,
            'conn_type': str(conn_type).lower(),
        })

    # Add additional row configurations for other connectivity types if present
    for ct in sorted(set(available_types or [])):
        if not cohorts_dir:
            continue
        if str(ct).lower() == str(conn_type).lower():
            continue
        cluster_file = os.path.join(cohorts_dir, f'combined_cohort_F32_global_{ct}_connectivity_clusters.csv')
        if os.path.exists(cluster_file):
            df_ct = pd.read_csv(cluster_file)
            df_ct = _ensure_group_column(df_ct)
        else:
            continue
        df_ct['Cluster'] = df_ct['Cluster'].apply(_normalize_cluster_label)
        motion_col = motion_lookup.get(ct, motion_metric)
        row_configs.append({
            'title': f"By Cluster ({str(ct).capitalize()})",
            'df': df_ct,
            'x_col': 'Cluster',
            'motion_col': motion_col,
            'approach': str(ct).capitalize(),
            'conn_type': str(ct).lower(),
        })

    # Create a figure with one row per configuration and one column per covariate
    n_rows = len(row_configs)
    n_cols = len(covariate_columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)

    # Main plotting loop: compute tests and render base covariates + ICD panels
    print("  Creating multi-row covariate plots...")
    summary_rows = []
    test_rows = []
    for row_idx, cfg in enumerate(row_configs):
        row_axes = axes[row_idx]
        row_conn_type = cfg.get('conn_type') or str(conn_type).lower()
        category_colors = _category_colors_for_row(row_conn_type)
        row_sig, log_df = _compute_covariate_tests(cfg)
        if log_df is not None and not log_df.empty:
            row_slug = re.sub(r'[^0-9A-Za-z]+', '_', str(cfg.get('title', 'row'))).strip('_')
            log_path = os.path.join(out_dir, f"global_covariate_tests_{row_slug}.csv")
            log_df.to_csv(log_path, index=False)
            log_df = log_df.copy()
            log_df['row_title'] = cfg.get('title')
            test_rows.append(log_df)
        # Draw the three base panels (age, sex, motion) and ICD panels for this row
        _plot_base_covariates(row_axes, cfg['df'], cfg['x_col'], cfg['motion_col'], row_sig, category_colors)
        _plot_icd(row_axes, cfg['df'], cfg['x_col'], row_sig, category_colors)

        # Build a small list of covariates present in this configuration to summarize
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

        # For each covariate compute group-wise numeric summaries and collect them
        for cov_name, cov_col in covariate_specs:
            rows = _summarize_covariate_values(df_cfg, x_col, cov_name, cov_col, groups)
            for row in rows:
                row['row_title'] = cfg.get('title')
                row['x_col'] = x_col
            summary_rows.extend(rows)

    # Set column titles for each covariate panel
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

    # Layout and save the combined figure
    fig.subplots_adjust(hspace=0.4 if n_rows > 1 else 0.3)
    if n_rows > 2:
        out_name = 'global_covariates_by_group_clusters_across_types.svg'
    else:
        out_name = f'global_{conn_type}_covariates_by_group_clusters.svg'

    fig.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved covariate distribution plots")

    # If we collected any numeric summaries, merge them with statistical test results
    # and write a human-readable table to disk for downstream inspection.
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if test_rows:
            tests_df = pd.concat(test_rows, ignore_index=True)
            # Align column names for merging: tests use 'covariate' as column name
            tests_df = tests_df.rename(columns={
                'covariate': 'covariate_name',
            })
            # Merge the numeric summaries with raw/FDR p-values from the tests
            summary_df = summary_df.merge(
                tests_df,
                how='left',
                left_on=['row_title', 'covariate'],
                right_on=['row_title', 'covariate_name'],
            )
            summary_df = summary_df.drop(columns=[c for c in ['covariate_name'] if c in summary_df.columns])

        table_path = os.path.join(out_dir, 'global_covariate_distribution_summary.txt')
        # Write the merged table as a plain-text fixed-width table for quick reading
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(summary_df.to_string(index=False))
            f.write("\n")
        print(f"  Saved covariate distribution summary table to: {table_path}")

# ==============================================================================
# HIGH-LEVEL PIPELINE ORCHESTRATION FUNCTIONS
# ==============================================================================
def load_and_prepare_cohort_data(combined_cohort_path, depression_cohort_path, 
                                 head_motion_path, save_if_modified=True):
    """Load cohort data and merge head motion if needed.
    
    Parameters
    ----------
    combined_cohort_path : str
        Path to combined cohort CSV
    depression_cohort_path : str
        Path to depression cohort CSV
    head_motion_path : str
        Path to head motion CSV
    save_if_modified : bool, default=True
        Whether to save cohorts if motion data was merged
    
    Returns
    -------
    dict
        Dictionary with keys:
        - combined_data : pd.DataFrame
        - depression_cohort : pd.DataFrame
        - head_motion : pd.DataFrame
        - depression_subject_ids : list of str
        - control_subject_ids : list of str
    """
    # Read CSV inputs. `combined_data` is expected to contain both control
    # and depressed subjects and a column named `depression_status` used
    # to split cohorts; `depression_cohort` may be a subset prepared for
    # cluster-specific analyses. `head_motion` contains per-subject motion
    # metrics used to augment cohort tables when missing.
    combined_data = pd.read_csv(combined_cohort_path)
    depression_cohort = pd.read_csv(depression_cohort_path)
    head_motion = pd.read_csv(head_motion_path, compression='infer')
    
    # Merge head motion into cohort tables when motion columns are missing.
    # The specific column names used here are project conventions for
    # fMRI/dMRI motion metrics. The helper `find_overlap_individuals`
    # performs a join on subject identifiers and can return a merged
    # DataFrame; when `save_if_modified` is True the updated CSVs are
    # written back to their original paths to persist the augmentation.
    required_cols = {'p24441_i2', 'p24453_i2'}

    if not required_cols.issubset(set(depression_cohort.columns)):
        from cohort_definition.cohort_selection_utils import find_overlap_individuals
        # `return_type='merged'` instructs the helper to merge motion
        # columns into the cohort table rather than just returning an
        # overlap mask. `dropna=False` preserves rows even when some
        # motion values are missing.
        depression_cohort = find_overlap_individuals(
            depression_cohort, head_motion, return_type="merged", dropna=False
        )
        if save_if_modified:
            depression_cohort.to_csv(depression_cohort_path, index=False)
            print("  Saved updated depression cohort with motion data")

    if not required_cols.issubset(set(combined_data.columns)):
        from cohort_definition.cohort_selection_utils import find_overlap_individuals
        combined_data = find_overlap_individuals(
            combined_data, head_motion, return_type="merged", dropna=False
        )
        if save_if_modified:
            combined_data.to_csv(combined_cohort_path, index=False)
            print("  Saved updated combined cohort with motion data")
    
    # Extract and sort `eid` identifiers for depressed and control groups.
    # These ID lists are used to drive per-subject file loading (matrices)
    # and must be string-typed for consistent filesystem joins.
    depression_subject_ids = combined_data[combined_data['depression_status'] == 1]['eid'].astype(str).tolist()
    depression_subject_ids.sort()

    control_subject_ids = combined_data[combined_data['depression_status'] == 0]['eid'].astype(str).tolist()
    control_subject_ids.sort()
    
    return {
        'combined_data': combined_data,
        'depression_cohort': depression_cohort,
        'head_motion': head_motion,
        'depression_subject_ids': depression_subject_ids,
        'control_subject_ids': control_subject_ids,
    }

def perform_clustering_validation(X_mat, Z, clusters, conn_type, out_dir, cluster_stability_bootstrap_iter):
    """Perform comprehensive clustering validation: silhouette, CH, bootstrap.
    
    Parameters
    ----------
    X_mat : np.ndarray
        Feature matrix used for clustering
    Z : np.ndarray
        Linkage matrix
    clusters : np.ndarray
        Cluster labels
    conn_type : str
        Connectivity type ('functional', 'structural', 'sfc')
    out_dir : str
        Output directory
    
    Returns
    -------
    dict
        Dictionary with validation results:
        - sil_df : silhouette scores dataframe
        - ch_df : Calinski-Harabasz scores dataframe
        - stability_results : bootstrap stability results
    """
    print("  Computing silhouette scores...")
    sil_df = compute_silhouette_scores(X_mat, Z)
    sil_df.to_csv(os.path.join(out_dir, f'global_{conn_type}_silhouette_scores.csv'), index=False)
    
    best_sil = sil_df.loc[sil_df['silhouette_score'].idxmax()]
    print(f"    Best k={int(best_sil['k'])}, score={best_sil['silhouette_score']:.4f}")
    
    print("  Computing Calinski-Harabasz scores...")
    ch_df = compute_calinski_harabasz_scores(X_mat, Z)
    ch_df.to_csv(os.path.join(out_dir, f'global_{conn_type}_calinski_harabasz_scores.csv'), index=False)
    
    best_ch = ch_df.loc[ch_df['calinski_harabasz'].idxmax()]
    print(f"    Best k={int(best_ch['k'])}, score={best_ch['calinski_harabasz']:.4f}")
    
    # Combined validation plot
    fig, (ax_sil, ax_ch) = plt.subplots(1, 2, figsize=(12, 4))
    
    valid_sil = sil_df.dropna()
    if not valid_sil.empty:
        ax_sil.plot(valid_sil['k'], valid_sil['silhouette_score'], marker='o')
        ax_sil.set_xlabel('Number of clusters (k)')
        ax_sil.set_ylabel('Silhouette score')
        ax_sil.set_title(f'Silhouette scores ({conn_type})')
        ax_sil.grid(alpha=0.3)
        # Force integer x-tick labels (display k as full integers)
        try:
            xticks = valid_sil['k'].astype(int).tolist()
            ax_sil.set_xticks(xticks)
            ax_sil.set_xticklabels([str(int(x)) for x in xticks])
        except Exception:
            pass
    
    valid_ch = ch_df.dropna()
    if not valid_ch.empty:
        ax_ch.plot(valid_ch['k'], valid_ch['calinski_harabasz'], marker='o')
        ax_ch.set_xlabel('Number of clusters (k)')
        ax_ch.set_ylabel('Calinski-Harabasz score')
        ax_ch.set_title(f'Calinski-Harabasz scores ({conn_type})')
        ax_ch.grid(alpha=0.3)
        # Force integer x-tick labels (display k as full integers)
        try:
            xticks_ch = valid_ch['k'].astype(int).tolist()
            ax_ch.set_xticks(xticks_ch)
            ax_ch.set_xticklabels([str(int(x)) for x in xticks_ch])
        except Exception:
            pass
    
    fig.savefig(os.path.join(out_dir, f'global_{conn_type}_cluster_validation_metrics.svg'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Computing bootstrap stability ({cluster_stability_bootstrap_iter} iterations)...")
    stability_results = bootstrap_clustering_stability(X_mat, clusters, n_boot=cluster_stability_bootstrap_iter)
    
    print(f"    Mean per-subject stability: {np.nanmean(stability_results['per_subject_stability']):.3f}")
    print(f"    Mean NMI: {np.nanmean(stability_results['nmi_list']):.3f}")
    print(f"    Mean Jaccard (Cluster 0): {np.nanmean(stability_results['jaccard_cluster0']):.3f}")
    print(f"    Mean Jaccard (Cluster 1): {np.nanmean(stability_results['jaccard_cluster1']):.3f}")
    
    return {
        'sil_df': sil_df,
        'ch_df': ch_df,
        'stability_results': stability_results,
    }


def save_clustering_results(stability_results, clusters, subject_scalar_depression,
                            conn_type, out_dir):
    """Save clustering validation results to CSV files.
    
    Parameters
    ----------
    stability_results : dict
        Bootstrap stability results
    clusters : np.ndarray
        Cluster labels
    subject_scalar_depression : np.ndarray
        Scalar connectivity summaries
    conn_type : str
        Connectivity type ('functional', 'structural', 'sfc')
    out_dir : str
        Output directory
    
    Returns
    -------
    None
    """
    # Persist clustering summaries: a per-subject CSV with connectivity
    # scalar summary, assigned cluster, and bootstrap-derived stability
    # score. This file is useful for downstream reporting and merging
    # with clinical/demographic tables.
    cluster_df = pd.DataFrame({
        'Connectivity': subject_scalar_depression,
        'Cluster': clusters,
        'cluster_stability': stability_results['per_subject_stability']
    })
    cluster_df.to_csv(os.path.join(out_dir, f'global_{conn_type}_cluster_stability.csv'), index=False)

    # Save bootstrap diagnostics (NMI and Jaccard lists) as a summary
    # table so that iteration-level results can be inspected or plotted
    # independently of the multi-panel SVG created elsewhere.
    boot_diag_df = pd.DataFrame({
        'bootstrap': np.arange(len(stability_results['nmi_list'])),
        'nmi': stability_results['nmi_list'],
        'jaccard_cluster0': stability_results['jaccard_cluster0'],
        'jaccard_cluster1': stability_results['jaccard_cluster1'],
    })
    boot_diag_df.to_csv(os.path.join(out_dir, f'global_{conn_type}_bootstrap_diagnostics.csv'), index=False)

def merge_combined_cohort_connectivity_clusters(
    cohorts_dir,
    conn_types=('functional', 'structural', 'sfc'),
    how='inner',
):
    """Merge per-conn_type combined cohort cluster CSVs into one dataframe.

    This function reads the combined cohort files created by
    `create_combined_dataframe`, renames the connectivity and cluster columns
    per modality, drops the `Connectivity_Type` column, and merges on `eid`.

    Parameters
    ----------
    cohorts_dir : str
        Base directory containing per-conn_type combined cohort CSVs.
    conn_types : tuple of str, default=('functional','structural','sfc')
        Connectivity types to merge.
    how : str, default='inner'
        Merge strategy for `eid` (e.g., 'inner', 'outer').

    Returns
    -------
    pd.DataFrame
        Merged dataframe with one set of shared covariates and
        modality-specific connectivity and cluster columns.
    """
    merged_df = None
    for idx, ct in enumerate(conn_types):
        cluster_file = os.path.join(
            cohorts_dir,
            f'combined_cohort_F32_global_{ct}_connectivity_clusters.csv',
        )
        if not os.path.exists(cluster_file):
            raise FileNotFoundError(f"Missing cluster file: {cluster_file}")

        df = pd.read_csv(cluster_file)
        if 'eid' not in df.columns:
            raise ValueError(f"Missing 'eid' column in {cluster_file}")

        df['eid'] = df['eid'].astype(str)
        if 'Connectivity_Type' in df.columns:
            df = df.drop(columns=['Connectivity_Type'])

        if 'Connectivity' not in df.columns or 'Cluster' not in df.columns:
            raise ValueError(
                f"Expected 'Connectivity' and 'Cluster' in {cluster_file}"
            )

        df = df.rename(
            columns={
                'Connectivity': f'{ct}_connectivity',
                'Cluster': f'{ct}_cluster',
            }
        )

        if idx == 0:
            merged_df = df
        else:
            df = df[['eid', f'{ct}_connectivity', f'{ct}_cluster']].copy()
            merged_df = merged_df.merge(df, on='eid', how=how)
    if merged_df is None:
        return merged_df

    preferred_cols = ['eid']
    for ct in conn_types:
        preferred_cols.append(f'{ct}_connectivity')
        preferred_cols.append(f'{ct}_cluster')

    existing_preferred = [c for c in preferred_cols if c in merged_df.columns]
    remaining_cols = [c for c in merged_df.columns if c not in existing_preferred]
    merged_df = merged_df.loc[:, existing_preferred + remaining_cols]

    return merged_df

def create_combined_dataframe(control_subject_ids, depression_subject_ids,
                              subject_scalar_control, subject_scalar_depression,
                              clusters, combined_data, conn_type=None):
    """Create combined dataframe with connectivity, cluster labels, and demographics.
    
    Parameters
    ----------
    control_subject_ids : list of str
        Control subject IDs
    depression_subject_ids : list of str
        Depression subject IDs
    subject_scalar_control : np.ndarray
        Scalar connectivity for controls
    subject_scalar_depression : np.ndarray
        Scalar connectivity for depression
    clusters : np.ndarray
        Cluster labels
    combined_data : pd.DataFrame
        Combined cohort data with demographics
    conn_type : str, optional
        Connectivity type ('functional', 'structural', or Structure-Function Coupling)
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all necessary columns
    """
    # Construct the combined per-subject table used for downstream
    # analyses and plotting. We include `eid`, a scalar connectivity
    # summary (`Connectivity`), the text `Cluster` label for plotting/R,
    # and a `Group` column ('Control'/'Depression'). The ordering of rows
    # matches the concatenation of control then depressed subjects which
    # aligns with how scalar arrays are supplied to this function.
    combined_df = pd.DataFrame({
        'eid': np.concatenate([control_subject_ids, depression_subject_ids]),
        'Connectivity': np.concatenate([subject_scalar_control, subject_scalar_depression]),
        'Cluster': np.concatenate([
            ['Control'] * len(subject_scalar_control),
            [f'Cluster {c}' for c in clusters],
        ]),
        'Group': ['Control'] * len(subject_scalar_control) + ['Depression'] * len(subject_scalar_depression),
    })
    
    # Add connectivity type if provided
    if conn_type is not None:
        combined_df['Connectivity_Type'] = conn_type
    
    combined_df['eid'] = combined_df['eid'].astype(str)
    combined_data['eid'] = combined_data['eid'].astype(str)
    combined_df = combined_df.merge(combined_data, on='eid', how='inner')
    
    return combined_df
