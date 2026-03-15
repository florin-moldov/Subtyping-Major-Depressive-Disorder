"""
Utility functions for per-edge covariate association analysis.

Provides helpers for loading connectivity matrices, building edge-feature
matrices, aligning covariate vectors, running per-edge correlation tests
with FDR correction, and visualizing results via Manhattan-style plots.

Main pipeline entry point: run_per_edge_associations().

Architecture
------------
The script is organized into functional sections:

- **Utilities**

- **Data loading and preprocessing**

- **Correlation analysis**

- **Visualization**

- **High-level pipeline**
"""

import os
from typing import Iterable, Optional
import pandas as pd
import numpy as np
import scipy.stats as sp_stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# ==============================================================================
# UTILITIES
# ==============================================================================
def get_motion_columns(conn_type, fMRI_MOTION_METRIC, dMRI_MOTION_METRIC):
    """Return a label-to-column mapping for the head motion covariate of a connectivity modality.

    Functional connectivity uses the fMRI motion metric and structural connectivity
    uses the dMRI motion metric. Any unrecognized connectivity type (e.g., 'sfc')
    falls back to a single 'motion' key pointing at the dMRI column.

    Parameters
    ----------
    conn_type : str
        Connectivity modality. Supported values: 'functional', 'structural'.
        Any other value falls back to a generic 'motion' key.
    fMRI_MOTION_METRIC : str
        DataFrame column name for the fMRI head motion metric (e.g., 'p24441_i2').
    dMRI_MOTION_METRIC : str
        DataFrame column name for the dMRI head motion metric (e.g., 'p24453_i2').

    Returns
    -------
    dict[str, str]
        Mapping of motion covariate label to DataFrame column name.
        Examples:
          - 'functional' -> {'motion_fmri': 'p24441_i2'}
          - 'structural' -> {'motion_dmri': 'p24453_i2'}
          - other       -> {'motion': 'p24453_i2'} (fallback)
    """

    # Map each connectivity type to its motion metric label and column name
    MOTION_METRICS = {
        'functional': {'motion_fmri': fMRI_MOTION_METRIC},
        'structural': {'motion_dmri': dMRI_MOTION_METRIC},
    }
    # Look up a modality-specific mapping (if missing, use a generic fallback)
    mapping = MOTION_METRICS.get(conn_type)
    if mapping is None:
        return {'motion': 'p24453_i2'}
    return mapping

def _stack_matrices(mats):
    """Coerce a connectivity matrix collection to a NumPy array.

    Handles three input types without copying when possible:
    - np.memmap: returned as-is.
    - list of 2-D np.ndarray: stacked along a new axis 0 -> shape (n, rows, cols).
    - any other type: passed through np.asarray.

    Parameters
    ----------
    mats : np.memmap, list of np.ndarray, or array-like
        Collection of connectivity matrices or a pre-stacked array.

    Returns
    -------
    np.ndarray or np.memmap
        3-D array of shape (n_subjects, n_nodes, n_nodes) when input is a list
        of 2-D arrays; otherwise the input converted or returned unchanged.
    """
    # Preserve memmap arrays to avoid loading into memory
    if isinstance(mats, np.memmap):
        return mats
    if isinstance(mats, list):
        # If list is empty, return an empty array
        if len(mats) == 0:
            return np.asarray(mats)
        # If list contains 2-D matrices, stack into 3-D
        if isinstance(mats[0], np.ndarray) and mats[0].ndim == 2:
            return np.stack(mats, axis=0)
    # Fallback: coerce to a NumPy array
    return np.asarray(mats)

def vectorize_connectivity_matrices(mats, tri_indices=None, k=1):
    """Vectorize symmetric connectivity matrices into upper-triangle edge vectors.

    Extracts the upper triangle (excluding or including the diagonal depending on
    `k`) from each matrix in the stack, producing one row per subject.

    Parameters
    ----------
    mats : array-like or list of np.ndarray
        Connectivity matrices. Accepted forms:
        - 3-D np.ndarray of shape (n_subjects, n_nodes, n_nodes).
        - list of 2-D np.ndarray, each of shape (n_nodes, n_nodes).
        - np.memmap of shape (n_subjects, n_nodes, n_nodes).
    tri_indices : tuple of np.ndarray, optional
        Pre-computed upper-triangle indices from np.triu_indices. If None,
        indices are computed from the node dimension of `mats` using `k`.
    k : int, default=1
        Diagonal offset passed to np.triu_indices. k=1 excludes the main
        diagonal (self-connections); k=0 includes it.

    Returns
    -------
    X_vec : np.ndarray, shape (n_subjects, n_edges)
        Vectorized edge features; each row is one subject's upper-triangle
        values flattened in row-major order.
    tri_indices : tuple of np.ndarray
        Pair (row_indices, col_indices) of the upper triangle used; can be
        passed back in subsequent calls for consistency.

    Raises
    ------
    ValueError
        If the stacked array is not 3-D after coercion.
    """
    # Coerce input to a 3-D array stack (subjects x nodes x nodes)
    mats_arr = _stack_matrices(mats)
    if mats_arr.ndim != 3:
        raise ValueError(f"Expected 3D matrix stack, got shape {mats_arr.shape}")
    # Infer node count from the second dimension
    n_nodes = mats_arr.shape[1]
    # Use provided triangle indices or compute a new upper triangle mask
    if tri_indices is None:
        tri_indices = np.triu_indices(n_nodes, k=k)
    # Allocate output matrix (subjects x edges)
    X_vec = np.empty((mats_arr.shape[0], len(tri_indices[0])), dtype=float)
    # Fill each row with upper-triangle values from the subject matrix
    for i in range(mats_arr.shape[0]):
        X_vec[i, :] = mats_arr[i][tri_indices]
    return X_vec, tri_indices

def describe_significant_edges(corr_map_df):
    """Summarize the significant edges from a per-edge correlation map.

    Counts edges flagged by FDR correction and reports the range and median
    of their correlation coefficients.

    Parameters
    ----------
    corr_map_df : pd.DataFrame
        Output of per_edge_correlation_map. Must contain columns:
        - 'reject_fdr' (bool): True for FDR-significant edges.
        - 'r' (float): correlation coefficient per edge.

    Returns
    -------
    count : int
        Number of edges where reject_fdr is True.
    percentage : float
        Percentage of total edges that are FDR-significant (0–100).
    rmin : float
        Minimum r among FDR-significant edges.
    rmax : float
        Maximum r among FDR-significant edges.
    median_r : float
        Median r among FDR-significant edges.

    Raises
    ------
    ValueError
        If 'reject_fdr' column is absent from corr_map_df.
    """
    # Validate that the FDR flag exists in the input
    if 'reject_fdr' not in corr_map_df.columns:
        raise ValueError("corr_map_df must contain 'reject_fdr' column")
    else:
        # Count significant edges and compute summary stats on r values
        count = int(corr_map_df['reject_fdr'].sum())
        percentage = count / corr_map_df.shape[0] if corr_map_df.shape[0] > 0 else 0
        percentage *= 100
        range_sig = corr_map_df.loc[corr_map_df['reject_fdr'] == True, 'r']
        if not range_sig.empty:
            rmin = float(np.min(range_sig.to_numpy(dtype=float)))
            rmax = float(np.max(range_sig.to_numpy(dtype=float)))
            median_r = float(np.median(range_sig.to_numpy(dtype=float)))
    return count, percentage, rmin, rmax, median_r

def encode_sex_to_binary(sex_values):
    """Encode sex labels or numeric codes to a binary 0/1 float array.

    Handles mixed inputs: string labels, single-byte codes, and numeric codings
    (both 0/1 and 1/2 conventions). Unrecognized or missing values become NaN.

    Parameters
    ----------
    sex_values : array-like
        Array of sex values. Accepted encodings:
        - Strings: 'Male'/'Female', 'M'/'F' (case-insensitive).
        - Numeric 0/1: 0 treated as male, 1 as female.
        - Numeric 1/2: 1 treated as male, 2 as female.
        - None or NaN: mapped to NaN.

    Returns
    -------
    np.ndarray of float, shape (n_subjects,)
        Binary-coded sex: 0.0 = male, 1.0 = female, NaN = unknown/missing.

    Notes
    -----
    Coding convention: 0 = male, 1 = female. In point-biserial correlation
    this means a negative r indicates males have higher values and a positive
    r indicates females have higher values.
    """
    # Normalize input to an array and pre-allocate output
    arr = np.asarray(sex_values)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    
    for i, v in enumerate(arr):
        # Skip missing values
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        
        # Handle string-coded sex values
        if isinstance(v, (str, bytes)):
            s = v.decode() if isinstance(v, bytes) else v
            s = s.strip().lower()
            if s in ('male', 'm'):
                out[i] = 0.0
            elif s in ('female', 'f'):
                out[i] = 1.0
        else:
            # Handle numeric codings (0/1 or 1/2)
            try:
                fv = float(v)
                if np.isfinite(fv) and fv >= 0:
                    if fv in (0.0, 1.0):
                        out[i] = fv
                    elif fv in (1.0, 2.0):
                        out[i] = 1.0 if fv == 2.0 else 0.0
            except Exception:
                continue
    
    return out

def align_motion_vector(combined_df, subject_ids, motion_col):
    """Extract and align a head motion covariate vector to a feature matrix row order.

    Looks up `motion_col` in `combined_df` by subject ID ('eid') and returns
    values in the exact order of `subject_ids`. Subjects absent from the
    DataFrame receive NaN, preserving alignment with the feature matrix.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Cohort DataFrame containing at least columns 'eid' and `motion_col`.
        'eid' values are cast to str before matching.
    subject_ids : list of str
        Ordered subject IDs corresponding to rows of the feature matrix.
    motion_col : str
        Name of the head motion column to extract
        (e.g., 'p24441_i2' for fMRI, 'p24453_i2' for dMRI).

    Returns
    -------
    np.ndarray of float, shape (n_subjects,)
        Motion values aligned to `subject_ids` order.
        Entries for subjects not found in `combined_df` are NaN.
    """
    # Keep only the ID and motion column to minimize memory
    tmp = combined_df.loc[:, ['eid', motion_col]].copy()
    tmp['eid'] = tmp['eid'].astype(str)
    # Use eid as index so reindex preserves subject ordering
    tmp = tmp.set_index('eid')
    # Reindex to subject_ids; missing IDs become NaN
    return tmp.reindex([str(x) for x in subject_ids])[motion_col].to_numpy(dtype=float)


def align_scalar_vector(combined_df, subject_ids, col, dtype=float):
    """Extract and align an arbitrary scalar covariate to a feature matrix row order.

    General-purpose alternative to align_motion_vector that supports any
    covariate column and an optional dtype cast. Use dtype=None to retain
    object/string arrays (e.g., for categorical sex values before encoding).

    Parameters
    ----------
    combined_df : pd.DataFrame
        Cohort DataFrame containing at least columns 'eid' and `col`.
        'eid' values are cast to str before matching.
    subject_ids : list of str
        Ordered subject IDs corresponding to rows of the feature matrix.
    col : str
        Name of the covariate column to extract
        (e.g., 'p21003_i2' for age, 'p31' for sex).
    dtype : type or None, default=float
        NumPy dtype for the output array. Pass None to keep the original
        object dtype (useful for string-valued columns such as sex).

    Returns
    -------
    np.ndarray, shape (n_subjects,)
        Covariate values aligned to `subject_ids` order, cast to `dtype`.
        Entries for subjects not found in `combined_df` are NaN (float)
        or None (object dtype).
    """
    # Extract only the ID and requested covariate column
    tmp = combined_df.loc[:, ['eid', col]].copy()
    tmp['eid'] = tmp['eid'].astype(str)
    # Use eid as index to align by subject IDs
    tmp = tmp.set_index('eid')
    s = tmp.reindex([str(x) for x in subject_ids])[col]
    # Preserve object dtype if requested (e.g., for categorical values)
    if dtype is None:
        return s.to_numpy()
    return s.to_numpy(dtype=dtype)

def load_single_connectivity_matrix(subject_id, cohort_dir, conn_type='functional'):
    """Load and return a single subject's connectivity matrix from disk.

    Looks for files under `<cohort_dir>/<subject_id>/i2/`. Functional matrices
    are stored as `.npy` files; structural matrices as gzip-compressed CSV files.
    The main diagonal is zeroed out in both cases to remove self-connections.

    Parameters
    ----------
    subject_id : str
        Subject identifier used to construct the directory and filename path.
    cohort_dir : str
        Root directory containing one sub-directory per subject.
    conn_type : str, default='functional'
        Connectivity type to load. Supported values:
        - 'functional': loads `<subject_id>_connectivity.npy`.
        - 'structural': loads `connectome_streamline_count_10M.csv.gz`.

    Returns
    -------
    np.ndarray of float, shape (n_nodes, n_nodes), or None
        Symmetric connectivity matrix with zeroed diagonal.
        Returns None if the expected file is not found.

    Raises
    ------
    ValueError
        If `conn_type` is not 'functional' or 'structural'.
    """
    # Build the subject-specific directory path
    subject_dir = os.path.join(cohort_dir, subject_id, 'i2')
    if conn_type == 'functional':
        # Functional connectivity stored as NumPy array
        file_path = os.path.join(subject_dir, f'{subject_id}_connectivity.npy')
        if os.path.isfile(file_path):
            data = np.load(file_path)
            # Remove self-connections
            np.fill_diagonal(data, 0)
            return data.astype(float)
        print(f"Warning: {file_path} not found for subject {subject_id}")
        return None

    if conn_type == 'structural':
        # Structural connectivity stored as gzip-compressed CSV
        file_path = os.path.join(subject_dir, 'connectome_streamline_count_10M.csv.gz')
        if os.path.isfile(file_path):
            data = pd.read_csv(file_path, compression='infer', header=None).to_numpy()
            # Remove self-connections
            np.fill_diagonal(data, 0)
            return data.astype(float)
        print(f"Warning: {file_path} not found for subject {subject_id}")
        return None

    raise ValueError("conn_type must be 'functional' or 'structural' for matrix loading")


def build_edge_feature_matrix_from_connectivity(
    subject_ids: Iterable[str],
    cohort_dir: str,
    conn_type: str,
    batch_size: int = 50,
    cache_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    dtype: np.dtype = np.float32,
) -> tuple:
    """Build a subjects-by-edges feature matrix from raw connectivity matrices.

    Iterates over `subject_ids`, filters to those with a file on disk, then
    fills a memory-mapped (disk-backed) array with upper-triangle edge values.
    Processing is done in batches to limit peak RAM usage. Rows for subjects
    whose matrix fails to load or has an unexpected shape remain NaN.

    The memmap file is written to `cache_dir` and reused within the same run;
    a new file is created each call (mode='w+'), overwriting any prior file
    with the same name.

    Parameters
    ----------
    subject_ids : Iterable[str]
        Subject identifiers to include. Converted to a list internally.
    cohort_dir : str
        Root directory containing one sub-directory per subject
        (see load_single_connectivity_matrix for expected structure).
    conn_type : str
        Connectivity type: 'functional' or 'structural'.
    batch_size : int, default=50
        Number of subjects to process per batch. Lower values reduce peak
        memory at the cost of more I/O round-trips.
    cache_dir : str, optional
        Directory for the memmap backing file. Defaults to '/tmp/edge_cache'.
        Created automatically if it does not exist.
    prefix : str, optional
        Filename prefix for the memmap file.
        Defaults to '<conn_type>_edges'.
    dtype : np.dtype, default=np.float32
        NumPy dtype for the memmap array. float32 halves storage compared to
        float64 at the cost of reduced precision.

    Returns
    -------
    X_edges : np.memmap, shape (n_kept_subjects, n_edges), or None
        Disk-backed edge-feature matrix. n_edges = n_nodes*(n_nodes-1)/2.
        Returns None if no valid subjects are found.
    n_nodes : int, or None
        Number of nodes inferred from the first successfully loaded matrix.
    kept_ids : list of str
        Subject IDs whose connectivity file was found on disk (not necessarily
        all successfully loaded; failed rows remain NaN).
        Returns empty list if no valid subjects are found.

    Notes
    -----
    The returned tuple has three elements (X_edges, n_nodes, kept_ids).
    `tri_indices` is computed internally but not returned; reconstruct with
    np.triu_indices(n_nodes, k=1) when needed downstream.
    """
    # Ensure subject IDs are concrete and iterable multiple times
    subject_ids = list(subject_ids)
    kept_ids = []
    for sid in subject_ids:
        # Only keep subjects with an existing connectivity file
        subject_dir = os.path.join(cohort_dir, sid, 'i2')
        if conn_type == 'functional':
            file_path = os.path.join(subject_dir, f"{sid}_connectivity.npy")
        else:
            file_path = os.path.join(subject_dir, 'connectome_streamline_count_10M.csv.gz')
        if os.path.isfile(file_path):
            kept_ids.append(sid)

    if not kept_ids:
        return None, None, None, []

    # Probe the first available matrix to infer shape
    first_mat = None
    for sid in kept_ids:
        first_mat = load_single_connectivity_matrix(sid, cohort_dir, conn_type)
        if first_mat is not None:
            break
    if first_mat is None:
        return None, None, None, []

    # Determine number of nodes and upper-triangle indices
    n_nodes = first_mat.shape[0]
    tri_indices = np.triu_indices(n_nodes, k=1)
    n_edges = len(tri_indices[0])

    # Default to a temporary cache directory for the memmap file
    if cache_dir is None:
        cache_dir = os.path.join('/tmp', 'edge_cache')
    os.makedirs(cache_dir, exist_ok=True)
    if prefix is None:
        prefix = f"{conn_type}_edges"

    # Create the memmap and initialize with NaNs
    mmap_path = os.path.join(cache_dir, f"{prefix}_{len(kept_ids)}x{n_edges}.dat")
    X_edges = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=(len(kept_ids), n_edges))
    X_edges[:] = np.nan

    # Fill the memmap in batches to limit peak memory usage
    for start in range(0, len(kept_ids), batch_size):
        end = min(start + batch_size, len(kept_ids))
        batch_ids = kept_ids[start:end]
        for offset, sid in enumerate(batch_ids):
            mat = load_single_connectivity_matrix(sid, cohort_dir, conn_type)
            if mat is None:
                continue
            if mat.shape[0] != n_nodes or mat.shape[1] != n_nodes:
                print(f"Warning: {sid} matrix shape {mat.shape} != ({n_nodes}, {n_nodes})")
                continue
            # Vectorize upper triangle into the feature row
            X_edges[start + offset, :] = mat[tri_indices]

    return X_edges, n_nodes, kept_ids

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_and_prepare_cohort_data(combined_cohort_path, depression_cohort_path, 
                                 head_motion_path, save_if_modified=True):
    """Load cohort CSVs and ensure head motion columns are present.

    Reads the combined cohort, the depression-only cohort, and the head motion
    file. If either cohort DataFrame is missing the required motion columns
    ('p24441_i2', 'p24453_i2'), the head motion DataFrame is merged in using
    find_overlap_individuals. The updated DataFrames are optionally saved back
    to disk.

    Subject ID lists for depression (depression_status==1) and control
    (depression_status==0) groups are extracted from the combined cohort,
    cast to str, and sorted.

    Parameters
    ----------
    combined_cohort_path : str
        Path to the combined cohort CSV (controls + depressed subjects).
    depression_cohort_path : str
        Path to the depression-only cohort CSV.
    head_motion_path : str
        Path to the head motion CSV (gzip-compressed or plain).
        Must contain 'eid', 'p24441_i2' (fMRI motion), 'p24453_i2' (dMRI motion).
    save_if_modified : bool, default=True
        If True, overwrite cohort CSV files when motion columns were merged.

    Returns
    -------
    dict with keys:
        combined_data : pd.DataFrame
            Full combined cohort with motion columns guaranteed present.
        depression_cohort : pd.DataFrame
            Depression cohort with motion columns guaranteed present.
        head_motion : pd.DataFrame
            Raw head motion DataFrame as loaded from disk.
        depression_subject_ids : list of str
            Sorted subject IDs with depression_status == 1.
        control_subject_ids : list of str
            Sorted subject IDs with depression_status == 0.
    """
    # Load primary cohort tables and head motion file
    combined_data = pd.read_csv(combined_cohort_path)
    depression_cohort = pd.read_csv(depression_cohort_path)
    head_motion = pd.read_csv(head_motion_path, compression='infer')
    
    # Merge head motion if needed
    # Motion columns required for downstream analysis
    required_cols = {'p24441_i2', 'p24453_i2'}
    
    if not required_cols.issubset(set(depression_cohort.columns)):
        from cohort_definition.cohort_selection_utils import find_overlap_individuals
        # Merge head motion into the depression cohort
        depression_cohort = find_overlap_individuals(
            depression_cohort, head_motion, return_type="merged", dropna=False
        )
        if save_if_modified:
            depression_cohort.to_csv(depression_cohort_path, index=False)
            print("  Saved updated depression cohort with motion data")
    
    if not required_cols.issubset(set(combined_data.columns)):
        from cohort_definition.cohort_selection_utils import find_overlap_individuals
        # Merge head motion into the combined cohort
        combined_data = find_overlap_individuals(
            combined_data, head_motion, return_type="merged", dropna=False
        )
        if save_if_modified:
            combined_data.to_csv(combined_cohort_path, index=False)
            print("  Saved updated combined cohort with motion data")
    
    # Extract subject IDs
    # Extract and sort subject IDs for each group
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

# ==============================================================================
# CORRELATION ANALYSIS
# ==============================================================================
def per_edge_correlation_map(X_subject_by_node, y_scalar, method='spearman', 
                             fdr_alpha=0.05):
    """Compute a per-edge correlation map between a feature matrix and a scalar covariate.

    For each column (edge) in X, computes the correlation with y and tests for
    significance. FDR correction (Benjamini-Hochberg) is applied across all edges
    jointly. Edges with fewer than 3 valid observations are skipped (r=NaN, p=NaN).

    Parameters
    ----------
    X_subject_by_node : np.ndarray, shape (n_subjects, n_edges)
        Feature matrix where rows are subjects and columns are edges (or nodes).
        NaN values are excluded per-edge before computing correlations.
    y_scalar : np.ndarray, shape (n_subjects,)
        Scalar covariate aligned to X rows. Accepted types depend on `method`:
        - 'pearson' / 'spearman': numeric float array; NaN entries are excluded.
        - 'pointbiserial': binary (0/1), numeric 1/2, or string
          ('Male'/'Female', 'M'/'F') values; mapped to 0/1 internally.
    method : str, default='spearman'
        Correlation method to apply per edge. One of:
        - 'pearson': Pearson product-moment correlation.
        - 'spearman': Spearman rank-order correlation (robust to outliers).
        - 'pointbiserial': Point-biserial correlation for binary y.
    fdr_alpha : float, default=0.05
        Alpha level for Benjamini-Hochberg FDR correction.

    Returns
    -------
    pd.DataFrame, shape (n_edges, 6)
        One row per edge/column of X, with columns:
        - 'edge' (int): column index in X.
        - 'n_used' (int): number of subjects with valid (non-NaN) data for this edge.
        - 'r' (float): correlation coefficient; NaN if fewer than 3 valid subjects.
        - 'p' (float): uncorrected two-sided p-value; NaN if fewer than 3 valid subjects.
        - 'q' (float): FDR-BH corrected q-value; NaN for edges without a valid p.
        - 'reject_fdr' (bool): True if q < fdr_alpha.

    Raises
    ------
    ValueError
        If `method` is not one of 'pearson', 'spearman', 'pointbiserial'.
        If X is not 2-D.
        If y is not 1-D or has length != n_subjects.
        If pointbiserial y does not resolve to exactly two binary categories.

    Notes
    -----
    For 'pointbiserial', a negative r means the group coded 0 (male) has higher
    feature values; a positive r means the group coded 1 (female) has higher values.
    """
    # Normalize inputs to NumPy arrays for vectorized operations
    Xn = np.asarray(X_subject_by_node)
    y_raw = np.asarray(y_scalar)
    
    # Choose representation for y depending on correlation type
    if method in ('pearson', 'spearman'):
        y = np.asarray(y_raw, dtype=float)
    elif method == 'pointbiserial':
        y = y_raw
    else:
        raise ValueError("method must be 'pearson', 'spearman', or 'pointbiserial'")
    
    # Validate expected input dimensions
    if Xn.ndim != 2:
        raise ValueError(f"X_subject_by_node must be 2D, got shape {Xn.shape}")
    if y.ndim != 1 or y.shape[0] != Xn.shape[0]:
        raise ValueError(f"y_scalar must be 1D of length {Xn.shape[0]}, got shape {y.shape}")
    
    # Prepare binary-coded y for point-biserial
    y_for_pointbiserial = None
    # Prepare binary-coded y for point-biserial correlation
    if method == 'pointbiserial':
        y_num = np.full(y.shape[0], np.nan, dtype=float)
        unknown_tokens = set()
        
        for i, v in enumerate(y):
            # Handle missing values explicitly
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            
            # Parse string values for sex-like encodings
            if isinstance(v, (str, bytes)):
                s = v.decode() if isinstance(v, bytes) else v
                s = s.strip().lower()
                if s in ('male', 'm'):
                    y_num[i] = 0.0
                elif s in ('female', 'f'):
                    y_num[i] = 1.0
                elif s in ('', 'nan'):
                    continue
                else:
                    unknown_tokens.add(s)
            else:
                # Attempt to coerce numeric values
                try:
                    y_num[i] = float(v)
                except Exception:
                    unknown_tokens.add(str(v))
        
        # Fail early when categorical values are unrecognized
        if unknown_tokens:
            raise ValueError(
                f"pointbiserial: unrecognized category values: {sorted(list(unknown_tokens))[:10]}"
            )
        
        y_for_pointbiserial = np.full_like(y_num, np.nan, dtype=float)
        nonneg = np.isfinite(y_num) & (y_num >= 0)
        u = np.unique(y_num[nonneg])
        u_set = set([float(v) for v in u.tolist()])
        
        # Reduce multi-valued y to a binary encoding when possible
        if len(u_set) > 2:
            if {0.0, 1.0}.issubset(u_set):
                keep = np.isfinite(y_num) & ((y_num == 0) | (y_num == 1))
                y_for_pointbiserial[keep] = y_num[keep]
            elif {1.0, 2.0}.issubset(u_set):
                keep = np.isfinite(y_num) & ((y_num == 1) | (y_num == 2))
                y_for_pointbiserial[keep] = y_num[keep]
            else:
                raise ValueError(
                    f"pointbiserial requires binary y; found: {sorted(u_set)}"
                )
        else:
            y_for_pointbiserial[nonneg] = y_num[nonneg]
        
        # Ensure exactly two categories exist before mapping to 0/1
        u = np.unique(y_for_pointbiserial[np.isfinite(y_for_pointbiserial)])
        if u.size != 2:
            raise ValueError(
                f"pointbiserial requires exactly 2 categories; found: {u.tolist()}"
            )
        
        # Map to 0/1, preserving NaN entries
        u = np.sort(u)
        mapped = np.full_like(y_for_pointbiserial, np.nan, dtype=float)
        keep = np.isfinite(y_for_pointbiserial)
        mapped[keep] = (y_for_pointbiserial[keep] == u[1]).astype(float)
        y_for_pointbiserial = mapped
    
    # Allocate output arrays per edge
    n_nodes = Xn.shape[1]
    r_vals = np.full(n_nodes, np.nan, dtype=float)
    p_vals = np.full(n_nodes, np.nan, dtype=float)
    n_used = np.zeros(n_nodes, dtype=int)
    
    for j in range(n_nodes):
        # Extract the edge vector and mask invalid entries
        xj = Xn[:, j].astype(float)
        if method == 'pointbiserial':
            mask = np.isfinite(xj) & np.isfinite(y_for_pointbiserial)
        else:
            mask = np.isfinite(xj) & np.isfinite(y)
        
        n_used[j] = int(mask.sum())
        # Require at least 3 valid observations to compute correlation
        if n_used[j] < 3:
            continue
        
        # Choose correlation method for this edge
        if method == 'pearson':
            r, p = sp_stats.pearsonr(xj[mask], y[mask])
        elif method == 'spearman':
            r, p = sp_stats.spearmanr(xj[mask], y[mask])
        elif method == 'pointbiserial':
            yb = y_for_pointbiserial[mask]
            if np.unique(yb).size != 2:
                continue
            r, p = sp_stats.pointbiserialr(xj[mask], yb)
        
        r_vals[j] = float(r)
        p_vals[j] = float(p)
    
    # FDR correction across all edges
    q_vals = np.full_like(p_vals, np.nan)
    reject = np.zeros(n_nodes, dtype=bool)
    valid = np.isfinite(p_vals)
    
    if valid.any():
        rej, q, _, _ = multipletests(p_vals[valid], alpha=fdr_alpha, method='fdr_bh')
        q_vals[valid] = q
        reject[valid] = rej

    
    return pd.DataFrame({
        'edge': np.arange(n_nodes, dtype=int),
        'n_used': n_used,
        'r': r_vals,
        'p': p_vals,
        'q': q_vals,
        'reject_fdr': reject,
    })

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_manhattan_style_association(dep_map, ctrl_map, conn_type, out_dir,
                                     scalar_label, connectivity_metric, out_filename, alpha_uncorr=0.05,
                                     rasterize=True, dpi=150):
    """Generate and save Manhattan-style association plots for depression and control cohorts.

    Creates two separate figures (one per cohort) plotting –log10(p) for each
    edge against its feature index, coloured by correlation sign. Reference
    lines mark the uncorrected alpha threshold and the FDR significance
    threshold (inferred from the maximum p among FDR-significant edges).
    FDR-significant edges are highlighted with an outlined marker.

    Parameters
    ----------
    dep_map : pd.DataFrame
        Per-edge correlation map for the depression cohort; output of
        per_edge_correlation_map. Required columns: 'p', 'r', 'reject_fdr'.
    ctrl_map : pd.DataFrame
        Per-edge correlation map for the control cohort; same structure as dep_map.
    conn_type : str
        Connectivity type label used in the figure title
        ('functional', 'structural', or 'sfc'). 'sfc' is displayed as
        'Structure-Function Coupling'.
    out_dir : str
        Directory where the two output figures will be saved.
    scalar_label : str
        Human-readable label for the covariate (e.g., 'motion_fmri', 'age', 'sex').
        Used in the figure title.
    connectivity_metric : str
        Label for the connectivity measure plotted (e.g., 'edge weight').
        Used in the figure title.
    out_filename : str
        Base filename for output SVG files (extension included).
        Saved as '<base>_depression.<ext>' and '<base>_control.<ext>'.
    alpha_uncorr : float, default=0.05
        Uncorrected p-value threshold shown as a green dashed reference line.
    rasterize : bool, default=True
        If True, scatter points are rasterized to reduce file size for SVG output.
    dpi : int, default=150
        Resolution for saved figures.

    Returns
    -------
    None
        Saves two figures to out_dir and prints their paths.

    Notes
    -----
    - Points are coloured red for positive r and blue for negative r.
    - FDR-significant edges are additionally outlined in their sign colour.
    - If no edges are FDR-significant, a text annotation replaces the FDR line.
    - The correlation range and median for FDR-significant edges are annotated
      in the top-left corner of each figure.
    """
    # Helper: compute the p-value threshold for FDR-significant points
    def _p_fdr_star(df_map):
        if ('reject_fdr' not in df_map.columns) or ('p' not in df_map.columns):
            return np.nan
        sig = df_map.loc[df_map['reject_fdr'] == True, 'p']
        if sig.empty:
            return np.nan
        return float(np.nanmax(sig.to_numpy(dtype=float)))
    
    # Helper: convert p-values to -log10 scale with clipping for stability
    def _manhattan_y(p):
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-300, 1.0)
        return -np.log10(p)
    
    # Make SFC display label more descriptive
    display_metric = connectivity_metric

    # Helper: plot a single cohort map
    def _plot_single(m, label):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        node = m['node'].to_numpy(dtype=int) if 'node' in m.columns else np.arange(m.shape[0])
        p = m['p'].to_numpy(dtype=float)
        r = m['r'].to_numpy(dtype=float) if 'r' in m.columns else np.zeros_like(p)
        
        # Sort by node index for a cleaner Manhattan plot
        order = np.argsort(node)
        node = node[order]
        p = p[order]
        r = r[order]
        y = _manhattan_y(p)
        
        # Color edges by sign of correlation
        colors = np.where(r >= 0, 'tab:red', 'tab:blue')
        ax.scatter(node, y, s=8, c=colors, alpha=0.25,
                   linewidths=0, rasterized=rasterize)
        
        # Uncorrected significance threshold reference line
        ax.axhline(_manhattan_y(alpha_uncorr), color='tab:green', ls='--', lw=1.2,
                   label=f'uncorr p={alpha_uncorr:g}')
        
        # FDR threshold line if any edges are significant
        p_star = _p_fdr_star(m)
        if np.isfinite(p_star):
            ax.axhline(_manhattan_y(p_star), color='black', ls='-', lw=1.2,
                       label=f'FDR p*={p_star:.2e}')
        else:
            ax.text(0.01, 0.92, 'No FDR-significant nodes', 
                   transform=ax.transAxes, ha='left', va='top', fontsize=9)
        
        # Highlight FDR-significant edges with outlined markers
        if 'reject_fdr' in m.columns:
            sig = m['reject_fdr'].to_numpy(dtype=bool)[order]
            if sig.any():
                ax.scatter(node[sig], y[sig], s=18, facecolors='none',
                          edgecolors=colors[sig], linewidths=0.9, alpha=0.9,
                          label='FDR-significant', rasterized=rasterize)
                
                # Annotate range/median for significant correlations
                r_sig = r[sig]
                r_sig = r_sig[np.isfinite(r_sig)]
                if r_sig.size > 0:
                    rmin, rmax, rmed = float(np.min(r_sig)), float(np.max(r_sig)), float(np.median(r_sig))
                    ax.text(0.01, 0.86, f'FDR-sig r: [{rmin:.3f}, {rmax:.3f}]\nmedian: {rmed:.3f}',
                           transform=ax.transAxes, ha='left', va='top', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{label}: {scalar_label} vs {display_metric}')
        ax.set_ylabel(r'$-\log_{10}(p)$')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=9, frameon=True)
        ax.set_xlabel('Feature index')
        ax.set_xlim([0, m.shape[0]])
        fig.tight_layout()
        return fig

    # Save separate plots for depression and control
    base_name = os.path.splitext(out_filename)[0]
    dep_fig = _plot_single(dep_map, 'Depression')
    ctrl_fig = _plot_single(ctrl_map, 'Control')

    dep_path = os.path.join(out_dir, f"{base_name}_depression.png")
    ctrl_path = os.path.join(out_dir, f"{base_name}_control.png")
    dep_fig.savefig(dep_path, dpi=dpi, bbox_inches='tight', format='png')
    plt.close(dep_fig)
    ctrl_fig.savefig(ctrl_path, dpi=dpi, bbox_inches='tight', format='png')
    plt.close(ctrl_fig)
    print(f"Saved Manhattan plot to: {dep_path}")
    print(f"Saved Manhattan plot to: {ctrl_path}")

# ==============================================================================
# HIGH-LEVEL PIPELINE
# ==============================================================================
def run_per_edge_associations(X_dep, X_ctrl, combined_data, depression_subject_ids,
                              control_subject_ids, conn_type, out_dir,
                              covariates=['motion', 'age', 'sex'], motion_metric='p24441_i2',
                              motion_metrics=None, analysis_level='global',
                              connectivity_metric='node_strength'):
    """Run per-edge covariate association analysis for depression and control cohorts.

    For each requested covariate, aligns the covariate vector to the feature
    matrix row order, computes per-edge correlations (with FDR correction) for
    both cohorts separately, saves results to CSV, and generates Manhattan plots.

    Supports 2-D input arrays (pre-vectorized edges) and 3-D matrix stacks
    (vectorized internally via the upper triangle). When motion_metrics contains
    multiple entries, each motion label is analyzed and stored independently.

    Parameters
    ----------
    X_dep : np.ndarray or np.memmap, shape (n_dep, n_edges) or (n_dep, n_nodes, n_nodes)
        Edge-feature matrix or connectivity matrix stack for depression subjects.
    X_ctrl : np.ndarray or np.memmap, shape (n_ctrl, n_edges) or (n_ctrl, n_nodes, n_nodes)
        Edge-feature matrix or connectivity matrix stack for control subjects.
        Must have the same number of edges/nodes as X_dep.
    combined_data : pd.DataFrame
        Combined cohort DataFrame containing 'eid', 'depression_status', age
        ('p21003_i2'), sex ('p31'), and motion columns as required.
    depression_subject_ids : list of str
        Ordered subject IDs corresponding to rows of X_dep.
    control_subject_ids : list of str
        Ordered subject IDs corresponding to rows of X_ctrl.
    conn_type : str
        Connectivity type label ('functional', 'structural', 'sfc').
        Used for output filenames, titles, and structural-specific NaN handling.
    out_dir : str
        Directory for CSV result files and Manhattan plot figures.
    covariates : list of str, default=['motion', 'age', 'sex']
        Covariates to analyze. Supported values:
        - 'motion': head motion (Spearman); uses motion_metrics mapping.
        - 'age': age in years ('p21003_i2', Spearman).
        - 'sex': biological sex ('p31', point-biserial).
    motion_metric : str, default='p24441_i2'
        Fallback motion column when motion_metrics is None.
    motion_metrics : dict[str, str], optional
        Mapping of motion label to DataFrame column name. Overrides motion_metric.
        Each label is analyzed and stored separately.
        Example: {'motion_fmri': 'p24441_i2', 'motion_dmri': 'p24453_i2'}.
    analysis_level : str, default='global'
        Tag prepended to output filenames to distinguish analysis levels
        (e.g., 'global', 'modular').
    connectivity_metric : str, default='node_strength'
        Label for the connectivity measure; used in figure titles. Automatically
        set to 'edge weight' when 3-D matrix input is detected.

    Returns
    -------
    results : dict[str, pd.DataFrame]
        Mapping of result key to per-edge correlation DataFrame.
        Keys follow the pattern '<label>_dep_map' and '<label>_ctrl_map' where
        <label> is the covariate name ('age', 'sex') or the motion label
        (e.g., 'motion_fmri', 'motion_dmri').
        Each DataFrame is the output of per_edge_correlation_map with columns:
        'edge', 'n_used', 'r', 'p', 'q', 'reject_fdr'.

    Notes
    -----
    For structural connectivity, non-finite r values are set to 0.0 before
    storing results, because sparse streamline matrices can produce undefined
    correlations at zero-variance edges.
    """

    # Helper: replace non-finite r values with zeros
    def _zero_undefined_r(df_map):
        if 'r' not in df_map.columns:
            return df_map
        r = df_map['r'].to_numpy(dtype=float)
        r[~np.isfinite(r)] = 0.0
        df_map = df_map.copy()
        df_map['r'] = r
        return df_map

    # Split cohorts and initialize output dict
    results = {}
    dep_df = combined_data[combined_data['depression_status'] == 1]
    ctrl_df = combined_data[combined_data['depression_status'] == 0]
    # Resolve motion label->column mapping
    motion_metric_map = motion_metrics or {'motion': motion_metric}

    # Normalize input arrays and ensure dimensionality is supported
    X_dep_arr = _stack_matrices(X_dep)
    X_ctrl_arr = _stack_matrices(X_ctrl)
    if X_dep_arr.ndim == 3:
        # Vectorize upper triangle for 3-D matrix stacks
        X_dep_vec, tri_idx = vectorize_connectivity_matrices(X_dep_arr)
        X_ctrl_vec, _ = vectorize_connectivity_matrices(X_ctrl_arr, tri_indices=tri_idx)
        X_dep_use = X_dep_vec
        X_ctrl_use = X_ctrl_vec
        if connectivity_metric == 'node_strength':
            connectivity_metric = 'edge weight'
        if analysis_level == 'matrix':
            analysis_tag = 'matrix_edges'
        else:
            analysis_tag = f"{analysis_level}_matrix_edges"
    elif X_dep_arr.ndim == 2:
        # Pre-vectorized edges: use as-is
        X_dep_use = X_dep_arr
        X_ctrl_use = X_ctrl_arr
        analysis_tag = analysis_level
    else:
        raise ValueError(f"X_dep must be 2D or 3D, got {X_dep_arr.shape}")

    # Loop through requested covariates
    for cov in covariates:
        if cov == 'motion':
            # Motion can map to multiple labels/columns
            motion_items = list(motion_metric_map.items())
            method = 'auto'
        elif cov == 'age':
            col = 'p21003_i2'
            method = 'auto'
        elif cov == 'sex':
            col = 'p31'
            method = 'pointbiserial'
        else:
            raise ValueError(f"Unknown covariate: {cov}")

        print(f"\n  Analyzing {cov} associations...")

        if cov == 'motion':
            for motion_label, col in motion_items:
                print(f"    Using motion covariate: {motion_label} ({col})")
                # Align motion vectors to X rows
                y_dep = align_motion_vector(dep_df, depression_subject_ids, col)
                y_ctrl = align_motion_vector(ctrl_df, control_subject_ids, col)

                prefix = f"{analysis_tag}_{conn_type}_per_edge_{motion_label}"

                # Compute per-edge correlations in each cohort
                dep_map = per_edge_correlation_map(X_dep_use, y_dep, method='spearman')
                ctrl_map = per_edge_correlation_map(X_ctrl_use, y_ctrl, method='spearman')

                if conn_type == 'structural':
                    dep_map = _zero_undefined_r(dep_map)
                    ctrl_map = _zero_undefined_r(ctrl_map)

                # Persist correlation maps
                dep_map.to_csv(os.path.join(out_dir, f"{prefix}_corr_depression.csv"), index=False)
                ctrl_map.to_csv(os.path.join(out_dir, f"{prefix}_corr_control.csv"), index=False)

                # Manhattan-style plots for each cohort
                plot_manhattan_style_association(
                    dep_map, ctrl_map, conn_type, out_dir, motion_label,
                    connectivity_metric, f"{prefix}_manhattan.svg"
                )

                # Store maps in results dict
                results[f'{motion_label}_dep_map'] = dep_map
                results[f'{motion_label}_ctrl_map'] = ctrl_map
            continue

        if method == 'pointbiserial':
            # Preserve raw labels for point-biserial encoding inside the function
            y_dep = align_scalar_vector(dep_df, depression_subject_ids, col, dtype=None)
            y_ctrl = align_scalar_vector(ctrl_df, control_subject_ids, col, dtype=None)

            # Compute correlations for each cohort
            dep_map = per_edge_correlation_map(X_dep_use, y_dep, method='pointbiserial')
            ctrl_map = per_edge_correlation_map(X_ctrl_use, y_ctrl, method='pointbiserial')

            if conn_type == 'structural':
                dep_map = _zero_undefined_r(dep_map)
                ctrl_map = _zero_undefined_r(ctrl_map)

            prefix = f"{analysis_tag}_{conn_type}_per_edge_{cov}"
            # Persist correlation maps
            dep_map.to_csv(os.path.join(out_dir, f"{prefix}_corr_depression.csv"), index=False)
            ctrl_map.to_csv(os.path.join(out_dir, f"{prefix}_corr_control.csv"), index=False)
        else:
            # Default to Spearman for continuous covariates
            y_dep = align_motion_vector(dep_df, depression_subject_ids, col)
            y_ctrl = align_motion_vector(ctrl_df, control_subject_ids, col)

            prefix = f"{analysis_tag}_{conn_type}_per_edge_{cov}"

            dep_map = per_edge_correlation_map(X_dep_use, y_dep, method='spearman')
            ctrl_map = per_edge_correlation_map(X_ctrl_use, y_ctrl, method='spearman')

            if conn_type == 'structural':
                dep_map = _zero_undefined_r(dep_map)
                ctrl_map = _zero_undefined_r(ctrl_map)

            # Persist correlation maps
            dep_map.to_csv(os.path.join(out_dir, f"{prefix}_corr_depression.csv"), index=False)
            ctrl_map.to_csv(os.path.join(out_dir, f"{prefix}_corr_control.csv"), index=False)

        # Generate Manhattan plots for this covariate
        plot_manhattan_style_association(
            dep_map, ctrl_map, conn_type, out_dir, cov, connectivity_metric, f"{prefix}_manhattan.svg"
        )

        # Store maps in results dict
        results[f'{cov}_dep_map'] = dep_map
        results[f'{cov}_ctrl_map'] = ctrl_map

    return results


