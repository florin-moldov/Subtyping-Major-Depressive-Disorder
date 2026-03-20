"""Utilities for NaN verification and averaging structural connectomes across subjects.

This module refactors the original `avg_strct.py` script into reusable functions.

Primary workflow
----------------
1) List subject IDs in a cohort directory (optionally excluding subjects)
2) Load each subject's structural connectome matrix (CSV/CSV.GZ)
3) Skip connectome issues tracking:
   - missing file per subject
   - NaNs in a subject matrix
4) Compute the arithmetic mean across successfully included subjects
5) Save the average matrix and small QC reports
6) Visualize the average matrix with nilearn's matrix plot

Architecture
------------
The script is organized into functional sections:

- **Utilities**

- **Data loading and preprocessing**

- **Connectivity averaging**

- **Visualization**

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# UTILITIES
# ==============================================================================
@dataclass(frozen=True)
class StructuralAverageResult:
    """Result bundle for average structural connectivity computation.

    Attributes
    ----------
    avg_matrix : np.ndarray
        The computed average structural connectivity matrix (n_rois x n_rois).
    n_included : int
        Number of subjects included in the average.
    n_candidates : int
        Number of candidate subjects attempted.
    missing_subjects : List[str]
        List of subject IDs for which the expected connectome file was missing.
    nan_subjects : pd.DataFrame
        DataFrame reporting subjects with NaNs and the number of NaNs per subject.
    """

    avg_matrix: np.ndarray
    n_included: int
    n_candidates: int
    missing_subjects: List[str]
    nan_subjects: pd.DataFrame

def load_labels_txt(labels_path: Path) -> List[str]:
    """Load ROI labels (one per line) from a text file.

    Parameters
    ----------
    labels_path : Path
        Path to a UTF-8 encoded text file with one ROI label per line.

    Returns
    -------
    List[str]
        List of label strings in file order.

    Raises
    ------
    FileNotFoundError
        If ``labels_path`` does not exist.
    """

    labels_path = Path(labels_path)
    return labels_path.read_text(encoding="utf-8").splitlines()

def list_subject_ids(data_dir: Path) -> List[str]:
    """List subject directory names in ``data_dir`` (sorted).

    Parameters
    ----------
    data_dir : Path
        Directory that contains one subdirectory per subject.

    Returns
    -------
    List[str]
        Sorted list of directory names found in ``data_dir``; entries are
        returned only when ``is_dir()`` evaluates to True.
    """

    data_dir = Path(data_dir)
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

def connectome_path(
    data_dir: Path,
    subject_id: str,
    subdir: str = "i2",
    filename_template: str = "connectome_streamline_count_10M.csv.gz",
) -> Path:
    """Build the expected connectome path for a subject.

    Parameters
    ----------
    data_dir : Path
        Root cohort directory.
    subject_id : str
        Subject identifier (directory name under ``data_dir``).
    subdir : str
        Subdirectory inside the subject folder where the connectome file lives.
    filename_template : str
        Filename or template for the connectome CSV file.

    Returns
    -------
    Path
        Full path to the expected connectome file for the subject.
    """

    return Path(data_dir) / str(subject_id) / subdir / filename_template

def exclude_subjects_by_eid(
    df,
    eids_source,
    eid_col: str = "eid",
    source_eid_col: str = "eid",
    inplace: bool = False,
    return_count: bool = True,
    out_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[int]] | pd.DataFrame:
    """Filter rows from ``df`` whose IDs appear in ``eids_source``.

    Parameters
    ----------
    df : pandas.DataFrame or str or Path
        If a string or Path is provided, it is interpreted as a path to a CSV
        file which will be loaded into a DataFrame. Otherwise it must be a
        pandas DataFrame.
    eids_source : str | Path | pandas.DataFrame | Iterable
        Source of eids to exclude. If a path is provided the function will
        attempt to read a CSV (falling back to a plain text file with one id
        per line). If a DataFrame is provided the column named by
        ``source_eid_col`` will be used. If an iterable is provided it is
        treated as a collection of ids.
    eid_col : str
        Column name in ``df`` that contains subject IDs (default: ``'eid'``).
    source_eid_col : str
        Column name in ``eids_source`` DataFrame to read IDs from (default: ``'eid'``).
    inplace : bool
        If True, modify the provided DataFrame in place and return it; if
        False, return a filtered copy.
    return_count : bool
        If True, return a tuple ``(filtered_df, n_excluded)``; otherwise return
        just the filtered DataFrame.
    out_path : Optional[Path]
        If provided, the filtered DataFrame will be saved as a CSV at this path.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, int)
        Filtered DataFrame, optionally accompanied by the number of excluded rows.

    Raises
    ------
    FileNotFoundError
        If a provided file path does not exist.
    KeyError
        If required columns are missing in provided DataFrames.
    TypeError
        If input types are unsupported.
    """

    import os
    import pandas as pd
    import numpy as np

    # 1) load original dataframe (if not already provided)
    # Accept string paths and pathlib.Path objects
    if isinstance(df, (str, Path)):
        path = df
        if not os.path.exists(path):
            raise FileNotFoundError(f"DataFrame path not found: {path}")
        df = pd.read_csv(df, dtype={eid_col: str})
    elif isinstance(df, pd.DataFrame):
        pass
    else:
        raise TypeError("df must be a path (str) or pandas DataFrame")

    # 2) obtain iterable of eids to exclude
    # Accept string paths and pathlib.Path objects
    if isinstance(eids_source, (str, Path)):
        path = eids_source
        if not os.path.exists(path):
            raise FileNotFoundError(f"eids_source path not found: {path}")
        # try CSV first, fallback to plain text lines
        try:
            src_df = pd.read_csv(path, dtype={source_eid_col: str})
            eids_iter = src_df[source_eid_col].astype(str).tolist()
        except Exception:
            # plain text: one eid per line
            with open(path, "r") as f:
                eids_iter = [line.strip() for line in f if line.strip()]
    elif isinstance(eids_source, pd.DataFrame):
        if source_eid_col not in eids_source.columns:
            raise KeyError(f"source_eid_col '{source_eid_col}' not in provided DataFrame")
        eids_iter = eids_source[source_eid_col].astype(str).tolist()
    elif isinstance(eids_source, (list, tuple, set, np.ndarray)):
        eids_iter = list(eids_source)
    else:
        raise TypeError("eids_source must be a path (str), DataFrame, or iterable of ids")

    # 3) normalize to set of strings (strip whitespace, ignore NaN)
    exclude_set = set(str(x).strip() for x in eids_iter if pd.notnull(x))

    # 4) build boolean mask of rows to remove (matching after casting df[eid_col] to str)
    if eid_col not in df.columns:
        raise KeyError(f"eid_col '{eid_col}' not found in dataframe")
    eid_series = df[eid_col].astype(str).str.strip()
    remove_mask = eid_series.isin(exclude_set)

    n_excluded = int(remove_mask.sum())

    # 5) Save filtered dataframe 
    if inplace:
        df.drop(index=df.index[remove_mask], inplace=True)
        filtered = df
    else:
        filtered = df.loc[~remove_mask].copy()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(out_path, index=False)
        print(f"Saved filtered dataframe to: {out_path}. Excluded {n_excluded} subjects.")

    # 6) Return filtered dataframe (and count if requested)
    if return_count:
        return filtered, n_excluded
    else:
        return filtered

def normalize_for_plot(
    mat: np.ndarray,
    *,
    log_transform: bool = False,
    log_epsilon: float = 1e-4,
) -> np.ndarray:
    """Prepare a matrix for plotting by optional log transform and [0,1] scaling.

    Parameters
    ----------
    mat : np.ndarray
        2-D array to normalize.
    log_transform : bool
        If True, apply ``np.log(mat + log_epsilon)`` prior to scaling.
    log_epsilon : float
        Small positive constant added before the log to avoid -inf.

    Returns
    -------
    np.ndarray
        Normalized matrix with values in [0, 1]. If the matrix is constant
        (max==min) returns a zero matrix of the same shape.
    """

    mat = np.asarray(mat, dtype=np.float64)
    if log_transform:
        mat = np.log(mat + float(log_epsilon))

    vmin = float(np.min(mat))
    vmax = float(np.max(mat))
    if vmax == vmin:
        return np.zeros_like(mat, dtype=np.float64)

    return (mat - vmin) / (vmax - vmin)

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_excluded_subject_ids(excluded_subjects_csv: Path, column: str = "subject_id") -> List[str]:
    """Load excluded subject IDs from a CSV file.

    Parameters
    ----------
    excluded_subjects_csv : Path
        Path to a CSV file containing at least the column named by ``column``.
    column : str
        Column name to read subject IDs from (default: ``'subject_id'``).

    Returns
    -------
    List[str]
        List of subject IDs as strings.

    Raises
    ------
    ValueError
        If the requested ``column`` is not present in the CSV file.
    """

    excluded_subjects_csv = Path(excluded_subjects_csv)
    df = pd.read_csv(excluded_subjects_csv)
    if column not in df.columns:
        raise ValueError(f"Expected column '{column}' in {excluded_subjects_csv}; got {list(df.columns)}")
    return df[column].astype(str).tolist()

def load_connectome_matrix(path: Path) -> np.ndarray:
    """Load a connectome matrix from CSV/CSV.GZ into a float numpy array.

    Parameters
    ----------
    path : Path
        Path to a CSV or gzipped CSV containing a square matrix with no header.

    Returns
    -------
    np.ndarray
        Matrix as ``np.float64``.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    """

    path = Path(path)
    mat = pd.read_csv(path, compression="infer", header=None).to_numpy()
    return np.asarray(mat, dtype=np.float64)

# ==============================================================================
# CONNECTIVITY AVERAGING
# ==============================================================================
def compute_average_structural_connectivity(
    data_dir: Path,
    subject_ids: Iterable[str],
    *,
    subdir: str = "i2",
    filename_template: str = "connectome_streamline_count_10M.csv.gz",
    skip_nan_subjects: bool = True,
    show_progress: bool = True,
) -> StructuralAverageResult:
    """Compute the average structural connectome across subjects.

    Parameters
    ----------
    data_dir : Path
        Cohort directory containing subject folders.
    subject_ids : Iterable[str]
        Iterable of subject identifiers to process.
    subdir : str
        Subdirectory under each subject directory containing the connectome file.
    filename_template : str
        Filename template of the connectome file inside each subject subdir.
    skip_nan_subjects : bool
        If True, subjects with any NaN entries in their matrix are excluded
        from the average. If False, such matrices are included and NaNs will
        propagate to the result.
    show_progress : bool
        If True, display a tqdm progress bar when available.

    Returns
    -------
    StructuralAverageResult
        Dataclass containing the average matrix, counts of included/candidate
        subjects, a list of missing subject IDs and a DataFrame of NaN
        diagnostics (columns include ``subject_id`` and ``n_nans``).

    Raises
    ------
    ValueError
        If no valid connectome matrices were included and an average cannot be
        computed.
    """
    

    data_dir = Path(data_dir)
    subject_ids = [str(s) for s in subject_ids]

    missing_subjects: List[str] = []
    nan_rows: List[dict] = []

    iterator = subject_ids
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(subject_ids, total=len(subject_ids), desc="Averaging structural connectomes", unit="subj")
        except Exception:
            iterator = subject_ids

    running: Optional[np.ndarray] = None
    n_included = 0

    for subj_id in iterator:
        conn_file = connectome_path(
            data_dir=data_dir,
            subject_id=subj_id,
            subdir=subdir,
            filename_template=filename_template,
        )
        if not conn_file.exists():
            missing_subjects.append(subj_id)
            continue

        mat = load_connectome_matrix(conn_file)

        if np.isnan(mat).any():
            nan_rows.append({"subject_id": subj_id, "n_nans": int(np.isnan(mat).sum()), "prop_nan": float(np.isnan(mat).mean())})
            if skip_nan_subjects:
                continue

        if running is None:
            running = np.zeros_like(mat, dtype=np.float64)

        running += mat
        n_included += 1

    if running is None or n_included == 0:
        raise ValueError("No valid connectome matrices were included; cannot compute average.")

    avg = running / float(n_included)

    nan_subjects = pd.DataFrame(nan_rows, columns=["subject_id", "n_nans"])

    return StructuralAverageResult(
        avg_matrix=avg,
        n_included=n_included,
        n_candidates=len(subject_ids),
        missing_subjects=missing_subjects,
        nan_subjects=nan_subjects,
    )

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_connectivity_matrix(
    mat: np.ndarray,
    *,
    labels: Optional[List[str]] = None,
    title: str = "Average Structural Connectivity Matrix",
    cmap: str = "Reds",
    figure: Tuple[int, int] = (20, 20),
    reorder: bool = False,
    out_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Plot a connectivity matrix using Nilearn and optionally save it to disk.

    Parameters
    ----------
    mat : np.ndarray
        Square connectivity matrix (n_rois x n_rois).
    labels : Optional[List[str]]
        Optional list of labels for axes; length must match matrix size.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    figure : Tuple[int, int]
        Figure size (width, height).
    reorder : bool
        If True, Nilearn will reorder regions for visualization.
    out_path : Optional[Path]
            If provided, the plotted figure is saved as a SVG at this path.
        dpi : int
            Dots-per-inch for the saved SVG.

        Side effects
        ------------
        - Calls ``nilearn.plotting.plot_matrix`` and ``plotting.show()``; this may
          open an interactive window in non-headless environments.
        - If ``out_path`` is provided, the file is written to disk.
        """

    from nilearn import plotting
    import matplotlib.pyplot as plt

    plotting.plot_matrix(
        mat,
        figure=figure,
        colorbar=True,
        title=title,
        cmap=cmap,
        labels=labels,
        reorder=reorder,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=dpi, format='svg')

    plotting.show()
    plt.close()
