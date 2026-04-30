"""Utilities for resting-state FC averaging (Schaefer1000 + Tian subcortex).

This module provides utilities to prepare per-subject merged cortical +
subcortical resting-state time series, compute per-subject connectivity
matrices, average them across a cohort and visualise the resulting group
connectivity matrix.

Primary workflow
----------------
1) Prepare merged time series per subject
    - Load cortical and subcortical parcellated time series
    - Identify any missing subjects
    - Handle NaNs via per-ROI interpolation
    - Save merged time series as ``.npy`` per subject
    - Save a metadata CSV linking ``subject_id`` -> ``merged_timeseries_path``

2) Compute subject connectivity and average connectivity
    - Load merged time series for each subject using the metadata CSV
    - Optionally clean the time series with ``nilearn.signal.clean``
    - Compute correlation connectivity matrices using ``nilearn.connectome``
    - Average across subjects (arithmetic mean or Fisher-z averaging)
    - Optionally save per-subject matrices and the group mean

3) Plot example subject's connectivity matrix and the group average matrix

Expected `data_dir` layout
--------------------------
Each subject is expected to have a subdirectory under ``data_dir`` named by
their subject id. Within each subject directory the script expects cortical and
subcortical time series files at paths matching the default suffixes (see
``prepare_merged_timeseries`` parameters). Example:

        /path/to/data_dir/<subject_id>/i2/fMRI.Schaefer7n1000p.csv.gz
        /path/to/data_dir/<subject_id>/i2/fMRI.Tian_Subcortex_S4_3T.csv.gz

Files written (side-effects)
----------------------------
- Per-subject merged time series: ``{subject_dir}/i2/{merged_timeseries_name}``
- ``merged_resting_state_timeseries_paths.csv`` (metadata CSV with columns
    ``subject_id,merged_timeseries_path``)
- Per-subject connectivity matrices (when enabled):
    ``{subject_dir}/i2/{subject_id}_connectivity.npy``
- Group average matrix: saved to ``{data_dir}/{average_matrix_name}``
- NaN / missing-subject audit CSVs (see return value of
    ``prepare_merged_timeseries``)

Notes
-----
- If you prefer Fisher-z averaging, set ``fisher_z_average=True`` in
    ``ConnectivityConfig``.

Architecture
------------
The script is organized into functional sections:

- **Utilities**

- **Data loading and preprocessing**

- **Merging timeseries**

- **Connectivity computation and averaging**

- **Visualization**

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure

# ==============================================================================
# UTILITIES
# ==============================================================================
@dataclass(frozen=True)
class NaNHandlingConfig:
    """Configuration for missing-data handling in time series.

    Attributes
    ----------
    interp_method : str
        Interpolation method passed to ``pandas.Series.interpolate`` (e.g.
        ``'linear'``, ``'time'``). Default is ``'linear'``.
    roi_nan_ratio_threshold : float
        Fraction threshold (0-1) above which a region-of-interest is marked as
        'bad' due to excessive NaNs. Default is ``0.05`` (5%% of timepoints).
    """

    interp_method: str = "linear"
    roi_nan_ratio_threshold: float = 0.05

@dataclass(frozen=True)
class ConnectivityConfig:
    """Configuration for connectivity computation.

    Attributes
    ----------
    kind : str
        Type of connectivity to compute (e.g. ``'correlation'``). Passed to
        ``nilearn.connectome.ConnectivityMeasure``.
    standardize : str
        Standardization mode passed to ``ConnectivityMeasure`` (see nilearn
        docs). Typical values include ``'zscore_sample'`` or ``None``.
    clean_kwargs : Optional[Dict[str, Any]]
        If provided, keyword arguments forwarded to ``nilearn.signal.clean``
        when cleaning time series. If ``None`` no cleaning is performed.
    fisher_z_average : bool
        If True, subject correlation matrices are converted to Fisher-z,
        averaged, then inverse-transformed back to correlation values.
    """

    kind: str = "correlation"
    standardize: str = "zscore_sample"

    # Optional nilearn.signal.clean parameters
    # If None, no additional cleaning is performed.
    clean_kwargs: Optional[Dict[str, Any]] = None

    # Optional averaging method for correlation matrices
    fisher_z_average: bool = False

def list_subject_ids(data_dir: Path) -> List[str]:
    """List subject directory names in ``data_dir`` (sorted).

    Parameters
    ----------
    data_dir : Path
        Directory containing one subdirectory per subject.

    Returns
    -------
    List[str]
        Sorted list of directory names found in ``data_dir``. These are the
        raw directory names and are not validated beyond ``is_dir()`` checks.
    """

    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

def save_labels(labels: Iterable[str], out_path: Path) -> None:
    """Write a sequence of label strings to a text file (one label per line).

    The function will create parent directories as needed and will overwrite
    an existing file at ``out_path``.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")

def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    """Fisher Z transform for correlations (with safe clipping)."""

    r = np.asarray(r, dtype=float)
    r = np.clip(r, -0.9999999, 0.9999999)
    z = np.arctanh(r)
    np.fill_diagonal(z, 0.0)
    return z


def inverse_fisher_z(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher Z transform back to correlations."""

    z = np.asarray(z, dtype=float)
    r = np.tanh(z)
    np.fill_diagonal(r, 1.0)
    return r


def _maybe_clean_timeseries(ts: np.ndarray, clean_kwargs: Optional[Dict[str, Any]]) -> np.ndarray:
    """Optionally clean a time series using ``nilearn.signal.clean``.

    Parameters
    ----------
    ts : np.ndarray
        Time series array with shape (timepoints, rois) or (timepoints,) for a
        single ROI. If ``clean_kwargs`` is ``None`` this function returns
        ``ts`` unchanged.
    clean_kwargs : dict or None
        Keyword arguments forwarded to ``nilearn.signal.clean``. Example keys
        include ``standardize``, ``detrend``, ``low_pass``, ``high_pass``,
        ``t_r``.

    Returns
    -------
    np.ndarray
        Cleaned time series array (same shape as input).
    """

    if clean_kwargs is None:
        return ts

    from nilearn import signal

    return signal.clean(ts, **clean_kwargs)

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_timeseries_csv_gz(path: Path) -> np.ndarray:
    """Load a UKB parcellated time series CSV (.csv.gz) into a (T, R) array.

    The UKB files typically store ROI labels in a ``label_name`` column and
    time points along columns; the file is transposed so the returned array is
    (timepoints, rois).

    Parameters
    ----------
    path : Path
        Path to the ``.csv.gz`` time series file. Must contain a ``label_name``
        column and one column per timepoint.

    Returns
    -------
    np.ndarray
        Array of shape (timepoints, rois) with dtype inferred from the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the expected ``label_name`` column is missing.
    """

    df = pd.read_csv(path, compression="gzip")
    ts = df.drop(columns=["label_name"]).to_numpy().T
    return ts

def load_label_names(path: Path) -> List[str]:
    """Load the ``label_name`` entries from a UKB time series CSV (.csv.gz).

    Parameters
    ----------
    path : Path
        Path to the time series file. The function reads the ``label_name``
        column and returns its values in file order.

    Returns
    -------
    List[str]
        List of label names in the order they appear in the file.
    """

    df = pd.read_csv(path, compression="gzip", usecols=["label_name"])
    return df["label_name"].tolist()

def impute_timeseries_nans(
    ts: np.ndarray,
    cfg: NaNHandlingConfig = NaNHandlingConfig(),
) -> Tuple[np.ndarray, int, float, int, float]:
    """Impute NaNs per ROI in a (T, R) time series array.

    Strategy
    --------
    - For each ROI (column) interpolate missing values along time using
      ``pandas.Series.interpolate`` with ``cfg.interp_method``.
    - At the start/end of the series fill in both directions (``limit_direction='both'``).
    - Any remaining NaNs after interpolation are replaced by the ROI mean; if
      the mean is NaN (all values NaN), the ROI is set to 0.
    - A ROI is considered "bad" if its NaN ratio exceeds
      ``cfg.roi_nan_ratio_threshold``.

    Parameters
    ----------
    ts : np.ndarray
        2-D array with shape (timepoints, rois). Must be convertible to float.
    cfg : NaNHandlingConfig
        Configuration for interpolation and bad-ROI threshold.

    Returns
    -------
    Tuple[np.ndarray, int, float, int, float]
        - cleaned_ts : np.ndarray
            The cleaned timeseries array with NaNs imputed and dtype ``np.float32``.
        - total_nans : int
            Total number of NaN entries found in the input.
        - pct_nans : float
            Percentage of entries that were NaN (0-100).
        - bad_rois : int
            Number of ROIs whose NaN ratio exceeded ``cfg.roi_nan_ratio_threshold``.
        - pct_bad_rois : float
            Percentage of ROIs marked as bad (0-100).

    Raises
    ------
    ValueError
        If ``ts`` is not a 2-D array.
    """

    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 2:
        raise ValueError(f"Expected ts to be 2D (T,R); got shape {ts.shape}")

    T, R = ts.shape
    total_nans = int(np.count_nonzero(~np.isfinite(ts)))
    pct_nans = 100.0 * total_nans / ts.size if ts.size else 0.0

    bad_rois = 0
    for j in range(R):
        col = ts[:, j]
        mask = ~np.isfinite(col)
        nan_ratio = float(np.count_nonzero(mask)) / T if T else 0.0

        if np.all(mask):
            bad_rois += 1
            ts[:, j] = 0.0
            continue

        if np.any(mask):
            s = pd.Series(col)
            col_imp = s.interpolate(method=cfg.interp_method, limit_direction="both").to_numpy()
            if not np.isfinite(col_imp).all():
                mean_val = np.nanmean(col_imp)
                if not np.isfinite(mean_val):
                    mean_val = 0.0
                col_imp = np.where(np.isfinite(col_imp), col_imp, mean_val)
            ts[:, j] = col_imp

        if nan_ratio > cfg.roi_nan_ratio_threshold:
            bad_rois += 1

    pct_bad_rois = 100.0 * bad_rois / R if R else 0.0

    ts = np.where(np.isfinite(ts), ts, 0.0).astype(np.float32, copy=False)
    return ts, total_nans, pct_nans, bad_rois, pct_bad_rois

# ==============================================================================
# MERGING TIMESERIES 
# ==============================================================================
def prepare_merged_timeseries(
    data_dir: Path,
    cortical_suffix: str = "i2/fMRI.Schaefer7n1000p.csv.gz",
    subcortical_suffix: str = "i2/fMRI.Tian_Subcortex_S4_3T.csv.gz",
    merged_timeseries_name: str = "fMRI.Schaefer7n1000p_TianSubcortexS4_merged_timeseries.npy",
    labels_txt_name: str = "Schaefer7n1000p_TianSubcortexS4_labels.txt",
    nan_cfg: NaNHandlingConfig = NaNHandlingConfig(),
) -> Dict[str, Path]:
    """Prepare merged cortical+subcortical time series for all subjects.

    It writes:
    - merged `.npy` time series per subject
    - `merged_resting_state_timeseries_paths.csv`
    - `cortical_resting_state_timeseries_nans_info.csv`
    - `subcortical_resting_state_timeseries_nans_info.csv`
    - `missing_subjects_resting_state_timeseries.csv`
    - merged labels txt

    Returns
    -------
    dict
        Paths to the generated metadata files.
    """

    data_dir = Path(data_dir)
    subject_ids = list_subject_ids(data_dir)
    if not subject_ids:
        raise FileNotFoundError(f"No subject directories found under {data_dir}")

    # Create merged labels based on first subject
    first_subj = subject_ids[0]
    cortical_path0 = data_dir / first_subj / cortical_suffix
    subcortical_path0 = data_dir / first_subj / subcortical_suffix
    merged_labels = load_label_names(cortical_path0) + load_label_names(subcortical_path0)

    labels_output_path = data_dir / labels_txt_name
    save_labels(merged_labels, labels_output_path)

    missing_subjects: List[str] = []
    metadata_rows: List[Dict[str, str]] = []
    cortical_nans_rows: List[Dict[str, Any]] = []
    subcortical_nans_rows: List[Dict[str, Any]] = []

    iterator = tqdm(
    subject_ids,
    total=len(subject_ids),
    desc="Preparing merged time series",
    unit="subj",
    dynamic_ncols=True,
    )

    for subj_id in iterator:
        cortical_path = data_dir / subj_id / cortical_suffix
        subcortical_path = data_dir / subj_id / subcortical_suffix
        if not cortical_path.exists() or not subcortical_path.exists():
            missing_subjects.append(subj_id)
            continue

        cortical_ts = load_timeseries_csv_gz(cortical_path)
        cortical_ts, n_nans_c, pct_nans_c, n_bad_c, pct_bad_c = impute_timeseries_nans(
            cortical_ts, cfg=nan_cfg
        )
        action_c = "imputed" if n_nans_c > 0 else "clean"
        cortical_nans_rows.append(
            {
                "subject_id": subj_id,
                "n_nans": n_nans_c,
                "pct_nans": pct_nans_c,
                "n_bad_rois": n_bad_c,
                "pct_bad_rois": pct_bad_c,
                "action": action_c,
            }
        )

        subcortical_ts = load_timeseries_csv_gz(subcortical_path)
        subcortical_ts, n_nans_s, pct_nans_s, n_bad_s, pct_bad_s = impute_timeseries_nans(
            subcortical_ts, cfg=nan_cfg
        )
        action_s = "imputed" if n_nans_s > 0 else "clean"
        subcortical_nans_rows.append(
            {
                "subject_id": subj_id,
                "n_nans": n_nans_s,
                "pct_nans": pct_nans_s,
                "n_bad_rois": n_bad_s,
                "pct_bad_rois": pct_bad_s,
                "action": action_s,
            }
        )

        merged_ts = np.hstack((cortical_ts, subcortical_ts))
        merged_ts_path = data_dir / subj_id / "i2" / merged_timeseries_name
        merged_ts_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(merged_ts_path, merged_ts)

        metadata_rows.append(
            {"subject_id": subj_id, "merged_timeseries_path": str(merged_ts_path)}
        )

    # Write summary files
    cortical_nans_info_path = data_dir / "cortical_resting_state_timeseries_nans_info.csv"
    subcortical_nans_info_path = data_dir / "subcortical_resting_state_timeseries_nans_info.csv"
    metadata_csv_path = data_dir / "merged_resting_state_timeseries_paths.csv"
    missing_subjects_path = data_dir / "missing_subjects_resting_state_timeseries.csv"

    pd.DataFrame(cortical_nans_rows).to_csv(cortical_nans_info_path, index=False)
    pd.DataFrame(subcortical_nans_rows).to_csv(subcortical_nans_info_path, index=False)
    pd.DataFrame(metadata_rows).to_csv(metadata_csv_path, index=False)
    pd.DataFrame({"subject_id": missing_subjects}).to_csv(missing_subjects_path, index=False)

    return {
        "labels_path": labels_output_path,
        "metadata_paths_csv": metadata_csv_path,
        "cortical_nans_csv": cortical_nans_info_path,
        "subcortical_nans_csv": subcortical_nans_info_path,
        "missing_subjects_csv": missing_subjects_path,
    }

# ==============================================================================
# CONNECTIVITY COMPUTATION AND AVERAGING
# ==============================================================================
def compute_average_connectivity(
    metadata_paths_csv: Path,
    data_dir: Path,
    cfg: ConnectivityConfig = ConnectivityConfig(),
    save_subject_matrices: bool = True,
    subject_matrix_suffix: str = "_connectivity.npy",
    average_matrix_name: str = "average_resting_state_connectivity_matrix.npy",
) -> np.ndarray:
    """Compute per-subject connectivity matrices and the average across subjects.

    The function reads a metadata CSV containing per-subject paths to merged
    timeseries (see ``prepare_merged_timeseries``). For each row the
    corresponding timeseries file is loaded, optionally cleaned, and a
    connectivity matrix is computed using ``nilearn.connectome.ConnectivityMeasure``.

    Parameters
    ----------
    metadata_paths_csv : Path
        CSV file containing at least the columns ``subject_id`` and
        ``merged_timeseries_path``. Each ``merged_timeseries_path`` must point
        to a saved ``.npy`` file containing the subject's timeseries (timepoints x ROIs).
    data_dir : Path
        Root directory used to save subject matrices and the final average
        matrix (the file ``{data_dir}/{average_matrix_name}`` is written).
    cfg : ConnectivityConfig
        Connectivity configuration (see ``ConnectivityConfig`` dataclass).
    save_subject_matrices : bool
        If True, writes each subject's connectivity matrix to
        ``{data_dir}/{subject_id}/i2/{subject_id}{subject_matrix_suffix}``.
    subject_matrix_suffix : str
        Suffix appended to each per-subject filename when saving connectivity matrices.
    average_matrix_name : str
        Filename used to save the group average matrix in ``data_dir``.

    Returns
    -------
    np.ndarray
        The average connectivity matrix (square, symmetric) as a numpy array.

    Raises
    ------
    ValueError
        If the metadata CSV is empty or required columns are missing.
    """

    data_dir = Path(data_dir)
    metadata_paths_csv = Path(metadata_paths_csv)

    meta = pd.read_csv(metadata_paths_csv)
    if meta.empty:
        raise ValueError(f"No rows found in {metadata_paths_csv}")

    conn_measure = ConnectivityMeasure(kind=cfg.kind, standardize=cfg.standardize)

    # First subject
    first_ts = np.load(meta["merged_timeseries_path"].iloc[0]).astype(np.float32)
    first_ts = _maybe_clean_timeseries(first_ts, cfg.clean_kwargs)
    first_conn = conn_measure.fit_transform([first_ts])[0]

    if save_subject_matrices:
        first_id = str(meta["subject_id"].iloc[0])
        out_path = data_dir / first_id / "i2" / f"{first_id}{subject_matrix_suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, first_conn)

    if cfg.fisher_z_average:
        running = fisher_z_transform(first_conn).astype(np.float64)
    else:
        running = first_conn.astype(np.float64)

    n_subjects = len(meta)

    iterator = tqdm(
    range(1, n_subjects),
    total=n_subjects,
    desc="Computing connectivity",
    unit="subj",
    dynamic_ncols=True,
    )

    for i in iterator:
        ts = np.load(meta["merged_timeseries_path"].iloc[i]).astype(np.float32)
        ts = _maybe_clean_timeseries(ts, cfg.clean_kwargs)
        conn = conn_measure.fit_transform([ts])[0]

        if save_subject_matrices:
            subj_id = str(meta["subject_id"].iloc[i])
            out_path = data_dir / subj_id / "i2" / f"{subj_id}{subject_matrix_suffix}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, conn)

        if cfg.fisher_z_average:
            running += fisher_z_transform(conn).astype(np.float64)
        else:
            running += conn.astype(np.float64)

    running /= float(n_subjects)

    if cfg.fisher_z_average:
        avg = inverse_fisher_z(running)
    else:
        avg = running

    avg_out = data_dir / average_matrix_name
    np.save(avg_out, avg)
    return avg

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_connectivity_matrix(
    avg_connectivity: np.ndarray,
    labels_path: Optional[Path],
    out_path: Path,
    title: str = None,
    figure: Tuple[int, int] = (20, 20),
    cmap: str = "seismic",
) -> None:
    """Plot and save the average connectivity matrix.

    Parameters
    ----------
    avg_connectivity : np.ndarray
        Square connectivity matrix (n_rois x n_rois). The matrix is expected
        to be symmetric; diagonal values are typically 0 (z) or 1 (r).
    labels_path : Path or None
        Optional path to a text file with one label per line used to annotate
        axes. If ``None`` or the file does not exist, no labels are shown.
    out_path : Path
        Path to save the generated plot (SVG). Parent directories will be
        created if missing; the file is written with ``dpi=300``.
    title : str
        Figure title.
    figure : Tuple[int, int]
        Figure size (width, height).
    cmap : str
        Colormap used for the matrix visualization (default: ``'seismic'``).

    Side effects
    ------------
    - Saves a SVG to ``out_path`` and calls ``nilearn.plotting.show()`` (may
      open an interactive window in non-headless environments).
    """

    labels = None
    if labels_path is not None and Path(labels_path).exists():
        labels = Path(labels_path).read_text(encoding="utf-8").splitlines()

    vmax = float(np.max(np.abs(avg_connectivity)))
    plotting.plot_matrix(
        avg_connectivity,
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax,
        title=title,
        labels=labels,
        reorder=False,
        figure=figure,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plotting.show()
    plt.close()
