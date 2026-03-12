"""Main entrypoint for resting-state FC averaging.

This script computes individual and average resting-state functional connectivity matrices
from merged time series data for a specified cohort using utilities from `mgng_avg_rest_utils.py`.
Also, it visualizes the average connectivity matrix.

Outputs
--------------------------------
- `merged_resting_state_timeseries_paths.csv`
- per-subject `*_connectivity.npy` matrices (optional)
- `cortical_resting_state_timeseries_nans_info.csv`
- `subcortical_resting_state_timeseries_nans_info.csv`
- `missing_subjects_resting_state_timeseries.csv`
- `average_resting_state_connectivity_matrix.npy`
- `*_average_func_conn_matrix_Schaefer7n1000p_TianSubcortexS4.png`

Edit the configuration in `main()` to match your cohort and directories.
"""

from __future__ import annotations
from pathlib import Path

from mgng_avg_rest_utils import (
    ConnectivityConfig,
    NaNHandlingConfig,
    compute_average_connectivity,
    plot_average_connectivity_matrix,
    prepare_merged_timeseries,
)


def main() -> None:
    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------
    project_root = Path(".../subtyping_depression")

    cohort_type = "control"  # 'F32' or 'control'
    data_dir = project_root / "data" / "UKB" / f"{cohort_type}_notask_STRCO_RSSCHA_RSTIA"
    vis_dir = project_root  / "reports" / "figures" / "schaefer1000+tian54" / "functional_con"

    # If True, assumes merged time series + metadata CSV already exist
    skip_prepare = False

    # Time series preparation
    nan_cfg = NaNHandlingConfig(
        interp_method="linear",
        roi_nan_ratio_threshold=0.05,
        subject_bad_roi_ratio_threshold=0.10,
    )

    # Connectivity computation
    conn_cfg = ConnectivityConfig(
        kind="correlation",
        standardize="zscore_sample",
        clean_kwargs=None,
        fisher_z_average=False,
    )

    save_subject_matrices = True

    # ---------------------------------------------------------------------
    # Step 1: Prepare merged time series (optional)
    # ---------------------------------------------------------------------
    if not skip_prepare:
        print("[1/3] Preparing merged time series...")
        outputs = prepare_merged_timeseries(data_dir=data_dir, nan_cfg=nan_cfg)
        labels_path = outputs["labels_path"]
        metadata_paths_csv = outputs["metadata_paths_csv"]
        print("  labels:", labels_path)
        print("  metadata:", metadata_paths_csv)
    else:
        print("[1/3] Skipping time series preparation...")
        labels_path = data_dir / "Schaefer7n1000p_TianSubcortexS4_labels.txt"
        metadata_paths_csv = data_dir / "merged_resting_state_timeseries_paths.csv"

    # ---------------------------------------------------------------------
    # Step 2: Compute average connectivity
    # ---------------------------------------------------------------------
    print("[2/3] Computing average connectivity...")
    avg_connectivity = compute_average_connectivity(
        metadata_paths_csv=metadata_paths_csv,
        data_dir=data_dir,
        cfg=conn_cfg,
        save_subject_matrices=save_subject_matrices,
    )
    print("  avg shape:", avg_connectivity.shape)

    # ---------------------------------------------------------------------
    # Step 3: Plot
    # ---------------------------------------------------------------------
    print("[3/3] Plotting and saving average matrix...")
    out_png = vis_dir / f"{cohort_type}_average_func_conn_matrix_Schaefer7n1000p_TianSubcortexS4.png"
    plot_average_connectivity_matrix(
        avg_connectivity=avg_connectivity,
        labels_path=labels_path,
        out_path=out_png,
    )
    print("  saved:", out_png)


if __name__ == "__main__":
    main()
