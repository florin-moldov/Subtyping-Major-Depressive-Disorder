"""Main entrypoint for resting-state FC averaging.

This script computes individual and average resting-state functional connectivity matrices
from merged time series data for a specified cohort using utilities from `mgng_avg_rest_utils.py`.
Also, it visualizes the average connectivity matrix.

Outputs
--------------------------------
- `merged_resting_state_timeseries_paths.csv`
- per-subject `*_connectivity.npy` matrices
- `cortical_resting_state_timeseries_nans_info.csv`
- `subcortical_resting_state_timeseries_nans_info.csv`
- `missing_subjects_resting_state_timeseries.csv`
- `average_resting_state_connectivity_matrix.npy`
- example subject `*_func_conn_matrix_Schaefer7n1000p_TianSubcortexS4.png`
- `*_average_func_conn_matrix_Schaefer7n1000p_TianSubcortexS4.png`

Edit the configuration in `main()` to match your cohort and directories.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from mgng_avg_rest_utils import (
    ConnectivityConfig,
    NaNHandlingConfig,
    compute_average_connectivity,
    plot_connectivity_matrix,
    prepare_merged_timeseries,
)


def main() -> None:
    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------
    project_root = Path(".../subtyping_depression")

    cohort_type = "F32"  # 'F32' or 'control'
    data_dir = project_root / "data" / "UKB" / f"{cohort_type}_notask_STRCO_RSSCHA_RSTIA"
    vis_dir = project_root  / "reports" / "figures" / "schaefer1000+tian54" / "functional_con"

    example_subject_id = "..."  # Change to a valid subject ID from your cohort to visualize their connectivity matrix

    # Time series preparation
    nan_cfg = NaNHandlingConfig(
        interp_method="linear",
        roi_nan_ratio_threshold=0.05,
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
    # Step 1: Prepare merged time series
    # ---------------------------------------------------------------------
    print("[1/3] Preparing merged time series...")
    outputs = prepare_merged_timeseries(data_dir=data_dir, nan_cfg=nan_cfg)
    labels_path = outputs["labels_path"]
    metadata_paths_csv = outputs["metadata_paths_csv"]
    print("  labels:", labels_path)
    print("  metadata:", metadata_paths_csv)

    # ---------------------------------------------------------------------
    # Step 2: Compute average connectivity (and save subject matrices if configured)
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
    print("[3/3] Plotting and saving example subject's connectivity matrix...")
    example_matrix_path = data_dir / f"{example_subject_id}" / "i2" / f"{example_subject_id}_connectivity.npy"
    if example_matrix_path.exists():
        example_matrix = np.load(example_matrix_path)
        out_png_example = vis_dir / f"{cohort_type}_{example_subject_id}_func_conn_matrix_Schaefer7n1000p_TianSubcortexS4.png"
        plot_connectivity_matrix(
            avg_connectivity=example_matrix,
            title=None,  # No title for subject matrix
            labels_path=None,  # No labels for subject matrix
            out_path=out_png_example,
        )
        print("  saved:", out_png_example)
    else:
        print(f"  Warning: Example subject matrix not found at {example_matrix_path}. Skipping example plot.")
    
    print("[3/3] Plotting and saving average matrix...")
    out_png = vis_dir / f"{cohort_type}_average_func_conn_matrix_Schaefer7n1000p_TianSubcortexS4.png"
    plot_connectivity_matrix(
        avg_connectivity=avg_connectivity,
        title="Average Functional Connectivity Matrix\n(Schaefer7n1000p + TianSubcortexS4)",
        labels_path=labels_path,
        out_path=out_png,
    )
    print("  saved:", out_png)


if __name__ == "__main__":
    main()
