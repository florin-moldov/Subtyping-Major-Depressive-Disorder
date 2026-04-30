"""Compute and plot the average structural connectome for a cohort.
Primary workflow
----------------
1) List subject IDs in a cohort directory (optionally excluding subjects)
2) Load each subject's structural connectome matrix (CSV/CSV.GZ)
3) Skip connectome issues tracking:
   - missing file per subject
   - NaNs in a subject matrix
4) Compute the arithmetic mean across successfully included subjects
5) Save the average matrix and small QC reports
6) Visualize example subject's structural connectome alongside the average matrix with nilearn's matrix plot
7) Exclude subjects with missing functional and structural connectomes from depression and combined cohort files

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from avg_strct_utils import (
    compute_average_structural_connectivity,
    list_subject_ids,
    load_excluded_subject_ids,
    load_labels_txt,
    normalize_for_plot,
    plot_connectivity_matrix,
    exclude_subjects_by_eid,
)


def main() -> None:
    # ---------------------------------------------------------------------
    # Configuration (edit and run)
    # ---------------------------------------------------------------------
    project_root = Path(".../subtyping_depression")

    cohort_type = "F32"  # 'F32' or 'control'

    data_dir = project_root / "data" / "UKB" / f"{cohort_type}_notask_STRCO_RSSCHA_RSTIA"
    combined_cohort_file = project_root / "data" / "UKB" / "cohorts" / "combined_cohort_F32.csv"
    vis_dir = project_root / "reports" / "figures" / "schaefer1000+tian54" / "structural_con"

    example_subject_id = "..."  # Change to a valid subject ID from your cohort to visualize their connectivity matrix

    labels_path = data_dir / "Schaefer7n1000p_TianSubcortexS4_labels.txt"

    connectome_subdir = "i2"
    connectome_template = "connectome_streamline_count_10M.csv.gz"

    show_progress = True
    log_transform = True

    # ---------------------------------------------------------------------
    # Step 1: Load labels and subject IDs
    # ---------------------------------------------------------------------
    labels = load_labels_txt(labels_path)

    labels = load_labels_txt(labels_path)
    excluded_subjects_csv = data_dir / "missing_subjects_resting_state_timeseries.csv"
    excluded_subjects_column = "subject_id"

    subject_ids = list_subject_ids(data_dir)
    print(f"Found {len(subject_ids)} subjects.")
    print(subject_ids[:5])

    if excluded_subjects_csv.exists():
        excluded = set(load_excluded_subject_ids(excluded_subjects_csv, column=excluded_subjects_column))
        subject_ids = [sid for sid in subject_ids if sid not in excluded]
        print(f"After exclusion, {len(subject_ids)} subjects remain.")
        subject_ids.sort()
        print(subject_ids[:5])

    if not subject_ids:
        raise ValueError(f"No subject directories to process under {data_dir}")
    
    # ---------------------------------------------------------------------
    # Step 2: Compute average structural connectivity matrix (while checking
    #         for missing/NaN subjects)
    # ---------------------------------------------------------------------

    result = compute_average_structural_connectivity(
        data_dir=data_dir,
        subject_ids=subject_ids,
        subdir=connectome_subdir,
        filename_template=connectome_template,
        skip_nan_subjects=True,
        show_progress=show_progress,
        )

    print(f"Average connectivity matrix shape: {result.avg_matrix.shape}")

    out_nans_csv = data_dir / f"structural_connectivity_matrices_streamline_count_nans.csv"
    out_missing_csv = data_dir / "missing_subjects_structural_connectomes.csv"
    out_avg_npy = data_dir / f"average_structural_connectivity_matrix_streamline_count_10M.npy"

    result.nan_subjects.to_csv(out_nans_csv, index=False)
    pd.DataFrame({"subject_id": result.missing_subjects}).to_csv(out_missing_csv, index=False)
    np.save(out_avg_npy, result.avg_matrix)

    print(f"Saved NaN report to {out_nans_csv}.")
    print(f"Saved missing subjects report to {out_missing_csv}.")
    print(f"Saved average structural connectivity matrix to {out_avg_npy}.")

    # ---------------------------------------------------------------------
    # Step 3: Visualize example subject's structural connectivity matrix
    # and the average structural connectivity matrix
    # ---------------------------------------------------------------------
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("[3/3] Plotting and saving example subject's structural connectivity matrix...")
    example_matrix_path = data_dir / f"{example_subject_id}" / connectome_subdir / connectome_template
    if example_matrix_path.exists():
        example_matrix = pd.read_csv(example_matrix_path, compression="infer", header=None).to_numpy()
        plot_example_matrix = normalize_for_plot(example_matrix, log_transform=log_transform)
        out_png_example = vis_dir / f"{cohort_type}_{example_subject_id}_struct_conn_matrix_streamline_count_Schaefer7n1000p_TianSubcortexS4.png"
        plot_connectivity_matrix(
            mat = plot_example_matrix,
            title=None,  # No title for subject matrix
            labels=None,  # No labels for subject matrix
            out_path=out_png_example,
            cmap="Reds",
        )
        print(f"Saved visualization of example subject's structural connectivity matrix to {out_png_example}.")
    else:
        print(f"  Warning: Example subject's structural connectome not found at {example_matrix_path}. Skipping example plot.")
    
    print("[3/3] Plotting and saving average structural connectivity matrix...")
    plot_avg_mat = normalize_for_plot(result.avg_matrix, log_transform=log_transform)

    out_fig = vis_dir / f"{cohort_type}_average_structural_connectivity_matrix_streamline_count_10M.png"

    plot_connectivity_matrix(
        mat = plot_avg_mat,
        labels=labels,
        title="Average Structural Connectivity Matrix",
        cmap="Reds",
        figure=(20, 20),
        reorder=False,
        out_path=out_fig,
        dpi=300,
    )

    print(f"Saved visualization of average structural connectivity matrix to {out_fig}.")

    # ---------------------------------------------------------------------
    # Step 4: Exclude subjects with missing functional or structural data
    # ---------------------------------------------------------------------
    missing_subjects_functional = pd.read_csv(data_dir / f"missing_subjects_resting_state_timeseries.csv")
    missing_subjects_structural = pd.read_csv(data_dir / f"missing_subjects_structural_connectomes.csv")

    missing_subjects = pd.concat([missing_subjects_functional, missing_subjects_structural]).drop_duplicates().reset_index(drop=True)

    if cohort_type == "F32":
        depression_cohort_file = project_root / "data" / "UKB" / "cohorts" / "depression_cohort_F32.csv"
        exclude_subjects_by_eid(depression_cohort_file, missing_subjects, eid_col="eid", 
                            source_eid_col="subject_id", out_path=depression_cohort_file, inplace=True, return_count=False)

    exclude_subjects_by_eid(combined_cohort_file, missing_subjects, eid_col="eid", 
                        source_eid_col="subject_id", out_path=combined_cohort_file, inplace=True, return_count=False)

if __name__ == "__main__":
    main()
