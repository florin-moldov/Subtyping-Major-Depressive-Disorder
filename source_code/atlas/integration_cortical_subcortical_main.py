"""Integrate cortical and subcortical atlases into a single labelled atlas.

This module provides a small, robust utility to merge two integer-labelled
NIfTI atlases (cortical and subcortical) into one combined atlas. It produces
a combined NIfTI file and an optional CSV label-mapping table that records
where each new label came from.

Pipeline Steps
--------------
1. Load both NIfTI atlases (``load_nifti``) — rounds float values to the
   nearest integer to correct for any floating-point resampling artefacts.
2. Optionally resample the subcortical atlas to the cortical atlas grid using
   nearest-neighbour interpolation (``integrate_atlases`` with
   ``resample=True``). Requires ``nibabel >= 3`` (preferred) or ``nilearn``.
3. Compute per-atlas unique non-zero labels and decide a label offset so that
   cortical and subcortical label integers never collide in the combined volume.
4. Remap each atlas's labels into the collision-free space and merge the two
   remapped arrays. Where a voxel is non-zero in both atlases, the atlas
   specified by ``cortex_precedence`` wins.
5. Optionally load human-readable label names from JSON or delimited text
   files (``_load_label_names``) for both atlases.
6. Save the combined array as an int32 NIfTI (``save_nifti``) and, if
   requested, write a CSV mapping table
   ``(new_label, source, original_label, original_name)``.

API
----------
- ``load_nifti(path)``          — Load a NIfTI, return (data, affine, header).
- ``save_nifti(...)``           — Save a numpy array as a NIfTI file.
- ``integrate_atlases(...)``    — Main merge function; usable as a library call.

CLI Usage
---------
    python integration_cortical_subcortical_gpt5.py \\
        --cortical  path/to/cortex.nii.gz \\
        --subcortical path/to/subcortex.nii.gz \\
        --out combined_atlas.nii.gz \\
        [--labels combined_labels.csv] \\
        [--resample] \\
        [--cortex-labels cortex_labels.txt] [--cortex-label-columns 2] \\
        [--subcortex-labels subcortex_labels.txt] [--subcortex-label-columns 2-3] \\
        [--cortex-precedence | --subcortex-precedence] \\
        [--debug]

Inputs
------
- Cortical NIfTI  : integer-labelled 3-D image in a common reference space.
- Subcortical NIfTI: integer-labelled 3-D image; must share shape/affine with
  the cortical image unless ``--resample`` is passed.
- Optional label files: JSON (``{label: name}``) or delimited text files
  (TSV / CSV / TXT) containing label-to-name mappings for either atlas.

Outputs
-------
- Combined NIfTI  : int32, same affine/header as the cortical atlas.
- Optional CSV    : columns ``new_label``, ``source`` (``'cortical'`` or
  ``'subcortical'``), ``original_label``, ``original_name``.
- Function return : ``dict`` mapping ``new_label -> (source, original_label,
  original_name)``.

Edge Cases Handled
------------------
- Shape / affine mismatch: raises ``ValueError`` unless ``resample=True``.
- Zero voxels are treated as background throughout and are never remapped.
- Label collisions: the non-precedent atlas is offset by the maximum label of
  the precedent atlas so integers never overlap.
- Missing resampling backend: raises ``RuntimeError`` with install hint.
- Debug mode (``--debug``): logs per-atlas label counts, voxel counts, and
  sample mapping entries via the ``logging`` module.

Optional Dependencies
---------------------
- ``nibabel >= 3`` for ``nibabel.processing.resample_from_to`` (preferred).
- ``nilearn`` for ``nilearn.image.resample_to_img`` (fallback).
  If neither is installed, resampling is unavailable but the script runs
  normally when the two atlases already share a grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Dict, Tuple, Optional

import nibabel as nib
import numpy as np

# Resampling: prefer nibabel.processing.resample_from_to; fall back to nilearn if available
try:
    from nibabel.processing import resample_from_to as _nib_resample
except Exception:
    _nib_resample = None

try:
    from nilearn.image import resample_to_img as _nilearn_resample
except Exception:
    _nilearn_resample = None


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a NIfTI file and return its integer data, affine, and header.

    The image data are loaded as float32, rounded to the nearest integer
    (``numpy.rint``), and cast to ``int64``. This avoids tiny floating-point
    artefacts that can be introduced by resampling (e.g. a label stored as
    4.999999 is correctly recovered as 5).

    Parameters
    ----------
    path : str
        Absolute or relative path to a NIfTI file (.nii or .nii.gz).

    Returns
    -------
    data : numpy.ndarray, dtype int64
        3-D array of integer label values.
    affine : numpy.ndarray, shape (4, 4)
        Voxel-to-world affine matrix from the NIfTI header.
    header : nibabel.Nifti1Header
        Full NIfTI header object (used to preserve metadata when saving).
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    # round then cast to int to avoid tiny floating point resampling artifacts
    return np.rint(data).astype(np.int64), img.affine, img.header


def save_nifti(data: np.ndarray, affine: np.ndarray, out_path: str, header=None) -> None:
    """Save a numpy array as a NIfTI file with int32 data type.

    The array is cast to ``int32`` before writing. If a header is supplied it
    is passed through to the new ``Nifti1Image``, preserving metadata such as
    voxel dimensions and orientation codes from the source atlas.

    Parameters
    ----------
    data : numpy.ndarray
        3-D array of label values to save.
    affine : numpy.ndarray, shape (4, 4)
        Voxel-to-world affine matrix to embed in the NIfTI file.
    out_path : str
        Destination file path (.nii or .nii.gz). Parent directory must exist.
    header : nibabel.Nifti1Header or None, optional
        Existing NIfTI header to copy metadata from. If ``None`` nibabel
        generates a default header.

    Returns
    -------
    None
        File is written to ``out_path`` as a side effect.
    """
    nii = nib.Nifti1Image(data.astype(np.int32), affine, header=header)
    nib.save(nii, out_path)


def _parse_columns(columns: Optional[str]) -> Optional[list]:
    """Parse a 1-based column selection string into a list of 0-based column indices.

    Accepts a comma-separated list of individual column numbers and/or inclusive
    ranges joined with a hyphen, for example:

    - ``'2'``      → ``[1]``
    - ``'2,4'``    → ``[1, 3]``
    - ``'2-4'``    → ``[1, 2, 3]``
    - ``'2,4-6'``  → ``[1, 3, 4, 5]``

    Column numbers are 1-based (matching spreadsheet / cut convention);
    they are converted to 0-based indices for direct use with Python lists.
    Invalid tokens and empty parts are silently skipped.

    Parameters
    ----------
    columns : str or None
        Column selection string, e.g. ``'2'``, ``'2,3'``, or ``'2-4'``.

    Returns
    -------
    list of int or None
        Sorted 0-based column indices, or ``None`` if ``columns`` is falsy or
        no valid tokens are found.
    """
    if not columns:
        return None
    cols = []
    for part in str(columns).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                a_i = int(a) - 1
                b_i = int(b) - 1
            except ValueError:
                continue
            cols.extend(list(range(a_i, b_i + 1)))
        else:
            try:
                cols.append(int(part) - 1)
            except ValueError:
                continue
    return cols if cols else None


def _load_label_names(path: Optional[str], columns: Optional[str] = None) -> Dict[int, str]:
    """Load a label-to-name mapping from a JSON or delimited text file.

    Supports two file formats:

    **JSON** (``.json`` extension)
        The file must be a flat object whose keys are integer labels (as
        strings) and values are the corresponding names, e.g.
        ``{"1": "LH_Vis_1", "2": "LH_Vis_2", ...}``.

    **Delimited text** (TSV / CSV / TXT or any whitespace-separated file)
        Each non-empty, non-comment line is parsed as follows:

        - If the first token is an integer it is used as the label ID and the
          name is taken from subsequent columns.
        - If the first token is *not* an integer (e.g. a bare region name), the
          1-based line number is used as the label ID. This matches the simple
          plain-text label files shipped with Tian/Schaefer parcellations.

        The ``columns`` parameter (1-based) controls which columns are joined
        with ``' | '`` to form the name string. If ``columns`` is ``None``,
        all columns after the ID column are used.

    Lines starting with ``#`` and blank lines are ignored. The delimiter is
    auto-detected: tab → comma → whitespace.

    Parameters
    ----------
    path : str or None
        Path to the label file. Returns an empty dict immediately if ``None``
        or an empty string.
    columns : str or None, optional
        1-based column selection string parsed by ``_parse_columns``
        (e.g. ``'2'``, ``'2,3'``, ``'2-4'``). Applied only to text files;
        ignored for JSON files.

    Returns
    -------
    dict mapping int -> str
        ``{label_integer: name_string}``.
        Returns an empty dict if ``path`` is falsy.

    Raises
    ------
    FileNotFoundError
        If ``path`` is a non-empty string that does not exist on disk.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {int(k): str(v) for k, v in data.items()}

    mapping: Dict[int, str] = {}
    cols = _parse_columns(columns)
    
    with open(path, "r", encoding="utf-8") as fh:
        line_idx = 0
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            line_idx += 1
            
            # Detect delimiter
            if "\t" in line:
                fields = line.split("\t")
            elif "," in line:
                fields = line.split(",")
            else:
                fields = line.split() # Splits on any whitespace
            
            if len(fields) < 1:
                continue

            # Try to parse the first column as an ID
            is_explicit_id = False
            try:
                lab = int(fields[0].strip())
                is_explicit_id = True
            except ValueError:
                # First column is not an integer. 
                # Assume implicit numbering (Line 1 = Label 1)
                lab = line_idx
            
            # Extract Name
            if is_explicit_id:
                # If first col was ID, name is in subsequent columns
                if cols is None:
                    name_parts = fields[1:]
                else:
                    name_parts = []
                    for ci in cols:
                        if 0 <= ci < len(fields) and ci != 0:
                            name_parts.append(fields[ci].strip())
            else:
                # If first col was NOT ID, the whole line (or specific cols) is the name
                if cols is None:
                    name_parts = fields
                else:
                    name_parts = []
                    for ci in cols:
                        if 0 <= ci < len(fields):
                            name_parts.append(fields[ci].strip())

            name = " | ".join([p for p in name_parts if p])
            # Fallback if parsing failed but we have a label
            if not name and not is_explicit_id:
                name = line
                
            mapping[lab] = name
            
    return mapping


def integrate_atlases(cortical_path: str,
                      subcortical_path: str,
                      out_nifti: str,
                      out_labels_csv: Optional[str] = None,
                      cortex_precedence: bool = True,
                      resample: bool = False,
                      cortex_labels_path: Optional[str] = None,
                      subcortex_labels_path: Optional[str] = None,
                      cortex_label_columns: Optional[str] = None,
                      subcortex_label_columns: Optional[str] = None) -> Dict[int, Tuple[str, int, Optional[str]]]:
    """Merge cortical and subcortical integer-labelled NIfTI atlases into one.

    Algorithm
    ---------
    1. Load both images; optionally resample the subcortical atlas to the
       cortical atlas grid (nearest-neighbour, preserving discrete labels).
    2. Compute the maximum non-zero label of the *precedent* atlas, then add
       that value as an offset to all non-zero labels of the *non-precedent*
       atlas. This guarantees no integer collision between the two label sets.
    3. Build ``combined``: start from zeros, write the precedent atlas labels
       first, then fill remaining background voxels with the non-precedent
       atlas labels.
    4. Load optional human-readable names via ``_load_label_names`` for both
       atlases and build the return mapping.
    5. Save the combined array as ``int32`` NIfTI (borrowing the cortical
       affine and header) and, if requested, write a CSV label table.

    Parameters
    ----------
    cortical_path : str
        Path to the cortical parcellation NIfTI (integer labels, 3-D).
    subcortical_path : str
        Path to the subcortical parcellation NIfTI (integer labels, 3-D).
    out_nifti : str
        Destination path for the combined atlas NIfTI (.nii or .nii.gz).
    out_labels_csv : str or None, optional
        If provided, write a CSV table with columns
        ``new_label, source, original_label, original_name``.
    cortex_precedence : bool, optional
        If ``True`` (default), cortical labels take priority where both
        atlases cover the same voxel; subcortical labels are offset by the
        maximum cortical label. If ``False``, the roles are reversed.
    resample : bool, optional
        If ``True``, resample the subcortical atlas to the cortical atlas
        grid when shapes or affines differ (nearest-neighbour interpolation).
        Requires ``nibabel >= 3`` or ``nilearn``. Default ``False``.
    cortex_labels_path : str or None, optional
        Path to a label-name file for the cortical atlas (JSON or text).
        Passed to ``_load_label_names``.
    subcortex_labels_path : str or None, optional
        Path to a label-name file for the subcortical atlas (JSON or text).
        Passed to ``_load_label_names``.
    cortex_label_columns : str or None, optional
        1-based column selector for the cortical label text file
        (e.g. ``'2'`` or ``'2-3'``). Passed to ``_load_label_names``.
    subcortex_label_columns : str or None, optional
        1-based column selector for the subcortical label text file.
        Passed to ``_load_label_names``.

    Returns
    -------
    dict mapping int -> tuple(str, int, str or None)
        ``{new_label: (source, original_label, original_name)}`` where
        *source* is ``'cortical'`` or ``'subcortical'``, *original_label* is
        the integer label in the source atlas, and *original_name* is the
        human-readable name from the label file (or ``None`` if not provided).

    Raises
    ------
    ValueError
        If the two atlases differ in shape or affine and ``resample=False``.
    RuntimeError
        If resampling is requested but no compatible backend is installed.
    """
    cort_img = nib.load(cortical_path)
    sub_img = nib.load(subcortical_path)

    # If needed, resample sub to cortical grid
    if (cort_img.shape != sub_img.shape) or (not np.allclose(cort_img.affine, sub_img.affine)):
        if not resample:
            raise ValueError(
                f"Shape/affine mismatch: cortical {cort_img.shape}/{cort_img.affine} vs "
                f"subcortical {sub_img.shape}/{sub_img.affine}. Set resample=True to resample subcortical to cortical grid."
            )
        # perform resampling (nearest neighbor)
        logging.info("Resampling subcortical atlas to cortical atlas grid (nearest neighbor)...")
        if _nib_resample is not None:
            sub_img_rs = _nib_resample(sub_img, cort_img, order=0)
        elif _nilearn_resample is not None:
            sub_img_rs = _nilearn_resample(sub_img, cort_img, interpolation='nearest')
        else:
            raise RuntimeError("No resampling backend available: please install nibabel>=3 or nilearn to enable resampling.")
        sub_img = sub_img_rs

    # Use rounding before casting to integers to avoid floating point
    # artifacts introduced during resampling (e.g., 4.999999 -> 5)
    cort_data = np.rint(cort_img.get_fdata(dtype=np.float32)).astype(np.int64)
    sub_data = np.rint(sub_img.get_fdata(dtype=np.float32)).astype(np.int64)
    cort_affine = cort_img.affine
    cort_hdr = cort_img.header

    cort_labels = np.unique(cort_data)
    sub_labels = np.unique(sub_data)

    cort_nonzero = cort_labels[cort_labels != 0]
    sub_nonzero = sub_labels[sub_labels != 0]

    logging.info("Cortical non-zero labels: %s", cort_nonzero)
    logging.info("Subcortical non-zero labels: %s", sub_nonzero)

    # Decide label remapping to avoid accidental overlaps
    if cortex_precedence:
        # Keep cortical labels as-is; offset subcortical labels so they don't collide
        max_cort = int(cort_nonzero.max()) if cort_nonzero.size else 0
        offset = max_cort 
        sub_mapping = {int(lbl): int(lbl) + offset for lbl in sub_nonzero}
        cort_mapping = {int(lbl): int(lbl) for lbl in cort_nonzero}
    else:
        max_sub = int(sub_nonzero.max()) if sub_nonzero.size else 0
        offset = max_sub 
        cort_mapping = {int(lbl): int(lbl) + offset for lbl in cort_nonzero}
        sub_mapping = {int(lbl): int(lbl) for lbl in sub_nonzero}

    # Build remapped arrays
    remapped_cort = np.zeros_like(cort_data)
    remapped_sub = np.zeros_like(sub_data)
    for orig, new in cort_mapping.items():
        remapped_cort[cort_data == orig] = new
    for orig, new in sub_mapping.items():
        remapped_sub[sub_data == orig] = new

    # Merge according to precedence: where both non-zero, use chosen atlas
    combined = np.zeros_like(cort_data)
    if cortex_precedence:
        # cortical wins where cortical non-zero; else take sub
        mask_cort = remapped_cort != 0
        combined[mask_cort] = remapped_cort[mask_cort]
        mask_sub_only = (remapped_sub != 0) & (~mask_cort)
        combined[mask_sub_only] = remapped_sub[mask_sub_only]
    else:
        mask_sub = remapped_sub != 0
        combined[mask_sub] = remapped_sub[mask_sub]
        mask_cort_only = (remapped_cort != 0) & (~mask_sub)
        combined[mask_cort_only] = remapped_cort[mask_cort_only]

    # Diagnostic info (optionally printed)
    # We'll compute some statistics after remapping to help debug missing labels
    def _count_nonzero(arr: np.ndarray) -> int:
        return int(np.count_nonzero(arr))

    # Create mapping new_label -> (source, original_label)
    # Load label name mappings (if provided)
    cortex_label_names = _load_label_names(cortex_labels_path, columns=cortex_label_columns)
    subcortex_label_names = _load_label_names(subcortex_labels_path, columns=subcortex_label_columns)

    mapping: Dict[int, Tuple[str, int, Optional[str]]] = {}
    for orig, new in cort_mapping.items():
        mapping[new] = ("cortical", orig, cortex_label_names.get(orig))
    for orig, new in sub_mapping.items():
        # If a new label already present from cortical mapping it means we offset differently
        if new in mapping:
            # This shouldn't happen since we offset to avoid collisions; warn
            logging.warning("Label collision for new label %s (orig sub %s).", new, orig)
            # prefer existing mapping
            continue
        mapping[new] = ("subcortical", orig, subcortex_label_names.get(orig))

    # If debug requested, print diagnostics to help find why subcortical labels might be missing
    if hasattr(integrate_atlases, "__debug_enabled") and integrate_atlases.__debug_enabled:
        logging.info("DEBUG: cortical unique labels (nonzero): %d", len(cort_nonzero))
        logging.info("DEBUG: subcortical unique labels (nonzero): %d", len(sub_nonzero))
        logging.info("DEBUG: remapped_cort nonzero voxels: %d", _count_nonzero(remapped_cort))
        logging.info("DEBUG: remapped_sub nonzero voxels: %d", _count_nonzero(remapped_sub))
        logging.info("DEBUG: combined nonzero voxels: %d", _count_nonzero(combined))
        # show first few sub mappings
        sub_items = list(sub_mapping.items())[:20]
        logging.info("DEBUG: sample sub_mapping (orig->new): %s", sub_items)
        # show whether mapping contains subcortical entries
        sub_mapped_keys = [k for k, v in mapping.items() if v[0] == "subcortical"]
        logging.info("DEBUG: mapping contains %d subcortical new-label entries", len(sub_mapped_keys))
        logging.info("DEBUG: sample mapping entries (first 10): %s", list(mapping.items())[:10])

    # Save outputs
    save_nifti(combined, cort_affine, out_nifti, header=cort_hdr)
    logging.info("Saved combined atlas to %s", out_nifti)

    if out_labels_csv:
        with open(out_labels_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["new_label", "source", "original_label", "original_name"])
            for new_label in sorted(mapping.keys()):
                source, orig, name = mapping[new_label]
                writer.writerow([new_label, source, orig, name or ""])
        logging.info("Saved label mapping to %s", out_labels_csv)

    return mapping


def _build_argparser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for the integration script.

    Defines all command-line arguments accepted by the ``__main__`` entry
    point (input paths, output paths, resampling flag, label files, column
    selectors, precedence flags, and debug mode). The returned parser can
    also be used programmatically to validate arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser ready to call ``.parse_args()`` on.
    """
    p = argparse.ArgumentParser(description="Integrate cortical and subcortical atlases.")
    p.add_argument("--cortical", required=True, help="Path to cortical atlas NIfTI (integer labels)")
    p.add_argument("--subcortical", required=True, help="Path to subcortical atlas NIfTI (integer labels)")
    p.add_argument("--out", required=True, help="Output combined atlas NIfTI path")
    p.add_argument("--labels", required=False, help="Output CSV mapping file (new_label,source,original_label,original_name)")
    p.add_argument("--resample", action="store_true", default=False,
                   help="If set, resample the subcortical atlas to the cortical atlas grid when shapes/affines differ.")
    p.add_argument("--cortex-labels", required=False,
                   help="Optional label->name mapping file for cortical atlas. JSON, TSV, CSV, or TXT.")
    p.add_argument("--cortex-label-columns", required=False,
                   help="Optional columns to include from cortex label text file (1-based indices, e.g. '2' or '2,4' or '2-4').")
    p.add_argument("--subcortex-labels", required=False,
                   help="Optional label->name mapping file for subcortical atlas. JSON, TSV, CSV, or TXT.")
    p.add_argument("--subcortex-label-columns", required=False,
                   help="Optional columns to include from subcortex label text file (1-based indices, e.g. '2' or '2,4' or '2-4').")
    p.add_argument("--cortex-precedence", action="store_true", default=True,
                   help="If set, cortical labels take precedence where both atlases label the same voxel (default true).")
    p.add_argument("--subcortex-precedence", dest="cortex_precedence", action="store_false",
                   help="If set, subcortical labels take precedence where both atlases label the same voxel.")
    p.add_argument("--debug", action="store_true", default=False,
                   help="Enable debug logging and print diagnostics about remapping and label counts.")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_argparser()
    args = parser.parse_args()

    # Wire debug flag into integrate_atlases for diagnostic printing
    if args.debug:
        integrate_atlases.__debug_enabled = True
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.cortical):
        raise SystemExit(f"Cortical atlas not found: {args.cortical}")
    if not os.path.exists(args.subcortical):
        raise SystemExit(f"Subcortical atlas not found: {args.subcortical}")

    mapping = integrate_atlases(
        args.cortical,
        args.subcortical,
        args.out,
        out_labels_csv=args.labels,
        cortex_precedence=bool(args.cortex_precedence),
        resample=bool(args.resample),
        cortex_labels_path=args.cortex_labels,
        subcortex_labels_path=args.subcortex_labels,
        cortex_label_columns=args.cortex_label_columns,
        subcortex_label_columns=args.subcortex_label_columns,
    )

    logging.info("Integration finished. %d labels in combined atlas.", len(mapping))