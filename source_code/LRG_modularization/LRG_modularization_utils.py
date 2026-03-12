"""
Utility functions for LRG modularization and module visualization.
Provides:
- entropy(G, steps) -> (S, dS, VarL, Len, t)
- plot_entropy_network(network, title=None)
- compute_dendrogram(G, tau, distance_metric) -> linkage_matrix, dists
- plot_dendrogram(linkage_matrix, Th, xlabel="Distance", ylabel="Nodes", figsize=(3,4), title=None)
- level_dictionary(T, lvl) -> dict of module labels to ROI indices
- reorder_matrix_by_modules(matrix, CM) -> matrix_reordered, sorted_indices
- _basename_if_path(x) -> str (helper to robustly handle subject_id entries that may be bare ids or paths)
- individual_reordering_by_modules(data_dir, CM, missing_subjects_file, connectivity_type)
- get_module_colors(n_modules, alpha=0.7) -> array of RGBA colors
- plot_all_communities_figure(conn_matrix, community_labels, atlas_img_path, output_path, output_dir=None, alpha=0.85, background_white=True, resample_on_mismatch=False, reference_img_path=None, resample_interpolation='nearest', save_niftis=True, nifti_dir=None, nifti_suffix='.nii.gz')
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.linalg import expm
import networkx as nx
from matplotlib.colors import ListedColormap
from nilearn import plotting
import warnings
from tqdm import tqdm
import nibabel as nib
from nilearn import image

def entropy(G,steps):
    w=nx.laplacian_spectrum(G)
    t1=np.log10(1/np.max(w[w>1e-10]))-1
    t2=np.log10(10/np.min(w[w>1e-10]))+1
    t = np.logspace(t1,t2, int(steps))
    cont=0
    S=np.zeros(len(t))
    VarL=np.zeros(len(t))
    N=len(G.nodes())
    
    L=nx.laplacian_matrix(G)
    L1=L.todense()    
    Len=np.zeros(len(t))

    
    for tau in tqdm(t):
        Tr=np.nansum(np.exp(-tau*w))
        T1=np.divide(np.exp(-w*tau),Tr)
        S[cont]=-np.nansum(T1*np.log(T1))/np.log(N)
        Med=np.nansum(np.multiply(w,np.exp(-tau*w)))/Tr
        Sqr=np.nansum(np.multiply(np.multiply(w,w),np.exp(-tau*w)))/Tr
        VarL[cont]=(Sqr-Med*Med)
        cont=cont+1
        
    dS=np.log(N)*np.diff(1-S)/np.diff(np.log(t))
    return 1-S,dS,VarL,Len, t

def plot_entropy_network(network, title): #filename,title=None):
    # Crear la figura y los ejes usando subplots
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Paletas de colores
    pal = sns.color_palette('Reds', n_colors=3, desat=1.0)
    pal1 = sns.color_palette('Blues', n_colors=3, desat=1.0)

    # Crear un segundo eje (twinx) para la gráfica secundaria
    ax2 = ax1.twinx()

    # Entropía para el 'network'
    # Supongo que la función "entropy" está definida previamente
    [S, dS, VarL, Len, t] = entropy(network, 1e3)

    # Definir t1 como los puntos medios del tiempo t
    t1 = (t[1:] + t[:-1]) / 2.0

    # Graficar los valores de S en el primer eje
    ax1.plot(t, S, marker='None', lw=2, ls='--', color=pal.pop(0), alpha=0.8, label=r'$1-S$')

    # Graficar los valores de dS en el segundo eje
    ax2.plot(t1, dS, marker='None', ls='-', lw=2, color=pal1.pop(0), zorder=-15, alpha=1.0)

    # Personalizar las leyendas y las líneas horizontales
    ax2.legend(loc='upper right')
    ax2.axhline(y=0.5, ls='--', color='black', lw=1.5)

    # Ajustes de escala y límites en los ejes
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$1-S$')
    ax2.set_ylabel(r'$C~\log N$')
    ax1.set_xlabel(r'$\tau$')
    #ax1.set_xlim(1e-6, 1e3)
    #ax1.set_ylim(-0.01, 1.05)
    #ax2.set_ylim(-0.01, 4)

    if title:
        ax1.set_title(title)

    # Guardar la gráfica
    #plt.savefig(f'{filename}')
    plt.show()


def compute_dendrogram(G, tau, distance_metric):
    L=nx.laplacian_matrix(G)
    L1=L.todense().astype(np.float64)
    #debug_matrix(L1, "Laplacian")

    num=expm((-tau*L1))
    #debug_matrix(num, "Exponential")

    den=np.trace(num)
    # print(f"\nTrace of expm(-tau*L): {den}")
    # Evitar división por cero
    if np.isclose(den, 0):
        raise ValueError("Trace is too close to zero, potential numerical instability.")
    
    rho=num/den
    #debug_matrix(rho, "Rho")
    # Evitar valores pequeños en rho antes de invertir
    epsilon = 1e-12
    rho[rho < epsilon] = epsilon  # Evitar divisiones por cero

    Trho=np.copy(1.0/rho)#1/adj2
    Trho = np.maximum(Trho, Trho.transpose() )
    #debug_matrix(Trho, "Trho (1/rho)")

    np.fill_diagonal(Trho, 0)
    #debug_matrix(Trho, "Trho with diagonal zeroed")
    # Paso 7: Conversión a distancias con squareform
    dists = squareform(Trho)
    linkage_matrix = linkage(dists, distance_metric) #try "average" and "ward"
    # print("\nLinkage matrix successfully computed.")
    tmax=linkage_matrix[::, 2][-1]#+0.01*linkage_matrix[::, 2][-1]
    linkage_matrix = linkage(dists/tmax,distance_metric)
    # print("\nLinkage matrix normalized successfully computed.")
    
    return linkage_matrix, dists




def plot_dendrogram(linkage_matrix, Th, xlabel="Distance", ylabel="Nodes", figsize=(3,4), title=None):
    
    """
    Dibuja un dendrograma horizontal coloreado por clusters definidos por 'maxclust',
    marcando el umbral correspondiente con una línea roja.
    
    Parámetros:
    - linkage_matrix: matriz de linkage (output de scipy.cluster.hierarchy.linkage)
    - Th: número deseado de clusters
    - xlabel, ylabel: etiquetas de los ejes
    - figsize: tamaño de la figura
    """

    # Función interna para encontrar el valor de distancia que da exactamente Th clusters
    for t in np.linspace(0, linkage_matrix[-1, 2], 500):
        n_cuts = len(np.unique(fcluster(linkage_matrix, t=t, criterion='distance')))
        if n_cuts == Th:
            cut_threshold_distance = t
            break
    # Plot
    ax_dict = plt.figure(constrained_layout=True, figsize=figsize).subplot_mosaic(
        """
         A
        """
    )

    dendrogram(linkage_matrix, ax=ax_dict['A'], orientation='right',
               color_threshold=cut_threshold_distance, above_threshold_color='k',
               leaf_rotation=0, leaf_font_size=1)
    #ax_dict['A'].axhline(y=cut_threshold_distance, color='#ED2939', linestyle='--')
    ax_dict['A'].axvline(x=cut_threshold_distance, color='#ED2939', linestyle='--')
    ax_dict['A'].set_xlabel(xlabel)
    ax_dict['A'].set_ylabel(ylabel)

    if title:
        ax_dict['A'].set_title(title)

    plt.savefig(f'dendrogram_Th_{Th}.png')

    plt.show()


def level_dictionary(T, lvl):
    l_dict = {}
    for i in range(1, lvl + 1):
        rois_in_clust = np.where(T == i)[0]
        if len(rois_in_clust) != 0:
            desc = "lvl_" + str(lvl) + "_mod_" + str(i)
            l_dict[desc] = rois_in_clust.tolist()
        else:
            warnings.warn(
                "Empty cluster found in level " + str(lvl) + ", module " + str(i) + "!"
            )
    return l_dict


def reorder_matrix_by_modules(matrix, CM):
    # Reordenar la matriz original SC según la partición en módulos
    sorted_indices = np.argsort(CM)  # Obtener los índices ordenados por módulo
    matrix_reordered = matrix[np.ix_(sorted_indices, sorted_indices)]  # Reordenar filas y columnas
    return matrix_reordered, sorted_indices


def _basename_if_path(x):
    """Return basename for a path-like value or the stripped string for simple ids."""
    return os.path.basename(str(x)).strip()


def individual_reordering_by_modules(data_dir, CM, missing_subjects_file, connectivity_type):
    missing = pd.read_csv(missing_subjects_file)
    missing_ids = missing['subject_id'].tolist()

    # normalize missing ids once for fast membership tests
    _missing_ids_set = {str(x).strip() for x in missing_ids}

    # robustly filter subject_ids: handle both bare ids and path-like entries,
    # preserve the original subject_id entry (path or id) if its basename is not missing
    _filtered_subject_ids = []

    subject_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for sid in subject_ids:
        if _basename_if_path(sid) not in _missing_ids_set:
            _filtered_subject_ids.append(sid)

    # optional: keep a deterministic order (uncomment if desired)
    _filtered_subject_ids.sort()

    subject_ids = _filtered_subject_ids

    for subject_id in subject_ids:
        subject_dir = os.path.join(data_dir, subject_id, 'i2')
        print(f"Processing subject {subject_id}...")

        # Load individual connectivity matrix
        if connectivity_type == 'structural':
            conn_matrix = pd.read_csv(os.path.join(subject_dir, 'connectome_streamline_count_10M.csv.gz'), compression="infer", header=None).to_numpy()
        else:  # functional
            conn_matrix = np.load(os.path.join(subject_dir, f'{subject_id}_connectivity.npy'))

        # Reorder matrix
        conn_matrix_reordered, sorted_indices = reorder_matrix_by_modules(conn_matrix, CM)

        # Save reordered matrix
        np.save(os.path.join(subject_dir, f'{subject_id}_{connectivity_type}_connectivity_matrix_reordered.npy'), conn_matrix_reordered)
        print(f"Saved reordered connectivity matrix for subject {subject_id}.")

def get_module_colors(n_modules, alpha=0.7):
    """
    Generate perceptually optimized colors for brain module visualization.
    Uses a curated palette of distinct, high-contrast colors that work well
    on both white and dark backgrounds and are reasonably colorblind-friendly.
    """
    # Curated palette: distinct hues with good saturation and brightness
    base_colors = [
        (0.90, 0.20, 0.20),  # Red
        (0.20, 0.60, 0.90),  # Blue
        (0.20, 0.80, 0.40),  # Green
        (0.95, 0.60, 0.10),  # Orange
        (0.60, 0.30, 0.80),  # Purple
        (0.95, 0.85, 0.20),  # Yellow
        (0.40, 0.80, 0.85),  # Cyan
        (0.90, 0.45, 0.70),  # Pink
        (0.55, 0.35, 0.20),  # Brown
        (0.50, 0.80, 0.30),  # Lime
        (0.30, 0.45, 0.70),  # Steel blue
        (0.85, 0.55, 0.55),  # Salmon
    ]
    # Extend with tab20 if more colors needed
    if n_modules > len(base_colors):
        extra = plt.cm.tab20(np.linspace(0, 1, n_modules - len(base_colors)))
        base_colors.extend([tuple(c[:3]) for c in extra])
    colors = np.array([(*base_colors[i % len(base_colors)], alpha) for i in range(n_modules)])
    return colors

def plot_all_communities_figure(conn_matrix: np.ndarray,
                                community_labels: np.ndarray,
                                atlas_img_path: str,
                                output_path: str,
                                output_dir: str = None,
                                alpha: float = 0.85,
                                background_white: bool = True,
                                resample_on_mismatch: bool = False,
                                reference_img_path: str = None,
                                resample_interpolation: str = 'nearest',
                                save_niftis: bool = True,
                                nifti_dir: str | None = None,
                                nifti_suffix: str = '.nii.gz'):
    """Plot all individual community glass brains plus overlay in a single large figure.
    
    Also saves individual figures for each module and the overlay if output_dir is provided.

    Parameters:
        - conn_matrix: NxN connectivity matrix (used only for shape validation)
        - community_labels: length N array of module labels for each ROI
        - atlas_img_path: path to atlas NIfTI file with integer region codes corresponding to ROIs in conn_matrix
        - output_path: path to save the combined figure
        - output_dir: directory to save individual module figures (optional)
        - alpha: transparency for module overlays
        - background_white: whether to use a white background for the figures
        - resample_on_mismatch: whether to resample the atlas to a reference image if there is a mismatch
        - reference_img_path: path to reference image for resampling
        - resample_interpolation: interpolation method for resampling
        - save_niftis: whether to save individual module NIfTI files
        - nifti_dir: directory to save NIfTI files (optional)
        - nifti_suffix: suffix for NIfTI files (default '.nii.gz')
    """
    import os
    from matplotlib.gridspec import GridSpec
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    atlas_img = image.load_img(atlas_img_path)
    atlas_data = atlas_img.get_fdata()
    labels = np.asarray(community_labels, dtype=int)
    N = conn_matrix.shape[0]

    # Validate shapes and basic assumptions
    if labels.size != N:
        raise ValueError(f"community_labels length ({labels.size}) does not match connectivity matrix size ({N}).\n"
                         "Ensure you provide one label per ROI and that ordering matches the connectivity matrix.")

    # Identify unique positive region codes in the atlas
    region_codes = np.unique(atlas_data)
    region_codes = region_codes[region_codes > 0]

    if region_codes.size != N:
        # Optionally attempt resampling to a provided reference image to resolve simple grid/space mismatches
        if resample_on_mismatch and reference_img_path is not None:
            ref_img = image.load_img(reference_img_path)
            try:
                atlas_img_rs = image.resample_to_img(atlas_img, ref_img, interpolation=resample_interpolation)
            except Exception as e:
                raise RuntimeError(f"Atlas resampling failed: {e}") from e
            atlas_img = atlas_img_rs
            atlas_data = atlas_img.get_fdata()
            region_codes = np.unique(atlas_data)
            region_codes = region_codes[region_codes > 0]
            if region_codes.size != N:
                raise ValueError(
                    f"After resampling to reference image, atlas unique region count ({region_codes.size}) still != connectivity matrix size ({N}).\n"
                    "Automatic resampling could not resolve the mismatch. Provide a matching atlas or a label->ROI mapping."
                )
            else:
                print(f"[INFO] Atlas resampled to '{reference_img_path}' and label count now matches ({N}).")
        else:
            # Do not silently pad/truncate region codes; surface a helpful error with next steps
            raise ValueError(
                f"Atlas unique region count ({region_codes.size}) != connectivity matrix size ({N}).\n"
                "This indicates the parcellation in the provided atlas does not match the ROI ordering/number used to build the connectivity matrix.\n"
                "Options to resolve:\n"
                "  1) Provide an atlas whose non-zero region labels count equals the number of ROIs in the matrix.\n"
                "  2) If you have a label->ROI mapping file, use it to map atlas labels to ROI indices before calling this function.\n"
                "  3) If the atlas is in a different grid/space, set resample_on_mismatch=True and provide reference_img_path to resample using nilearn.image.resample_to_img.\n"
            )

    roi_to_code = dict(zip(range(N), region_codes))
    communities = np.unique(labels)
    K = len(communities)
    colors = get_module_colors(K, alpha=0.7)

    # Determine grid layout: overlay on top row (centered), individual modules below
    ncols = 3
    nrows_modules = int(np.ceil(K / ncols))
    nrows = 1 + nrows_modules  # 1 row for overlay + rows for modules
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.2)

    # Build per-module mask images
    mask_imgs = []
    for idx_comm, comm in enumerate(communities):
        roi_idx = np.where(labels == comm)[0]
        codes = [roi_to_code[i] for i in roi_idx]
        # Determine nifti output directory (if requested)
        if save_niftis:
            nifti_dir = nifti_dir or output_dir or os.path.dirname(output_path) or os.getcwd()
            os.makedirs(nifti_dir, exist_ok=True)

        # Create a binary mask (uint8) and attach a copy of the atlas header to preserve metadata
        mask = np.isin(atlas_data, codes).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask, atlas_img.affine, header=atlas_img.header.copy())
        mask_imgs.append((comm, roi_idx.size, mask_img, colors[idx_comm]))
        # Optionally save per-module NIfTI masks for external viewers (MRIcroGL, etc.)
        if save_niftis:
            nifti_path = os.path.join(nifti_dir, f'module_{int(comm):02d}_mask{nifti_suffix}')
            try:
                nib.save(mask_img, nifti_path)
                print(f"[SAVE] {nifti_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save module mask NIfTI '{nifti_path}': {e}")

    # Build overlay image (all modules)
    overlay = np.zeros_like(atlas_data, dtype=np.int32)
    for roi_idx, code in enumerate(region_codes[:labels.size]):
        overlay[atlas_data == code] = labels[roi_idx] + 1
    # Create colormap: need K colors for K unique module labels (1 to K)
    # plot_roi maps unique non-zero values to colormap colors sequentially
    overlay_colors = [colors[i][:3] for i in range(K)]
    cmap_overlay = ListedColormap(overlay_colors)
    overlay_img = nib.Nifti1Image(overlay.astype(np.uint8), atlas_img.affine, header=atlas_img.header.copy())

    # Plot overlay centered on top row (spanning middle column)
    ax_overlay = fig.add_subplot(gs[0, 1])
    display = plotting.plot_roi(overlay_img,
                                cmap=cmap_overlay,
                                colorbar=False,
                                display_mode='ortho',
                                black_bg=not background_white,
                                alpha=alpha,
                                title='All Modules Overlay',
                                axes=ax_overlay)

    # Save individual overlay figure
    if output_dir is not None:
        fig_overlay = plt.figure()
        disp_overlay = plotting.plot_roi(overlay_img,
                         cmap=cmap_overlay,
                         colorbar=False,
                         display_mode='ortho',
                         black_bg=not background_white,
                         alpha=alpha,
                         title='All Modules Overlay',
                         figure=fig_overlay)
        overlay_ind_path = os.path.join(output_dir, 'all_modules_overlay.png')
        disp_overlay.savefig(overlay_ind_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig_overlay)
        print(f"[SAVE] {overlay_ind_path}")

    # Plot each module on rows below overlay
    for idx, (comm, size, mask_img, color) in enumerate(mask_imgs):
        row, col = divmod(idx, ncols)
        row += 1  # shift down by 1 to leave top row for overlay
        ax = fig.add_subplot(gs[row, col])
        # Create single-color colormap for this module's mask
        module_cmap = ListedColormap([color[:3]])
        title = f"M{int(comm)} (n={size})"
        display = plotting.plot_roi(mask_img,
                            cmap=module_cmap,
                            colorbar=False,
                            display_mode='ortho',
                            black_bg=not background_white,
                            alpha=alpha,
                            title=title,
                            axes=ax)

        # Save individual module figure
        if output_dir is not None:
                fig_ind = plt.figure()
                disp_ind = plotting.plot_roi(mask_img,
                                    cmap=module_cmap,
                                    colorbar=False,
                                    display_mode='ortho',
                                    black_bg=not background_white,
                                    alpha=alpha,
                                    title=title,
                                    figure=fig_ind)
                ind_path = os.path.join(output_dir, f'module_{int(comm):02d}.png')
                disp_ind.savefig(ind_path, dpi=300, bbox_inches='tight', format='png')
                plt.close(fig_ind)
                print(f"[SAVE] {ind_path}")

    plt.suptitle('Module ROI Visualizations', fontsize=18, fontweight='bold', y=0.92)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {output_path}")