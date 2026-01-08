"""
Manuscript-ready version of the methodological choice analysis for consensus size.
Generates variability/stability plots for raw and aligned eigenvectors and a summary
panel comparing alignment methods across consensus sizes.

Saves figures to FIGURES/Manuscript.
"""
import os
import sys
import random
import numpy as np
import scipy.io as sio
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to import custom libs
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import lib.func_GSP as gsp
from lib import fcn_groups_bin

# Matplotlib styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Aptos', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 10

# Paths
example_dir = os.path.join(project_root, "DATA/EEG")
sc_path = os.path.join(project_root, 'DATA/Individual_Connectomes.mat')
roi_info_path = os.path.join(project_root, 'data/label/roi_info.xlsx')
figures_dir = os.path.join(project_root, 'FIGURES/Manuscript')
os.makedirs(figures_dir, exist_ok=True)

# Load data
SC = sio.loadmat(sc_path)
SC = SC['connMatrices']['SC'][0][0][1][0]
roi_info = pd.read_excel(roi_info_path, sheet_name='SCALE 2')
cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
matMetric = SC
x = np.asarray(roi_info['x-pos'])[cort_rois]
y = np.asarray(roi_info['y-pos'])[cort_rois]
z = np.asarray(roi_info['z-pos'])[cort_rois]
coordMat = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))

# Parameters
ls_bins = [1, 5, 10, 20, 25]
nbPerm = 100
nbins = 41
total_participant = matMetric.shape[2]
nROIs = matMetric.shape[0]
idxs = list(range(total_participant))
hemii = np.ones(len(Euc))
hemii[int(len(hemii) / 2):] = 2

RandCons = np.zeros((nROIs, nROIs, nbPerm, len(ls_bins)))
ShuffIdxs = np.zeros((len(idxs), nbPerm, len(ls_bins)))

# Consensus generation
for b, bi in enumerate(ls_bins):
    for p in np.arange(nbPerm):
        random.shuffle(idxs)
        ShuffIdxs[:, p, b] = idxs
        idxs_tmp = idxs[0:bi]
        G, Gc = fcn_groups_bin.fcn_groups_bin(matMetric[:, :, idxs_tmp], Euc, hemii, nbins)
        avg = np.mean(matMetric[:, :, idxs_tmp], 2)
        RandCons[:, :, p, b] = Gc * avg
print(f'nROIs={RandCons.shape[0]}, number of bins={RandCons.shape[3]}, number of randomization={RandCons.shape[2]}')

# Eigenvectors
nb_eig2keep = nROIs
eigenvectors_perm = np.zeros((len(cort_rois), nb_eig2keep, len(ls_bins) * nbPerm))
eigenvalues_perm = np.zeros((nb_eig2keep, len(ls_bins) * nbPerm))
eigenvectors_perm_mat = np.zeros((len(cort_rois), nb_eig2keep, len(ls_bins), nbPerm))
eigenvalues_perm_mat = np.zeros((nb_eig2keep, len(ls_bins), nbPerm))
labels_perm = []

for b, bi in enumerate(ls_bins):
    for p in np.arange(nbPerm):
        try:
            eigenvalues_perm_mat[:, b, p], eigenvectors_perm_mat[:, :, b, p], Ln_ind, An_ind = gsp.cons_normalized_lap(
                RandCons[:, :, p, b], Euc, plot=False)
            labels_perm.append(f'Bin{bi}')
        except np.linalg.LinAlgError:
            print(f"Warning: SVD convergence failed for bin {bi}, permutation {p}. Skipping...")
            eigenvalues_perm_mat[:, b, p] = np.nan
            eigenvectors_perm_mat[:, :, b, p] = np.nan
            labels_perm.append(f'Bin{bi}')

# Alignments
max_retries = 3
eigenvalues_perm_mat_rot = np.zeros_like(eigenvalues_perm_mat)
eigenvectors_perm_mat_rot = np.zeros_like(eigenvectors_perm_mat)
eigenvalues_perm_mat_ortho = np.zeros_like(eigenvalues_perm_mat)
eigenvectors_perm_mat_ortho = np.zeros_like(eigenvectors_perm_mat)
eigenvalues_perm_mat_matched = np.zeros_like(eigenvalues_perm_mat)
eigenvectors_perm_mat_matched = np.zeros_like(eigenvectors_perm_mat)
R_all = np.zeros_like(eigenvectors_perm_mat)
scale_R = np.zeros((len(ls_bins), nbPerm))

for b, bi in enumerate(ls_bins):
    print(bi)
    # Generalized Procrustes with retry
    for retry in range(max_retries):
        try:
            eigenvectors_perm_mat_rot[:, :, b, :], eigenvalues_perm_mat_rot[:, b, :], A, B = gsp.rotation_procrustes(
                eigenvectors_perm_mat[:, :, b, :], eigenvalues_perm_mat[:, b, :], plot=False, p=f'bin{bi}')
            break
        except np.linalg.LinAlgError:
            if retry < max_retries - 1:
                print(f"  Generalized Procrustes SVD failed for bin {bi}, retry {retry+1}/{max_retries}...")
                eigenvectors_perm_mat[:, :, b, :] += np.random.randn(*eigenvectors_perm_mat[:, :, b, :].shape) * 1e-10
            else:
                print(f"  Generalized Procrustes failed for bin {bi} after {max_retries} retries, filling with NaN")
                eigenvectors_perm_mat_rot[:, :, b, :] = np.nan
                eigenvalues_perm_mat_rot[:, b, :] = np.nan

    # Orthogonal Procrustes with retry
    for retry in range(max_retries):
        try:
            eigenvectors_perm_mat_ortho[:, :, b, :], eigenvalues_perm_mat_ortho[:, b, :], R_all[:, :, b, :], scale_R[b, :] = (
                gsp.orthogonal_rotation_procrustes(eigenvectors_perm_mat[:, :, b, :], eigenvalues_perm_mat[:, b, :],
                                                   plot=False, p=f'bin{bi}'))
            break
        except np.linalg.LinAlgError:
            if retry < max_retries - 1:
                print(f"  Orthogonal Procrustes SVD failed for bin {bi}, retry {retry+1}/{max_retries}...")
                eigenvectors_perm_mat[:, :, b, :] += np.random.randn(*eigenvectors_perm_mat[:, :, b, :].shape) * 1e-10
            else:
                print(f"  Orthogonal Procrustes failed for bin {bi} after {max_retries} retries, filling with NaN")
                eigenvectors_perm_mat_ortho[:, :, b, :] = np.nan
                eigenvalues_perm_mat_ortho[:, b, :] = np.nan
                R_all[:, :, b, :] = np.nan
                scale_R[b, :] = np.nan

    # Hungarian matching
    for q in np.arange(nbPerm):
        perm, total_cost = gsp.match_eigenvectors(eigenvectors_perm_mat[:, :, b, 0], eigenvectors_perm_mat[:, :, b, q])
        eigenvectors_perm_mat_matched[:, :, b, q] = eigenvectors_perm_mat[:, perm, b, q]
        eigenvalues_perm_mat_matched[:, b, q] = eigenvalues_perm_mat[perm, b, q]

# Similarity matrices
Dist_eigvec_perm = np.zeros((len(ls_bins) * nbPerm, len(ls_bins) * nbPerm, nb_eig2keep))
Dist_eigvec_perm_ortho = np.zeros_like(Dist_eigvec_perm)
Dist_eigvec_perm_rot = np.zeros_like(Dist_eigvec_perm)
Dist_eigvec_perm_matched = np.zeros_like(Dist_eigvec_perm)

for eigvec_nb in np.arange(nb_eig2keep):
    MatDist = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb, :]), metric='correlation')
    Dist_eigvec_perm[:, :, eigvec_nb] = scipy.spatial.distance.squareform(MatDist)
    MatDist_ortho = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_ortho[:, eigvec_nb, :]), metric='correlation')
    Dist_eigvec_perm_ortho[:, :, eigvec_nb] = scipy.spatial.distance.squareform(MatDist_ortho)
    MatDist_rot = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_rot[:, eigvec_nb, :]), metric='correlation')
    Dist_eigvec_perm_rot[:, :, eigvec_nb] = scipy.spatial.distance.squareform(MatDist_rot)
    MatDist_matched = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_matched[:, eigvec_nb, :]), metric='correlation')
    Dist_eigvec_perm_matched[:, :, eigvec_nb] = scipy.spatial.distance.squareform(MatDist_matched)

# Flatten and remove zeros
Dist_eigvec_perm_vec = np.reshape(Dist_eigvec_perm, (len(ls_bins) * nbPerm * len(ls_bins) * nbPerm, nb_eig2keep))
Dist_eigvec_perm_vec = np.abs(Dist_eigvec_perm_vec)
Dist_eigvec_perm_ortho_vec = np.reshape(Dist_eigvec_perm_ortho, (len(ls_bins) * nbPerm * len(ls_bins) * nbPerm, nb_eig2keep))
Dist_eigvec_perm_ortho_vec = np.abs(Dist_eigvec_perm_ortho_vec)
Dist_eigvec_perm_rot_vec = np.reshape(Dist_eigvec_perm_rot, (len(ls_bins) * nbPerm * len(ls_bins) * nbPerm, nb_eig2keep))
Dist_eigvec_perm_rot_vec = np.abs(Dist_eigvec_perm_rot_vec)
Dist_eigvec_perm_matched_vec = np.reshape(Dist_eigvec_perm_matched, (len(ls_bins) * nbPerm * len(ls_bins) * nbPerm, nb_eig2keep))
Dist_eigvec_perm_matched_vec = np.abs(Dist_eigvec_perm_matched_vec)

for i in np.arange(nb_eig2keep):
    tmp = Dist_eigvec_perm_vec[:, i]
    tmp2 = Dist_eigvec_perm_ortho_vec[:, i]
    tmp3 = Dist_eigvec_perm_rot_vec[:, i]
    tmp4 = Dist_eigvec_perm_matched_vec[:, i]
    if i == 0:
        Dist_eigvec_perm_vec_nz = np.zeros((len(tmp), nb_eig2keep))
        Dist_eigvec_perm_ortho_vec_nz = np.zeros((len(tmp2), nb_eig2keep))
        Dist_eigvec_perm_rot_vec_nz = np.zeros((len(tmp3), nb_eig2keep))
        Dist_eigvec_perm_matched_vec_nz = np.zeros((len(tmp4), nb_eig2keep))
    Dist_eigvec_perm_vec_nz[:, i] = tmp
    Dist_eigvec_perm_ortho_vec_nz[:, i] = tmp2
    Dist_eigvec_perm_rot_vec_nz[:, i] = tmp3
    Dist_eigvec_perm_matched_vec_nz[:, i] = tmp4

# Aggregate by bin
bin_variability = np.zeros((len(ls_bins), nb_eig2keep, 2))
bin_variability_ortho = np.zeros_like(bin_variability)
bin_variability_matched = np.zeros_like(bin_variability)
bin_variability_rot = np.zeros_like(bin_variability)

labels_perm = np.array(labels_perm)
labels_perm_mat = []
for i in np.arange(len(labels_perm)):
    for j in np.arange(len(labels_perm)):
        labels_perm_mat.append(f'{labels_perm[i]}_{labels_perm[j]}')
labels_perm_mat = np.array(labels_perm_mat)

for b, bi in enumerate(ls_bins):
    idxs_bin = np.where(labels_perm_mat == f'Bin{bi}_Bin{bi}')[0]
    for i in np.arange(nb_eig2keep):
        bin_variability[b, i, 0] = np.median(Dist_eigvec_perm_vec_nz[idxs_bin, i])
        bin_variability[b, i, 1] = np.std(Dist_eigvec_perm_vec_nz[idxs_bin, i])
        bin_variability_ortho[b, i, 0] = np.median(Dist_eigvec_perm_ortho_vec_nz[idxs_bin, i])
        bin_variability_ortho[b, i, 1] = np.std(Dist_eigvec_perm_ortho_vec_nz[idxs_bin, i])
        bin_variability_matched[b, i, 0] = np.median(Dist_eigvec_perm_matched_vec_nz[idxs_bin, i])
        bin_variability_matched[b, i, 1] = np.std(Dist_eigvec_perm_matched_vec_nz[idxs_bin, i])
        bin_variability_rot[b, i, 0] = np.median(Dist_eigvec_perm_rot_vec_nz[idxs_bin, i])
        bin_variability_rot[b, i, 1] = np.std(Dist_eigvec_perm_rot_vec_nz[idxs_bin, i])

# Main figure (2x2)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()
colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
handles = []

for b, bi in enumerate(ls_bins):
    line, = ax[0].plot(bin_variability[b, :, 0], linewidth=2, color=colors_palette[b], label=f'n={bi}')
    if b < len(ls_bins):
        handles.append(line)
    ax[1].plot(bin_variability_ortho[b, :, 0], linewidth=2, color=colors_palette[b], label=f'n={bi}')
    ax[2].plot(bin_variability_rot[b, :, 0], linewidth=2, color=colors_palette[b], label=f'n={bi}')
    ax[3].plot(bin_variability_matched[b, :, 0], linewidth=2, color=colors_palette[b], label=f'n={bi}')

    mean_raw = np.mean(bin_variability[b, :, 0])
    mean_ortho = np.mean(bin_variability_ortho[b, :, 0])
    mean_rot = np.mean(bin_variability_rot[b, :, 0])
    mean_matched = np.mean(bin_variability_matched[b, :, 0])

    ax[0].axhline(y=mean_raw, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[0].text(nb_eig2keep + 1, mean_raw, f'{mean_raw:.2f}', color=colors_palette[b], fontsize=9, va='center', ha='left', fontweight='bold')
    ax[1].axhline(y=mean_ortho, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[1].text(nb_eig2keep + 1, mean_ortho, f'{mean_ortho:.2f}', color=colors_palette[b], fontsize=9, va='center', ha='left', fontweight='bold')
    ax[2].axhline(y=mean_rot, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[2].text(nb_eig2keep + 1, mean_rot, f'{mean_rot:.2f}', color=colors_palette[b], fontsize=9, va='center', ha='left', fontweight='bold')
    ax[3].axhline(y=mean_matched, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[3].text(nb_eig2keep + 1, mean_matched, f'{mean_matched:.2f}', color=colors_palette[b], fontsize=9, va='center', ha='left', fontweight='bold')

    upper_bound = bin_variability[b, :, 0] + bin_variability[b, :, 1]
    lower_bound = bin_variability[b, :, 0] - bin_variability[b, :, 1]
    upper_bound_ortho = bin_variability_ortho[b, :, 0] + bin_variability_ortho[b, :, 1]
    lower_bound_ortho = bin_variability_ortho[b, :, 0] - bin_variability_ortho[b, :, 1]
    upper_bound_matched = bin_variability_matched[b, :, 0] + bin_variability_matched[b, :, 1]
    lower_bound_matched = bin_variability_matched[b, :, 0] - bin_variability_matched[b, :, 1]
    upper_bound_rot = bin_variability_rot[b, :, 0] + bin_variability_rot[b, :, 1]
    lower_bound_rot = bin_variability_rot[b, :, 0] - bin_variability_rot[b, :, 1]

    ax[0].fill_between(range(nb_eig2keep), lower_bound, upper_bound, alpha=0.15, color=colors_palette[b])
    ax[1].fill_between(range(nb_eig2keep), lower_bound_ortho, upper_bound_ortho, alpha=0.15, color=colors_palette[b])
    ax[2].fill_between(range(nb_eig2keep), lower_bound_rot, upper_bound_rot, alpha=0.15, color=colors_palette[b])
    ax[3].fill_between(range(nb_eig2keep), lower_bound_matched, upper_bound_matched, alpha=0.15, color=colors_palette[b])

for x in range(4):
    ax[x].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
    ax[x].set_xticks(range(0, nb_eig2keep, 20))
    ax[x].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax[x].set_ylim([0, 1.02])
    ax[x].set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax[x].spines['top'].set_visible(False)
    ax[x].spines['right'].set_visible(False)
    ax[x].tick_params(labelsize=10)
    ax[x].legend(handles=handles, labels=[f'n={bi}' for bi in ls_bins],
                 loc='lower right', frameon=True, fancybox=False, shadow=False,
                 fontsize=9, title='Consensus size', title_fontsize=9)

ax[0].set_title('A. Raw eigenvector similarity', fontsize=13, fontweight='bold', loc='left', pad=10)
ax[1].set_title('B. Orthogonal Procrustes', fontsize=13, fontweight='bold', loc='left', pad=10)
ax[2].set_title('C. Generalized Procrustes', fontsize=13, fontweight='bold', loc='left', pad=10)
ax[3].set_title('D. Hungarian matching', fontsize=13, fontweight='bold', loc='left', pad=10)

plt.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.12)
fig_path_png = os.path.join(figures_dir, 'bin_variability_analysis.png')
fig_path_pdf = os.path.join(figures_dir, 'bin_variability_analysis.pdf')
plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Plots saved as '{fig_path_png}' and '{fig_path_pdf}'")

# Summary figure (1x3)
method_labels = ['Raw', 'Orthogonal Procrustes', 'Hungarian matching', 'Generalized Procrustes']
method_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

mean_modes = [
    np.mean(bin_variability[:, :, 0], axis=0),
    np.mean(bin_variability_ortho[:, :, 0], axis=0),
    np.mean(bin_variability_matched[:, :, 0], axis=0),
    np.mean(bin_variability_rot[:, :, 0], axis=0),
]
std_modes = [
    np.std(bin_variability[:, :, 0], axis=0),
    np.std(bin_variability_ortho[:, :, 0], axis=0),
    np.std(bin_variability_matched[:, :, 0], axis=0),
    np.std(bin_variability_rot[:, :, 0], axis=0),
]

bin_means = [
    np.mean(bin_variability[:, :, 0], axis=1),
    np.mean(bin_variability_ortho[:, :, 0], axis=1),
    np.mean(bin_variability_matched[:, :, 0], axis=1),
    np.mean(bin_variability_rot[:, :, 0], axis=1),
]
bin_stds = [
    np.std(bin_variability[:, :, 0], axis=1),
    np.std(bin_variability_ortho[:, :, 0], axis=1),
    np.std(bin_variability_matched[:, :, 0], axis=1),
    np.std(bin_variability_rot[:, :, 0], axis=1),
]

# Improvements relative to raw
delta_modes = [
    mean_modes[1] - mean_modes[0],
    mean_modes[2] - mean_modes[0],
    mean_modes[3] - mean_modes[0],
]

fig2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
ax2 = ax2.flatten()

# Panel 1: Effect of consensus size (grouped bars)
x = np.arange(len(ls_bins))
width = 0.18
bar_handles = []
for idx, label in enumerate(method_labels):
    h = ax2[0].bar(x + (idx - 1.5) * width, bin_means[idx], width=width, color=method_colors[idx],
                   yerr=bin_stds[idx], capsize=3, alpha=0.9, label=label)
    bar_handles.append(h)
ax2[0].set_xticks(x)
ax2[0].set_xticklabels([f'n={b}' for b in ls_bins])
ax2[0].set_ylabel('Mean correlation across modes')
ax2[0].set_title('Effect of consensus size', fontsize=12, fontweight='bold')
ax2[0].set_ylim([0, 1.5])
ax2[0].grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

# Add significance tests across bins for each method
for method_idx in range(4):
    # ANOVA to test if this method differs across bin sizes
    data_by_bin = []
    for b, bi in enumerate(ls_bins):
        idxs_bin = np.where(labels_perm_mat == f'Bin{bi}_Bin{bi}')[0]
        if method_idx == 0:
            data_by_bin.append(Dist_eigvec_perm_vec_nz[idxs_bin, :].flatten())
        elif method_idx == 1:
            data_by_bin.append(Dist_eigvec_perm_ortho_vec_nz[idxs_bin, :].flatten())
        elif method_idx == 2:
            data_by_bin.append(Dist_eigvec_perm_matched_vec_nz[idxs_bin, :].flatten())
        else:
            data_by_bin.append(Dist_eigvec_perm_rot_vec_nz[idxs_bin, :].flatten())
    f_stat, p_val = f_oneway(*data_by_bin)
    p_str = f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}"
    # Place asterisk to the right side of each method's color
    x_pos = len(ls_bins) - 0.5 + (method_idx - 1.5) * width
    y_pos = 1.42
    ax2[0].text(x_pos, y_pos, p_str, fontsize=9, fontweight='bold', 
                ha='center', va='center', color=method_colors[method_idx], 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=method_colors[method_idx], linewidth=1.5))

# Panel 2: Mean stability across bins
line_handles = []
for idx, label in enumerate(method_labels):
    line, = ax2[1].plot(mean_modes[idx], color=method_colors[idx], linewidth=2, label=label)
    line_handles.append(line)
    ax2[1].fill_between(range(nb_eig2keep), mean_modes[idx] - std_modes[idx], mean_modes[idx] + std_modes[idx],
                        color=method_colors[idx], alpha=0.12)
ax2[1].set_title('Mean stability across bins', fontsize=12, fontweight='bold')
ax2[1].set_xlabel('Eigenmode')
ax2[1].set_ylabel('Correlation')
ax2[1].set_ylim([0, 1.02])
ax2[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Panel 3: Alignment gain vs raw
delta_labels = ['Orthogonal - Raw', 'Generalized - Raw', 'Hungarian - Raw']
delta_colors = [method_colors[1], method_colors[2], method_colors[3]]
delta_handles = []
for idx, label in enumerate(delta_labels):
    line, = ax2[2].plot(delta_modes[idx], color=delta_colors[idx], linewidth=2, label=label)
    delta_handles.append(line)
ax2[2].axhline(0, color='gray', linestyle='--', linewidth=1)
ax2[2].set_title('Alignment gain vs raw', fontsize=12, fontweight='bold')
ax2[2].set_xlabel('Eigenmode')
ax2[2].set_ylabel('Δ correlation')
ax2[2].grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Add significance asterisks near peak of each delta curve
for align_idx, align in enumerate(['Orthogonal', 'Generalized', 'Hungarian']):
    stats = sig_tests_panel3[align]
    p_val = stats['p_val']
    p_str = f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}"
    # Place asterisk near the peak of each line
    peak_idx = np.argmax(np.abs(delta_modes[align_idx]))
    peak_y = delta_modes[align_idx][peak_idx]
    ax2[2].text(peak_idx, peak_y, p_str, fontsize=10, fontweight='bold',
                ha='center', va='bottom', color=delta_colors[align_idx])

for axis in ax2.flatten():
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(labelsize=10)

# Single consolidated legend for summary figure
all_handles = line_handles + delta_handles
all_labels = method_labels + delta_labels
fig2.legend(handles=all_handles, labels=all_labels,
           loc='upper center', frameon=True, fancybox=False, shadow=False,
           fontsize=9, ncol=7, bbox_to_anchor=(0.5, 1.08), 
           title='Methods and comparisons', title_fontsize=9)

fig2.tight_layout()
fig2_path_png = os.path.join(figures_dir, 'stability_alignment_summary.png')
fig2_path_pdf = os.path.join(figures_dir, 'stability_alignment_summary.pdf')
fig2.savefig(fig2_path_png, dpi=300, bbox_inches='tight', facecolor='white')
fig2.savefig(fig2_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Summary plots saved as '{fig2_path_png}' and '{fig2_path_pdf}'")

plt.show()
