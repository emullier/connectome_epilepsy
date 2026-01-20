"""
Manuscript-ready analysis concatenating methodological choice (script 12) and alignment comparison (script 10).

Part 1 (from script 12): Methodological choice analysis
- Creates consensus matrices with varying numbers of participants (n=1,5,10,20,25)
- Applies different alignment methods (Generalized Procrustes, Orthogonal Procrustes, Hungarian)
- Measures variability within each consensus size
- Generates Figure 1: Within-bin variability (2x2 grid)

Part 2 (from script 10): Alignment comparison to reference
- Compares SC IND (27 controls) vs SC HC (Geneva dataset)
- Shows how alignment methods improve similarity to reference
- Generates Figure 2: Similarity to reference consensus (2x2 grid)
- Generates Figure 3: Summary comparison (1x3 panels)

Saves figures to FIGURES.
"""
import os
import sys
import random
import numpy as np
import scipy.io as sio
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to import custom libs
project_root = os.path.dirname(os.path.abspath(__file__))
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
figures_dir = os.path.join(project_root, 'FIGURES')
output_dir = os.path.join(project_root, 'OUTPUT')
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

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

# Reshape 4D arrays to 3D for distance computation
eigenvectors_perm_ortho = np.reshape(eigenvectors_perm_mat_ortho, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm_rot = np.reshape(eigenvectors_perm_mat_rot, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm_matched = np.reshape(eigenvectors_perm_mat_matched, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm = np.reshape(eigenvectors_perm_mat, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))

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
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
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
    std_raw = np.std(bin_variability[b, :, 0])
    mean_ortho = np.mean(bin_variability_ortho[b, :, 0])
    std_ortho = np.std(bin_variability_ortho[b, :, 0])
    mean_rot = np.mean(bin_variability_rot[b, :, 0])
    std_rot = np.std(bin_variability_rot[b, :, 0])
    mean_matched = np.mean(bin_variability_matched[b, :, 0])
    std_matched = np.std(bin_variability_matched[b, :, 0])

    ax[0].axhline(y=mean_raw, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[0].text(nb_eig2keep + 1, mean_raw, f'{mean_raw:.2f}±{std_raw:.2f}', color=colors_palette[b], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[1].axhline(y=mean_ortho, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[1].text(nb_eig2keep + 1, mean_ortho, f'{mean_ortho:.2f}±{std_ortho:.2f}', color=colors_palette[b], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[2].axhline(y=mean_rot, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[2].text(nb_eig2keep + 1, mean_rot, f'{mean_rot:.2f}±{std_rot:.2f}', color=colors_palette[b], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[3].axhline(y=mean_matched, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[3].text(nb_eig2keep + 1, mean_matched, f'{mean_matched:.2f}±{std_matched:.2f}', color=colors_palette[b], fontsize=8, va='center', ha='left', fontweight='bold')

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

ax[0].set_title('A. Raw eigenvector similarity', fontsize=13, fontweight='bold', loc='left', pad=10)
ax[1].set_title('B. Orthogonal Procrustes', fontsize=13, fontweight='bold', loc='left', pad=10)
ax[2].set_title('C. Generalized Procrustes', fontsize=13, fontweight='bold', loc='left', pad=10)
ax[3].set_title('D. Hungarian matching', fontsize=13, fontweight='bold', loc='left', pad=10)

# Single legend outside
fig.legend(handles=handles, labels=[f'n={bi}' for bi in ls_bins],
           loc='center', frameon=True, fancybox=False, shadow=False,
           fontsize=10, title='Consensus size', title_fontsize=10,
           bbox_to_anchor=(0.5, 0.5))

fig_path_png = os.path.join(figures_dir, 'Fig4_bin_variability_analysis.png')
plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Plot saved as '{fig_path_png}'")

# ============================================================================
# PART 2: Comparison between SC IND and SC HC datasets (from script 10)
# ============================================================================
print("\n" + "="*80)
print("PART 2: Comparing SC IND (27 controls) vs SC HC reference")
print("="*80)

# Load datasets
print("Loading SC IND and SC HC datasets...")
consensus_HC_DSI = np.load("DATA/SC/matMetric_HC_DSI_number_of_fibers.npy")
consensus_schz = np.load("DATA/SC/matMetric_SCHZ_CTRL.npy")
consensus_HC_ref = np.mean(consensus_HC_DSI, axis=2)
consensus_schz_mean = np.mean(consensus_schz, axis=0)
EucDist = np.load("DATA/EucMat/EucMat_HC_dsi_number_of_fibers.npy")

print("Generating harmonics from both consensus matrices...")
P_ref, Q_ref, Ln_ref, An_ref = gsp.cons_normalized_lap(consensus_HC_ref, EucDist, plot=False)
P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus_schz_mean, EucDist, plot=False)

print("Applying alignment methods...")
# Generalized Procrustes
Qind_rotated, Qind_HC_centered, disparity = scipy.spatial.procrustes(Q_ref, Q_ind)

# Orthogonal Procrustes
R, _ = scipy.linalg.orthogonal_procrustes(Q_ref, Q_ind)
Qind_ortho_rotated = Q_ind @ R

# Hungarian matching
perm, total_cost = gsp.match_eigenvectors(Q_ref, Q_ind)
Qind_matched = Q_ind[:, perm]

# Compute similarity between harmonics
print("Computing similarity metrics...")
nb_eig = Q_ref.shape[1]
similarity_ind = np.zeros(nb_eig)
similarity_rotated = np.zeros(nb_eig)
similarity_ortho = np.zeros(nb_eig)
similarity_matched = np.zeros(nb_eig)

for eigvec_nb in range(nb_eig):
    # Correlation-based similarity (1 - correlation distance)
    similarity_ind[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Q_ind[:, eigvec_nb])
    similarity_rotated[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_rotated[:, eigvec_nb])
    similarity_ortho[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_ortho_rotated[:, eigvec_nb])
    similarity_matched[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_matched[:, eigvec_nb])

# Take absolute values to compensate for sign flips
similarity_ind = np.abs(similarity_ind)
similarity_rotated = np.abs(similarity_rotated)
similarity_ortho = np.abs(similarity_ortho)
similarity_matched = np.abs(similarity_matched)

print(f"\nHarmonic Similarity:")
print(f"  SC IND vs SC HC:           Mean={np.mean(similarity_ind):.4f}, Median={np.median(similarity_ind):.4f}")
print(f"  Gen. Procrustes vs SC HC:  Mean={np.mean(similarity_rotated):.4f}, Median={np.median(similarity_rotated):.4f}")
print(f"  Ortho. Procrustes vs SC HC: Mean={np.mean(similarity_ortho):.4f}, Median={np.median(similarity_ortho):.4f}")
print(f"  Hungarian vs SC HC:        Mean={np.mean(similarity_matched):.4f}, Median={np.median(similarity_matched):.4f}")

# Figure 2: Similarity between SC HC (ref) and SC IND harmonics - 2x2 layout
fig2, ax2 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
ax2 = ax2.flatten()

n_harmonics = len(similarity_ind)
harmonic_indices = np.arange(n_harmonics)

# Define colors for consistency
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
similarity_ylim = (0, 1.02)

# A. Before alignment (SC IND vs SC HC)
ax2[0].plot(harmonic_indices, similarity_ind, linewidth=2, color=colors[0])
mean_ind = np.mean(similarity_ind)
std_ind = np.std(similarity_ind)
ax2[0].axhline(y=mean_ind, color=colors[0], linestyle=':', linewidth=1.5, alpha=0.7)
ax2[0].text(n_harmonics+1, mean_ind, f'{mean_ind:.3f}±{std_ind:.3f}', color=colors[0], fontsize=8, va='center', ha='left', fontweight='bold')
ax2[0].fill_between(harmonic_indices, similarity_ind - std_ind, similarity_ind + std_ind, alpha=0.15, color=colors[0])
ax2[0].set_title(f'A. Before alignment (r={mean_ind:.3f}±{std_ind:.3f})', fontsize=13, fontweight='bold', loc='left', pad=10)
ax2[0].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
ax2[0].set_ylabel('Correlation', fontsize=12, fontweight='bold')

# B. Orthogonal Procrustes
ax2[1].plot(harmonic_indices, similarity_ortho, linewidth=2, color=colors[1])
mean_ortho = np.mean(similarity_ortho)
std_ortho = np.std(similarity_ortho)
ax2[1].axhline(y=mean_ortho, color=colors[1], linestyle=':', linewidth=1.5, alpha=0.7)
ax2[1].text(n_harmonics+1, mean_ortho, f'{mean_ortho:.3f}±{std_ortho:.3f}', color=colors[1], fontsize=8, va='center', ha='left', fontweight='bold')
ax2[1].fill_between(harmonic_indices, similarity_ortho - std_ortho, similarity_ortho + std_ortho, alpha=0.15, color=colors[1])
ax2[1].set_title(f'B. Orthogonal Procrustes (r={mean_ortho:.3f}±{std_ortho:.3f})', fontsize=13, fontweight='bold', loc='left', pad=10)
ax2[1].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
ax2[1].set_ylabel('Correlation', fontsize=12, fontweight='bold')

# C. Generalized Procrustes
ax2[2].plot(harmonic_indices, similarity_rotated, linewidth=2, color=colors[2])
mean_rotated = np.mean(similarity_rotated)
std_rotated = np.std(similarity_rotated)
ax2[2].axhline(y=mean_rotated, color=colors[2], linestyle=':', linewidth=1.5, alpha=0.7)
ax2[2].text(n_harmonics+1, mean_rotated, f'{mean_rotated:.3f}±{std_rotated:.3f}', color=colors[2], fontsize=8, va='center', ha='left', fontweight='bold')
ax2[2].fill_between(harmonic_indices, similarity_rotated - std_rotated, similarity_rotated + std_rotated, alpha=0.15, color=colors[2])
ax2[2].set_title(f'C. Generalized Procrustes (r={mean_rotated:.3f}±{std_rotated:.3f})', fontsize=13, fontweight='bold', loc='left', pad=10)
ax2[2].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
ax2[2].set_ylabel('Correlation', fontsize=12, fontweight='bold')

# D. Hungarian matching
ax2[3].plot(harmonic_indices, similarity_matched, linewidth=2, color=colors[3])
mean_matched = np.mean(similarity_matched)
std_matched = np.std(similarity_matched)
ax2[3].axhline(y=mean_matched, color=colors[3], linestyle=':', linewidth=1.5, alpha=0.7)
ax2[3].text(n_harmonics+1, mean_matched, f'{mean_matched:.3f}±{std_matched:.3f}', color=colors[3], fontsize=8, va='center', ha='left', fontweight='bold')
ax2[3].fill_between(harmonic_indices, similarity_matched - std_matched, similarity_matched + std_matched, alpha=0.15, color=colors[3])
ax2[3].set_title(f'D. Hungarian matching (r={mean_matched:.3f}±{std_matched:.3f})', fontsize=13, fontweight='bold', loc='left', pad=10)
ax2[3].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
ax2[3].set_ylabel('Correlation', fontsize=12, fontweight='bold')

# Format all subplots
for i in range(4):
    ax2[i].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax2[i].set_ylim(similarity_ylim)
    ax2[i].set_xticks(range(0, n_harmonics, 20))
    ax2[i].spines['top'].set_visible(False)
    ax2[i].spines['right'].set_visible(False)
    ax2[i].spines['left'].set_linewidth(1.5)
    ax2[i].spines['bottom'].set_linewidth(1.5)
    ax2[i].tick_params(labelsize=11)

fig2.suptitle('Similarity between SC HC (ref) and SC IND harmonics', fontsize=16, fontweight='bold', y=0.98)

fig2_path_png = os.path.join(figures_dir, 'Fig3_harmonic_similarity_SC_IND_vs_SC_HC.png')
plt.savefig(fig2_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Part 2 figure saved as '{fig2_path_png}'")

# ============================================================================
# Figure 3: Cutoff frequency comparison (from script 10)
# ============================================================================
print("\nGenerating cutoff frequency comparison figure...")

# Load EEG data to compute cutoff frequencies
X_RS_allPat = gsp.load_EEG_example("./DATA/EEG")

# Compute cutoff frequencies for each alignment method
ls_cutoff = []
ls_cutoff_ref = []
ls_cutoff_rotated = []
ls_cutoff_ortho_rotated = []
ls_cutoff_matched = []

for p in np.arange(len(X_RS_allPat)):
    X_RS = X_RS_allPat[p]['X_RS']
    
    PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, X_RS)
    ls_cutoff.append(NN)
    
    PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ref, X_RS)
    ls_cutoff_ref.append(NN)
    
    PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_rotated, X_RS)
    ls_cutoff_rotated.append(NN)
    
    PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_ortho_rotated, X_RS)
    ls_cutoff_ortho_rotated.append(NN)
    
    PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_matched, X_RS)
    ls_cutoff_matched.append(NN)

print(f"Cutoff frequencies computed for {len(X_RS_allPat)} subjects")

# Figure 3: Cutoff frequency comparison - boxplot and scatter
from scipy.stats import pearsonr as scipy_pearsonr, ttest_rel

fig3, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Panel A: Boxplot with individual data points
bp = axs[0].boxplot([ls_cutoff_ref, ls_cutoff, ls_cutoff_rotated, ls_cutoff_ortho_rotated, ls_cutoff_matched],
                     tick_labels=['SC HC (ref)', 'SC IND', 'Gen. Procrustes', 'Ortho. Procrustes', 'Hungarian'], 
                     patch_artist=True, widths=0.6)

# Color boxes according to method
for i, (patch, color) in enumerate(zip(bp['boxes'], ['#1f77b4', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_linewidth(1.5)

for whisker in bp['whiskers']:
    whisker.set(linewidth=1.5, color='black')
for cap in bp['caps']:
    cap.set(linewidth=1.5, color='black')
for median in bp['medians']:
    median.set(linewidth=2, color='black')

# Overlay scatter points with jitter
np.random.seed(42)
jitter_strength = 0.04
positions = np.arange(1, 6)
data_list = [ls_cutoff_ref, ls_cutoff, ls_cutoff_rotated, ls_cutoff_ortho_rotated, ls_cutoff_matched]
colors_list = ['#1f77b4', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

for pos, data, color in zip(positions, data_list, colors_list):
    x_jitter = np.random.normal(pos, jitter_strength, size=len(data))
    axs[0].scatter(x_jitter, data, alpha=0.4, s=30, color=color, edgecolors='none')

# Add pairwise significance tests
t1, p1 = ttest_rel(ls_cutoff_ref, ls_cutoff)
t2, p2 = ttest_rel(ls_cutoff_ref, ls_cutoff_rotated)
t3, p3 = ttest_rel(ls_cutoff_ref, ls_cutoff_ortho_rotated)
t4, p4 = ttest_rel(ls_cutoff_ref, ls_cutoff_matched)

# Get y-axis limits for significance bars
y_max = max(max(ls_cutoff), max(ls_cutoff_ref), max(ls_cutoff_rotated), max(ls_cutoff_ortho_rotated), max(ls_cutoff_matched))
y_min = min(min(ls_cutoff), min(ls_cutoff_ref), min(ls_cutoff_rotated), min(ls_cutoff_ortho_rotated), min(ls_cutoff_matched))
y_range = y_max - y_min
bar_height = y_range * 0.05
bar_y = y_max + y_range * 0.05

# Draw significance bars for p < 0.05
if p1 < 0.05:
    axs[0].plot([1, 2], [bar_y, bar_y], 'k-', linewidth=1.5)
    axs[0].text(1.5, bar_y + bar_height*0.5, f'p={p1:.3e}', ha='center', va='bottom', fontsize=7)
    bar_y += bar_height * 2

if p2 < 0.05:
    axs[0].plot([1, 3], [bar_y, bar_y], 'k-', linewidth=1.5)
    axs[0].text(2, bar_y + bar_height*0.5, f'p={p2:.3e}', ha='center', va='bottom', fontsize=7)
    bar_y += bar_height * 2

if p3 < 0.05:
    axs[0].plot([1, 4], [bar_y, bar_y], 'k-', linewidth=1.5)
    axs[0].text(2.5, bar_y + bar_height*0.5, f'p={p3:.3e}', ha='center', va='bottom', fontsize=7)
    bar_y += bar_height * 2

if p4 < 0.05:
    axs[0].plot([1, 5], [bar_y, bar_y], 'k-', linewidth=1.5)
    axs[0].text(3, bar_y + bar_height*0.5, f'p={p4:.3e}', ha='center', va='bottom', fontsize=7)

axs[0].set_ylabel('Cutoff frequency (Hz)', fontsize=13, fontweight='bold')
axs[0].set_title('A. Cutoff frequency distribution', fontsize=13, fontweight='bold', loc='left', pad=10)
axs[0].tick_params(axis='x', rotation=45, labelsize=10)
axs[0].tick_params(axis='y', labelsize=11)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_linewidth(1.5)
axs[0].spines['bottom'].set_linewidth(1.5)

# Panel B: Scatter plot - SC IND vs aligned methods
axs[1].scatter(ls_cutoff, ls_cutoff_ref, alpha=0.7, s=80, color='#1f77b4', label='SC HC (ref)', edgecolors='white', linewidth=0.5, marker='s')
axs[1].scatter(ls_cutoff, ls_cutoff_rotated, alpha=0.7, s=80, color='#2ca02c', label='Gen. Procrustes', edgecolors='white', linewidth=0.5, marker='^')
axs[1].scatter(ls_cutoff, ls_cutoff_ortho_rotated, alpha=0.7, s=80, color='#ff7f0e', label='Ortho. Procrustes', edgecolors='white', linewidth=0.5, marker='v')
axs[1].scatter(ls_cutoff, ls_cutoff_matched, alpha=0.7, s=80, color='#d62728', label='Hungarian', edgecolors='white', linewidth=0.5)

# Add regression lines
x_range = np.linspace(np.min(ls_cutoff), np.max(ls_cutoff), 100)

# SC HC (ref) regression
z0 = np.polyfit(ls_cutoff, ls_cutoff_ref, 1)
p0 = np.poly1d(z0)
axs[1].plot(x_range, p0(x_range), color='#1f77b4', linewidth=2, linestyle='--', alpha=0.8)
r0, p0_val = scipy_pearsonr(ls_cutoff, ls_cutoff_ref)

# Gen. Procrustes regression
z1 = np.polyfit(ls_cutoff, ls_cutoff_rotated, 1)
p1 = np.poly1d(z1)
axs[1].plot(x_range, p1(x_range), color='#2ca02c', linewidth=2, linestyle='--', alpha=0.8)
r1, p1_val = scipy_pearsonr(ls_cutoff, ls_cutoff_rotated)

# Ortho. Procrustes regression
z2 = np.polyfit(ls_cutoff, ls_cutoff_ortho_rotated, 1)
p2 = np.poly1d(z2)
axs[1].plot(x_range, p2(x_range), color='#ff7f0e', linewidth=2, linestyle='--', alpha=0.8)
r2, p2_val = scipy_pearsonr(ls_cutoff, ls_cutoff_ortho_rotated)

# Hungarian regression
z3 = np.polyfit(ls_cutoff, ls_cutoff_matched, 1)
p3 = np.poly1d(z3)
axs[1].plot(x_range, p3(x_range), color='#d62728', linewidth=2, linestyle='--', alpha=0.8)
r3, p3_val = scipy_pearsonr(ls_cutoff, ls_cutoff_matched)

axs[1].set_xlabel('SC IND cutoff (Hz)', fontsize=13, fontweight='bold')
axs[1].set_ylabel('Aligned cutoff (Hz)', fontsize=13, fontweight='bold')
axs[1].set_title(f'B. SC IND vs aligned\nSC HC (r={r0:.3f}, p={p0_val:.3e}), Gen.P (r={r1:.3f}, p={p1_val:.3e})\nOrtho.P (r={r2:.3f}, p={p2_val:.3e}), Hung (r={r3:.3f}, p={p3_val:.3e})', 
                fontsize=11, fontweight='bold', loc='left', pad=10)
axs[1].legend(fontsize=10, frameon=False, loc='upper left')
axs[1].tick_params(labelsize=11)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_linewidth(1.5)
axs[1].spines['bottom'].set_linewidth(1.5)

fig3.suptitle('Cutoff frequency comparison across alignment methods', fontsize=16, fontweight='bold', y=1.02)

fig3_path_png = os.path.join(figures_dir, 'FigS5_cutoff_frequency_comparison.png')
plt.savefig(fig3_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Cutoff frequency figure saved as '{fig3_path_png}'")

plt.show()
