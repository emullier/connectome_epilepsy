
"""
Manuscript-ready analysis concatenating methodological choice and alignment comparison 

Part 1 : Methodological choice analysis
- Creates consensus matrices with varying numbers of participants (n=1,5,10,20,25)
- Applies different alignment methods (Generalized Procrustes, Orthogonal Procrustes, Hungarian)
- Measures variability within each consensus size
- Generates Figure 1: Within-bin variability (2x2 grid)

Part 2 : Alignment comparison to reference
- Compares SC IND (27 controls) vs SC HC (Geneva dataset)
- Shows how alignment methods improve similarity to reference
- Generates Figure 2: Similarity to reference consensus (1x3 grid; no Orthogonal panel)
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
import lib.func_GSP as gsp
from lib import fcn_groups_bin
from lib.func_plot import plot_rois_pyvista_noaxes
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kruskal, levene, mannwhitneyu, friedmanchisquare
from scipy.spatial import procrustes
from scipy.interpolate import make_interp_spline


# Add project root to import custom libs
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

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
# Use SC IND matrices for Fig4/Fig5 analyses
matMetric_ind = np.load(os.path.join(project_root, "DATA", "SC", "matMetric_SCHZ_CTRL.npy"))
if matMetric_ind.ndim == 3 and matMetric_ind.shape[0] != matMetric_ind.shape[1]:
    matMetric = np.transpose(matMetric_ind, (1, 2, 0))
else:
    matMetric = matMetric_ind

# Match Fig4 pipeline to cortical ROIs used to build Euc
if matMetric.shape[0] == 118:
    cort_rois = np.concatenate((np.arange(0, 57), np.arange(59, 116)))
else:
    cort_rois_raw = np.where(roi_info['Structure'] == 'cort')[0]
    cort_rois = cort_rois_raw[cort_rois_raw < matMetric.shape[0]]

matMetric = matMetric[cort_rois, :, :]
matMetric = matMetric[:, cort_rois, :]
x = np.asarray(roi_info['x-pos'])[cort_rois]
y = np.asarray(roi_info['y-pos'])[cort_rois]
z = np.asarray(roi_info['z-pos'])[cort_rois]
coordMat = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))

# Parameters
ls_bins = [1, 9]
nbPerm = 5
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

Dist_eigvec_perm_vec_nz = Dist_eigvec_perm_vec
Dist_eigvec_perm_ortho_vec_nz = Dist_eigvec_perm_ortho_vec
Dist_eigvec_perm_rot_vec_nz = Dist_eigvec_perm_rot_vec
Dist_eigvec_perm_matched_vec_nz = Dist_eigvec_perm_matched_vec

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


# ============================================================================
# Figure 4 - Statistical tests: bin size effect on mean similarity
# Spearman rank correlation: does increasing bin size increase mean similarity?
# Kruskal-Wallis: do means differ across bin sizes?
# Levene's test: do variances differ across bin sizes?
# ============================================================================

# Compute mean similarities for each bin size and alignment method
bin_sizes = np.array(ls_bins)
mean_similarities = {'raw': [], 'ortho': [], 'rotated': [],'matched': [] }
std_similarities = {'raw': [],'ortho': [],'rotated': [],'matched': []}

for b, bi in enumerate(ls_bins):
    mean_similarities['raw'].append(np.mean(bin_variability[b, :, 0]))
    std_similarities['raw'].append(np.std(bin_variability[b, :, 0]))
    mean_similarities['ortho'].append(np.mean(bin_variability_ortho[b, :, 0]))
    std_similarities['ortho'].append(np.std(bin_variability_ortho[b, :, 0]))
    mean_similarities['rotated'].append(np.mean(bin_variability_rot[b, :, 0]))
    std_similarities['rotated'].append(np.std(bin_variability_rot[b, :, 0]))
    mean_similarities['matched'].append(np.mean(bin_variability_matched[b, :, 0]))
    std_similarities['matched'].append(np.std(bin_variability_matched[b, :, 0]))

mean_similarities = {k: np.array(v) for k, v in mean_similarities.items()}
std_similarities = {k: np.array(v) for k, v in std_similarities.items()}

# Compute Spearman rank correlation for each method
spearman_results = {}
spearman_harmonic_results = {}
kruskal_results = {}
levene_results = {}
method_names_fig4 = ['raw', 'rotated', 'matched']
method_labels_fig4 = ['Raw', 'Procrustes', 'Hungarian matching']

print("\n" + "="*80)
print("Figure 4 — Statistical tests: effect of bin size on similarity metrics")
print("="*80)

# Prepare data for Kruskal-Wallis and Levene tests
# Each bin's data is bin_variability[b, :, 0] (all harmonics for that bin)
kw_data_by_method = {
    'raw': [bin_variability[b, :, 0] for b in range(len(ls_bins))],
    'rotated': [bin_variability_rot[b, :, 0] for b in range(len(ls_bins))],
    'matched': [bin_variability_matched[b, :, 0] for b in range(len(ls_bins))]
}

print(f"\n{'Method':<25} {'Spearman ρ':>12} {'Kruskal-Wallis H':>18} {'Levene F':>12}")
print(f"{'':25} {'p-value':>12} {'p-value':>18} {'p-value':>12}")
print("-" * 80)

for method, label in zip(method_names_fig4, method_labels_fig4):
    # Spearman correlation
    rho, p_spear = spearmanr(bin_sizes, mean_similarities[method])
    sig_spear = '***' if p_spear < 0.001 else ('**' if p_spear < 0.01 else ('*' if p_spear < 0.05 else 'n.s.'))
    spearman_results[method] = {'rho': rho, 'p': p_spear, 'sig': sig_spear}
    
    # Kruskal-Wallis test (compare means across bin sizes)
    H, p_kw = kruskal(*kw_data_by_method[method])
    sig_kw = '***' if p_kw < 0.001 else ('**' if p_kw < 0.01 else ('*' if p_kw < 0.05 else 'n.s.'))
    kruskal_results[method] = {'H': H, 'p': p_kw, 'sig': sig_kw}
    
    # Levene's test (compare variances across bin sizes)
    F, p_lev = levene(*kw_data_by_method[method])
    sig_lev = '***' if p_lev < 0.001 else ('**' if p_lev < 0.01 else ('*' if p_lev < 0.05 else 'n.s.'))
    levene_results[method] = {'F': F, 'p': p_lev, 'sig': sig_lev}
    
    print(f"{label:<25} {rho:>12.3f}    {H:>16.3f}    {F:>12.3f}")
    print(f"{'':25} {p_spear:>12.3e}    {p_kw:>16.3e}    {p_lev:>12.3e}")
    print(f"{'':25} {sig_spear:>12}    {sig_kw:>16}    {sig_lev:>12}")
    print()

    # Harmonic-level Spearman correlation (more informative than 5-point mean curve)
    x_rep = np.concatenate([np.full(len(arr), bin_sizes[idx]) for idx, arr in enumerate(kw_data_by_method[method])])
    y_rep = np.concatenate(kw_data_by_method[method])
    rho_h, p_h = spearmanr(x_rep, y_rep)
    sig_h = '***' if p_h < 0.001 else ('**' if p_h < 0.01 else ('*' if p_h < 0.05 else 'n.s.'))
    spearman_harmonic_results[method] = {'rho': rho_h, 'p': p_h, 'sig': sig_h}

print("-" * 80)
print("Spearman: correlation between bin size and mean similarity (monotonic trend)")
print("Kruskal-Wallis: test if mean similarity differs across bin sizes (non-parametric)")
print("Levene: test if variance in similarity differs across bin sizes\n")

print("Harmonic-level Spearman (bin size vs per-harmonic similarity):")
for method, label in zip(method_names_fig4, method_labels_fig4):
    rho_h = spearman_harmonic_results[method]['rho']
    p_h = spearman_harmonic_results[method]['p']
    sig_h = spearman_harmonic_results[method]['sig']
    print(f"  {label:<25} rho={rho_h:.3f}, p={p_h:.3e} {sig_h}")
print()

# FIGURE 4 - Main figure (3x1): Raw, Procrustes, Hungarian
fig, ax = plt.subplots(3, 1, figsize=(8.5, 13), constrained_layout=True)
# Generate color palette for 8 bin sizes using a colormap
cmap_fig4 = plt.cm.get_cmap('tab20')
colors_palette = [cmap_fig4(i / len(ls_bins)) for i in range(len(ls_bins))]
handles = []

for b, bi in enumerate(ls_bins):
    line, = ax[0].plot(bin_variability[b, :, 0], linewidth=2, color=colors_palette[b], label=f'n={bi}')
    if b < len(ls_bins):
        handles.append(line)
    ax[1].plot(bin_variability_rot[b, :, 0], linewidth=2, color=colors_palette[b], label=f'n={bi}')
    ax[2].plot(bin_variability_matched[b, :, 0], linewidth=2, color=colors_palette[b], label=f'n={bi}')

    mean_raw = np.mean(bin_variability[b, :, 0])
    std_raw = np.std(bin_variability[b, :, 0])
    mean_rotated = np.mean(bin_variability_rot[b, :, 0])
    std_rotated = np.std(bin_variability_rot[b, :, 0])
    mean_matched = np.mean(bin_variability_matched[b, :, 0])
    std_matched = np.std(bin_variability_matched[b, :, 0])

    ax[0].axhline(y=mean_raw, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[0].text(nb_eig2keep + 1, mean_raw, f'{mean_raw:.2f}±{std_raw:.2f}', color=colors_palette[b], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[1].axhline(y=mean_rotated, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[1].text(nb_eig2keep + 1, mean_rotated, f'{mean_rotated:.2f}±{std_rotated:.2f}', color=colors_palette[b], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[2].axhline(y=mean_matched, color=colors_palette[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[2].text(nb_eig2keep + 1, mean_matched, f'{mean_matched:.2f}±{std_matched:.2f}', color=colors_palette[b], fontsize=8, va='center', ha='left', fontweight='bold')

    upper_bound = bin_variability[b, :, 0] + bin_variability[b, :, 1]
    lower_bound = bin_variability[b, :, 0] - bin_variability[b, :, 1]
    upper_bound_matched = bin_variability_matched[b, :, 0] + bin_variability_matched[b, :, 1]
    lower_bound_matched = bin_variability_matched[b, :, 0] - bin_variability_matched[b, :, 1]
    upper_bound_rotated = bin_variability_rot[b, :, 0] + bin_variability_rot[b, :, 1]
    lower_bound_rotated = bin_variability_rot[b, :, 0] - bin_variability_rot[b, :, 1]

    ax[0].fill_between(range(nb_eig2keep), lower_bound, upper_bound, alpha=0.15, color=colors_palette[b])
    ax[1].fill_between(range(nb_eig2keep), lower_bound_rotated, upper_bound_rotated, alpha=0.15, color=colors_palette[b])
    ax[2].fill_between(range(nb_eig2keep), lower_bound_matched, upper_bound_matched, alpha=0.15, color=colors_palette[b])

for x in [0, 1, 2]:
    ax[x].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
    ax[x].set_xticks(range(0, nb_eig2keep, 20))
    ax[x].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax[x].set_ylim([0, 1.02])
    ax[x].set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax[x].spines['top'].set_visible(False)
    ax[x].spines['right'].set_visible(False)
    ax[x].tick_params(labelsize=10)

ax[0].set_title('A. Raw eigenvector similarity',
                fontsize=13, fontweight='bold', loc='left', pad=10)
ax[1].set_title('B. Procrustes',
                fontsize=13, fontweight='bold', loc='left', pad=10)
ax[2].set_title('C. Hungarian matching',
                fontsize=13, fontweight='bold', loc='left', pad=10)

fig.legend(handles=handles, labels=[f'n={bi}' for bi in ls_bins],
           loc='lower center', frameon=True, fancybox=False, shadow=False,
           fontsize=10, title='Consensus size', title_fontsize=10,
           ncol=4, bbox_to_anchor=(0.5, -0.01))

fig_path_png = os.path.join(figures_dir, 'Fig4_bin_variability_analysis.png')
plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Plot saved as '{fig_path_png}'")
plt.close()



# ============================================================================
# PART 2: Comparison between SC IND and SC HC datasets (from script 10)
# ============================================================================
print("\n" + "="*80)
print("PART 2: Comparing SC IND (27 controls) vs SC HC reference")
print("="*80)

print("Loading SC IND and SC HC datasets...")
consensus_HC_DSI = np.load(os.path.join(project_root, "DATA", "SC", "matMetric_HC_dsi_number_of_fibers.npy"))
consensus_IND = np.load(os.path.join(project_root, "DATA", "SC", "matMetric_SCHZ_CTRL.npy"))
EucDist_ref = np.load(os.path.join(project_root, "DATA", "EucMat", "EucMat_HC_dsi_number_of_fibers.npy"))

consensus_HC_ref = np.mean(consensus_HC_DSI, axis=2)
consensus_IND_mean = np.mean(consensus_IND, axis=0)

print("Generating harmonics from both consensus matrices...")
P_ref, Q_ref, Ln_ref, An_ref = gsp.cons_normalized_lap(consensus_HC_ref, EucDist_ref, plot=False)
P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus_IND_mean, EucDist_ref, plot=False)

print("Applying alignment methods...")
Qind_rotated_raw, Qind_HC_centered, disparity = scipy.spatial.procrustes(Q_ref, Q_ind)
_Uind, _, _Vtind = scipy.linalg.svd(Qind_rotated_raw, full_matrices=False)
Qind_rotated = _Uind @ _Vtind
perm_ind, total_cost_ind = gsp.match_eigenvectors(Q_ref, Q_ind)
Qind_matched = Q_ind[:, perm_ind]

print("Computing similarity metrics...")
nb_eig = Q_ref.shape[1]
similarity_ind = np.zeros(nb_eig)
similarity_rotated = np.zeros(nb_eig)
similarity_matched = np.zeros(nb_eig)

for eigvec_nb in range(nb_eig):
    similarity_ind[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Q_ind[:, eigvec_nb])
    similarity_rotated[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_rotated[:, eigvec_nb])
    similarity_matched[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_matched[:, eigvec_nb])

similarity_ind = np.abs(similarity_ind)
similarity_rotated = np.abs(similarity_rotated)
similarity_matched = np.abs(similarity_matched)

print("Loading EEG data and SDI summaries...")
X_RS_allPat = gsp.load_EEG_example(example_dir)
lat_labels = np.array([str(patient['lat'][0]) for patient in X_RS_allPat])
lateralizations = {
    'LT': np.where(lat_labels == 'Ltle')[0],
    'RT': np.where(lat_labels == 'Rtle')[0],
}
nbSurr = 100

methods = {
    'SC_HC_ref': Q_ref,
    'SC_IND': Q_ind,
    'Gen_Procrustes': Qind_rotated,
    'Hungarian': Qind_matched,
}

sdi_results = {}
method_file_prefixes = {
    'SC_HC_ref': 'SC_HC_ref',
    'SC_IND': 'SC_IND',
    'Gen_Procrustes': 'Gen_Procrustes',
    'Hungarian': 'Hungarian',
}
cutoff_file_prefixes = {
    'SC_HC_ref': 'HC',
    'SC_IND': 'IND',
    'Gen_Procrustes': 'Gen_Procrustes',
    'Hungarian': 'Hungarian',
}

for method_name, Q_method in methods.items():
    for lateralization in ['LT', 'RT']:
        surr_thresh_path = os.path.join(output_dir, f"SDI_surr_thresh_{method_file_prefixes[method_name]}_{lateralization}.npy")
        surr_thresh = np.load(surr_thresh_path, allow_pickle=True)

        cutoff_values = []
        for patient in X_RS_allPat:
            _, NN, _, _ = gsp.get_cutoff_freq(Q_method, patient['X_RS'])
            cutoff_values.append(NN)
        cutoff_values = np.array(cutoff_values)

        cutoff_path = os.path.join(output_dir, f"cutoff_{cutoff_file_prefixes[method_name]}_{lateralization}.npy")
        np.save(cutoff_path, cutoff_values)

        sdi_results[f"{method_name}_{lateralization}"] = {
            'surr_thresh': surr_thresh,
            'cutoff_frequencies': cutoff_values,
        }


# ============================================================================
# Figure S5: Cutoff frequency comparison across alignment methods
# ============================================================================
print("\nGenerating Figure S5: Cutoff frequency comparison...")

# Create figure with two subplots (LT and RT)
fig_s5, axs_s5 = plt.subplots(1, 2, figsize=(14, 6))

box_labels = ['SC HC (ref)', 'SC IND', 'Procrustes', 'Hungarian']
box_palette = ['#1f77b4', '#1f77b4', '#2ca02c', '#d62728']
comparisons = [(0, 1), (0, 2), (0, 3)]

for panel_idx, lateralization in enumerate(['LT', 'RT']):
    ax_s5 = axs_s5[panel_idx]
    cutoff_sc_hc = sdi_results[f'SC_HC_ref_{lateralization}']['cutoff_frequencies']
    cutoff_sc_ind = sdi_results[f'SC_IND_{lateralization}']['cutoff_frequencies']
    cutoff_gen_procrustes = sdi_results[f'Gen_Procrustes_{lateralization}']['cutoff_frequencies']
    cutoff_hungarian = sdi_results[f'Hungarian_{lateralization}']['cutoff_frequencies']

    box_data = [cutoff_sc_hc, cutoff_sc_ind, cutoff_gen_procrustes, cutoff_hungarian]

    bp = ax_s5.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (data, color) in enumerate(zip(box_data, box_palette)):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax_s5.scatter(x, data, alpha=0.4, s=30, color=color, edgecolors='none')

    if panel_idx == 0:
        panel_title = 'A. LT cutoff frequency distribution'
    else:
        panel_title = 'B. RT cutoff frequency distribution'

    ax_s5.set_title(panel_title, fontsize=12, fontweight='bold', loc='left')
    ax_s5.set_ylabel('Cutoff frequency (Hz)', fontsize=12, fontweight='bold')
    ax_s5.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax_s5.tick_params(labelsize=10)
    ax_s5.set_xticklabels(box_labels, rotation=15, ha='right', fontsize=10)

    y_max = max([np.max(d) for d in box_data])
    y_step = 0.05 * y_max
    for i, (x1, x2) in enumerate(comparisons):
        stat, pval = mannwhitneyu(box_data[x1], box_data[x2], alternative='two-sided')
        y = y_max + (i + 1) * y_step * 2
        ax_s5.plot([x1 + 1, x1 + 1, x2 + 1, x2 + 1], [y, y + y_step * 0.5, y + y_step * 0.5, y], 'k-', linewidth=1)
        ax_s5.text((x1 + x2) / 2 + 1, y + y_step * 0.5, f"p={pval:.3e}", ha='center', va='bottom', fontsize=8)

    ax_s5.set_ylim(top=y_max + (len(comparisons) + 2) * y_step * 2)

fig_s5.suptitle('Cutoff frequency comparison across alignment methods', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

fig_s5_path = os.path.join(figures_dir, 'FigS5_cutoff_frequency_comparison_LT_RT.png')
plt.savefig(fig_s5_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure S5 saved as '{fig_s5_path}'")
plt.close()

# ============================================================================
# PyVista Brain Visualization: Apply thr=5 threshold like in Generation script
# ============================================================================
print("\nGenerating PyVista brain visualizations with thr=5 thresholding...")

thr = 5  # Same threshold index as in Generation script

for lateralization in ['LT', 'RT']:
    print(f"\n  Plotting {lateralization} lateralization...")
    for method_name in ['SC_HC_ref', 'SC_IND', 'Gen_Procrustes', 'Hungarian']:
        key = f"{method_name}_{lateralization}"
        surr_thresh = sdi_results[key]['surr_thresh']
        
        # Apply same formula as Generation script: mean_SDI * abs(SDI_sig)
        thresholded_sdi = surr_thresh[thr]['mean_SDI'] * np.abs(surr_thresh[thr]['SDI_sig'])
        # plot_rois_pyvista_noaxes(thresholded_sdi, scale=2, out_dir=figures_dir, vmin=-2, vmax=2, center_at_zero=True, label=f'FigS7_SDI_thr{thr}_{method_name}_{lateralization}',cmap='coolwarm',fmt='png')

print(f"Thresholded SDI brain visualizations saved to {figures_dir}")

thr = 0
for lateralization in ['LT', 'RT']:
    print(f"\n  Plotting {lateralization} lateralization...")
    for method_name in ['SC_HC_ref', 'SC_IND', 'Gen_Procrustes', 'Hungarian']:
        key = f"{method_name}_{lateralization}"
        surr_thresh = sdi_results[key]['surr_thresh']
        
        # Apply same formula as Generation script: mean_SDI * abs(SDI_sig)
        thresholded_sdi = surr_thresh[thr]['mean_SDI'] * np.abs(surr_thresh[thr]['SDI_sig'])
        #plot_rois_pyvista_noaxes(thresholded_sdi, scale=2, out_dir=figures_dir,vmin=-2,vmax=2,center_at_zero=True,label=f'FigS7_SDI_thr{thr}_{method_name}_{lateralization}',cmap='coolwarm',fmt='png')

print(f"Thresholded SDI brain visualizations saved to {figures_dir}")

# ============================================================================
# FigS7bis: Summary brain plot - ROI activation consistency across alignments
# ============================================================================
print("\n" + "="*80)
print("Creating FigS7bis - Summary brain plot showing ROI activation consistency")
print("="*80)

thr = 5  # Use same threshold as main FigS7
alignment_methods = ['SC_HC_ref', 'SC_IND', 'Gen_Procrustes', 'Hungarian']

for lateralization in ['LT', 'RT']:
    print(f"\nGenerating FigS7bis for {lateralization}...")
    
    # Initialize summary vector (118 ROIs)
    summary_vector = np.zeros(118)
    
    # For each ROI, count how many alignment methods have it as significant at thr=5
    for roi_idx in range(118):
        count = 0
        for method_name in alignment_methods:
            key = f"{method_name}_{lateralization}"
            surr_thresh = sdi_results[key]['surr_thresh']
            # Check if this ROI is significant in this method at threshold 5
            if surr_thresh[thr]['SDI_sig'][roi_idx] != 0:
                count += 1
        summary_vector[roi_idx] = count
    
    # Only show ROIs that are active in at least one alignment method (values 1-4)
    summary_vector_masked = np.where(summary_vector > 0, summary_vector, np.nan)
    
    # Create brain plot with values indicating consistency across methods
    # plot_rois_pyvista_noaxes(summary_vector_masked, scale=2, out_dir=figures_dir, vmin=1, vmax=4, cmap='YlOrRd',label=f'FigS7bis_summary_alignment_consistency_{lateralization}')
    
    print(f"  - ROIs active in 1 alignment method: {np.sum(summary_vector == 1)}")
    print(f"  - ROIs active in 2 alignment methods: {np.sum(summary_vector == 2)}")
    print(f"  - ROIs active in 3 alignment methods: {np.sum(summary_vector == 3)}")
    print(f"  - ROIs active in all 4 alignment methods: {np.sum(summary_vector == 4)}")

# ============================================================================
print("\n" + "="*80)
print("Creating comparison table for alignment methods (LT and RT)")
print("="*80)

# Load ROI labels
df_roi = pd.read_csv(os.path.join(project_root, 'DATA/label/labels_rois_118.csv'))
labels_118 = np.array(df_roi['Label Lausanne2008'])

# Collect all indices that are significant in any method or lateralization (threshold = 5)
all_idx = set()
for key in [
    'SC_HC_ref_LT', 'SC_IND_LT', 'Gen_Procrustes_LT', 'Hungarian_LT',
    'SC_HC_ref_RT', 'SC_IND_RT', 'Gen_Procrustes_RT', 'Hungarian_RT'
]:
    surr_thresh = sdi_results[key]['surr_thresh']
    sig_idx = np.where(surr_thresh[thr]['SDI_sig'] != 0)[0]
    all_idx.update(sig_idx)
all_idx = sorted(list(all_idx))

# Build a dictionary for DataFrame with MultiIndex columns
data = {("ROI", ""): [labels_118[idx] for idx in all_idx]}

for lateralization in ['LT', 'RT']:
    for method_name in ['SC_HC_ref', 'SC_IND', 'Gen_Procrustes', 'Hungarian']:
        col_name = (lateralization, method_name)
        values = []
        key = f"{method_name}_{lateralization}"
        surr_thresh = sdi_results[key]['surr_thresh']
        for idx in all_idx:
            if surr_thresh[thr]['SDI_sig'][idx] != 0:
                values.append(round(surr_thresh[thr]['mean_SDI'][idx], 2))
            else:
                values.append(np.nan)
        data[col_name] = values

# Create DataFrame with MultiIndex columns
df_comparison = pd.DataFrame(data)
df_comparison.columns = pd.MultiIndex.from_tuples(df_comparison.columns)

# Print table
print("\n" + "="*80)
print(f"Significant ROIs across alignment methods (threshold={thr})")
print("="*80)
print(df_comparison)

# Export to Excel
excel_path = "SDI_alignment_comparison_table.xlsx"
df_comparison.to_excel(excel_path, index=True)
print(f"\nTable saved to: {excel_path}")

# Summary statistics
print("\n" + "="*80)
print("Summary: Number of significant ROIs per method and lateralization")
print("="*80)
for lateralization in ['LT', 'RT']:
    print(f"\n{lateralization}:")
    for method_name in ['SC_HC_ref', 'SC_IND', 'Gen_Procrustes', 'Hungarian']:
        key = f"{method_name}_{lateralization}"
        surr_thresh = sdi_results[key]['surr_thresh']
        n_sig = len(np.where(surr_thresh[thr]['SDI_sig'] != 0)[0])
        print(f"  {method_name:20s}: {n_sig:3d} ROIs")

# ============================================================================
# Figure S6: Number of significant ROIs across thresholds and alignment methods
# ============================================================================
print("\nGenerating Figure S6: Number of significant ROIs across thresholds...")

# Count significant ROIs for each method, lateralization, and threshold
# Check how many thresholds are available
n_thresholds = len(sdi_results['SC_HC_ref_LT']['surr_thresh'])
method_names_list = ['SC_HC_ref', 'SC_IND', 'Gen_Procrustes', 'Hungarian']
colors_methods = ['#1f77b4', '#1f77b4', '#2ca02c', '#d62728']
markers_methods = ['s', 'o', '^', 'd']

fig_s6, axes_s6 = plt.subplots(1, 2, figsize=(16, 6))

for lat_idx, lateralization in enumerate(['LT', 'RT']):
    ax = axes_s6[lat_idx]
    lateralization_label = 'Left IED' if lateralization == 'LT' else 'Right IED'
    
    for method_idx, method_name in enumerate(method_names_list):
        key = f"{method_name}_{lateralization}"
        surr_thresh = sdi_results[key]['surr_thresh']
        
        # Count significant ROIs for each threshold
        n_sig_per_threshold = []
        for thr_idx in range(n_thresholds):
            n_sig = len(np.where(surr_thresh[thr_idx]['SDI_sig'] != 0)[0])
            n_sig_per_threshold.append(n_sig)
        
        # Plot with method-specific color and marker
        ax.plot(range(n_thresholds), n_sig_per_threshold, 
                marker=markers_methods[method_idx], 
                linewidth=2.5, 
                markersize=8,
                color=colors_methods[method_idx],
                label=method_name.replace('_', ' '))
        
        # Add value labels on markers
        for thr_idx, n_sig in enumerate(n_sig_per_threshold):
            ax.text(thr_idx, n_sig + 2, f'{int(n_sig)}', 
                   fontsize=7, ha='center', va='bottom', 
                   color=colors_methods[method_idx], fontweight='bold')
    
    ax.set_xlabel('Threshold (# subjects)', fontsize=12, fontweight='bold')
    ax.set_ylabel('# Significant ROIs', fontsize=12, fontweight='bold')
    ax.set_title(f'Number of significant ROIs across alignment methods ({lateralization_label})', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(n_thresholds))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.tick_params(labelsize=10)

plt.tight_layout()

fig_s6_path = os.path.join(figures_dir, 'FigS6_nbROIs_across_methods.png')
plt.savefig(fig_s6_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure S6 saved as '{fig_s6_path}'")
plt.close()

# ============================================================================
# Statistical tests: each alignment method vs no alignment (Fig3a)
# Wilcoxon signed-rank test, paired over harmonics (n=nb_eig values per array)
# alternative='greater' → tests whether alignment INCREASES per-harmonic similarity
# ============================================================================
from scipy.stats import wilcoxon as _wilcoxon

def _wilcoxon_vs_raw(aligned, raw):
    """Return (W, p, sig_label) comparing aligned > raw over harmonics."""
    diff = aligned - raw
    if np.all(diff == 0):
        return np.nan, np.nan, 'n.s.'
    W, p = _wilcoxon(aligned, raw, alternative='greater')
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    return W, p, sig

W_rotated, p_rotated, sig_rotated = _wilcoxon_vs_raw(similarity_rotated,  similarity_ind)
W_matched, p_matched, sig_matched = _wilcoxon_vs_raw(similarity_matched,  similarity_ind)

print("\n" + "="*80)
print("Fig3a — Statistical tests: aligned vs no-alignment (Wilcoxon signed-rank, paired over harmonics)")
print("  H1 (alternative='greater'): alignment increases per-harmonic similarity")
print("="*80)
print(f"  {'Method':<25} {'W':>10}  {'p-value':>12}  {'Δmean':>8}  {'sig':>5}")
print("  " + "-"*65)
for lbl, W, p, sig, arr in [
    ('Procrustes',           W_rotated, p_rotated, sig_rotated, similarity_rotated),
    ('Hungarian',            W_matched, p_matched, sig_matched, similarity_matched),
]:
    delta = np.mean(arr) - np.mean(similarity_ind)
    print(f"  {lbl:<25} {W:>10.1f}  {p:>12.4e}  {delta:>+8.4f}  {sig:>5}")
print("  " + "-"*65)
print("  Δmean = mean(aligned) − mean(raw);  positive = alignment improves similarity")

# Figure 2: Similarity between SC HC (ref) and SC IND harmonics - 1x3 layout
fig2, ax2 = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

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
ax2[0].set_title(f'A. Before alignment (mean r={mean_ind:.3f}±{std_ind:.3f})', fontsize=13, fontweight='bold', loc='left', pad=10)
ax2[0].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
ax2[0].set_ylabel('Correlation', fontsize=12, fontweight='bold')

# B. Procrustes
ax2[1].plot(harmonic_indices, similarity_rotated, linewidth=2, color=colors[2])
mean_rotated = np.mean(similarity_rotated)
std_rotated = np.std(similarity_rotated)
ax2[1].axhline(y=mean_rotated, color=colors[2], linestyle=':', linewidth=1.5, alpha=0.7)
ax2[1].text(n_harmonics+1, mean_rotated, f'{mean_rotated:.3f}±{std_rotated:.3f}', color=colors[2], fontsize=8, va='center', ha='left', fontweight='bold')
ax2[1].fill_between(harmonic_indices, similarity_rotated - std_rotated, similarity_rotated + std_rotated, alpha=0.15, color=colors[2])
ax2[1].set_title(f'B. Procrustes (mean r={mean_rotated:.3f}±{std_rotated:.3f})\nvs before alignment, p={p_rotated:.3e} {sig_rotated} (Wilcoxon signed-rank)',
                 fontsize=11, fontweight='bold', loc='left', pad=10)
ax2[1].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
ax2[1].set_ylabel('Correlation', fontsize=12, fontweight='bold')

# C. Hungarian matching
ax2[2].plot(harmonic_indices, similarity_matched, linewidth=2, color=colors[3])
mean_matched = np.mean(similarity_matched)
std_matched = np.std(similarity_matched)
ax2[2].axhline(y=mean_matched, color=colors[3], linestyle=':', linewidth=1.5, alpha=0.7)
ax2[2].text(n_harmonics+1, mean_matched, f'{mean_matched:.3f}±{std_matched:.3f}', color=colors[3], fontsize=8, va='center', ha='left', fontweight='bold')
ax2[2].fill_between(harmonic_indices, similarity_matched - std_matched, similarity_matched + std_matched, alpha=0.15, color=colors[3])
ax2[2].set_title(f'C. Hungarian matching (mean r={mean_matched:.3f}±{std_matched:.3f})\nvs before alignment, p={p_matched:.3e} {sig_matched} (Wilcoxon signed-rank)',
                 fontsize=11, fontweight='bold', loc='left', pad=10)
ax2[2].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
ax2[2].set_ylabel('Correlation', fontsize=12, fontweight='bold')

# Format all subplots
for i in range(3):
    ax2[i].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax2[i].set_ylim(similarity_ylim)
    ax2[i].set_xticks(range(0, n_harmonics, 20))
    ax2[i].spines['top'].set_visible(False)
    ax2[i].spines['right'].set_visible(False)
    ax2[i].spines['left'].set_linewidth(1.5)
    ax2[i].spines['bottom'].set_linewidth(1.5)
    ax2[i].tick_params(labelsize=11)

fig2_path_png = os.path.join(figures_dir, 'Fig3a_harmonic_similarity_SC_IND_vs_SC_HC.png')
plt.savefig(fig2_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 3a saved as '{fig2_path_png}'")

plt.close()
print(f"Figure 3a saved as '{fig2_path_png}'")

plt.close()



# Define colors for each method
colors = ['#1f77b4', '#2ca02c', '#d62728']
method_names = ['Before alignment', 'Procrustes', 'Hungarian matching']

n_harmonics = len(similarity_ind)
harmonic_indices = np.arange(n_harmonics)

# Figure 3b: Scatter plots of SDI correlation
# ============================================================================
print("\nGenerating scatter plots of SDI correlation (Fig 3b)...")

# Create figure with 1 row x 3 columns for scatter plots
fig3b, axs3b = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

# Get mean SDI values for SC HC ref and each alignment method (using threshold 5)
thr = 5

# SC HC reference SDI (for both LT and RT, we'll use combined for simplicity or pick one lateralization)
# Let's use LT lateralization as reference
sdi_sc_hc_ref = sdi_results['SC_HC_ref_LT']['surr_thresh'][thr]['mean_SDI']

# Find global min/max for consistent axis limits
all_sdi_values = [sdi_sc_hc_ref]
for method_name in ['SC_IND', 'Gen_Procrustes', 'Hungarian']:
    all_sdi_values.append(sdi_results[f'{method_name}_LT']['surr_thresh'][thr]['mean_SDI'])
global_min = np.min([np.min(v) for v in all_sdi_values])
global_max = np.max([np.max(v) for v in all_sdi_values])

# Scatter plots (SDI values: SC HC ref vs aligned methods)
for i, method_name in enumerate(['SC_IND', 'Gen_Procrustes', 'Hungarian']):
    ax = axs3b[i]
    
    # Get SDI values for this method (LT lateralization)
    sdi_method = sdi_results[f'{method_name}_LT']['surr_thresh'][thr]['mean_SDI']
    
    # Calculate correlation and p-value
    from scipy.stats import pearsonr
    r_corr, p_val = pearsonr(sdi_sc_hc_ref, sdi_method)
    
    # Scatter plot
    ax.scatter(sdi_sc_hc_ref, sdi_method, alpha=0.7, s=40, color=colors[i], edgecolors='none')
    slope, intercept = np.polyfit(sdi_sc_hc_ref, sdi_method, 1)
    x_line = np.array([global_min, global_max])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, linestyle='--', linewidth=1.8, color='black', alpha=0.9)
    
    # Formatting with explicit bold styling
    ax.set_xlabel('SDI SC HC', fontsize=12, fontweight='bold', family='sans-serif')
    ax.set_ylabel('SDI SC IND', fontsize=12, fontweight='bold', family='sans-serif')
    title = ax.set_title(f'r={r_corr:.3f}, p={p_val:.2e}', fontsize=11, loc='center', pad=5, family='sans-serif')
    title.set_fontweight('bold')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(labelsize=10)
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([global_min, global_max])
    ax.set_ylim([global_min, global_max])
    
    # Add method label in corner
    ax.text(0.05, 0.95, f'{method_names[i]}', 
            transform=ax.transAxes, fontsize=10, fontweight='bold', family='sans-serif',
            verticalalignment='top', horizontalalignment='left')

plt.tight_layout()

fig3b_path_png = os.path.join(figures_dir, 'Fig3b_harmonic_similarity.png')
plt.savefig(fig3b_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 3b (scatter plots) saved as '{fig3b_path_png}'")

plt.close()


# ============================================================================
# Figure 5: Fully optimized + persistent caching (CLEAN VERSION)
# ============================================================================

import os
import numpy as np
import pickle
import hashlib
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.stats import spearmanr
from scipy.spatial import procrustes

import lib.func_GSP as gsp

# ============================================================================
# PATHS (EDIT THESE IF NEEDED)
# ============================================================================
project_root = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(project_root, "DATA")
figures_dir = os.path.join(project_root, "FIGURES")
cache_dir = os.path.join(project_root, "cache_fig5")

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

example_dir = os.path.join(data_dir, "EEG")

# ============================================================================
# CACHE UTILS
# ============================================================================
def make_cache_key(*args):
    key_str = "_".join(map(str, args))
    return hashlib.md5(key_str.encode()).hexdigest()

def cache_path(prefix, key):
    return os.path.join(cache_dir, f"{prefix}_{key}.npy")

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================
def load_or_compute_laplacian(consensus_SC, perm_idxs, sc_label, p, bi):
    key = make_cache_key("lap", sc_label, p, bi, tuple(perm_idxs))
    fpath = cache_path("lap", key)

    if os.path.exists(fpath):
        return np.load(fpath)

    sub = np.mean(consensus_SC[:, :, perm_idxs], axis=2)
    _, Q, _, _ = gsp.cons_normalized_lap(sub, EucDist_fig5, plot=False)

    np.save(fpath, Q)
    return Q


def compute_sdi_matrix_cached(Q, sc_label, label, p, bi):
    key = make_cache_key("sdimat", sc_label, label, p, bi, Q.shape, float(np.sum(Q)))
    fpath = cache_path("sdimat", key)

    if os.path.exists(fpath):
        return np.load(fpath)

    SDI_mat = np.column_stack([
        gsp.compute_SDI(pat['X_RS'], Q)[0]
        for pat in X_RS_allPat_fig5
    ])

    np.save(fpath, SDI_mat)
    return SDI_mat


def compute_sdi_ref_cached(Q_ref, sc_label):
    key = make_cache_key("sdi_ref", sc_label)
    fpath = cache_path("sdi_ref", key)

    if os.path.exists(fpath):
        return np.load(fpath)

    SDI_ref = np.column_stack([
        gsp.compute_SDI(p['X_RS'], Q_ref)[0]
        for p in X_RS_allPat_fig5
    ])

    np.save(fpath, SDI_ref)
    return SDI_ref


# ============================================================================
# CONFIG
# ============================================================================
print("\nGenerating Figure 5 (optimized & cached)...")

sc_configs = [
    {'label': 'SC-TLE', 'path': os.path.join(data_dir, "SC", "matMetric_HC_dsi_number_of_fibers.npy")},
    {'label': 'SC-HC',  'path': os.path.join(data_dir, "SC", "matMetric_EP_dsi_number_of_fibers.npy")},
    {'label': 'SC-IND', 'path': os.path.join(data_dir, "SC", "matMetric_SCHZ_CTRL.npy")},
]

EucDist_fig5 = np.load(os.path.join(data_dir, "EucMat", "EucMat_HC_dsi_number_of_fibers.npy"))
X_RS_allPat_fig5 = gsp.load_EEG_example(example_dir)

nbPerm = 20
selected_bins_base = [2, 4] + list(range(5, 13))

all_sc_results = {}

# ============================================================================
# MAIN LOOP
# ============================================================================
for sc_cfg in sc_configs:

    sc_label = sc_cfg['label']
    print(f"\nProcessing {sc_label}")



# Figure S4: Same scatter plots but for LT and RT lateralizations
# ============================================================================
print("\nGenerating scatter plots of SDI correlation for LT and RT (Fig S4)...")

# Create figure with 2 rows x 3 columns (LT/RT scatter panels only)
figS4, axsS4 = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

method_names_rt = ['SC_IND', 'Gen_Procrustes', 'Hungarian']
method_labels_rt = ['Before alignment', 'Procrustes', 'Hungarian']
method_names_display = ['Before alignment', 'Procrustes', 'Hungarian']

rows = [('LT', 'A'), ('RT', 'B')]
n_boot = 500
rng = np.random.default_rng(42)

for row_idx, (lateralization, row_letter) in enumerate(rows):
    sdi_sc_hc_ref = sdi_results[f'SC_HC_ref_{lateralization}']['surr_thresh'][thr]['mean_SDI']
    all_sdi_values = [sdi_sc_hc_ref]
    for method_name in method_names_rt:
        all_sdi_values.append(sdi_results[f'{method_name}_{lateralization}']['surr_thresh'][thr]['mean_SDI'])
    global_min = np.min([np.min(v) for v in all_sdi_values])
    global_max = np.max([np.max(v) for v in all_sdi_values])

    corr_boot = {method_name: [] for method_name in method_names_rt}
    for _ in range(n_boot):
        boot_idx = rng.integers(0, len(sdi_sc_hc_ref), size=len(sdi_sc_hc_ref))
        ref_boot = sdi_sc_hc_ref[boot_idx]
        for method_name in method_names_rt:
            method_boot = sdi_results[f'{method_name}_{lateralization}']['surr_thresh'][thr]['mean_SDI'][boot_idx]
            r_boot, _ = pearsonr(ref_boot, method_boot)
            corr_boot[method_name].append(r_boot)

    friedman_stat, friedman_p = friedmanchisquare(*[corr_boot[m] for m in method_names_rt])

    # Scatter plots
    for i, method_name in enumerate(method_names_rt):
        ax = axsS4[row_idx, i]
        sdi_method = sdi_results[f'{method_name}_{lateralization}']['surr_thresh'][thr]['mean_SDI']
        r_corr, p_val = pearsonr(sdi_sc_hc_ref, sdi_method)
        lateralization_color = '#1f77b4' if lateralization == 'LT' else '#2ca02c'

        ax.scatter(sdi_sc_hc_ref, sdi_method, alpha=0.7, s=40, color=lateralization_color, edgecolors='none')
        slope, intercept = np.polyfit(sdi_sc_hc_ref, sdi_method, 1)
        x_line = np.array([global_min, global_max])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, linestyle='--', linewidth=1.8, color=lateralization_color, alpha=0.9)
        ax.set_xlabel(r'SDI SC$_{HC}$', fontsize=12, fontweight='bold', family='sans-serif')
        ax.set_ylabel(r'SDI SC$_{IND}$', fontsize=12, fontweight='bold', family='sans-serif')
        title = ax.set_title(f'{method_names_display[i]} — r={r_corr:.3f}, p={p_val:.2e}', fontsize=11, loc='center', pad=5, family='sans-serif')
        title.set_fontweight('bold')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['top'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([global_min, global_max])
        ax.set_ylim([global_min, global_max])
        ax.text(0.05, 0.95, f'{method_names_display[i]}', 
                transform=ax.transAxes, fontsize=10, fontweight='bold', family='sans-serif',
                verticalalignment='top', horizontalalignment='left')

figS4.suptitle('SDI correlation for LT and RT', fontsize=14, fontweight='bold')
plt.tight_layout()



# Figure S4bis and ter: Mixed LT/RT triangle heatmaps of inter-method correlations
# ============================================================================
print("\nGenerating mixed LT/RT inter-method correlation heatmaps (Fig S4ter)...")

groups_heatmap_s4ter = [
    ('SC_HC_ref', r'SC$_{HC}$ reference'),
    ('SC_IND', r'Before alignment'),
    ('Gen_Procrustes', r'Procrustes'),
    ('Hungarian', r'Hungarian'),
]

matrices_by_side_s4ter = {}
for lateralization in ['LT', 'RT']:
    sdi_vectors = [
        sdi_results[f'{group_name}_{lateralization}']['surr_thresh'][thr]['mean_SDI']
        for group_name, _ in groups_heatmap_s4ter
    ]
    n_groups = len(sdi_vectors)
    r_mat = np.eye(n_groups)
    p_mat = np.zeros((n_groups, n_groups))

    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                continue
            r_tmp, p_tmp = pearsonr(sdi_vectors[i], sdi_vectors[j])
            r_mat[i, j] = r_tmp
            p_mat[i, j] = p_tmp

    r_df = pd.DataFrame(r_mat, index=[label for _, label in groups_heatmap_s4ter], columns=[label for _, label in groups_heatmap_s4ter])
    p_df = pd.DataFrame(p_mat, index=[label for _, label in groups_heatmap_s4ter], columns=[label for _, label in groups_heatmap_s4ter])
    matrices_by_side_s4ter[lateralization] = {'r_df': r_df, 'p_df': p_df, 'r_mat': r_mat, 'p_mat': p_mat}

purple_cmap_s4ter = sns.light_palette("purple", as_cmap=True)

mask_lower_s4ter = np.triu(np.ones_like(matrices_by_side_s4ter['LT']['r_df'], dtype=bool), k=0)
mask_upper_s4ter = np.tril(np.ones_like(matrices_by_side_s4ter['LT']['r_df'], dtype=bool), k=0)

# ---- Figure S4bis: correlations ----
figS4bis, axS4bis = plt.subplots(1, 1, figsize=(6.5, 5.8), constrained_layout=True)

sns.heatmap(
    matrices_by_side_s4ter['LT']['r_df'],
    mask=mask_lower_s4ter,
    ax=axS4bis,
    cmap='Blues',
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar=False,
    annot=False,
)
sns.heatmap(
    matrices_by_side_s4ter['RT']['r_df'],
    mask=mask_upper_s4ter,
    ax=axS4bis,
    cmap='Greens',
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar=False,
    annot=False,
)

for d in range(n_groups):
    axS4bis.add_patch(
        plt.Rectangle((d, d), 1, 1, facecolor='lightgray', edgecolor='white', linewidth=0.5, zorder=3)
    )
    axS4bis.text(d + 0.5, d + 0.5, '—', ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

for i in range(n_groups):
    for j in range(n_groups):
        if i > j:
            p_lt = matrices_by_side_s4ter['LT']['p_mat'][i, j]
            stars_lt = '***' if p_lt < 0.001 else '**' if p_lt < 0.01 else '*' if p_lt < 0.05 else ''
            axS4bis.text(j + 0.5, i + 0.5, f"{matrices_by_side_s4ter['LT']['r_mat'][i, j]:.2f}{stars_lt}",
                         ha='center', va='center', fontsize=14,
                         fontweight='bold' if p_lt < 0.05 else 'normal')
        elif i < j:
            p_rt = matrices_by_side_s4ter['RT']['p_mat'][i, j]
            stars_rt = '***' if p_rt < 0.001 else '**' if p_rt < 0.01 else '*' if p_rt < 0.05 else ''
            axS4bis.text(j + 0.5, i + 0.5, f"{matrices_by_side_s4ter['RT']['r_mat'][i, j]:.2f}{stars_rt}",
                         ha='center', va='center', fontsize=14,
                         fontweight='bold' if p_rt < 0.05 else 'normal')

axS4bis.tick_params(axis='x', rotation=20, labelsize=10)
axS4bis.tick_params(axis='y', rotation=0, labelsize=10)

figS4bis_path_png = os.path.join(figures_dir, 'FigS4bis_inter_method_correlation_LT_RT.png')
plt.savefig(figS4bis_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure S4bis saved as '{figS4bis_path_png}'")
plt.close()

# ---- Figure S4ter: p-values ----
figS4ter, axS4ter = plt.subplots(1, 1, figsize=(6.5, 5.8), constrained_layout=True)
figS4ter.suptitle("SDI correlations p-values: SC HC and SC IND realigned", fontsize=14, fontweight='bold')

sns.heatmap(
    matrices_by_side_s4ter['LT']['p_df'],
    mask=mask_lower_s4ter,
    ax=axS4ter,
    cmap=purple_cmap_s4ter,
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar=False,
    annot=False,
)
sns.heatmap(
    matrices_by_side_s4ter['RT']['p_df'],
    mask=mask_upper_s4ter,
    ax=axS4ter,
    cmap=purple_cmap_s4ter,
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar=False,
    annot=False,
)

for d in range(n_groups):
    axS4ter.add_patch(
        plt.Rectangle((d, d), 1, 1, facecolor='lightgray', edgecolor='white', linewidth=0.5, zorder=3)
    )
    axS4ter.text(d + 0.5, d + 0.5, '—', ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

for i in range(n_groups):
    for j in range(n_groups):
        if i > j:
            p_lt = matrices_by_side_s4ter['LT']['p_mat'][i, j]
            stars_lt = '***' if p_lt < 0.001 else '**' if p_lt < 0.01 else '*' if p_lt < 0.05 else ''
            axS4ter.text(j + 0.5, i + 0.5, f"{p_lt:.1e}{stars_lt}",
                         ha='center', va='center', fontsize=14,
                         fontweight='bold' if p_lt < 0.05 else 'normal', color='black')
        elif i < j:
            p_rt = matrices_by_side_s4ter['RT']['p_mat'][i, j]
            stars_rt = '***' if p_rt < 0.001 else '**' if p_rt < 0.01 else '*' if p_rt < 0.05 else ''
            axS4ter.text(j + 0.5, i + 0.5, f"{p_rt:.1e}{stars_rt}",
                         ha='center', va='center', fontsize=14,
                         fontweight='bold' if p_rt < 0.05 else 'normal', color='black')

axS4ter.set_title('p-value: lower Left IED, upper Right IED', fontsize=12, fontweight='bold')
axS4ter.tick_params(axis='x', rotation=20, labelsize=10)
axS4ter.tick_params(axis='y', rotation=0, labelsize=10)

figS4ter_path_png = os.path.join(figures_dir, 'FigS4ter_inter_method_pvalue_LT_RT.png')
plt.savefig(figS4ter_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure S4ter saved as '{figS4ter_path_png}'")
plt.close()
