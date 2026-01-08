
''' This script runs the analysis comparing how changing the number of matrices for the consensus changes the variability of the harmonics'''

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import seaborn as sns
import numpy as np 
import scipy.io as sio
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from lib.func_plot import plot_rois, plot_rois_pyvista
import random
from lib import fcn_groups_bin
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import lib.func_GSP as gsp


example_dir = "DATA/EEG"
### Reading the data
SC = sio.loadmat('./DATA/Individual_Connectomes.mat')   
SC = SC['connMatrices']['SC'][0][0][1][0]
roi_info_path = 'data/label/roi_info.xlsx'
roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
matMetric = SC
x = np.asarray(roi_info['x-pos'])[cort_rois] 
y = np.asarray(roi_info['y-pos'])[cort_rois]
z = np.asarray(roi_info['z-pos'])[cort_rois]
coordMat = np.concatenate((x[:,None],y[:,None],z[:,None]),1)
Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))  
#matMetric = np.load("DATA/SC/matMetric_SCHZ_CTRL.npy")
#matMetric = np.transpose(matMetric, (1, 2, 0))
#Euc = np.load("DATA/EucMat/EucMat_HC_DSI_number_of_fibers.npy")
#Euc = np.mean(Euc, axis=2)  # Average the Euclidean distance matrix across participants
#cort_rois = np.arange(len(Euc))

### Generate random group based on different number of participants
total_participant = np.shape(matMetric)[2]; nROIs = np.shape(matMetric)[0]
idxs = list(range(total_participant))
ls_bins = [1,5,10, 20, 25]
nbPerm = 100
nbins = 41
hemii = np.ones(len(Euc))
hemii[int(len(hemii)/2):] = 2
RandCons = np.zeros((nROIs, nROIs, nbPerm, len(ls_bins)))
ShuffIdxs = np.zeros((len(idxs), nbPerm, len(ls_bins))  )

for b,bi in enumerate(ls_bins):
    for p in np.arange(nbPerm):
        random.shuffle(idxs)
        ShuffIdxs[:,p,b] = idxs
        idxs_tmp = idxs[0:bi]
        [G, Gc] = fcn_groups_bin.fcn_groups_bin(matMetric[:,:, idxs_tmp], Euc, hemii, nbins) 
        avg = np.mean(matMetric[:,:, idxs_tmp], 2) 
        RandCons[:,:,p,b] = Gc*avg
print('nROIs=%d, number of bins=%d, number of randomization=%d'%(np.shape(RandCons)[0], np.shape(RandCons)[3], np.shape(RandCons)[2]))


### Generate the eigenvectors 
nb_eig2keep = nROIs
eigenvectors_perm = np.zeros((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm))
eigenvalues_perm = np.zeros((nb_eig2keep, len(ls_bins)*nbPerm))
eigenvectors_perm_mat = np.zeros((len(cort_rois), nb_eig2keep, len(ls_bins), nbPerm))
eigenvalues_perm_mat = np.zeros((nb_eig2keep, len(ls_bins), nbPerm))
labels_perm = []

k = 0
for b,bi in enumerate(ls_bins):
    for p in np.arange(nbPerm):
        try:
            eigenvalues_perm_mat[:, b, p], eigenvectors_perm_mat[:, :, b, p],Ln_ind, An_ind = gsp.cons_normalized_lap(RandCons[:,:,p,b], Euc, plot=False)
            labels_perm.append('Bin%d'%(bi))
        except np.linalg.LinAlgError as e:
            print(f"Warning: SVD convergence failed for bin {bi}, permutation {p}. Skipping...")
            # Fill with NaN values to indicate failure
            eigenvalues_perm_mat[:, b, p] = np.nan
            eigenvectors_perm_mat[:, :, b, p] = np.nan
            labels_perm.append('Bin%d'%(bi))
        k = k+1 
     
     
     
#P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
#Qind_rotated, Qind_HC_RT_centered, disparity_RT = scipy.spatial.procrustes(Qind_ref, Q_ind)
#R_RT, _ = scipy.linalg.orthogonal_procrustes(Qind_ref, Q_ind)
#Qind_ortho_rotated=Q_ind@R_RT 
#perm, total_cost = gsp.match_eigenvectors(Qind_ref, Q_ind)
#Qind_matched = Q_ind[:,perm]   
        
### Rotate for each bin only
eigenvalues_perm_mat_rot = np.zeros(np.shape(eigenvalues_perm_mat)); eigenvectors_perm_mat_rot = np.zeros(np.shape(eigenvectors_perm_mat))
eigenvalues_perm_mat_ortho = np.zeros(np.shape(eigenvalues_perm_mat)); eigenvectors_perm_mat_ortho = np.zeros(np.shape(eigenvectors_perm_mat))
eigenvalues_perm_mat_matched = np.zeros(np.shape(eigenvalues_perm_mat)); eigenvectors_perm_mat_matched = np.zeros(np.shape(eigenvectors_perm_mat))
R_all = np.zeros(np.shape(eigenvectors_perm_mat)); 
scale_R = np.zeros((len(ls_bins), nbPerm)); disparity = np.copy(scale_R)
for b,bi in enumerate(ls_bins):
    print(bi)
    ### Generalized Procrustes (with retry)
    max_retries = 3
    for retry in range(max_retries):
        try:
            eigenvectors_perm_mat_rot[:,:,b,:], eigenvalues_perm_mat_rot[:,b,:], A, B = gsp.rotation_procrustes(eigenvectors_perm_mat[:,:,b,:], eigenvalues_perm_mat[:,b,:], plot=False, p='bin%d'%bi)
            break
        except np.linalg.LinAlgError:
            if retry < max_retries - 1:
                print(f"  Generalized Procrustes SVD failed for bin {bi}, retry {retry+1}/{max_retries}...")
                eigenvectors_perm_mat[:,:,b,:] += np.random.randn(*eigenvectors_perm_mat[:,:,b,:].shape) * 1e-10
            else:
                print(f"  Generalized Procrustes failed for bin {bi} after {max_retries} retries, filling with NaN")
                eigenvectors_perm_mat_rot[:,:,b,:] = np.nan
                eigenvalues_perm_mat_rot[:,b,:] = np.nan
    
    ### Orthogonal Procrustes (robust to SVD failures with retry)
    for retry in range(max_retries):
        try:
            eigenvectors_perm_mat_ortho[:,:,b,:], eigenvalues_perm_mat_ortho[:,b,:],  R_all[:,:,b,:], scale_R[b,:] = gsp.orthogonal_rotation_procrustes(eigenvectors_perm_mat[:,:,b,:], eigenvalues_perm_mat[:,b,:], plot=False, p='bin%d'%bi)
            break  # Success, exit retry loop
        except np.linalg.LinAlgError:
            if retry < max_retries - 1:
                print(f"  Orthogonal Procrustes SVD failed for bin {bi}, retry {retry+1}/{max_retries}...")
                # Add small random noise to break degeneracy
                eigenvectors_perm_mat[:,:,b,:] += np.random.randn(*eigenvectors_perm_mat[:,:,b,:].shape) * 1e-10
            else:
                print(f"  Orthogonal Procrustes failed for bin {bi} after {max_retries} retries, filling with NaN")
                eigenvectors_perm_mat_ortho[:,:,b,:] = np.nan
                eigenvalues_perm_mat_ortho[:,b,:] = np.nan
                R_all[:,:,b,:] = np.nan
                scale_R[b,:] = np.nan
    # Compare the first permutation to another available one (avoid hard-coded index 5 when nbPerm < 6)
    if nbPerm > 1 and not np.isnan(eigenvectors_perm_mat_ortho[:,:,b,:]).all():
        compare_idx = min(nbPerm - 1, 5)
        cos_sim_ortho = np.diag(cosine_similarity(eigenvectors_perm_mat_ortho[:,:,b,0], eigenvectors_perm_mat_ortho[:,:,b,compare_idx]))
        cos_sim = np.diag(cosine_similarity(eigenvectors_perm_mat[:,:,b,0], eigenvectors_perm_mat[:,:,b,compare_idx]))
    else:
        cos_sim_ortho = np.array([])
        cos_sim = np.array([])
    ### Hungarian algorithm for matching the eigenvectors
    for q in np.arange(nbPerm):
        perm, total_cost = gsp.match_eigenvectors(eigenvectors_perm_mat[:,:,b,0], eigenvectors_perm_mat[:,:,b,q])
        eigenvectors_perm_mat_matched[:,:,b,q] =  eigenvectors_perm_mat[:,perm,b,q]
    #eigenvectors_perm_mat_rot[:,:,b,:], eigenvalues_perm_mat_ortho[:,b,:], A, B = gsp.orthogonal_rotation_procrustes(eigenvectors_perm_mat[:,:,b,:], eigenvalues_perm_mat[:,b,:], plot=False, p='bin%d'%bi)


eigenvectors_perm_ortho = np.reshape(eigenvectors_perm_mat_ortho, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm_rot = np.reshape(eigenvectors_perm_mat_rot, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm_matched = np.reshape(eigenvectors_perm_mat_matched, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm = np.reshape(eigenvectors_perm_mat, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))

X_RS_allPat = gsp.load_EEG_example(example_dir)

P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(np.mean(matMetric, axis=2), Euc, plot=False)

### Generate the corresponding labels
### Generate the corresponding labels
labels_perm_mat = []
labels_perm_bin = []
for i in np.arange(len(labels_perm)):
    for j in np.arange(len(labels_perm)):  
        labels_perm_mat.append('%s_%s'%(labels_perm[i], labels_perm[j]))
        if labels_perm[i]==labels_perm[j]:
            labels_perm_bin.append('%s'%(labels_perm[i]))
        else:
            labels_perm_bin.append('Different bins')
labels_perm_bin = np.array(labels_perm_bin)
labels_perm_mat = np.array(labels_perm_mat)

### Compute the similarity betwen all the eigenvectors (all bins and randomization)
Dist_eigvec_perm = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_ortho = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_rot = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_matched = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep))
for eigvec_nb in np.arange(nb_eig2keep):
    MatDist = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb,:]), metric='correlation')
    #MatDist = scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb,:]), metric='euclidean')
    Dist_eigvec_perm[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist)
    #MatDist_rot = scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_rot[:, eigvec_nb,:]), metric='euclidean')
    MatDist_ortho = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_ortho[:, eigvec_nb,:]), metric='correlation')
    Dist_eigvec_perm_ortho[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist_ortho)
    MatDist_rot = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_rot[:, eigvec_nb,:]), metric='correlation')
    Dist_eigvec_perm_rot[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist_rot)
    MatDist_matched = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_matched[:, eigvec_nb,:]), metric='correlation')
    Dist_eigvec_perm_matched[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist_matched)


#### Remove the 0 values corresponding to the similarity between identical vectors
#### Remove the 0 values corresponding to the similarity between identical vectors
Dist_eigvec_perm_vec = np.reshape(Dist_eigvec_perm, (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_vec = np.abs(Dist_eigvec_perm_vec) ### Take absolute values for compensating for sign change 
Dist_eigvec_perm_ortho_vec = np.reshape(Dist_eigvec_perm_ortho, (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_ortho_vec = np.abs(Dist_eigvec_perm_ortho_vec) ### Take absolute values for compensating for sign change 
Dist_eigvec_perm_rot_vec = np.reshape(Dist_eigvec_perm_rot, (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_rot_vec = np.abs(Dist_eigvec_perm_rot_vec) ### Take absolute values for compensating for sign change
Dist_eigvec_perm_matched_vec = np.reshape(Dist_eigvec_perm_matched, (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_matched_vec = np.abs(Dist_eigvec_perm_matched_vec) ### Take absolute values for compensating for sign change
tmp2 = Dist_eigvec_perm_ortho_vec[:,30]; tmp2 = tmp2[np.where(tmp2>0)]


### Remove the 0 values corresponding here to the diagonal
for i in np.arange(nb_eig2keep):
    idxs_nz = np.where(Dist_eigvec_perm_vec[:,i])
    tmp = Dist_eigvec_perm_vec[:,i]; #tmp = tmp[idxs_nz]
    tmp2 = Dist_eigvec_perm_ortho_vec[:,i]; #tmp2 = tmp2[idxs_nz]
    tmp3 = Dist_eigvec_perm_rot_vec[:,i]; #tmp3 = tmp3[idxs_nz]
    tmp4 = Dist_eigvec_perm_matched_vec[:,i]; #tmp4 = tmp4[idxs_nz]
    if i==0:
           Dist_eigvec_perm_vec_nz = np.zeros((len(tmp), nb_eig2keep))
           Dist_eigvec_perm_ortho_vec_nz = np.zeros((len(tmp2), nb_eig2keep))
           Dist_eigvec_perm_rot_vec_nz = np.zeros((len(tmp3), nb_eig2keep))
           Dist_eigvec_perm_matched_vec_nz = np.zeros((len(tmp4), nb_eig2keep))
    Dist_eigvec_perm_vec_nz[:,i] = tmp 
    Dist_eigvec_perm_ortho_vec_nz[:,i] = tmp2
    Dist_eigvec_perm_rot_vec_nz[:,i] = tmp3
    Dist_eigvec_perm_matched_vec_nz[:,i] = tmp4
#labels_perm_bin = labels_perm_bin[idxs_nz] 
#labels_perm_mat = labels_perm_mat[idxs_nz] 

bin_variability = np.zeros((len(ls_bins), nb_eig2keep, 2))
bin_variability_ortho = np.zeros((len(ls_bins), nb_eig2keep, 2))
bin_variability_matched = np.zeros((len(ls_bins), nb_eig2keep, 2))
bin_variability_rot = np.zeros((len(ls_bins), nb_eig2keep, 2))
for b,bi in enumerate(ls_bins):
    idxs = np.where(labels_perm_mat=='Bin%d_Bin%d'%(bi,bi))[0]
    for i in np.arange(nb_eig2keep):
        #print(np.median(Dist_eigvec_perm_vec_nz[idxs,i]))
        bin_variability[b,i,0] = np.median(Dist_eigvec_perm_vec_nz[idxs,i])
        bin_variability[b,i,1] = np.std(Dist_eigvec_perm_vec_nz[idxs,i])
        bin_variability_ortho[b,i,0] = np.median(Dist_eigvec_perm_ortho_vec_nz[idxs,i])
        bin_variability_ortho[b,i,1] = np.std(Dist_eigvec_perm_ortho_vec_nz[idxs,i])        
        bin_variability_matched[b,i,0] = np.median(Dist_eigvec_perm_matched_vec_nz[idxs,i])
        bin_variability_matched[b,i,1] = np.std(Dist_eigvec_perm_matched_vec_nz[idxs,i])
        bin_variability_rot[b,i,0] = np.median(Dist_eigvec_perm_rot_vec_nz[idxs,i])
        bin_variability_rot[b,i,1] = np.std(Dist_eigvec_perm_rot_vec_nz[idxs,i])

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()  # Make indexing easier
# Define publication-quality color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
handles = []

for b,bi in enumerate(ls_bins):
    line, = ax[0].plot(bin_variability[b,:,0], linewidth=2, color=colors[b], label=f'n={bi}'); 
    if b < len(ls_bins):
        handles.append(line)
    ax[1].plot(bin_variability_ortho[b,:,0], linewidth=2, color=colors[b], label=f'n={bi}');
    ax[2].plot(bin_variability_rot[b,:,0], linewidth=2, color=colors[b], label=f'n={bi}');
    ax[3].plot(bin_variability_matched[b,:,0], linewidth=2, color=colors[b], label=f'n={bi}');
    
    # Calculate mean and std values for horizontal lines
    mean_raw = np.mean(bin_variability[b,:,0])
    std_raw = np.std(bin_variability[b,:,0])
    mean_ortho = np.mean(bin_variability_ortho[b,:,0])
    std_ortho = np.std(bin_variability_ortho[b,:,0])
    mean_rot = np.mean(bin_variability_rot[b,:,0])
    std_rot = np.std(bin_variability_rot[b,:,0])
    mean_matched = np.mean(bin_variability_matched[b,:,0])
    std_matched = np.std(bin_variability_matched[b,:,0])
    
    # Add horizontal dotted lines at mean values with text annotations on the right
    ax[0].axhline(y=mean_raw, color=colors[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[0].text(nb_eig2keep+1, mean_raw, f'{mean_raw:.2f}±{std_raw:.2f}', color=colors[b], fontsize=8, va='center', ha='left', fontweight='bold')
    
    ax[1].axhline(y=mean_ortho, color=colors[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[1].text(nb_eig2keep+1, mean_ortho, f'{mean_ortho:.2f}±{std_ortho:.2f}', color=colors[b], fontsize=8, va='center', ha='left', fontweight='bold')
    
    ax[2].axhline(y=mean_rot, color=colors[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[2].text(nb_eig2keep+1, mean_rot, f'{mean_rot:.2f}±{std_rot:.2f}', color=colors[b], fontsize=8, va='center', ha='left', fontweight='bold')
    
    ax[3].axhline(y=mean_matched, color=colors[b], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[3].text(nb_eig2keep+1, mean_matched, f'{mean_matched:.2f}±{std_matched:.2f}', color=colors[b], fontsize=8, va='center', ha='left', fontweight='bold')

    upper_bound = bin_variability[b, :, 0] + bin_variability[b, :, 1]
    lower_bound = bin_variability[b, :, 0] - bin_variability[b, :, 1]
    upper_bound_ortho = bin_variability_ortho[b, :, 0] + bin_variability_ortho[b, :, 1]
    lower_bound_ortho = bin_variability_ortho[b, :, 0] - bin_variability_ortho[b, :, 1]
    upper_bound_matched = bin_variability_matched[b, :, 0] + bin_variability_matched[b, :, 1]
    lower_bound_matched = bin_variability_matched[b, :, 0] - bin_variability_matched[b, :, 1]
    upper_bound_rot = bin_variability_rot[b, :, 0] + bin_variability_rot[b, :, 1]
    lower_bound_rot = bin_variability_rot[b, :, 0] - bin_variability_rot[b, :, 1]

    ax[0].fill_between(range(nb_eig2keep), lower_bound, upper_bound, alpha=0.15, color=colors[b])
    ax[1].fill_between(range(nb_eig2keep), lower_bound_ortho, upper_bound_ortho, alpha=0.15, color=colors[b])
    ax[2].fill_between(range(nb_eig2keep), lower_bound_rot, upper_bound_rot, alpha=0.15, color=colors[b])
    ax[3].fill_between(range(nb_eig2keep), lower_bound_matched, upper_bound_matched, alpha=0.15, color=colors[b])

for x in range(4):
    ax[x].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
    ax[x].set_xticks(range(0, nb_eig2keep, 20))
    ax[x].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax[x].set_ylim([0,1.02])
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

plt.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.12)
plt.savefig('bin_variability_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('bin_variability_analysis.pdf', bbox_inches='tight', facecolor='white')
print("Plots saved as 'bin_variability_analysis.png' and '.pdf'")

# Summary figure to compare stability and alignment across methods
method_labels = ['Raw', 'Orthogonal Procrustes', 'Hungarian matching', 'Generalized Procrustes']
method_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

# Mean and std across bins for each eigenmode
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

# Mean across eigenmodes for each bin size
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

# Improvements relative to raw (per eigenmode)
delta_modes = [
    mean_modes[1] - mean_modes[0],
    mean_modes[2] - mean_modes[0],
    mean_modes[3] - mean_modes[0],
]


fig2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
ax2 = ax2.flatten()

# Panel 1: Mean stability by consensus size (grouped bars)
x = np.arange(len(ls_bins))
width = 0.18
bar_handles = []
for idx, label in enumerate(method_labels):
    h = ax2[0].bar(x + (idx - 1.5) * width, bin_means[idx], width=width, color=method_colors[idx],
                  yerr=bin_stds[idx], capsize=3, label=label, alpha=0.9)
    bar_handles.append(h)
ax2[0].set_xticks(x)
ax2[0].set_xticklabels([f'n={b}' for b in ls_bins])
ax2[0].set_ylabel('Mean correlation across modes')
ax2[0].set_title('Effect of consensus size', fontsize=12, fontweight='bold')
ax2[0].set_ylim([0, 1.5])
ax2[0].grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

# Panel 2: Mean stability by method across bins
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

# Panel 3: Improvement over raw (per eigenmode)
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

# Panel 4 removed; keep layout 1x3

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
fig2.savefig('stability_alignment_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
fig2.savefig('stability_alignment_summary.pdf', bbox_inches='tight', facecolor='white')
print("Summary plots saved as 'stability_alignment_summary.png' and '.pdf'")
plt.show()