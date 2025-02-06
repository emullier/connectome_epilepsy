
import numpy as np
import pandas as pd
from nilearn import datasets
import nibabel as nb
import scipy
import random
import scipy.io as sio
import lib.func_reading as reading 
from lib import fcn_groups_bin
from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.gradient import GradientMaps
from brainspace.plotting import plot_hemispheres
from brainspace.gradient import embedding
import pyvista as pv
from lib.func_plot import plot_rois, plot_rois_pyvista
import matplotlib.pyplot as plt
#from brainspace.gradient.alignment import ProcrustesAlignment
from brainspace.gradient.alignment import procrustes_alignment
import networkx as nx
from lib import func_SDI as sdi
from lib import func_ML as ML
import seaborn as sns
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn



config_path = 'config.json' #'config_PEP3.json'
config_defaults = reading.check_config_file(config_path)

### Reading the data
SC = sio.loadmat('./data/Individual_Connectomes.mat')   
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

### Generate random group based on different number of participants
total_participant = np.shape(matMetric)[2]; nROIs = np.shape(matMetric)[0]
idxs = list(range(total_participant))
ls_bins = [1, 2,10,20,30, 40, 50] 
#ls_bins = [10, 50]
nbPerm = 100
nbins = 41
hemii = np.ones(len(Euc))
hemii[int(len(hemii)/2):] = 2
RandCons = np.zeros((nROIs, nROIs, nbPerm, len(ls_bins)))
ShuffIdxs = np.zeros((len(idxs), nbPerm, len(ls_bins)))

for b,bi in enumerate(ls_bins):
    for p in np.arange(nbPerm):
        random.shuffle(idxs)
        ShuffIdxs[:,p,b] = idxs
        idxs_tmp = idxs[0:bi]
        [G, Gc] = fcn_groups_bin.fcn_groups_bin(matMetric[:,:, idxs_tmp], Euc, hemii, nbins) 
        avg = np.mean(matMetric[:,:, idxs_tmp], 2) 
        tmp = Gc*avg
        RandCons[:,:,p,b] = tmp
        #if p%2==0:
        #    RandCons[:,:,p,b] = tmp
        #    #RandCons[:,:,p,b] = np.random.rand(*tmp.shape)
        #else:
        #    RandCons[:,:,p,b] = np.random.rand(*tmp.shape)
        #flattened = tmp.flatten()
        #np.random.shuffle(flattened)
        #shuffled_matrix = flattened.reshape(tmp.shape)
        #RandCons[:,:,p,b] = shuffled_matrix
    print('nROIs=%d, number of bins=%d, number of randomization=%d'%(np.shape(RandCons)[0], np.shape(RandCons)[3], np.shape(RandCons)[2]))


################################################

### Generate the eigenvectors 
nb_eig2keep = nROIs-1
n_components = 114
eigenvectors_perm = np.zeros((len(cort_rois), n_components-1, len(ls_bins)*nbPerm))
eigenvectors_perm_mat = np.zeros((len(cort_rois), n_components-1, len(ls_bins), nbPerm))
eigenvectors_perm_mat_rot = np.zeros(np.shape(eigenvectors_perm_mat))
labels_perm = []



k = 0
gref = GradientMaps(kernel=None, approach='le', n_components=n_components)
galign_proc = GradientMaps(kernel=None, approach='le', alignment='procrustes',  n_components=n_components)
reconstruct_sc = np.zeros(np.shape(RandCons))
X_RS_allPat = sdi.load_EEG_example()
ls_cutoff = np.zeros((len(X_RS_allPat), nbPerm, len(ls_bins)))
#galign_joint = GradientMaps(kernel='normalized_angle', approach='le', alignment='joint', n_components=n_components)
for b,bi in enumerate(ls_bins):
    print('Bin %d' % bi)
    mask = np.zeros(RandCons.shape, dtype=bool)
    mask[:,:,:,b] = True
    for p in np.arange(nbPerm):
        if p%10==0:
            print('Perm %d' %p)
        binRandCons = RandCons*mask
        mean_excluding_slice = binRandCons.mean(axis=(2,3))
        gref.fit(mean_excluding_slice)
        galign_proc.fit(RandCons[:,:,p,b], reference=gref.gradients_)
        #galign_joint.fit(RandCons[:,:,p,b], reference=gref.gradients_)
        lap = embedding.laplacian_eigenmaps(RandCons[:,:,p,b], n_components=n_components, norm_laplacian=True, random_state=None)
        eigenvectors_perm_mat[:,:,b,p] = galign_proc.gradients_ #lap[0]
        eigenvectors_perm_mat_rot[:, :, b, p] = galign_proc.aligned_
        reconstruct_sc[:,:,p,b] = galign_proc.aligned_ @ np.diag(galign_proc.lambdas_) @ np.transpose(galign_proc.aligned_)
        binMat = np.copy(RandCons[:,:,p,b])
        binMat[np.where(RandCons[:,:,p,b]>0)]=1
        reconstruct_sc[:,:,p,b] = np.abs(reconstruct_sc[:,:,p,b])*binMat
        np.fill_diagonal(reconstruct_sc[:,:,p,b], 0)
        labels_perm.append('Bin%d'%(bi))
        P_ind, Q_ind, Ln_ind, An_ind = ML.cons_normalized_lap(RandCons[:,:,p,b], Euc, plot=False)
        for t in np.arange(len(X_RS_allPat)):
            X_RS = X_RS_allPat[t]['X_RS']
            idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59,116)))
            X_RS = X_RS[idxs_tmp, :, :]
            PSD,NN, Vlow, Vhigh = sdi.get_cutoff_freq(Q_ind, X_RS)
            ls_cutoff[t,p,b] = NN 

        k = k+1 
        
eigenvectors_perm_rot = np.reshape(eigenvectors_perm_mat_rot, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm = np.reshape(eigenvectors_perm_mat, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))

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
Corr_eigvec_perm = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep)); Euc_eigvec_perm = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep))
Corr_eigvec_perm_rot = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep)); Euc_eigvec_perm_rot = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep))
for eigvec_nb in np.arange(nb_eig2keep):
    MatDist = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb,:]), metric='correlation')
    Corr_eigvec_perm[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist)
    MatDist_rot = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_rot[:, eigvec_nb,:]), metric='correlation')
    Corr_eigvec_perm_rot[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist_rot)
    MatDist = scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb,:]), metric='euclidean')
    Euc_eigvec_perm[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist)
    MatDist_rot = scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_rot[:, eigvec_nb,:]), metric='euclidean')
    Euc_eigvec_perm_rot[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist_rot)

#### Remove the 0 values corresponding to the similarity between identical vectors
Corr_eigvec_perm_vec = np.reshape(np.abs(Corr_eigvec_perm), (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Corr_eigvec_perm_rot_vec = np.reshape(np.abs(Corr_eigvec_perm_rot), (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Euc_eigvec_perm_vec = np.reshape(np.abs(Euc_eigvec_perm), (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Euc_eigvec_perm_rot_vec = np.reshape(np.abs(Euc_eigvec_perm_rot), (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
#tmp2 = Dist_eigvec_perm_rot_vec[:,30]; tmp2 = tmp2[np.where(tmp2>0)]


### Remove the 0 values corresponding here to the diagonal
for i in np.arange(nb_eig2keep):
    idxs_nz = np.where(Corr_eigvec_perm_vec[:,i])
    tmp = Corr_eigvec_perm_vec[:,i]; tmp = tmp[idxs_nz]
    tmp2 = Corr_eigvec_perm_rot_vec[:,i]; tmp2 = tmp2[idxs_nz]
    tmp3 = Euc_eigvec_perm_vec[:,i]; tmp3 = tmp3[idxs_nz]
    tmp4 = Euc_eigvec_perm_rot_vec[:,i]; tmp4 = tmp4[idxs_nz]
    if i==0:
           Corr_eigvec_perm_vec_nz = np.zeros((len(tmp), nb_eig2keep))
           Corr_eigvec_perm_rot_vec_nz = np.zeros((len(tmp2), nb_eig2keep))
           Euc_eigvec_perm_vec_nz = np.zeros((len(tmp), nb_eig2keep))
           Euc_eigvec_perm_rot_vec_nz = np.zeros((len(tmp2), nb_eig2keep))
    Corr_eigvec_perm_vec_nz[:,i] = tmp 
    Corr_eigvec_perm_rot_vec_nz[:,i] = tmp2
    Euc_eigvec_perm_vec_nz[:,i] = tmp3 
    Euc_eigvec_perm_rot_vec_nz[:,i] = tmp4
labels_perm_bin = labels_perm_bin[idxs_nz] 
labels_perm_mat = labels_perm_mat[idxs_nz] 

corr_bin_variability = np.zeros((len(ls_bins), nb_eig2keep, 2))
corr_bin_variability_rot = np.zeros((len(ls_bins), nb_eig2keep, 2))
euc_bin_variability = np.zeros((len(ls_bins), nb_eig2keep, 2))
euc_bin_variability_rot = np.zeros((len(ls_bins), nb_eig2keep, 2))
for b,bi in enumerate(ls_bins):
    idxs = np.where(labels_perm_mat=='Bin%d_Bin%d'%(bi,bi))[0]
    for i in np.arange(nb_eig2keep):
        corr_bin_variability[b,i,0] = np.median(Corr_eigvec_perm_vec_nz[idxs,i])
        corr_bin_variability[b,i,1] = np.std(Corr_eigvec_perm_vec_nz[idxs,i])
        corr_bin_variability_rot[b,i,0] = np.median(Corr_eigvec_perm_rot_vec_nz[idxs,i])
        corr_bin_variability_rot[b,i,1] = np.std(Corr_eigvec_perm_rot_vec_nz[idxs,i])    
        euc_bin_variability[b,i,0] = np.median(Euc_eigvec_perm_vec_nz[idxs,i])
        euc_bin_variability[b,i,1] = np.std(Euc_eigvec_perm_vec_nz[idxs,i])
        euc_bin_variability_rot[b,i,0] = np.median(Euc_eigvec_perm_rot_vec_nz[idxs,i])
        euc_bin_variability_rot[b,i,1] = np.std(Euc_eigvec_perm_rot_vec_nz[idxs,i])      
fig, ax = plt.subplots(2,2,figsize=(15,5))
handles = []; handles_rot=[]  # To store the handles for lines in the plot
for b,bi in enumerate(ls_bins):
    line, = ax[0,0].plot(corr_bin_variability[b,:,0]); handles.append(line)
    line_rot, = ax[1,0].plot(corr_bin_variability_rot[b,:,0]); handles_rot.append(line_rot)
    corr_upper_bound = corr_bin_variability[b, :, 0] + corr_bin_variability[b, :, 1]
    corr_lower_bound = corr_bin_variability[b, :, 0] - corr_bin_variability[b, :, 1]
    corr_upper_bound_rot = corr_bin_variability_rot[b, :, 0] + corr_bin_variability_rot[b, :, 1]
    corr_lower_bound_rot = corr_bin_variability_rot[b, :, 0] - corr_bin_variability_rot[b, :, 1]
    ax[0,0].fill_between(range(nb_eig2keep), corr_lower_bound, corr_upper_bound, alpha=0.1)
    ax[1,0].fill_between(range(nb_eig2keep), corr_lower_bound_rot, corr_upper_bound_rot, alpha=0.1)
for x in range(2):
    ax[x,0].set_xlabel('Eigenmode'); ax[x,0].set_xticks(range(0, nb_eig2keep, 10))
    ax[x,0].grid('on'); ax[x,0].set_ylim([0,1.02]); ax[x,0].set_ylabel('Correlation'); ax[x,0].set_title('Similarity between network harmonics', fontsize=15);
ax[0,0].legend(handles=handles, labels=ls_bins, loc='lower left', title='Bin');
ax[1,0].legend(handles=handles_rot, labels=ls_bins, loc='lower left', title='Bin');

handles = []; handles_rot=[]  # To store the handles for lines in the plot
for b,bi in enumerate(ls_bins):
    line, = ax[0,1].plot(euc_bin_variability[b,:,0]); handles.append(line)
    line_rot, = ax[1,1].plot(euc_bin_variability_rot[b,:,0]); handles_rot.append(line_rot)
    euc_upper_bound = euc_bin_variability[b, :, 0] + euc_bin_variability[b, :, 1]
    euc_lower_bound = euc_bin_variability[b, :, 0] - euc_bin_variability[b, :, 1]
    euc_upper_bound_rot = euc_bin_variability_rot[b, :, 0] + euc_bin_variability_rot[b, :, 1]
    euc_lower_bound_rot = euc_bin_variability_rot[b, :, 0] - euc_bin_variability_rot[b, :, 1]
    ax[0,1].fill_between(range(nb_eig2keep), euc_lower_bound, euc_upper_bound, alpha=0.1)
    ax[1,1].fill_between(range(nb_eig2keep), euc_lower_bound_rot, euc_upper_bound_rot, alpha=0.1)
for x in range(2):
    ax[x,1].set_xlabel('Eigenmode'); ax[x,1].set_xticks(range(0, nb_eig2keep, 10))
    ax[x,1].grid('on'); ax[x,1].set_ylabel('Euclidean Distance'); ax[x,1].set_title('Similarity between network harmonics', fontsize=15);
ax[0,1].legend(handles=handles, labels=ls_bins, loc='lower left', title='Bin');
ax[1,1].legend(handles=handles_rot, labels=ls_bins, loc='lower left', title='Bin');
#plt.show(block=True)

marker_shapes = ['o', 's', '^', 'd', 'P'] 
colormaps = ['viridis', 'plasma', 'cool', 'cividis', 'magma']

fig, axes = plt.subplots(3, 2, figsize=(12, 15))
p=0; k=0
# Plot each bin in its own subplot
for b, bi in enumerate(ls_bins):
    # Create a color gradient for the points in the bin
    num_points = corr_bin_variability.shape[1]
    gradient_colors = plt.cm.viridis(np.linspace(0, 1, num_points))
    
    if (p>2):
        k=1; p=0
    
    # Scatter plot for the current bin
    axes[p, k].scatter(corr_bin_variability[b, :, 0], euc_bin_variability[b, :, 0], c=gradient_colors, marker='o', edgecolor='k', alpha=0.9)
    axes[p, k].set_title('Number of participants %d'%bi)  # Add title for each bin
    axes[p, k].set_xlabel('Correlation')
    axes[p, k].set_ylabel('Euclidean Distance')

    p=p+1

plt.tight_layout()
#plt.show(block=True)


RandCons_vec = np.reshape(RandCons, (114,114, len(ls_bins)*nbPerm))
reconstruct_sc_vec = np.reshape(reconstruct_sc, (114,114, len(ls_bins)*nbPerm))
clustering_coeffs = []; efficiencies = []
clustering_coeffs_rec = []; efficiencies_rec = []
for i in range(RandCons_vec.shape[2]):
    if i%100==0:
        print('Matrix %d/%d'%(i,len(ls_bins)*nbPerm))
    matrix = RandCons_vec[:, :, i]
    matrix_rec = reconstruct_sc_vec[:,:,i]
    G = nx.from_numpy_array(matrix)
    G_rec = nx.from_numpy_array(matrix_rec)
    global_clustering = nx.transitivity(G) 
    global_clustering_rec = nx.transitivity(G_rec) 
    clustering_coeffs.append(global_clustering)
    clustering_coeffs_rec.append(global_clustering_rec)
    global_efficiency = nx.global_efficiency(G)
    global_efficiency_rec = nx.global_efficiency(G_rec)
    efficiencies.append(global_efficiency)
    efficiencies_rec.append(global_efficiency_rec)
clustering_coeffs = np.array(clustering_coeffs); efficiencies = np.array(efficiencies)
clustering_coeffs_rec = np.array(clustering_coeffs_rec); efficiencies_rec = np.array(efficiencies_rec)

fig, axes = plt.subplots(1, 2, figsize=(12, 15))
im1 = axes[0].imshow(matrix); axes[0].set_title('Original SC')
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
im2 = axes[1].imshow(np.abs(matrix_rec)); axes[1].set_title('Reconstructed SC (from aligned harmonics)')
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)


fig, axes = plt.subplots(1, 2, figsize=(12, 15))
axes[0].scatter(efficiencies, efficiencies_rec)
axes[0].set_xlabel('Efficiency - Original SC'); axes[0].set_ylabel('Efficiency - Reconstructed SC')
axes[1].scatter(clustering_coeffs, clustering_coeffs_rec)
axes[1].set_xlabel('Efficiency - Original SC'); axes[1].set_ylabel('Efficiency - Reconstructed SC')
#axes[0].plot(efficiencies); axes[0].plot(efficiencies_rec)
#axes[1].plot(clustering_coeffs); axes[1].plot(clustering_coeffs_rec)


nSubs = np.shape(ls_cutoff)[0]
nPerm = np.shape(ls_cutoff)[1]
nBins = np.shape(ls_cutoff)[2]
# Convert data to long format
participants = np.arange(nSubs).repeat(nPerm * nBins)
permutations = np.tile(np.arange(nPerm), nSubs * nBins)
bins = np.tile(np.arange(nBins), nSubs * nPerm)
values = ls_cutoff.flatten()
# Create a DataFrame
df = pd.DataFrame({'Participant': participants, 'Permutation': permutations, 'Bin': bins, 'Value': values})

# Use Seaborn to plot
plt.figure(figsize=(12, 6))
boxplot = sns.boxplot(data=df, x='Participant', y='Value', hue='Bin', palette='Set2', showfliers=False)
plt.title('Cut-off frequency')
plt.xlabel('Participant'); plt.ylabel('Cutoff Frequency')
plt.legend(title='Bin'); plt.grid('on')
sns.stripplot(data=df, x='Participant', y='Value', hue='Bin',  dodge=True, palette='dark:.3',  alpha=0.8,  size=2, linewidth=0.5,  edgecolor='gray')
handles, labels = boxplot.get_legend_handles_labels()  # Get handles and labels from the boxplot
plt.legend(handles=handles[:len(df['Bin'].unique())],  title='Bin', bbox_to_anchor=(1.05, 1), loc='upper left', labels=ls_bins) 

grouped_data = [df[df['Bin'] == bin]['Value'] for bin in df['Bin'].unique()]

# Perform Kruskal-Wallis test
stat, p_value = kruskal(*grouped_data)
print(f"Kruskal-Wallis test: H-statistic={stat:.3f}, p-value={p_value:.3f}")

posthoc = posthoc_dunn(df, val_col='Value', group_col='Bin', p_adjust='bonferroni')
print(posthoc)

plt.show(block=True)


### Reference gradient
#conn_matrix = RandCons[:,:,p,1]
#conn_matrix2 = RandCons[:,:,p,0]
#gref.fit(conn_matrix2)
#galign = GradientMaps(kernel='normalized_angle', approach='le', alignment='procrustes', n_components=114)
#galign.fit(conn_matrix, reference= gref.gradients_)
#plot_rois_pyvista(galign.gradients_[:,110]*100, config_defaults["Parameters"]["scale"], config_defaults, vmin=-2.5, vmax=2.5, label='TestAlignmentBrainspace')

## Load GIFTI files using nibabel
#fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
#right_surf = nb.load(fsaverage['pial_right'])  # GIFTI file for the right hemisphere
#left_surf = nb.load(fsaverage['pial_left'])    # GIFTI file for the left hemisphere
## Extract vertices and faces from the GIFTI file
#right_verts = right_surf.darrays[0].data  # vertices
#right_faces = right_surf.darrays[1].data  # faces
#left_verts = left_surf.darrays[0].data   # vertices
#left_faces = left_surf.darrays[1].data   # faces
    
## Check the format of right_faces and left_faces
#if right_faces.shape[1] == 3:
#    right_faces_pv = np.hstack([np.full((right_faces.shape[0], 1), 3), right_faces])
#    left_faces_pv = np.hstack([np.full((left_faces.shape[0], 1), 3), left_faces])
#else:
#    raise ValueError('Unexpected face format. Faces should have 3 vertices per face.')

## Create PolyData objects for right and left hemispheres
#surf_rh = pv.PolyData(right_verts, right_faces_pv)
#surf_lh = pv.PolyData(left_verts, left_faces_pv)
    
##plot_hemispheres(surf_lh, surf_rh, array_name=galign, size=(1200, 400), cmap='viridis_r', color_bar=True, label_text=['1'], zoom=1.5)
