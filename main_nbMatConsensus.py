
''' This script runs the analysis comparing how changing the number of matrices for the consensus changes the variability of the harmonics'''

import seaborn as sns
import numpy as np 
import scipy.io as sio
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import lib.func_reading as reading 
import lib.func_SDI as sdi
import lib.func_ML as ML
from lib.func_plot import plot_rois, plot_rois_pyvista
import random
from lib import fcn_groups_bin
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

config_path = 'config.json' #'config_PEP3.json'
config_defaults = reading.check_config_file(config_path)

### Reading the data
SC = sio.loadmat('./data/Individual_Connectomes.mat')   
#print(np.shape(SC['connMatrices']['SC'][0][0][1][0]))
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
#ls_bins = [2,10,20,30,40,50]
ls_bins = [10, 50]
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
        eigenvalues_perm_mat[:, b, p], eigenvectors_perm_mat[:, :, b, p],Ln_ind, An_ind = ML.cons_normalized_lap(RandCons[:,:,p,b], Euc, plot=False)
        labels_perm.append('Bin%d'%(bi))
        k = k+1 
        
        
### rotate for each bin only
eigenvalues_perm_mat_rot = np.zeros(np.shape(eigenvalues_perm_mat))
eigenvectors_perm_mat_rot = np.zeros(np.shape(eigenvectors_perm_mat))
R_all = np.zeros(np.shape(eigenvectors_perm_mat)); 
scale_R = np.zeros((len(ls_bins), nbPerm)); disparity = np.copy(scale_R)
for b,bi in enumerate(ls_bins):
    print(bi)
    ### Generalized Procrustes
    # eigenvectors_perm_mat_rot[:,:,b,:], eigenvalues_perm_mat_rot[:,b,:],  R_all[:,:,b,:], scale_R[b,:] = ML.rotation_procrustes(eigenvectors_perm_mat[:,:,b,:], eigenvalues_perm_mat[:,b,:], plot=True, p='bin%d'%bi)
    ### Orthogonal Procrustes
    eigenvectors_perm_mat_rot[:,:,b,:], eigenvalues_perm_mat_rot[:,b,:],  R_all[:,:,b,:], scale_R[b,:] = ML.orthogonal_rotation_procrustes(eigenvectors_perm_mat[:,:,b,:], eigenvalues_perm_mat[:,b,:], plot=True, p='bin%d'%bi)
    cos_sim_rot = np.diag(cosine_similarity(eigenvectors_perm_mat_rot[:,:,b,0], eigenvectors_perm_mat_rot[:,:,b,5]))
    cos_sim = np.diag(cosine_similarity(eigenvectors_perm_mat[:,:,b,0], eigenvectors_perm_mat[:,:,b,5]))
    plt.figure()
    plt.plot(np.abs(cos_sim))
    plt.plot(cos_sim_rot)
    plt.legend(['Before alignment', 'After alignment'])
    plt.title('Pairwise Cosine Similarity %s'%bi)
    plt.grid('on')
    plt.xlabel('Eigenvector'); plt.ylabel('Cosine Similarity')
    print('ok')
##for p in np.arange(nbPerm):
##    eigenvectors_perm_mat_rot[:,:,:,p],eigenvalues_perm_mat_rot[:,:,p],  R_all, scale_R = ML.rotation_procrustes(eigenvectors_perm_mat[:,:,:,p], eigenvalues_perm_mat[:,:,p], plot=True, p='bin%d'%p)#eigenvectors_perm = np.reshape(eigenvectors_perm_mat, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm_rot = np.reshape(eigenvectors_perm_mat_rot, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
eigenvectors_perm = np.reshape(eigenvectors_perm_mat, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))

X_RS_allPat = sdi.load_EEG_example()
P_ind, Q_ind, Ln_ind, An_ind = ML.cons_normalized_lap(matMetric, Euc, plot=False)




### Rotate for each bin and perm
#eigenvectors_perm = np.reshape(eigenvectors_perm_mat, ((len(cort_rois), nb_eig2keep, len(ls_bins)*nbPerm)))
#eigenvectors_perm_rot = np.zeros(np.shape(eigenvectors_perm))
#eigenvalues_perm_rot = np.zeros((nb_eig2keep, len(ls_bins)*nbPerm))
#eigenvectors_perm_rot, eigenvalues_perm_rot,  R_all, scale_R, disparity = ML.rotation_procrustes(eigenvectors_perm, eigenvalues_perm, plot=True, p='bin%d'%p)

#fig, axs = plt.subplots(1,1)
#axs.scatter(eigenvectors_perm_rot[:,50,50], eigenvectors_perm[:,50,50])
#plt.show()

#eig=1
#plot_rois_pyvista(eigenvectors_perm[:,eig,50]*10, config_defaults["Parameters"]["scale"], config_defaults, vmin=-2.5, vmax=2.5, label='NotRotated%d'%eig)
#plot_rois_pyvista(eigenvectors_perm_rot[:,eig,50]*100, config_defaults["Parameters"]["scale"], config_defaults, vmin=-2.5, vmax=2.5, label='Rotated%d'%eig)

#corr_eigvec_perm_before_after_rot = np.zeros((nb_eig2keep, len(ls_bins)*nbPerm, len(ls_bins)*nbPerm))
#for eig in np.arange(nb_eig2keep):
#    for k1 in np.arange(len(ls_bins)*nbPerm):
#        for k2 in np.arange(len(ls_bins)*nbPerm):    
#            tmp = np.corrcoef(eigenvectors_perm_rot[:,eig,k1],eigenvectors_perm_rot[:,eig,k2])
#            corr_eigvec_perm_before_after_rot[eig, k1,k2] = tmp[0,1]
#fig, axs = plt.subplots(1,1)
#axs.imshow(corr_eigvec_perm_before_after_rot[48,:,:]); 
#plt.show()

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
Dist_eigvec_perm_rot = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nb_eig2keep))
for eigvec_nb in np.arange(nb_eig2keep):
    MatDist = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb,:]), metric='correlation')
    #MatDist = scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb,:]), metric='euclidean')
    Dist_eigvec_perm[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist)
    #MatDist_rot = scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_rot[:, eigvec_nb,:]), metric='euclidean')
    MatDist_rot = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_rot[:, eigvec_nb,:]), metric='correlation')
    Dist_eigvec_perm_rot[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist_rot)


#### Remove the 0 values corresponding to the similarity between identical vectors
Dist_eigvec_perm_vec = np.reshape(Dist_eigvec_perm, (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_vec = np.abs(Dist_eigvec_perm_vec) ### Take absolute values for compensating for sign change 
Dist_eigvec_perm_rot_vec = np.reshape(Dist_eigvec_perm_rot, (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nb_eig2keep))
Dist_eigvec_perm_rot_vec = np.abs(Dist_eigvec_perm_rot_vec) ### Take absolute values for compensating for sign change 
tmp2 = Dist_eigvec_perm_rot_vec[:,30]; tmp2 = tmp2[np.where(tmp2>0)]


### Remove the 0 values corresponding here to the diagonal
for i in np.arange(nb_eig2keep):
    idxs_nz = np.where(Dist_eigvec_perm_vec[:,i])
    tmp = Dist_eigvec_perm_vec[:,i]; tmp = tmp[idxs_nz]
    tmp2 = Dist_eigvec_perm_rot_vec[:,i]; tmp2 = tmp2[idxs_nz]
    if i==0:
           Dist_eigvec_perm_vec_nz = np.zeros((len(tmp), nb_eig2keep))
           Dist_eigvec_perm_rot_vec_nz = np.zeros((len(tmp2), nb_eig2keep))
    Dist_eigvec_perm_vec_nz[:,i] = tmp 
    Dist_eigvec_perm_rot_vec_nz[:,i] = tmp2
labels_perm_bin = labels_perm_bin[idxs_nz] 
labels_perm_mat = labels_perm_mat[idxs_nz] 

bin_variability = np.zeros((len(ls_bins), nb_eig2keep, 2))
bin_variability_rot = np.zeros((len(ls_bins), nb_eig2keep, 2))
for b,bi in enumerate(ls_bins):
    idxs = np.where(labels_perm_mat=='Bin%d_Bin%d'%(bi,bi))[0]
    for i in np.arange(nb_eig2keep):
        #print(np.median(Dist_eigvec_perm_vec_nz[idxs,i]))
        bin_variability[b,i,0] = np.median(Dist_eigvec_perm_vec_nz[idxs,i])
        bin_variability[b,i,1] = np.std(Dist_eigvec_perm_vec_nz[idxs,i])
    #idxs_rot = np.where(labels_perm_rot_bin=='Bin%d'%bi)[0]
    #for i in np.arange(nb_eig2keep):
        bin_variability_rot[b,i,0] = np.median(Dist_eigvec_perm_rot_vec_nz[idxs,i])
        bin_variability_rot[b,i,1] = np.std(Dist_eigvec_perm_rot_vec_nz[idxs,i])        
fig, ax = plt.subplots(2,1,figsize=(15,5))
handles = []; handles_rot=[]  # To store the handles for lines in the plot
for b,bi in enumerate(ls_bins):
    line, = ax[0].plot(bin_variability[b,:,0]); handles.append(line);
    line_rot, = ax[1].plot(bin_variability_rot[b,:,0]); handles_rot.append(line_rot);
    upper_bound = bin_variability[b, :, 0] + bin_variability[b, :, 1]
    lower_bound = bin_variability[b, :, 0] - bin_variability[b, :, 1]
    upper_bound_rot = bin_variability_rot[b, :, 0] + bin_variability_rot[b, :, 1]
    lower_bound_rot = bin_variability_rot[b, :, 0] - bin_variability_rot[b, :, 1]
    ax[0].fill_between(range(nb_eig2keep), lower_bound, upper_bound, alpha=0.1)
    ax[1].fill_between(range(nb_eig2keep), lower_bound_rot, upper_bound_rot, alpha=0.1)
for x in range(2):
    ax[x].set_xlabel('Eigenmode'); ax[x].set_xticks(range(0, nb_eig2keep, 10))
    ax[x].grid('on'); ax[x].set_ylim([0,1.02]); ax[x].set_ylabel('Correlation'); ax[x].set_title('Similarity between network harmonics', fontsize=15);
ax[0].legend(handles=handles, labels=ls_bins, loc='lower left', title='Bin');
ax[1].legend(handles=handles_rot, labels=ls_bins, loc='lower left', title='Bin');


    
plt.show()