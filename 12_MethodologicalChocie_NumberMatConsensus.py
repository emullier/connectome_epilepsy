
''' This script runs the analysis comparing how changing the number of matrices for the consensus changes the variability of the harmonics'''

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


example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"

### Reading the data
#SC = sio.loadmat('./data/Individual_Connectomes.mat')   
#SC = SC['connMatrices']['SC'][0][0][1][0]
#roi_info_path = 'data/label/roi_info.xlsx'
#roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
#cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
#matMetric = SC
#x = np.asarray(roi_info['x-pos'])[cort_rois] 
#y = np.asarray(roi_info['y-pos'])[cort_rois]
#z = np.asarray(roi_info['z-pos'])[cort_rois]
#coordMat = np.concatenate((x[:,None],y[:,None],z[:,None]),1)
#Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))  
matMetric = np.load("DATA/matMetric_SCHZ_CTRL.npy")
matMetric = np.transpose(matMetric, (1, 2, 0))
Euc = np.load("DATA/EucMat_HC_DSI_number_of_fibers.npy")
Euc = np.mean(Euc, axis=2)  # Average the Euclidean distance matrix across participants
cort_rois = np.arange(len(Euc))

### Generate random group based on different number of participants
total_participant = np.shape(matMetric)[2]; nROIs = np.shape(matMetric)[0]
idxs = list(range(total_participant))
ls_bins = [1,5,10,15,20,25, 30]
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
        eigenvalues_perm_mat[:, b, p], eigenvectors_perm_mat[:, :, b, p],Ln_ind, An_ind = gsp.cons_normalized_lap(RandCons[:,:,p,b], Euc, plot=False)
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
    ### Generalized Procrustes
    eigenvectors_perm_mat_rot[:,:,b,:], eigenvalues_perm_mat_rot[:,b,:], A, B = gsp.rotation_procrustes(eigenvectors_perm_mat[:,:,b,:], eigenvalues_perm_mat[:,b,:], plot=False, p='bin%d'%bi)
    ### Orthogonal Procrustes
    eigenvectors_perm_mat_ortho[:,:,b,:], eigenvalues_perm_mat_ortho[:,b,:],  R_all[:,:,b,:], scale_R[b,:] = gsp.orthogonal_rotation_procrustes(eigenvectors_perm_mat[:,:,b,:], eigenvalues_perm_mat[:,b,:], plot=False, p='bin%d'%bi)
    cos_sim_ortho = np.diag(cosine_similarity(eigenvectors_perm_mat_ortho[:,:,b,0], eigenvectors_perm_mat_ortho[:,:,b,5]))
    cos_sim = np.diag(cosine_similarity(eigenvectors_perm_mat[:,:,b,0], eigenvectors_perm_mat[:,:,b,5]))
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

fig, ax = plt.subplots(4,1,figsize=(15,5))
handles = []; handles_rot=[]  # To store the handles for lines in the plot
for b,bi in enumerate(ls_bins):
    line, = ax[0].plot(bin_variability[b,:,0]); handles.append(line);
    line_ortho, = ax[1].plot(bin_variability_ortho[b,:,0]); handles_rot.append(line_ortho);
    line_matched, = ax[2].plot(bin_variability_matched[b,:,0]); handles_rot.append(line_matched);
    line_rot, = ax[3].plot(bin_variability_rot[b,:,0]); handles_rot.append(line_rot);
    upper_bound = bin_variability[b, :, 0] + bin_variability[b, :, 1]
    lower_bound = bin_variability[b, :, 0] - bin_variability[b, :, 1]
    upper_bound_ortho = bin_variability_ortho[b, :, 0] + bin_variability_ortho[b, :, 1]
    lower_bound_ortho = bin_variability_ortho[b, :, 0] - bin_variability_ortho[b, :, 1]
    upper_bound_matched = bin_variability_matched[b, :, 0] + bin_variability_matched[b, :, 1]
    lower_bound_matched = bin_variability_matched[b, :, 0] - bin_variability_matched[b, :, 1]
    upper_bound_rot = bin_variability_rot[b, :, 0] + bin_variability_rot[b, :, 1]
    lower_bound_rot = bin_variability_rot[b, :, 0] - bin_variability_rot[b, :, 1]
    ax[0].fill_between(range(nb_eig2keep), lower_bound, upper_bound, alpha=0.1)
    ax[1].fill_between(range(nb_eig2keep), lower_bound_ortho, upper_bound_ortho, alpha=0.1)
    ax[2].fill_between(range(nb_eig2keep), lower_bound_matched, upper_bound_matched, alpha=0.1)
    ax[3].fill_between(range(nb_eig2keep), lower_bound_rot, upper_bound_rot, alpha=0.1)
for x in range(4):
    ax[x].set_xlabel('Eigenmode'); ax[x].set_xticks(range(0, nb_eig2keep, 10))
    ax[x].grid('on'); ax[x].set_ylim([0,1.02]); ax[x].set_ylabel('Correlation'); #ax[x].set_title('Similarity between network harmonics', fontsize=15);
ax[0].legend(handles=handles, labels=ls_bins, loc='lower left', title='Bin');
ax[1].legend(handles=handles_rot, labels=ls_bins, loc='lower left', title='Bin');
ax[2].legend(handles=handles, labels=ls_bins, loc='lower left', title='Bin');
ax[3].legend(handles=handles_rot, labels=ls_bins, loc='lower left', title='Bin');
ax[0].set_title('Similarity between network harmonics', fontsize=15);
ax[1].set_title('After Orthogonal Procrustes', fontsize=15);
ax[2].set_title('After Hungarian matching', fontsize=15);
ax[3].set_title('After Generalized Procrustes', fontsize=15);

    
plt.show()