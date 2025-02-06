
'''This script plots the variability of the harmonics using different consensus with the GVA data. '''

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
import lib.func_utils as utils
import os

config_path = 'config.json' #'config_PEP3.json'
config_defaults = reading.check_config_file(config_path)
df_info = reading.read_info(config_defaults['Parameters']['info_path'])
filters = utils.compare_pdkeys_list(df_info, config_defaults['Parameters']['filters'])
df, ls_subs = reading.filtered_dataframe(df_info, filters, config_defaults)

MatMat = {}; EucMat = {}
procs = list(config_defaults["Parameters"]["processing"].keys())
for p, proc in enumerate(procs):    
    idxs_tmp = np.where((df[proc] == 1) | (df[proc] == '1'))[0]
    df_tmp = df.iloc[idxs_tmp]
    tmp_path = os.path.join(config_defaults["Parameters"]["data_dir"], config_defaults["Parameters"]["processing"][proc]) 
    MatMat[proc], EucMat[proc], df_info = reading.load_matrices(df_tmp, tmp_path, config_defaults['Parameters']['scale'], config_defaults['Parameters']['metric'])

df = df_info
matMetric = MatMat['shore']
EucMat = np.mean(EucMat['shore'],axis=2)

G_dist = np.zeros((114,114,4))
gp = ['HC', 'EP']
dwi = ['dsi', 'multishell']
P_ind = np.zeros((114,4))
Q_ind = np.zeros((114,114,4))

labels_perm = []
k=0
for g,group in enumerate(gp):
    for d,dw in enumerate(dwi):
        idxs = np.where((df['group']==group)*df['dwi']==dw)[0]
        hemii = np.ones(np.shape(EucMat)[0])
        hemii[int(len(hemii)/2):] = 2
        nbins = 42
        tmp, G_unif = fcn_groups_bin.fcn_groups_bin(matMetric[:,:,idxs], EucMat, hemii, nbins) 
        G_dist[:,:,k] = np.mean(matMetric[:,:,idxs],axis=2)
        #G_dist_wei, G_unif_wei = reading.save_consensus(MatMat[:,:,idxs], config["Parameters"]["metric"], G_dist, G_unif, config["Parameters"]["output_dir"], config["Parameters"]["processing"])
        P_ind[:,k], Q_ind[:,:,k], Ln_ind, An_ind = ML.cons_normalized_lap(G_dist[:,:,k], EucMat,  plot=False)
        if np.shape(Q_ind[:,:,k])!=114:
            Q_ind[:,:,k] = utils.extract_ctx_ROIs(Q_ind[:,:,k])
        labels_perm.append('%s_%s'%(group, dw))
        k = k+1
    
Q_ind_rot, P_ind_rot,  R_all, scale_R = ML.rotation_procrustes(Q_ind, P_ind, plot=True, p='k%d'%k)
    
np.save(os.path.join(os.getcwd(), 'output', 'epilepsy','Qind_%d_beforeRot_main'%(k)), Q_ind)
np.save(os.path.join(os.getcwd(), 'output', 'epilepsy','Qind_rot_%d_afterRot_main'%(k)), Q_ind_rot)
### Generate the eigenvectors 
nb_eig2keep = np.shape(Q_ind)[0]

labels_perm_mat = []
labels_perm_bin = []
for i in np.arange(len(labels_perm)):
    for j in np.arange(len(labels_perm)):  
        labels_perm_mat.append('%s_%s'%(labels_perm[i],labels_perm[j]))
        if labels_perm[i]==labels_perm[j]:
            labels_perm_bin.append(0)
        else:
            labels_perm_bin.append(1)
labels_perm_bin = np.array(labels_perm_bin)
labels_perm_mat = np.array(labels_perm_mat)


### Compute the similarity betwen all the eigenvectors (all bins and randomization)
Dist_eigvec_perm = np.zeros((4, 4, nb_eig2keep))
Dist_eigvec_perm_rot = np.zeros((4,4, nb_eig2keep))
for eigvec_nb in np.arange(nb_eig2keep):
    MatDist = 1 - scipy.spatial.distance.pdist(np.transpose(Q_ind[:,eigvec_nb,:]), metric='correlation')
    #MatDist = scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb,:]), metric='euclidean')
    Dist_eigvec_perm[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist)
    #MatDist_rot = scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm_rot[:, eigvec_nb,:]), metric='euclidean')
    MatDist_rot = 1 - scipy.spatial.distance.pdist(np.transpose(Q_ind_rot[:,eigvec_nb,:]), metric='correlation')
    Dist_eigvec_perm_rot[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist_rot)



#### Remove the 0 values corresponding to the similarity between identical vectors
Dist_eigvec_perm_vec = np.reshape(Dist_eigvec_perm, (16, nb_eig2keep))
Dist_eigvec_perm_vec = np.abs(Dist_eigvec_perm_vec) ### Take absolute values for compensating for sign change 
Dist_eigvec_perm_rot_vec = np.reshape(Dist_eigvec_perm_rot, (16, nb_eig2keep))
Dist_eigvec_perm_rot_vec = np.abs(Dist_eigvec_perm_rot_vec) ### Take absolute values for compensating for sign change 


### Remove the 0 values corresponding here to the diagonal
#for i in np.arange(nb_eig2keep):
#    idxs_nz = np.where(Dist_eigvec_perm_vec[:,i])
#    tmp = Dist_eigvec_perm_vec[:,i]; tmp = tmp[idxs_nz]
#    tmp2 = Dist_eigvec_perm_rot_vec[:,i]; tmp2 = tmp2[idxs_nz]
#    if i==0:
#           Dist_eigvec_perm_vec_nz = np.zeros((len(tmp), nb_eig2keep))
#           Dist_eigvec_perm_rot_vec_nz = np.zeros((len(tmp2), nb_eig2keep))
#    Dist_eigvec_perm_vec_nz[:,i] = tmp 
#    Dist_eigvec_perm_rot_vec_nz[:,i] = tmp2
Dist_eigvec_perm_rot_vec_nz = Dist_eigvec_perm_rot_vec
Dist_eigvec_perm_vec_nz = Dist_eigvec_perm_vec
#labels_perm_bin = labels_perm_bin[idxs_nz] 
#labels_perm_mat = labels_perm_mat[idxs_nz] 

print(labels_perm_mat)

fig, ax = plt.subplots(2,1,figsize=(15,5))
handles = []; handles_rot=[]
idxs = np.where(labels_perm_bin>0)[0] 
for b in np.arange(len(idxs)):
    line, = ax[0].plot(Dist_eigvec_perm_vec_nz[idxs[b],:]); handles.append(line);
    line_rot, = ax[1].plot(Dist_eigvec_perm_rot_vec_nz[idxs[b],:]); handles_rot.append(line_rot);    
ax[1].legend(labels_perm_mat[idxs])
ax[0].grid('on')
    
#bin_variability = np.zeros((16, nb_eig2keep, 2))
#bin_variability_rot = np.zeros((16, nb_eig2keep, 2))
#for b,dw in enumerate(dwi):
###    idxs = np.where('dsi' in labels_perm_mat)[0]
  #  for i in np.arange(nb_eig2keep):
  #      #print(np.median(Dist_eigvec_perm_vec_nz[idxs,i]))
   #     bin_variability[b,i,0] = np.median(Dist_eigvec_perm_vec_nz[idxs,i])
   #     bin_variability[b,i,1] = np.std(Dist_eigvec_perm_vec_nz[idxs,i])
   # #idxs_rot = np.where(labels_perm_rot_bin=='Bin%d'%bi)[0]
   # #for i in np.arange(nb_eig2keep):
   #     bin_variability_rot[b,i,0] = np.median(Dist_eigvec_perm_rot_vec_nz[idxs,i])
   #     bin_variability_rot[b,i,1] = np.std(Dist_eigvec_perm_rot_vec_nz[idxs,i])        
#fig, ax = plt.subplots(2,1,figsize=(15,5))
#handles = []; handles_rot=[]  # To store the handles for lines in the plot
#for b,dw in enumerate(dwi):
#    line, = ax[0].plot(bin_variability[b,:,0]); handles.append(line);
#    line_rot, = ax[1].plot(bin_variability_rot[b,:,0]); handles_rot.append(line_rot);
#    upper_bound = bin_variability[b, :, 0] + bin_variability[b, :, 1]
#    lower_bound = bin_variability[b, :, 0] - bin_variability[b, :, 1]
#    upper_bound_rot = bin_variability_rot[b, :, 0] + bin_variability_rot[b, :, 1]
#    lower_bound_rot = bin_variability_rot[b, :, 0] - bin_variability_rot[b, :, 1]
#    ax[0].fill_between(range(nb_eig2keep), lower_bound, upper_bound, alpha=0.1)
#    ax[1].fill_between(range(nb_eig2keep), lower_bound_rot, upper_bound_rot, alpha=0.1)
#for x in range(2):
#    ax[x].set_xlabel('Eigenmode'); ax[x].set_xticks(range(0, nb_eig2keep, 10))
#    ax[x].grid('on'); ax[x].set_ylim([0,1.02]); ax[x].set_ylabel('Correlation'); ax[x].set_title('Similarity between network harmonics', fontsize=15);
#ax[0].legend(handles=handles, loc='lower left', title='Bin');
#ax[1].legend(handles=handles_rot, loc='lower left', title='Bin');


    
plt.show()