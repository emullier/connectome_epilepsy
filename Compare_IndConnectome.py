import os
import numpy as np 
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.stats import linregress
import seaborn as sns
import pandas as pd
import scipy
from scipy.stats import wilcoxon
from sklearn.cross_decomposition import CCA
import lib.func_GSP as gsp
from scipy.stats import friedmanchisquare
from scipy.stats import kruskal
import scipy.io as sio
from lib import fcn_groups_bin

data_path = "DATA/Connectome_scale-2.mat"
matMetric = sio.loadmat(data_path)
matMetric = matMetric['num']
cort_rois = np.concatenate((np.arange(0,57), [62,63], np.arange(64,121), [126,127]))
matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
consensus_ind = matMetric

metric = "number_of_fibers" #'fiber_length_mean' #"number_of_fibers"
consensus_HC_DSI = np.load("DATA/matMetric_HC_dsi_%s.npy"%metric) 
consensus_EP_DSI = np.load("DATA/matMetric_EP_dsi_%s.npy"%metric) 

mat_schz = np.load("DATA/matMetric_SCHZ_CTRL.npy")
#print(np.mean(np.mean(consensus_HC_DSI,axis=2)))
#print(np.mean(consensus_ind))
EucDist = np.load("DATA/EucMat_HC_dsi_number_of_fibers.npy")

fig, axs = plt.subplots(2, 4, figsize=(10, 10), constrained_layout=True)
HC_dsi_mat = np.mean(consensus_HC_DSI, axis=2)
EP_dsi_mat = np.mean(consensus_EP_DSI, axis=2)
np.fill_diagonal(HC_dsi_mat, 0)
np.fill_diagonal(EP_dsi_mat, 0)
HC_dsi_vec = scipy.stats.zscore(HC_dsi_mat.flatten())
EP_dsi_vec = scipy.stats.zscore(EP_dsi_mat.flatten())
ind_vec = scipy.stats.zscore(consensus_ind.flatten())
hemii = np.ones(np.shape(EucDist)[0])
hemii[int(len(hemii)/2):] = 2
[G, Gc] = fcn_groups_bin.fcn_groups_bin(np.transpose(mat_schz, (1, 2, 0)), np.mean(EucDist, axis=2), hemii, 200)
consensus_schz = G
im0 = axs[0, 0].imshow(HC_dsi_mat); axs[0, 0].set_title('HC DSI')
fig.colorbar(im0, ax=axs[0, 0])
im1 = axs[0, 1].imshow(consensus_ind); axs[0, 1].set_title('Ind')
fig.colorbar(im1, ax=axs[0, 1])
im2 = axs[0, 2].imshow(G); axs[0, 2].set_title('Schz')
fig.colorbar(im2, ax=axs[0, 2])
im3 = axs[0, 3].imshow(EP_dsi_mat); axs[0, 3].set_title('EP DSI')
fig.colorbar(im3, ax=axs[0, 3])
schz_vec = scipy.stats.zscore(consensus_schz.flatten())
idxs = np.where((HC_dsi_vec > 0) * (ind_vec > 0))[0]
axs[1, 0].scatter(HC_dsi_vec[idxs], ind_vec[idxs], c='k', alpha=0.5)
idxs = np.where((HC_dsi_vec > 0) * (schz_vec > 0))[0]
axs[1, 1].scatter(HC_dsi_vec[idxs], schz_vec[idxs], c='k', alpha=0.5)
idxs = np.where((ind_vec > 0) * (schz_vec > 0))[0]
axs[1, 2].scatter(ind_vec[idxs], schz_vec[idxs], c='k', alpha=0.5)
idxs = np.where((EP_dsi_vec > 0) * (HC_dsi_vec > 0))[0]
axs[1, 3].scatter(EP_dsi_vec[idxs], HC_dsi_vec[idxs], c='k', alpha=0.5)




### Comparison deterministic vs probabilistic
deter_path = "DATA/deterministic_matrices"
proba_path = "DATA/probabilistic_matrices"

tmp = os.listdir(deter_path)
ls_subs_deter = []
for s,sub in enumerate(tmp):
    if sub.startswith('sub'):
        ls_subs_deter.append(sub)
ls_subs_deter = np.array(ls_subs_deter)
tmp = os.listdir(proba_path)
ls_subs_proba = []
for s,sub in enumerate(tmp):
    if sub.startswith('sub'):
        ls_subs_proba.append(sub)
ls_subs_proba = np.array(ls_subs_proba)

for s,sub in enumerate(ls_subs_deter):
    tmp = sio.loadmat(os.path.join(deter_path, sub))
    tmp = tmp['sc'][metric][0][0]
    if s==0:
        matDeter = np.zeros((np.shape(tmp)[0], np.shape(tmp)[0], len(ls_subs_deter)))
    matDeter[:,:,s] = tmp
for s,sub in enumerate(ls_subs_proba):
    tmp = sio.loadmat(os.path.join(proba_path, sub))
    tmp = tmp['sc'][metric][0][0]
    if s==0:
        matProba = np.zeros((np.shape(tmp)[0], np.shape(tmp)[0], len(ls_subs_proba)))
    matProba[:,:,s] = tmp

df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
rois_118 = df_118['ID Lausanne2008']
rois_118 = np.array(rois_118)
cons_deter = np.mean(matDeter,axis=2)
cons_proba = np.mean(matProba,axis=2)
cons_deter = cons_deter[rois_118,:]; cons_deter = cons_deter[:, rois_118]
cons_proba = cons_proba[rois_118,:]; cons_proba = cons_proba[:, rois_118]
np.save("DATA/matMetric_deterministic.npy", cons_deter)
np.save("DATA/matMetric_probabilistic.npy", cons_proba)

#deter = sio.loadmat("deterministic_sub-01_atlas-L2018_res-scale1_conndata-network_connectivity.mat") 
#proba = sio.loadmat("probabilistic_sub-01_atlas-L2018_res-scale1_conndata-network_connectivity.mat") 
#matDeter = deter['sc'][metric][0][0]
#matProba = proba['sc'][metric][0][0] 

fig, axs = plt.subplots(1, 2, figsize=(10, 10), constrained_layout=True)
im0 = axs[0].imshow(np.mean(matDeter,axis=2)); axs[0].set_title('Deterministic')
#fig.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(np.mean(matProba,axis=2)); axs[1].set_title('Probabilistic')
#fig.colorbar(im1, ax=axs[1])



# Prepare the figure and axes for the distributions
# fig, axs = plt.subplots(1, 1, figsize=(15, 5), constrained_layout=True)
#HC_dsi_mat = np.mean(consensus_HC_DSI, axis=2)
#np.fill_diagonal(HC_dsi_mat, 0)
#HC_dsi_vec = scipy.stats.zscore(HC_dsi_mat.flatten())
#ind_vec = scipy.stats.zscore(consensus_ind.flatten())
#schz_vec = scipy.stats.zscore(np.mean(consensus_schz, axis=0).flatten())
##sns.kdeplot(HC_dsi_vec, ax=axs, color='blue', fill=True, alpha=0.6)
#sns.kdeplot(ind_vec, ax=axs, color='red', fill=True, alpha=0.6)
#sns.kdeplot(schz_vec, ax=axs, color='green', fill=True, alpha=0.6)

plt.show()