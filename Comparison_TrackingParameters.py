

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

data_path = "DATA/Connectome_scale-2.mat"
matMetric = sio.loadmat(data_path)
matMetric = matMetric['num']
cort_rois = np.concatenate((np.arange(0,57), [62,63], np.arange(64,121), [126,127]))
matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
consensus_ind = matMetric

#data_path = "DATA/matrices_tracking3"
#tmp = os.listdir(data_path)
#ls_subs = []
#for s,sub in enumerate(tmp):
#    if sub.startswith('sub-'):
#        ls_subs.append(sub)
#ls_subs = np.array(ls_subs)
#matMetric = np.zeros((len(ls_subs), matMetric.shape[0], matMetric.shape[1]))
#for s,sub in enumerate(ls_subs):
#    tmp = sio.loadmat(os.path.join(data_path,sub))['sc']['number_of_fibers'][0][0]
#    df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
#    rois_118 = df_118['ID Lausanne2008']
#    rois_118 = np.array(rois_118)
#    tmp = tmp[rois_118,:]; tmp = tmp[:, rois_118]
#    matMetric[s,:,:] = tmp
#np.save('DATA/matMetric_tracking3.npy', matMetric)


consensus_HC_DSI = np.load("DATA/matMetric_HC_DSI_number_of_fibers.npy")
consensus_schz = np.load("DATA/matMetric_SCHZ_CTRL.npy")
consensus_deter = np.load("DATA/matMetric_deterministic.npy")
consensus_proba = np.load("DATA/matMetric_probabilistic.npy")
consensus_tracking1 = np.load("DATA/matMetric_tracking1.npy")
consensus_tracking2 = np.load("DATA/matMetric_tracking2.npy")
consensus_tracking3 = np.load("DATA/matMetric_tracking3.npy")
HC_dsi_mat = np.mean(consensus_HC_DSI,axis=2)
#np.fill_diagonal(HC_dsi_mat, 0)
HC_dsi_vec = (HC_dsi_mat.flatten())
ind_vec = (consensus_ind.flatten())
deter_vec = (consensus_deter.flatten())
proba_vec = (consensus_proba.flatten())
tracking1_vec = (consensus_tracking1.mean(axis=0).flatten())
tracking2_vec = (consensus_tracking2.mean(axis=0).flatten())
tracking3_vec = (consensus_tracking3.mean(axis=0).flatten())
schz_vec = (np.mean(consensus_schz, axis=0).flatten())

fig,axs = plt.subplots(2,5,figsize=(10,10), constrained_layout=True)
axs[0,0].imshow(np.mean(consensus_HC_DSI,axis=2)); axs[0,0].set_title('Consensus HC DSI GVA');
axs[0,1].imshow(consensus_ind); axs[0,1].set_title('Independent 70p. Consensus')
axs[0,2].imshow(np.mean(consensus_schz, axis=0)); axs[0,2].set_title('Independent 27p. Consensus')
axs[0,3].imshow(consensus_deter); axs[0,3].set_title('Deterministic 15p.')
axs[0,4].imshow(consensus_proba); axs[0,4].set_title('Probabilistic 15p.')
idxs = np.where((ind_vec>0)*(HC_dsi_vec>0))[0]
axs[1,0].scatter(ind_vec[idxs], HC_dsi_vec[idxs], c='k', alpha=0.5)   
axs[1,0].set_xlabel('Ind. 70p.'); axs[1,0].set_ylabel('HC DSI GVA')
idxs = np.where((HC_dsi_vec>0)*(schz_vec>0))[0]
axs[1,1].scatter(HC_dsi_vec[idxs], schz_vec[idxs], c='k', alpha=0.5)
axs[1,1].set_xlabel('HC DSI GVA'); axs[1,1].set_ylabel('Ind. 27p.')   
idxs = np.where((ind_vec>0)*(schz_vec>0))[0]
axs[1,2].scatter(ind_vec[idxs], schz_vec[idxs], c='k', alpha=0.5) 
axs[1,2].set_xlabel('Ind. 70p.'); axs[1,2].set_ylabel('Ind. 27p.')
idxs = np.where((ind_vec>0)*(deter_vec>0))[0]
axs[1,3].scatter(ind_vec[idxs], deter_vec[idxs], c='k', alpha=0.5)
axs[1,3].set_xlabel('Ind. 70p.'); axs[1,3].set_ylabel('Deterministic 15p.')
idxs = np.where((ind_vec>0)*(proba_vec>0))[0]
axs[1,4].scatter(ind_vec[idxs], proba_vec[idxs], c='k', alpha=0.5)
axs[1,4].set_xlabel('Ind. 70p.'); axs[1,4].set_ylabel('Probabilistic 15p.')



fig,axs = plt.subplots(2,3,figsize=(10,10), constrained_layout=True)

axs[0,0].imshow(consensus_tracking1.mean(axis=0)); axs[0,0].set_title('Tracking 1')
axs[0,1].imshow(consensus_tracking2.mean(axis=0)); axs[0,1].set_title('Tracking 2')
axs[0,2].imshow(consensus_tracking3.mean(axis=0)); axs[0,2].set_title('Tracking 3')
HC_dsi_mat = np.mean(consensus_HC_DSI,axis=2)
idxs = np.where((ind_vec>0)*(tracking1_vec>0))[0]
axs[1,0].scatter(ind_vec[idxs], tracking1_vec[idxs], c='k', alpha=0.5)
idxs = np.where((ind_vec>0)*(tracking2_vec>0))[0]
axs[1,1].scatter(ind_vec[idxs], tracking2_vec[idxs], c='k', alpha=0.5)
idxs = np.where((ind_vec>0)*(tracking3_vec>0))[0]
axs[1,2].scatter(ind_vec[idxs], tracking3_vec[idxs], c='k', alpha=0.5)
plt.show()