
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

consensus_HC_DSI = np.load("DATA/matMetric_HC_DSI_number_of_fibers.npy")
consensus_schz = np.load("DATA/matMetric_SCHZ_CTRL.npy")
consensus_deter = np.load("DATA/matMetric_deterministic.npy")
consensus_proba = np.load("DATA/matMetric_probabilistic.npy")
#print(np.mean(np.mean(consensus_HC_DSI,axis=2)))
#print(np.mean(consensus_ind))

fig,axs = plt.subplots(2,5,figsize=(10,10), constrained_layout=True)
axs[0,0].imshow(np.mean(consensus_HC_DSI,axis=2)); axs[0,0].set_title('Consensus HC DSI GVA');
axs[0,1].imshow(consensus_ind); axs[0,1].set_title('Independent 70p. Consensus')
axs[0,2].imshow(np.mean(consensus_schz, axis=0)); axs[0,2].set_title('Independent 27p. Consensus')
axs[0,3].imshow(consensus_deter); axs[0,3].set_title('Deterministic 15p.')
axs[0,4].imshow(consensus_proba); axs[0,4].set_title('Probabilistic 15p.')
HC_dsi_mat = np.mean(consensus_HC_DSI,axis=2)
#np.fill_diagonal(HC_dsi_mat, 0)
HC_dsi_vec = scipy.stats.zscore(HC_dsi_mat.flatten())
ind_vec = scipy.stats.zscore(consensus_ind.flatten())
deter_vec = scipy.stats.zscore(consensus_deter.flatten())
proba_vec = scipy.stats.zscore(consensus_proba.flatten())
schz_vec = scipy.stats.zscore(np.mean(consensus_schz, axis=0).flatten())
idxs = np.where((HC_dsi_vec>0)*(ind_vec>0))[0]
axs[1,0].scatter(HC_dsi_vec[idxs], ind_vec[idxs], c='k', alpha=0.5)   
axs[1,0].set_xlabel('HC DSI GVA'); axs[1,0].set_ylabel('Ind. 70p.')
idxs = np.where((HC_dsi_vec>0)*(schz_vec>0))[0]
axs[1,1].scatter(HC_dsi_vec[idxs], schz_vec[idxs], c='k', alpha=0.5)
axs[1,1].set_xlabel('HC DSI GVA'); axs[1,1].set_ylabel('Ind. 27p.')   
idxs = np.where((ind_vec>0)*(schz_vec>0))[0]
axs[1,2].scatter(ind_vec[idxs], schz_vec[idxs], c='k', alpha=0.5) 
axs[1,2].set_xlabel('Ind. 70p.'); axs[1,2].set_ylabel('Ind. 27p.')
idxs = np.where((deter_vec>0)*(proba_vec>0))[0]
axs[1,3].scatter(deter_vec[idxs], proba_vec[idxs], c='k', alpha=0.5)
axs[1,3].set_xlabel('Deterministic 15p.'); axs[1,3].set_ylabel('Probabilistic 15p.')
idxs = np.where((HC_dsi_vec>0)*(deter_vec>0))[0]
axs[1,4].scatter(HC_dsi_vec[idxs], deter_vec[idxs], c='k', alpha=0.5)
axs[1,4].set_xlabel('HC DSI GVA'); axs[1,4].set_ylabel('Deterministic 15p.')


### Comparision Cutoff frequencies
cutoff_HC = np.load('./OUTPUT/cutoff_HC_dsi.npy')
cutoff_EP = np.load('./OUTPUT/cutoff_EP_dsi.npy')
cutoff_Iso = np.load('./OUTPUT/cutoff_Iso.npy')
cutoff_schz = np.load('./OUTPUT/cutoff_schz.npy')
cutoff_deter = np.load('./OUTPUT/cutoff_deterministic.npy')
cutoff_proba = np.load('./OUTPUT/cutoff_probabilistic.npy')

stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_EP, alternative='two-sided')
print(f"HC - EP \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_Iso, alternative='two-sided')
print(f"HC - Independent \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_EP, cutoff_Iso, alternative='two-sided')
print(f"EP - Independent \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_deter, cutoff_proba, alternative='two-sided')
print(f"Deterministic \n Probabilistic = {stat:.3f}, p-value = {p_mwu:.3g}")
r_value, p_value = scipy.stats.pearsonr(cutoff_HC, cutoff_EP)
stat, p_kw = kruskal(cutoff_HC, cutoff_EP, cutoff_Iso, cutoff_schz)
print(f"Kruskal–Wallis H = {stat:.3f}, p = {p_kw:.3g}")


fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].scatter(cutoff_HC, cutoff_EP, c='tab:blue', alpha=0.6, edgecolors='w', s=60)
ax[0].set_title(f"Pearson r = {r_value:.3f}, p = {p_value:.3f}", fontsize=10)
ax[0].set_xlabel('Cutoff frequency HC')
ax[0].set_ylabel('Cutoff frequency EP')
ax[0].grid(True)
sns.boxplot(data=[cutoff_HC, cutoff_EP, cutoff_Iso], ax=ax[1], width=0.5, palette='colorblind')
sns.stripplot(data=[cutoff_HC, cutoff_EP, cutoff_Iso], ax=ax[1], color='black', size=4, jitter=True, dodge=True)
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(['HC Consensus', 'EP Consensus', 'Independent \n Consensus'])
ax[1].set_ylabel('Cutoff frequency')
ax[1].set_title(f'Kruskal–Wallis H = {stat:.3f}, p = {p_kw:.3g}', fontsize=10)
ax[1].grid(True, axis='y', linestyle='--', alpha=0.5)
ax[2].scatter(cutoff_deter, cutoff_proba, c='tab:blue', alpha=0.6, edgecolors='w', s=60)
ax[2].set_title(f"Pearson r = {r_value:.3f}, p = {p_value:.3f}", fontsize=10)
ax[2].set_xlabel('Cutoff frequency deterministic')
ax[2].set_ylabel('Cutoff frequency probabilistic')
ax[2].grid(True)
plt.tight_layout()

### Comparison SDI for each participant
SDI_HC_LT = np.load('./OUTPUT/SDI_HC_dsi_LT.npy')
SDI_EP_LT = np.load('./OUTPUT/SDI_EP_dsi_LT.npy')
SDI_HC_RT = np.load('./OUTPUT/SDI_HC_dsi_RT.npy')
SDI_EP_RT = np.load('./OUTPUT/SDI_EP_dsi_RT.npy')
SDI_Iso_RT = np.load('./OUTPUT/SDI_Iso_RT.npy')
SDI_Iso_LT = np.load('./OUTPUT/SDI_Iso_LT.npy')
SDI_schz_LT = np.load('./OUTPUT/SDI_schz_LT.npy')   
SDI_schz_RT = np.load('./OUTPUT/SDI_schz_RT.npy')
SDI_deter_RT = np.load('./OUTPUT/SDI_deterministic_RT.npy')
SDI_proba_RT = np.load('./OUTPUT/SDI_probabilistic_RT.npy')
SDI_deter_LT = np.load('./OUTPUT/SDI_deterministic_LT.npy')
SDI_proba_LT = np.load('./OUTPUT/SDI_probabilistic_LT.npy')


### Comparison SDI for each participant - singificant values only
surr_thresh_HC_LT = np.load('./OUTPUT/SDI_surr_thresh_HC_dsi_LT.npy', allow_pickle=True)
surr_thresh_HC_RT = np.load('./OUTPUT/SDI_surr_thresh_HC_dsi_RT.npy', allow_pickle=True)
surr_thresh_EP_RT = np.load('./OUTPUT/SDI_surr_thresh_EP_dsi_RT.npy', allow_pickle=True)
surr_thresh_EP_LT = np.load('./OUTPUT/SDI_surr_thresh_EP_dsi_LT.npy', allow_pickle=True)
surr_thresh_Iso_RT = np.load('./OUTPUT/SDI_surr_thresh_Iso_RT.npy', allow_pickle=True)
surr_thresh_Iso_LT = np.load('./OUTPUT/SDI_surr_thresh_Iso_LT.npy', allow_pickle=True)
surr_thresh_schz_LT = np.load('./OUTPUT/SDI_surr_thresh_schz_LT.npy', allow_pickle=True)
surr_thresh_schz_RT = np.load('./OUTPUT/SDI_surr_thresh_schz_RT.npy', allow_pickle=True)
surr_thresh_deter_RT = np.load('./OUTPUT/SDI_surr_thresh_deterministic_RT.npy', allow_pickle=True)
surr_thresh_deter_LT = np.load('./OUTPUT/SDI_surr_thresh_deterministic_LT.npy', allow_pickle=True)
surr_thresh_proba_RT = np.load('./OUTPUT/SDI_surr_thresh_probabilistic_RT.npy', allow_pickle=True)
surr_thresh_proba_LT = np.load('./OUTPUT/SDI_surr_thresh_probabilistic_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_HC_LT = np.load('./OUTPUT/SDI_sig_subjectwise_HC_dsi_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_LT = np.load('./OUTPUT/SDI_sig_subjectwise_EP_dsi_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_HC_RT = np.load('./OUTPUT/SDI_sig_subjectwise_HC_dsi_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_RT = np.load('./OUTPUT/SDI_sig_subjectwise_EP_dsi_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_Iso_RT = np.load('./OUTPUT/SDI_sig_subjectwise_Iso_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_Iso_LT = np.load('./OUTPUT/SDI_sig_subjectwise_Iso_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_schz_LT = np.load('./OUTPUT/SDI_sig_subjectwise_schz_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_schz_RT = np.load('./OUTPUT/SDI_sig_subjectwise_schz_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_deter_RT = np.load('./OUTPUT/SDI_sig_subjectwise_deterministic_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_deter_LT = np.load('./OUTPUT/SDI_sig_subjectwise_deterministic_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_proba_RT = np.load('./OUTPUT/SDI_sig_subjectwise_probabilistic_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_proba_LT = np.load('./OUTPUT/SDI_sig_subjectwise_probabilistic_LT.npy', allow_pickle=True)

nROIs = 118
mean_SDI_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_HC_LT = np.copy(mean_SDI_HC_RT); mean_SDI_EP_LT = np.copy(mean_SDI_HC_RT); mean_SDI_EP_RT = np.copy(mean_SDI_HC_RT)
SDI_sig_HC_LT = np.copy(SDI_sig_HC_RT); SDI_sig_EP_LT = np.copy(SDI_sig_HC_RT); SDI_sig_EP_RT = np.copy(SDI_sig_HC_RT)
mean_SDI_Iso_RT = np.copy(mean_SDI_HC_RT); mean_SDI_Iso_LT = np.copy(mean_SDI_HC_RT)
SDI_sig_Iso_RT = np.copy(SDI_sig_HC_RT); SDI_sig_Iso_LT = np.copy(SDI_sig_HC_RT)
mean_SDI_Iso_LT = np.copy(mean_SDI_HC_RT); SDI_sig_Iso_LT = np.copy(SDI_sig_HC_RT)  
SDI_sig_Iso_LT = np.copy(SDI_sig_HC_RT); SDI_sig_Iso_RT = np.copy(SDI_sig_HC_RT)
mean_SDI_schz_LT = np.copy(mean_SDI_HC_RT); mean_SDI_schz_RT = np.copy(mean_SDI_HC_RT)
SDI_sig_schz_LT = np.copy(SDI_sig_HC_RT); SDI_sig_schz_RT = np.copy(SDI_sig_HC_RT)
mean_SDI_deter_RT = np.zeros((np.shape(surr_thresh_deter_RT)[0], nROIs))
mean_SDI_deter_LT = np.zeros((np.shape(surr_thresh_deter_LT)[0], nROIs))
mean_SDI_proba_RT = np.zeros((np.shape(surr_thresh_proba_RT)[0], nROIs))
mean_SDI_proba_LT = np.zeros((np.shape(surr_thresh_proba_LT)[0], nROIs))
SDI_sig_deter_RT = np.copy(mean_SDI_deter_RT); SDI_sig_deter_LT = np.copy(mean_SDI_deter_LT)
SDI_sig_proba_RT = np.copy(mean_SDI_proba_RT); SDI_sig_proba_LT = np.copy(mean_SDI_proba_LT)

### Comparing average SDI values LT   
for s in np.arange(np.shape(surr_thresh_HC_RT)[0]):
    th = surr_thresh_HC_RT[s]['threshold']
    mean_SDI_HC_RT[s,:] = surr_thresh_HC_RT[s]['mean_SDI']
    SDI_sig_HC_RT[s,:] = surr_thresh_HC_RT[s]['SDI_sig']
    mean_SDI_EP_RT[s,:] = surr_thresh_EP_RT[s]['mean_SDI']
    SDI_sig_EP_RT[s,:] = surr_thresh_EP_RT[s]['SDI_sig']
    mean_SDI_Iso_RT[s,:] = surr_thresh_Iso_RT[s]['mean_SDI']
    SDI_sig_Iso_RT[s,:] = surr_thresh_Iso_RT[s]['SDI_sig']
    mean_SDI_schz_RT[s,:] = surr_thresh_schz_RT[s]['mean_SDI']
    SDI_sig_schz_RT[s,:] = surr_thresh_schz_RT[s]['SDI_sig']
    mean_SDI_deter_RT[s,:] = surr_thresh_deter_RT[s]['mean_SDI']
    SDI_sig_deter_RT[s,:] = surr_thresh_deter_RT[s]['SDI_sig']
    mean_SDI_proba_RT[s,:] = surr_thresh_proba_RT[s]['mean_SDI']
    SDI_sig_proba_RT[s,:] = surr_thresh_proba_RT[s]['SDI_sig']
for s in np.arange(np.shape(surr_thresh_HC_LT)[0]):
    th = surr_thresh_HC_LT[s]['threshold']
    mean_SDI_HC_LT[s,:] = surr_thresh_HC_LT[s]['mean_SDI']
    SDI_sig_HC_LT[s,:] = surr_thresh_HC_LT[s]['SDI_sig']
    mean_SDI_EP_LT[s,:] = surr_thresh_EP_LT[s]['mean_SDI']
    SDI_sig_EP_LT[s,:] = surr_thresh_EP_LT[s]['SDI_sig']
    mean_SDI_Iso_LT[s,:] = surr_thresh_Iso_LT[s]['mean_SDI']
    SDI_sig_Iso_LT[s,:] = surr_thresh_Iso_LT[s]['SDI_sig']
    mean_SDI_schz_LT[s,:] = surr_thresh_schz_LT[s]['mean_SDI']
    SDI_sig_schz_LT[s,:] = surr_thresh_schz_LT[s]['SDI_sig']    
    mean_SDI_deter_RT[s,:] = surr_thresh_deter_RT[s]['mean_SDI']
    mean_SDI_deter_LT[s,:] = surr_thresh_deter_LT[s]['mean_SDI']    
    SDI_sig_deter_LT[s,:] = surr_thresh_deter_LT[s]['SDI_sig']
    mean_SDI_proba_LT[s,:] = surr_thresh_proba_LT[s]['mean_SDI']
    SDI_sig_proba_LT[s,:] = surr_thresh_proba_LT[s]['SDI_sig']

fig, axs = plt.subplots(4, 2, figsize=(10,10), constrained_layout=True) 
axs[0,0].scatter(mean_SDI_HC_RT[0,:], mean_SDI_EP_RT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_RT[s,:], mean_SDI_EP_RT[s,:])
axs[0,0].set_title(f"RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0,0].set_xlabel('Mean SDI HC SC'); axs[0,0].set_ylabel('Mean SDI RTLE SC')
axs[0,1].scatter(mean_SDI_HC_LT[0,:], mean_SDI_EP_LT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_LT[s,:], mean_SDI_EP_LT[s,:])
axs[0,1].set_title(f"LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0,1].set_xlabel('Mean SDI SC'); axs[0,1].set_ylabel('Mean SDI LTLE SC')
axs[1,1].scatter(mean_SDI_Iso_LT[0,:], mean_SDI_HC_LT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_Iso_LT[s,:], mean_SDI_HC_LT[s,:])
axs[1,1].set_title(f"Ind. LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)        
axs[1,1].set_xlabel('Mean SDI Ind SC'); axs[1,1].set_ylabel('Mean SDI HC SC')
axs[1,0].scatter(mean_SDI_Iso_RT[0,:], mean_SDI_HC_RT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_Iso_RT[s,:], mean_SDI_HC_RT[s,:])
axs[1,0].set_title(f"Ind. RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[1,0].set_xlabel('Mean SDI Ind SC'); axs[1,0].set_ylabel('Mean SDI HC SC')
axs[2,0].scatter(mean_SDI_schz_RT[0,:], mean_SDI_HC_RT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_schz_RT[s,:], mean_SDI_HC_RT[s,:]) 
axs[2,0].set_title(f"27 subs RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[2,0].set_xlabel('Mean SDI 27 subs SC'); axs[2,0].set_ylabel('Mean SDI HC SC')                           
axs[2,1].scatter(mean_SDI_schz_LT[0,:], mean_SDI_HC_LT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_schz_LT[s,:], mean_SDI_HC_LT[s,:]) 
axs[2,1].set_title(f"27 subs LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[2,1].set_xlabel('Mean SDI 27 subs SC'); axs[2,1].set_ylabel('Mean SDI HC SC')
#axs[3,0].scatter(mean_SDI_schz_LT[0,:], mean_SDI_Iso_LT[0,:], c='k', alpha=0.5)
#[r,p]= scipy.stats.pearsonr(mean_SDI_schz_LT[s,:], mean_SDI_Iso_LT[s,:])    
#axs[3,0].set_title(f"27 subs LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
#axs[3,0].set_xlabel('Mean SDI 27 subs SC'); axs[3,0].set_ylabel('Mean SDI Ind SC')
#axs[3,1].scatter(mean_SDI_schz_RT[0,:], mean_SDI_Iso_RT[0,:], c='k', alpha=0.5)
#[r,p]= scipy.stats.pearsonr(mean_SDI_schz_RT[s,:], mean_SDI_Iso_RT[s,:])
#axs[3,1].set_title(f"27 subs RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
#axs[3,1].set_xlabel('Mean SDI 27 subs SC'); axs[3,1].set_ylabel('Mean SDI Ind SC')
axs[3,0].scatter(mean_SDI_deter_LT[0,:], mean_SDI_proba_LT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_deter_LT[s,:], mean_SDI_proba_LT[s,:])    
axs[3,0].set_title(f"Deterministic x Probabilistic LT\n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[3,0].set_xlabel('Deterministic'); axs[3,0].set_ylabel('Probabilistic')
axs[3,1].scatter(mean_SDI_deter_RT[0,:], mean_SDI_proba_RT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_deter_RT[s,:], mean_SDI_proba_RT[s,:])
axs[3,1].set_title(f"Deterministic x Probabilistic RT \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[3,1].set_xlabel('Deterministic'); axs[3,1].set_ylabel('Probabilistic')


### Threshold 3 participants
#labels = pd.read_csv('./DATA/label/labels_114.csv')
labels = pd.read_csv('./DATA/label/labels_rois_118.csv')
#labels = df_tmp['Label Lausanne2008']
df_SDI_3part_sig = pd.DataFrame()

mean_SDI_sig = surr_thresh_Iso_LT[2]['mean_SDI']*surr_thresh_Iso_LT[2]['SDI_sig']
idxs = np.where(mean_SDI_sig>0)[0]
#df_SDI_3part_sig['ROIs'] = labels['labels'].iloc[idxs]
df_SDI_3part_sig['ROIs'] = labels['Label Lausanne2008'].iloc[idxs]

df_SDI_3part_sig['SDI'] = surr_thresh_Iso_LT[2]['mean_SDI'][idxs]
print(df_SDI_3part_sig)
df_SDI_3part_sig.to_csv('./OUTPUT/SDI_3part_sig_Iso_LT.csv')

### Comparing individual SDI values LT
n_plots = SDI_EP_LT.shape[1]
fig, ax = plt.subplots(2, n_plots, figsize=(5 * n_plots, 6), constrained_layout=True)
for s in range(n_plots):
    ax[0,s].scatter(np.log(SDI_EP_LT[:, s]), np.log(SDI_HC_LT[:, s]), color='steelblue', alpha=0.6, edgecolor='k', s=50)
    r, p = scipy.stats.pearsonr(np.log(SDI_EP_LT[:, s]), np.log(SDI_HC_LT[:, s]))
    common_pos = (SDI_sig_subjectwise_EP_LT[:, s] == 1) & (SDI_sig_subjectwise_HC_LT[:, s] == 1)
    common_neg = (SDI_sig_subjectwise_EP_LT[:, s] == -1) & (SDI_sig_subjectwise_HC_LT[:, s] == -1)
    num_common_pos = np.sum(common_pos)
    num_common_neg = np.sum(common_neg)
    ax[0,s].set_title(f"Patient {s+1}\n$r$ = {r:.2f}, $p$ = {p:.2e}\nCommon sig = {num_common_pos + num_common_neg}", fontsize=12)
    ax[0,s].set_xlabel('log(SDI) EP LT', fontsize=11)
    ax[0,s].set_ylabel('log(SDI) HC LT', fontsize=11)
    ax[0,s].tick_params(axis='both', labelsize=10)
    ax[0,s].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax[1,s].scatter(np.log(SDI_Iso_LT[:, s]), np.log(SDI_schz_LT[:, s]), color='steelblue', alpha=0.6, edgecolor='k', s=50)
    r, p = scipy.stats.pearsonr(np.log(SDI_Iso_LT[:, s]), np.log(SDI_HC_LT[:, s]))
    common_pos = (SDI_sig_subjectwise_Iso_LT[:, s] == 1) & (SDI_sig_subjectwise_schz_LT[:, s] == 1)
    common_neg = (SDI_sig_subjectwise_schz_LT[:, s] == -1) & (SDI_sig_subjectwise_schz_LT[:, s] == -1)
    num_common_pos = np.sum(common_pos)
    num_common_neg = np.sum(common_neg)
    ax[1,s].set_title(f"Ind. {s+1}\n$r$ = {r:.2f}, $p$ = {p:.2e}\n Common sig = {num_common_pos + num_common_neg}", fontsize=12)
    ax[1,s].set_xlabel('log(SDI) Ind. LT', fontsize=11)
    ax[1,s].set_ylabel('log(SDI) schz LT', fontsize=11)
    ax[1,s].tick_params(axis='both', labelsize=10)
    ax[1,s].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
fig.suptitle("Comparison of Individual SDI Values (LTLE)", fontsize=16)

### Comparing individual SDI values RT
n_plots = SDI_EP_LT.shape[1]
fig, ax = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6), constrained_layout=True)
for s in range(n_plots):
    ax[s].scatter(np.log(SDI_EP_RT[:, s]), np.log(SDI_HC_RT[:, s]), color='steelblue', alpha=0.6, edgecolor='k', s=50)
    r, p = scipy.stats.pearsonr(np.log(SDI_EP_RT[:, s]), np.log(SDI_HC_RT[:, s]))
    common_pos = (SDI_sig_subjectwise_EP_RT[:, s] == 1) & (SDI_sig_subjectwise_HC_RT[:, s] == 1)
    common_neg = (SDI_sig_subjectwise_EP_RT[:, s] == -1) & (SDI_sig_subjectwise_HC_RT[:, s] == -1)
    num_common_pos = np.sum(common_pos)
    num_common_neg = np.sum(common_neg)
    ax[s].set_title(f"Patient {s+1}\n$r$ = {r:.2f}, $p$ = {p:.2e}\nCommon sig = {num_common_pos + num_common_neg}", fontsize=12)
    ax[s].set_xlabel('log(SDI) EP RT', fontsize=11)
    ax[s].set_ylabel('log(SDI) HC RT', fontsize=11)
    ax[s].tick_params(axis='both', labelsize=10)
    ax[s].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
fig.suptitle("Comparison of Individual SDI Values (RTLE)", fontsize=16)

### Comparing harmonics values
Qind_HC_RT = np.load('./OUTPUT/Q_ind_HC_dsi_RT.npy')
Qind_HC_LT = np.load('./OUTPUT/Q_ind_HC_dsi_LT.npy')
Qind_EP_RT = np.load('./OUTPUT/Q_ind_EP_dsi_RT.npy')
Qind_EP_LT = np.load('./OUTPUT/Q_ind_EP_dsi_LT.npy')
Qind_Iso_LT = np.load('./OUTPUT/Q_ind_Iso_LT.npy')
Qind_Iso_RT = np.load('./OUTPUT/Q_ind_Iso_RT.npy')
Qind_schz_RT = np.load('./OUTPUT/Q_ind_schz_RT.npy')
Qind_schz_LT = np.load('./OUTPUT/Q_ind_schz_LT.npy')
#Q_rand = np.random.permutation(Qind_HC_RT.flatten()).reshape(Qind_HC_RT.shape)
Q_rand2 = np.random.rand(*Qind_HC_RT.shape)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(Q_rand2, cmap='gray'); plt.title('Randomized Matrix')
plt.colorbar(fraction=0.046, pad=0.04, aspect=20)
plt.subplot(1,2,2)
plt.imshow(Qind_HC_RT, cmap='gray'); plt.title('HC DSI')
plt.colorbar(fraction=0.046, pad=0.04, aspect=20)


Qind_EP_RT_rotated, Qind_HC_RT_centered, disparity_RT = scipy.spatial.procrustes(Qind_HC_RT, Qind_EP_RT)
Qind_EP_LT_rotated, Qind_HC_LT_centered, disparity_LT = scipy.spatial.procrustes(Qind_HC_LT, Qind_EP_LT)
Qind_Iso_RT_rotated, Qind_HC_RT_centered, disparity_RT = scipy.spatial.procrustes(Qind_HC_RT, Qind_Iso_RT)
Qind_rand_rotated, Qind_HC_RT_centered, disparity_RT = scipy.spatial.procrustes(Qind_HC_RT, Q_rand2)
Qind_schz_RT_rotated, Qind_HC_RT_centered, disparity_RT = scipy.spatial.procrustes(Qind_HC_RT, Qind_schz_RT)
R_RT, _ = scipy.linalg.orthogonal_procrustes(Qind_HC_RT, Qind_EP_RT)
Qind_EP_RT_ortho_rotated=Qind_EP_RT@R_RT 
R_LT, _ = scipy.linalg.orthogonal_procrustes(Qind_HC_LT, Qind_EP_LT)
Qind_EP_LT_ortho_rotated=Qind_EP_LT@R_LT 
R_Iso_RT, _ = scipy.linalg.orthogonal_procrustes(Qind_HC_RT, Qind_Iso_RT)
Qind_Iso_RT_ortho_rotated=Qind_Iso_RT@R_Iso_RT 
R_rand, _ = scipy.linalg.orthogonal_procrustes(Qind_HC_RT, Q_rand2)
Qind_rand_ortho_rotated=Q_rand2@R_rand
R_schz_RT, _ = scipy.linalg.orthogonal_procrustes(Qind_HC_RT, Qind_schz_RT)
Qind_schz_RT_ortho_rotated=Qind_schz_RT@R_schz_RT

perm, total_cost = gsp.match_eigenvectors(Qind_HC_RT, Qind_EP_RT)
Qind_EP_RT_matched = Qind_EP_RT[:,perm]
perm, total_cost = gsp.match_eigenvectors(Qind_HC_LT, Qind_EP_LT)
Qind_EP_LT_matched = Qind_EP_LT[:,perm]
perm, total_cost = gsp.match_eigenvectors(Qind_HC_RT, Qind_Iso_RT)
Qind_Iso_RT_matched = Qind_Iso_RT[:,perm]
perm, total_cost = gsp.match_eigenvectors(Qind_HC_RT, Q_rand2)
Qind_rand_matched = Q_rand2[:,perm]
perm, total_cost = gsp.match_eigenvectors(Qind_HC_RT, Qind_schz_RT)
Qind_schz_RT_matched = Qind_schz_RT[:,perm]

pearson_corrs_LT = np.zeros(Qind_EP_LT.shape[1])
pearson_corrs_RT = np.zeros(Qind_EP_RT.shape[1])
pearson_corrs_Iso_RT = np.zeros(Qind_Iso_RT.shape[1])
pearson_corrs_rand = np.zeros(Q_rand2.shape[1])
pearson_corrs_schz_RT = np.zeros(Qind_schz_RT.shape[1])
pearson_corrs_LT_rot = np.zeros(Qind_EP_LT.shape[1])
pearson_corrs_RT_rot = np.zeros(Qind_EP_RT.shape[1])
pearson_corrs_Iso_RT_rot = np.zeros(Qind_Iso_RT.shape[1])
pearson_corrs_rand_rot = np.zeros(Q_rand2.shape[1])
pearson_corrs_schz_RT_rot = np.zeros(Qind_schz_RT.shape[1])
pearson_corrs_LT_orth_rot = np.zeros(Qind_EP_LT.shape[1])
pearson_corrs_RT_orth_rot = np.zeros(Qind_EP_RT.shape[1])
pearson_corrs_Iso_RT_orth_rot = np.zeros(Qind_Iso_RT.shape[1])
pearson_corrs_rand_orth_rot = np.zeros(Q_rand2.shape[1])
pearson_corrs_schz_RT_orth_rot = np.zeros(Qind_schz_RT.shape[1])
pearson_corrs_LT_matched = np.zeros(Qind_EP_LT.shape[1])
pearson_corrs_RT_matched = np.zeros(Qind_EP_RT.shape[1])
pearson_corrs_Iso_RT_matched = np.zeros(Qind_Iso_RT.shape[1])
pearson_corrs_rand_matched = np.zeros(Q_rand2.shape[1])
pearson_corrs_schz_RT_matched = np.zeros(Qind_schz_RT.shape[1])

for i in range(Qind_EP_LT.shape[1]):
    # Calculate Pearson correlation between the i-th column of A and B
    corr_LT, _ = scipy.stats.pearsonr(Qind_EP_LT[:,i], Qind_HC_LT[:, i])
    corr_RT, _ = scipy.stats.pearsonr(Qind_EP_RT[:, i], Qind_HC_RT[:, i])
    corr_Iso_RT, _ = scipy.stats.pearsonr(Qind_Iso_RT[:, i], Qind_HC_RT[:, i])
    corr_rand, _ = scipy.stats.pearsonr(Q_rand2[:, i], Qind_HC_RT[:, i])
    corr_schz_RT, _ = scipy.stats.pearsonr(Qind_schz_RT[:, i], Qind_HC_RT[:, i])
    corr_LT_rot, _ = scipy.stats.pearsonr(Qind_EP_LT_rotated[:,i], Qind_HC_LT_centered[:, i])
    corr_RT_rot, _ = scipy.stats.pearsonr(Qind_EP_RT_rotated[:, i], Qind_HC_RT_centered[:, i])
    corr_Iso_RT_rot, _ = scipy.stats.pearsonr(Qind_Iso_RT_rotated[:, i], Qind_HC_RT_centered[:, i])
    corr_rand_rot, _ = scipy.stats.pearsonr(Qind_rand_rotated[:, i], Qind_HC_RT_centered[:, i])
    corr_schz_RT_rot, _ = scipy.stats.pearsonr(Qind_schz_RT_rotated[:, i], Qind_HC_RT_centered[:, i])
    corr_LT_orth_rot, _ = scipy.stats.pearsonr(Qind_EP_LT_ortho_rotated[:,i], Qind_HC_LT[:, i])
    corr_RT_orth_rot, _ = scipy.stats.pearsonr(Qind_EP_RT_ortho_rotated[:, i], Qind_HC_RT[:, i])
    corr_Iso_RT_orth_rot, _ = scipy.stats.pearsonr(Qind_Iso_RT_ortho_rotated[:, i], Qind_HC_RT[:, i])
    corr_rand_orth_rot, _ = scipy.stats.pearsonr(Qind_rand_ortho_rotated[:, i], Qind_HC_RT[:, i])
    corr_schz_RT_orth_rot, _ = scipy.stats.pearsonr(Qind_schz_RT_ortho_rotated[:, i], Qind_HC_RT[:, i])
    corr_LT_matched, _ = scipy.stats.pearsonr(Qind_EP_LT_matched[:,i], Qind_HC_LT_centered[:, i])
    corr_RT_matched, _ = scipy.stats.pearsonr(Qind_EP_RT_matched[:, i], Qind_HC_RT_centered[:, i])
    corr_Iso_RT_matched, _ = scipy.stats.pearsonr(Qind_Iso_RT_matched[:, i], Qind_HC_RT_centered[:, i])
    corr_rand_matched, _ = scipy.stats.pearsonr(Qind_rand_matched[:, i], Qind_HC_RT_centered[:, i])
    corr_schz_RT_matched, _ = scipy.stats.pearsonr(Qind_schz_RT_matched[:, i], Qind_HC_RT_centered[:, i])
    pearson_corrs_LT[i] = np.abs(corr_LT)
    pearson_corrs_RT[i] = np.abs(corr_RT)
    pearson_corrs_Iso_RT[i] = np.abs(corr_Iso_RT)
    pearson_corrs_rand[i] = np.abs(corr_rand)
    pearson_corrs_schz_RT[i] = np.abs(corr_schz_RT)
    pearson_corrs_LT_rot[i] = np.abs(corr_LT_rot)
    pearson_corrs_RT_rot[i] = np.abs(corr_RT_rot)
    pearson_corrs_Iso_RT_rot[i] = np.abs(corr_Iso_RT_rot)
    pearson_corrs_rand_rot[i] = np.abs(corr_rand_rot)
    pearson_corrs_schz_RT_rot[i] = np.abs(corr_schz_RT_rot)
    pearson_corrs_LT_orth_rot[i] = np.abs(corr_LT_orth_rot)
    pearson_corrs_RT_orth_rot[i] = np.abs(corr_RT_orth_rot)
    pearson_corrs_Iso_RT_orth_rot[i] = np.abs(corr_Iso_RT_orth_rot)
    pearson_corrs_rand_orth_rot[i] = np.abs(corr_rand_orth_rot)
    pearson_corrs_schz_RT_orth_rot[i] = np.abs(corr_schz_RT_orth_rot)
    pearson_corrs_LT_matched[i] = np.abs(corr_LT_matched)
    pearson_corrs_RT_matched[i] = np.abs(corr_RT_matched)
    pearson_corrs_Iso_RT_matched[i] = np.abs(corr_Iso_RT_matched)
    pearson_corrs_rand_matched[i] = np.abs(corr_rand_matched)
    pearson_corrs_schz_RT_matched[i] = np.abs(corr_schz_RT_matched)
    
# Plot the Pears
# on correlation coefficients for each pair of columns
fig, axs = plt.subplots(4,1,figsize=(15, 20), constrained_layout=True)
axs[0].plot(range(1, len(pearson_corrs_LT) + 1), pearson_corrs_LT, linestyle='-', color='b')
axs[0].plot(range(1, len(pearson_corrs_RT) + 1), pearson_corrs_RT, linestyle='-', color='r')
axs[0].plot(range(1, len(pearson_corrs_Iso_RT) + 1), pearson_corrs_Iso_RT, linestyle='-', color='g')   
axs[0].plot(range(1, len(pearson_corrs_rand) + 1), pearson_corrs_rand, linestyle='-', color='k') 
#axs[0].plot(range(1, len(pearson_corrs_schz_RT) + 1), pearson_corrs_schz_RT, linestyle='-', color='m')
axs[0].set_title("Pearson Correlation between Corresponding Eigenvectors ")
axs[0].set_xlabel("Eigenvector Index"); axs[0].set_ylabel("Pearson Correlation Coefficient")
axs[0].legend(["LTLE", "RTLE", "Ind.", "Rand."]); axs[0].grid(True)
axs[3].plot(range(1, len(pearson_corrs_LT_rot) + 1), pearson_corrs_LT_rot, linestyle='-', color='b')
axs[3].plot(range(1, len(pearson_corrs_RT_rot) + 1), pearson_corrs_RT_rot, linestyle='-', color='r')
axs[3].plot(range(1, len(pearson_corrs_Iso_RT_rot) + 1), pearson_corrs_Iso_RT_rot, linestyle='-', color='g')
axs[3].plot(range(1, len(pearson_corrs_rand_rot) + 1), pearson_corrs_rand_rot, linestyle='-', color='k')
#axs[3].plot(range(1, len(pearson_corrs_schz_RT_rot) + 1), pearson_corrs_schz_RT_rot, linestyle='-', color='m')
axs[3].set_title("Pearson Correlation between Rotated Eigenvectors ")
axs[3].set_xlabel("Eigenvector Index"); axs[3].set_ylabel("Pearson Correlation Coefficient")
axs[3].legend(["LTLE", "RTLE", "Ind.", "Rand."]); axs[3].grid(True)
axs[1].plot(range(1, len(pearson_corrs_LT_matched) + 1), pearson_corrs_LT_matched, linestyle='-', color='b')
axs[1].plot(range(1, len(pearson_corrs_RT_matched) + 1), pearson_corrs_RT_matched, linestyle='-', color='r')
axs[1].plot(range(1, len(pearson_corrs_Iso_RT_matched) + 1), pearson_corrs_Iso_RT_matched, linestyle='-', color='g')
axs[1].plot(range(1, len(pearson_corrs_rand_matched) + 1), pearson_corrs_rand_matched, linestyle='-', color='k')
#axs[1].plot(range(1, len(pearson_corrs_schz_RT_matched) + 1), pearson_corrs_schz_RT_matched, linestyle='-', color='m')
axs[1].set_title("Pearson Correlation between Matched Eigenvectors ")
axs[1].set_xlabel("Eigenvector Index"); axs[1].set_ylabel("Pearson Correlation Coefficient")
axs[1].legend(["LTLE", "RTLE", "Ind.", "Rand."]); axs[1].grid(True)
axs[2].plot(range(1, len(pearson_corrs_LT_orth_rot) + 1), pearson_corrs_LT_orth_rot, linestyle='-', color='b')
axs[2].plot(range(1, len(pearson_corrs_RT_orth_rot) + 1), pearson_corrs_RT_orth_rot, linestyle='-', color='r')
axs[2].plot(range(1, len(pearson_corrs_Iso_RT_orth_rot) + 1), pearson_corrs_Iso_RT_orth_rot, linestyle='-', color='g')
axs[2].plot(range(1, len(pearson_corrs_rand_orth_rot) + 1), pearson_corrs_rand_orth_rot, linestyle='-', color='k')
#axs[2].plot(range(1, len(pearson_corrs_schz_RT_orth_rot) + 1), pearson_corrs_schz_RT_orth_rot, linestyle='-', color='m')
axs[2].set_title("Pearson Correlation between Orthogonal Rotated Eigenvectors ")
axs[2].set_xlabel("Eigenvector Index"); axs[2].set_ylabel("Pearson Correlation Coefficient")
axs[2].legend(["LTLE", "RTLE", "Ind.", "Rand."]); axs[2].grid(True)

nbROIs_EP_RT = np.load('./OUTPUT/nbROIs_sig_EP_dsi_RT.npy')
nbROIs_EP_LT = np.load('./OUTPUT/nbROIs_sig_EP_dsi_LT.npy')
nbROIs_HC_RT = np.load('./OUTPUT/nbROIs_sig_HC_dsi_RT.npy')
nbROIs_HC_LT = np.load('./OUTPUT/nbROIs_sig_HC_dsi_LT.npy')
nbROIs_Iso_RT = np.load('./OUTPUT/nbROIs_sig_Iso_RT.npy')
nbROIs_Iso_LT = np.load('./OUTPUT/nbROIs_sig_Iso_LT.npy')
nbROIs_schz_RT = np.load('./OUTPUT/nbROIs_sig_schz_RT.npy')
nbROIs_schz_LT = np.load('./OUTPUT/nbROIs_sig_schz_LT.npy')


fig, ax = plt.subplots(1,1)
#ls_nbROIs_sig = [nbROIs_HC_RT, nbROIs_HC_LT, nbROIs_EP_RT, nbROIs_EP_LT]
#ls_surr_thresh = [surr_thresh_HC_RT, surr_thresh_HC_LT, surr_thresh_EP_RT, surr_thresh_EP_LT]
#ls_labels = ["HC RT", "HC LT", "EP RT", "EP LT"]
ls_nbROIs_sig = [nbROIs_HC_RT, nbROIs_HC_LT, nbROIs_EP_RT, nbROIs_EP_LT, nbROIs_Iso_RT, nbROIs_Iso_LT]
ls_surr_thresh = [surr_thresh_HC_RT, surr_thresh_HC_LT, surr_thresh_EP_RT, surr_thresh_EP_LT, surr_thresh_Iso_RT, surr_thresh_Iso_LT]
ls_labels = ["HC RT", "HC LT", "EP RT", "EP LT", "Ind. RT", "Ind. LT"]
ls_labels = np.array(ls_labels)

for t in np.arange(len(ls_labels)):
    surr_thresh = ls_surr_thresh[t]
    nbROIs_sig = ls_nbROIs_sig[t]
    ax.plot(np.arange(np.shape(surr_thresh)[0]), np.array(nbROIs_sig), marker='x', linewidth=2)
    for i, y_value in enumerate(nbROIs_sig):
        #ax.scatter(i, y_value, marker='x', s=20)  # Cross marker
        ax.text(i, y_value, f'{y_value}', fontsize=8, ha='left', va='bottom', color='k')
    #ax.axvline(x=0.75*np.shape(surr_thresh)[0], color='r', linestyle='--', linewidth=2, label='75% of participants')
    ax.set_xlabel('Number of EEG participants'); ax.set_ylabel('#ROIs with significant SDI')
    ax.set_xticks(np.arange(0, np.shape(surr_thresh)[0]+1))
    ax.grid('on', alpha=.2)
ax.legend(ls_labels, loc='upper right')
ax.set_title('Number of significant SDI ROIs')



#df_SDI_EP_LT = pd.read_csv('./OUTPUT/SDI_3part_sig_EP_LT.csv')
#df_SDI_EP_RT = pd.read_csv('./OUTPUT/SDI_3part_sig_EP_RT.csv')
#df_SDI_HC_LT = pd.read_csv('./OUTPUT/SDI_3part_sig_HC_LT.csv')
#df_SDI_HC_RT = pd.read_csv('./OUTPUT/SDI_3part_sig_HC_RT.csv')

#common_rois = set(df_SDI_EP_LT['ROIs']) & set(df_SDI_EP_RT['ROIs']) & set(df_SDI_HC_LT['ROIs']) & set(df_SDI_HC_RT['ROIs'])
#print("Common ROIs across all dataframes:")
#print(sorted(common_rois))

#common_rois_RT =  set(df_SDI_EP_RT['ROIs'])  & set(df_SDI_HC_RT['ROIs'])
#print("Common ROIs RT:")
#print(sorted(common_rois_RT))

#common_rois_LT =  set(df_SDI_EP_LT['ROIs'])  & set(df_SDI_HC_LT['ROIs'])
#print("Common ROIs LT:")
#print(sorted(common_rois_LT))
                                
#common_rois_HC =  set(df_SDI_HC_LT['ROIs'])  & set(df_SDI_HC_RT['ROIs'])
#print("Common ROIs HC:")
#print(sorted(common_rois_HC))

#common_rois_EP =  set(df_SDI_EP_LT['ROIs'])  & set(df_SDI_EP_RT['ROIs'])
#print("Common ROIs EP:")
#print(sorted(common_rois_EP))

plt.show()