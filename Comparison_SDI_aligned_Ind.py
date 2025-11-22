
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

ref = 'HCdsi'

### Comparison Cutoff frequencies
cutoff_HC = np.load('./OUTPUT/EPvsCTRL/cutoff_HC_dsi.npy')
cutoff_EP = np.load('./OUTPUT/INDvsCTRL/cutoff_schz.npy')
cutoff_EP_rotated = np.load('./OUTPUT/ALIGN/cutoff_rotated_schz_ref%s.npy'%ref)
cutoff_EP_ortho_rotated = np.load('./OUTPUT/ALIGN/cutoff_ortho_rotated_schz_ref%s.npy'%ref)
cutoff_EP_matched = np.load('./OUTPUT/ALIGN/cutoff_matched_schz_ref%s.npy'%ref)

stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_EP, alternative='two-sided')
print(f"HC - EP \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
#stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_Iso, alternative='two-sided')
#print(f"HC - Independent \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
#stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_EP, cutoff_Iso, alternative='two-sided')
#print(f"EP - Independent \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
#r_value, p_value = scipy.stats.pearsonr(cutoff_HC, cutoff_EP)
stat, p_kw = kruskal(cutoff_HC, cutoff_EP, cutoff_EP_rotated, cutoff_EP_ortho_rotated, cutoff_EP_matched)
print(f"Kruskal–Wallis H = {stat:.3f}, p = {p_kw:.3g}")

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.boxplot(data=[cutoff_HC, cutoff_EP, cutoff_EP_rotated, cutoff_EP_ortho_rotated, cutoff_EP_matched], ax=ax, width=0.5, palette='colorblind')
sns.stripplot(data=[cutoff_HC, cutoff_EP, cutoff_EP_rotated, cutoff_EP_ortho_rotated, cutoff_EP_matched], ax=ax, color='black', size=4, jitter=True, dodge=True)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['HC Consensus', 'EP Consensus', 'EP rotated', 'EP ortho rotated', 'EP matched'])
ax.set_ylabel('Cutoff frequency')
ax.set_title(f'Kruskal–Wallis H = {stat:.3f}, p = {p_kw:.3g}', fontsize=10)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()


### Comparison SDI for each participant
SDI_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_HC_dsi_RT.npy')
SDI_EP_RT = np.load('./OUTPUT/INDvsCTRL/SDI_schz_RT.npy')
SDI_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_HC_dsi_LT.npy')
SDI_EP_LT = np.load('./OUTPUT/INDvsCTRL/SDI_schz_LT.npy')
SDI_EP_RT_rotated = np.load('./OUTPUT/ALIGN/SDI_rotated_schz_RT_ref%s.npy'%ref)
SDI_EP_RT_ortho_rotated = np.load('./OUTPUT/ALIGN/SDI_ortho_rotated_schz_RT_ref%s.npy'%ref)
SDI_EP_RT_matched = np.load('./OUTPUT/ALIGN/SDI_matched_schz_RT_ref%s.npy'%ref)
SDI_EP_LT_rotated = np.load('./OUTPUT/ALIGN/SDI_rotated_schz_LT_ref%s.npy'%ref)
SDI_EP_LT_ortho_rotated = np.load('./OUTPUT/ALIGN/SDI_ortho_rotated_schz_LT_ref%s.npy'%ref)
SDI_EP_LT_matched = np.load('./OUTPUT/ALIGN/SDI_matched_schz_LT_ref%s.npy'%ref)


surr_thresh_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_HC_dsi_RT.npy', allow_pickle=True)
surr_thresh_EP_RT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_schz_RT.npy', allow_pickle=True)
surr_thresh_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_HC_dsi_LT.npy', allow_pickle=True)
surr_thresh_EP_LT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_schz_LT.npy', allow_pickle=True)
surr_thresh_EP_RT_rotated = np.load('./OUTPUT/ALIGN/SDI_surr_thresh_rotated_schz_RT_ref%s.npy'%ref, allow_pickle=True)
surr_thresh_EP_RT_ortho_rotated = np.load('./OUTPUT/ALIGN/SDI_surr_thresh_ortho_rotated_schz_RT_ref%s.npy'%ref, allow_pickle=True)
surr_thresh_EP_RT_matched = np.load('./OUTPUT/ALIGN/SDI_surr_thresh_matched_schz_RT_ref%s.npy'%ref, allow_pickle=True)
surr_thresh_EP_LT_rotated = np.load('./OUTPUT/ALIGN/SDI_surr_thresh_rotated_schz_LT_ref%s.npy'%ref, allow_pickle=True)
surr_thresh_EP_LT_ortho_rotated = np.load('./OUTPUT/ALIGN/SDI_surr_thresh_ortho_rotated_schz_LT_ref%s.npy'%ref, allow_pickle=True)
surr_thresh_EP_LT_matched = np.load('./OUTPUT/ALIGN/SDI_surr_thresh_matched_schz_LT_ref%s.npy'%ref, allow_pickle=True)
SDI_sig_subjectwise_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_HC_dsi_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_RT = np.load('./OUTPUT/INDvsCTRL/SDI_sig_subjectwise_schz_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_RT_rotated = np.load('./OUTPUT/ALIGN/SDI_sig_subjectwise_rotated_schz_RT_ref%s.npy'%ref, allow_pickle=True)    
SDI_sig_subjectwise_EP_RT_ortho_rotated = np.load('./OUTPUT/ALIGN/SDI_sig_subjectwise_ortho_rotated_schz_RT_ref%s.npy'%ref, allow_pickle=True)
SDI_sig_subjectwise_EP_RT_matched = np.load('./OUTPUT/ALIGN/SDI_sig_subjectwise_matched_schz_RT_ref%s.npy'%ref, allow_pickle=True)
SDI_sig_subjectwise_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_HC_dsi_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_LT = np.load('./OUTPUT/INDvsCTRL/SDI_sig_subjectwise_schz_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_LT_rotated = np.load('./OUTPUT/ALIGN/SDI_sig_subjectwise_rotated_schz_LT_ref%s.npy'%ref, allow_pickle=True)
SDI_sig_subjectwise_EP_LT_ortho_rotated = np.load('./OUTPUT/ALIGN/SDI_sig_subjectwise_ortho_rotated_schz_LT_ref%s.npy'%ref, allow_pickle=True)
SDI_sig_subjectwise_EP_LT_matched = np.load('./OUTPUT/ALIGN/SDI_sig_subjectwise_matched_schz_LT_ref%s.npy'%ref, allow_pickle=True)

nROIs = 118
mean_SDI_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_HC_LT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_HC_LT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_EP_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_EP_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_EP_RT_rotated = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_EP_RT_rotated = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_EP_RT_ortho_rotated = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_EP_RT_ortho_rotated = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_EP_RT_matched = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_EP_RT_matched = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_EP_LT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_EP_LT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_EP_LT_rotated = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_EP_LT_rotated = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_EP_LT_ortho_rotated = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_EP_LT_ortho_rotated = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_EP_LT_matched = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_EP_LT_matched = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))

### Comparing average SDI values LT   
for s in np.arange(np.shape(surr_thresh_HC_RT)[0]):
    th = surr_thresh_HC_RT[s]['threshold']
    mean_SDI_HC_RT[s,:] = surr_thresh_HC_RT[s]['mean_SDI']
    SDI_sig_HC_RT[s,:] = surr_thresh_HC_RT[s]['SDI_sig']
    mean_SDI_EP_RT[s,:] = surr_thresh_EP_RT[s]['mean_SDI']
    SDI_sig_EP_RT[s,:] = surr_thresh_EP_RT[s]['SDI_sig']
    mean_SDI_EP_RT_rotated[s,:] = surr_thresh_EP_RT_rotated[s]['mean_SDI']
    SDI_sig_EP_RT_rotated[s,:] = surr_thresh_EP_RT_rotated[s]['SDI_sig']
    mean_SDI_EP_RT_ortho_rotated[s,:] = surr_thresh_EP_RT_ortho_rotated[s]['mean_SDI']
    SDI_sig_EP_RT_ortho_rotated[s,:] = surr_thresh_EP_RT_ortho_rotated[s]['SDI_sig']
    mean_SDI_EP_RT_matched[s,:] = surr_thresh_EP_RT_matched[s]['mean_SDI']
    SDI_sig_EP_RT_matched[s,:] = surr_thresh_EP_RT_matched[s]['SDI_sig']
for s in np.arange(np.shape(surr_thresh_HC_LT)[0]):
    mean_SDI_HC_LT[s,:] = surr_thresh_HC_LT[s]['mean_SDI']
    SDI_sig_HC_LT[s,:] = surr_thresh_HC_LT[s]['SDI_sig']
    mean_SDI_EP_LT[s,:] = surr_thresh_EP_LT[s]['mean_SDI']
    SDI_sig_EP_LT[s,:] = surr_thresh_EP_LT[s]['SDI_sig']
    mean_SDI_EP_LT_rotated[s,:] = surr_thresh_EP_LT_rotated[s]['mean_SDI']
    SDI_sig_EP_LT_rotated[s,:] = surr_thresh_EP_LT_rotated[s]['SDI_sig']
    mean_SDI_EP_LT_ortho_rotated[s,:] = surr_thresh_EP_LT_ortho_rotated[s]['mean_SDI']
    SDI_sig_EP_LT_ortho_rotated[s,:] = surr_thresh_EP_LT_ortho_rotated[s]['SDI_sig']
    mean_SDI_EP_LT_matched[s,:] = surr_thresh_EP_LT_matched[s]['mean_SDI']
    SDI_sig_EP_LT_matched[s,:] = surr_thresh_EP_LT_matched[s]['SDI_sig']
fig, axs = plt.subplots(2, 4, figsize=(10,10), constrained_layout=True) 
axs[0,0].scatter(mean_SDI_HC_RT[0,:], mean_SDI_EP_RT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_RT[s,:], mean_SDI_EP_RT[s,:])
axs[0,0].set_title(f"RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0,0].set_xlabel('Mean SDI HC SC'); axs[0,0].set_ylabel('Mean SDI RTLE SC')
axs[0,1].scatter(mean_SDI_HC_RT[0,:], mean_SDI_EP_RT_rotated[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_RT[s,:], mean_SDI_EP_RT_rotated[s,:])   
axs[0,1].set_title(f"Rotated RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0,1].set_xlabel('Mean SDI HC SC'); axs[0,1].set_ylabel('Mean SDI Rotated RTLE SC')
axs[0,2].scatter(mean_SDI_HC_RT[0,:], mean_SDI_EP_RT_ortho_rotated[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_RT[s,:], mean_SDI_EP_RT_ortho_rotated[s,:])
axs[0,2].set_title(f"Ortho rotated RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0,2].set_xlabel('Mean SDI HC SC'); axs[0,2].set_ylabel('Mean SDI Ortho rotated RTLE SC')
axs[0,3].scatter(mean_SDI_HC_RT[0,:], mean_SDI_EP_RT_matched[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_RT[s,:], mean_SDI_EP_RT_matched[s,:])
axs[0,3].set_title(f"Matched RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0,3].set_xlabel('Mean SDI HC SC'); axs[0,3].set_ylabel('Mean SDI Matched RTLE SC')
axs[1,0].scatter(mean_SDI_HC_LT[0,:], mean_SDI_EP_LT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_LT[s,:], mean_SDI_EP_LT[s,:])
axs[1,0].set_title(f"LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[1,0].set_xlabel('Mean SDI HC SC'); axs[1,0].set_ylabel('Mean SDI LTLE SC')
axs[1,1].scatter(mean_SDI_HC_LT[0,:], mean_SDI_EP_LT_rotated[0,:], c='k', alpha=0.5)    
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_LT[s,:], mean_SDI_EP_LT_rotated[s,:])
axs[1,1].set_title(f"Rotated LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[1,1].set_xlabel('Mean SDI HC SC'); axs[1,1].set_ylabel('Mean SDI Rotated LTLE SC')
axs[1,2].scatter(mean_SDI_HC_LT[0,:], mean_SDI_EP_LT_ortho_rotated[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_LT[s,:], mean_SDI_EP_LT_ortho_rotated[s,:])
axs[1,2].set_title(f"Ortho rotated LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[1,2].set_xlabel('Mean SDI HC SC'); axs[1,2].set_ylabel('Mean SDI Ortho rotated LTLE SC')
axs[1,3].scatter(mean_SDI_HC_LT[0,:], mean_SDI_EP_LT_matched[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_LT[s,:], mean_SDI_EP_LT_matched[s,:])
axs[1,3].set_title(f"Matched LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[1,3].set_xlabel('Mean SDI HC SC'); axs[1,3].set_ylabel('Mean SDI Matched LTLE SC')



### Comparing harmonics values
Qind_HC_RT = np.load('./OUTPUT/EPvsCTRL/Q_ind_HC_dsi_RT.npy')
Qind_HC_LT = np.load('./OUTPUT/EPvsCTRL/Q_ind_HC_dsi_LT.npy')
Qind_EP_RT = np.load('./OUTPUT/INDvsCTRL/Q_ind_schz_RT.npy')
Qind_EP_RT_rotated = np.load('./OUTPUT/ALIGN/Q_ind_rotated_schz_RT_ref%s.npy'%ref)
Qind_EP_RT_ortho_rotated = np.load('./OUTPUT/ALIGN/Q_ind_ortho_rotated_schz_RT_ref%s.npy'%ref)
Qind_EP_RT_matched = np.load('./OUTPUT/ALIGN/Q_ind_matched_schz_RT_ref%s.npy'%ref)
Qind_EP_LT = np.load('./OUTPUT/INDvsCTRL/Q_ind_schz_LT.npy')
Qind_EP_LT_rotated = np.load('./OUTPUT/ALIGN/Q_ind_rotated_schz_LT_ref%s.npy'%ref)
Qind_EP_LT_ortho_rotated = np.load('./OUTPUT/ALIGN/Q_ind_ortho_rotated_schz_LT_ref%s.npy'%ref)
Qind_EP_LT_matched = np.load('./OUTPUT/ALIGN/Q_ind_matched_schz_LT_ref%s.npy'%ref)

pearson_corrs_RT = np.zeros(Qind_EP_RT.shape[1])
pearson_corrs_RT_rot = np.zeros(Qind_EP_RT.shape[1])
pearson_corrs_RT_orth_rot = np.zeros(Qind_EP_RT.shape[1])
pearson_corrs_RT_matched = np.zeros(Qind_EP_RT.shape[1])
pearson_corrs_LT = np.zeros(Qind_EP_LT.shape[1])
pearson_corrs_LT_rot = np.zeros(Qind_EP_LT.shape[1])
pearson_corrs_LT_orth_rot = np.zeros(Qind_EP_LT.shape[1])
pearson_corrs_LT_matched = np.zeros(Qind_EP_LT.shape[1])

for i in range(Qind_EP_RT.shape[1]):
    # Calculate Pearson correlation between the i-th column of A and B
    corr_RT, _ = scipy.stats.pearsonr(Qind_EP_RT[:, i], Qind_HC_RT[:, i])
    corr_RT_rot, _ = scipy.stats.pearsonr(Qind_EP_RT_rotated[:, i], Qind_HC_RT[:, i])
    corr_RT_orth_rot, _ = scipy.stats.pearsonr(Qind_EP_RT_ortho_rotated[:, i], Qind_HC_RT[:, i])
    corr_RT_matched, _ = scipy.stats.pearsonr(Qind_EP_RT_matched[:, i], Qind_HC_RT[:, i])
    corr_LT, _ = scipy.stats.pearsonr(Qind_EP_LT[:, i], Qind_HC_RT[:, i])
    corr_LT_rot, _ = scipy.stats.pearsonr(Qind_EP_LT_rotated[:, i], Qind_HC_RT[:, i])
    corr_LT_orth_rot, _ = scipy.stats.pearsonr(Qind_EP_LT_ortho_rotated[:, i], Qind_HC_RT[:, i])
    corr_LT_matched, _ = scipy.stats.pearsonr(Qind_EP_LT_matched[:, i], Qind_HC_RT[:, i])
    pearson_corrs_RT[i] = np.abs(corr_RT)
    pearson_corrs_RT_rot[i] = np.abs(corr_RT_rot)
    pearson_corrs_RT_orth_rot[i] = np.abs(corr_RT_orth_rot)
    pearson_corrs_RT_matched[i] = np.abs(corr_RT_matched)
    pearson_corrs_LT[i] = np.abs(corr_LT)
    pearson_corrs_LT_rot[i] = np.abs(corr_LT_rot)
    pearson_corrs_LT_orth_rot[i] = np.abs(corr_LT_orth_rot)
    pearson_corrs_LT_matched[i] = np.abs(corr_LT_matched)

# Plot the Pears
# on correlation coefficients for each pair of columns
fig, axs = plt.subplots(2,1,figsize=(15, 20), constrained_layout=True)
axs[0].plot(range(1, len(pearson_corrs_RT) + 1), pearson_corrs_RT, linestyle='-')
axs[0].plot(range(1, len(pearson_corrs_RT_rot) + 1), pearson_corrs_RT_rot, linestyle='-')
axs[0].plot(range(1, len(pearson_corrs_RT_orth_rot) + 1), pearson_corrs_RT_orth_rot, linestyle='-')
axs[0].plot(range(1, len(pearson_corrs_RT_matched) + 1), pearson_corrs_RT_matched, linestyle='-')
axs[0].set_title("RTLE"); 
axs[0].set_xlabel("Eigenvector Index"); axs[0].set_ylabel("Pearson Correlation Coefficient")
axs[0].legend(["Original", "Gen.Procrustes", "Ortho.Procrusters", "Matched"]); axs[0].grid(True)
axs[1].plot(range(1, len(pearson_corrs_LT) + 1), pearson_corrs_LT, linestyle='-')
axs[1].plot(range(1, len(pearson_corrs_LT_rot) + 1), pearson_corrs_LT_rot, linestyle='-')
axs[1].plot(range(1, len(pearson_corrs_LT_orth_rot) + 1), pearson_corrs_LT_orth_rot, linestyle='-')
axs[1].plot(range(1, len(pearson_corrs_LT_matched) + 1), pearson_corrs_LT_matched, linestyle='-')
axs[1].set_title("RTLE"); 
axs[1].set_xlabel("Eigenvector Index"); axs[1].set_ylabel("Pearson Correlation Coefficient")
axs[1].legend(["Original", "Gen.Procrustes", "Ortho.Procrusters", "Matched"]); axs[1].grid(True)

fig, axs = plt.subplots(3,1,figsize=(15, 20), constrained_layout=True)
axs[0].scatter(pearson_corrs_LT, pearson_corrs_LT_rot, c=np.arange(len(pearson_corrs_LT)), alpha=0.5)
[r,p] = scipy.stats.pearsonr(pearson_corrs_LT, pearson_corrs_LT_rot)
axs[0].set_title(f"LTLE \n Similarities of harmonics \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0].set_xlabel('Original'); axs[0].set_ylabel('Gen.Procrustes')
axs[1].scatter(pearson_corrs_LT, pearson_corrs_LT_orth_rot, c=np.arange(len(pearson_corrs_LT)), alpha=0.5)
[r,p] = scipy.stats.pearsonr(pearson_corrs_LT, pearson_corrs_LT_orth_rot)
axs[1].set_title(f"LTLE \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[1].set_xlabel('Original'); axs[1].set_ylabel('Ortho.Procrusters')
axs[2].scatter(pearson_corrs_LT, pearson_corrs_LT_matched, c=np.arange(len(pearson_corrs_LT)), alpha=0.5)
[r,p] = scipy.stats.pearsonr(pearson_corrs_LT, pearson_corrs_LT_matched)
axs[2].set_title(f"LTLE \n Similarities of harmonics \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[2].set_xlabel('Original'); axs[2].set_ylabel('Matched')


stat, p_kw = kruskal(pearson_corrs_LT, pearson_corrs_LT_rot, pearson_corrs_LT_orth_rot, pearson_corrs_LT_matched)
print(f"Kruskal–Wallis H = {stat:.3f}, p = {p_kw:.3g}")
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.boxplot(data=[pearson_corrs_LT, pearson_corrs_LT_rot, pearson_corrs_LT_orth_rot, pearson_corrs_LT_matched], ax=ax, width=0.5, palette='colorblind')
sns.stripplot(data=[pearson_corrs_LT, pearson_corrs_LT_rot, pearson_corrs_LT_orth_rot, pearson_corrs_LT_matched], ax=ax,  c='k', size=4, jitter=True, dodge=True)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['EP Consensus', 'EP rotated', 'EP ortho rotated', 'EP matched'])
ax.set_ylabel('Cutoff frequency')
ax.set_title(f'Kruskal–Wallis H = {stat:.3f}, p = {p_kw:.3g}', fontsize=10)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()