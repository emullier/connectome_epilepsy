

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.func_GSP as gsp
from lib.func_plot import plot_rois, plot_rois_pyvista, plot_rois_pyvista_superior
import scipy
import seaborn as sns



tracking = "tracking1" # "probabilistic
ls_lateralization = ["RT", "LT"]

for l, lateralization in enumerate(ls_lateralization):

    data_path = "DATA/matMetric_%s.npy"%(tracking)
    example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"
    infoGVA_path = './DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv'
    scale = 2

    ### Generate the structural harmonics
    #########################################

    print('###################')
    print('Consensus Matrices')

    consensus = np.load(data_path)
    consensus = consensus.mean(axis=0)
    EucDist = np.load("DATA/EucMat_HC_dsi_number_of_fibers.npy")

    print("Generate harmonics from the consensus")
    ### Generate the harmonics
    P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
    np.save('./OUTPUT/DIFF_PROC/Q_ind_%s_%s.npy'%(tracking, lateralization), Q_ind)
    np.save('./OUTPUT/DIFF_PROC/P_ind_%s_%s.npy'%(tracking, lateralization), P_ind)

    ### Project the functional signals
    ########################################
    print("Load EEG example data for SDI")
    X_RS_allPat = gsp.load_EEG_example(example_dir)

    ### Estimate SDI
    ls_cutoff = []
    SDI_tmp = np.zeros((118, len(X_RS_allPat)))
    ls_lat = []; SDI={}; SDI_surr={}
    cutoff_path= './OUTPUT/DIFF_PROC/cutoff_%s.npy'%(tracking)
    for p in np.arange(len(X_RS_allPat)):
        X_RS = X_RS_allPat[p]['X_RS']
        ls_lat.append(X_RS_allPat[p]['lat'][0])
        PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, X_RS)
        ls_cutoff.append(NN)
        SDI_tmp[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind)
    np.save(cutoff_path, ls_cutoff)
    ls_lat = np.array(ls_lat)
    SDI = SDI_tmp
    if lateralization=='RT':
        idxs_lat = np.where(ls_lat=='Rtle')[0]
    elif lateralization=='LT':
        idxs_lat = np.where(ls_lat=='Ltle')[0]
 
    SDI = SDI[:, idxs_lat]
    SDI_path = './OUTPUT/DIFF_PROC/SDI_%s_%s.npy'%(tracking,lateralization)
    np.save(SDI_path, SDI)

    plot_rois_pyvista(np.mean(SDI,axis=1), scale, './FIGURES/DIFF_PROC/', vmin=-1, vmax=1, label='SDImean_%s'%(tracking))

    ### Surrogate part
    nbSurr = 100
    surr_path = './OUTPUT/DIFF_PROC/SDI_surr_%s_%s.npy'%(tracking, lateralization)
    if not os.path.exists(surr_path):
        SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate 
        np.save(surr_path, SDI_surr) # Save the surrogate
    else:   
        SDI_surr = np.load(surr_path)
        print('Surrogate SDI already generated')

    idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
    surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
    surr_thresh_path = './OUTPUT/DIFF_PROC/SDI_surr_thresh_%s_%s.npy'%(tracking, lateralization)
    surr_sig_subjectwise_path = './OUTPUT/DIFF_PROC/SDI_sig_subjectwise_%s_%s.npy'%(tracking, lateralization)
    np.save(surr_thresh_path, surr_thresh, allow_pickle=True) # Save the surrogate
    np.save(surr_sig_subjectwise_path, SDI_sig_subjectwise, allow_pickle=True) # Save the surrogate

    nbROIs_sig = []
    for p in np.arange(np.shape(surr_thresh)[0]):
        nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
    np.save('./OUTPUT/DIFF_PROC/nbROIs_sig_%s_%s.npy'%(tracking, lateralization), nbROIs_sig)

    thr = 2
    plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'], scale, './FIGURES/DIFF_PROC/', vmin=-1, vmax=1, label='SDImean_thr%d_%s_%s'%(thr, tracking, lateralization))



### Plot the consensus connectome
########################################
consensus_HC = np.load("DATA/matMetric_HC_DSI_number_of_fibers.npy")
consensus_HC = np.mean(consensus_HC, axis=2)
consensus = np.load("DATA/matMetric_%s.npy"%(tracking))
consensus_ind = consensus.mean(axis=0)

cons_HC_vec = consensus_HC.flatten()
cons_ind_vec = consensus_ind.flatten()

idxs = np.where((cons_HC_vec>0)*(cons_ind_vec>0))[0]
fig, axs = plt.subplots(1,3, figsize=(10,10), constrained_layout=True)
axs[0].imshow(consensus_HC); axs[0].set_title('Consensus HC DSI GVA \n #streamlines %d'% np.sum(consensus_HC))
axs[1].imshow(consensus_ind); axs[1].set_title('Consensus %s \n #streamlines %d'% (tracking, np.sum(consensus_ind)))
axs[2].scatter(cons_HC_vec[idxs], cons_ind_vec[idxs]); axs[2].set_xlabel('HC'); axs[2].set_ylabel('%s'% (tracking))



### Plot the cutoff frequencies
######################################
cutoff_HC = np.load('./OUTPUT/EPvsCTRL/cutoff_HC_dsi.npy')
cutoff_EP = np.load('./OUTPUT/DIFF_PROC/cutoff_%s.npy'%(tracking))

stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_EP, alternative='two-sided')
print(f"HC - EP \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
r_value, p_value = scipy.stats.pearsonr(cutoff_HC, cutoff_EP)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(cutoff_HC, cutoff_EP, c='tab:blue', alpha=0.6, edgecolors='w', s=60)
ax[0].set_title(f"Pearson r = {r_value:.3f}, p = {p_value:.3f}", fontsize=10)
ax[0].set_xlabel('Cutoff frequency HC'); ax[0].set_ylabel('Cutoff frequency Ind.'); ax[0].grid(True)
sns.boxplot(data=[cutoff_HC, cutoff_EP], ax=ax[1], width=0.5, palette='colorblind')
sns.stripplot(data=[cutoff_HC, cutoff_EP], ax=ax[1], color='black', size=4, jitter=True, dodge=True)
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['HC Consensus', 'Ind. Consensus'])
ax[1].set_ylabel('Cutoff frequency')
ax[1].set_title(f'Mann-Whitney U U = {stat:.3f}, p = {p_mwu:.3g}', fontsize=10)
ax[1].grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()


### Plot the cutoff frequencies
#######################################
nbROIs_sig_RT = np.load('./OUTPUT/DIFF_PROC/nbROIs_sig_%s_RT.npy'%tracking, allow_pickle=True)
nbROIs_sig_LT = np.load('./OUTPUT/DIFF_PROC/nbROIs_sig_%s_LT.npy'%tracking, allow_pickle=True)
surr_thresh_RT = np.load('./OUTPUT/DIFF_PROC/SDI_surr_thresh_%s_RT.npy'%tracking, allow_pickle=True)
surr_thresh_LT = np.load('./OUTPUT/DIFF_PROC/SDI_surr_thresh_%s_LT.npy'%tracking, allow_pickle=True)
fig, ax = plt.subplots(1,1)
ax.plot(np.arange(np.shape(surr_thresh_RT)[0]), np.array(nbROIs_sig_RT), label='RT')
for i, y_value in enumerate(nbROIs_sig_RT):
    ax.scatter(i, y_value, color='k', marker='x', s=20)  # Cross marker
    ax.text(i, y_value, f'{y_value}', fontsize=8, ha='left', va='bottom', color='k')
#ax.axvline(x=0.75*np.shape(surr_thresh)[0], color='r', linestyle='--', linewidth=2, label='75% of participants')
ax.plot(np.arange(np.shape(surr_thresh_LT)[0]), np.array(nbROIs_sig_LT), label='LT')
for i, y_value in enumerate(nbROIs_sig_LT):
    ax.scatter(i, y_value, color='k', marker='x', s=20)  # Cross marker
    ax.text(i, y_value, f'{y_value}', fontsize=8, ha='left', va='bottom', color='k')
ax.set_xlabel('Threshold #Subs'); ax.set_ylabel('#ROIs with significant SDI')
ax.grid('on', alpha=.2); ax.legend()
ax.set_title('Number of significant ROIs for each threshold')
plt.show()

### Plot the harmonics
##############################
Qind_HC_RT = np.load('./OUTPUT/EPvsCTRL/Q_ind_HC_dsi_RT.npy')
Qind_HC_LT = np.load('./OUTPUT/EPvsCTRL/Q_ind_HC_dsi_LT.npy')
Qind_EP_RT = np.load('./OUTPUT/DIFF_PROC/Q_ind_%s_RT.npy'%tracking)
Qind_EP_LT = np.load('./OUTPUT/DIFF_PROC/Q_ind_%s_LT.npy'%tracking)

pearson_corrs_LT = np.zeros(Qind_EP_LT.shape[1])
pearson_corrs_RT = np.zeros(Qind_EP_RT.shape[1])
for i in range(Qind_EP_LT.shape[1]):
    # Calculate Pearson correlation between the i-th column of A and B
    corr_LT, _ = scipy.stats.pearsonr(Qind_EP_LT[:,i], Qind_HC_LT[:, i])
    corr_RT, _ = scipy.stats.pearsonr(Qind_EP_RT[:, i], Qind_HC_RT[:, i])
    pearson_corrs_LT[i] = np.abs(corr_LT)
    pearson_corrs_RT[i] = np.abs(corr_RT)
fig, axs = plt.subplots(1,1,figsize=(30, 10), constrained_layout=True)
axs.plot(range(1, len(pearson_corrs_LT) + 1), pearson_corrs_LT, linestyle='-', color='b')
axs.plot(range(1, len(pearson_corrs_RT) + 1), pearson_corrs_RT, linestyle='-', color='r')
axs.set_title("Pearson Correlation between Corresponding Eigenvectors ")
axs.set_xlabel("Eigenvector Index"); axs.set_ylabel("Pearson Correlation Coefficient")
axs.legend(["LTLE", "RTLE"]); axs.grid(True)


### Plot the SDI
#######################
SDI_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_HC_dsi_LT.npy')
SDI_EP_LT = np.load('./OUTPUT/DIFF_PROC/SDI_%s_LT.npy'%tracking)
SDI_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_HC_dsi_RT.npy')
SDI_EP_RT = np.load('./OUTPUT/DIFF_PROC/SDI_%s_RT.npy'%tracking)
surr_thresh_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_HC_dsi_LT.npy', allow_pickle=True)
surr_thresh_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_HC_dsi_RT.npy', allow_pickle=True)
surr_thresh_EP_RT = np.load('./OUTPUT/DIFF_PROC/SDI_surr_thresh_%s_RT.npy'%tracking, allow_pickle=True)
surr_thresh_EP_LT = np.load('./OUTPUT/DIFF_PROC/SDI_surr_thresh_%s_LT.npy'%tracking, allow_pickle=True)
SDI_sig_subjectwise_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_HC_dsi_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_LT = np.load('./OUTPUT/DIFF_PROC/SDI_sig_subjectwise_%s_LT.npy'%tracking, allow_pickle=True)
SDI_sig_subjectwise_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_HC_dsi_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_RT = np.load('./OUTPUT/DIFF_PROC/SDI_sig_subjectwise_%s_RT.npy'%tracking, allow_pickle=True)

print((np.where(surr_thresh_HC_LT[5]['SDI_sig']!=0)[0])) # 6 out of 8 patients
print((np.where(surr_thresh_HC_RT[5]['SDI_sig']!=0)[0])) # 7 out of 9 patients
df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
labels_118 = df_118['Label Lausanne2008']
labels_118 = np.array(labels_118)


print(labels_118[np.where(surr_thresh_HC_LT[4]['SDI_sig']!=0)[0]])
print(labels_118[np.where(surr_thresh_HC_RT[5]['SDI_sig']!=0)[0]])

#print(len(np.where(SDI_sig_subjectwise_HC_LT[:,6]!=0)[0]))

nROIs = 118
mean_SDI_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_HC_LT = np.copy(mean_SDI_HC_RT); mean_SDI_EP_LT = np.copy(mean_SDI_HC_RT); mean_SDI_EP_RT = np.copy(mean_SDI_HC_RT)
SDI_sig_HC_LT = np.copy(SDI_sig_HC_RT); SDI_sig_EP_LT = np.copy(SDI_sig_HC_RT); SDI_sig_EP_RT = np.copy(SDI_sig_HC_RT) 
for s in np.arange(np.shape(surr_thresh_HC_RT)[0]):
    th = surr_thresh_HC_RT[s]['threshold']
    mean_SDI_HC_RT[s,:] = surr_thresh_HC_RT[s]['mean_SDI']
    SDI_sig_HC_RT[s,:] = surr_thresh_HC_RT[s]['SDI_sig']
    mean_SDI_EP_RT[s,:] = surr_thresh_EP_RT[s]['mean_SDI']
    SDI_sig_EP_RT[s,:] = surr_thresh_EP_RT[s]['SDI_sig']
for s in np.arange(np.shape(surr_thresh_HC_LT)[0]):
    th = surr_thresh_HC_LT[s]['threshold']
    mean_SDI_HC_LT[s,:] = surr_thresh_HC_LT[s]['mean_SDI']
    SDI_sig_HC_LT[s,:] = surr_thresh_HC_LT[s]['SDI_sig']
    mean_SDI_EP_LT[s,:] = surr_thresh_EP_LT[s]['mean_SDI']
    SDI_sig_EP_LT[s,:] = surr_thresh_EP_LT[s]['SDI_sig']

fig, axs = plt.subplots(1, 2, figsize=(20,10), constrained_layout=True) 
axs[0].scatter(mean_SDI_HC_RT[0,:], mean_SDI_EP_RT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_RT[s,:], mean_SDI_EP_RT[s,:])
axs[0].set_title(f"RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0].set_xlabel('Mean SDI HC SC'); axs[0].set_ylabel('Mean SDI RTLE SC')
axs[1].scatter(mean_SDI_HC_LT[0,:], mean_SDI_EP_LT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_LT[s,:], mean_SDI_EP_LT[s,:])
axs[1].set_title(f"LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[1].set_xlabel('Mean SDI SC'); axs[1].set_ylabel('Mean SDI LTLE SC')

nbROIs_EP_RT = np.load('./OUTPUT/DIFF_PROC/nbROIs_sig_%s_RT.npy'%tracking)
nbROIs_EP_LT = np.load('./OUTPUT/DIFF_PROC/nbROIs_sig_%s_LT.npy'%tracking)
nbROIs_HC_RT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_HC_dsi_RT.npy')
nbROIs_HC_LT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_HC_dsi_LT.npy')

fig, ax = plt.subplots(1,1)
ls_nbROIs_sig = [nbROIs_HC_RT, nbROIs_HC_LT, nbROIs_EP_RT, nbROIs_EP_LT]
ls_surr_thresh = [surr_thresh_HC_RT, surr_thresh_HC_LT, surr_thresh_EP_RT, surr_thresh_EP_LT]
ls_labels = ["HC RT", "HC LT", "Ind. RT", "Ind. LT"]
ls_labels = np.array(ls_labels)

for t in np.arange(len(ls_labels)):
    surr_thresh = ls_surr_thresh[t]
    nbROIs_sig = ls_nbROIs_sig[t]
    ax.plot(np.arange(np.shape(surr_thresh)[0]), np.array(nbROIs_sig), marker='x', linewidth=2)
    for i, y_value in enumerate(nbROIs_sig):
        ax.text(i, y_value, f'{y_value}', fontsize=8, ha='left', va='bottom', color='k')
    ax.set_xlabel('Number of EEG participants'); ax.set_ylabel('#ROIs with significant SDI')
    ax.set_xticks(np.arange(0, np.shape(surr_thresh)[0]+1))
    ax.grid('on', alpha=.2)
ax.legend(ls_labels, loc='upper right')
ax.set_title('Number of significant SDI ROIs') 






plt.show()
