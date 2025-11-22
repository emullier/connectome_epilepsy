


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.func_GSP as gsp
from lib.func_plot import plot_rois, plot_rois_pyvista
import scipy.io as sio
import scipy
import seaborn as sns


ls_groups = ["last10SCHZ", "first10SCHZ", "27SCHZ"]
ls_lateralization = ["RT", "LT"]
                     
for g,suff in enumerate(ls_groups):
    for l, lateralization in enumerate(ls_lateralization):

        #data_path = "DATA/Connectome_scale-2.mat"
        example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"
        infoGVA_path = './DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv'
        scale = 2

        ### Generate the structural harmonics
        #########################################
        ### Load the data
        matMetric = np.load("DATA/matMetric_SCHZ_CTRL.npy")
        if suff == "last10SCHZ":
            consensus = np.mean(matMetric[-10:-1,:,:], axis=0)
            np.save('./OUTPUT/IND10/consensus_last10SCHZ.npy', consensus)
        elif suff == "first10SCHZ":
            consensus = np.mean(matMetric[:10,:,:], axis=0)
            np.save('./OUTPUT/IND10/consensus_first10SCHZ.npy', consensus)
        elif suff == "27SCHZ":
            consensus = np.mean(matMetric, axis=0)
            np.save('./OUTPUT/IND10/consensus_27SCHZ.npy', consensus)
        ### which one is used in Isotta paper
        EucDist = np.load("DATA/EucMat_HC_DSI_number_of_fibers.npy")

        print("Generate harmonics from the consensus")
        ### Generate the harmonics
        P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
        np.save('./OUTPUT/IND10/Q_ind_%s_%s.npy'%(suff, lateralization), Q_ind)
        np.save('./OUTPUT/IND10/P_ind_%s_%s.npy'%(suff, lateralization), P_ind)

        ### Project the functional signals
        ########################################
        print("Load EEG example data for SDI")
        X_RS_allPat = gsp.load_EEG_example(example_dir)

        ### Estimate SDI
        ls_cutoff = []
        SDI_tmp = np.zeros((118, len(X_RS_allPat)))
        ls_lat = []; SDI={}; SDI_surr={}
        cutoff_path= './OUTPUT/IND10/cutoff_%s.npy'%(suff)
        for p in np.arange(len(X_RS_allPat)):
            X_RS = X_RS_allPat[p]['X_RS']
            ls_lat.append(X_RS_allPat[p]['lat'][0])
            #idx_ctx = np.concatenate((np.arange(0,57), np.arange(59,116)))
            #PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, X_RS[idx_ctx,:,:])
            PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, X_RS)
            ls_cutoff.append(NN)
            ### Function to have the cutoff frequency from Sipes paper to be added as well
            #SDI_tmp[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS[idx_ctx,:,:], Q_ind)
            SDI_tmp[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind)
        np.save(cutoff_path, ls_cutoff)
        ls_lat = np.array(ls_lat)
        SDI = SDI_tmp
        if lateralization=='RT':
            idxs_lat = np.where(ls_lat=='Rtle')[0]
        elif lateralization=='LT':
            idxs_lat = np.where(ls_lat=='Ltle')[0]
 
        SDI = SDI[:, idxs_lat]
        SDI_path = './OUTPUT/IND10/SDI_%s_%s.npy'%(suff,lateralization)                                                                                                
        np.save(SDI_path, SDI)

        plot_rois_pyvista(np.mean(SDI,axis=1), scale, './FIGURES/IND10', vmin=-1, vmax=1, label='SDImean_%s_%s'%(suff, lateralization))

        ### Surrogate part
        nbSurr = 100
        surr_path = './OUTPUT/IND10/SDI_surr_%s_%s.npy'%(suff, lateralization)
        if not os.path.exists(surr_path):
            SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=True) # Generate the surrogate 
            np.save(surr_path, SDI_surr) # Save the surrogate
        else:   
            SDI_surr = np.load(surr_path)
            print('Surrogate SDI already generated')

        idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
        surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
        surr_thresh_path = './OUTPUT/IND10/SDI_surr_thresh_%s_%s.npy'%(suff, lateralization)
        surr_sig_subjectwise_path = './OUTPUT/IND10/SDI_sig_subjectwise_%s_%s.npy'%(suff, lateralization)
        np.save(surr_thresh_path, surr_thresh, allow_pickle=True) # Save the surrogate
        np.save(surr_sig_subjectwise_path, SDI_sig_subjectwise, allow_pickle=True) # Save the surrogate

        nbROIs_sig = []
        for p in np.arange(np.shape(surr_thresh)[0]):
            nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
        np.save('./OUTPUT/IND10/nbROIs_sig_%s_%s.npy'%(suff, lateralization), nbROIs_sig)




### Plot the consensus connectomes
#####################################
df_info_orig = pd.read_csv(infoGVA_path)
idxs2keep = np.where((df_info_orig['Inclusion']==1)*(df_info_orig['group']=='EP')*(df_info_orig['dwi']=='dsi'))[0]
df_info= df_info_orig.iloc[idxs2keep]

consensus_HC = np.load("DATA/matMetric_HC_DSI_number_of_fibers.npy")
consensus_HC = np.mean(consensus_HC, axis=2)
idxs_RT = np.where(df_info['Lateralization']=="RT")[0]
idxs_LT = np.where(df_info['Lateralization']=="LT")[0]
consensus_first10SCHZ = np.load('./OUTPUT/IND10/consensus_first10SCHZ.npy')
consensus_last10SCHZ = np.load('./OUTPUT/IND10/consensus_last10SCHZ.npy')
consensus_27SCHZ = np.load('./OUTPUT/IND10/consensus_27SCHZ.npy')

cons_HC_vec = consensus_HC.flatten()
consensus_first10SCHZ_vec = consensus_first10SCHZ.flatten()
consensus_last10SCHZ_vec = consensus_last10SCHZ.flatten()
consensus_27SCHZ_vec = consensus_27SCHZ.flatten()

fig, axs = plt.subplots(2,4, figsize=(10,10), constrained_layout=True)
axs[0,0].imshow(consensus_HC); axs[0,0].set_title('Consensus HC DSI GVA')
axs[0,1].imshow(consensus_first10SCHZ); axs[0,1].set_title('Consensus first 10 SCHZ')
axs[0,2].imshow(consensus_last10SCHZ); axs[0,2].set_title('Consensus last 10 SCHZ')
axs[0,3].imshow(consensus_27SCHZ); axs[0,3].set_title('Consensus 27 SCHZ')
axs[1,0].scatter(cons_HC_vec, consensus_first10SCHZ); axs[1,0].set_xlabel('HC'); axs[1,0].set_ylabel('First 10 SCHZ')
axs[1,1].scatter(cons_HC_vec, consensus_last10SCHZ); axs[1,1].set_xlabel('HC'); axs[1,1].set_ylabel('Last 10 SCHZ')
axs[1,2].scatter(cons_HC_vec, consensus_27SCHZ); axs[1,2].set_xlabel('HC'); axs[1,0].set_ylabel('27 SCHZ')


### Plot the cutoff frequencies
######################################
cutoff_HC = np.load('./OUTPUT/EPvsCTRL/cutoff_HC_dsi.npy')
cutoff_first10SCHZ= np.load('./OUTPUT/IND10/cutoff_first10SCHZ.npy')
cutoff_last10SCHZ = np.load('./OUTPUT/IND10/cutoff_last10SCHZ.npy')
cutoff_27SCHZ = np.load('./OUTPUT/IND10/cutoff_27SCHZ.npy')

stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_first10SCHZ, alternative='two-sided')
print(f"HC - EP \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
r_value, p_value = scipy.stats.pearsonr(cutoff_HC, cutoff_first10SCHZ)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#ax[0].scatter(cutoff_HC, cutoff_first10SCHZ, c='tab:blue', alpha=0.6, edgecolors='w', s=60)
#ax[0].set_title(f"Pearson r = {r_value:.3f}, p = {p_value:.3f}", fontsize=10)
#ax[0].set_xlabel('Cutoff frequency HC'); ax[0].set_ylabel('Cutoff frequency first 10 SCHZ'); ax[0].grid(True)
sns.boxplot(data=[cutoff_HC, cutoff_first10SCHZ, cutoff_last10SCHZ, cutoff_27SCHZ], ax=ax[0], width=0.5, palette='colorblind')
sns.stripplot(data=[cutoff_HC, cutoff_first10SCHZ, cutoff_last10SCHZ, cutoff_27SCHZ], ax=ax[0], color='black', size=4, jitter=True, dodge=True)
ax[0].set_xticks([0, 1, 2, 3])
ax[0].set_xticklabels(['HC Consensus', 'first 10 SCHZ', 'last 10 SCHZ', '27 SCHZ'])
ax[0].set_ylabel('Cutoff frequency')
ax[0].set_title(f'Mann-Whitney U U = {stat:.3f}, p = {p_mwu:.3g}', fontsize=10)
ax[0].grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()


### Plot the harmonics
##############################
Qind_HC_RT = np.load('./OUTPUT/EPvsCTRL/Q_ind_HC_dsi_RT.npy')
Qind_HC_LT = np.load('./OUTPUT/EPvsCTRL/Q_ind_HC_dsi_LT.npy')
Qind_first10SCHZ_RT = np.load('./OUTPUT/IND10/Q_ind_first10SCHZ_RT.npy')
Qind_first10SCHZ_LT = np.load('./OUTPUT/IND10/Q_ind_first10SCHZ_LT.npy')
Qind_last10SCHZ_RT = np.load('./OUTPUT/IND10/Q_ind_last10SCHZ_RT.npy')
Qind_last10SCHZ_LT = np.load('./OUTPUT/IND10/Q_ind_last10SCHZ_LT.npy')
Qind_27SCHZ_RT = np.load('./OUTPUT/IND10/Q_ind_27SCHZ_RT.npy')
Qind_27SCHZ_LT = np.load('./OUTPUT/IND10/Q_ind_27SCHZ_LT.npy')

pearson_corrs_first10SCHZ_LT = np.zeros(Qind_first10SCHZ_LT.shape[1]); pearson_corrs_first10SCHZ_RT = np.zeros(Qind_first10SCHZ_RT.shape[1])
pearson_corrs_last10SCHZ_LT = np.zeros(Qind_first10SCHZ_LT.shape[1]); pearson_corrs_last10SCHZ_RT = np.zeros(Qind_first10SCHZ_RT.shape[1])
pearson_corrs_27SCHZ_LT = np.zeros(Qind_first10SCHZ_LT.shape[1]); pearson_corrs_27SCHZ_RT = np.zeros(Qind_first10SCHZ_RT.shape[1])
pearson_first10_last10_LT = np.zeros((Qind_first10SCHZ_LT.shape[1], 2)); pearson_first10_last10_RT = np.zeros((Qind_first10SCHZ_RT.shape[1], 2));
pearson_first10_27schz_LT = np.zeros((Qind_first10SCHZ_LT.shape[1], 2)); pearson_first10_27schz_RT = np.zeros((Qind_first10SCHZ_RT.shape[1], 2));
pearson_last10_27schz_LT = np.zeros((Qind_first10SCHZ_LT.shape[1], 2)); pearson_last10_27schz_RT = np.zeros((Qind_first10SCHZ_RT.shape[1], 2));
for i in range(Qind_first10SCHZ_LT.shape[1]):
    # Calculate Pearson correlation between the i-th column of A and B
    corr_first10SCHZ_LT, _ = scipy.stats.pearsonr(Qind_first10SCHZ_LT[:,i], Qind_HC_LT[:, i])
    corr_first10SCHZ_RT, _ = scipy.stats.pearsonr(Qind_first10SCHZ_RT[:, i], Qind_HC_RT[:, i])
    corr_last10SCHZ_LT, _ = scipy.stats.pearsonr(Qind_last10SCHZ_LT[:,i], Qind_HC_LT[:, i])
    corr_last10SCHZ_RT, _ = scipy.stats.pearsonr(Qind_last10SCHZ_RT[:, i], Qind_HC_RT[:, i])
    corr_27SCHZ_LT, _ = scipy.stats.pearsonr(Qind_27SCHZ_LT[:,i], Qind_HC_LT[:, i])
    corr_27SCHZ_RT, _ = scipy.stats.pearsonr(Qind_27SCHZ_RT[:, i], Qind_HC_RT[:, i])
    corr_first10_last10_LT, _ = scipy.stats.pearsonr(Qind_first10SCHZ_LT[:,i], Qind_last10SCHZ_LT[:, i])
    corr_first10_last10_RT, _ = scipy.stats.pearsonr(Qind_first10SCHZ_RT[:, i], Qind_last10SCHZ_RT[:, i])
    corr_first10_27schz_LT, _ = scipy.stats.pearsonr(Qind_first10SCHZ_LT[:,i], Qind_27SCHZ_LT[:, i])
    corr_first10_27schz_RT, _ = scipy.stats.pearsonr(Qind_first10SCHZ_RT[:, i], Qind_27SCHZ_RT[:, i])
    corr_last10_27schz_LT, _ = scipy.stats.pearsonr(Qind_last10SCHZ_LT[:,i], Qind_27SCHZ_LT[:, i])
    corr_last10_27schz_RT, _ = scipy.stats.pearsonr(Qind_last10SCHZ_RT[:, i], Qind_27SCHZ_RT[:, i])
    pearson_corrs_first10SCHZ_LT[i] = np.abs(corr_first10SCHZ_LT); pearson_corrs_first10SCHZ_RT[i] = np.abs(corr_first10SCHZ_RT)
    pearson_corrs_last10SCHZ_LT[i] = np.abs(corr_last10SCHZ_LT); pearson_corrs_last10SCHZ_RT[i] = np.abs(corr_last10SCHZ_RT)
    pearson_corrs_27SCHZ_LT[i] = np.abs(corr_27SCHZ_LT); pearson_corrs_27SCHZ_RT[i] = np.abs(corr_27SCHZ_RT)
    pearson_corrs_last10SCHZ_LT[i] = np.abs(corr_last10SCHZ_LT); pearson_corrs_last10SCHZ_RT[i] = np.abs(corr_last10SCHZ_RT)
    pearson_first10_last10_LT[i, 0] = np.abs(corr_first10_last10_LT); pearson_first10_last10_RT[i, 0] = np.abs(corr_first10_last10_RT)
    pearson_first10_27schz_LT[i, 0] = np.abs(corr_first10_27schz_LT); pearson_first10_27schz_RT[i, 0] = np.abs(corr_first10_27schz_RT)
    pearson_last10_27schz_LT[i, 0] = np.abs(corr_last10_27schz_LT); pearson_last10_27schz_RT[i, 0] = np.abs(corr_last10_27schz_RT)
    
    
fig, axs = plt.subplots(1,1,figsize=(30, 10), constrained_layout=True)
axs.plot(range(1, len(pearson_corrs_first10SCHZ_LT) + 1), pearson_corrs_first10SCHZ_LT, linestyle='-', color='r')
#axs.plot(range(1, len(pearson_corrs_first10SCHZ_RT) + 1), pearson_corrs_first10SCHZ_RT, linestyle='-', color='r')
axs.plot(range(1, len(pearson_corrs_last10SCHZ_LT) + 1), pearson_corrs_last10SCHZ_LT, linestyle='--', color='b')
#axs.plot(range(1, len(pearson_corrs_last10SCHZ_RT) + 1), pearson_corrs_last10SCHZ_RT, linestyle='--', color='r')
axs.plot(range(1, len(pearson_corrs_27SCHZ_LT) + 1), pearson_corrs_27SCHZ_LT, linestyle=':', color='g')
#axs.plot(range(1, len(pearson_corrs_27SCHZ_RT) + 1), pearson_corrs_27SCHZ_RT, linestyle=':', color='r')
axs.plot(range(1, len(pearson_first10_last10_LT) + 1), pearson_first10_last10_LT[:, 0], linestyle='-', color='orange')
axs.plot(range(1, len(pearson_first10_27schz_LT) + 1), pearson_first10_27schz_LT[:, 0], linestyle='-', color='purple')
axs.plot(range(1, len(pearson_last10_27schz_LT) + 1), pearson_last10_27schz_LT[:, 0], linestyle='-', color='brown')
axs.set_title("Pearson Correlation between Corresponding Eigenvectors ")
axs.set_xlabel("Eigenvector Index"); axs.set_ylabel("Pearson Correlation Coefficient")
axs.legend(["first 10 SCHZ", "last 10 SCHZ", "27 SCHZ", "first-last", "first-27", "last-27"]); axs.grid(True)


    
### Plot the SDI
#######################
SDI_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_HC_dsi_LT.npy'); SDI_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_HC_dsi_RT.npy')
SDI_first10SCHZ_LT = np.load('./OUTPUT/IND10/SDI_first10SCHZ_LT.npy'); SDI_first10SCHZ_RT = np.load('./OUTPUT/IND10/SDI_first10SCHZ_RT.npy')
SDI_last10SCHZ_LT = np.load('./OUTPUT/IND10/SDI_last10SCHZ_LT.npy'); SDI_last10SCHZ_RT = np.load('./OUTPUT/IND10/SDI_last10SCHZ_RT.npy')
SDI_27SCHZ_LT = np.load('./OUTPUT/IND10/SDI_27SCHZ_LT.npy'); SDI_27SCHZ_RT = np.load('./OUTPUT/IND10/SDI_27SCHZ_RT.npy')
surr_thresh_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_HC_dsi_LT.npy', allow_pickle=True); surr_thresh_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_HC_dsi_RT.npy', allow_pickle=True)
surr_thresh_first10SCHZ_LT = np.load('./OUTPUT/IND10/SDI_surr_thresh_first10SCHZ_LT.npy', allow_pickle=True); surr_thresh_first10SCHZ_RT = np.load('./OUTPUT/IND10/SDI_surr_thresh_first10SCHZ_RT.npy', allow_pickle=True)
surr_thresh_last10SCHZ_LT = np.load('./OUTPUT/IND10/SDI_surr_thresh_last10SCHZ_LT.npy', allow_pickle=True); surr_thresh_last10SCHZ_RT = np.load('./OUTPUT/IND10/SDI_surr_thresh_last10SCHZ_RT.npy', allow_pickle=True)
surr_thresh_27SCHZ_LT = np.load('./OUTPUT/IND10/SDI_surr_thresh_27SCHZ_LT.npy', allow_pickle=True); surr_thresh_27SCHZ_RT = np.load('./OUTPUT/IND10/SDI_surr_thresh_27SCHZ_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_HC_dsi_RT.npy', allow_pickle=True); SDI_sig_subjectwise_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_HC_dsi_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_first10SCHZ_RT = np.load('./OUTPUT/IND10/SDI_sig_subjectwise_first10SCHZ_RT.npy', allow_pickle=True); SDI_sig_subjectwise_first10SCHZ_LT = np.load('./OUTPUT/IND10/SDI_sig_subjectwise_first10SCHZ_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_last10SCHZ_RT = np.load('./OUTPUT/IND10/SDI_sig_subjectwise_last10SCHZ_RT.npy', allow_pickle=True); SDI_sig_subjectwise_last10SCHZ_LT = np.load('./OUTPUT/IND10/SDI_sig_subjectwise_last10SCHZ_LT.npy', allow_pickle=True)  
SDI_sig_subjectwise_27SCHZ_RT = np.load('./OUTPUT/IND10/SDI_sig_subjectwise_27SCHZ_RT.npy', allow_pickle=True); SDI_sig_subjectwise_27SCHZ_LT = np.load('./OUTPUT/IND10/SDI_sig_subjectwise_27SCHZ_LT.npy', allow_pickle=True)


nROIs = 118
mean_SDI_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs)); mean_SDI_HC_LT = np.copy(mean_SDI_HC_RT)
SDI_sig_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs)); SDI_sig_HC_LT = np.copy(SDI_sig_HC_RT)
mean_SDI_first10SCHZ_LT = np.copy(mean_SDI_HC_RT); mean_SDI_first10SCHZ_RT = np.copy(mean_SDI_HC_RT)
SDI_sig_first10SCHZ_LT = np.copy(SDI_sig_HC_RT); SDI_sig_first10SCHZ_RT = np.copy(SDI_sig_HC_RT)
mean_SDI_last10SCHZ_LT = np.copy(mean_SDI_HC_RT); mean_SDI_last10SCHZ_RT = np.copy(mean_SDI_HC_RT)
SDI_sig_last10SCHZ_LT = np.copy(SDI_sig_HC_RT); SDI_sig_last10SCHZ_RT = np.copy(SDI_sig_HC_RT)
mean_SDI_27SCHZ_LT = np.copy(mean_SDI_HC_RT); mean_SDI_27SCHZ_RT = np.copy(mean_SDI_HC_RT)
SDI_sig_27SCHZ_LT = np.copy(SDI_sig_HC_RT); SDI_sig_27SCHZ_RT = np.copy(SDI_sig_HC_RT)
 
for s in np.arange(np.shape(surr_thresh_HC_RT)[0]):
    th = surr_thresh_HC_RT[s]['threshold']
    mean_SDI_HC_RT[s,:] = surr_thresh_HC_RT[s]['mean_SDI']
    SDI_sig_HC_RT[s,:] = surr_thresh_HC_RT[s]['SDI_sig']
    mean_SDI_first10SCHZ_RT[s,:] = surr_thresh_first10SCHZ_RT[s]['mean_SDI']
    SDI_sig_first10SCHZ_RT[s,:] = surr_thresh_first10SCHZ_RT[s]['SDI_sig']
    mean_SDI_last10SCHZ_RT[s,:] = surr_thresh_last10SCHZ_RT[s]['mean_SDI']
    SDI_sig_last10SCHZ_RT[s,:] = surr_thresh_last10SCHZ_RT[s]['SDI_sig']
    mean_SDI_27SCHZ_RT[s,:] = surr_thresh_27SCHZ_RT[s]['mean_SDI']
    SDI_sig_27SCHZ_RT[s,:] = surr_thresh_27SCHZ_RT[s]['SDI_sig']
for s in np.arange(np.shape(surr_thresh_HC_LT)[0]):
    th = surr_thresh_HC_LT[s]['threshold']
    mean_SDI_HC_LT[s,:] = surr_thresh_HC_LT[s]['mean_SDI']
    SDI_sig_HC_LT[s,:] = surr_thresh_HC_LT[s]['SDI_sig']
    mean_SDI_first10SCHZ_LT[s,:] = surr_thresh_first10SCHZ_LT[s]['mean_SDI']
    SDI_sig_first10SCHZ_LT[s,:] = surr_thresh_first10SCHZ_LT[s]['SDI_sig']
    mean_SDI_last10SCHZ_LT[s,:] = surr_thresh_last10SCHZ_LT[s]['mean_SDI']
    SDI_sig_last10SCHZ_LT[s,:] = surr_thresh_last10SCHZ_LT[s]['SDI_sig']
    mean_SDI_27SCHZ_LT[s,:] = surr_thresh_27SCHZ_LT[s]['mean_SDI']
    SDI_sig_27SCHZ_LT[s,:] = surr_thresh_27SCHZ_LT[s]['SDI_sig']
    
fig, axs = plt.subplots(1, 2, figsize=(20,10), constrained_layout=True) 
axs[0].scatter(mean_SDI_HC_RT[0,:], mean_SDI_first10SCHZ_RT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_RT[s,:], mean_SDI_first10SCHZ_RT[s,:])
axs[0].set_title(f"RTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[0].set_xlabel('Mean SDI HC SC'); axs[0].set_ylabel('Mean SDI first10SCHZ RT SC')
axs[1].scatter(mean_SDI_HC_LT[0,:], mean_SDI_first10SCHZ_LT[0,:], c='k', alpha=0.5)
[r,p]= scipy.stats.pearsonr(mean_SDI_HC_LT[s,:], mean_SDI_first10SCHZ_LT[s,:])
axs[1].set_title(f"LTLE \n Correlation between mean SDI \n (r = {r:.2f}, p = {p:.2e}) ", fontsize=10)
axs[1].set_xlabel('Mean SDI SC'); axs[1].set_ylabel('Mean SDI first10SCHZ LT SC')

nbROIs_first10SCHZ_RT = np.load('./OUTPUT/IND10/nbROIs_sig_first10SCHZ_RT.npy')
nbROIs_first10SCHZ_LT = np.load('./OUTPUT/IND10/nbROIs_sig_first10SCHZ_LT.npy')
nbROIs_HC_RT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_HC_dsi_RT.npy')
nbROIs_HC_LT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_HC_dsi_LT.npy')

fig, ax = plt.subplots(1,1)
ls_nbROIs_sig = [nbROIs_HC_RT, nbROIs_HC_LT, nbROIs_first10SCHZ_RT, nbROIs_first10SCHZ_LT]
ls_surr_thresh = [surr_thresh_HC_RT, surr_thresh_HC_LT, surr_thresh_first10SCHZ_RT, surr_thresh_first10SCHZ_LT]
ls_labels = ["HC RT", "HC LT", "first10SCHZ RT", "first10SCHZ LT"]
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


thr = 2
plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'], scale, './FIGURES/IND10', vmin=-1, vmax=1, label='SDImean_thr%d_%s_%s'%(thr, suff, lateralization))

df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
labels_118 = df_118['Label Lausanne2008']
labels_118 = np.array(labels_118)
print(labels_118[np.where(surr_thresh[5]['SDI_sig']!=0)[0]])
print(surr_thresh[5]['mean_SDI'][np.where(surr_thresh[5]['SDI_sig']!=0)[0]])


plt.show()

