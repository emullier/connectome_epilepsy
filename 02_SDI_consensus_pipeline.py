


''' This script computes the SDI values on the EEG data of the 17 patients (Rigoni,2023) using the consensus structural connectivity matrices 
generated from the Geneva datasets (HC and EP with RTLE and LTLE). 

Fig 1: Generate consensus structural connectivity matrices from Geneva datasets (HC and EP with RTLE and LTLE)
Fig 2: Compare cutoff frequencies when using consensus SC HC and consensus SC EP
Fig 3: Similarities between structural harmonics using consensus SC HC and consensus SC EP for right and left TLE patients
Fig 4: Correlation between mean SDI using consensus SC HC and consensus SC EP for right and left TLE patients (1 point = 1 ROI)
Fig 5: Number of ROIs with significant SDI depending on the number of subjects included in the analysis for HC and EP with RTLE and LTLE patients
Saved fig: Brain plot of the mean SDI values for different thresholds (2, 5) for HC and EP with RTLE and LTLE patients.

Parameters to modify:
- metric: metric to use to generate the consensus SC ('number_of_fibers', 'normalized_fiber_density', 'fiber_length_mean')
- dwi: DSI or multishell
- group: HC or EP   

Last modified: EM, 20.11.2025 
Created: 11.03.2025, Emeline Mullier
University of Geneva & Lausanne University Hospital '''


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.func_GSP as gsp
from lib.func_plot import plot_rois, plot_rois_pyvista, plot_rois_pyvista_superior
import scipy
import seaborn as sns
from tabulate import tabulate

ls_groups = ["EP", "HC"]
ls_lateralization = ["RT", "LT"]
                     
for g,group in enumerate(ls_groups):
    for l, lateralization in enumerate(ls_lateralization):

        #metric = "normalized_fiber_density" # "number_of_fibers"
        metric = "number_of_fibers" #'fiber_length_mean' #"number_of_fibers", "normalized_fiber_density" # 'shore_gfa_mean'
        dwi = "dsi"
        data_path = "DATA/matMetric_%s_%s_%s.npy"%(group, dwi, metric)
        example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"
        infoGVA_path = './DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv'
        scale = 2

        ### Load info
        ################### 
        df_info_orig = pd.read_csv(infoGVA_path)
        idxs2keep = np.where((df_info_orig['Inclusion']==1)*(df_info_orig['group']==group)*(df_info_orig['dwi']==dwi))[0]
        df_info= df_info_orig.iloc[idxs2keep]

        ### Generate the structural harmonics
        #########################################
        ### Load the data
        matMetric = np.load(data_path)
        if group=='EP':
            idxs = np.where(df_info['Lateralization']==lateralization)[0]
            matMetric = matMetric[:,:,idxs]

        print('###################')
        print('Consensus Matrices')
        #G_dist, G_unif, EucDist = ML.consensus(MatMat, config["Parameters"]["processing"],  dict_df, EucMat, config["CONSENSUS"]["nbins"])
        #G_dist_wei, G_unif_wei = reading.save_consensus(MatMat, config["Parameters"]["metric"], G_dist, G_unif, config["Parameters"]["output_dir"], config["Parameters"]["processing"])
        #np.save('tests/MatMat_main', MatMat[procs[0]])

        consensus = np.mean(matMetric, axis=2)
        #np.fill_diagonal(consensus, 0)
        #EucDist = consensus #### To be replaced by proper Euclidean matrix
        #EucDist = (EucDist + EucDist.T)/2
        EucDist = np.load("DATA/EucMat_%s_%s_%s.npy"%(group, dwi, metric))

        print("Generate harmonics from the consensus")
        ### Generate the harmonics
        P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
        np.save('./OUTPUT/EPvsCTRL/Q_ind_%s_%s_%s_%s.npy'%(metric, group, dwi, lateralization), Q_ind)
        np.save('./OUTPUT/EPvsCTRL/P_ind_%s_%s_%s_%s.npy'%(metric, group, dwi, lateralization), P_ind)
        #### maybe add the reference input to know to which input to realign. 
        ### Create the functions to do all the alignments possible
        ### NOT IN THE CASE OF CONSENSUS
        #Q_all_rotated, P_all_rotated, R_all, scale_R = gsp.rotation_procrustes(Q_ind, P_ind,plot=True)

        ### Project the functional signals
        ########################################
        print("Load EEG example data for SDI")
        X_RS_allPat = gsp.load_EEG_example(example_dir)

        ### Estimate SDI
        ls_cutoff = []
        SDI_tmp = np.zeros((118, len(X_RS_allPat)))
        ls_lat = []; SDI={}; SDI_surr={}
        cutoff_path= './OUTPUT/EPvsCTRL/cutoff_%s_%s_%s.npy'%(metric, group, dwi)
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
        SDI_path = './OUTPUT/EPvsCTRL/SDI_%s_%s_%s_%s.npy'%(metric, group, dwi, lateralization)
        np.save(SDI_path, SDI)

        #vmin = np.min(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'])
        #vmax = np.max(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'])
        vmin=-2; vmax=2
        #plot_rois_pyvista(np.mean(SDI,axis=1), scale, './FIGURES/EPvsCTRL', vmin=vmin, vmax=vmax, label='SDImean__%s_%s_%s_%s'%(metric, group, dwi, lateralization))

        ### Surrogate part
        nbSurr = 100
        surr_path = './OUTPUT/EPvsCTRL/SDI_surr_%s_%s_%s_%s.npy'%(metric, group, dwi, lateralization)
        if not os.path.exists(surr_path):
            SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate 
            np.save(surr_path, SDI_surr) # Save the surrogate
        else:   
            SDI_surr = np.load(surr_path)
            print('Surrogate SDI already generated')

        idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
        surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
        surr_thresh_path = './OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_%s_%s_%s.npy'%(metric, group, dwi, lateralization)
        surr_sig_subjectwise_path = './OUTPUT/EPvsCTRL/SDI_sig_subjectwise_%s_%s_%s_%s.npy'%(metric, group, dwi, lateralization)
        np.save(surr_thresh_path, surr_thresh, allow_pickle=True) # Save the surrogate
        np.save(surr_sig_subjectwise_path, SDI_sig_subjectwise, allow_pickle=True) # Save the surrogate

        nbROIs_sig = []
        for p in np.arange(np.shape(surr_thresh)[0]):
            nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
        np.save('./OUTPUT/EPvsCTRL/nbROIs_sig_%s_%s_%s_%s.npy'%(metric, group, dwi, lateralization), nbROIs_sig)


        thr = 2
        #vmin = np.min(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'])
        #vmax = np.max(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'])
        vmin=-2; vmax=2
        #plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'], scale, './FIGURES/EPvsCTRL', vmin=vmin, vmax=vmax, label='SDImean_thr%d_%s_%s_%s_%s'%(thr, metric, group, dwi, lateralization))

        thr = 5
        #vmin = np.min(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'])
        #vmax = np.max(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'])
        vmin=-2; vmax=2
        plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'], scale, './FIGURES/EPvsCTRL', vmin=vmin, vmax=vmax, label='SDImean_thr%d_%s_%s_%s_%s'%(thr, metric, group, dwi, lateralization))



### Plot the consensus connectomes
#####################################
df_info_orig = pd.read_csv(infoGVA_path)
idxs2keep = np.where((df_info_orig['Inclusion']==1)*(df_info_orig['group']=='EP')*(df_info_orig['dwi']=='dsi'))[0]
df_info= df_info_orig.iloc[idxs2keep]

consensus_HC = np.load("DATA/matMetric_HC_DSI_number_of_fibers.npy")
matMetric = np.load("DATA/matMetric_EP_DSI_number_of_fibers.npy")
idxs_RT = np.where(df_info['Lateralization']=="RT")[0]
idxs_LT = np.where(df_info['Lateralization']=="LT")[0]
consensus_HC = np.mean(consensus_HC, axis=2)
consensus_EP_RT = np.mean(matMetric[:,:,idxs_RT], axis=2)
consensus_EP_LT = np.mean(matMetric[:,:,idxs_LT], axis=2)

cons_HC_vec = consensus_HC.flatten()
cons_EP_RT_vec = consensus_EP_RT.flatten()
cons_EP_LT_vec = consensus_EP_LT.flatten()

fig, axs = plt.subplots(2,3, figsize=(10,10), constrained_layout=True)
axs[0,0].imshow(consensus_HC); axs[0,0].set_title('Consensus HC DSI GVA')
axs[0,1].imshow(consensus_EP_RT); axs[0,1].set_title('Consensus EP RT')
axs[0,2].imshow(consensus_EP_LT); axs[0,2].set_title('Consensus EP LT')
axs[1,0].scatter(cons_HC_vec, cons_EP_RT_vec); axs[1,0].set_xlabel('HC'); axs[1,0].set_ylabel('EP RT')
axs[1,1].scatter(cons_HC_vec, cons_EP_LT_vec); axs[1,1].set_xlabel('HC'); axs[1,1].set_ylabel('EP LT')
axs[1,2].scatter(cons_EP_RT_vec, cons_EP_LT_vec); axs[1,2].set_xlabel('EP RT'); axs[1,0].set_ylabel('EP LT')


### Plot the cutoff frequencies
######################################
cutoff_HC = np.load('./OUTPUT/EPvsCTRL/cutoff_%s_HC_%s.npy'%(metric, dwi))
cutoff_EP = np.load('./OUTPUT/EPvsCTRL/cutoff_%s_EP_%s.npy'%(metric,dwi))

stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_EP, alternative='two-sided')
print(f"HC - EP \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
r_value, p_value = scipy.stats.pearsonr(cutoff_HC, cutoff_EP)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(cutoff_HC, cutoff_EP, c='tab:blue', alpha=0.6, edgecolors='w', s=60)
ax[0].set_title(f"Pearson r = {r_value:.3f}, p = {p_value:.3f}", fontsize=10)
ax[0].set_xlabel('Cutoff frequency HC'); ax[0].set_ylabel('Cutoff frequency EP'); ax[0].grid(True)
sns.boxplot(data=[cutoff_HC, cutoff_EP], ax=ax[1], width=0.5, palette='colorblind')
sns.stripplot(data=[cutoff_HC, cutoff_EP], ax=ax[1], color='black', size=4, jitter=True, dodge=True)
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['HC Consensus', 'EP Consensus'])
ax[1].set_ylabel('Cutoff frequency')
ax[1].set_title(f'Mann-Whitney U U = {stat:.3f}, p = {p_mwu:.3g}', fontsize=10)
ax[1].grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()



### Plot the harmonics
##############################
Qind_HC_RT = np.load('./OUTPUT/EPvsCTRL/Q_ind_%s_HC_%s_RT.npy'%(metric,dwi))
Qind_HC_LT = np.load('./OUTPUT/EPvsCTRL/Q_ind_%s_HC_%s_LT.npy'%(metric,dwi))
Qind_EP_RT = np.load('./OUTPUT/EPvsCTRL/Q_ind_%s_EP_%s_RT.npy'%(metric,dwi))
Qind_EP_LT = np.load('./OUTPUT/EPvsCTRL/Q_ind_%s_EP_%s_LT.npy'%(metric,dwi))

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

### 
vmin=-2; vmax=2
for i in np.arange(5):
    #plot_rois_pyvista(scipy.stats.zscore(Qind_HC_RT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_HC_RT_%d'%i)
    #plot_rois_pyvista(scipy.stats.zscore(Qind_EP_RT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_EP_RT_%d'%i)
    #plot_rois_pyvista(scipy.stats.zscore(Qind_HC_LT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_HC_LT_%d'%i)
    #plot_rois_pyvista(scipy.stats.zscore(Qind_EP_LT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_EP_LT_%d'%i)
    plot_rois_pyvista_superior(scipy.stats.zscore(Qind_HC_RT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics', vmin=vmin, vmax=vmax, label='Qind_HC_RT_%s_%d'%(metric,i))
    plot_rois_pyvista_superior(scipy.stats.zscore(Qind_EP_RT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics', vmin=vmin, vmax=vmax, label='Qind_EP_RT_%s_%d'%(metric,i))
    plot_rois_pyvista_superior(scipy.stats.zscore(Qind_HC_LT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  vmin=vmin, vmax=vmax,label='Qind_HC_LT_%s_%d' %(metric,i))
    plot_rois_pyvista_superior(scipy.stats.zscore(Qind_EP_LT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  vmin=vmin, vmax=vmax,label='Qind_EP_LT_%s_%d'%(metric,i))
    tmp = np.abs(scipy.stats.zscore(Qind_HC_RT[:,i])) - np.abs(scipy.stats.zscore(Qind_EP_RT[:,i]))
    plot_rois_pyvista_superior(tmp, scale, './FIGURES/EPvsCTRL/harmonics', label='Qind_HC-EP_RT_%s_%d'%(metric,i))
    tmp = np.abs(scipy.stats.zscore(Qind_HC_LT[:,i])) - np.abs(scipy.stats.zscore(Qind_EP_LT[:,i]))
    plot_rois_pyvista_superior(tmp, scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_HC-EP_LT_%s_%d'%(metric,i))
    tmp = np.abs(scipy.stats.zscore(Qind_EP_LT[:,i])) - np.abs(scipy.stats.zscore(Qind_EP_RT[:,i]))
    plot_rois_pyvista_superior(tmp, scale, './FIGURES/EPvsCTRL/harmonics', label='Qind_LT-RT_%s_%d'%(metric,i))
    
### Plot the SDI
#######################
SDI_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_%s_HC_%s_LT.npy'%(metric,dwi))
SDI_EP_LT = np.load('./OUTPUT/EPvsCTRL/SDI_%s_EP_%s_LT.npy'%(metric,dwi))
SDI_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_%s_HC_%s_RT.npy'%(metric,dwi))
SDI_EP_RT = np.load('./OUTPUT/EPvsCTRL/SDI_%s_EP_%s_RT.npy'%(metric,dwi))
surr_thresh_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_HC_%s_LT.npy'%(metric,dwi), allow_pickle=True)
surr_thresh_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_HC_%s_RT.npy'%(metric,dwi), allow_pickle=True)
surr_thresh_EP_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_EP_%s_RT.npy'%(metric,dwi), allow_pickle=True)
surr_thresh_EP_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_EP_%s_LT.npy'%(metric,dwi), allow_pickle=True)
SDI_sig_subjectwise_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_%s_HC_%s_LT.npy'%(metric,dwi), allow_pickle=True)
SDI_sig_subjectwise_EP_LT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_%s_EP_%s_LT.npy'%(metric,dwi), allow_pickle=True)
SDI_sig_subjectwise_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_%s_HC_%s_RT.npy'%(metric,dwi), allow_pickle=True)
SDI_sig_subjectwise_EP_RT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_%s_EP_%s_RT.npy'%(metric,dwi), allow_pickle=True)

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

nbROIs_EP_RT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_%s_EP_%s_RT.npy'%(metric,dwi))
nbROIs_EP_LT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_%s_EP_%s_LT.npy'%(metric,dwi))
nbROIs_HC_RT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_%s_HC_%s_RT.npy'%(metric,dwi))
nbROIs_HC_LT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_%s_HC_%s_LT.npy'%(metric,dwi))

fig, ax = plt.subplots(1,1)
ls_nbROIs_sig = [nbROIs_HC_RT, nbROIs_HC_LT, nbROIs_EP_RT, nbROIs_EP_LT]
ls_surr_thresh = [surr_thresh_HC_RT, surr_thresh_HC_LT, surr_thresh_EP_RT, surr_thresh_EP_LT]
ls_labels = ["HC RT", "HC LT", "EP RT", "EP LT"]
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

#print((np.where(surr_thresh_HC_LT[5]['SDI_sig']!=0)[0])) # 6 out of 8 patients
#print((np.where(surr_thresh_EP_LT[5]['SDI_sig']!=0)[0])) # 7 out of 9 patients
df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
labels_118 = df_118['Label Lausanne2008']
labels_118 = np.array(labels_118)
print('Significant ROIs for threshold 5:')
tmp = np.where(surr_thresh_HC_LT[5]['SDI_sig']!=0)[0]
for i in range(len(tmp)):
    print(f"HC_RT {labels_118[tmp[i]]} {surr_thresh_HC_RT[5]['mean_SDI'][tmp[i]]}")
tmp = np.where(surr_thresh_HC_RT[5]['SDI_sig']!=0)[0]
for i in range(len(tmp)):
    print(f"HC_RT {labels_118[tmp[i]]} {surr_thresh_HC_RT[5]['mean_SDI'][tmp[i]]}")
tmp = np.where(surr_thresh_EP_LT[5]['SDI_sig']!=0)[0]
for i in range(len(tmp)):
    print(f"EP_LT {labels_118[tmp[i]]} {surr_thresh_EP_LT[5]['mean_SDI'][tmp[i]]}")
tmp = np.where(surr_thresh_EP_RT[5]['SDI_sig']!=0)[0]
for i in range(len(tmp)):
    print(f"EP_RT {labels_118[tmp[i]]} {surr_thresh_EP_RT[5]['mean_SDI'][tmp[i]]}")

# Print table with significant ROIs and their mean SDI values
groups = {"HC_LT": surr_thresh_HC_LT,"HC_RT": surr_thresh_HC_RT,
    "EP_LT": surr_thresh_EP_LT,"EP_RT": surr_thresh_EP_RT,}

# Collect all indices that are significant in any group
all_idx = set()
for surr in groups.values():
    sig_idx = np.where(surr[5]['SDI_sig'] != 0)[0]
    all_idx.update(sig_idx)
all_idx = sorted(list(all_idx))

# Build a dictionary for DataFrame
data = {("ROI", ""): [labels_118[idx] for idx in all_idx]}  # make ROI a tuple
for side in ["LT", "RT"]:
    for group in ["HC", "EP"]:
        col_name = (side, group)  # multi-index column
        values = []
        for idx in all_idx:
            key = f"{group}_{side}"
            if groups[key][5]['SDI_sig'][idx] != 0:
                values.append(round(groups[key][5]['mean_SDI'][idx],2))
            else:
                values.append(np.nan)
        data[col_name] = values

# Create DataFrame with MultiIndex columns
df = pd.DataFrame(data)
df.columns = pd.MultiIndex.from_tuples(df.columns)

# Print DataFrame
print(df)

# Optional: export to Excel
df.to_excel("SDI_comparison_table_EPvsCTRL.xlsx", index=True)

#print(labels_118[np.where(surr_thresh_HC_LT[5]['SDI_sig']!=0)[0]])
#print(labels_118[np.where(surr_thresh_EP_LT[5]['SDI_sig']!=0)[0]])
#print(labels_118[np.where(surr_thresh_HC_RT[5]['SDI_sig']!=0)[0]])
#print(labels_118[np.where(surr_thresh_EP_RT[5]['SDI_sig']!=0)[0]])

#print(surr_thresh_HC_LT[5]['mean_SDI'][np.where(surr_thresh_HC_LT[5]['SDI_sig']!=0)[0]])
#print(surr_thresh_EP_LT[5]['mean_SDI'][np.where(surr_thresh_EP_LT[5]['SDI_sig']!=0)[0]])
#print(surr_thresh_HC_RT[5]['mean_SDI'][np.where(surr_thresh_HC_RT[5]['SDI_sig']!=0)[0]])
#print(surr_thresh_EP_RT[5]['mean_SDI'][np.where(surr_thresh_EP_RT[5]['SDI_sig']!=0)[0]])

plt.show()