

''' This script reproduces the results of the paper (Rigoni,2023), as a validation of 
the functions recreated in python from the original matlab code provided by the authors.

Last modified: EM, 20.11.2025 
Created: 11.03.2025, Emeline Mullier
University of Geneva & Lausanne University Hospital '''


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.func_GSP as gsp
from lib.func_plot import plot_rois, plot_rois_pyvista,  plot_rois_pyvista_superior
import scipy.io as sio
import scipy
import seaborn as sns

ls_lateralization = ["RT", "LT"]
metric = "number_of_fibers" 

for l, lateralization in enumerate(ls_lateralization):

    #lateralization="LT"
    data_path = "DATA/Connectome_scale-2.mat"
    example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"
    infoGVA_path = './DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv'
    scale = 2

    ### Generate the structural harmonics
    #########################################
    ### Load the data
    matMetric = sio.loadmat(data_path)
    matMetric = matMetric['num']
    cort_rois = np.concatenate((np.arange(0,57), [62,63], np.arange(64,121), [126,127]))
    matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
    consensus = matMetric
    ### which one is used in Isotta paper
    EucDist = np.load("DATA/EucMat_HC_DSI_number_of_fibers.npy")

    print("Generate harmonics from the consensus")
    ### Generate the harmonics
    P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
    np.save('./OUTPUT/INDvsCTRL/Q_ind_Iso_%s.npy'%(lateralization), Q_ind)
    np.save('./OUTPUT/INDvsCTRL/P_ind_Iso_%s.npy'%(lateralization), P_ind)

    ### Project the functional signals
    ########################################
    print("Load EEG example data for SDI")
    X_RS_allPat = gsp.load_EEG_example(example_dir)
    
    
    ### Estimate SDI
    ls_cutoff = []
    SDI_tmp = np.zeros((118, len(X_RS_allPat)))
    ls_lat = []; SDI={}; SDI_surr={}
    cutoff_path= './OUTPUT/INDvsCTRL/cutoff_Iso.npy'
    for p in np.arange(len(X_RS_allPat)):
        
        
        X_RS = X_RS_allPat[p]['X_RS']
        zX_RS = scipy.stats.zscore(X_RS, axis=1, ddof=0) ### added 5.05

        ### zscore in Iso matlab code - 05.05
        ls_lat.append(X_RS_allPat[p]['lat'][0])
        PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, zX_RS)
        ls_cutoff.append(NN)
        SDI_tmp[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(zX_RS, Q_ind)
                 
    np.save(cutoff_path, ls_cutoff)



    ls_lat = np.array(ls_lat)
    SDI = SDI_tmp
    if lateralization=='RT':
        idxs_lat = np.where(ls_lat=='Rtle')[0]
    elif lateralization=='LT':
        idxs_lat = np.where(ls_lat=='Ltle')[0]
 
    SDI = SDI[:, idxs_lat]
    SDI_path = './OUTPUT/INDvsCTRL/SDI_Iso_%s.npy'%(lateralization)    
    Xc_norm_path = './OUTPUT/INDvsCTRL/Xc_norm_Iso_%s.npy'%(lateralization)
    Xd_norm_path = './OUTPUT/INDvsCTRL/Xd_norm_Iso_%s.npy'%(lateralization)
    
    
    np.save(Xc_norm_path, X_c_norm, allow_pickle=True)
    np.save(Xd_norm_path, X_d_norm, allow_pickle=True) # Save the surrogate                                                                                            
    np.save(SDI_path, SDI)

    #plot_rois_pyvista(np.mean(SDI,axis=1), scale, './FIGURES/INDvsCTRL', label='SDImean_Iso_%s'%(lateralization))

    ### Surrogate part
    nbSurr = 10
    surr_path = './OUTPUT/INDvsCTRL/SDI_surr_Iso_%s.npy'%( lateralization)
    if not os.path.exists(surr_path):
        SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate 
        np.save(surr_path, SDI_surr) # Save the surrogate
    else:   
        SDI_surr = np.load(surr_path)
        print('Surrogate SDI already generated')


    idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
    surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
    surr_thresh_path = './OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_%s.npy'%(lateralization)
    surr_sig_subjectwise_path = './OUTPUT/INDvsCTRL/SDI_sig_subjectwise_Iso_%s.npy'%(lateralization)
    np.save(surr_thresh_path, surr_thresh, allow_pickle=True) # Save the surrogate
    np.save(surr_sig_subjectwise_path, SDI_sig_subjectwise, allow_pickle=True) # Save the surrogate

    nbROIs_sig = []
    for p in np.arange(np.shape(surr_thresh)[0]):
        nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
    np.save('./OUTPUT/INDvsCTRL/nbROIs_sig_Iso_%s.npy'%(lateralization), nbROIs_sig)

    thr = 2
    plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'], scale, './FIGURES/INDvsCTRL', label='SDImean_thr%d_Iso_%s'%(thr, lateralization))
    plt.show()


### Plot the consensus connectome
########################################
consensus_HC = np.load("DATA/matMetric_HC_DSI_number_of_fibers.npy")
consensus_HC = np.mean(consensus_HC, axis=2)
data_path = "DATA/Connectome_scale-2.mat"
matMetric = sio.loadmat(data_path)
matMetric = matMetric['num']
cort_rois = np.concatenate((np.arange(0,57), [62,63], np.arange(64,121), [126,127]))
matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
consensus_ind = matMetric

cons_HC_vec = consensus_HC.flatten()
cons_ind_vec = consensus_ind.flatten()

idxs = np.where((cons_HC_vec>0)*(cons_ind_vec>0))[0]
fig, axs = plt.subplots(1,3, figsize=(10,10), constrained_layout=True)
axs[0].imshow(consensus_HC); axs[0].set_title('Consensus HC DSI GVA \n #streamlines %d'% np.sum(consensus_HC))
axs[1].imshow(consensus_ind); axs[1].set_title('Consensus EP RT \n #streamlines %d'% np.sum(consensus_ind))
axs[2].scatter(cons_HC_vec[idxs], cons_ind_vec[idxs]); axs[2].set_xlabel('HC'); axs[2].set_ylabel('Ind.')


### Plot coupled/decoupled signals
######################################
Xc_norm_RT = np.load('./OUTPUT/INDvsCTRL/Xc_norm_Iso_RT.npy', allow_pickle=True)
Xc_norm_LT = np.load('./OUTPUT/INDvsCTRL/Xc_norm_Iso_LT.npy', allow_pickle=True)
Xd_norm_RT = np.load('./OUTPUT/INDvsCTRL/Xd_norm_Iso_RT.npy', allow_pickle=True)
Xd_norm_LT = np.load('./OUTPUT/INDvsCTRL/Xd_norm_Iso_LT.npy', allow_pickle=True)


fig, ax = plt.subplots(2,1,figsize=(15,10), constrained_layout=True)
ax[0].plot(np.mean(Xc_norm_RT, axis=1), label='Xc'); ax[0].plot(np.mean(Xd_norm_RT, axis=1), label='Xd')
ax[0].set_title('Normalized energy of ROI time courses \n RT') 
ax[0].set_xlabel('Time (s)'); ax[0].set_ylabel('Normalized energy'); ax[0].legend(['Coupled', 'Decoupled'])
ax[1].plot(np.mean(Xc_norm_RT, axis=1), label='Xc'); ax[1].plot(np.mean(Xd_norm_RT, axis=1), label='Xd')
ax[1].set_title('Normalized energy of ROI time courses \n LT')
ax[1].set_xlabel('Time (s)'); ax[1].set_ylabel('Normalized energy'); ax[1].legend(['Coupled', 'Decoupled'])

### Plot the cutoff frequencies
######################################
cutoff_HC = np.load('./OUTPUT/EPvsCTRL/cutoff_%s_HC_dsi.npy'%metric)
cutoff_EP = np.load('./OUTPUT/INDvsCTRL/cutoff_Iso.npy')

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
nbROIs_sig_RT = np.load('./OUTPUT/INDvsCTRL/nbROIs_sig_Iso_RT.npy', allow_pickle=True)
nbROIs_sig_LT = np.load('./OUTPUT/INDvsCTRL/nbROIs_sig_Iso_LT.npy', allow_pickle=True)
surr_thresh_RT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_RT.npy', allow_pickle=True)
surr_thresh_LT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_LT.npy', allow_pickle=True)
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
Qind_EP_RT = np.load('./OUTPUT/INDvsCTRL/Q_ind_Iso_RT.npy')
Qind_EP_LT = np.load('./OUTPUT/INDvsCTRL/Q_ind_Iso_LT.npy')

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
    plot_rois_pyvista_superior(scipy.stats.zscore(Qind_HC_RT[:,i]), scale, './FIGURES/INDvsCTRL/harmonics', vmin=vmin, vmax=vmax, label='Qind_HC_RT_%d'%i)
    plot_rois_pyvista_superior(scipy.stats.zscore(Qind_EP_RT[:,i]), scale, './FIGURES/INDvsCTRL/harmonics', vmin=vmin, vmax=vmax, label='Qind_Iso_RT_%d'%i)
    plot_rois_pyvista_superior(scipy.stats.zscore(Qind_HC_LT[:,i]), scale, './FIGURES/INDvsCTRL/harmonics',  vmin=vmin, vmax=vmax,label='Qind_HC_LT_%d'%i)
    plot_rois_pyvista_superior(scipy.stats.zscore(Qind_EP_LT[:,i]), scale, './FIGURES/INDvsCTRL/harmonics',  vmin=vmin, vmax=vmax,label='Qind_Iso_LT_%d'%i)
    tmp = np.abs(scipy.stats.zscore(Qind_HC_RT[:,i])) - np.abs(scipy.stats.zscore(Qind_EP_RT[:,i]))
    plot_rois_pyvista_superior(tmp, scale, './FIGURES/INDvsCTRL/harmonics', label='Qind_HC-Iso_RT_%d'%i)
    tmp = np.abs(scipy.stats.zscore(Qind_HC_LT[:,i])) - np.abs(scipy.stats.zscore(Qind_EP_LT[:,i]))
    plot_rois_pyvista_superior(tmp, scale, './FIGURES/INDvsCTRL/harmonics',  label='Qind_HC-Iso_LT_%d'%i)
    tmp = np.abs(scipy.stats.zscore(Qind_EP_LT[:,i])) - np.abs(scipy.stats.zscore(Qind_EP_RT[:,i]))
    plot_rois_pyvista_superior(tmp, scale, './FIGURES/INDvsCTRL/harmonics', label='Qind_LT-RT_%d'%i)
    
### Plot the SDI
#######################
SDI_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_HC_dsi_LT.npy')
SDI_EP_LT = np.load('./OUTPUT/INDvsCTRL/SDI_Iso_LT.npy')
SDI_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_HC_dsi_RT.npy')
SDI_EP_RT = np.load('./OUTPUT/INDvsCTRL/SDI_Iso_RT.npy')
surr_thresh_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_HC_dsi_LT.npy', allow_pickle=True)
surr_thresh_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_HC_dsi_RT.npy', allow_pickle=True)
surr_thresh_EP_RT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_RT.npy', allow_pickle=True)
surr_thresh_EP_LT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_HC_dsi_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_LT = np.load('./OUTPUT/INDvsCTRL/SDI_sig_subjectwise_Iso_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_HC_dsi_RT.npy', allow_pickle=True)
SDI_sig_subjectwise_EP_RT = np.load('./OUTPUT/INDvsCTRL/SDI_sig_subjectwise_Iso_RT.npy', allow_pickle=True)

#print((np.where(surr_thresh_HC_LT[5]['SDI_sig']!=0)[0])) # 6 out of 8 patients
#print((np.where(surr_thresh_HC_RT[5]['SDI_sig']!=0)[0])) # 7 out of 9 patients
df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
labels_118 = df_118['Label Lausanne2008']
labels_118 = np.array(labels_118)


print(labels_118[np.where(surr_thresh_EP_LT[5]['SDI_sig']!=0)[0]])
print(labels_118[np.where(surr_thresh_EP_RT[5]['SDI_sig']!=0)[0]])
print(surr_thresh_EP_LT[5]['mean_SDI'][np.where(surr_thresh_EP_LT[5]['SDI_sig']!=0)[0]])
print(surr_thresh_EP_RT[5]['mean_SDI'][np.where(surr_thresh_EP_RT[5]['SDI_sig']!=0)[0]])


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

nbROIs_EP_RT = np.load('./OUTPUT/INDvsCTRL/nbROIs_sig_Iso_RT.npy')
nbROIs_EP_LT = np.load('./OUTPUT/INDvsCTRL/nbROIs_sig_Iso_LT.npy')
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