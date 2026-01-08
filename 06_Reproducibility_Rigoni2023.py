

''' This script reproduces the results of the paper (Rigoni,2023), as a validation of 
the functions recreated in python from the original matlab code provided by the authors.

- Fig 1: Comparison Consensus connectome from Isotta paper to HC consensus connectome from DSI GVA data
- Fig 2: Normalized energy of coupled/decoupled signals in RTLE and LTLE patients
- Fig 3: Cutoff frequencies comparison between HC consensus and Isotta consensus
- Fig 4: Number of significant ROIs for different thresholds in RTLE and LTLE

Missing:
- Table of significant ROIs in RTLE and LTLE patients as in 02_SDI_consensus_pipeline.py
- SDI visualization on the brain ? Check in the saved figures
- Boxplot for significant ROIs (Fig 5 in the original paper)

Last modified: EM, 25.11.2025
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

# Set font configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Aptos', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 10

# Define colors - consistent with code 02 and 07
COLOR_HC_RT = '#ADD8E6'  # Pale blue for HC RT
COLOR_HC_LT = '#90EE90'  # Light green for HC LT
COLOR_IND_RT = '#1f77b4'  # Blue for IND (Isotta) RT
COLOR_IND_LT = '#FF6B6B'  # Red for IND (Isotta) LT
COLOR_IND27_RT = '#FFB366'  # Pale orange for IND27 RT
COLOR_IND27_LT = '#FFA500'  # Orange for IND27 LT
COLOR_IND = '#1f77b4'  # Blue for independent datasets
COLORS_COMPARISON = ['#ADD8E6', '#90EE90', '#1f77b4', '#FF6B6B']  # HC RT, HC LT, IND RT, IND LT - matching code 07
MARKERS_COMPARISON = ['o', 's', '^', 'D']  # Distinct markers for each group
COLORS_EXTENDED = [COLOR_HC_RT, COLOR_HC_LT, COLOR_IND_RT, COLOR_IND_LT, COLOR_IND27_RT, COLOR_IND27_LT]
MARKERS_EXTENDED = ['o', 's', '^', 'D', 'v', 'P']

ls_lateralization = ["RT", "LT"]
metric = "number_of_fibers" 

for l, lateralization in enumerate(ls_lateralization):

    #lateralization="LT"
    data_path = "DATA/Connectome_scale-2.mat"
    example_dir = "DATA/EEG"
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
    EucDist = np.load("DATA/EucMat/EucMat_HC_DSI_number_of_fibers.npy")

    print("Generate harmonics from the consensus")
    if not os.path.exists('./OUTPUT/INDvsCTRL'):
        os.makedirs('./OUTPUT/INDvsCTRL')
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

    
    if not os.path.exists('./FIGURES/INDvsCTRL'):
        os.makedirs('./FIGURES/INDvsCTRL')
    thr = 2
    plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), scale, './FIGURES/INDvsCTRL', label='SDImean_thr%d_Iso_%s'%(thr, lateralization))
    thr = 5
    plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), scale, './FIGURES/INDvsCTRL', label='SDImean_thr%d_Iso_%s'%(thr, lateralization))


### Process IND27 dataset (27 healthy controls consensus)
#########################################################
for l, lateralization in enumerate(ls_lateralization):
    
    ### Load IND27 data
    matMetric_IND27 = np.load("DATA/SC/matMetric_SCHZ_CTRL.npy")  # 27 independent controls
    consensus_IND27 = np.mean(matMetric_IND27, axis=0)
    EucDist = np.load("DATA/EucMat/EucMat_HC_DSI_number_of_fibers.npy")
    
    print(f"Generate harmonics from IND27 consensus for {lateralization}")
    ### Generate the harmonics
    P_ind27, Q_ind27, Ln_ind27, An_ind27 = gsp.cons_normalized_lap(consensus_IND27, EucDist, plot=False)
    np.save('./OUTPUT/INDvsCTRL/Q_ind_IND27_%s.npy'%(lateralization), Q_ind27)
    np.save('./OUTPUT/INDvsCTRL/P_ind_IND27_%s.npy'%(lateralization), P_ind27)
    
    ### Project the functional signals
    print("Load EEG example data for SDI (IND27)")
    X_RS_allPat = gsp.load_EEG_example(example_dir)
    
    ### Estimate SDI
    ls_cutoff_IND27 = []
    SDI_tmp_IND27 = np.zeros((118, len(X_RS_allPat)))
    ls_lat = []
    cutoff_path_IND27 = './OUTPUT/INDvsCTRL/cutoff_IND27.npy'
    
    for p in np.arange(len(X_RS_allPat)):
        X_RS = X_RS_allPat[p]['X_RS']
        zX_RS = scipy.stats.zscore(X_RS, axis=1, ddof=0)
        ls_lat.append(X_RS_allPat[p]['lat'][0])
        PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind27, zX_RS)
        ls_cutoff_IND27.append(NN)
        SDI_tmp_IND27[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(zX_RS, Q_ind27)
    
    np.save(cutoff_path_IND27, ls_cutoff_IND27)
    
    ls_lat = np.array(ls_lat)
    SDI_IND27 = SDI_tmp_IND27
    if lateralization=='RT':
        idxs_lat = np.where(ls_lat=='Rtle')[0]
    elif lateralization=='LT':
        idxs_lat = np.where(ls_lat=='Ltle')[0]
    
    SDI_IND27 = SDI_IND27[:, idxs_lat]
    SDI_path_IND27 = './OUTPUT/INDvsCTRL/SDI_IND27_%s.npy'%(lateralization)
    np.save(SDI_path_IND27, SDI_IND27)
    
    ### Surrogate part for IND27
    nbSurr = 10
    surr_path_IND27 = './OUTPUT/INDvsCTRL/SDI_surr_IND27_%s.npy'%(lateralization)
    if not os.path.exists(surr_path_IND27):
        SDI_surr_IND27 = gsp.surrogate_sdi(Q_ind27, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False)
        np.save(surr_path_IND27, SDI_surr_IND27)
    else:
        SDI_surr_IND27 = np.load(surr_path_IND27)
        print('Surrogate SDI for IND27 already generated')
    
    idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
    surr_thresh_IND27, SDI_sig_subjectwise_IND27 = gsp.select_significant_sdi(SDI_IND27, SDI_surr_IND27[:,:,idxs_lat])
    surr_thresh_path_IND27 = './OUTPUT/INDvsCTRL/SDI_surr_thresh_IND27_%s.npy'%(lateralization)
    surr_sig_subjectwise_path_IND27 = './OUTPUT/INDvsCTRL/SDI_sig_subjectwise_IND27_%s.npy'%(lateralization)
    np.save(surr_thresh_path_IND27, surr_thresh_IND27, allow_pickle=True)
    np.save(surr_sig_subjectwise_path_IND27, SDI_sig_subjectwise_IND27, allow_pickle=True)
    
    nbROIs_sig_IND27 = []
    for p in np.arange(np.shape(surr_thresh_IND27)[0]):
        nbROIs_sig_IND27.append(len(np.where(np.abs(surr_thresh_IND27[p]['SDI_sig']))[0]))
    np.save('./OUTPUT/INDvsCTRL/nbROIs_sig_IND27_%s.npy'%(lateralization), nbROIs_sig_IND27)
    
    thr = 2
    plot_rois_pyvista(surr_thresh_IND27[thr]['mean_SDI']*np.abs(surr_thresh_IND27[thr]['SDI_sig']), scale, './FIGURES/INDvsCTRL', label='SDImean_thr%d_IND27_%s'%(thr, lateralization))
    thr = 5
    plot_rois_pyvista(surr_thresh_IND27[thr]['mean_SDI']*np.abs(surr_thresh_IND27[thr]['SDI_sig']), scale, './FIGURES/INDvsCTRL', label='SDImean_thr%d_IND27_%s'%(thr, lateralization))


### Plot the consensus connectome
########################################
consensus_HC = np.load("DATA/SC/matMetric_HC_DSI_number_of_fibers.npy")
consensus_HC = np.mean(consensus_HC, axis=2)
data_path = "DATA/Connectome_scale-2.mat"
matMetric = sio.loadmat(data_path)
matMetric = matMetric['num']
cort_rois = np.concatenate((np.arange(0,57), [62,63], np.arange(64,121), [126,127]))
matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
consensus_ind = matMetric

# Load IND27 consensus
matMetric_IND27 = np.load("DATA/SC/matMetric_SCHZ_CTRL.npy")
consensus_ind27 = np.mean(matMetric_IND27, axis=0)

cons_HC_vec = consensus_HC.flatten()
cons_ind_vec = consensus_ind.flatten()
cons_ind27_vec = consensus_ind27.flatten()

idxs = np.where((cons_HC_vec>0)*(cons_ind_vec>0))[0]
idxs_ind27 = np.where((cons_HC_vec>0)*(cons_ind27_vec>0))[0]

fig, axs = plt.subplots(2,3, figsize=(15,10), constrained_layout=True)
axs[0,0].imshow(consensus_HC); axs[0,0].set_title('Consensus HC DSI GVA \n #streamlines %d'% np.sum(consensus_HC))
axs[0,1].imshow(consensus_ind); axs[0,1].set_title('Consensus IND (Isotta) \n #streamlines %d'% np.sum(consensus_ind))
axs[0,2].imshow(consensus_ind27); axs[0,2].set_title('Consensus IND27 \n #streamlines %d'% np.sum(consensus_ind27))
axs[1,0].scatter(cons_HC_vec[idxs], cons_ind_vec[idxs], alpha=0.6, s=50); axs[1,0].set_xlabel('HC'); axs[1,0].set_ylabel('IND (Isotta)'); axs[1,0].grid(True)
axs[1,1].scatter(cons_HC_vec[idxs_ind27], cons_ind27_vec[idxs_ind27], alpha=0.6, s=50); axs[1,1].set_xlabel('HC'); axs[1,1].set_ylabel('IND27'); axs[1,1].grid(True)
axs[1,2].scatter(cons_ind_vec[idxs], cons_ind27_vec[idxs], alpha=0.6, s=50); axs[1,2].set_xlabel('IND (Isotta)'); axs[1,2].set_ylabel('IND27'); axs[1,2].grid(True)


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
if not os.path.exists('./FIGURES/INDvsCTRL'):
    os.makedirs('./FIGURES/INDvsCTRL')
cutoff_HC = np.load('./OUTPUT/EPvsCTRL/cutoff_%s_HC_dsi.npy'%metric)
cutoff_IND = np.load('./OUTPUT/INDvsCTRL/cutoff_Iso.npy')
cutoff_IND27 = np.load('./OUTPUT/INDvsCTRL/cutoff_IND27.npy')

stat, p_mwu = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_IND, alternative='two-sided')
print(f"HC - IND (Isotta) \n Mann-Whitney U test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
r_value, p_value = scipy.stats.pearsonr(cutoff_HC, cutoff_IND)

stat27, p_mwu27 = scipy.stats.mannwhitneyu(cutoff_HC, cutoff_IND27, alternative='two-sided')
print(f"HC - IND27 \n Mann-Whitney U test statistic = {stat27:.3f}, p-value = {p_mwu27:.3g}")
r_value27, p_value27 = scipy.stats.pearsonr(cutoff_HC, cutoff_IND27)

fig, ax = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

# Merged scatter plot: HC vs IND (Isotta) and HC vs IND27
ax[0,0].scatter(cutoff_HC, cutoff_IND, c='#1f77b4', alpha=.6, edgecolors='darkgray', s=80, linewidth=0.5, label='HC vs IND (Isotta)')
# Add regression line for HC vs IND
z_ind = np.polyfit(cutoff_HC, cutoff_IND, 1)
p_ind = np.poly1d(z_ind)
x_line = np.linspace(cutoff_HC.min(), cutoff_HC.max(), 100)
ax[0,0].plot(x_line, p_ind(x_line), color='#1f77b4', linestyle='--', linewidth=2, alpha=0.8)

ax[0,0].scatter(cutoff_HC, cutoff_IND27, c='#ff7f0e', alpha=.6, edgecolors='darkgray', s=80, linewidth=0.5, label='HC vs IND27')
# Add regression line for HC vs IND27
z_ind27 = np.polyfit(cutoff_HC, cutoff_IND27, 1)
p_ind27 = np.poly1d(z_ind27)
ax[0,0].plot(x_line, p_ind27(x_line), color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.8)

ax[0,0].set_title(f"Cutoff Frequency Comparison\nIND (Isotta): r={r_value:.3f}, p={p_value:.3g} | IND27: r={r_value27:.3f}, p={p_value27:.3g}", fontsize=11, fontweight='bold', pad=10)
ax[0,0].set_xlabel('Cutoff frequency HC', fontsize=11, fontweight='bold')
ax[0,0].set_ylabel('Cutoff frequency (IND or IND27)', fontsize=11, fontweight='bold')
ax[0,0].grid(True, alpha=0.3, linestyle='--')
ax[0,0].tick_params(labelsize=10)
ax[0,0].legend(fontsize=10, loc='upper left')

# Boxplot: HC vs IND (Isotta) vs IND27
box_palette_all = [COLOR_IND, COLOR_IND, COLOR_IND]
sns.boxplot(data=[cutoff_HC, cutoff_IND, cutoff_IND27], ax=ax[0,1], width=0.5, palette=box_palette_all)
for patch in ax[0,1].patches:
    patch.set_alpha(0.6)
sns.stripplot(data=[cutoff_HC, cutoff_IND, cutoff_IND27], ax=ax[0,1], color='black', size=5, jitter=True, dodge=True, alpha=0.6)
ax[0,1].set_xticks([0, 1, 2])
ax[0,1].set_xticklabels(['HC Consensus', 'IND (Isotta)', 'IND27'], fontsize=11, fontweight='bold')
ax[0,1].set_ylabel('Cutoff frequency', fontsize=11, fontweight='bold')
ax[0,1].set_title(f'Boxplot Comparison\nHC-IND: MWU={stat:.3f}, p={p_mwu:.3g} | HC-IND27: MWU={stat27:.3f}, p={p_mwu27:.3g}', fontsize=11, fontweight='bold', pad=10)
ax[0,1].grid(True, axis='y', linestyle='--', alpha=0.3)
ax[0,1].tick_params(labelsize=10)

# Hide the bottom two subplots
ax[1,0].axis('off')
ax[1,1].axis('off')


### Plot the number of significant ROIs
#######################################
nbROIs_HC_RT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_%s_HC_dsi_RT.npy'%metric)
nbROIs_HC_LT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_%s_HC_dsi_LT.npy'%metric)
nbROIs_sig_RT = np.load('./OUTPUT/INDvsCTRL/nbROIs_sig_Iso_RT.npy', allow_pickle=True)
nbROIs_sig_LT = np.load('./OUTPUT/INDvsCTRL/nbROIs_sig_Iso_LT.npy', allow_pickle=True)
nbROIs_sig_IND27_RT = np.load('./OUTPUT/INDvsCTRL/nbROIs_sig_IND27_RT.npy', allow_pickle=True)
nbROIs_sig_IND27_LT = np.load('./OUTPUT/INDvsCTRL/nbROIs_sig_IND27_LT.npy', allow_pickle=True)
surr_thresh_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_HC_dsi_RT.npy'%metric, allow_pickle=True)
surr_thresh_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_HC_dsi_LT.npy'%metric, allow_pickle=True)
surr_thresh_RT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_RT.npy', allow_pickle=True)
surr_thresh_LT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_LT.npy', allow_pickle=True)
surr_thresh_IND27_RT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_IND27_RT.npy', allow_pickle=True)
surr_thresh_IND27_LT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_IND27_LT.npy', allow_pickle=True)

# Define extended color palette for 6 lines
COLORS_EXTENDED = ['#ADD8E6', '#90EE90', '#1f77b4', '#FF6B6B', '#FFB6C1', '#FFA500']  # HC RT, HC LT, IND RT, IND LT, IND27 RT, IND27 LT
MARKERS_EXTENDED = ['o', 's', '^', 'D', 'v', 'P']

fig, ax = plt.subplots(1,1, figsize=(14, 7), constrained_layout=True)
# HC and IND (Isotta) lines using COLORS_COMPARISON to match code 07
ls_nbROIs_first_4 = [nbROIs_HC_RT, nbROIs_HC_LT, nbROIs_sig_RT, nbROIs_sig_LT]
ls_surr_thresh_first_4 = [surr_thresh_HC_RT, surr_thresh_HC_LT, surr_thresh_RT, surr_thresh_LT]
ls_labels_first_4 = ["HC RT", "HC LT", "IND (Isotta) RT", "IND (Isotta) LT"]

for t in np.arange(len(ls_labels_first_4)):
    surr_thresh = ls_surr_thresh_first_4[t]
    nbROIs_sig = ls_nbROIs_first_4[t]
    ax.plot(np.arange(np.shape(surr_thresh)[0]), np.array(nbROIs_sig), marker=MARKERS_COMPARISON[t], linewidth=2.5, markersize=5, color=COLORS_COMPARISON[t], label=ls_labels_first_4[t])
    for i, y_value in enumerate(nbROIs_sig):
        ax.text(i, y_value+0.3, f'{int(y_value)}', fontsize=6, ha='center', va='bottom', fontweight='bold')

# IND27 lines (dashed)
ax.plot(np.arange(np.shape(surr_thresh_IND27_RT)[0]), np.array(nbROIs_sig_IND27_RT), label='IND27 RT', marker='v', linewidth=2.5, markersize=5, linestyle='--', color=COLOR_IND27_RT)
for i, y_value in enumerate(nbROIs_sig_IND27_RT):
    ax.text(i, y_value+0.5, f'{int(y_value)}', fontsize=6, ha='center', va='bottom', fontweight='bold')
ax.plot(np.arange(np.shape(surr_thresh_IND27_LT)[0]), np.array(nbROIs_sig_IND27_LT), label='IND27 LT', marker='P', linewidth=2.5, markersize=5, linestyle='--', color=COLOR_IND27_LT)
for i, y_value in enumerate(nbROIs_sig_IND27_LT):
    ax.text(i, y_value+0.5, f'{int(y_value)}', fontsize=6, ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Number of EEG participants', fontsize=12, fontweight='bold')
ax.set_ylabel('# ROIs with significant SDI', fontsize=12, fontweight='bold')
ax.set_xticks(np.arange(0, np.shape(surr_thresh_HC_RT)[0]))
ax.grid('on', alpha=0.3, linestyle='--')
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.set_title('Number of significant ROIs for each threshold', fontsize=13, fontweight='bold', pad=10)
ax.tick_params(labelsize=10)


### Plot the harmonics
##############################
Qind_HC_RT = np.load('./OUTPUT/EPvsCTRL/Q_ind_%s_HC_dsi_RT.npy'%metric)
Qind_HC_LT = np.load('./OUTPUT/EPvsCTRL/Q_ind_%s_HC_dsi_LT.npy'%metric)
Qind_IND_RT = np.load('./OUTPUT/INDvsCTRL/Q_ind_Iso_RT.npy')
Qind_IND_LT = np.load('./OUTPUT/INDvsCTRL/Q_ind_Iso_LT.npy')

pearson_corrs_LT = np.zeros(Qind_IND_LT.shape[1])
pearson_corrs_RT = np.zeros(Qind_IND_RT.shape[1])
for i in range(Qind_IND_LT.shape[1]):
    # Calculate Pearson correlation between the i-th column of A and B
    corr_LT, _ = scipy.stats.pearsonr(Qind_IND_LT[:,i], Qind_HC_LT[:, i])
    corr_RT, _ = scipy.stats.pearsonr(Qind_IND_RT[:, i], Qind_HC_RT[:, i])
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
if not os.path.exists('./FIGURES/INDvsCTRL/harmonics'):
    os.makedirs('./FIGURES/INDvsCTRL/harmonics')
    for i in np.arange(5):
        #plot_rois_pyvista(scipy.stats.zscore(Qind_HC_RT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_HC_RT_%d'%i)
        #plot_rois_pyvista(scipy.stats.zscore(Qind_IND_RT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_IND_RT_%d'%i)
        #plot_rois_pyvista(scipy.stats.zscore(Qind_HC_LT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_HC_LT_%d'%i)
        #plot_rois_pyvista(scipy.stats.zscore(Qind_IND_LT[:,i]), scale, './FIGURES/EPvsCTRL/harmonics',  label='Qind_IND_LT_%d'%i)
        plot_rois_pyvista_superior(scipy.stats.zscore(Qind_HC_RT[:,i]), scale, './FIGURES/INDvsCTRL/harmonics', vmin=vmin, vmax=vmax, label='Qind_HC_RT_%d'%i)
        plot_rois_pyvista_superior(scipy.stats.zscore(Qind_IND_RT[:,i]), scale, './FIGURES/INDvsCTRL/harmonics', vmin=vmin, vmax=vmax, label='Qind_Iso_RT_%d'%i)
        plot_rois_pyvista_superior(scipy.stats.zscore(Qind_HC_LT[:,i]), scale, './FIGURES/INDvsCTRL/harmonics',  vmin=vmin, vmax=vmax,label='Qind_HC_LT_%d'%i)
        plot_rois_pyvista_superior(scipy.stats.zscore(Qind_IND_LT[:,i]), scale, './FIGURES/INDvsCTRL/harmonics',  vmin=vmin, vmax=vmax,label='Qind_Iso_LT_%d'%i)
        tmp = np.abs(scipy.stats.zscore(Qind_HC_RT[:,i])) - np.abs(scipy.stats.zscore(Qind_IND_RT[:,i]))
        plot_rois_pyvista_superior(tmp, scale, './FIGURES/INDvsCTRL/harmonics', label='Qind_HC-Iso_RT_%d'%i)
        tmp = np.abs(scipy.stats.zscore(Qind_HC_LT[:,i])) - np.abs(scipy.stats.zscore(Qind_IND_LT[:,i]))
        plot_rois_pyvista_superior(tmp, scale, './FIGURES/INDvsCTRL/harmonics',  label='Qind_HC-Iso_LT_%d'%i)
        tmp = np.abs(scipy.stats.zscore(Qind_IND_LT[:,i])) - np.abs(scipy.stats.zscore(Qind_IND_RT[:,i]))
        plot_rois_pyvista_superior(tmp, scale, './FIGURES/INDvsCTRL/harmonics', label='Qind_LT-RT_%d'%i)
    
### Plot the SDI
#######################
SDI_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_%s_HC_dsi_LT.npy'%metric)
SDI_IND_LT = np.load('./OUTPUT/INDvsCTRL/SDI_Iso_LT.npy')
SDI_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_%s_HC_dsi_RT.npy'%metric)
SDI_IND_RT = np.load('./OUTPUT/INDvsCTRL/SDI_Iso_RT.npy')
surr_thresh_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_HC_dsi_LT.npy'%metric, allow_pickle=True)
surr_thresh_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_%s_HC_dsi_RT.npy'%metric, allow_pickle=True)
surr_thresh_IND_RT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_RT.npy', allow_pickle=True)
surr_thresh_IND_LT = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_Iso_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_%s_HC_dsi_LT.npy'%metric, allow_pickle=True)
SDI_sig_subjectwise_IND_LT = np.load('./OUTPUT/INDvsCTRL/SDI_sig_subjectwise_Iso_LT.npy', allow_pickle=True)
SDI_sig_subjectwise_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_sig_subjectwise_%s_HC_dsi_RT.npy'%metric, allow_pickle=True)
SDI_sig_subjectwise_IND_RT = np.load('./OUTPUT/INDvsCTRL/SDI_sig_subjectwise_Iso_RT.npy', allow_pickle=True)

#print((np.where(surr_thresh_HC_LT[5]['SDI_sig']!=0)[0])) # 6 out of 8 patients
#print((np.where(surr_thresh_HC_RT[5]['SDI_sig']!=0)[0])) # 7 out of 9 patients
df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
labels_118 = df_118['Label Lausanne2008']
labels_118 = np.array(labels_118)


print(labels_118[np.where(surr_thresh_IND_LT[5]['SDI_sig']!=0)[0]])
print(labels_118[np.where(surr_thresh_IND_RT[5]['SDI_sig']!=0)[0]])
print(surr_thresh_IND_LT[5]['mean_SDI'][np.where(surr_thresh_IND_LT[5]['SDI_sig']!=0)[0]])
print(surr_thresh_IND_RT[5]['mean_SDI'][np.where(surr_thresh_IND_RT[5]['SDI_sig']!=0)[0]])


#print(len(np.where(SDI_sig_subjectwise_HC_LT[:,6]!=0)[0]))

nROIs = 118
mean_SDI_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
SDI_sig_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_HC_LT = np.copy(mean_SDI_HC_RT); mean_SDI_IND_LT = np.copy(mean_SDI_HC_RT); mean_SDI_IND_RT = np.copy(mean_SDI_HC_RT)
mean_SDI_IND27_RT = np.copy(mean_SDI_HC_RT); mean_SDI_IND27_LT = np.copy(mean_SDI_HC_RT)
SDI_sig_HC_LT = np.copy(SDI_sig_HC_RT); SDI_sig_IND_LT = np.copy(SDI_sig_HC_RT); SDI_sig_IND_RT = np.copy(SDI_sig_HC_RT) 
for s in np.arange(np.shape(surr_thresh_HC_RT)[0]):
    th = surr_thresh_HC_RT[s]['threshold']
    mean_SDI_HC_RT[s,:] = surr_thresh_HC_RT[s]['mean_SDI']
    SDI_sig_HC_RT[s,:] = surr_thresh_HC_RT[s]['SDI_sig']
    mean_SDI_IND_RT[s,:] = surr_thresh_IND_RT[s]['mean_SDI']
    SDI_sig_IND_RT[s,:] = surr_thresh_IND_RT[s]['SDI_sig']
for s in np.arange(np.shape(surr_thresh_HC_LT)[0]):
    th = surr_thresh_HC_LT[s]['threshold']
    mean_SDI_HC_LT[s,:] = surr_thresh_HC_LT[s]['mean_SDI']
    SDI_sig_HC_LT[s,:] = surr_thresh_HC_LT[s]['SDI_sig']
    mean_SDI_IND_LT[s,:] = surr_thresh_IND_LT[s]['mean_SDI']
    SDI_sig_IND_LT[s,:] = surr_thresh_IND_LT[s]['SDI_sig']
for s in np.arange(np.shape(surr_thresh_IND27_RT)[0]):
    mean_SDI_IND27_RT[s,:] = surr_thresh_IND27_RT[s]['mean_SDI']
for s in np.arange(np.shape(surr_thresh_IND27_LT)[0]):
    mean_SDI_IND27_LT[s,:] = surr_thresh_IND27_LT[s]['mean_SDI']

fig, axs = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True) 

# Row 1: HC vs IND (Isotta)
# RT plot
r_RT, p_RT = scipy.stats.pearsonr(mean_SDI_HC_RT[0,:], mean_SDI_IND_RT[0,:])
axs[0,0].scatter(mean_SDI_HC_RT[0,:], mean_SDI_IND_RT[0,:], c=COLORS_COMPARISON[0], alpha=0.6, s=80, edgecolors='darkgray', linewidth=0.5)
axs[0,0].set_title(f"HC vs IND (Isotta) - RT\n(r = {r_RT:.2f}, p = {p_RT:.2e})", fontsize=12, fontweight='bold', pad=15)
axs[0,0].set_xlabel('Mean SDI HC', fontsize=11, fontweight='bold')
axs[0,0].set_ylabel('Mean SDI IND (Isotta)', fontsize=11, fontweight='bold')
axs[0,0].grid(True, alpha=0.3, linestyle='--')
axs[0,0].tick_params(labelsize=10)

# LT plot
r_LT, p_LT = scipy.stats.pearsonr(mean_SDI_HC_LT[0,:], mean_SDI_IND_LT[0,:])
axs[0,1].scatter(mean_SDI_HC_LT[0,:], mean_SDI_IND_LT[0,:], c=COLORS_COMPARISON[1], alpha=0.6, s=80, edgecolors='darkgray', linewidth=0.5)
axs[0,1].set_title(f"HC vs IND (Isotta) - LT\n(r = {r_LT:.2f}, p = {p_LT:.2e})", fontsize=12, fontweight='bold', pad=15)
axs[0,1].set_xlabel('Mean SDI HC', fontsize=11, fontweight='bold')
axs[0,1].set_ylabel('Mean SDI IND (Isotta)', fontsize=11, fontweight='bold')
axs[0,1].grid(True, alpha=0.3, linestyle='--')
axs[0,1].tick_params(labelsize=10)

# Row 2: HC vs IND27
# RT plot
r_RT27, p_RT27 = scipy.stats.pearsonr(mean_SDI_HC_RT[0,:], mean_SDI_IND27_RT[0,:])
axs[1,0].scatter(mean_SDI_HC_RT[0,:], mean_SDI_IND27_RT[0,:], c='#ff7f0e', alpha=0.6, s=80, edgecolors='darkgray', linewidth=0.5)
axs[1,0].set_title(f"HC vs IND27 - RT\n(r = {r_RT27:.2f}, p = {p_RT27:.2e})", fontsize=12, fontweight='bold', pad=15)
axs[1,0].set_xlabel('Mean SDI HC', fontsize=11, fontweight='bold')
axs[1,0].set_ylabel('Mean SDI IND27', fontsize=11, fontweight='bold')
axs[1,0].grid(True, alpha=0.3, linestyle='--')
axs[1,0].tick_params(labelsize=10)

# LT plot
r_LT27, p_LT27 = scipy.stats.pearsonr(mean_SDI_HC_LT[0,:], mean_SDI_IND27_LT[0,:])
axs[1,1].scatter(mean_SDI_HC_LT[0,:], mean_SDI_IND27_LT[0,:], c='#ff7f0e', alpha=0.6, s=80, edgecolors='darkgray', linewidth=0.5)
axs[1,1].set_title(f"HC vs IND27 - LT\n(r = {r_LT27:.2f}, p = {p_LT27:.2e})", fontsize=12, fontweight='bold', pad=15)
axs[1,1].set_xlabel('Mean SDI HC', fontsize=11, fontweight='bold')
axs[1,1].set_ylabel('Mean SDI IND27', fontsize=11, fontweight='bold')
axs[1,1].grid(True, alpha=0.3, linestyle='--')
axs[1,1].tick_params(labelsize=10)

### 25.11.2025: Adapt this part to print the table of significant ROIs, where instead of EP and HC, it would be HC and INd,
### In this context, EP refers to the Ind. consensus connectome from Isotta data
### And HC refers to the HC consensus connectome from DSI GVA data
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
tmp = np.where(surr_thresh_IND_LT[5]['SDI_sig']!=0)[0]
for i in range(len(tmp)):
    print(f"EP_LT {labels_118[tmp[i]]} {surr_thresh_IND_LT[5]['mean_SDI'][tmp[i]]}")
tmp = np.where(surr_thresh_IND_RT[5]['SDI_sig']!=0)[0]
for i in range(len(tmp)):
    print(f"EP_RT {labels_118[tmp[i]]} {surr_thresh_IND_RT[5]['mean_SDI'][tmp[i]]}")

# Print table with significant ROIs and their mean SDI values
surr_thresh_IND27_RT_final = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_IND27_RT.npy', allow_pickle=True)
surr_thresh_IND27_LT_final = np.load('./OUTPUT/INDvsCTRL/SDI_surr_thresh_IND27_LT.npy', allow_pickle=True)

groups = {"HC_LT": surr_thresh_HC_LT,"HC_RT": surr_thresh_HC_RT,
    "IND_LT": surr_thresh_IND_LT,"IND_RT": surr_thresh_IND_RT,
    "IND27_LT": surr_thresh_IND27_LT_final, "IND27_RT": surr_thresh_IND27_RT_final}

# Collect all indices that are significant in any group
all_idx = set()
for surr in groups.values():
    sig_idx = np.where(surr[5]['SDI_sig'] != 0)[0]
    all_idx.update(sig_idx)
all_idx = sorted(list(all_idx))

# Build a dictionary for DataFrame
data = {("ROI", ""): [labels_118[idx] for idx in all_idx]}  # make ROI a tuple
for side in ["LT", "RT"]:
    for group in ["HC", "IND", "IND27"]:
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
df.to_excel("SDI_comparison_table_INDvsCTRL_with_IND27.xlsx", index=True)

plt.show()