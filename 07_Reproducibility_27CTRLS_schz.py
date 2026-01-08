

''' This script generates the results of (Rigoni,2023) but using a consensus of 27 healthy controls rather than the
original one used by the authors in their paper.
Why 27, because uncertainty on the processing used on the data used in the original paper.

Last modified: EM, 20.11.2025 
Created: 11.03.2025, Emeline Mullier
University of Geneva & Lausanne University Hospital '''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.func_GSP as gsp
from lib.func_plot import plot_rois, plot_rois_pyvista, plot_rois_pyvista_noaxes
import scipy.io as sio
import seaborn as sns
import scipy

# Set matplotlib font to Aptos Body (with fallback to sans-serif)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Aptos', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 10

# Define consistent color palette
COLOR_IND = '#1f77b4'   # Blue for IND (independent dataset)

ls_lateralization = ["RT", "LT"]
#data_path = "DATA/Connectome_scale-2.mat"
example_dir = "DATA/EEG"
infoGVA_path = './DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv'
scale = 2

for l, lateralization in enumerate(ls_lateralization):

    ### Generate the structural harmonics
    #########################################
    ### Load the data
    matMetric = np.load("DATA/SC/matMetric_SCHZ_CTRL.npy")  # Independent dataset
    consensus = np.mean(matMetric, axis=0)
    ### which one is used in Isotta paper
    EucDist = np.load("DATA/EucMat/EucMat_HC_DSI_number_of_fibers.npy")

    print("Generate harmonics from the consensus")
    ### Generate the harmonics
    P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
    if not os.path.exists('./OUTPUT/IND27vsCTRL/'):
        os.makedirs('./OUTPUT/IND27vsCTRL/')
    np.save('./OUTPUT/IND27vsCTRL/Q_ind_ind_%s.npy'%(lateralization), Q_ind)
    np.save('./OUTPUT/IND27vsCTRL/P_ind_ind_%s.npy'%(lateralization), P_ind)

    ### Project the functional signals
    ########################################
    print("Load EEG example data for SDI")
    X_RS_allPat = gsp.load_EEG_example(example_dir)

    ### Estimate SDI
    ls_cutoff = []
    SDI_tmp = np.zeros((118, len(X_RS_allPat)))
    ls_lat = []; SDI={}; SDI_surr={}
    cutoff_path= './OUTPUT/IND27vsCTRL/cutoff_ind.npy'
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
    SDI_path = './OUTPUT/IND27vsCTRL/SDI_ind_%s.npy'%(lateralization)                                                                                                
    np.save(SDI_path, SDI)

    if not os.path.exists('./FIGURES/IND27vsCTRL/'):
        os.makedirs('./FIGURES/IND27vsCTRL/')
    plot_rois_pyvista_noaxes(np.mean(SDI,axis=1), scale, './FIGURES/IND27vsCTRL', vmin=-2, vmax=2, label='SDImean_ind_%s'%(lateralization))

    ### Surrogate part
    nbSurr = 100
    surr_path = './OUTPUT/SDI_surr_ind_%s.npy'%( lateralization)
    if not os.path.exists(surr_path):
        SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate 
        np.save(surr_path, SDI_surr) # Save the surrogate
    else:   
        SDI_surr = np.load(surr_path)
        print('Surrogate SDI already generated')

    idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
    surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
    surr_thresh_path = './OUTPUT/IND27vsCTRL/SDI_surr_thresh_ind_%s.npy'%(lateralization)
    surr_sig_subjectwise_path = './OUTPUT/IND27vsCTRL/SDI_sig_subjectwise_ind_%s.npy'%(lateralization)
    np.save(surr_thresh_path, surr_thresh, allow_pickle=True) # Save the surrogate
    np.save(surr_sig_subjectwise_path, SDI_sig_subjectwise, allow_pickle=True) # Save the surrogate

    nbROIs_sig = []
    for p in np.arange(np.shape(surr_thresh)[0]):
        nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
    np.save('./OUTPUT/IND27vsCTRL/nbROIs_sig_ind_%s.npy'%(lateralization), nbROIs_sig)

    thr = 2
    plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), scale, './FIGURES/IND27vsCTRL', vmin=-2, vmax=2, label='SDImean_thr%d_ind_%s'%(thr, lateralization))

    thr = 5
    plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), scale, './FIGURES/IND27vsCTRL', vmin=-2, vmax=2, label='SDImean_thr%d_ind_%s'%(thr, lateralization))

# Plot the consensus connectome comparison (only once, outside the loop)
########################################
consensus_HC = np.load("DATA/SC/matMetric_HC_DSI_number_of_fibers.npy")
consensus_HC = np.mean(consensus_HC, axis=2)
matMetric_IND = np.load("DATA/SC/matMetric_SCHZ_CTRL.npy")  # Independent dataset
consensus_IND = np.mean(matMetric_IND, axis=0)

cons_HC_vec = consensus_HC.flatten()
cons_IND_vec = consensus_IND.flatten()

idxs = np.where((cons_HC_vec>0)*(cons_IND_vec>0))[0]
fig, axs = plt.subplots(1,3, figsize=(15, 5), constrained_layout=True)
axs[0].imshow(consensus_HC); axs[0].set_title('Consensus HC DSI GVA \n #streamlines %d'% np.sum(consensus_HC), fontsize=12, fontweight='bold')
axs[1].imshow(consensus_IND); axs[1].set_title('Consensus IND \n #streamlines %d'% np.sum(consensus_IND), fontsize=12, fontweight='bold')
axs[2].scatter(cons_HC_vec[idxs], cons_IND_vec[idxs], color=COLOR_IND, alpha=0.6, edgecolors='w', s=60)
axs[2].set_xlabel('HC', fontsize=11, fontweight='bold'); axs[2].set_ylabel('IND', fontsize=11, fontweight='bold')
r_value, p_value = scipy.stats.pearsonr(cons_HC_vec[idxs], cons_IND_vec[idxs])
axs[2].set_title(f"Pearson r = {r_value:.3f}, p = {p_value:.3f}", fontsize=12, fontweight='bold')
axs[2].grid(True, alpha=0.3, linestyle='--')
for ax in axs:
    ax.tick_params(labelsize=10)

### Plot the cutoff frequencies comparison
########################################
cutoff_HC = np.load('./OUTPUT/EPvsCTRL/cutoff_number_of_fibers_HC_dsi.npy')
cutoff_IND = np.load('./OUTPUT/IND27vsCTRL/cutoff_ind.npy')

stat, p_mwu = scipy.stats.wilcoxon(cutoff_HC, cutoff_IND, alternative='two-sided')
print(f"HC - IND \n Wilcoxon signed-rank test statistic = {stat:.3f}, p-value = {p_mwu:.3g}")
r_value, p_value = scipy.stats.pearsonr(cutoff_HC, cutoff_IND)

fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
ax[0].scatter(cutoff_HC, cutoff_IND, c='k', alpha=.6, edgecolors='darkgray', s=80, linewidth=0.5)
ax[0].set_title(f"Pearson r = {r_value:.3f}, p = {p_value:.3f}", fontsize=12, fontweight='bold', pad=10)
ax[0].set_xlabel('Cutoff frequency HC', fontsize=11, fontweight='bold'); ax[0].set_ylabel('Cutoff frequency IND', fontsize=11, fontweight='bold'); ax[0].grid(True, alpha=0.3, linestyle='--'); ax[0].tick_params(labelsize=10)

box_palette = ['#1f77b4', '#9467bd']  # HC (blue) and IND (purple)
sns.boxplot(data=[cutoff_HC, cutoff_IND], ax=ax[1], width=0.5, palette=box_palette)
for patch in ax[1].patches:
    patch.set_alpha(0.8)
sns.stripplot(data=[cutoff_HC, cutoff_IND], ax=ax[1], color='black', size=5, jitter=True, dodge=True, alpha=0.6)
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['HC Consensus', 'IND Consensus'], fontsize=11, fontweight='bold')
ax[1].set_ylabel('Cutoff frequency', fontsize=11, fontweight='bold')
ax[1].set_title(f'Wilcoxon signed-rank test statistic = {stat:.3f}, p = {p_mwu:.3g}', fontsize=12, fontweight='bold', pad=10)
ax[1].grid(True, axis='y', linestyle='--', alpha=0.3); ax[1].tick_params(labelsize=10)

### Load SDI data for correlation plot
########################################
SDI_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_number_of_fibers_HC_dsi_RT.npy')
SDI_IND_RT = np.load('./OUTPUT/IND27vsCTRL/SDI_ind_RT.npy')
SDI_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_number_of_fibers_HC_dsi_LT.npy')
SDI_IND_LT = np.load('./OUTPUT/IND27vsCTRL/SDI_ind_LT.npy')

surr_thresh_HC_RT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_number_of_fibers_HC_dsi_RT.npy', allow_pickle=True)
surr_thresh_IND_RT = np.load('./OUTPUT/IND27vsCTRL/SDI_surr_thresh_ind_RT.npy', allow_pickle=True)
surr_thresh_HC_LT = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_number_of_fibers_HC_dsi_LT.npy', allow_pickle=True)
surr_thresh_IND_LT = np.load('./OUTPUT/IND27vsCTRL/SDI_surr_thresh_ind_LT.npy', allow_pickle=True)

nROIs = 118
mean_SDI_HC_RT = np.zeros((np.shape(surr_thresh_HC_RT)[0], nROIs))
mean_SDI_IND_RT = np.zeros((np.shape(surr_thresh_IND_RT)[0], nROIs))
mean_SDI_HC_LT = np.zeros((np.shape(surr_thresh_HC_LT)[0], nROIs))
mean_SDI_IND_LT = np.zeros((np.shape(surr_thresh_IND_LT)[0], nROIs))

for s in np.arange(np.shape(surr_thresh_HC_RT)[0]):
    mean_SDI_HC_RT[s,:] = surr_thresh_HC_RT[s]['mean_SDI']
    mean_SDI_IND_RT[s,:] = surr_thresh_IND_RT[s]['mean_SDI']
for s in np.arange(np.shape(surr_thresh_HC_LT)[0]):
    mean_SDI_HC_LT[s,:] = surr_thresh_HC_LT[s]['mean_SDI']
    mean_SDI_IND_LT[s,:] = surr_thresh_IND_LT[s]['mean_SDI']

# Correlation plot
fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True) 

# RTLE plot
r_RT, p_RT = scipy.stats.pearsonr(mean_SDI_HC_RT[0,:], mean_SDI_IND_RT[0,:])
axs[0].scatter(mean_SDI_HC_RT[0,:], mean_SDI_IND_RT[0,:], c='k', alpha=0.6, s=80, edgecolors='darkgray', linewidth=0.5)
axs[0].set_title(f"RTLE\nCorrelation between mean SDI\n(r = {r_RT:.2f}, p = {p_RT:.2e})", fontsize=14, fontweight='bold', pad=15)
axs[0].set_xlabel('Mean SDI HC', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Mean SDI IND', fontsize=12, fontweight='bold')
axs[0].grid(True, alpha=0.3, linestyle='--')
axs[0].tick_params(labelsize=11)

# LTLE plot
r_LT, p_LT = scipy.stats.pearsonr(mean_SDI_HC_LT[0,:], mean_SDI_IND_LT[0,:])
axs[1].scatter(mean_SDI_HC_LT[0,:], mean_SDI_IND_LT[0,:], c='k', alpha=0.6, s=80, edgecolors='darkgray', linewidth=0.5)
axs[1].set_title(f"LTLE \nCorrelation between mean SDI\n(r = {r_LT:.2f}, p = {p_LT:.2e})", fontsize=14, fontweight='bold', pad=15)
axs[1].set_xlabel('Mean SDI HC', fontsize=12, fontweight='bold')
axs[1].set_ylabel('Mean SDI IND', fontsize=12, fontweight='bold')
axs[1].grid(True, alpha=0.3, linestyle='--')
axs[1].tick_params(labelsize=11)

### Plot the number of significant ROIs for different thresholds
########################################
nbROIs_HC_RT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_number_of_fibers_HC_dsi_RT.npy')
nbROIs_HC_LT = np.load('./OUTPUT/EPvsCTRL/nbROIs_sig_number_of_fibers_HC_dsi_LT.npy')
nbROIs_IND_RT = np.load('./OUTPUT/IND27vsCTRL/nbROIs_sig_ind_RT.npy')
nbROIs_IND_LT = np.load('./OUTPUT/IND27vsCTRL/nbROIs_sig_ind_LT.npy')

fig, ax = plt.subplots(1,1, figsize=(12, 7), constrained_layout=True)
ls_nbROIs_sig = [nbROIs_HC_RT, nbROIs_HC_LT, nbROIs_IND_RT, nbROIs_IND_LT]
ls_surr_thresh = [surr_thresh_HC_RT, surr_thresh_HC_LT, surr_thresh_IND_RT, surr_thresh_IND_LT]
ls_labels = ["HC RT", "HC LT", "IND RT", "IND LT"]
COLORS_COMPARISON = ['#1f77b4', '#ff7f0e', '#9467bd', '#e377c2']  # HC RT (blue), HC LT (orange), IND RT (purple), IND LT (pink)
MARKERS_COMPARISON = ['o', 's', '^', 'D']

for t in np.arange(len(ls_labels)):
    surr_thresh = ls_surr_thresh[t]
    nbROIs_sig = ls_nbROIs_sig[t]
    ax.plot(np.arange(np.shape(surr_thresh)[0]), np.array(nbROIs_sig), marker=MARKERS_COMPARISON[t], linewidth=2.5, markersize=5, color=COLORS_COMPARISON[t], label=ls_labels[t])
    for i, y_value in enumerate(nbROIs_sig):
        ax.text(i, y_value+0.3, f'{int(y_value)}', fontsize=8, ha='center', va='bottom', fontweight='bold')
ax.set_xlabel('Number of EEG participants', fontsize=12, fontweight='bold'); ax.set_ylabel('# ROIs with significant SDI', fontsize=12, fontweight='bold')
ax.set_xticks(np.arange(0, np.shape(surr_thresh)[0]+1))
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_title('Number of Significant SDI ROIs per Threshold', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='best', framealpha=0.9); ax.tick_params(labelsize=10)

# Generate table for RT and LT (after processing both lateralizations)
########################################
df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
labels_118 = df_118['Label Lausanne2008']
labels_118 = np.array(labels_118)

# Load surrogates for both lateralizations and build combined table
groups = {}
for lat in ["RT", "LT"]:
    surr_thresh_HC = np.load('./OUTPUT/EPvsCTRL/SDI_surr_thresh_number_of_fibers_HC_dsi_%s.npy'%(lat), allow_pickle=True)
    surr_thresh_IND = np.load('./OUTPUT/IND27vsCTRL/SDI_surr_thresh_ind_%s.npy'%(lat), allow_pickle=True)
    groups[f"HC_{lat}"] = surr_thresh_HC
    groups[f"IND_{lat}"] = surr_thresh_IND

# Collect all indices that are significant in any group
all_idx = set()
for surr in groups.values():
    sig_idx = np.where(surr[5]['SDI_sig'] != 0)[0]
    all_idx.update(sig_idx)
all_idx = sorted(list(all_idx))

# Build a dictionary for DataFrame with MultiIndex columns
data = {("ROI", ""): [labels_118[idx] for idx in all_idx]}  # make ROI a tuple
for lat in ["LT", "RT"]:
    for group in ["HC", "IND"]:
        col_name = (lat, group)  # multi-index column
        values = []
        for idx in all_idx:
            key = f"{group}_{lat}"
            if groups[key][5]['SDI_sig'][idx] != 0:
                values.append(round(groups[key][5]['mean_SDI'][idx], 2))
            else:
                values.append(np.nan)
        data[col_name] = values

# Create DataFrame with MultiIndex columns
df = pd.DataFrame(data)
df.columns = pd.MultiIndex.from_tuples(df.columns)

# Print DataFrame
print("\nSignificant ROIs for HC and IND (threshold=5):")
print(df)

# Optional: export to Excel
df.to_excel("SDI_comparison_table_IND27_vs_HC.xlsx", index=True)

#plt.show()

