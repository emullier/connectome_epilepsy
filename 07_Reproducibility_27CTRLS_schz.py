

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
from lib.func_plot import plot_rois, plot_rois_pyvista
import scipy.io as sio

lateralization="LT"
#data_path = "DATA/Connectome_scale-2.mat"
example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"
infoGVA_path = './DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv'
scale = 2

### Generate the structural harmonics
#########################################
### Load the data
matMetric = np.load("DATA/matMetric_SCHZ_CTRL.npy")
consensus = np.mean(matMetric, axis=0)
### which one is used in Isotta paper
EucDist = np.load("DATA/EucMat_HC_DSI_number_of_fibers.npy")

print("Generate harmonics from the consensus")
### Generate the harmonics
P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
np.save('./OUTPUT/Q_ind_schz_%s.npy'%(lateralization), Q_ind)
np.save('./OUTPUT/P_ind_schz_%s.npy'%(lateralization), P_ind)

### Project the functional signals
########################################
print("Load EEG example data for SDI")
X_RS_allPat = gsp.load_EEG_example(example_dir)

### Estimate SDI
ls_cutoff = []
SDI_tmp = np.zeros((118, len(X_RS_allPat)))
ls_lat = []; SDI={}; SDI_surr={}
cutoff_path= './OUTPUT/cutoff_schz.npy'
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
SDI_path = './OUTPUT/SDI_schz_%s.npy'%(lateralization)                                                                                                
np.save(SDI_path, SDI)

plot_rois_pyvista(np.mean(SDI,axis=1), scale, './FIGURES', vmin=-1, vmax=1, label='SDImean_schz_%s'%(lateralization))

### Surrogate part
nbSurr = 100
surr_path = './OUTPUT/SDI_surr_schz_%s.npy'%( lateralization)
if not os.path.exists(surr_path):
    SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate 
    np.save(surr_path, SDI_surr) # Save the surrogate
else:   
    SDI_surr = np.load(surr_path)
    print('Surrogate SDI already generated')

idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
surr_thresh_path = './OUTPUT/SDI_surr_thresh_schz_%s.npy'%(lateralization)
surr_sig_subjectwise_path = './OUTPUT/SDI_sig_subjectwise_schz_%s.npy'%(lateralization)
np.save(surr_thresh_path, surr_thresh, allow_pickle=True) # Save the surrogate
np.save(surr_sig_subjectwise_path, SDI_sig_subjectwise, allow_pickle=True) # Save the surrogate

nbROIs_sig = []
for p in np.arange(np.shape(surr_thresh)[0]):
    nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
np.save('./OUTPUT/nbROIs_sig_schz_%s.npy'%(lateralization), nbROIs_sig)


fig, ax = plt.subplots(1,1)
ax.plot(np.arange(np.shape(surr_thresh)[0]), np.array(nbROIs_sig), color='k')
for i, y_value in enumerate(nbROIs_sig):
    ax.scatter(i, y_value, color='k', marker='x', s=20)  # Cross marker
    ax.text(i, y_value, f'{y_value}', fontsize=8, ha='left', va='bottom', color='k')
ax.axvline(x=0.75*np.shape(surr_thresh)[0], color='r', linestyle='--', linewidth=2, label='75% of participants')
ax.set_xlabel('Threshold #Subs'); ax.set_ylabel('#ROIs with significant SDI')
ax.grid('on', alpha=.2)
ax.set_title('SDI schz %s'%(lateralization))


fig, ax = plt.subplots(1,1)
ax.plot(np.arange(np.shape(ls_cutoff)[0]), np.array(ls_cutoff), marker='x', color='k')
ax.set_title('Cutoff frequency schz %s'%(lateralization))
ax.set_xlabel('Participants'); ax.set_ylabel('Cutoff frequency')
#plt.show()

thr = 2
plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'], scale, './FIGURES', vmin=-1, vmax=1, label='SDImean_thr%d_schz_%s'%(thr, lateralization))
plt.show()

