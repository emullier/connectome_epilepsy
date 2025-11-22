
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.func_GSP as gsp
from lib.func_plot import plot_rois, plot_rois_pyvista
import scipy.io as sio
import scipy

lateralization = 'LT' 
data_path = "DATA/Connectome_scale-2.mat"
matMetric = sio.loadmat(data_path)
matMetric = matMetric['num']
cort_rois = np.concatenate((np.arange(0,57), [62,63], np.arange(64,121), [126,127]))
matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
consensus_ind = matMetric
consensus_HC_DSI = np.load("DATA/matMetric_HC_DSI_number_of_fibers.npy")
consensus_schz = np.load("DATA/matMetric_SCHZ_CTRL.npy")
example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"
scale = 2

#Qind_ref = consensus_ind
Qind_ref = np.mean(consensus_HC_DSI, axis=2)
ref = 'HCdsi'
consensus = np.mean(consensus_schz,axis=0)

#np.fill_diagonal(consensus, 0)
#EucDist = consensus #### To be replaced by proper Euclidean matrix
#EucDist = (EucDist + EucDist.T)/2
EucDist = np.load("DATA/EucMat_HC_dsi_number_of_fibers.npy")


print("Generate harmonics from the consensus")
### Generate the harmonics
P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)


Qind_rotated, Qind_HC_RT_centered, disparity_RT = scipy.spatial.procrustes(Qind_ref, Q_ind)
R_RT, _ = scipy.linalg.orthogonal_procrustes(Qind_ref, Q_ind)
Qind_ortho_rotated=Q_ind@R_RT 
perm, total_cost = gsp.match_eigenvectors(Qind_ref, Q_ind)
Qind_matched = Q_ind[:,perm]

np.save('./OUTPUT/INDvsCTRL/Q_ind_schz_%s.npy'%( lateralization), Q_ind)
np.save('./OUTPUT/ALIGN/P_ind_schz_%s_ref%s.npy'%(lateralization, ref), P_ind)
np.save('./OUTPUT/ALIGN/Q_ind_rotated_schz_%s_ref%s.npy'%( lateralization, ref), Qind_rotated)
np.save('./OUTPUT/ALIGN/Q_ind_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref), Qind_ortho_rotated) 
np.save('./OUTPUT/ALIGN/Q_ind_matched_schz_%s_ref%s.npy'%( lateralization, ref), Qind_matched)

### Project the functional signals
########################################
print("Load EEG example data for SDI")
X_RS_allPat = gsp.load_EEG_example(example_dir)

### Estimate SDI
ls_cutoff = []
ls_cutoff_rotated = []
ls_cutoff_ortho_rotated = []    
ls_cutoff_matched = []
SDI_tmp = np.zeros((118, len(X_RS_allPat)))
SDI_tmp_rotated = np.zeros((118, len(X_RS_allPat)))
SDI_tmp_ortho_rotated = np.zeros((118, len(X_RS_allPat)))
SDI_tmp_matched = np.zeros((118, len(X_RS_allPat)))
ls_lat = []; SDI={}; SDI_surr={}
cutoff_path= './OUTPUT/INDvsCTRL/cutoff_schz.npy'
cutoff_path_rotated= './OUTPUT/ALIGN/cutoff_rotated_schz_ref%s.npy'%ref
cutoff_path_ortho_rotated= './OUTPUT/ALIGN/cutoff_ortho_rotated_schz_ref%s.npy'%ref
cutoff_path_matched= './OUTPUT/ALIGN/cutoff_matched_schz_ref%s.npy'%ref 
for p in np.arange(len(X_RS_allPat)):
    X_RS = X_RS_allPat[p]['X_RS']
    ls_lat.append(X_RS_allPat[p]['lat'][0])
    PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, X_RS); ls_cutoff.append(NN)
    SDI_tmp[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind)
    PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_rotated, X_RS); ls_cutoff_rotated.append(NN)
    SDI_tmp_rotated[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Qind_rotated)
    PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_ortho_rotated, X_RS); ls_cutoff_ortho_rotated.append(NN)
    SDI_tmp_ortho_rotated[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Qind_ortho_rotated)
    PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_matched, X_RS); ls_cutoff_matched.append(NN)
    SDI_tmp_matched[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Qind_matched)
    
np.save(cutoff_path, ls_cutoff)
np.save(cutoff_path_rotated, ls_cutoff_rotated)
np.save(cutoff_path_ortho_rotated, ls_cutoff_ortho_rotated)
np.save(cutoff_path_matched, ls_cutoff_matched)
ls_lat = np.array(ls_lat)
if lateralization=='RT':
    idxs_lat = np.where(ls_lat=='Rtle')[0]
elif lateralization=='LT':
    idxs_lat = np.where(ls_lat=='Ltle')[0]
 
SDI = SDI_tmp[:, idxs_lat]; SDI_path = './OUTPUT/INDvsCTRL/SDI_schz_%s.npy'%( lateralization); np.save(SDI_path, SDI)
SDI_rotated = SDI_tmp_rotated[:, idxs_lat]; SDI_rotated_path = './OUTPUT/ALIGN/SDI_rotated_schz_%s_ref%s.npy'%( lateralization, ref); np.save(SDI_rotated_path, SDI_rotated)
SDI_ortho_rotated = SDI_tmp_ortho_rotated[:, idxs_lat]; SDI_ortho_rotated_path = './OUTPUT/ALIGN/SDI_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref); np.save(SDI_ortho_rotated_path, SDI_ortho_rotated)
SDI_matched = SDI_tmp_matched[:, idxs_lat]; SDI_matched_path = './OUTPUT/ALIGN/SDI_matched_schz_%s_ref%s.npy'%( lateralization, ref); np.save(SDI_matched_path, SDI_matched)

plot_rois_pyvista(np.mean(SDI,axis=1), scale, './FIGURES/ALIGN', vmin=-1, vmax=1, label='SDImean_schz_%s'%( lateralization))
plot_rois_pyvista(np.mean(SDI_rotated,axis=1), scale, './FIGURES/ALIGN', vmin=-1, vmax=1, label='SDImean_rotated_schz_%s_ref%s'%( lateralization, ref))
plot_rois_pyvista(np.mean(SDI_ortho_rotated,axis=1), scale, './FIGURES/ALIGN', vmin=-1, vmax=1, label='SDImean_ortho_rotated_schz_%s_ref%s'%( lateralization, ref))  
plot_rois_pyvista(np.mean(SDI_matched,axis=1), scale, './FIGURES/ALIGN', vmin=-1, vmax=1, label='SDImean_matched_schz_%s_ref%s'%( lateralization, ref))  

### Surrogate part
nbSurr = 100
surr_path = './OUTPUT/INDvsCTRL/SDI_surr_schz_%s.npy'%( lateralization)
surr_path_rotated = './OUTPUT/ALIGN/SDI_surr_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
surr_path_ortho_rotated = './OUTPUT/ALIGN/SDI_surr_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
surr_path_matched = './OUTPUT/ALIGN/SDI_surr_matched_schz_%s_ref%s.npy'%( lateralization, ref)
if not os.path.exists(surr_path_matched):
    SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate 
    np.save(surr_path, SDI_surr) # Save the surrogate
    SDI_surr_rotated = gsp.surrogate_sdi(Qind_rotated, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate
    np.save(surr_path_rotated, SDI_surr_rotated) # Save the surrogate
    SDI_surr_ortho_rotated = gsp.surrogate_sdi(Qind_ortho_rotated, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate
    np.save(surr_path_ortho_rotated, SDI_surr_ortho_rotated) # Save the surrogate
    SDI_surr_matched = gsp.surrogate_sdi(Qind_matched, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate
    np.save(surr_path_matched, SDI_surr_matched) # Save the surrogate
else:   
    print('Surrogate SDI already generated')
    SDI_surr = np.load(surr_path)
    SDI_surr_rotated = np.load(surr_path_rotated)
    SDI_surr_ortho_rotated = np.load(surr_path_ortho_rotated)           
    SDI_surr_matched = np.load(surr_path_matched)


idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
surr_thresh_rotated, SDI_sig_subjectwise_rotated = gsp.select_significant_sdi(SDI_rotated, SDI_surr_rotated[:,:,idxs_lat])
surr_thresh_ortho_rotated, SDI_sig_subjectwise_ortho_rotated = gsp.select_significant_sdi(SDI_ortho_rotated, SDI_surr_ortho_rotated[:,:,idxs_lat])  
surr_thresh_matched, SDI_sig_subjectwise_matched = gsp.select_significant_sdi(SDI_matched, SDI_surr_matched[:,:,idxs_lat])
surr_thresh_path = './OUTPUT/INDvsCTRL/SDI_surr_thresh_schz_%s.npy'%( lateralization)
surr_thresh_rotated_path = './OUTPUT/ALIGN/SDI_surr_thresh_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
surr_thresh_ortho_rotated_path = './OUTPUT/ALIGN/SDI_surr_thresh_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
surr_thresh_matched_path = './OUTPUT/ALIGN/SDI_surr_thresh_matched_schz_%s_ref%s.npy'%( lateralization, ref)
surr_sig_subjectwise_path = './OUTPUT/INDvsCTRL/SDI_sig_subjectwise_schz_%s.npy'%( lateralization)
surr_sig_subjectwise_rotated_path = './OUTPUT/ALIGN/SDI_sig_subjectwise_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
surr_sig_subjectwise_ortho_rotated_path = './OUTPUT/ALIGN/SDI_sig_subjectwise_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
surr_sig_subjectwise_matched_path = './OUTPUT/ALIGN/SDI_sig_subjectwise_matched_schz_%s_ref%s.npy'%( lateralization, ref)
np.save(surr_thresh_path, surr_thresh, allow_pickle=True) # Save the surrogate
np.save(surr_sig_subjectwise_path, SDI_sig_subjectwise, allow_pickle=True) # Save the surrogate
np.save(surr_thresh_rotated_path, surr_thresh_rotated, allow_pickle=True) # Save the surrogate
np.save(surr_sig_subjectwise_rotated_path, SDI_sig_subjectwise_rotated, allow_pickle=True) # Save the surrogate
np.save(surr_thresh_ortho_rotated_path, surr_thresh_ortho_rotated, allow_pickle=True) # Save the surrogate
np.save(surr_sig_subjectwise_ortho_rotated_path, SDI_sig_subjectwise_ortho_rotated, allow_pickle=True) # Save the surrogate
np.save(surr_thresh_matched_path, surr_thresh_matched, allow_pickle=True) # Save the surrogate
np.save(surr_sig_subjectwise_matched_path, SDI_sig_subjectwise_matched, allow_pickle=True) # Save the surrogate

nbROIs_sig = []
nbROIs_sig_rotated = []
nbROIs_sig_ortho_rotated = []
nbROIs_sig_matched = []
    
for p in np.arange(np.shape(surr_thresh)[0]):
    nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
    nbROIs_sig_rotated.append(len(np.where(np.abs(surr_thresh_rotated[p]['SDI_sig']))[0]))
    nbROIs_sig_ortho_rotated.append(len(np.where(np.abs(surr_thresh_ortho_rotated[p]['SDI_sig']))[0]))
    nbROIs_sig_matched.append(len(np.where(np.abs(surr_thresh_matched[p]['SDI_sig']))[0]))
np.save('./OUTPUT/INDvsCTRL/nbROIs_sig_schz_%s.npy'%( lateralization), nbROIs_sig)
np.save('./OUTPUT/ALIGN/nbROIs_sig_rotated_schz_%s_ref%s.npy'%(lateralization, ref), nbROIs_sig_rotated)
np.save('./OUTPUT/ALIGN/nbROIs_sig_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref), nbROIs_sig_ortho_rotated)
np.save('./OUTPUT/ALIGN/nbROIs_sig_matched_schz_%s_ref%s.npy'%( lateralization, ref), nbROIs_sig_matched)


fig, ax = plt.subplots(1,1)
ax.plot(np.arange(np.shape(surr_thresh)[0]), np.array(nbROIs_sig), color='k')
for i, y_value in enumerate(nbROIs_sig):
    ax.scatter(i, y_value, color='k', marker='x', s=20)  # Cross marker
    ax.text(i, y_value, f'{y_value}', fontsize=8, ha='left', va='bottom', color='k')
ax.axvline(x=0.75*np.shape(surr_thresh)[0], color='r', linestyle='--', linewidth=2, label='75% of participants')
ax.set_xlabel('Threshold #Subs'); ax.set_ylabel('#ROIs with significant SDI')
ax.grid('on', alpha=.2)
ax.set_title('SDI schz %s'%( lateralization))


fig, ax = plt.subplots(1,1)
ax.plot(np.arange(np.shape(ls_cutoff)[0]), np.array(ls_cutoff), marker='x', color='k')
ax.set_title('Cutoff frequency schz %s'%( lateralization))
ax.set_xlabel('Participants'); ax.set_ylabel('Cutoff frequency')
#plt.show()

thr = 2
plot_rois_pyvista(surr_thresh[thr]['mean_SDI']*surr_thresh[thr]['SDI_sig'], scale, './FIGURES/INDvsCTRL', vmin=-1, vmax=1, label='SDImean_thr%d_schz_%s'%(thr, lateralization))
plot_rois_pyvista(surr_thresh_rotated[thr]['mean_SDI']*surr_thresh_rotated[thr]['SDI_sig'], scale, './FIGURES/ALIGN', vmin=-1, vmax=1, label='SDImean_rotated_thr%d_schz_%s_ref%s'%(thr,  lateralization, ref))
plot_rois_pyvista(surr_thresh_ortho_rotated[thr]['mean_SDI']*surr_thresh_ortho_rotated[thr]['SDI_sig'], scale, './FIGURES/ALIGN', vmin=-1, vmax=1, label='SDImean_ortho_rotated_thr%d_schz_%s_ref%s'%(thr, lateralization, ref))
plot_rois_pyvista(surr_thresh_matched[thr]['mean_SDI']*surr_thresh_matched[thr]['SDI_sig'], scale, './FIGURES/ALIGN', vmin=-1, vmax=1, label='SDImean_matched_thr%d_schz_%s_ref%s'%(thr,  lateralization, ref))  
plt.show()

