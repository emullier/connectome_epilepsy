

import numpy as np 
import scipy.io as sio
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import lib.func_reading as reading 
import lib.func_SDI as sdi
import lib.func_ML as ML
from lib.func_plot import plot_rois, plot_rois_pyvista

config_path = 'config.json' #'config_PEP3.json'
config_defaults = reading.check_config_file(config_path)

### Reading the data
SC = sio.loadmat('./data/Individual_Connectomes.mat')
SC = SC['connMatrices']['SC'][0][0][scale-1][0]
roi_info_path = 'data/label/roi_info.xlsx'
roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
matMetric = SC
x = np.asarray(roi_info['x-pos'])[cort_rois]
y = np.asarray(roi_info['y-pos'])[cort_rois]
z = np.asarray(roi_info['z-pos'])[cort_rois]
coordMat = np.concatenate((x[:,None],y[:,None],z[:,None]),1)
Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))  

### Compute SDI
X_RS_allPat = sdi.load_EEG_example()
P_ind, Q_ind, Ln_ind, An_ind = ML.cons_normalized_lap(matMetric, Euc, plot=False)
#np.save('tests/matMetric_mainrigoni', matMetric)

### Estimate SDI
#SDI_tmp = np.zeros((118, len(X_RS_allPat)))
SDI_tmp = np.zeros((114, len(X_RS_allPat)))
ls_lat = []
for p in np.arange(len(X_RS_allPat)):
    X_RS = X_RS_allPat[p]['X_RS']
    idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59,116)))
    X_RS = X_RS[idxs_tmp, :, :]
    ls_lat.append(X_RS_allPat[p]['lat'][0])
    PSD,NN, Vlow, Vhigh = sdi.get_cutoff_freq(Q_ind, X_RS)
    SDI_tmp[:,p], X_c_norm, X_d_norm = sdi.compute_SDI(X_RS, Q_ind)
ls_lat = np.array(ls_lat)
SDI = SDI_tmp
#np.save('tests/SDI_mainrigoni', SDI)


### Surrogate part
SDI_surr = sdi.surrogate_sdi(Q_ind, Vlow, Vhigh, config_defaults, nbSurr=20, example=True) # Generate the surrogate 
SDI_surr = SDI_surr[idxs_tmp, :, :]
    
fig,ax = plt.subplots(1,2)
for l, lat in enumerate(np.unique(ls_lat)):
    ### not related here, the functional signales are always the same, they should be loaded all the time the same way
    SDI_tmp = SDI; SDI_surr_tmp = SDI_surr
    SDI_surr_tmp = SDI_surr_tmp[idxs_tmp,:,:]; 
    #print(np.shape(SDI_tmp))
    surr_thresh = sdi.select_significant_sdi(SDI_tmp[:,np.where(ls_lat==lat)[0]], SDI_surr_tmp[:,:,np.where(ls_lat==lat)[0]])
    thr = 1
    nbROIsig = []; ls_th = []
    for t in np.arange(np.shape(surr_thresh)[0]):
        th = surr_thresh[t]['threshold'] 
        ls_th.append(th)
        tmp = surr_thresh[t]['mean_SDI']*np.abs(surr_thresh[t]['SDI_sig'])
        nbROIsig.append(len(np.where(np.abs(surr_thresh[t]['SDI_sig']))[0]))
        #idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
        #tmp = tmp[idxs_tmp]
        #print(np.shape(tmp))
        plot_rois_pyvista(tmp, config_defaults["Parameters"]["scale"], config_defaults, vmin=-1, vmax=1, label='SDI_th%d_%s'%(surr_thresh[t]['threshold'], lat))
        if th==3:
            tmp = surr_thresh[t]['mean_SDI']
            roi = roi_info['Label Lausanne2008']
            roi = roi[cort_rois]
