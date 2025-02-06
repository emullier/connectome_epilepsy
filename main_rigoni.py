'''This script is to validate that the code reproduces the results of the paper Rigoni2023 with original code in  matlab'''

import numpy as np 
import scipy.io as sio
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import lib.func_reading as reading 
import lib.func_SDI as sdi
import lib.func_ML as ML
from lib.func_plot import plot_rois, plot_rois_pyvista
from scipy.signal import butter, filtfilt


config_path = 'config.json' #'config_PEP3.json'
config_defaults = reading.check_config_file(config_path)

### Reading the data
SC  = sio.loadmat('./data/Connectome_scale-%d.mat'%2)['num']
roi_info_path = 'data/label/roi_info.xlsx'
roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
#cort_rois = np.concatenate((np.arange(0,57), np.arange(62,121), [126,127]))
SC = SC[cort_rois,:]; SC = SC[:, cort_rois]
matMetric = SC
x = np.asarray(roi_info['x-pos'])[cort_rois]; y = np.asarray(roi_info['y-pos'])[cort_rois]; z = np.asarray(roi_info['z-pos'])[cort_rois]
coordMat = np.concatenate((x[:,None],y[:,None],z[:,None]),1)
Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))  

### Compute SDI
X_RS_allPat = sdi.load_EEG_example()
P_ind, Q_ind, Ln_ind, An_ind = ML.cons_normalized_lap(matMetric, Euc, plot=False)
np.save('tests/matMetric_mainrigoni', matMetric)

fs = 1000
# Design a bandstop filter
lowcut = 30  # Lower edge of the band to remove
highcut = 60  # Upper edge of the band to remove
order = 6  # Filter order
b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype="bandstop")


### Estimate SDI
#SDI_tmp = np.zeros((118, len(X_RS_allPat)))
SDI_tmp = np.zeros((114, len(X_RS_allPat)))
corr_Xc = np.zeros((114,114, len(X_RS_allPat)))
corr_Xd = np.zeros((114,114, len(X_RS_allPat)))
Xc_all = np.ones((114, 0)); Xd_all = np.empty((114, 0))
ls_lat = []
for p in np.arange(len(X_RS_allPat)):
    X_RS = X_RS_allPat[p]['X_RS']
    idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59,116)))
    X_RS = X_RS[idxs_tmp, :, :]
    ls_lat.append(X_RS_allPat[p]['lat'][0])
    PSD,NN, Vlow, Vhigh = sdi.get_cutoff_freq(Q_ind, X_RS)
    SDI_tmp[:,p], X_c, X_d, SD_hat = sdi.compute_SDI(X_RS, Q_ind)
    Xc_concat =X_c.reshape(X_c.shape[0], -1)  # Concatenate over the third dimension
    Xd_concat = X_d.reshape(X_d.shape[0], -1)  # Concatenate over the third dimension
    Xc_concat = filtfilt(b, a, Xc_concat)
    Xd_concat = filtfilt(b, a, Xd_concat)
    Xc_all = np.concatenate([Xc_all, Xc_concat], axis=1)
    Xd_all = np.concatenate([Xd_all, Xd_concat], axis=1)
    corr_Xc[:,:,p] = np.corrcoef(np.transpose(Xc_concat), rowvar=False)
    corr_Xd[:,:,p] = np.corrcoef(np.transpose(Xd_concat), rowvar=False)
ls_lat = np.array(ls_lat)
SDI = SDI_tmp
np.save('tests/SDI_mainrigoni', SDI)


### Surrogate part
SDI_surr = sdi.surrogate_sdi(Q_ind, Vlow, Vhigh, config_defaults, nbSurr=20, example=True) # Generate the surrogate 
SDI_surr = SDI_surr[idxs_tmp, :, :]
    
fig,ax = plt.subplots(1,2)
for l, lat in enumerate(np.unique(ls_lat)):
    ### not related here, the functional signales are always the same, they should be loaded all the time the same way
    SDI_tmp = SDI; SDI_surr_tmp = SDI_surr
    #SDI_surr_tmp = SDI_surr_tmp[idxs_tmp,:,:]; 
    #print(np.shape(SDI_tmp))
    surr_thresh = sdi.select_significant_sdi(SDI_tmp[:,np.where(ls_lat==lat)[0]], SDI_surr_tmp[:,:,np.where(ls_lat==lat)[0]])
    thr = 1
    nbROIsig = []; ls_th = []
    for t in np.arange(np.shape(surr_thresh)[0]):
        th = surr_thresh[t]['threshold'] 
        ls_th.append(th)
        tmp = surr_thresh[t]['mean_SDI']*np.abs(surr_thresh[t]['SDI_sig'])
        nbROIsig.append(len(np.where(np.abs(surr_thresh[t]['SDI_sig']))[0]))
        #plot_rois_pyvista(tmp, config_defaults["Parameters"]["scale"], config_defaults, vmin=-1, vmax=1, label='SDI_th%d_%s'%(surr_thresh[t]['threshold'], lat))
        if th==3:
            tmp = surr_thresh[t]['mean_SDI']
            roi = roi_info['Label Lausanne2008']
            #roi = np.array(roi[np.where(roi_info['Structure'] == 'cort')[0]])
            roi = roi[cort_rois]
            #print(roi)
            #print(roi[np.where(tmp>0)[0]])
            print(np.shape(roi))
            print(np.shape(tmp))
            np.save('tests/th3_%s_main_rigoni'%lat, tmp)
       

sub = 8   
fig,ax = plt.subplots(2,1)
#ax[0].imshow(corr_Xc[:,:,sub])
#ax[1].imshow(corr_Xd[:,:,sub])
ax[0].plot(Xc_all[0,:])
ax[0].plot(Xd_all[0,:])
plt.show()