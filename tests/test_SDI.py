

import os
import numpy as np
import scipy.io as sio 
import scipy
import pygsp
import lib.func_ML as ML
import lib.func_reading as reading
import lib.func_SDI as sdi

    
config_path = 'config.json'
config = reading.check_config_file(config_path)   

func_path = 'data/func_data.mat'
struct_path = 'data/struct_data.mat'

func = sio.loadmat(func_path)
struct = sio.loadmat(struct_path)

SC = struct['struct_data']['SC'][0][0]
func = func['func_data']
#names_tmp = func['names']
names = np.array([item[0] for item in func['name'].flatten()])

for n, names in enumerate(names[:1]):
    time_w = [.3, .7]
    data_sub = func[0][n]
    sub = data_sub[0]; lat = data_sub[1]; ROI_traces = data_sub[2][0]
    elec = ROI_traces['elec']; time = ROI_traces['time']; fsample = ROI_traces['fsample'][0][0][0]
    label = ROI_traces['label']; trial = ROI_traces['trial'][0].flatten()
    ##define cut-off frequency for each subject on the 400 ms around the IED
    for t in np.arange(len(trial)):
        tmp_trial = trial[t]
        tmp = tmp_trial[:,int(time_w[0]*fsample):int(time_w[1]*fsample-1)]
        if t==0:
            X_RS = np.zeros((np.shape(tmp)[0], np.shape(tmp)[1], len(trial)))
        X_RS[:,:,t] = tmp
    zX_RS = scipy.stats.zscore(X_RS, axis=None)
    EucDist, cort_rois, hemii = ML.ROIs_euclidean_distance(config["Parameters"]["scale"])
    sc = pygsp.graphs.Graph(SC, lap_type='normalized', coords=EucDist)
    sc.compute_fourier_basis()
    print(np.shape(trial))
    #print(np.shape(sc.U))
    [PSD,NN,Vlow, Vhigh] = sdi.get_cutoff_freq(sc.U, zX_RS); #split harmonics in high and low frequency and get PSD

    ## get the part of the signal that is COUPLED and DECOUPLED from the structure
    X_c, X_d, N_c, N_d, SDI = sdi.filter_signal_with_harmonics(sc.U,zX_RS,Vlow,Vhigh)

    ## normalise X_c and X_d and get Broadcasting Direction
    BD, X_c_norm, X_d_norm = sdi.getBD(zX_RS,X_c,X_d)
    
    ## average across trials for final stats and plot
    c_avg_timeseries_norm = np.mean(X_c_norm, axis=1)
    d_avg_timeseries_norm = np.mean(X_d_norm, axis=1)
    
    print(np.shape(SDI))

  
