

import os
import numpy as np
import scipy.io as sio 
import scipy
import pygsp
import lib.func_ML as ML
import lib.func_reading as reading
from scipy.stats import binom
import h5py
import lib.func_utils as utils


def get_cutoff_freq(sc, data):
    X_hat_L = np.zeros(np.shape(data))
    #sc = np.squeeze(sc)
    ## compute CUT-OFF FREQUENCY
    for ep in np.arange(np.shape(data)[2]):
        X_hat_L[:,:,ep]=(sc)@data[:,:,ep]
    ## power
    pow=np.square(np.abs(X_hat_L))
    ## mean across time
    PSD = np.squeeze(np.mean(pow,axis=1))
    ## mean across subjects/epochs
    mPSD = np.mean(PSD, axis=1);
    ## total area under the curve
    AUCTOT = np.trapz(mPSD[:sc.shape[0]]) ##total area under the curve  
    i=1; AUC=0;
    while AUC<AUCTOT/2:
        i=i+1; AUC=np.trapz(mPSD[0:i])
    NN=i-1; #CUTOFF FREQUENCY : number of low frequency eigenvalues to consider in order to have the same energy as the high freq ones
    ## split structural harmonics in high/low frequency
    Vlow=np.zeros(np.shape(sc)); Vhigh=np.zeros(np.shape(sc))
    Vhigh[:,NN+1:]=sc[:,NN+1:] # high frequencies= decoupled
    Vlow[:,1:NN]=sc[:,1:NN] #low frequencies = coupled
    
    return PSD,NN,Vlow, Vhigh

def filter_signal_with_harmonics(sc,data,Vlow,Vhigh):
    ## sc = harmonics of the structural connectome [ROI x HARM]
    ## Vlow = low freq harmonics [ROI x HARM]
    ## Vhigh = high freq harmonics [ROI x HARM]
    X_hat = np.zeros(np.shape(data))
    X_c = np.copy(X_hat); X_d = np.copy(X_hat); 
    N_d = np.zeros((np.shape(data)[0],np.shape(data)[2])); N_c = np.copy(N_d); N_hat = np.copy(N_d)
    ## compute ESI HF/LF portions
    for ep in np.arange(np.shape(data)[2]):
        X_hat[:,:,ep] = sc @ data[:,:,ep]
        X_c[:,:,ep]=Vlow@X_hat[:,:,ep]
        X_d[:,:,ep]=Vhigh@X_hat[:,:,ep];
        # norms  of the weights over time
        for r in np.arange(np.shape(data)[0]):
            N_c[r,ep]=np.linalg.norm(X_c[r,:,ep])
            N_d[r,ep]=np.linalg.norm(X_d[r,:,ep])
            N_hat[r,ep]=np.linalg.norm(X_hat[r,:,ep])
            
    ## STRUCTURAL DECOUPLING INDEX
    SDI=np.mean(N_d,1)/np.mean(N_c,1); #emipirical individual SDI
    SD_hat = np.mean(N_hat, 1)
    return SD_hat, X_c, X_d, N_c, N_d, SDI

def getBD(zX_RS,X_c,X_d):
    ## LF/HF content
    ## calculate power in time for normalization
    power_in_time = np.zeros((np.shape(zX_RS)[2], np.shape(zX_RS)[1]))
    for ep in np.arange(np.shape(power_in_time)[0]):
        for t in np.arange(np.shape(zX_RS)[1]):
            power_in_time[ep,t] = np.linalg.norm(zX_RS[:,t,ep])
    power_in_time = np.transpose(power_in_time)
    X_c_norm = np.zeros((np.shape(zX_RS)[1], np.shape(zX_RS)[2]))
    X_d_norm = np.zeros((np.shape(zX_RS)[1], np.shape(zX_RS)[2]))
    ## normalize backprojected time series
    for ep in np.arange(np.shape(zX_RS)[2]):
        for t in np.arange(np.shape(zX_RS)[1]):
            # normalize by the norm of the power of the original signal
            X_c_norm[t,ep]=((np.linalg.norm(np.squeeze(X_c[:,t,ep])))/power_in_time[t,ep])
            X_d_norm[t,ep]=((np.linalg.norm(np.squeeze(X_d[:,t,ep])))/power_in_time[t,ep])
    ## get Broadcasting Direction BD
    BD=(np.mean(X_d_norm,axis=1)- np.transpose(np.mean(X_c_norm,axis=1))) # BDnorm

    return BD, X_c_norm, X_d_norm


def compute_SDI(X_RS, scU):
    ## X_RS dimensions (nROIs, ntimepoints, ntrials)
    ## scU Laplacian/eigenvectors
    zX_RS = scipy.stats.zscore(X_RS, axis=None)
    [PSD,NN, Vlow, Vhigh] = get_cutoff_freq(scU, zX_RS); #split harmonics in high and low frequency and get PSD
    ## get the part of the signal that is COUPLED and DECOUPLED from the structure
    SD_hat, X_c, X_d, N_c, N_d, SDI = filter_signal_with_harmonics(scU,zX_RS,Vlow,Vhigh)
    ## normalise X_c and X_d and get Broadcasting Direction
    BD, X_c_norm, X_d_norm = getBD(zX_RS,X_c,X_d)
    #SDI=np.mean(N_d,1)/np.mean(N_c,1);#(np.shape(SDI_surr))
    return SDI, X_c, X_d, SD_hat

def load_EEG_example():
    func_path = 'data/func_data.mat'
    func = sio.loadmat(func_path)['func_data']
    names = np.array([item[0] for item in func['name'].flatten()])
    X_RS_allPat = []
    for n, name in enumerate(names):
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
        X_RS_allPat.append({'name': name, 'X_RS': X_RS, 'lat': lat})
    return X_RS_allPat


def surrogate_sdi(scU,  Vlow, Vhigh, config, nbSurr=1000, example=False): 
    X_RS_allPat = load_EEG_example()
    SDI_surr = np.zeros((114, 19, len(X_RS_allPat)))
    
    if example==True:
        SDI_surr = np.zeros((118, 19, len(X_RS_allPat)))

        with h5py.File(os.path.join(config["Parameters"]["example_dir"], 'PHI.mat'), 'r') as f:
            tmp = f['PHI'][()]
        #PHI = utils.extract_ctx_ROIs(tmp)
        PHI = tmp
        nbSurr = np.shape(PHI)[2]

        GSP2_surr = sio.loadmat(os.path.join(config["Parameters"]["example_dir"],'data_GSP2_surr.mat'))
        GSP2_surr = GSP2_surr['data_GSP2_surr'][0]

        for s in np.arange(np.shape(GSP2_surr)[0]):
            sub = GSP2_surr[s][0]
            lat = GSP2_surr[s][1]
            surr = GSP2_surr[s][2][0][0][0]
            #idxs_ctxs = np.concatenate((np.arange(0,57), np.arange(60,117)))
            #SDI_surr[:,:,s] = surr[idxs_ctxs,:]
            SDI_surr[:,:,s] = surr

    else:
        #PHI = np.zeros((np.shape(scU)[0], np.shape(scU)[0], nbSurr))
        #for n in np.arange(nbSurr):
        #    # %randomize sign of Fourier coefficients
        #    PHIdiag= np.round(np.random.rand(np.shape(scU)[0]))
        #    PHIdiag[np.where(PHIdiag==0)] = -1
        #    PHI[:,:,n] = np.diag(PHIdiag)
        
        with h5py.File(os.path.join(config["Parameters"]["example_dir"], 'PHI.mat'), 'r') as f:
            tmp = f['PHI'][()]
        PHI = utils.extract_ctx_ROIs(tmp)
        nbSurr = np.shape(PHI)[2]

        for s in np.arange(len(X_RS_allPat)):
            for n in np.arange(19):
                print('sub-%d, n%d'%(s,n))
                X_RS = X_RS_allPat[s]['X_RS']
                idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59,116)))
                #print(idxs_tmp)
                X_RS = X_RS[idxs_tmp, :, :]
                zX_RS = scipy.stats.zscore(X_RS, axis=None)
                XrandS = np.zeros(np.shape(X_RS))
                PHI_curr = np.squeeze(PHI[:,:,n])
                for p in np.arange(np.shape(X_RS)[2]):
                    zX_RS_curr = scipy.stats.zscore(zX_RS[:,:,p])
                    XrandS[:,:,p] = scU@PHI_curr@np.transpose(scU)@zX_RS_curr
                    #  X_hat=M'X, normally reconstructed signal would be Xrecon=M*X_hat=MM'X, instead of M, M*PHI is V with randomized signs
                SD_hat, X_c, X_d, N_c, N_d, SDI = filter_signal_with_harmonics(scU, XrandS, Vlow, Vhigh)
                SDI_surr[:,n,s]=np.mean(N_d,1)/np.mean(N_c,1);#(np.shape(SDI_surr))
    return SDI_surr


def select_significant_sdi(SDI, SDI_surr):
    ### initiation of max and min for threshold
    max_SDI_surr = np.zeros((np.shape(SDI)))
    min_SDI_surr = np.copy(max_SDI_surr)
    SDI_thr_max = np.copy(min_SDI_surr); SDI_thr_min = np.copy(min_SDI_surr)
    ### conversion to log SDI
    SDI_surr = np.log(SDI_surr)
    SDI = np.log(SDI); SDI=np.squeeze(SDI)
    ### mean SDI
    mean_SDI = np.mean(SDI, axis=1)
    ### find threshold
    for s in np.arange(np.shape(SDI_surr)[2]):
        max_SDI_surr[:,s] = np.max(SDI_surr[:,:,s],axis=1)
        min_SDI_surr[:,s] = np.min(SDI_surr[:,:,s],axis=1)    
        ### select significant SDI for each subject, across surrogates individual th, first screening
    for s in np.arange(np.shape(SDI)[1]):
        SDI_thr_max[:,s] = SDI[:,s] > max_SDI_surr[:,s]
        SDI_thr_min[:,s] = SDI[:,s] < min_SDI_surr[:,s]
        detect_max = np.sum(SDI_thr_max, axis=1)  # Sums along the first axis (rows)
        detect_min = np.sum(SDI_thr_min, axis=1)
        
    ### for every region, test across subjects 0.05, correcting for the number oftests (regions), 0.05/118
    x = np.arange(0, 101)
    # Calculate the complementary binomial cumulative distribution function (1 - cdf)
    y = binom.sf(x, 100, 0.05)
    # Find the first index where y is less than 0.05 / size(mean_SDI, 0)
    threshold_index = np.min(np.where(y < 0.05 / mean_SDI.shape[0]))
    # Get the corresponding value of x
    THRsubjects = x[threshold_index]
    # Final calculation for THRsubjects
    #THRsubjects = int(np.floor(SDI.shape[0] / 100 * THRsubjects))+1
    THRsubjects = 2
        
    surr_thresh = []
    for thr in range(THRsubjects, SDI.shape[1]):
        SDI_sig_higher = detect_max > thr
        SDI_sig_lower = detect_min > thr
        SDI_sig = np.zeros(mean_SDI.shape[0])
        SDI_sig[SDI_sig_higher] = 1
        SDI_sig[SDI_sig_lower] = -1
        surr_thresh.append({'threshold':thr, 'mean_SDI': mean_SDI, 'SDI_sig':SDI_sig})
    return surr_thresh