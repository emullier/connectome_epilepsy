

import os
import numpy as np
import scipy.io as sio 
import scipy
import pygsp
from scipy.stats import binom
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def extract_ctx_ROIs(Mat):
    # Get the number of regions in the matrix (assuming the matrix is 3D: nbROIs x nbROIs x N)
    nbROIs = np.shape(Mat)[0]

    if Mat.ndim==2:
        Mat = Mat[:, :, np.newaxis]

    # Check if the number of ROIs is odd (remove brainstem if necessary)
    if nbROIs % 2 != 0:
        Mat = Mat[:-1, :-1, :]  # Remove the last region (brainstem)
        nbROIs -= 1  # Adjust the number of ROIs after removing brainstem

    # Divide the matrix into two halves
    half_nbROIs = nbROIs // 2
    right_hemisphere_indices = np.arange(half_nbROIs)
    left_hemisphere_indices = np.arange(half_nbROIs, nbROIs)

    # Define the number of cortical regions (57 per hemisphere)
    cortical_regions_count = 57

    # Extract indices for cortical regions
    right_cortical_indices = right_hemisphere_indices[:cortical_regions_count]
    left_cortical_indices = left_hemisphere_indices[:cortical_regions_count]

    # Combine indices
    cortical_indices = np.concatenate([right_cortical_indices, left_cortical_indices])

    # Extract the submatrix corresponding to cortical regions for all 3D slices
    cortical_ROIs = Mat[np.ix_(cortical_indices, cortical_indices, np.arange(Mat.shape[2]))]

    # Plot the first slice of the 3D matrix (for example)
    fig, axs = plt.subplots(1, 1)
    axs.imshow(cortical_ROIs[:, :, 0])  # Visualize the first 2D slice of the 3D matrix

    cortical_ROIs = np.squeeze(cortical_ROIs)

    return cortical_ROIs


def normalize_Lap(A):
    ''' Takes the adjacency matrix as input and returns the corresponding symmetric normalized Laplacian matrix'''
    indices_diag = np.diag_indices(len(A))
    A[indices_diag] = 0
    D = np.sum(A,axis=1)
    epsilon = 1e-10
    D = np.where(D == 0, epsilon, D)
    D = np.diag(D)
    Dn = np.power(D, -0.5)
    Dn = np.diag(np.diag(Dn))
    # symmetric normalize Adjacency
    An = Dn@A@Dn
    Ln = np.diag(np.full(len(An),1)) - An
    # Ln = np.diag(np.sum(An,axis=1)) - An
    return Ln, An

def get_cutoff_freq(sc, data):
    X_hat_L = np.zeros(np.shape(data))
    #sc = np.squeeze(sc)
    ## compute CUT-OFF FREQUENCY
    for ep in np.arange(np.shape(data)[2]):
        X_hat_L[:,:,ep]=np.transpose(sc)@data[:,:,ep] ## added 5.05
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
    Vhigh[:,NN:]=sc[:,NN:] # high frequencies= decoupled
    Vlow[:,:NN]=sc[:,:NN] #low frequencies = coupled
    
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
        X_hat[:,:,ep] = np.transpose(sc) @ data[:,:,ep]
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
    return SDI, X_c_norm, X_d_norm, SD_hat

def load_EEG_example(example_dir):
    func_path = os.path.join(example_dir,'func_data.mat')
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
            tmp = tmp_trial[:,int(time_w[0]*fsample-1):int(time_w[1]*fsample-1)]
            if t==0:
                X_RS = np.zeros((np.shape(tmp)[0], np.shape(tmp)[1], len(trial)))
            X_RS[:,:,t] = tmp
        X_RS_allPat.append({'name': name, 'X_RS': X_RS, 'lat': lat})
    return X_RS_allPat


def surrogate_sdi(scU,  Vlow, Vhigh, example_dir, nbSurr=1000, example=False): 
    X_RS_allPat = load_EEG_example(example_dir)
    #SDI_surr = np.zeros((114, 19, len(X_RS_allPat)))
    SDI_surr = np.zeros((118, 19, len(X_RS_allPat)))
    
    if example==True:
        SDI_surr = np.zeros((118, 19, len(X_RS_allPat)))

        with h5py.File(os.path.join(example_dir, 'PHI.mat'), 'r') as f:
            tmp = f['PHI'][()]
        #PHI = utils.extract_ctx_ROIs(tmp)
        PHI = tmp
        nbSurr = np.shape(PHI)[2]

        GSP2_surr = sio.loadmat(os.path.join(example_dir,'data_GSP2_surr.mat'))
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
        
        with h5py.File(os.path.join(example_dir, 'PHI.mat'), 'r') as f:
            tmp = f['PHI'][()]
        #PHI =extract_ctx_ROIs(tmp)
        PHI = tmp
        nbSurr = np.shape(PHI)[2]

        for s in np.arange(len(X_RS_allPat)):
            for n in np.arange(19):
                print('sub-%d, n%d'%(s,n))
                X_RS = X_RS_allPat[s]['X_RS']
                #idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59,116)))
                #X_RS = X_RS[idxs_tmp, :, :]
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
    SDI_surr = np.log2(SDI_surr)
    SDI = np.log2(SDI); SDI=np.squeeze(SDI)
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
    THRsubjects = 0
    
    # Final significance map per subject
    SDI_sig_subjectwise = np.zeros(SDI.shape)  # shape: (regions, subjects)
    SDI_sig_subjectwise[SDI_thr_max == 1] = 1
    SDI_sig_subjectwise[SDI_thr_min == 1] = -1

    surr_thresh = []
    for thr in range(THRsubjects, SDI.shape[1]+1):
        SDI_sig_higher = detect_max > thr
        SDI_sig_lower = detect_min > thr
        SDI_sig = np.zeros(mean_SDI.shape[0])
        SDI_sig[SDI_sig_higher] = 1
        SDI_sig[SDI_sig_lower] = -1
        surr_thresh.append({'threshold':thr, 'mean_SDI': mean_SDI, 'SDI_sig':SDI_sig})
    return surr_thresh, SDI_sig_subjectwise


def cons_normalized_lap(Mat, EucDist, plot=False):
    tmp = Mat
    diag_zeros = np.diag(np.diag(tmp))
    tmp = tmp - diag_zeros
    Ln, An  = normalize_Lap(tmp)
    sc = pygsp.graphs.Graph(tmp, lap_type='normalized', coords=EucDist)
    sc.compute_fourier_basis()
    P = sc.e
    Q = sc.U
    return P, Q, Ln, An

def rotation_procrustes(Q_all, P_all,  plot=False, p=''):
    if np.shape(Q_all)[2]>1:
        Q_all_rotated = np.zeros(np.shape(Q_all))
        Q_all_new = np.zeros(np.shape(Q_all))
        Q_all_rotated[:,:,0] = Q_all[:,:,0]
        R_all = np.zeros(np.shape(Q_all))
        scale_R = np.zeros(np.shape(Q_all)[2])
        Q_all[np.isnan(Q_all)]=0; Q_all[np.isinf(Q_all)]=0

        for i in range(1, np.shape(Q_all)[2]):
            Q_all_new[:,:,i], Q_all_rotated[:,:,i], disparity = scipy.spatial.procrustes(Q_all[:,:,0], Q_all[:,:,i])
        ### take the average of the rotated eigenvectors
        Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
        ###second round of Procrustes transformation
        P_all_rotated = np.zeros((np.shape(Q_all)[0], np.shape(Q_all)[2]))
        for i in range(1, np.shape(Q_all)[2]):
            Q_all[:,:,i], Q_all_rotated[:,:,i], disparity = scipy.spatial.procrustes(Q_mean_rotated, Q_all[:,:,i])
            P_all_rotated[:,i] = P_all[:,i]        
            Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
            P_mean = np.mean(P_all,axis=1); P_mean_rotated = np.mean(P_all_rotated, axis=1)


        if plot==True:
            fig, ax = plt.subplots(2,2, figsize=(10,3))            
            ax[0,0].imshow(Q_mean_rotated,  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[0,0].set_title('Average of rotated eigenvectors');  ax[0,0].set_aspect('equal')
            cax1 = ax[1,0].imshow(np.mean(Q_all, axis=2),  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[1,0].set_title('Average of original eigenvectors'); ax[1,0].set_aspect('equal')
            
            #gs = ax[0, 2].get_gridspec()
            #for a in [ax[0, 2], ax[1, 2]]:
            #    a.remove()
            #ax_big = fig.add_subplot(gs[:, 2])
            #ax_big.plot(range(np.shape(Q_all)[0]), P_mean, range(np.shape(Q_all)[0]), P_mean_rotated)
            #ax_big.set_title('Original and Rotated Eigenvectors '); ax_big.set_xlabel('eigenvalue index'); ax_big.set_ylabel('eigenvalues'); ax_big.legend(['Original Eigenvalues', 'Rotated Eigenvalues'])

        A = Q_all[:,:,0].T; B = Q_all[:,:,1].T; A_cos = np.dot(A, B.T)
        if plot==True:
            ax[0,1].imshow(A_cos,cmap = 'seismic',vmin = -1,vmax=1)
            ax[0,1].set_title('Cosine Similarity Before Rotation'); ax[0,1].set_xlabel('Subject 1 eigenvectors'), ax[0,1].set_ylabel('Subject 2 eigenvectors')
        
        A = Q_all_rotated[:,:,0].T; B = Q_all_rotated[:,:,1].T; A_cos = np.dot(A,B.T)
        if plot==True:
            ax[1,1].imshow(A_cos,cmap = 'seismic', vmin = -1,vmax=1); ax[1,1].set_title('Cosine Similarity After Rotation'); ax[1,1].set_xlabel('Subject 1 eigenvectors'); ax[1,1].set_ylabel('Subject 2 eigenvectors')        
            fig.suptitle('%s'%p); plt.show(block=False)
            plt.savefig('./public/static/images/RotationProcrustes%s.png'%p)
    
    else:
        Q_all_rotated = Q_all
        P_all_rotated = P_all
        R_all = 0; scale_R = 0
        print('Not Procrustes alignment performed')
    
    return Q_all_rotated, P_all_rotated, R_all, scale_R

def orthogonal_rotation_procrustes(Q_all, P_all,  plot=False, p=''):
    if np.shape(Q_all)[2]>1:
        Q_all_rotated = np.zeros(np.shape(Q_all))
        Q_all_new = np.zeros(np.shape(Q_all))
        Q_all_rotated[:,:,0] = Q_all[:,:,0]
        R_all = np.zeros(np.shape(Q_all))
        scale_R = np.zeros(np.shape(Q_all)[2])
        Q_all[np.isnan(Q_all)]=0; Q_all[np.isinf(Q_all)]=0

        for i in range(1, np.shape(Q_all)[2]):
            R_all[:,:,i], _ = scipy.linalg.orthogonal_procrustes(Q_all[:,:,0], Q_all[:,:,i])
            Q_all_rotated[:,:,i] = Q_all[:,:,i] @ R_all[:,:,i]    
        ### take the average of the rotated eigenvectors
        Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
        ###second round of Procrustes transformation
        P_all_rotated = np.zeros((np.shape(Q_all)[0], np.shape(Q_all)[2]))
        for i in range(1, np.shape(Q_all)[2]):
            R, _ = scipy.linalg.orthogonal_procrustes(Q_mean_rotated, Q_all[:,:,i])
            Q_all_rotated[:,:,i]  = Q_all[:,:,i] @ R
            #eig_rotated = R@Q_all[:,:,i]@np.diag(P_all[:,i])
            #P_all_rotated[:,i] = np.sqrt(np.sum(np.multiply(eig_rotated,eig_rotated),axis=0))
            P_all_rotated[:,i] = P_all[:,i]

        
            Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
            P_mean = np.mean(P_all,axis=1); P_mean_rotated = np.mean(P_all_rotated, axis=1)


        if plot==True:
            fig, ax = plt.subplots(2,2, figsize=(10,3))            
            ax[0,0].imshow(Q_mean_rotated,  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[0,0].set_title('Average of rotated eigenvectors');  ax[0,0].set_aspect('equal')
            cax1 = ax[1,0].imshow(np.mean(Q_all, axis=2),  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[1,0].set_title('Average of original eigenvectors'); ax[1,0].set_aspect('equal')
            
        A = Q_all[:,:,0].T; B = Q_all[:,:,1].T; A_cos = np.dot(A, B.T)
        if plot==True:
            ax[0,1].imshow(A_cos,cmap = 'seismic',vmin = -1,vmax=1)
            ax[0,1].set_title('Cosine Similarity Before Rotation'); ax[0,1].set_xlabel('Subject 1 eigenvectors'), ax[0,1].set_ylabel('Subject 2 eigenvectors')
        
        A = Q_all_rotated[:,:,0].T; B = Q_all_rotated[:,:,1].T; A_cos = np.dot(A,B.T)
        if plot==True:
            ax[1,1].imshow(A_cos,cmap = 'seismic', vmin = -1,vmax=1); ax[1,1].set_title('Cosine Similarity After Rotation'); ax[1,1].set_xlabel('Subject 1 eigenvectors'); ax[1,1].set_ylabel('Subject 2 eigenvectors')        
            fig.suptitle('%s'%p); plt.show(block=False)
            plt.savefig('./public/static/images/RotationProcrustes%s.png'%p)
    
    else:
        Q_all_rotated = Q_all
        P_all_rotated = P_all
        R_all = 0; scale_R = 0
        print('Not Procrustes alignment performed')
    
    return Q_all_rotated, P_all_rotated, R_all, scale_R

def reconstruct_SC(MatMat, df, P, Q, k=None, plot=False, p=''):
    Ln_group_recon = np.zeros(np.shape(MatMat))
    MatMat_recon = np.zeros(np.shape(MatMat))
    
    for s, sub in enumerate(list(df['sub'])):
        Qs = Q[:,:,s]
        Ps = P[:,s]
        
        if np.any(np.isnan(Qs)) or np.any(np.isinf(Qs)):
            print(f"NaN or Inf found in Q matrix slice {s}. Replacing with zeros.")
            Qs = np.nan_to_num(Qs)
        
        if k is not None:
            Qs = Qs[:, :k]  # Select the first k eigenvectors
            Ps = Ps[:k]     # Select the first k eigenvalues
        
        try:
            Q_pinv = np.linalg.pinv(Qs)
        except np.linalg.LinAlgError:
            print(f"SVD did not converge for slice {s}. Applying stronger regularization.")
            Q_pinv = np.linalg.pinv(Qs + np.eye(Qs.shape[0]) * 1e-8)
            if np.linalg.cond(Qs) > 1e10:  # Check the condition number
                print(f"Condition number is too high for slice {s}. Further regularization.")
                Q_pinv = np.linalg.pinv(Qs + np.eye(Qs.shape[0]) * 1e-6)
        
        Ln_group_recon[:,:,s] = Qs @ np.diag(Ps) @ Q_pinv
        MatMat_recon[:,:,s] = np.diag(np.full(len(MatMat[:,:,s]), 1)) - Ln_group_recon[:,:,s]
        MatMat_recon[:,:,s] = np.diag(np.diag(Ln_group_recon[:,:,s])) - Ln_group_recon[:,:,s]
    
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(9, 4))
        im1 = axs[0].imshow(MatMat_recon[:,:,0])
        axs[0].set_title('Reconstructed Normalized SC - Subject 1')
        im2 = axs[1].imshow(MatMat[:,:,0])
        axs[1].set_title('Raw Normalized SC - Subject 1')
        im3 = axs[2].scatter(MatMat[:,:,0], MatMat_recon[:,:,0])
        axs[2].set_title('Pearson correlation - Subject 1'); axs[2].set_xlabel('SC'); axs[2].set_ylabel('Reconstructed SC');
        fig.suptitle('%s' % p)
        #plt.savefig('./public/static/images/reconstruct_SC_proc%s.png'%title)
        plt.savefig('./public/static/images/reconstruct_SC_proc%s.png'%p)
        plt.show(block=False)

    return MatMat_recon


def reconstruct_SC_part(MatMat, nbEig, df, P, Q, plot=False):
    Q = Q[:, :nbEig, :]; P = P[:nbEig,:]
    Ln_group_recon = np.zeros(np.shape(MatMat)); MatMat_recon = np.zeros(np.shape(MatMat))
    for s,sub in enumerate(list(df['sub'])):
        #Ln_group_recon[:,:,s] = Q[:,:,s]@np.diag(P[:,s])@np.linalg.inv(Q[:,:,s])
        Ln_group_recon[:,:,s] = Q[:,:,s]@np.diag(P[:,s])@np.linalg.pinv(Q[:,:,s]) ## use pseudo inverse because the matrix is not square anymore
        MatMat_recon[:,:,s] = np.diag(np.full(len(MatMat[:,:,s]),1)) - Ln_group_recon[:,:,s]
        MatMat_recon[:,:,s] = np.diag(np.diag(Ln_group_recon[:,:,s])) - Ln_group_recon[:,:,s]
    if plot==True:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        im1 = axs[0].imshow(MatMat_recon[:,:,1]); axs[0].set_title(' Reconstructed Normalized SC - Subject 1')
        im2 = axs[1].imshow(MatMat[:,:,1]); axs[1].set_title('Raw Normalized SC - Subject 1')
    plt.show(block=False)       

    return MatMat_recon

def match_eigenvectors(A, B, metric='cosine'):
    """
    Match columns of A to columns of B using minimal pairwise cost.

    Parameters:
    A, B (np.ndarray): Matrices of shape (n, n), e.g. eigenvectors
    metric (str): Distance metric to use ('cosine' or 'euclidean')

    Returns:
    perm (np.ndarray): Permutation indices for B to match A (i.e., B[:, perm] aligns with A)
    """
    n = A.shape[1]
    cost = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if metric == 'cosine':
                cost[i, j] = 1 - np.abs(np.dot(A[:, i], B[:, j])) / (
                    np.linalg.norm(A[:, i]) * np.linalg.norm(B[:, j])
                )
            elif metric == 'euclidean':
                cost[i, j] = np.linalg.norm(A[:, i] - B[:, j])
            else:
                raise ValueError("Unsupported metric")

    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind, cost[row_ind, col_ind].sum()


def normalize_columns(X):
    return X / np.linalg.norm(X, axis=0, keepdims=True)

def cosine_similarity_matrix(H1, H2):
    H1_norm = normalize_columns(H1)
    H2_norm = normalize_columns(H2)
    return H1_norm.T @ H2_norm  # shape: (k, k)

def hungarian_aligned_cosine_similarity(H1, H2):
    sim_matrix = cosine_similarity_matrix(H1, H2)
    # Convert to cost for Hungarian (minimization)
    cost_matrix = -sim_matrix
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Extract aligned similarities
    aligned_similarities = sim_matrix[row_ind, col_ind]
    # Create an aligned similarity matrix
    aligned_sim_matrix = np.zeros_like(sim_matrix)
    for i, j in zip(row_ind, col_ind):
        aligned_sim_matrix[i, j] = sim_matrix[i, j]
    return aligned_sim_matrix, aligned_similarities, (row_ind, col_ind)





def orthogonal_rotation_procrustes(Q_all, P_all,  plot=False, p=''):
    if np.shape(Q_all)[2]>1:
        Q_all_rotated = np.zeros(np.shape(Q_all))
        Q_all_new = np.zeros(np.shape(Q_all))
        Q_all_rotated[:,:,0] = Q_all[:,:,0]
        R_all = np.zeros(np.shape(Q_all))
        scale_R = np.zeros(np.shape(Q_all)[2])
        Q_all[np.isnan(Q_all)]=0; Q_all[np.isinf(Q_all)]=0

        for i in range(1, np.shape(Q_all)[2]):
            R_all[:,:,i], _ = scipy.linalg.orthogonal_procrustes(Q_all[:,:,0], Q_all[:,:,i])
            Q_all_rotated[:,:,i] = Q_all[:,:,i] @ R_all[:,:,i]    
        ### take the average of the rotated eigenvectors
        Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
        ###second round of Procrustes transformation
        P_all_rotated = np.zeros((np.shape(Q_all)[0], np.shape(Q_all)[2]))
        for i in range(1, np.shape(Q_all)[2]):
            R, _ = scipy.linalg.orthogonal_procrustes(Q_mean_rotated, Q_all[:,:,i])
            Q_all_rotated[:,:,i]  = Q_all[:,:,i] @ R
            #eig_rotated = R@Q_all[:,:,i]@np.diag(P_all[:,i])
            #P_all_rotated[:,i] = np.sqrt(np.sum(np.multiply(eig_rotated,eig_rotated),axis=0))
            P_all_rotated[:,i] = P_all[:,i]

        
            Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
            P_mean = np.mean(P_all,axis=1); P_mean_rotated = np.mean(P_all_rotated, axis=1)


        if plot==True:
            fig, ax = plt.subplots(2,2, figsize=(10,3))            
            ax[0,0].imshow(Q_mean_rotated,  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[0,0].set_title('Average of rotated eigenvectors');  ax[0,0].set_aspect('equal')
            cax1 = ax[1,0].imshow(np.mean(Q_all, axis=2),  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[1,0].set_title('Average of original eigenvectors'); ax[1,0].set_aspect('equal')
            
        A = Q_all[:,:,0].T; B = Q_all[:,:,1].T; A_cos = np.dot(A, B.T)
        if plot==True:
            ax[0,1].imshow(A_cos,cmap = 'seismic',vmin = -1,vmax=1)
            ax[0,1].set_title('Cosine Similarity Before Rotation'); ax[0,1].set_xlabel('Subject 1 eigenvectors'), ax[0,1].set_ylabel('Subject 2 eigenvectors')
        
        A = Q_all_rotated[:,:,0].T; B = Q_all_rotated[:,:,1].T; A_cos = np.dot(A,B.T)
        #if plot==True:
        #    ax[1,1].imshow(A_cos,cmap = 'seismic', vmin = -1,vmax=1); ax[1,1].set_title('Cosine Similarity After Rotation'); ax[1,1].set_xlabel('Subject 1 eigenvectors'); ax[1,1].set_ylabel('Subject 2 eigenvectors')        
        #    fig.suptitle('%s'%p); plt.show(block=False)
        #    plt.savefig('./public/static/images/RotationProcrustes%s.png'%p)
    
    else:
        Q_all_rotated = Q_all
        P_all_rotated = P_all
        R_all = 0; scale_R = 0
        print('Not Procrustes alignment performed')
    
    return Q_all_rotated, P_all_rotated, R_all, scale_R


def rotation_procrustes(Q_all, P_all,  plot=False, p=''):
    if np.shape(Q_all)[2]>1:
        Q_all_rotated = np.zeros(np.shape(Q_all))
        Q_all_new = np.zeros(np.shape(Q_all))
        Q_all_norm = np.zeros(np.shape(Q_all))
        Q_all_rotated[:,:,0] = Q_all[:,:,0]
        R_all = np.zeros(np.shape(Q_all))
        scale_R = np.zeros(np.shape(Q_all)[2])
        Q_all[np.isnan(Q_all)]=0; Q_all[np.isinf(Q_all)]=0

        for i in range(1, np.shape(Q_all)[2]):
            Q_all_new[:,:,i], Q_all_rotated[:,:,i], disparity = scipy.spatial.procrustes(Q_all[:,:,0], Q_all[:,:,i])
        ### take the average of the rotated eigenvectors
        Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
        ###second round of Procrustes transformation
        P_all_rotated = np.zeros((np.shape(Q_all)[0], np.shape(Q_all)[2]))
        for i in range(1, np.shape(Q_all)[2]):
            Q_all_norm[:,:,i], Q_all_rotated[:,:,i], disparity = scipy.spatial.procrustes(Q_mean_rotated, Q_all[:,:,i])
            P_all_rotated[:,i] = P_all[:,i]        
            Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
            P_mean = np.mean(P_all,axis=1); P_mean_rotated = np.mean(P_all_rotated, axis=1)


        if plot==True:
            fig, ax = plt.subplots(2,2, figsize=(10,3))            
            ax[0,0].imshow(Q_mean_rotated,  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[0,0].set_title('Average of rotated eigenvectors');  ax[0,0].set_aspect('equal')
            cax1 = ax[1,0].imshow(np.mean(Q_all, axis=2),  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[1,0].set_title('Average of original eigenvectors'); ax[1,0].set_aspect('equal')
            
        A = Q_all[:,:,0].T; B = Q_all[:,:,1].T; A_cos = np.dot(A, B.T)
        if plot==True:
            ax[0,1].imshow(A_cos,cmap = 'seismic',vmin = -1,vmax=1)
            ax[0,1].set_title('Cosine Similarity Before Rotation'); ax[0,1].set_xlabel('Subject 1 eigenvectors'), ax[0,1].set_ylabel('Subject 2 eigenvectors')
        
        A = Q_all_rotated[:,:,0].T; B = Q_all_rotated[:,:,1].T; A_cos = np.dot(A,B.T)
        if plot==True:
            ax[1,1].imshow(A_cos,cmap = 'seismic', vmin = -1,vmax=1); ax[1,1].set_title('Cosine Similarity After Rotation'); ax[1,1].set_xlabel('Subject 1 eigenvectors'); ax[1,1].set_ylabel('Subject 2 eigenvectors')        
            fig.suptitle('%s'%p); plt.show(block=False)
            plt.savefig('./public/static/images/RotationProcrustes%s.png'%p)
    
    else:
        Q_all_rotated = Q_all
        P_all_rotated = P_all
        R_all = 0; scale_R = 0
        print('Not Procrustes alignment performed')
    
    return Q_all_rotated, P_all_rotated, R_all, scale_R
