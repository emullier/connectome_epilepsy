
import os
import numpy as np
import pygsp
import scipy
import scipy.io as sio
from scipy.stats import binom
import h5py
from lib.func_plot import plot_rois_pyvista_noaxes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

def normalize_Lap(A):
    ''' Takes the adjacency matrix as input and returns the corresponding symmetric normalized Laplacian matrix'''
    indices_diag = np.diag_indices(len(A))
    A[indices_diag] = 0
    D = np.sum(A,axis=1)
    epsilon = 1e-10
    D = np.where(D == 0, epsilon, D)
    D = np.diag(D)
    #Dn = np.power(D, -0.5)
    #Dn = np.diag(np.diag(Dn))
    Dn = np.power(D, -0.5, where=D>0)
    Dn[D == 0] = 0
    # symmetric normalize Adjacency
    An = Dn@A@Dn
    Ln = np.diag(np.full(len(An),1)) - An
    # Ln = np.diag(np.sum(An,axis=1)) - An
    return Ln, An

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
    AUCTOT = np.trapezoid(mPSD[:sc.shape[0]]) ##total area under the curve  
    i=1; AUC=0;
    while AUC<AUCTOT/2:
        i=i+1; AUC=np.trapezoid(mPSD[0:i])
    NN=i-1; #CUTOFF FREQUENCY : number of low frequency eigenvalues to consider in order to have the same energy as the high freq ones
    ## split structural harmonics in high/low frequency
    Vlow=np.zeros(np.shape(sc)); Vhigh=np.zeros(np.shape(sc))
    Vhigh[:,NN:]=sc[:,NN:] # high frequencies= decoupled
    Vlow[:,:NN]=sc[:,:NN] #low frequencies = coupled
    
    return PSD,NN,Vlow, Vhigh



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




def compute_group_sdi(label, consensus, EucDist, lateralization,example_dir, output_dir, figures_dir,nbSurr=100, scale=2, surr_key=None):
    """
    Build GSP harmonics from a consensus SC matrix, estimate SDI per patient,
    run surrogate-based significance testing, and save/plot the results.

    Parameters
    ----------
    label : str
        Short group name used in output filenames and figure labels
        (e.g. 'HC', 'EP', 'IND').
    consensus : ndarray (ROI, ROI)
        Group consensus structural connectivity matrix.
    EucDist : ndarray (ROI, ROI)
        Euclidean distance matrix used to build the harmonics.
    lateralization : {'LT', 'RT'}
        Which TLE lateralization to keep for SDI estimation.
    surr_key : str, optional
        Filename stem for the cached surrogate SDI array.
        Defaults to f'SDI_surr_{label}_{lateralization}'.

    Returns
    -------
    surr_thresh : ndarray
        Per-threshold surrogate significance results.
    """
    print(f"\nProcessing {label} {lateralization}...")

    # Harmonics from the consensus matrix
    P, Q, Ln, An = cons_normalized_lap(consensus, EucDist, plot=False)

    # Project EEG onto the harmonics, estimate SDI + cutoff per patient
    X_RS_allPat = load_EEG_example(example_dir)
    ls_cutoff, ls_lat = [], []
    SDI_tmp = np.zeros((118, len(X_RS_allPat)))
    for p, patient in enumerate(X_RS_allPat):
        X_RS = patient['X_RS']
        ls_lat.append(patient['lat'][0])
        PSD, NN, Vlow, Vhigh = get_cutoff_freq(Q, X_RS)
        ls_cutoff.append(NN)
        SDI_tmp[:, p], X_c_norm, X_d_norm, SD_hat = compute_SDI(X_RS, Q)
    np.save(os.path.join(output_dir, f'cutoff_{label}_{lateralization}.npy'), ls_cutoff)

    # Keep only patients with the requested lateralization
    ls_lat = np.array(ls_lat)
    lat_key = 'Rtle' if lateralization == 'RT' else 'Ltle'
    idxs_lat = np.where(ls_lat == lat_key)[0]
    SDI = SDI_tmp[:, idxs_lat]
    np.save(os.path.join(output_dir, f'SDI_{label}_{lateralization}.npy'), SDI)

    # Surrogate-based significance testing (cached to disk)
    surr_key = surr_key or f'SDI_surr_{label}_{lateralization}'
    surr_path = os.path.join(output_dir, f'{surr_key}.npy')
    if os.path.exists(surr_path):
        SDI_surr = np.load(surr_path)
        print('Surrogate SDI already generated')
    else:
        SDI_surr = surrogate_sdi(Q, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False)
        np.save(surr_path, SDI_surr)

    surr_thresh, SDI_sig_subjectwise = select_significant_sdi(SDI, SDI_surr[:, :, idxs_lat])
    np.save(os.path.join(output_dir, f'SDI_surr_thresh_{label}_{lateralization}.npy'),
            surr_thresh, allow_pickle=True)

    # Count significant ROIs per threshold
    nbROIs_sig = [len(np.where(np.abs(surr_thresh[t]['SDI_sig']))[0])
                  for t in range(np.shape(surr_thresh)[0])]
    np.save(os.path.join(output_dir, f'nbROIs_sig_{label}_{lateralization}.npy'), nbROIs_sig)

    # Brain plots: uncorrected (thr=0) and manuscript threshold (thr=5)
    plot_rois_pyvista_noaxes(surr_thresh[0]['mean_SDI'], scale, figures_dir, vmin=-2, vmax=2,label=f'Fig2_SDImean_{label}_{lateralization}')
    plot_rois_pyvista_noaxes(surr_thresh[5]['mean_SDI'] * np.abs(surr_thresh[5]['SDI_sig']),scale, figures_dir, vmin=-2, vmax=2,label=f'Fig3_SDImean_thr5_{label}_{lateralization}')

    return surr_thresh


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


#### CACHES FUNCTION FOR FIG5
################################

# =====================================================
# GLOBAL CACHE HELPERS
# =====================================================
def load_or_compute(path, compute_fn, compress=False):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    result = compute_fn()
    if compress:
        np.savez_compressed(path, data=result)
    else:
        np.save(path, result)
    return result

def load_npz_dict(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k].tolist() for k in data}

# ============================================================================
# HELPERS
# ============================================================================
def align_all(Q_ref, Q):
    Q_rot, _, _ = procrustes(Q_ref, Q)
    U, _, Vt = scipy.linalg.svd(Q_rot, full_matrices=False)
    Q_rot = U @ Vt

    perm, _ = match_eigenvectors(Q_ref, Q)
    Q_match = Q[:, perm]

    return {"raw": Q, "rotated": Q_rot, "matched": Q_match}

def harmonic_similarity(Q_ref, Q):
    return np.abs(np.diag(Q_ref.T @ Q))

# ============================================================================
# CACHE WRAPPERS
# ============================================================================
def get_permutations(label, idxs, max_bin, nbPerm, OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, f"perms_{label}.npy")

    def compute():
        return np.array([
            np.random.choice(idxs, max_bin, replace=False)
            for _ in range(nbPerm)
        ])
    return load_or_compute(path, compute)

def get_Q(label, bi, p, SC, perm_idxs, Euc, OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, f"Q_{label}_bin{bi}_perm{p}.npy")

    def compute():
        SC_sub = np.mean(SC[:, :, perm_idxs], axis=2)
        _, Q, _, _ = cons_normalized_lap(SC_sub, Euc, plot=False)
        return Q

    return load_or_compute(path, compute)

def get_alignments(label, bi, p, Q_ref, Q, OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, f"ALIGN_{label}_bin{bi}_perm{p}.npz")

    if os.path.exists(path):
        data = np.load(path)
        return {"raw": data["raw"], "rotated": data["rotated"],"matched": data["matched"]}

    Qs = align_all(Q_ref, Q)
    np.savez(path,raw=Qs["raw"],rotated=Qs["rotated"],matched=Qs["matched"])

    return Qs

def get_SDI(label, bi, p, method_key, Qm, X_RS_allPat, OUTPUT_DIR):
    fname = f"SDI_{label}_bin{bi}_perm{p}_{method_key}.npy"
    path = os.path.join(OUTPUT_DIR, fname)

    def compute():
        return np.column_stack([compute_SDI(pat['X_RS'], Qm)[0] for pat in X_RS_allPat])

    return load_or_compute(path, compute)
