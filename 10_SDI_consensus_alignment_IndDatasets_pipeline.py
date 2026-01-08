
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.func_GSP as gsp
from lib.func_plot import plot_rois, plot_rois_pyvista, plot_rois_pyvista_noaxes
import scipy.io as sio
import scipy

# Loop through both lateralizations
ls_lateralization = ['LT', 'RT']

# Collect per-lateralization tables (thr=5) to merge at the end
combined_tables = {}

for lateralization in ls_lateralization:
    print(f"\n{'='*80}")
    print(f"Processing lateralization: {lateralization}")
    print(f"{'='*80}")
    
    data_path = "DATA/Connectome_scale-2.mat"
    matMetric = sio.loadmat(data_path)
    matMetric = matMetric['num']
    cort_rois = np.concatenate((np.arange(0,57), [62,63], np.arange(64,121), [126,127]))
    matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
    consensus_ind = matMetric
    consensus_HC_DSI = np.load("DATA/SC/matMetric_HC_DSI_number_of_fibers.npy")
    consensus_schz = np.load("DATA/SC/matMetric_SCHZ_CTRL.npy")
    example_dir = "./DATA/EEG"
    scale = 2

    #Qind_ref = consensus_ind
    consensus_HC_ref = np.mean(consensus_HC_DSI, axis=2)
    ref = 'HCdsi'
    consensus_schz = np.mean(consensus_schz,axis=0)

    #np.fill_diagonal(consensus, 0)
    #EucDist = consensus #### To be replaced by proper Euclidean matrix
    #EucDist = (EucDist + EucDist.T)/2
    EucDist = np.load("DATA/EucMat/EucMat_HC_dsi_number_of_fibers.npy")


    print("Generate harmonics from the consensus")
    ### Generate the harmonics
    P_ref, Q_ref, Ln_ref, An_ref = gsp.cons_normalized_lap(consensus_HC_ref, EucDist,  plot=False)
    P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus_schz, EucDist,  plot=False)
    Qind_rotated, Qind_HC_RT_centered, disparity_RT = scipy.spatial.procrustes(Q_ref, Q_ind)
    R_RT, _ = scipy.linalg.orthogonal_procrustes(Q_ref, Q_ind)
    Qind_ortho_rotated=Q_ind@R_RT 
    perm, total_cost = gsp.match_eigenvectors(Q_ref, Q_ind)
    Qind_matched = Q_ind[:,perm]

    ### Compute similarity between harmonics (using correlation-based distance as in script 12)
    nb_eig = Q_ref.shape[1]
    similarity_ind = np.zeros(nb_eig)
    similarity_rotated = np.zeros(nb_eig)
    similarity_ortho = np.zeros(nb_eig)
    similarity_matched = np.zeros(nb_eig)
    
    for eigvec_nb in range(nb_eig):
        # Compute correlation-based similarity (1 - correlation distance)
        # Higher values indicate greater similarity
        similarity_ind[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Q_ind[:, eigvec_nb])
        similarity_rotated[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_rotated[:, eigvec_nb])
        similarity_ortho[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_ortho_rotated[:, eigvec_nb])
        similarity_matched[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_matched[:, eigvec_nb])
    
    # Take absolute values to compensate for sign changes
    similarity_ind = np.abs(similarity_ind)
    similarity_rotated = np.abs(similarity_rotated)
    similarity_ortho = np.abs(similarity_ortho)
    similarity_matched = np.abs(similarity_matched)
    
    print(f"\nHarmonic Similarity ({lateralization}):")
    print(f"  SC IND vs SC HC:           Mean={np.mean(similarity_ind):.4f}, Median={np.median(similarity_ind):.4f}")
    print(f"  Gen. Procrustes vs SC HC:  Mean={np.mean(similarity_rotated):.4f}, Median={np.median(similarity_rotated):.4f}")
    print(f"  Ortho. Procrustes vs SC HC: Mean={np.mean(similarity_ortho):.4f}, Median={np.median(similarity_ortho):.4f}")
    print(f"  Hungarian vs SC HC:        Mean={np.mean(similarity_matched):.4f}, Median={np.median(similarity_matched):.4f}")

    if not os.path.exists('./OUTPUT/ALIGN/'):
        os.makedirs('./OUTPUT/ALIGN/')

    np.save('./OUTPUT/ALIGN/Q_ref_HC_%s_ref%s.npy'%( lateralization, ref), Q_ref)
    np.save('./OUTPUT/INDvsCTRL/Q_ind_schz_%s.npy'%( lateralization), Q_ind)
    np.save('./OUTPUT/ALIGN/P_ind_schz_%s_ref%s.npy'%(lateralization, ref), P_ind)
    np.save('./OUTPUT/ALIGN/Q_ind_rotated_schz_%s_ref%s.npy'%( lateralization, ref), Qind_rotated)
    np.save('./OUTPUT/ALIGN/Q_ind_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref), Qind_ortho_rotated) 
    np.save('./OUTPUT/ALIGN/Q_ind_matched_schz_%s_ref%s.npy'%( lateralization, ref), Qind_matched)
    
    # Save similarity metrics
    np.save('./OUTPUT/ALIGN/similarity_ind_%s_ref%s.npy'%( lateralization, ref), similarity_ind)
    np.save('./OUTPUT/ALIGN/similarity_rotated_%s_ref%s.npy'%( lateralization, ref), similarity_rotated)
    np.save('./OUTPUT/ALIGN/similarity_ortho_%s_ref%s.npy'%( lateralization, ref), similarity_ortho)
    np.save('./OUTPUT/ALIGN/similarity_matched_%s_ref%s.npy'%( lateralization, ref), similarity_matched)

    ### Project the functional signals
    ########################################
    print("Load EEG example data for SDI")
    X_RS_allPat = gsp.load_EEG_example(example_dir)

    ### Estimate SDI
    ls_cutoff = []
    ls_cutoff_ref = []
    ls_cutoff_rotated = []
    ls_cutoff_ortho_rotated = []    
    ls_cutoff_matched = []
    SDI_tmp = np.zeros((118, len(X_RS_allPat)))
    SDI_tmp_ref = np.zeros((118, len(X_RS_allPat)))
    SDI_tmp_rotated = np.zeros((118, len(X_RS_allPat)))
    SDI_tmp_ortho_rotated = np.zeros((118, len(X_RS_allPat)))
    SDI_tmp_matched = np.zeros((118, len(X_RS_allPat)))
    ls_lat = []; SDI={}; SDI_surr={}
    cutoff_path_ref = './OUTPUT/EPvsCTRL/cutoff_number_of_fibers_HC_dsi.npy'
    cutoff_path= './OUTPUT/ALIGN/cutoff_ref_schz_ref%s.npy'%ref
    cutoff_path_rotated= './OUTPUT/ALIGN/cutoff_rotated_schz_ref%s.npy'%ref
    cutoff_path_ortho_rotated= './OUTPUT/ALIGN/cutoff_ortho_rotated_schz_ref%s.npy'%ref
    cutoff_path_matched= './OUTPUT/ALIGN/cutoff_matched_schz_ref%s.npy'%ref 
    for p in np.arange(len(X_RS_allPat)):
        X_RS = X_RS_allPat[p]['X_RS']
        ls_lat.append(X_RS_allPat[p]['lat'][0])
        PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, X_RS); ls_cutoff.append(NN)
        SDI_tmp[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind)
        PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ref, X_RS); ls_cutoff_ref.append(NN)
        SDI_tmp_ref[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ref)
        PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_rotated, X_RS); ls_cutoff_rotated.append(NN)
        SDI_tmp_rotated[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Qind_rotated)
        PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_ortho_rotated, X_RS); ls_cutoff_ortho_rotated.append(NN)
        SDI_tmp_ortho_rotated[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Qind_ortho_rotated)
        PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Qind_matched, X_RS); ls_cutoff_matched.append(NN)
        SDI_tmp_matched[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Qind_matched)
    
    np.save(cutoff_path_ref, ls_cutoff_ref)
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
    SDI_ref = SDI_tmp_ref[:, idxs_lat]; SDI_ref_path = './OUTPUT/EPvsCTRL/SDI_sig_subjectwise_number_of_fibers_HC_%s.npy'%( lateralization); np.save(SDI_ref_path, SDI_ref)
    SDI_rotated = SDI_tmp_rotated[:, idxs_lat]; SDI_rotated_path = './OUTPUT/ALIGN/SDI_rotated_schz_%s_ref%s.npy'%( lateralization, ref); np.save(SDI_rotated_path, SDI_rotated)
    SDI_ortho_rotated = SDI_tmp_ortho_rotated[:, idxs_lat]; SDI_ortho_rotated_path = './OUTPUT/ALIGN/SDI_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref); np.save(SDI_ortho_rotated_path, SDI_ortho_rotated)
    SDI_matched = SDI_tmp_matched[:, idxs_lat]; SDI_matched_path = './OUTPUT/ALIGN/SDI_matched_schz_%s_ref%s.npy'%( lateralization, ref); np.save(SDI_matched_path, SDI_matched)

    if not os.path.exists("./FIGURES/ALIGN/"):
        os.makedirs("./FIGURES/ALIGN/")

    plot_rois_pyvista_noaxes(np.mean(SDI,axis=1), scale, './FIGURES/ALIGN', label='SDImean_schz_%s'%( lateralization), vmin=-2, vmax=2)
    plot_rois_pyvista(np.mean(SDI_ref,axis=1), scale, './FIGURES/ALIGN', vmin=-2, vmax=2, label='SDImean_HC_%s'%( lateralization))
    plot_rois_pyvista(np.mean(SDI_rotated,axis=1), scale, './FIGURES/ALIGN', vmin=-2, vmax=2, label='SDImean_rotated_schz_%s_ref%s'%( lateralization, ref))
    plot_rois_pyvista(np.mean(SDI_ortho_rotated,axis=1), scale, './FIGURES/ALIGN', vmin=-2, vmax=2, label='SDImean_ortho_rotated_schz_%s_ref%s'%( lateralization, ref))  
    plot_rois_pyvista(np.mean(SDI_matched,axis=1), scale, './FIGURES/ALIGN', vmin=-2, vmax=2, label='SDImean_matched_schz_%s_ref%s'%( lateralization, ref))  

    ### Surrogate part
    nbSurr = 100
    surr_path_ref = './OUTPUT/EPvsCTRL/SDI_surr_number_of_fibers_HC_dsi_%s.npy'%( lateralization)
    surr_path = './OUTPUT/INDvsCTRL/SDI_surr_schz_%s.npy'%( lateralization)
    surr_path_rotated = './OUTPUT/ALIGN/SDI_surr_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
    surr_path_ortho_rotated = './OUTPUT/ALIGN/SDI_surr_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
    surr_path_matched = './OUTPUT/ALIGN/SDI_surr_matched_schz_%s_ref%s.npy'%( lateralization, ref)
    if not os.path.exists(surr_path_matched):
        SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate 
        np.save(surr_path, SDI_surr) # Save the surrogate
        SDI_surr_ref = gsp.surrogate_sdi(Q_ref, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate
        np.save(surr_path_ref, SDI_surr_ref) # Save the surrogate
        SDI_surr_rotated = gsp.surrogate_sdi(Qind_rotated, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate
        np.save(surr_path_rotated, SDI_surr_rotated) # Save the surrogate
        SDI_surr_ortho_rotated = gsp.surrogate_sdi(Qind_ortho_rotated, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate
        np.save(surr_path_ortho_rotated, SDI_surr_ortho_rotated) # Save the surrogate
        SDI_surr_matched = gsp.surrogate_sdi(Qind_matched, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False) # Generate the surrogate
        np.save(surr_path_matched, SDI_surr_matched) # Save the surrogate
    else:   
        print('Surrogate SDI already generated')
        SDI_surr = np.load(surr_path)
        SDI_surr_ref = np.load(surr_path_ref)
        SDI_surr_rotated = np.load(surr_path_rotated)
        SDI_surr_ortho_rotated = np.load(surr_path_ortho_rotated)           
        SDI_surr_matched = np.load(surr_path_matched)


    idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
    surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
    surr_thresh_ref, SDI_sig_subjectwise_ref = gsp.select_significant_sdi(SDI_ref, SDI_surr_ref[:,:,idxs_lat])
    surr_thresh_rotated, SDI_sig_subjectwise_rotated = gsp.select_significant_sdi(SDI_rotated, SDI_surr_rotated[:,:,idxs_lat])
    surr_thresh_ortho_rotated, SDI_sig_subjectwise_ortho_rotated = gsp.select_significant_sdi(SDI_ortho_rotated, SDI_surr_ortho_rotated[:,:,idxs_lat])  
    surr_thresh_matched, SDI_sig_subjectwise_matched = gsp.select_significant_sdi(SDI_matched, SDI_surr_matched[:,:,idxs_lat])
    surr_thresh_path = './OUTPUT/EPvsCTRL/SDI_surr_thresh_number_of_fibers_HC_%s.npy'%( lateralization)
    surr_thresh_ref_path = './OUTPUT/EPvsCTRL/SDI_surr_thresh_number_of_fibers_HC_dsi_%s.npy'%( lateralization)
    surr_thresh_rotated_path = './OUTPUT/ALIGN/SDI_surr_thresh_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
    surr_thresh_ortho_rotated_path = './OUTPUT/ALIGN/SDI_surr_thresh_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
    surr_thresh_matched_path = './OUTPUT/ALIGN/SDI_surr_thresh_matched_schz_%s_ref%s.npy'%( lateralization, ref)
    surr_sig_subjectwise_path = './OUTPUT/INDvsCTRL/SDI_sig_subjectwise_schz_%s.npy'%( lateralization)
    surr_sig_subjectwise_ref_path = './OUTPUT/EPvsCTRL/SDI_sig_subjectwise_number_of_fibers_HC_dsi_%s.npy'%( lateralization)
    surr_sig_subjectwise_rotated_path = './OUTPUT/ALIGN/SDI_sig_subjectwise_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
    surr_sig_subjectwise_ortho_rotated_path = './OUTPUT/ALIGN/SDI_sig_subjectwise_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref)
    surr_sig_subjectwise_matched_path = './OUTPUT/ALIGN/SDI_sig_subjectwise_matched_schz_%s_ref%s.npy'%( lateralization, ref)
    np.save(surr_thresh_path, surr_thresh, allow_pickle=True) # Save the surrogate
    np.save(surr_sig_subjectwise_path, SDI_sig_subjectwise, allow_pickle=True) # Save the surrogate
    np.save(surr_thresh_ref_path, surr_thresh_ref, allow_pickle=True) # Save the surrogate
    np.save(surr_sig_subjectwise_ref_path, SDI_sig_subjectwise_ref, allow_pickle=True) # Save the surrogate
    np.save(surr_thresh_rotated_path, surr_thresh_rotated, allow_pickle=True) # Save the surrogate
    np.save(surr_sig_subjectwise_rotated_path, SDI_sig_subjectwise_rotated, allow_pickle=True) # Save the surrogate
    np.save(surr_thresh_ortho_rotated_path, surr_thresh_ortho_rotated, allow_pickle=True) # Save the surrogate
    np.save(surr_sig_subjectwise_ortho_rotated_path, SDI_sig_subjectwise_ortho_rotated, allow_pickle=True) # Save the surrogate
    np.save(surr_thresh_matched_path, surr_thresh_matched, allow_pickle=True) # Save the surrogate
    np.save(surr_sig_subjectwise_matched_path, SDI_sig_subjectwise_matched, allow_pickle=True) # Save the surrogate

    nbROIs_sig = []
    nvROIs_sig_ref = []
    nbROIs_sig_rotated = []
    nbROIs_sig_ortho_rotated = []
    nbROIs_sig_matched = []
    
    for p in np.arange(np.shape(surr_thresh)[0]):
        nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
        nvROIs_sig_ref.append(len(np.where(np.abs(surr_thresh_ref[p]['SDI_sig']))[0]))
        nbROIs_sig_rotated.append(len(np.where(np.abs(surr_thresh_rotated[p]['SDI_sig']))[0]))
        nbROIs_sig_ortho_rotated.append(len(np.where(np.abs(surr_thresh_ortho_rotated[p]['SDI_sig']))[0]))
        nbROIs_sig_matched.append(len(np.where(np.abs(surr_thresh_matched[p]['SDI_sig']))[0]))
    np.save('./OUTPUT/EPvsCTRL/nbROIs_sig_number_of_fibers_HC_%s.npy'%( lateralization), nbROIs_sig)
    np.save('./OUTPUT/EPvsCTRL/nbROIs_sig_ref_number_of_fibers_HC_dsi_%s.npy'%( lateralization), nvROIs_sig_ref)
    np.save('./OUTPUT/ALIGN/nbROIs_sig_rotated_schz_%s_ref%s.npy'%(lateralization, ref), nbROIs_sig_rotated)
    np.save('./OUTPUT/ALIGN/nbROIs_sig_ortho_rotated_schz_%s_ref%s.npy'%( lateralization, ref), nbROIs_sig_ortho_rotated)
    np.save('./OUTPUT/ALIGN/nbROIs_sig_matched_schz_%s_ref%s.npy'%( lateralization, ref), nbROIs_sig_matched)



    thr = 2
    plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), scale, './FIGURES/EPvsCTRL', label='SDImean_thr%d_number_of_fibers_HC_dsi_%s'%(thr, lateralization), vmin=-2, vmax=2)
    plot_rois_pyvista_noaxes(surr_thresh_ref[thr]['mean_SDI']*np.abs(surr_thresh_ref[thr]['SDI_sig']), scale, './FIGURES/EPvsCTRL', label='SDImean_thr%d_number_of_fibers_HC_dsi_%s_ref'%(thr, lateralization), vmin=-2, vmax=2)
    plot_rois_pyvista_noaxes(surr_thresh_rotated[thr]['mean_SDI']*np.abs(surr_thresh_rotated[thr]['SDI_sig']), scale, './FIGURES/ALIGN', label='SDImean_rotated_thr%d_schz_%s_ref%s'%(thr,  lateralization, ref), vmin=-2, vmax=2)
    plot_rois_pyvista_noaxes(surr_thresh_ortho_rotated[thr]['mean_SDI']*np.abs(surr_thresh_ortho_rotated[thr]['SDI_sig']), scale, './FIGURES/ALIGN', label='SDImean_ortho_rotated_thr%d_schz_%s_ref%s'%(thr, lateralization, ref), vmin=-2, vmax=2)
    plot_rois_pyvista_noaxes(surr_thresh_matched[thr]['mean_SDI']*np.abs(surr_thresh_matched[thr]['SDI_sig']), scale, './FIGURES/ALIGN', label='SDImean_matched_thr%d_schz_%s_ref%s'%(thr,  lateralization, ref), vmin=-2, vmax=2)

    # Also plot for thr=5
    thr = 5
    plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), scale, './FIGURES/EPvsCTRL', label='SDImean_thr%d_number_of_fibers_HC_dsi_%s'%(thr, lateralization), vmin=-2, vmax=2)
    plot_rois_pyvista_noaxes(surr_thresh_ref[thr]['mean_SDI']*np.abs(surr_thresh_ref[thr]['SDI_sig']), scale, './FIGURES/EPvsCTRL', label='SDImean_thr%d_number_of_fibers_HC_dsi_%s_ref'%(thr, lateralization), vmin=-2, vmax=2)
    plot_rois_pyvista_noaxes(surr_thresh_rotated[thr]['mean_SDI']*np.abs(surr_thresh_rotated[thr]['SDI_sig']), scale, './FIGURES/ALIGN', label='SDImean_rotated_thr%d_schz_%s_ref%s'%(thr,  lateralization, ref), vmin=-2, vmax=2)
    plot_rois_pyvista_noaxes(surr_thresh_ortho_rotated[thr]['mean_SDI']*np.abs(surr_thresh_ortho_rotated[thr]['SDI_sig']), scale, './FIGURES/ALIGN', label='SDImean_ortho_rotated_thr%d_schz_%s_ref%s'%(thr, lateralization, ref), vmin=-2, vmax=2)
    plot_rois_pyvista_noaxes(surr_thresh_matched[thr]['mean_SDI']*np.abs(surr_thresh_matched[thr]['SDI_sig']), scale, './FIGURES/ALIGN', label='SDImean_matched_thr%d_schz_%s_ref%s'%(thr,  lateralization, ref), vmin=-2, vmax=2)  

    ### Generate comparison figures for alignment methods
    print("\nGenerating comparison figures for different alignment methods...")

    # Count significant ROIs for each alignment method
    nbROIs_sig = []
    nbROIs_sig_ref = []
    nbROIs_sig_rotated = []
    nbROIs_sig_ortho_rotated = []
    nbROIs_sig_matched = []
    for p in np.arange(np.shape(surr_thresh)[0]):
        nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
        nbROIs_sig_ref.append(len(np.where(np.abs(surr_thresh_ref[p]['SDI_sig']))[0]))
        nbROIs_sig_rotated.append(len(np.where(np.abs(surr_thresh_rotated[p]['SDI_sig']))[0]))
        nbROIs_sig_ortho_rotated.append(len(np.where(np.abs(surr_thresh_ortho_rotated[p]['SDI_sig']))[0]))
        nbROIs_sig_matched.append(len(np.where(np.abs(surr_thresh_matched[p]['SDI_sig']))[0]))

    # Calculate mean SDI for scatter comparisons
    mean_sdi_raw = np.mean(SDI, axis=1)
    mean_sdi_ref = np.mean(SDI_ref, axis=1)
    mean_sdi_rotated = np.mean(SDI_rotated, axis=1)
    mean_sdi_ortho = np.mean(SDI_ortho_rotated, axis=1)
    mean_sdi_matched = np.mean(SDI_matched, axis=1)

    x_pos = np.arange(4)
    methods = ['ref: SC HC (ref)', 'Gen. Procrustes', 'Ortho. Procrustes', 'Hungarian']
    methods_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    # Figure 1: Scatter comparisons between alignment methods
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()
    from scipy.stats import pearsonr

    # Order: SC IND (axs[0]), Ortho.P (axs[1]), Gen.P (axs[2]), Hungarian (axs[3])
    # to match Figure 4 harmonic similarity order

    # SC HC (ref) vs SC IND
    r, p = pearsonr(mean_sdi_ref, mean_sdi_raw)
    axs[0].scatter(mean_sdi_ref, mean_sdi_raw, alpha=0.7, s=80, color='#1f77b4', edgecolors='white', linewidth=0.5)
    axs[0].set_xlabel('SC HC (ref)', fontsize=13, fontweight='bold')
    axs[0].set_ylabel('SC IND', fontsize=13, fontweight='bold')
    axs[0].set_title(f'r={r:.3f}, p={p:.3e}', fontsize=11, fontweight='bold')
    axs[0].tick_params(labelsize=11)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_linewidth(1.5)
    axs[0].spines['bottom'].set_linewidth(1.5)

    # SC HC (ref) vs Orthogonal Procrustes
    r, p = pearsonr(mean_sdi_ref, mean_sdi_ortho)
    axs[1].scatter(mean_sdi_ref, mean_sdi_ortho, alpha=0.7, s=80, color='#ff7f0e', edgecolors='white', linewidth=0.5)
    axs[1].set_xlabel('SC HC (ref)', fontsize=13, fontweight='bold')
    axs[1].set_ylabel('Ortho. Procrustes', fontsize=13, fontweight='bold')
    axs[1].set_title(f'r={r:.3f}, p={p:.3e}', fontsize=11, fontweight='bold')
    axs[1].tick_params(labelsize=11)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_linewidth(1.5)
    axs[1].spines['bottom'].set_linewidth(1.5)

    # SC HC (ref) vs Generalized Procrustes
    r, p = pearsonr(mean_sdi_ref, mean_sdi_rotated)
    axs[2].scatter(mean_sdi_ref, mean_sdi_rotated, alpha=0.7, s=80, color='#2ca02c', edgecolors='white', linewidth=0.5)
    axs[2].set_xlabel('SC HC (ref)', fontsize=13, fontweight='bold')
    axs[2].set_ylabel('Gen. Procrustes', fontsize=13, fontweight='bold')
    axs[2].set_title(f'r={r:.3f}, p={p:.3e}', fontsize=11, fontweight='bold')
    axs[2].tick_params(labelsize=11)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['left'].set_linewidth(1.5)
    axs[2].spines['bottom'].set_linewidth(1.5)

    # SC HC (ref) vs Hungarian
    r, p = pearsonr(mean_sdi_ref, mean_sdi_matched)
    axs[3].scatter(mean_sdi_ref, mean_sdi_matched, alpha=0.7, s=80, color='#d62728', edgecolors='white', linewidth=0.5)
    axs[3].set_xlabel('SC HC (ref)', fontsize=13, fontweight='bold')
    axs[3].set_ylabel('Hungarian', fontsize=13, fontweight='bold')
    axs[3].set_title(f'r={r:.3f}, p={p:.3e}', fontsize=11, fontweight='bold')
    axs[3].tick_params(labelsize=11)
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    axs[3].spines['left'].set_linewidth(1.5)
    axs[3].spines['bottom'].set_linewidth(1.5)

    fig.suptitle(f'SDI Alignment Methods Comparison ({lateralization})', fontsize=16, fontweight='bold', y=1.02)

    plt.savefig('./FIGURES/ALIGN/comparison_alignment_scatter_%s.png'%lateralization, dpi=300, bbox_inches='tight')
    plt.savefig('./FIGURES/ALIGN/comparison_alignment_scatter_%s.pdf'%lateralization, bbox_inches='tight')

    # Save significant ROIs with their SDI values as CSV table
    # Build table with significant ROIs for all thresholds and alignment methods
    from collections import defaultdict

    # Select threshold 5 for detailed results
    thr = 5

    # Collect all ROIs that are significant in any method at threshold 5
    sig_indices = {
        'SC_IND': np.where(surr_thresh[thr]['SDI_sig'] != 0)[0],
        'SC_HC': np.where(surr_thresh_ref[thr]['SDI_sig'] != 0)[0],
        'Gen_Procrustes': np.where(surr_thresh_rotated[thr]['SDI_sig'] != 0)[0],
        'Ortho_Procrustes': np.where(surr_thresh_ortho_rotated[thr]['SDI_sig'] != 0)[0],
        'Hungarian': np.where(surr_thresh_matched[thr]['SDI_sig'] != 0)[0],
    }
    
    # Load ROI labels first to preserve their order
    roi_labels = pd.read_csv('DATA/label/labels_rois_118.csv')
    labels_118 = roi_labels['Label Lausanne2008'].values if 'Label Lausanne2008' in roi_labels.columns else [f'ROI_{i}' for i in range(118)]
    
    # Collect all significant indices
    all_idx_set = set()
    for indices in sig_indices.values():
        all_idx_set.update(indices)
    
    # Collect all significant indices
    all_idx_set = set()
    for indices in sig_indices.values():
        all_idx_set.update(indices)
    
    # Sort indices (this preserves index order, which corresponds to labels_118 order)
    all_idx = sorted(list(all_idx_set))

    # Build a dictionary for DataFrame using MultiIndex like script 02
    data = {("ROI", ""): [labels_118[idx] for idx in all_idx]}
    
    methods = ['SC_HC', 'SC_IND', 'Ortho_Procrustes', 'Gen_Procrustes', 'Hungarian']
    method_display = {
        'SC_HC': 'SC HC (ref)',
        'SC_IND': 'SC IND',
        'Ortho_Procrustes': 'Ortho. Procrustes',
        'Gen_Procrustes': 'Gen. Procrustes',
        'Hungarian': 'Hungarian',
    }
    
    for method in methods:
        col_name = (lateralization, method_display[method])
        values = []
        for idx in all_idx:
            if idx in sig_indices[method]:
                if method == 'SC_HC':
                    values.append(round(surr_thresh_ref[thr]['mean_SDI'][idx], 3))
                elif method == 'SC_IND':
                    values.append(round(surr_thresh[thr]['mean_SDI'][idx], 3))
                elif method == 'Gen_Procrustes':
                    values.append(round(surr_thresh_rotated[thr]['mean_SDI'][idx], 3))
                elif method == 'Ortho_Procrustes':
                    values.append(round(surr_thresh_ortho_rotated[thr]['mean_SDI'][idx], 3))
                elif method == 'Hungarian':
                    values.append(round(surr_thresh_matched[thr]['mean_SDI'][idx], 3))
            else:
                values.append(np.nan)
        data[col_name] = values

    df_table = pd.DataFrame(data)
    df_table.columns = pd.MultiIndex.from_tuples(df_table.columns)
    combined_tables[lateralization] = df_table.copy()
    
    df_table.to_csv('./FIGURES/ALIGN/table_significant_ROIs_%s_thr%d.csv'%(lateralization, thr), index=False)
    print(f"\nSignificant ROIs table saved to ./FIGURES/ALIGN/table_significant_ROIs_{lateralization}_thr{thr}.csv")
    print("\nSignificant ROIs with SDI values at threshold %d:" % thr)
    print(df_table.to_string(index=False))

    # Figure 2: Number of significant ROIs comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(nbROIs_sig))

    # Plot lines for each method
    ax.plot(x, nbROIs_sig_ref, color='#1f77b4', linewidth=2.5, marker='s', markersize=8, linestyle='--', label='SC HC (ref)')
    ax.plot(x, nbROIs_sig, color='#1f77b4', linewidth=2.5, marker='o', markersize=8, label='SC IND')
    ax.plot(x, nbROIs_sig_rotated, color='#2ca02c', linewidth=2.5, marker='^', markersize=8, label='Gen. Procrustes')
    ax.plot(x, nbROIs_sig_ortho_rotated, color='#ff7f0e', linewidth=2.5, marker='v', markersize=8, label='Ortho. Procrustes')
    ax.plot(x, nbROIs_sig_matched, color='#d62728', linewidth=2.5, marker='D', markersize=8, label='Hungarian')

    # Add value labels on points
    for i, (y0, y1, y2, y3, y4) in enumerate(zip(nbROIs_sig, nbROIs_sig_ref, nbROIs_sig_rotated, nbROIs_sig_ortho_rotated, nbROIs_sig_matched)):
        ax.text(i-0.1, y0, f'{y0}', fontsize=8, ha='center', va='bottom', color='#1f77b4', fontweight='bold')
        ax.text(i+0.05, y1, f'{y1}', fontsize=8, ha='center', va='bottom', color='#1f77b4', fontweight='bold')
        ax.text(i+0.15, y2, f'{y2}', fontsize=8, ha='center', va='bottom', color='#2ca02c', fontweight='bold')
        ax.text(i+0.25, y3, f'{y3}', fontsize=8, ha='center', va='bottom', color='#ff7f0e', fontweight='bold')
        ax.text(i+0.35, y4, f'{y4}', fontsize=8, ha='center', va='bottom', color='#d62728', fontweight='bold')

    ax.set_xlabel('Threshold (# subjects)', fontsize=13, fontweight='bold')
    ax.set_ylabel('# Significant ROIs', fontsize=13, fontweight='bold')
    ax.set_title(f'Number of significant ROIs across alignment methods ({lateralization})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(len(nbROIs_sig))], fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig('./FIGURES/ALIGN/comparison_nbROIs_sig_%s.png'%lateralization, dpi=300, bbox_inches='tight')
    plt.savefig('./FIGURES/ALIGN/comparison_nbROIs_sig_%s.pdf'%lateralization, bbox_inches='tight')

    # Figure 3: Cutoff frequency comparison - scatter plot with regressions
    from scipy.stats import pearsonr as scipy_pearsonr

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Boxplot with reordered data: SC HC (ref) first
    bp = axs[0].boxplot([ls_cutoff_ref, ls_cutoff, ls_cutoff_rotated, ls_cutoff_ortho_rotated, ls_cutoff_matched],
                         tick_labels=['SC HC (ref)', 'SC IND', 'Gen. Procrustes', 'Ortho. Procrustes', 'Hungarian'], patch_artist=True, widths=0.6)
    for i, (patch, color) in enumerate(zip(bp['boxes'], ['#1f77b4', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, color='black')
    for cap in bp['caps']:
        cap.set(linewidth=1.5, color='black')
    for median in bp['medians']:
        median.set(linewidth=2, color='black')

    # Overlay scatter points for data distribution
    # Add jitter to x-positions for visibility
    np.random.seed(42)
    jitter_strength = 0.04
    
    positions = np.arange(1, 6)  # 5 boxes
    data_list = [ls_cutoff_ref, ls_cutoff, ls_cutoff_rotated, ls_cutoff_ortho_rotated, ls_cutoff_matched]
    colors_list = ['#1f77b4', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    for pos, data, color in zip(positions, data_list, colors_list):
        x_jitter = np.random.normal(pos, jitter_strength, size=len(data))
        axs[0].scatter(x_jitter, data, alpha=0.4, s=30, color=color, edgecolors='none')

    # Add pairwise significance tests on boxplot with bars
    from scipy.stats import ttest_rel
    # Reorder comparisons: now SC HC (ref) is position 1, SC IND is position 2
    t1, p1 = ttest_rel(ls_cutoff_ref, ls_cutoff)
    t2, p2 = ttest_rel(ls_cutoff_ref, ls_cutoff_rotated)
    t3, p3 = ttest_rel(ls_cutoff_ref, ls_cutoff_ortho_rotated)
    t4, p4 = ttest_rel(ls_cutoff_ref, ls_cutoff_matched)

    # Get y-axis limits
    y_max = max(max(ls_cutoff), max(ls_cutoff_ref), max(ls_cutoff_rotated), max(ls_cutoff_ortho_rotated), max(ls_cutoff_matched))
    y_min = min(min(ls_cutoff), min(ls_cutoff_ref), min(ls_cutoff_rotated), min(ls_cutoff_ortho_rotated), min(ls_cutoff_matched))
    y_range = y_max - y_min

    # Draw significance bars
    bar_height = y_range * 0.05
    bar_y = y_max + y_range * 0.05

    # SC HC (ref) vs SC IND (positions 1-2)
    if p1 < 0.05:
        axs[0].plot([1, 2], [bar_y, bar_y], 'k-', linewidth=1.5)
        axs[0].text(1.5, bar_y + bar_height*0.5, f'p={p1:.3e}', ha='center', va='bottom', fontsize=7)
        bar_y += bar_height * 2

    # SC HC (ref) vs Gen. Procrustes (positions 1-3)
    if p2 < 0.05:
        axs[0].plot([1, 3], [bar_y, bar_y], 'k-', linewidth=1.5)
        axs[0].text(2, bar_y + bar_height*0.5, f'p={p2:.3e}', ha='center', va='bottom', fontsize=7)
        bar_y += bar_height * 2

    # SC HC (ref) vs Ortho. Procrustes (positions 1-4)
    if p3 < 0.05:
        axs[0].plot([1, 4], [bar_y, bar_y], 'k-', linewidth=1.5)
        axs[0].text(2.5, bar_y + bar_height*0.5, f'p={p3:.3e}', ha='center', va='bottom', fontsize=7)
        bar_y += bar_height * 2

    # SC HC (ref) vs Hungarian (positions 1-5)
    if p4 < 0.05:
        axs[0].plot([1, 5], [bar_y, bar_y], 'k-', linewidth=1.5)
        axs[0].text(3, bar_y + bar_height*0.5, f'p={p4:.3e}', ha='center', va='bottom', fontsize=7)

    axs[0].set_ylabel('Cutoff frequency (Hz)', fontsize=13, fontweight='bold')
    axs[0].set_title(f'Cutoff frequency distribution ({lateralization})', fontsize=13, fontweight='bold')
    axs[0].tick_params(axis='x', rotation=45, labelsize=10)
    axs[0].tick_params(axis='y', labelsize=11)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_linewidth(1.5)
    axs[0].spines['bottom'].set_linewidth(1.5)

    # Scatter: SC IND vs aligned methods (with SC IND as x-axis)
    axs[1].scatter(ls_cutoff, ls_cutoff_ref, alpha=0.7, s=80, color='#1f77b4', label='SC HC (ref)', edgecolors='white', linewidth=0.5, marker='s')
    axs[1].scatter(ls_cutoff, ls_cutoff_rotated, alpha=0.7, s=80, color='#2ca02c', label='Gen. Procrustes', edgecolors='white', linewidth=0.5, marker='^')
    axs[1].scatter(ls_cutoff, ls_cutoff_ortho_rotated, alpha=0.7, s=80, color='#ff7f0e', label='Ortho. Procrustes', edgecolors='white', linewidth=0.5, marker='v')
    axs[1].scatter(ls_cutoff, ls_cutoff_matched, alpha=0.7, s=80, color='#d62728', label='Hungarian', edgecolors='white', linewidth=0.5)

    # Add regression lines for each method
    x_range = np.linspace(np.min(ls_cutoff), np.max(ls_cutoff), 100)

    # ref regression
    z0 = np.polyfit(ls_cutoff, ls_cutoff_ref, 1)
    p0 = np.poly1d(z0)
    axs[1].plot(x_range, p0(x_range), color='#1f77b4', linewidth=2, linestyle='--', alpha=0.8)
    r0, p0_val = scipy_pearsonr(ls_cutoff, ls_cutoff_ref)

    # Gen. Procrustes regression
    z1 = np.polyfit(ls_cutoff, ls_cutoff_rotated, 1)
    p1 = np.poly1d(z1)
    axs[1].plot(x_range, p1(x_range), color='#2ca02c', linewidth=2, linestyle='--', alpha=0.8)
    r1, p1_val = scipy_pearsonr(ls_cutoff, ls_cutoff_rotated)

    # Ortho. Procrustes regression
    z2 = np.polyfit(ls_cutoff, ls_cutoff_ortho_rotated, 1)
    p2 = np.poly1d(z2)
    axs[1].plot(x_range, p2(x_range), color='#ff7f0e', linewidth=2, linestyle='--', alpha=0.8)
    r2, p2_val = scipy_pearsonr(ls_cutoff, ls_cutoff_ortho_rotated)

    # Hungarian regression
    z3 = np.polyfit(ls_cutoff, ls_cutoff_matched, 1)
    p3 = np.poly1d(z3)
    axs[1].plot(x_range, p3(x_range), color='#d62728', linewidth=2, linestyle='--', alpha=0.8)
    r3, p3_val = scipy_pearsonr(ls_cutoff, ls_cutoff_matched)

    axs[1].set_xlabel('SC IND cutoff (Hz)', fontsize=13, fontweight='bold')
    axs[1].set_ylabel('Aligned cutoff (Hz)', fontsize=13, fontweight='bold')
    axs[1].set_title(f'Cutoff frequency: SC IND vs aligned ({lateralization})\nSC HC (ref) (r={r0:.3f}, p={p0_val:.3e}), Gen.P (r={r1:.3f}, p={p1_val:.3e})\nOrtho.P (r={r2:.3f}, p={p2_val:.3e}), Hung (r={r3:.3f}, p={p3_val:.3e})', fontsize=10, fontweight='bold')
    axs[1].legend(fontsize=11, frameon=False)
    axs[1].tick_params(labelsize=11)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_linewidth(1.5)
    axs[1].spines['bottom'].set_linewidth(1.5)

    plt.savefig('./FIGURES/ALIGN/comparison_cutoff_freq_%s.png'%lateralization, dpi=300, bbox_inches='tight')
    plt.savefig('./FIGURES/ALIGN/comparison_cutoff_freq_%s.pdf'%lateralization, bbox_inches='tight')

    # Figure 4: Harmonic similarity comparison (2x2 layout like script 12)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    
    n_harmonics = len(similarity_ind)
    harmonic_indices = np.arange(n_harmonics)
    
    # Define colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    similarity_ylim = (0, 1.02)
    
    # A. Before alignment (SC IND vs SC HC)
    ax[0].plot(harmonic_indices, similarity_ind, linewidth=2, color=colors[0])
    mean_ind = np.mean(similarity_ind)
    std_ind = np.std(similarity_ind)
    ax[0].axhline(y=mean_ind, color=colors[0], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[0].text(n_harmonics+1, mean_ind, f'{mean_ind:.3f}±{std_ind:.3f}', color=colors[0], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[0].fill_between(harmonic_indices, similarity_ind - std_ind, similarity_ind + std_ind, alpha=0.15, color=colors[0])
    ax[0].set_title('A. Before alignment', fontsize=13, fontweight='bold', loc='left', pad=10)
    ax[0].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
    ax[0].set_ylabel('Correlation', fontsize=12, fontweight='bold')
    
    # B. Orthogonal Procrustes
    ax[1].plot(harmonic_indices, similarity_ortho, linewidth=2, color=colors[1])
    mean_ortho = np.mean(similarity_ortho)
    std_ortho = np.std(similarity_ortho)
    ax[1].axhline(y=mean_ortho, color=colors[1], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[1].text(n_harmonics+1, mean_ortho, f'{mean_ortho:.3f}±{std_ortho:.3f}', color=colors[1], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[1].fill_between(harmonic_indices, similarity_ortho - std_ortho, similarity_ortho + std_ortho, alpha=0.15, color=colors[1])
    ax[1].set_title('B. Orthogonal Procrustes', fontsize=13, fontweight='bold', loc='left', pad=10)
    ax[1].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
    ax[1].set_ylabel('Correlation', fontsize=12, fontweight='bold')
    
    # C. Generalized Procrustes
    ax[2].plot(harmonic_indices, similarity_rotated, linewidth=2, color=colors[2])
    mean_rotated = np.mean(similarity_rotated)
    std_rotated = np.std(similarity_rotated)
    ax[2].axhline(y=mean_rotated, color=colors[2], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[2].text(n_harmonics+1, mean_rotated, f'{mean_rotated:.3f}±{std_rotated:.3f}', color=colors[2], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[2].fill_between(harmonic_indices, similarity_rotated - std_rotated, similarity_rotated + std_rotated, alpha=0.15, color=colors[2])
    ax[2].set_title('C. Generalized Procrustes', fontsize=13, fontweight='bold', loc='left', pad=10)
    ax[2].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
    ax[2].set_ylabel('Correlation', fontsize=12, fontweight='bold')
    
    # D. Hungarian matching
    ax[3].plot(harmonic_indices, similarity_matched, linewidth=2, color=colors[3])
    mean_matched = np.mean(similarity_matched)
    std_matched = np.std(similarity_matched)
    ax[3].axhline(y=mean_matched, color=colors[3], linestyle=':', linewidth=1.5, alpha=0.7)
    ax[3].text(n_harmonics+1, mean_matched, f'{mean_matched:.3f}±{std_matched:.3f}', color=colors[3], fontsize=8, va='center', ha='left', fontweight='bold')
    ax[3].fill_between(harmonic_indices, similarity_matched - std_matched, similarity_matched + std_matched, alpha=0.15, color=colors[3])
    ax[3].set_title('D. Hungarian matching', fontsize=13, fontweight='bold', loc='left', pad=10)
    ax[3].set_xlabel('Eigenmode', fontsize=12, fontweight='bold')
    ax[3].set_ylabel('Correlation', fontsize=12, fontweight='bold')
    
    # Format all subplots
    for i in range(4):
        ax[i].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax[i].set_ylim(similarity_ylim)
        ax[i].set_xticks(range(0, n_harmonics, 20))
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_linewidth(1.5)
        ax[i].spines['bottom'].set_linewidth(1.5)
        ax[i].tick_params(labelsize=11)
    
    fig.suptitle('Similarity between SC HC (ref) and SC IND harmonics', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    plt.savefig('./FIGURES/ALIGN/harmonic_similarity_%s.png'%lateralization, dpi=300, bbox_inches='tight')
    plt.savefig('./FIGURES/ALIGN/harmonic_similarity_%s.pdf'%lateralization, bbox_inches='tight')
    print(f"Harmonic similarity figure saved for {lateralization}")

    print("Comparison figures saved to ./FIGURES/ALIGN/")


# Merge RT and LT significant ROIs (thr=5) into a single table like script 02/08
if combined_tables:
    # Load labels to preserve order
    roi_labels = pd.read_csv('DATA/label/labels_rois_118.csv')
    labels_118 = roi_labels['Label Lausanne2008'].values if 'Label Lausanne2008' in roi_labels.columns else [f'ROI_{i}' for i in range(118)]
    
    # Collect all significant ROIs from both lateralizations
    all_significant_rois = set()
    for lat in ls_lateralization:
        df_lat = combined_tables.get(lat)
        if df_lat is not None:
            roi_col = [col for col in df_lat.columns if col[0] == 'ROI']
            if roi_col:
                all_significant_rois.update(df_lat[roi_col[0]].tolist())
    
    # Order ROIs according to labels_118
    all_rois = [roi for roi in labels_118 if roi in all_significant_rois]

    # Build combined data with all ROIs in labels_118 order
    combined_data = {("ROI", ""): all_rois}
    
    methods = ['SC HC (ref)', 'SC IND', 'Ortho. Procrustes', 'Gen. Procrustes', 'Hungarian']
    
    for lat in ls_lateralization:
        df_lat = combined_tables.get(lat)
        if df_lat is None:
            continue
        
        # Get ROI column
        roi_col = [col for col in df_lat.columns if col[0] == 'ROI'][0]
        
        # Create a mapping from ROI name to values for each method
        for method in methods:
            col_tuple = (lat, method)
            values = []
            for roi in all_rois:
                # Find the row with this ROI
                mask = df_lat[roi_col] == roi
                if mask.any():
                    val = df_lat.loc[mask, (lat, method)].values[0]
                    values.append(val if pd.notna(val) else np.nan)
                else:
                    values.append(np.nan)
            combined_data[col_tuple] = values

    combined_df = pd.DataFrame(combined_data)
    combined_df.columns = pd.MultiIndex.from_tuples(combined_df.columns)
    
    # Ensure numeric columns are floats; keep NaN as blanks in CSV
    combined_df_for_csv = combined_df.copy()
    for col in combined_df_for_csv.columns:
        if col[0] == 'ROI':
            continue
        combined_df_for_csv[col] = combined_df_for_csv[col].where(~combined_df_for_csv[col].isna(), '')

    output_path = './FIGURES/ALIGN/table_significant_ROIs_RT_LT_thr5.csv'
    combined_df_for_csv.to_csv(output_path, index=False)
    print(f"\nCombined RT+LT significant ROIs table saved to {output_path}")
    print(combined_df)

plt.show()