import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import lib.func_reading as reading
import lib.func_utils as utils
import lib.func_ML as ML
import lib.func_SDI as sdi
from lib.func_plot import plot_rois, plot_rois_pyvista



def load_data(config):
    ''' Log File '''
    logging.basicConfig(filename='%s/log_file.log'% config['Parameters']['output_dir'], level=logging.INFO,
        format='%(asctime)s - %(message)s',  datefmt='%Y-%m-%d %H:%M:%S' , filemode='w')      
    
    ''' Read the information file '''
    try:
        df_info = reading.read_info(config['Parameters']['info_path'])
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the information file: {e}")
        raise RuntimeError(f"An unexpected error occurred while reading the information file: {e}")
    
    ### Filter by demographic information
    try:
        filters = utils.compare_pdkeys_list(df_info, config['Parameters']['filters'])
        df, ls_subs = reading.filtered_dataframe(df_info, filters, config)
    except Exception as e:
        logging.error(f"An unexpected error occurred with your chosen demographic filters: {e}")
        raise RuntimeError(f"An unexpected error occurred with your chosen demographic filters: {e}")
    
    ## Generate list of processing types to include
    procs = list(config["Parameters"]["processing"].keys())
    try:
        ### Load the matrices
        MatMat = {}; EucMat = {}; dict_df = {}
        if len(procs) == 0:
            logging.error("No processing methods selected.")
            raise ValueError("No processing methods selected.")
        elif len(procs) > 2:
            logging.error("Please select a maximum of two processings.")
            raise ValueError("Please select a maximum of two processings.")
        elif len(procs)==1:
            logging.info('Processing %s selected'%(procs[0]))
        else:
            logging.info('Processings %s and %s selected'%(procs[0], procs[1]))
        for p, proc in enumerate(procs):
            idxs_tmp = np.where((df[proc] == 1) | (df[proc] == '1'))[0]
            df_tmp = df.iloc[idxs_tmp]
            logging.info("Number of subjects with processing %s: %d" % (proc, len(idxs_tmp)))
            tmp_path = os.path.join(config["Parameters"]["data_dir"], config["Parameters"]["processing"][proc])
            MatMat[proc], EucMat[proc], df_info = reading.load_matrices(df_tmp, tmp_path, config['Parameters']['scale'], config['Parameters']['metric'])
            ### Example-specific handling
            if proc == 'example':
                new_subs = ['sub-IC' + str(i) for i in range(1, np.shape(MatMat[proc])[2] + 1)]
                df_tmp = pd.DataFrame({'sub': new_subs, 'dwi': 'dsi', 'age': 40, 'group': 'HC', 'gender': 'M', 'Inclusion': 1, 'example': 1})
            dict_df[proc] = df_tmp
    except ValueError as ve:
        logging.error(f"ValueError occurred with your processing choice: {ve}")  # Log the error message
        raise  # Re-raise the ValueError after logging
    
    ### Check the compatibility between the parcellations 
    ### TO ADD: COMPATIBILITY WITH THE EEG DATA - TAKE ONLY CORTICAL REGIONS
    nbProcs = len(procs)
    comp_parc=0
    if len(procs)==2:
        if np.shape(MatMat[procs[0]])[0]==np.shape(MatMat[procs[1]])[0]:
            logging.info('Parcellation are compatible between the 2 selected processings')
            logging.info('Concatenating the data')
            MatConc, EucConc, df_conc = utils.concatenate_processings(MatMat, EucMat, dict_df, config)
            logging.info('Visual checking for bias between the 2 processings')
            X_pca, fig, axs = ML.bias_PCAplot_concat(MatConc, df_conc, config["BIAS"]["colors_key"], p='Concatenated')
            comp_parc=1
        else:
            logging.info('Parcellation are NOT compatible between the 2 selected processings')

    return MatMat, EucMat, dict_df, comp_parc, nbProcs


def run_analysis(config, MatMat, EucMat, dict_df, comp_parc, nbProcs):
    
    logging.info("Generate the GSP pipeline for each processing individually \n")
    logging.info("'''''''''")
    procs = list(config["Parameters"]["processing"].keys())
    
    logging.info("Load EEG example data for SDI")
    X_RS_allPat = sdi.load_EEG_example()
    
    #logging.info('SDI - Individual Matrices \n')
    #P_ind={}; Q_ind={}; Q_all_rotated={}; P_all_rotated={}; scale_R={}; R_all={}; Mat_recon={}; Ln_ind = {}; An_ind= {}
    #for p,proc in enumerate(procs):
    #    P_ind[proc], Q_ind[proc], Ln_ind[proc], An_ind[proc] = ML.ind_normalized_lap(MatMat[proc], EucMat[proc], dict_df[proc], plot=True)
    #    Q_all_rotated[proc], P_all_rotated[proc], R_all[proc], scale_R[proc] = ML.rotation_procrustes(Q_ind[proc], P_ind[proc], plot=True, p='%s'%proc)
    #    #Mat_recon[proc] = ML.reconstruct_SC(An_ind[proc], dict_df[proc], P_ind[proc], Q_ind[proc], k=100, plot=True, p='%s'%proc)

    logging.info('Consensus Matrices')
    G_dist, G_unif, EucDist = ML.consensus(MatMat, config["Parameters"]["processing"],  dict_df, EucMat, config["CONSENSUS"]["nbins"])
    G_dist_wei, G_unif_wei = reading.save_consensus(MatMat, config["Parameters"]["metric"], G_dist, G_unif, config["Parameters"]["output_dir"], config["Parameters"]["processing"])
    
    logging.info("Generate harmonics from the consensus")
    P_ind={}; Q_ind={}; Ln_ind = {}; An_ind= {}; SDI = {}; SDI_surr = {}
    for p,proc in enumerate(procs):
        
        ### Generate the harmonics
        P_ind[proc], Q_ind[proc], Ln_ind[proc], An_ind[proc] = ML.cons_normalized_lap(G_dist[proc], EucDist[proc], dict_df[proc], plot=True)
        if np.shape(Q_ind[proc])!=114:
            Q_ind[proc] = utils.extract_ctx_ROIs(Q_ind[proc])
                    
        ### Estimate SDI
        SDI_tmp = np.zeros((114, len(X_RS_allPat)))
        ls_lat = []
        for p in np.arange(len(X_RS_allPat)):
            X_RS = X_RS_allPat[p]['X_RS']
            ls_lat.append(X_RS_allPat[p]['lat'][0])
            idx_ctx = np.concatenate((np.arange(0,57), np.arange(59,116)))
            PSD,NN, Vlow, Vhigh = sdi.get_cutoff_freq(Q_ind[proc], X_RS[idx_ctx,:,:])
            SDI_tmp[:,p], X_c_norm, X_d_norm = sdi.compute_SDI(X_RS[idx_ctx,:,:], Q_ind[proc])
        ls_lat = np.array(ls_lat)
        SDI[proc] = SDI_tmp
        

        
        ### Surrogate part
        SDI_surr[proc], XRandS = sdi.surrogate_sdi(Q_ind[proc], Vlow, Vhigh, nbSurr=10) # Generate the surrogate 
        
    
        fig,ax = plt.subplots(1,2)
        for l, lat in enumerate(np.unique(ls_lat)):
            ### not related here, the functional signales are always the same, they should be loaded all the time the same way
            surr_thresh = sdi.select_significant_sdi(SDI_tmp[:,np.where(ls_lat==lat)[0]], SDI_surr[proc])
            thr = 1
            nbROIsig = []; ls_th = []
            for t in np.arange(np.shape(surr_thresh)[0]):
                th = surr_thresh[t]['threshold'] 
                ls_th.append(th)
                #if th==thr:
                tmp = surr_thresh[t]['mean_SDI']*np.abs(surr_thresh[t]['SDI_sig'])
                nbROIsig.append(len(np.where(np.abs(surr_thresh[t]['SDI_sig']))[0]))
                #plot_rois(tmp, config["Parameters"]["scale"], config, vmin=-1, vmax=1, label='SDI_Sig%d'%(surr_thresh[0]['threshold']))
                plot_rois_pyvista(tmp, config["Parameters"]["scale"], config, vmin=-1, vmax=1, label='SDI_th%d_%s'%(surr_thresh[t]['threshold'], lat))
        
            ax[l].plot(np.array(ls_th), np.array(nbROIsig))
            ax[l].set_xlabel('Threshold #Subs'); ax[l].set_ylabel('#ROIs with significant SDI')
            ax[l].set_title('%s'% lat)
            plt.suptitle('%s'%proc)
        
        
    ''' Part of the data concatenated'''
    #if ((nbProcs>1) and (comp_parc==1)):
    #    logging.info("Generate the GSP pipeline for the data concatenated for the 2 processings")
        
    
if __name__ == "__main__":
    # Load the default configuration from the file
    config_path = 'config.json' #'config_PEP3.json'
    config_defaults = reading.check_config_file(config_path)
    MatMat, EucMat, dict_df, compatibility, nbProcs = load_data(config_defaults)
    run_analysis(config_defaults, MatMat, EucMat, dict_df, compatibility, nbProcs)
    plt.show(block=True)