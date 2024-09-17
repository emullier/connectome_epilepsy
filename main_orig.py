
from flask import Flask, render_template, Response
import subprocess
import tkinter as tk
from config_gui import ConfigGUI
import os
import lib.func_reading as reading
import lib.func_utils as utils
import logging
import numpy as np
import pandas as pd
import lib.func_ML as ML
import matplotlib
import matplotlib.pyplot as plt
from lib.func_plot import plot_rois
import lib.func_metaanalysis as MA
import lib.func_SDI as sdi
import scipy.io as sio
import shutil
import random
matplotlib.use('TkAgg')


def run_processing(config):

    ### Empty the figures directory
    if os.path.exists('./public/static/images'):
        shutil.rmtree('./public/static/images')
    os.makedirs('./public/static/images')

    ### Create the log file
    logging.basicConfig(
        filename='%s/log_file.log'% config['Parameters']['output_dir'],  # Name of the log file
        level=logging.INFO,
        format='%(asctime)s - %(message)s',  # Log message format
        datefmt='%Y-%m-%d %H:%M:%S' ,
        filemode='w')  # 'w' to overwrite the file, 'a' to append to the file
    
    logging.info('DATA LOADING \n')
        
    ''' DATA LOADING'''
    ### Load the demographic information, check the filters (keys) and filter the subjects to be kept
    df_info = reading.read_info(config['Parameters']['info_path'])
    logging.info('Loading demographic information')
    filters = utils.compare_pdkeys_list(df_info, config['Parameters']['filters'])
    logging.info('Found Filters in the csv file: %s \n'  % filters)

    ### Filter the list of participants based on the filters in the configuration file
    df, ls_subs = reading.filtered_dataframe(df_info, filters, config)
    logging.info("Number of subjects kept after filtering: %d" % len(ls_subs)); 
    
    ## Generate list of processing types to include
    procs = list(config["Parameters"]["processing"].keys())
    logging.info("Included Processing: %s" % procs)
    
    ### Load the matrices
    MatMat = {}; EucMat = {}; dict_df = {}
    procs = config["Parameters"]["processing"]
    for p,proc in enumerate(config["Parameters"]["processing"]):
        idxs_tmp = np.where((df[proc]==1)|(df[proc]=='1'))[0]
        df_tmp = df.iloc[idxs_tmp]
        logging.info("Number of subjects with processing %s: %d" %(proc,len(idxs_tmp)))
        tmp_path = os.path.join(config["Parameters"]["data_dir"], config["Parameters"]["processing"][proc])
        MatMat[proc], EucMat[proc], df_info = reading.load_matrices(df_tmp, tmp_path, config['Parameters']['scale'], config['Parameters']['metric'])
        ### if loading the example data (from Ale), update df_info with 'fake' sub info for the code to run properly
        if proc=='example':
            new_subs = ['sub-IC' + str(i) for i in range(1, np.shape(MatMat[proc])[2]+1)]
            df_tmp = pd.DataFrame({'sub': new_subs, 'dwi':'dsi', 'age': 40, 'group':'HC', 'gender': 'M', 'Inclusion':1, 'example':1})
        dict_df[proc] = df_tmp 
    if len(config["Parameters"]["processing"])>1:
        MatConc, EucConc, df_conc = utils.concatenate_processings(MatMat, EucMat, dict_df, config)
    else:
        MatConc = MatMat[list(procs.keys())[0]]
        EucConc = EucMat[list(procs.keys())[0]]
        df_conc = dict_df[list(procs.keys())[0]]
        

    ''' BIAS INVESTIGATION'''
    ### Perform dimensionality reduction (via PCA) to look for evident bias in the connectivity matrices
    if config['BIAS']['perform_analysis']:
        logging.info('Starting Bias Analysis \n')
        if config['BIAS']['DimReduc']=="PCA":
            logging.info(config['BIAS']['DimReduc'])
            print(df_conc.keys())
            X_pca, fig, axs = ML.bias_PCAplot_concat(MatConc, df_conc, config["BIAS"]["colors_key"], p='Concatenated')
        for p,proc in enumerate(config["Parameters"]["processing"]):
            ML.bias_PCAplot_concat(MatMat[proc], dict_df[proc], config["BIAS"]["colors_key"], p=str(p+1))
        
    ### Estimate the consensus matrices for each selected processing (uniform consensus and distance-base consensus)
    if config['CONSENSUS']['perform_analysis']:
        logging.info('Starting Consensus Estimation \n')
        procs = list(config["Parameters"]["processing"].keys())
        # EucDist, cort_rois, hemii = ML.ROIs_euclidean_distance(config["Parameters"]["scale"])
        G_dist, G_unif = ML.consensus(MatMat, config["Parameters"]["processing"],  dict_df, EucMat, config["CONSENSUS"]["nbins"])
        G_dist_wei, G_unif_wei = reading.save_consensus(MatMat, config["Parameters"]["metric"], G_dist, G_unif, config["Parameters"]["output_dir"], config["Parameters"]["processing"])
        #if len(procs)>1:
        #    ML.compare_matrices(G_dist_wei[procs[0]], G_dist_wei[procs[1]], dict_df[procs[0]], dict_df[procs[1]], procs[0], procs[1], plot=True)
        #    MatDist_normError_inter, MatDist_corr_inter = ML.compare_matrices(MatMat[procs[0]], MatMat[procs[1]], dict_df[procs[0]], dict_df[procs[1]], procs[0], procs[1], plot=True)
        #    MatDist_normError_intra, MatDist_corr_intra = ML.compare_matrices(MatMat[procs[0]], MatMat[procs[0]], dict_df[procs[0]], dict_df[procs[0]], procs[0], procs[0], plot=True)
        #    ML.inter_vs_intra_compare_matrices(MatMat, dict_df, procs)
        logging.info("\n"); logging.info('""""')
        
    if config['HARMONICS']['generate_harmonics']:
        logging.info('Harmonics analysis')
        procs = list(config["Parameters"]["processing"].keys())
        ### Compute the structural harmonics
        P_group, Q_group = ML.group_normalized_lap(MatMat, EucMat, procs, plot=False)
            
        ### Concatenate the different processing before aligning them (along the participant dimension)
        P_ind_conc, Q_ind_conc, Ln_ind_conc, An_ind_conc = ML.ind_normalized_lap(MatConc, EucConc, df_conc, plot=True)
        Q_all_rotated_conc, P_all_rotated_conc, R_all_conc, scale_R_conc = ML.rotation_procrustes(Q_ind_conc, P_ind_conc, plot=True, p='Concatenated')
        Mat_recon_conc = ML.reconstruct_SC(An_ind_conc, df_conc, P_ind_conc, Q_ind_conc, plot=True, p='Concatenated')
        #Mat_recon_conc = ML.reconstruct_SC(An_ind_conc, df_conc, P_all_rotated_conc, Q_all_rotated_conc, k=100, plot=True, p='Concatenated_Rotated')
        #for i in np.arange(3):
        #   save_fname = plot_rois(Q_ind[:,i,0], config["Parameters"]["scale"], config, vmin=-.15, vmax=.15, label='test_eigv%d'%i)
        #   save_fname = plot_rois(Q_all_rotated[:,i,0], config["Parameters"]["scale"], config, vmin=-.15, vmax=.15, label='test_rotated_eigv%d'%i)
            
        ### Aligning for every processing independently
        P_ind={}; Q_ind={}; Q_all_rotated={}; P_all_rotated={}; scale_R={}; R_all={}; Mat_recon={}; Ln_ind = {}; An_ind= {}
        for p,proc in enumerate(procs):
            P_ind[proc], Q_ind[proc], Ln_ind[proc], An_ind[proc] = ML.ind_normalized_lap(MatMat[proc], EucMat[proc], dict_df[proc], plot=True)
            Q_all_rotated[proc], P_all_rotated[proc], R_all[proc], scale_R[proc] = ML.rotation_procrustes(Q_ind[proc], P_ind[proc], plot=True, p='%s'%proc)
            Mat_recon[proc] = ML.reconstruct_SC(An_ind[proc], dict_df[proc], P_ind[proc], Q_ind[proc], k=100, plot=True, p='%s'%proc)

         
    if config['HARMONICS']['compare_harmonics']:
        #diff_norm = utils.compare_large_rotations(R_all)
        #diff_harm = utils.compare_harmonicwise_rotations(R_all)
        if len(procs)>1:
            diff_eig = utils.diff_before_after_rotation(Q_ind_conc, Q_all_rotated_conc, df_conc)
        #print("Difference Norm (Frobenius):", np.mean(diff_norm))    
     
    if config['HARMONICS']['consensus']:
        procs = list(config["Parameters"]["processing"].keys())
        ConsConc = reading.reading_consensus(procs, config["Parameters"]["metric"], config["Parameters"]["output_dir"], dict_df)
        if len(procs)>1:
            ML.compare_matrices(ConsConc[:,:,0], ConsConc[:,:,1], dict_df[procs[0]], dict_df[procs[1]], procs[0], procs[1], plot=False)
        ### PROBLEM HERE BETWEEN THE IND CONS FOR CONSENSUS LOADING FOR PROCESSING EXAMPLE
        df_cons = pd.DataFrame([df_conc.iloc[0], df_conc.iloc[-1]])
        P_ind, Q_ind, Ln_ind, An_ind = ML.ind_normalized_lap(ConsConc, EucConc, df_cons, plot=False)   
        Q_all_rotated, P_all_rotated, R_all, scale_R = ML.rotation_procrustes(Q_ind, P_ind, plot=True, title='Concatenated')
        Mat_recon = ML.reconstruct_SC(An_ind, df_cons, P_ind, Q_ind, plot=False)
        #for c in np.arange(len(procs)):
        #    for i in np.arange(3):
        #        print('sub %d eigenv %d'%(c,i))
        #        save_fname = plot_rois(Q_ind[:,i,c], config["Parameters"]["scale"], config, vmin=-.15, vmax=.15, label='cons%d_eigv%d'%(c,i))
        #        save_fname = plot_rois(Q_all_rotated[:,i,c], config["Parameters"]["scale"], config, vmin=-.15, vmax=.15, label='cons%d_rotated_eigv%d'%(c,i))
        MNI_parcellation = os.path.join(os.getcwd(), 'data/lausanne2018.ctx+subc.scale2.maxprob_2x2x2.nii.gz')
        output_path = os.path.join(os.getcwd(), 'output/maps')
        EucDist, cort_rois, hemii = ML.ROIs_euclidean_distance(config["Parameters"]["scale"])
        ### next step takes time
        MA.create_nii_activation_maps(Q_ind, MNI_parcellation , cort_rois + 1, output_path)
        MA.binarize_map(output_path, 0.01)
        neurosynth = os.path.join(os.getcwd(), 'data/neurosynth_maps') 
        MA.binarize_map(neurosynth, 0.01)
        ### Need here to correct for different sizes of nifti file 
        ### Careful: Takes time
        index = MA.comparison_neurosynthmaps(output_path, neurosynth, p='Concatenated')
        
    
    if config["INDvsCONS"]["perform_analysis"]:
        #EucDist, cort_rois, hemii = ML.ROIs_euclidean_distance(config["Parameters"]["scale"])
        ### Concatenated matrices
        EucDist = np.mean(EucConc, axis=2)
        nbPerm = 50; ls_bins = [1,5,10,20]
        RandCons, df_random, ShuffIdxs = ML.generate_randomized_part_consensus(MatConc, nbPerm, EucDist, ls_bins)
        ### Before rotation
        eigenvalues_perm, eigenvalues_perm_mat, eigenvectors_perm, eigenvectors_perm_mat, labels_perm = ML.harmonics_randomized_part_consensus(MatConc, RandCons, nbPerm, EucDist, df_random, ls_bins)
        bin_variability = ML.plot_randomized_part_consensus(MatConc, eigenvectors_perm, nbPerm, labels_perm, ls_bins,plot=True, title='Concatenated')
        ### After Procrustes Rotation
        Q_all_rotated, P_all_rotated, R_all, scale_R = ML.rotation_procrustes(eigenvectors_perm, eigenvalues_perm, plot=False)
        bin_variability_rotated = ML.plot_randomized_part_consensus(MatConc, Q_all_rotated, nbPerm, labels_perm, ls_bins, plot=False, title='Concatenated')
            
        ### Not concatenated matrices
        procs = list(config["Parameters"]["processing"].keys())
        RandCons_p = {}; df_random_p = {}; eigenvectors_perm_p = {}; eigenvalues_perm_p = {}; Q_all_rotated_p = {}; bin_variability_p={}; bin_variability_rotated_p = {}
        for p,proc in enumerate(procs):
            RandCons_p[proc], df_random_p[proc], ShuffIdxs = ML.generate_randomized_part_consensus(MatMat[proc], nbPerm, EucDist, ls_bins)
            eigenvalues_perm_p[proc], eigenvalues_perm_mat, eigenvectors_perm_p[proc], eigenvectors_perm_mat, labels_perm = ML.harmonics_randomized_part_consensus(MatMat[proc], RandCons_p[proc], nbPerm, EucDist, df_random_p[proc], ls_bins)
            bin_variability_p[proc] = ML.plot_randomized_part_consensus(MatMat[proc], eigenvectors_perm_p[proc], nbPerm, labels_perm, ls_bins, plot=True, title='%s'%proc)
            Q_all_rotated_p[proc], P_all_rotated, R_all, scale_R = ML.rotation_procrustes(eigenvectors_perm_p[proc], eigenvalues_perm_p[proc], plot=False)
            bin_variability_rotated_p[proc] = ML.plot_randomized_part_consensus(MatMat[proc], Q_all_rotated_p[proc], nbPerm, labels_perm, ls_bins, plot=False, title='%s'%proc)

      ### Maybe here we will require some adjustements/checking between parcellations
    if config["SDI"]["perform_analysis"]:
        print('SDI')
        ### Load the EEG data
        P_ind_conc, Q_ind_conc, Ln_ind_conc, An_ind_conc = ML.ind_normalized_lap(MatConc, EucConc, df_conc, plot=False)
        Q_all_rotated_conc, P_all_rotated_conc, R_all_conc, scale_R_conc = ML.rotation_procrustes(Q_ind_conc, P_ind_conc, plot=False, p='Concatenated')
        X_RS_allPat = sdi.load_EEG_example()
        #scU = Q_all_rotated_conc[:,:,20] ### temporary scU
        
        scU = Q_ind_conc[:,:,0]
        #scU = np.mean(Q_ind_conc,axis=2)
        SDI = np.zeros((114, len(X_RS_allPat)))
        ls_lat = []
        for p in np.arange(len(X_RS_allPat)):
            X_RS = X_RS_allPat[p]['X_RS']
            ls_lat.append(X_RS_allPat[p]['lat'][0])
            idx_ctx = np.concatenate((np.arange(0,57), np.arange(59,116)))
            PSD,NN, Vlow, Vhigh = sdi.get_cutoff_freq(scU, X_RS[idx_ctx,:,:])
            #SDI[:,:,p], X_c_norm, X_d_norm = sdi.compute_SDI(X_RS[:114,:,:], scU)
            SDI[:,p], X_c_norm, X_d_norm = sdi.compute_SDI(X_RS[idx_ctx,:,:], scU)
        ls_lat = np.array(ls_lat)
        #fig, ax = plt.subplots(1,1); ax.plot(np.mean(X_c_norm,axis=1)); #ax.plot(np.mean(X_d_norm,axis=1))
        
        ### Surrogate part
        SDI_surr, XRandS = sdi.surrogate_sdi(scU, Vlow, Vhigh, nbSurr=100) # Generate the surrogate 
        ### Compare to surrogate part
        
        fig,ax = plt.subplots(1,2)
        for l, lat in enumerate(np.unique(ls_lat)):
            ### not related here, the functional signales are always the same, they should be loaded all the time the same way
            surr_thresh = sdi.select_significant_sdi(SDI[:,np.where(ls_lat==lat)[0]], SDI_surr)
            thr = 1
            nbROIsig = []; ls_th = []
            for t in np.arange(np.shape(surr_thresh)[0]):
                th = surr_thresh[t]['threshold'] 
                ls_th.append(th)
                #if th==thr:
                tmp = surr_thresh[t]['mean_SDI']*np.abs(surr_thresh[t]['SDI_sig'])
                    #plot_rois(tmp, config["Parameters"]["scale"], config, vmin=-1, vmax=1, label='test_SDI_Sig%d'%(thr))
                nbROIsig.append(len(np.where(np.abs(surr_thresh[t]['SDI_sig']))[0]))
                #plot_rois(tmp, config["Parameters"]["scale"], config, vmin=-1, vmax=1, label='SDI_th%d_%s'%(th, lat))
            ax[l].plot(np.array(ls_th), np.array(nbROIsig))
            ax[l].set_xlabel('Threshold #Subs'); ax[l].set_ylabel('#ROIs with significant SDI')
            ax[l].set_title('%s'%lat)
            


        #    print(tmp)
        #plot_rois(tmp, config["Parameters"]["scale"], config, vmin=-1, vmax=1, label='test_SDI_Sig%d'%(c+1))


        #procs = list(config["Parameters"]["processing"].keys())
        ### Plot SDI with consensus proc1 && plot consensus proc2 
        #SDI = np.zeros((114, np.shape(X_RS)[2], len(procs)))
    
        #SDI = np.zeros((np.shape(X_RS)[0], np.shape(X_RS)[1],  np.shape(X_RS)[2], len(procs)))
        #X_c_norm = np.zeros((114, np.shape(X_RS)[1], np.shape(X_RS)[2], len(procs))); X_d_norm = np.zeros(np.shape(X_c_norm))
        #P_ind={}; Q_ind={}; Q_all_rotated={}; P_all_rotated={}; scale_R={}; R_all={}; Mat_recon={}; Ln_ind = {}; An_ind= {}
        #for p,procs in enumerate(procs):
        #    P_ind[proc], Q_ind[proc], Ln_ind[proc], An_ind[proc] = ML.ind_normalized_lap(MatMat[proc], EucMat[proc], dict_df[proc], plot=True)
        #    Q_all_rotated[proc], P_all_rotated[proc], R_all[proc], scale_R[proc] = ML.rotation_procrustes(Q_ind[proc], P_ind[proc], plot=True, p='%s'%proc)
        #    SDI[:,:,p], X_c_norm[:,:,:,p], X_d_norm[:,:,:,p] = sdi.compute_SDI(X_RS[:114], Q_all_rotated_conc[:,:,20])     


        plt.show(block=True)


if __name__ == "__main__":
    # Load the default configuration from the file
    #config_path = 'config_PEP3.json'
    config_path = 'config.json'
    config_defaults = reading.check_config_file(config_path)
    
    #root = tk.Tk()
    #app = ConfigGUI(root, run_processing, config_defaults)
    #root.mainloop()
    run_processing(config_defaults)