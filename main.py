
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
matplotlib.use('TkAgg')



def run_processing(config):

    ''' DATA LOADING'''
    if config['Parameters']['load_dataset']:
        print(config['Parameters']['output_dir'])
        ### Create the log file
        logging.basicConfig(
            filename='%s/log_file.log'% config['Parameters']['output_dir'],  # Name of the log file
            level=logging.DEBUG,         # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
            filemode='w')  # 'w' to overwrite the file, 'a' to append to the file
        
        ### Load the demographic information, check the filters (keys) and filter the subjects to be kept
        df_info = reading.read_info(config['Parameters']['info_path'])
        logging.info('""""'); logging.info('Loading demographic information')
        filters = utils.compare_pdkeys_list(df_info, config['Parameters']['filters'])
        logging.info('""""'); logging.info("Found Filters in the csv file: %s" % filters)
    
        ### Filter the list of participants based on the filters in the configuration file
        df, ls_subs = reading.filtered_dataframe(df_info, filters, config)
        logging.info("Number of subjects kept after filtering: %d" % len(ls_subs)); logging.info("\n"); logging.info('""""')
    
        ## Generate list of processing types to include
        process = list(config["Parameters"]["processing"].keys())
        logging.info("Included Processing: %s" % process)
    
        ### Load the matrices
        MatMat = {}; EucMat = {}; dict_df = {}
        for p,proc in enumerate(config["Parameters"]["processing"]):
            idxs_tmp = np.where((df[proc]==1)|(df[proc]=='1'))[0]
            df_tmp = df.iloc[idxs_tmp]
            print("Number of subjects with processing %s: %d" %(proc,len(idxs_tmp)))
            tmp_path = os.path.join(config["Parameters"]["data_dir"], config["Parameters"]["processing"][proc])
            MatMat[proc], EucMat[proc], df_info = reading.load_matrices(df_tmp, tmp_path, config['Parameters']['scale'], config['Parameters']['metric'])
            if proc=='example':
                new_subs = ['sub-IC' + str(i) for i in range(1, 71)]
                df_tmp = pd.DataFrame({'sub': new_subs, 'dwi':'dsi','age': 40, 'group':'HC', 'gender': 'M', 'Inclusion':1, 'example':1})
            dict_df[proc] = df_tmp 
        MatConc, EucConc, df_conc = utils.concatenate_processings(MatMat, EucMat, dict_df, config)
        logging.info("\n"); logging.info('""""')

        ''' BIAS INVESTIGATION'''
        ### Perform dimensionality reduction (via PCA) to look for evident bias in the connectivity matrices
        if config['BIAS']['perform_analysis']:
            logging.info('Starting Bias Analysis')
            if config['BIAS']['DimReduc']=="PCA":
                print(config['BIAS']['DimReduc'])
                #X_pca, fig, axs = ML.bias_PCAplot(MatMat, dict_df, config["BIAS"]["colors_key"], config["Parameters"]["processing"])
                X_pca, fig, axs = ML.bias_PCAplot_concat(MatConc, df_conc, config["BIAS"]["colors_key"])
            logging.info("\n"); logging.info('""""')
        
        ### Estimate the consensus matrices for each selected processing (uniform consensus and distance-base consensus)
        if config['CONSENSUS']['perform_analysis']:
            logging.info('Starting Consensus Estimation')
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
            Q_all_rotated_conc, P_all_rotated_conc, R_all_conc, scale_R_conc = ML.rotation_procrustes(Q_ind_conc, P_ind_conc, plot=True, title='Concatenated')
            Mat_recon_conc = ML.reconstruct_SC(An_ind_conc, df_conc, P_ind_conc, Q_ind_conc, plot=True, title='Concatenated')
            #for i in np.arange(3):
            #   save_fname = plot_rois(Q_ind[:,i,0], config["Parameters"]["scale"], config, vmin=-.15, vmax=.15, label='test_eigv%d'%i)
            #   save_fname = plot_rois(Q_all_rotated[:,i,0], config["Parameters"]["scale"], config, vmin=-.15, vmax=.15, label='test_rotated_eigv%d'%i)
            
            ### Aligning for every processing independently
            P_ind={}; Q_ind={}; Q_all_rotated={}; P_all_rotated={}; scale_R={}; R_all={}; Mat_recon={}; Ln_ind = {}; An_ind= {}
            for p,proc in enumerate(procs):
                P_ind[proc], Q_ind[proc], Ln_ind[proc], An_ind[proc] = ML.ind_normalized_lap(MatMat[proc], EucMat[proc], dict_df[proc], plot=True)
                Q_all_rotated[proc], P_all_rotated[proc], R_all[proc], scale_R[proc] = ML.rotation_procrustes(Q_ind[proc], P_ind[proc], plot=True, title='%s'%proc)
                Mat_recon[proc] = ML.reconstruct_SC(An_ind[proc], dict_df[proc], P_ind[proc], Q_ind[proc], k=100, plot=True, title='%s'%proc)

         
        if config['HARMONICS']['compare_harmonics']:
            #diff_norm = utils.compare_large_rotations(R_all)
            #diff_harm = utils.compare_harmonicwise_rotations(R_all)
            diff_eig = utils.diff_before_after_rotation(Q_ind_conc, Q_all_rotated_conc, df_conc)
            #print("Difference Norm (Frobenius):", np.mean(diff_norm))    
     
        if config['HARMONICS']['consensus']:
            procs = list(config["Parameters"]["processing"].keys())
            ConsConc = reading.reading_consensus(procs, config["Parameters"]["metric"], config["Parameters"]["output_dir"], dict_df)
            #ML.compare_matrices(ConsConc[:,:,0], ConsConc[:,:,1], dict_df[procs[0]], dict_df[procs[1]], procs[0], procs[1], plot=True)
            ### PROBLEM HERE BETWEEN THE IND CONS FOR CONSENSUS LOADING FOR PROCESSING EXAMPLE
            #df_cons = pd.DataFrame([df_conc.iloc[0], df_conc.iloc[-1]])
            #P_ind, Q_ind, Ln_ind, An_ind = ML.ind_normalized_lap(ConsConc, EucConc, df_cons, plot=True)   
            #Q_all_rotated, P_all_rotated, R_all, scale_R = ML.rotation_procrustes(Q_ind, P_ind, plot=True)
            #Mat_recon = ML.reconstruct_SC(An_ind, df_cons, P_ind, Q_ind, plot=True)
            #for c in np.arange(len(procs)):
            #    for i in np.arange(3):
            #        print('sub %d eigenv %d'%(c,i))
            #        save_fname = plot_rois(Q_ind[:,i,c], config["Parameters"]["scale"], config, vmin=-.15, vmax=.15, label='cons%d_eigv%d'%(c,i))
            #        save_fname = plot_rois(Q_all_rotated[:,i,c], config["Parameters"]["scale"], config, vmin=-.15, vmax=.15, label='cons%d_rotated_eigv%d'%(c,i))
            MNI_parcellation = os.path.join(os.getcwd(), 'data/lausanne2018.ctx+subc.scale2.maxprob_2x2x2.nii.gz')
            output_path = os.path.join(os.getcwd(), 'output/maps')
            EucDist, cort_rois, hemii = ML.ROIs_euclidean_distance(config["Parameters"]["scale"])
            ### next step takes time
            #MA.create_nii_activation_maps(Q_ind, MNI_parcellation , cort_rois + 1, output_path)
            #MA.binarize_map(output_path, 0.01)
            neurosynth = os.path.join(os.getcwd(), 'data/neurosynth_maps') 
            #MA.binarize_map(neurosynth, 0.01)
            ### Need here to correct for different sizes of nifti file 
            ### Careful: Takes time
            #index = MA.comparison_neurosynthmaps(output_path, neurosynth)
    
        if config["INDvsCONS"]["perform_analysis"]:
            #EucDist, cort_rois, hemii = ML.ROIs_euclidean_distance(config["Parameters"]["scale"])
            ### Concatenated matrices
            EucDist = np.mean(EucConc, axis=2)
            nbPerm = 20; ls_bins = [1,3,5,8]
            RandCons, df_random, ShuffIdxs = ML.generate_randomized_part_consensus(MatConc, nbPerm, EucDist, ls_bins)
            ### Before rotation
            eigenvalues_perm, eigenvalues_perm_mat, eigenvectors_perm, eigenvectors_perm_mat, labels_perm = ML.harmonics_randomized_part_consensus(MatConc, RandCons, nbPerm, EucDist, df_random, ls_bins)
            bin_variability = ML.plot_randomized_part_consensus(MatConc, eigenvectors_perm, nbPerm, labels_perm, ls_bins,plot=True, title='Concatenated')
            ### After Procrustes Rotation
            Q_all_rotated, P_all_rotated, R_all, scale_R = ML.rotation_procrustes(eigenvectors_perm, eigenvalues_perm, plot=False)
            bin_variability_rotated = ML.plot_randomized_part_consensus(MatConc, Q_all_rotated, nbPerm, labels_perm, ls_bins, plot=False, title='Concatenated')
            
            ### Not concatenanted matrices
            #procs = list(config["Parameters"]["processing"].keys())
            #print(procs)
            #RandCons_p = {}; df_random_p = {}; eigenvectors_perm_p = {}; eigenvalues_perm_p = {}; Q_all_rotated_p = {}; bin_variability_p={}; bin_variability_rotated_p = {}
            #for p,proc in enumerate(procs):
            #    RandCons_p[proc], df_random_p[proc], ShuffIdxs = ML.generate_randomized_part_consensus(MatMat[proc], nbPerm, EucDist, ls_bins)
            #    eigenvalues_perm_p[proc], eigenvalues_perm_mat, eigenvectors_perm_p[proc], eigenvectors_perm_mat, labels_perm = ML.harmonics_randomized_part_consensus(MatMat[proc], RandCons_p[proc], nbPerm, EucDist, df_random_p[proc], ls_bins)
            #    bin_variability_p[proc] = ML.plot_randomized_part_consensus(MatMat[proc], eigenvectors_perm_p[proc], nbPerm, labels_perm, ls_bins, plot=True, title='%s'%proc)
            #    Q_all_rotated_p[proc], P_all_rotated, R_all, scale_R = ML.rotation_procrustes(eigenvectors_perm_p[proc], eigenvalues_perm_p[proc], plot=False)
            #    bin_variability_rotated_p[proc] = ML.plot_randomized_part_consensus(MatMat[proc], Q_all_rotated_p[proc], nbPerm, labels_perm, ls_bins, plot=False, title='%s'%proc)

        plt.show(block=True)


if __name__ == "__main__":
    # Load the default configuration from the file
    config_path = 'config_PEP3.json'
    config_defaults = reading.check_config_file(config_path)
    
    #root = tk.Tk()
    #app = ConfigGUI(root, run_processing, config_defaults)
    #root.mainloop()
    run_processing(config_defaults)