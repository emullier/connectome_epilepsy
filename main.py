

import os
import lib.func_reading as reading
import lib.func_utils as utils
import logging
import numpy as np
import pandas as pd
import lib.func_ML as ML
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

config_path = 'config_PEP3.json'
config = reading.check_config_file(config_path)


''' DATA LOADING'''

if config['Parameters']['load_dataset']:
    ### Create the log file
    logging.basicConfig(
        filename='%s/log_file.log'% config['Parameters']['output_dir'],  # Name of the log file
        level=logging.DEBUG,         # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        filemode='w'  # 'w' to overwrite the file, 'a' to append to the file
    )
    
    #logging.debug('This is a debug message')
    #logging.info('This is an info message')
    #logging.warning('This is a warning message')
    #logging.error('This is an error message')
    #logging.critical('This is a critical message')

    
    ### Load the demographic information, check the filters (keys) and filter the subjects to be kept
    df_info = reading.read_info(config['Parameters']['info_path'])
    print('""""')
    print('Loading demographic information')
    filters = utils.compare_pdkeys_list(df_info, config['Parameters']['filters'])
    print('""""')
    print("Found Filters in the csv file: %s" % filters)
    ##logging.info("Found Filters in the csv file: %s" % filters)
    
    df, ls_subs = reading.filtered_dataframe(df_info, filters, config)
    print("Number of subjects kept after filtering: %d" % len(ls_subs))
    print("\n")
    print('""""')
    
    ## Generate list of processing types to include
    process = list(config["Parameters"]["processing"].keys())
    print("Included Processing: %s" % process)
    
    MatMat = {}; EucMat = {}; dict_df = {}
    for p,proc in enumerate(config["Parameters"]["processing"]):
        idxs_tmp = np.where(df[proc]=='1')[0]
        df_tmp = df.iloc[idxs_tmp]
        print("Number of subjects with processing %s: %d" %(proc,len(idxs_tmp)))
        tmp_path = os.path.join(config["Parameters"]["data_dir"], config["Parameters"]["processing"][proc])
        MatMat[proc], EucMat[proc] = reading.load_matrices(df_tmp, tmp_path, config['Parameters']['scale'], config['Parameters']['metric'])
        dict_df[proc] = df_tmp 

    MatConc, EucConc, df_conc = utils.concatenate_processings(MatMat, EucMat, dict_df, config)
    print(df_conc['proc'])
    
    print("\n")
    print('""""')


    ''' BIAS INVESTIGATION'''
    if config['BIAS']['perform_analysis']:
        print('Starting Bias Analysis')
        if config['BIAS']['DimReduc']=="PCA":
            print(config['BIAS']['DimReduc'])
            #X_pca, fig, axs = ML.bias_PCAplot(MatMat, dict_df, config["BIAS"]["colors_key"], config["Parameters"]["processing"])
            X_pca, fig, axs = ML.bias_PCAplot_concat(MatConc, df_conc, config["BIAS"]["colors_key"])
    print("\n")
    print('""""')

    if config['CONSENSUS']['perform_analysis']:
        print('Starting Consensus Estimation')
        # EucDist, cort_rois, hemii = ML.ROIs_euclidean_distance(config["Parameters"]["scale"])
        # G_dist, G_unif = ML.consensus(MatMat, config["Parameters"]["processing"], cort_rois, dict_df, EucDist, hemii, config["CONSENSUS"]["nbins"])
        G_dist, G_unif = ML.consensus(MatMat, config["Parameters"]["processing"],  dict_df, EucMat, config["CONSENSUS"]["nbins"])
        reading.save_consensus(MatMat, config["Parameters"]["metric"], G_dist, G_unif, config["CONSENSUS"]["out_dir"], config["Parameters"]["processing"])
        
        
        
        