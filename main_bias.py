
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
matplotlib.use('TkAgg')


def run_bias_analysis(config):

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
                df_tmp = pd.DataFrame({'sub': new_subs, 'dwi':'dsi', 'age': 40, 'group':'HC', 'gender': 'M', 'Inclusion':1, 'example':1})
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
                X_pca, fig, axs = ML.bias_PCAplot_concat(MatConc, df_conc, config["BIAS"]["colors_key"], title='Concatenated')
            for p,proc in enumerate(config["Parameters"]["processing"]):
                ML.bias_PCAplot_concat(MatMat[proc], dict_df[proc], config["BIAS"]["colors_key"], title=proc)
            logging.info("\n"); logging.info('""""')
        

if __name__ == "__main__":
    # Load the default configuration from the file
    config_path = 'config.json'
    config_defaults = reading.check_config_file(config_path)
    
    run_bias_analysis(config_defaults)