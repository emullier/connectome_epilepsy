
''' This script computes the SDI values on the EEG data of the 17 patients (Rigoni,2023) using the individual structural connectivity matrices 
generated from the Geneva datasets (HC and EP with RTLE and LTLE)
Outputs: N matrices (Nb of individual matrices) of dimensions (nROIs x 17), 17 number of EEG patients

Last modified: EM, 20.11.2025 
Created: 11.03.2025, Emeline Mullier
University of Geneva & Lausanne University Hospital '''


import os
import numpy as np
import matplotlib.pyplot as plt
import lib.func_GSP as gsp
from lib.func_GSP import hungarian_aligned_cosine_similarity
from lib.func_plot import plot_rois, plot_rois_pyvista
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

metric = "number_of_fibers"
#ls_datasets = ['EP_DSI_%s'%metric, 'EP_multishell_%s'%metric, 'HC_DSI_%s'%metric, 'HC_multishell_%s'%metric]
ls_datasets = ['EP_DSI_%s'%metric, 'HC_DSI_%s'%metric]

example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"
X_RS_allPat = gsp.load_EEG_example(example_dir)

for d,dat in enumerate(ls_datasets):
    print('Dataset %s'%dat)
    data_path = "DATA/matMetric_%s.npy"%dat
    matMetric = np.load(data_path)

    for m1 in np.arange(np.shape(matMetric)[2]):
        consensus = matMetric[:,:,m1]
        EucDist = consensus    
        ### Generate the harmonics
        P_ind_1, Q_ind_m1, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
        ### Estimate SDI
        SDI_m1 = np.zeros((118, len(X_RS_allPat)))
        ls_lat = []; 
        for p in np.arange(len(X_RS_allPat)):
            X_RS = X_RS_allPat[p]['X_RS']
            ls_lat.append(X_RS_allPat[p]['lat'][0])
            PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind_m1, X_RS)
            SDI_m1[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind_m1)
        ls_lat = np.array(ls_lat)
        
        print(np.shape(SDI_m1))
        np.save("TMP_OUTPUT/SDI_%s_%d.npy"%(dat,m1), SDI_m1)
        
        

        


