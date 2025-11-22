

''' This script computes the SDI values on the EEG data of the 17 patients (Rigoni,2023) using the individual structural connectivity matrices 
generated from the Geneva datasets (HC and EP with RTLE and LTLE). Based on these SDI values, several similarity metrics between all the possible
pairewise combinations of SDI individual matrix based are computed:
    - Pearson correlation between the SDI
    - Pearson correlation between the Laplacian eigenvectors
    - Spectral distance between the SC matrices
    - Frobenius distance between the SC matrices
    - Principal angles between the Laplacian eigenvectors
    - Hungarian aligned cosine similarity between the Laplacian eigenvectors
    - Cosine similarity between the SDI
    - Spearman correlation between the SDI  
Resulting sizes of similarity matrices: (Nb individual matrices) x (Nb individual matrices)

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

    Mat_corr_SDI_out = "OUTPUT/EPvsCTRL/SIM/Mat_corr_SDI_%s.npy"%(dat)
    Mat_cutoff_out = "OUTPUT/EPvsCTRL/SIM/Mat_cutoffFreq_%s.npy"%(dat)
    Mat_corr_lap_out = "OUTPUT/EPvsCTRL/SIM/Mat_corr_laplacian_%s.npy"%(dat)
    Mat_SpectralDistance_SC_out = "OUTPUT/EPvsCTRL/SIM/Mat_SpectralDistance_SC_%s.npy"%(dat)
    Mat_Frobenius_SC_out = "OUTPUT/EPvsCTRL/SIM/Mat_Frobenius_SC_%s.npy"%(dat)
    Mat_PrincipalAngles_lap_out = "OUTPUT/EPvsCTRL/SIM/Mat_PrincipalAngles_laplacian_%s.npy"%(dat)
    Mat_HungCosSimMat_lap_out = "OUTPUT/EPvsCTRL/SIM/Mat_HungCosSimMat_laplacian_%s.npy"%(dat)
    Mat_CosineSim_SDI_out = "OUTPUT/EPvsCTRL/SIM/Mat_CosineSim_SDI_%s.npy"%(dat)
    Mat_Spearmanr_SDI_out = "OUTPUT/EPvsCTRL/SIM/Mat_Spearmanr_SDI_%s.npy"%(dat)
    if os.path.exists(Mat_corr_lap_out):
        Mat_corr_SDI = np.load(Mat_corr_SDI_out)
        Mat_cutoffFreq = np.load(Mat_cutoff_out)
        Mat_corr_laplacian = np.load(Mat_corr_lap_out)
        Mat_SpectralDistance_SC = np.load(Mat_SpectralDistance_SC_out)
        Mat_Frobenius_SC = np.load(Mat_Frobenius_SC_out)
        Mat_PrincipalAngles_lap = np.load(Mat_PrincipalAngles_lap_out)
    
        Mat_HungCosSimMat_lap = np.load(Mat_HungCosSimMat_lap_out)
        Mat_CosineSim_SDI = np.load(Mat_CosineSim_SDI_out)
        Mat_Spearmanr_SDI = np.load(Mat_Spearmanr_SDI_out)
    else:
        Mat_corr_SDI = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        Mat_cutoffFreq = np.zeros((np.shape(matMetric)[2], len(X_RS_allPat)))
        Mat_corr_laplacian = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        Mat_SpectralDistance_SC = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        Mat_Frobenius_SC = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        Mat_PrincipalAngles_lap = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        Mat_HungCosSimMat_lap = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        Mat_CosineSim_SDI = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        Mat_Spearmanr_SDI = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        for m1 in np.arange(np.shape(matMetric)[2]):
            consensus = matMetric[:,:,m1]
            EucDist = np.load("DATA/EucMat_HC_dsi_number_of_fibers.npy")  
            ### Generate the harmonics
            P_ind_1, Q_ind_m1, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
            ### Estimate SDI
            SDI_m1 = np.zeros((118, len(X_RS_allPat)))
            ls_lat = []; 
            for p in np.arange(len(X_RS_allPat)):
                X_RS = X_RS_allPat[p]['X_RS']
                ls_lat.append(X_RS_allPat[p]['lat'][0])
                #idx_ctx = np.concatenate((np.arange(0,57), np.arange(59,116)))
                #PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind_m1, X_RS[idx_ctx,:,:])
                PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind_m1, X_RS)
                Mat_cutoffFreq[m1,p] = NN
                #SDI_m1[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS[idx_ctx,:,:], Q_ind_m1)
                SDI_m1[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind_m1)
            ls_lat = np.array(ls_lat)
        
            for m2 in np.arange(np.shape(matMetric)[2]):
                print('(%d, %d) / %d'%(m1,m2, np.shape(matMetric)[2]))
                consensus = matMetric[:,:,m2]
                EucDist = consensus    
                ### Generate the harmonics
                P_ind_2, Q_ind_m2, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist,  plot=False)
                ### Estimate SDI
                SDI_m2 = np.zeros((118, len(X_RS_allPat)))
                ls_lat =[]
                for p in np.arange(len(X_RS_allPat)):
                    X_RS = X_RS_allPat[p]['X_RS']
                    ls_lat.append(X_RS_allPat[p]['lat'][0])
                    #idx_ctx = np.concatenate((np.arange(0,57), np.arange(59,116)))
                    #PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind_m2, X_RS[idx_ctx,:,:])
                    #SDI_m2[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS[idx_ctx,:,:], Q_ind_m2)
                    PSD,NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind_m2, X_RS)
                    SDI_m2[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind_m2)
            
                SDI_m1 = SDI_m1 + 1e-05
                SDI_m2 = SDI_m2 + 1e-05
                #[r,p] = scipy.stats.pearsonr(SDI_m1.flatten(), SDI_m2.flatten())
                [r,p] = scipy.stats.pearsonr(np.mean(SDI_m1, axis=1), np.mean(SDI_m2, axis=1))
                Mat_corr_SDI[m1,m2] = r
                [r,p] = scipy.stats.pearsonr(Q_ind_m1[0:50,:].flatten(), Q_ind_m2[0:50,:].flatten())
                Mat_corr_laplacian[m1,m2] = r
                #distance = np.linalg.norm(SDI_m1.flatten() - SDI_m2.flatten(), ord=2)
                #Mat_corr_SDI[m1,m2] = distance
                Mat_SpectralDistance_SC[m1,m2] = np.linalg.norm(P_ind_1 - P_ind_2)
                Mat_Frobenius_SC[m1,m2] = np.linalg.norm(matMetric[:,:,m1] - matMetric[:,:,m2], ord='fro')
                Q1, _ = np.linalg.qr(P_ind_1.reshape(-1, 1))
                Q2, _ = np.linalg.qr(P_ind_2.reshape(-1, 1))
                Mat_PrincipalAngles_lap[m1,m2] = scipy.linalg.subspace_angles(Q1, Q2)
                aligned_sim_matrix, aligned_similarities, (row_ind, col_ind) = hungarian_aligned_cosine_similarity(Q_ind_m1, Q_ind_m2)
                Mat_HungCosSimMat_lap[m1, m2] = np.mean(aligned_sim_matrix)
                Mat_CosineSim_SDI[m1,m2] = cosine_similarity(SDI_m1.reshape(1, -1), SDI_m2.reshape(1, -1))[0, 0]
                rho, pval = spearmanr(SDI_m1.reshape(1, -1), SDI_m2.reshape(1, -1))
                Mat_Spearmanr_SDI[m1,m2] = rho
            
        np.save(Mat_corr_SDI_out, Mat_corr_SDI)
        np.save(Mat_cutoff_out, Mat_cutoffFreq)
        np.save(Mat_corr_lap_out, Mat_corr_laplacian)
        np.save(Mat_SpectralDistance_SC_out, Mat_SpectralDistance_SC)
        np.save(Mat_Frobenius_SC_out, Mat_Frobenius_SC)
        np.save(Mat_PrincipalAngles_lap_out, Mat_PrincipalAngles_lap)
        np.save(Mat_HungCosSimMat_lap_out, Mat_HungCosSimMat_lap)
        np.save(Mat_CosineSim_SDI_out, Mat_CosineSim_SDI)
        np.save(Mat_Spearmanr_SDI_out, Mat_Spearmanr_SDI)


        


