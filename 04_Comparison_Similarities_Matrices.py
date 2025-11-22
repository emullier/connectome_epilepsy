

''' This script plots the  similarity matrices between all the possible
pairewise combinations of SDI individual matrix based on the script 03_SDOI_individual_pipeline.py:

Last modified: EM, 20.11.2025 
Created: 11.03.2025, Emeline Mullier
University of Geneva & Lausanne University Hospital '''


import os
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec


metric = "number_of_fibers"
#ls_datasets = ['EP_DSI_%s'%metric, 'EP_multishell_%s'%metric, 'HC_DSI_%s'%metric, 'HC_multishell_%s'%metric]
ls_datasets = ['EP_DSI_%s'%metric,  'HC_DSI_%s'%metric]

concat_similarity_clus = []; concat_similarity_cent = []
concat_corr_lap = []; concat_corr_sdi = []
concat_dataset = []; concat_diff_cutoff = []

fig, axs = plt.subplots(3,len(ls_datasets), figsize=(25, 30))
for d,dat in enumerate(ls_datasets):
    print('Dataset %s'%dat)
    
    ### Matrix Level    
    Mat_similarity_clus_out = "OUTPUT/EPvsCTRL/SIM/Mat_similarity_clustering_%s.npy"%(dat)
    Mat_similarity_cent_out = "OUTPUT/EPvsCTRL/SIM/Mat_similarity_centrality_%s.npy"%(dat)
    Mat_SpectralDistance_SC = "OUTPUT/EPvsCTRL/SIM/Mat_SpectralDistance_SC_%s.npy"%(dat)
    Mat_Frobenius_SC_out = "OUTPUT/EPvsCTRL/SIM/Mat_Frobenius_SC_%s.npy"%(dat)
    Mat_similarity_clus = np.load(Mat_similarity_clus_out)
    Mat_similarity_cent = np.load(Mat_similarity_cent_out)
    Mat_Frobenius_SC = np.load(Mat_Frobenius_SC_out)
    Mat_SpectralDistance_SC = np.load(Mat_SpectralDistance_SC)
    
    upper_sim_clus = np.triu(Mat_similarity_clus, k=1); similarity_clus = upper_sim_clus[upper_sim_clus != 0]
    upper_sim_cent = np.triu(Mat_similarity_cent, k=1); similarity_cent  = upper_sim_cent[upper_sim_cent != 0]
    upper_Frob = np.triu(Mat_Frobenius_SC, k=1); similarity_Frob = upper_Frob[upper_Frob != 0]
    upper_SpectralDistance = np.triu(Mat_SpectralDistance_SC, k=1); similarity_SpectralDistance = upper_SpectralDistance[upper_SpectralDistance != 0]

    ### Laplacian level
    Mat_corr_lap_out = "OUTPUT/EPvsCTRL/SIM/Mat_corr_laplacian_%s.npy"%(dat)
    Mat_PrincipalAngles_lap_out = "OUTPUT/EPvsCTRL/SIM/Mat_PrincipalAngles_laplacian_%s.npy"%(dat)
    Mat_HungCosSimMat_lap_out = "OUTPUT/EPvsCTRL/SIM/Mat_HungCosSimMat_laplacian_%s.npy"%(dat)
    Mat_corr_lap = np.load(Mat_corr_lap_out)
    Mat_PrincipalAngles_lap = np.load(Mat_PrincipalAngles_lap_out)
    Mat_HungCosSimMat_lap = np.load(Mat_HungCosSimMat_lap_out)
    upper_sim_lap = np.triu(Mat_corr_lap,k=1); corr_lap = upper_sim_lap[upper_sim_lap != 0]
    upper_PrincipalAngles_lap = np.triu(Mat_PrincipalAngles_lap,k=1); similarity_PrincipalAngles_lap = upper_PrincipalAngles_lap[upper_PrincipalAngles_lap != 0]
    upper_HungCosSimMat_lap = np.triu(Mat_HungCosSimMat_lap,k=1); similarity_HungCosSimMat_lap = upper_HungCosSimMat_lap[upper_HungCosSimMat_lap != 0]
        
    ### SDI level
    Mat_corr_SDI_out = "OUTPUT/EPvsCTRL/SIM/Mat_corr_SDI_%s.npy"%(dat)
    Mat_cutoff_out = "OUTPUT/EPvsCTRL/SIM/Mat_cutoffFreq_%s.npy"%(dat)
    Mat_CosineSim_SDI_out = "OUTPUT/EPvsCTRL/SIM/Mat_CosineSim_SDI_%s.npy"%(dat)
    Mat_Spearmanr_SDI_out = "OUTPUT/EPvsCTRL/SIM/Mat_Spearmanr_SDI_%s.npy"%(dat)
    Mat_corr_SDI = np.load(Mat_corr_SDI_out)
    Mat_cutoff = np.load(Mat_cutoff_out)
    Mat_CosineSim_SDI = np.load(Mat_CosineSim_SDI_out)
    Mat_Spearmanr_SDI = np.load(Mat_Spearmanr_SDI_out)
    upper_sim_SDI = np.triu(Mat_corr_SDI, k=1); corr_sdi = upper_sim_SDI[upper_sim_SDI != 0]
    upper_CosineSim_SDI = np.triu(Mat_CosineSim_SDI, k=1); similarity_CosineSim_SDI = upper_CosineSim_SDI[upper_CosineSim_SDI != 0]
    upper_Spearmanr_SDI = np.triu(Mat_Spearmanr_SDI, k=1); similarity_Spearmanr_SDI = upper_Spearmanr_SDI[upper_Spearmanr_SDI != 0]
    
    ### Difference in cutoff frequency
    Mat_diff_cutoff = np.zeros((np.shape(Mat_cutoff)[0], np.shape(Mat_cutoff)[0]))
    for m1 in np.arange(np.shape(Mat_cutoff)[0]):
        for m2 in np.arange(np.shape(Mat_cutoff)[0]):
            Mat_diff_cutoff[m1,m2] = np.abs(Mat_cutoff[m1,:] - Mat_cutoff[m2,:]).mean()
    upper_diff_cutoff = np.triu(Mat_diff_cutoff, k=1)
    diff_cutoff = upper_diff_cutoff[upper_diff_cutoff != 0]
    

    fs = 10
    
    ### SC MEASURES
    similarity_SW = similarity_cent/similarity_clus
    
    ### Similarity SW vs Similarity Frobenius
    #axs[0,d].scatter(similarity_SW, similarity_Frob)
    #[r,p] = scipy.stats.pearsonr(similarity_SW, similarity_Frob)
    #axs[0,d].set_title('%s \n r=%.2g, p=%.2g'%(dat,r,p), fontsize=fs)
    #axs[0,d].set_xlabel('Similarity Smallworldness', fontsize=fs); axs[0,d].set_ylabel('Frobenius Norm', fontsize=fs);
    #axs[0,d].tick_params(axis='x', labelsize=fs); axs[0,d].tick_params(axis='y', labelsize=fs)
    
    ### Similarity SW vs Similarity SpectralDistance
    #axs[1,d].scatter(similarity_SW, similarity_SpectralDistance)
    #[r,p] = scipy.stats.pearsonr(similarity_SW, similarity_SpectralDistance)
    #axs[1,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)
    #axs[1,d].set_xlabel('Similarity Smallworldness', fontsize=fs); axs[1,d].set_ylabel('Spectral Distance', fontsize=fs);
    #axs[1,d].tick_params(axis='x', labelsize=fs); axs[1,d].tick_params(axis='y', labelsize=fs) 

    ### Similarity Frobenius vs Similarity SpectralDistance
    #axs[2,d].scatter(similarity_Frob, similarity_SpectralDistance)
    #[r,p] = scipy.stats.pearsonr(similarity_Frob, similarity_SpectralDistance)
    #axs[2,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)
    #axs[2,d].set_xlabel('Frobenius norm', fontsize=fs); axs[2,d].set_ylabel('Spectral Distance', fontsize=fs);
    #axs[2,d].tick_params(axis='x', labelsize=fs); axs[2,d].tick_params(axis='y', labelsize=fs)
    
    

    ### LAPLACIAN MEASURES
    ### Correlation laplacian vs Principal angles laplacian
    #axs[0,d].scatter(corr_lap, similarity_PrincipalAngles_lap)
    #[r,p] = scipy.stats.pearsonr(corr_lap, similarity_PrincipalAngles_lap)
    #axs[0,d].set_title('%s \n r=%.2g, p=%.2g'%(dat,r,p), fontsize=fs)
    #axs[0,d].set_xlabel('Correlation Laplacian', fontsize=fs); axs[0,d].set_ylabel('Principal Angles Laplacian', fontsize=fs);
    #axs[0,d].tick_params(axis='x', labelsize=fs); axs[0,d].tick_params(axis='y', labelsize=fs)
    
    ### Correlation laplacian vs Hung Cosine Similarity Laplacian
    #axs[1,d].scatter(corr_lap, similarity_HungCosSimMat_lap)        
    #[r,p]=scipy.stats.pearsonr(corr_lap, similarity_HungCosSimMat_lap)
    #axs[1,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)     
    #axs[1,d].set_xlabel('Correlation Laplacian', fontsize=fs); axs[1,d].set_ylabel('Hung Cosine Similarity Laplacian', fontsize=fs);
    #axs[1,d].tick_params(axis='x', labelsize=fs); axs[1,d].tick_params(axis='y', labelsize=fs)
        
    ### Principal angles laplacian vs Hung Cosine Similarity Laplacian
    #axs[2,d].scatter(similarity_PrincipalAngles_lap, similarity_HungCosSimMat_lap)
    #[r,p]=scipy.stats.pearsonr(similarity_PrincipalAngles_lap, similarity_HungCosSimMat_lap)
    #axs[2,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)
    #axs[2,d].set_xlabel('Principal Angles Laplacian', fontsize=fs); axs[2,d].set_ylabel('Hung Cosine Similarity Laplacian', fontsize=fs);
    #axs[2,d].tick_params(axis='x', labelsize=fs); axs[2,d].tick_params(axis='y', labelsize=fs)
    
    ### SDI MEASURES
    #similarity_Spearmanr_SDI = np.zeros(np.shape(similarity_Spearmanr_SDI))

    ### Correlation sdi vs Similarity Cosine SDI
    #axs[0,d].scatter(corr_sdi, similarity_CosineSim_SDI)
    #[r,p]=scipy.stats.pearsonr(corr_sdi, similarity_CosineSim_SDI)
    #axs[0,d].set_title('%s \n r=%.2g, p=%.2g'%(dat,r,p), fontsize=fs)
    #axs[0,d].set_xlabel('Correlation SDI', fontsize=fs); axs[0,d].set_ylabel('Cosine Similarity SDI', fontsize=fs);
    #axs[0,d].tick_params(axis='x', labelsize=fs); axs[0,d].tick_params(axis='y', labelsize=fs)

    ### Correlation laplacian vs Similarity Spearmanr SDI
    #axs[1,d].scatter(corr_sdi, similarity_Spearmanr_SDI)
    #[r,p]=scipy.stats.pearsonr(corr_sdi, similarity_Spearmanr_SDI)
    #axs[1,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)
    #axs[1,d].set_xlabel('Correlation SDI', fontsize=fs); axs[1,d].set_ylabel('Spearmanr SDI', fontsize=fs);
    #axs[1,d].tick_params(axis='x', labelsize=fs); axs[1,d].tick_params(axis='y', labelsize=fs)

    ### Similarity Cosine SDI vs Similarity Spearmanr SDI
    #axs[2,d].scatter(similarity_CosineSim_SDI, similarity_Spearmanr_SDI)
    #[r,p]=scipy.stats.pearsonr(similarity_CosineSim_SDI, similarity_Spearmanr_SDI)
    #axs[2,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)
    #axs[2,d].set_xlabel('Cosine Similarity SDI', fontsize=fs); axs[2,d].set_ylabel('Spearmanr SDI', fontsize=fs);
    #axs[2,d].set_xlim(0,1); axs[2,d].set_ylim(0,1)
    #axs[2,d].tick_params(axis='x', labelsize=fs); axs[2,d].tick_params(axis='y', labelsize=fs)

    ### SIMILARITY PROPAGATION
    ## Frobenius norm vs Hungarian Cosine Similarity
    #axs[0,d].scatter(similarity_Frob, similarity_HungCosSimMat_lap)
    #[r,p]=scipy.stats.pearsonr(similarity_Frob, similarity_HungCosSimMat_lap)
    #axs[0,d].set_xlabel('Frobenius norm', fontsize=fs); axs[0,d].set_ylabel('Hungarian Cosine Similarity', fontsize=fs);
    axs[0,d].scatter(similarity_SpectralDistance, similarity_HungCosSimMat_lap)
    [r,p]=scipy.stats.pearsonr(similarity_SpectralDistance, similarity_HungCosSimMat_lap)
    axs[0,d].set_xlabel('Spectral Distance', fontsize=fs); axs[0,d].set_ylabel('Hungarian Cosine Similarity', fontsize=fs);
    axs[0,d].set_title('%s \n r=%.2g, p=%.2g'%(dat,r,p), fontsize=fs)
    axs[0,d].tick_params(axis='x', labelsize=fs); axs[0,d].tick_params(axis='y', labelsize=fs)


    ### Hungarian Cosine Similarity vs Correlation SDI
    axs[1,d].scatter(similarity_HungCosSimMat_lap, corr_sdi)
    [r,p]=scipy.stats.pearsonr(similarity_HungCosSimMat_lap, corr_sdi)
    axs[1,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)    
    axs[1,d].set_xlabel('Hungarian Cosine Similarity', fontsize=fs); axs[1,d].set_ylabel('Correlation SDI', fontsize=fs);
    axs[1,d].tick_params(axis='x', labelsize=fs); axs[1,d].tick_params(axis='y', labelsize=fs)

    ### Frobenius norm vs Correlation SDI
    axs[2,d].scatter(similarity_Frob, corr_sdi)
    [r,p]=scipy.stats.pearsonr(similarity_Frob, corr_sdi)
    axs[2,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)
    axs[2,d].set_xlabel('Frobenius Norm', fontsize=fs); axs[2,d].set_ylabel('Correlation SDI', fontsize=fs);
    axs[2,d].tick_params(axis='x', labelsize=fs); axs[2,d].tick_params(axis='y', labelsize=fs)

    ### Similarity Cosine SDI vs Similarity Spearmanr SDI
    #axs[2,d].scatter(similarity_CosineSim_SDI, similarity_Spearmanr_SDI)
    #[r,p]=scipy.stats.pearsonr(similarity_CosineSim_SDI, similarity_Spearmanr_SDI)
    #axs[2,d].set_title(' r=%.2g, p=%.2g'%(r,p), fontsize=fs)
    #axs[2,d].set_xlabel('Cosine Similarity SDI', fontsize=fs); axs[2,d].set_ylabel('Spearmanr SDI', fontsize=fs);
    #axs[2,d].set_xlim(0,1); axs[2,d].set_ylim(0,1)
    #axs[2,d].tick_params(axis='x', labelsize=fs); axs[2,d].tick_params(axis='y', labelsize=fs)

    
    plt.subplots_adjust(hspace=0.5) 




plt.show()