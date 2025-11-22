

''' This script computes the network properties of each individual structural connectivity matrices 
from the Geneva datasets (HC and EP with RTLE and LTLE) and generate the similarity matrices (pearson correlation)
between regional clustering coefficients and betweenness centrality between all the possible pairewise combinations of individual matrices.
Resulting sizes of similarity matrices: (Nb individual matrices) x (Nb individual matrices)

Last modified: EM, 20.11.2025 
Created: 11.03.2025, Emeline Mullier
University of Geneva & Lausanne University Hospital '''

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import scipy.stats

metric = "number_of_fibers"
#ls_datasets = ['EP_DSI_%s'%metric, 'EP_multishell_%s'%metric, 'HC_DSI_%s'%metric, 'HC_multishell_%s'%metric]
ls_datasets = ['EP_DSI_%s'%metric,  'HC_DSI_%s'%metric]

for d,dat in enumerate(ls_datasets):
    print('Dataset %s'%dat)
    data_path = "DATA/matMetric_%s.npy"%dat
    matMetric = np.load(data_path)
    
    Mat_similarity_clus_out = "OUTPUT/EPvsCTRL/SIM/Mat_similarity_clustering_%s.npy"%(dat)
    Mat_similarity_cent_out = "OUTPUT/EPvsCTRL/SIM/Mat_similarity_centrality_%s.npy"%(dat)
    if os.path.exists(Mat_similarity_clus_out):
        Mat_similarity_clus = np.load(Mat_similarity_clus_out)
        Mat_similarity_cent = np.load(Mat_similarity_cent_out)
    else:
        Mat_similarity_clus = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))
        Mat_similarity_cent = np.zeros((np.shape(matMetric)[2],np.shape(matMetric)[2]))

        for m1 in np.arange(np.shape(matMetric)[2]):
            Mat1 = matMetric[:,:,m1]
            np.fill_diagonal(Mat1, 0)
            graph1 = nx.from_numpy_array(Mat1)
            clustering_coeffs = nx.clustering(graph1)
            betweenness_centrality = nx.betweenness_centrality(graph1, weight='weight')  # Use edge weights if available
            clus1=np.array(list(clustering_coeffs.values()))
            cent1=np.array(list(betweenness_centrality.values()))
            
            for m2 in np.arange(np.shape(matMetric)[2]):
                Mat2 = matMetric[:,:,m2]
                np.fill_diagonal(Mat2, 0)
                graph2 = nx.from_numpy_array(Mat2)
                clustering_coeffs = nx.clustering(graph2)
                betweenness_centrality = nx.betweenness_centrality(graph2, weight='weight')  # Use edge weights if available
                clus2=np.array(list(clustering_coeffs.values()))
                cent2=np.array(list(betweenness_centrality.values()))
            
                [r,p] = scipy.stats.pearsonr(clus1, clus2)
                Mat_similarity_clus[m1,m2] = r
                [r,p] = scipy.stats.pearsonr(cent1, cent2)
                Mat_similarity_cent[m1,m2]=r
        
        np.save(Mat_similarity_clus_out, Mat_similarity_clus)
        np.save(Mat_similarity_cent_out, Mat_similarity_cent)