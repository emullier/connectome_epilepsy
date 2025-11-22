

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy

metric = "number_of_fibers"
#ls_datasets = ['EP_DSI_%s'%metric, 'EP_multishell_%s'%metric, 'HC_DSI_%s'%metric, 'HC_multishell_%s'%metric]
ls_datasets = ['EP_DSI_%s'%metric, 'HC_DSI_%s'%metric]  

for d,dat in enumerate(ls_datasets):
    print('Dataset %s'%dat)
    data_path = "DATA/matMetric_%s.npy"%dat
    matMetric = np.load(data_path)

    n_nodes = 118
    clustering_coeffs = np.zeros((np.shape(matMetric)[2], n_nodes))
    betweenness_centralities = np.zeros((np.shape(matMetric)[2], n_nodes))
    for m1 in np.arange(np.shape(matMetric)[2]):
        SDI_m1 = np.load("TMP_OUTPUT/SDI_%s_%d.npy"%(dat,m1))
        roi_SDI = np.mean(SDI_m1, axis=1)
        G = nx.from_numpy_array(matMetric[:,:,m1])
        clustering_dict = nx.clustering(G)
        clustering_coeffs[m1,:] = np.array([clustering_dict[n] for n in sorted(clustering_dict)])
        betweenness_dict = nx.betweenness_centrality(G, normalized=True)
        betweenness_centralities[m1,:] = np.array([betweenness_dict[n] for n in sorted(betweenness_dict)])
    
        [r,p] = scipy.stats.pearsonr(clustering_coeffs[m1,:], roi_SDI)
        print(r)
        
        #fig, ax = plt.subplots(1,1)
        
