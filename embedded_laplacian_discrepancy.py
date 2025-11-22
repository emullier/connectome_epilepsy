


#### Embedded Laplacian Discrepancy
#### https://arxiv.org/pdf/2201.12064

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def compute_laplacian_eigendecomposition(graph):
    L = nx.laplacian_matrix(graph).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues, eigenvectors

def construct_empirical_measure(eigenvalues, eigenvectors, n):
    measures = []
    for r in range(len(eigenvalues)):
        measure_points = np.concatenate([
            eigenvalues[r] * eigenvectors[:, r], 
            -eigenvalues[r] * eigenvectors[:, r]
        ])  # Dirac delta spikes at these points
        weights = np.ones_like(measure_points) / (2 * n)  # Uniform weights
        measures.append((measure_points, weights))  # Store as (locations, weights)
    return measures

import numpy as np
import networkx as nx

def estimate_max_wasserstein_distance(G1, G2):
    # Compute Laplacian matrices
    L1 = nx.laplacian_matrix(G1).toarray()
    L2 = nx.laplacian_matrix(G2).toarray()
    
    # Compute eigenvalues
    eigvals_G1 = np.linalg.eigvalsh(L1)  # Sorted automatically
    eigvals_G2 = np.linalg.eigvalsh(L2)
    
    # Compute the maximum possible Wasserstein distance
    lambda_max_G1 = eigvals_G1[-1]  # Largest eigenvalue of G1
    lambda_min_G1 = eigvals_G1[0]   # Smallest eigenvalue of G1
    
    lambda_max_G2 = eigvals_G2[-1]  # Largest eigenvalue of G2
    lambda_min_G2 = eigvals_G2[0]   # Smallest eigenvalue of G2
    
    # Maximum Wasserstein distance estimate
    W_max = max(abs(lambda_max_G1 - lambda_min_G2), abs(lambda_max_G2 - lambda_min_G1))
    
    return W_max



def embedded_laplacian_discrepancy(G1, G2, k):
    n1, n2 = len(G1.nodes), len(G2.nodes)
    k = min(k, n1, n2)  # Ensure k is valid
    
    # Compute Laplacian eigendecompositions
    eigvals_G1, eigvecs_G1 = compute_laplacian_eigendecomposition(G1)
    eigvals_G2, eigvecs_G2 = compute_laplacian_eigendecomposition(G2)
    
    # Construct empirical measures
    mu_G1 = construct_empirical_measure(eigvals_G1[:k], eigvecs_G1[:, :k], n1)
    nu_G2 = construct_empirical_measure(eigvals_G2[:k], eigvecs_G2[:, :k], n2)
    
    # Compute Wasserstein distances
  # Compute Wasserstein distances with weights
    wasserstein_distances = [wasserstein_distance(mu_G1[r][0], nu_G2[r][0], u_weights=mu_G1[r][1], v_weights=nu_G2[r][1]) 
    for r in range(k)]

    max_w_distance = estimate_max_wasserstein_distance(G1, G2)

    # Return average discrepancy
    #return np.mean(wasserstein_distances)/max_w_distance
    return np.mean(wasserstein_distances)

# Example usage
graph1 = nx.erdos_renyi_graph(10, 0.5)
graph2 = nx.erdos_renyi_graph(10, 0.5)
k = 5
similarity_score = embedded_laplacian_discrepancy(graph1, graph2, k)
print("Similarity Score between Erdo Renyi Graphs:", similarity_score)
max_w_distance = estimate_max_wasserstein_distance(graph1, graph2)
print("Estimated Maximum Wasserstein Distance:", max_w_distance)

# Differences between 2 conseneus
mat1_path = "DATA/matMetric_EP_DSI_number_of_fibers.npy"
mat2_path = "DATA/matMetric_HC_DSI_number_of_fibers.npy"
mat1 = np.load(mat1_path)
mat2 = np.load(mat2_path)
avgMat1 = np.mean(mat1, axis=2)
avgMat2 = np.mean(mat2, axis=2)
graph1 = nx.from_numpy_array(avgMat1)
graph2 = nx.from_numpy_array(avgMat2)
k = 5
similarity_score = embedded_laplacian_discrepancy(graph1, graph2, k)
print("Similarity Score between 2 Consensus:", similarity_score)


### Differences between individual graphs of each dataset
metric = "number_of_fibers"
ls_datasets = ['EP_DSI_%s'%metric, 'EP_multishell_%s'%metric, 'HC_DSI_%s'%metric, 'HC_multishell_%s'%metric]

fig,axs = plt.subplots(1,len(ls_datasets))
for d,dat in enumerate(ls_datasets):
    Mat_similarity_out = "OUTPUT/Mat_similarity_laplacian_%s.npy"%(dat)
    if os.path.exists(Mat_similarity_out):
        Mat_similarity_score = np.load(Mat_similarity_out)
    else:
        dataset_path = "DATA/matMetric_%s.npy"%dat
        data = np.load(dataset_path)
        Mat_similarity_score = np.zeros((np.shape(data)[2], np.shape(data)[2]))
        for m1 in np.arange(np.shape(data)[2]):
            for m2 in np.arange(np.shape(data)[2]):
                graph1 = nx.from_numpy_array(data[:,:,m1])
                graph2 = nx.from_numpy_array(data[:,:,m2])
                k = 5
                Mat_similarity_score[m1, m2] = embedded_laplacian_discrepancy(graph1, graph2, k)
        np.save(Mat_similarity_out, Mat_similarity_score)
    cax = axs[d].imshow(Mat_similarity_score); axs[d].set_title('Similarity between Laplacians \n %s'%dat)
    axs[d].set_xlabel('Subject ID'); axs[d].set_ylabel('Subject ID')
    fig.colorbar(cax, ax=axs[d], fraction=0.046, pad=0.04) 
plt.show()
