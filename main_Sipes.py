import os
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
from lib.func_harmonics import ev_zeroXings, zerocrossrate, match_eigs, perm_len, plot_iqr, matchpairs
import lib.func_reading as reading
import lib.func_utils as utils
from scipy.stats import kendalltau
import itertools


#ls_matMetric = ['matMetric_Ale.npy', 'matMetric_EP_DSI.npy', 'matMetric_HC_DSI.npy', 'matMetric_EP_multishell.npy', 'matMetric_HC_multishell.npy']
ls_matMetric = ['matMetric_Ale.npy','matMetric_Sipes.npy']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
labels = ['Independent', 'EP_DSI', 'HC_DSI', 'EP_multishell', 'HC_multishell']
unmatched_cost = 10


fig1, axes1 = plt.subplots(1,1)
fig2, axes2 = plt.subplots(2,2)
fig3, axes3 = plt.subplots(2,len(ls_matMetric))
fig4, axs4 = plt.subplots(3, len(ls_matMetric), figsize=(10, 15))

for m, metric in enumerate(ls_matMetric):

    matMetric = np.load('./data/%s'%metric)
    n_nodes = matMetric.shape[0]
    n_subjects = matMetric.shape[2]
    SC_ev_all = np.empty((n_nodes, n_nodes, n_subjects))

    for n in np.arange(n_subjects):
        L = laplacian(matMetric[:,:,n], normed=True)
        SC_ev, SC_U = np.linalg.eigh(L)
        SC_ev_all[:,:,n] = SC_ev
    
    SC_consensus = np.mean(matMetric, axis=2)
    SC_L = laplacian(SC_consensus, normed=True)
    SC_ev_consensus, SC_U_consensus = np.linalg.eigh(SC_L)


    ### FIGURE 1

    ### Eigenvalue Plot
    axes1.plot(SC_ev_consensus, linewidth=3, color=colors[m])
    median = np.median(SC_ev_all, axis=(0,2))
    q1 = np.percentile(SC_ev_all, 25, axis=(0,2))  # First quartile (25th percentile)
    q3 = np.percentile(SC_ev_all, 75, axis=(0,2))  # Third quartile (75th percentile)
    iqr = q3 - q1; x = np.arange(SC_ev_all.shape[1])  # X-values (index)
    axes1.plot(x, median, linestyle='-.', color=colors[m], label="Median")  # Median line
    axes1.fill_between(x, q1, q3,  alpha=0.2, label="IQR")  # IQR shaded area
    axes1.legend(['Consensus', 'Median', 'IQR'])
    axes1.set_xlabel('Harmonic Index'); axes1.set_ylabel('Eigenvalues');

    ### Visualize consensus harmonics on the brain
    ### TO BE ADDED

    ### Zerocrossrate, entropy, min-cut-max-flow, and support
    Adja = np.copy(SC_consensus)
    Adja[np.where(SC_consensus>0)] = 1
    fro_norm = np.linalg.norm(Adja, 'fro')
    Adja_norm = Adja / fro_norm
    RH = np.arange(0, int(n_nodes/2))
    LH = np.arange(int(n_nodes/2), n_nodes)
    Adja_L = Adja[LH,:]; Adja_L = Adja_L[:,LH] 
    Adja_R = Adja[RH,:]; Adja_R = Adja_R[:,RH]
    fro_L = np.linalg.norm(Adja_L, 'fro'); fro_R = np.linalg.norm(Adja_R, 'fro')
    Adja_L_norm = Adja_L/fro_L; Adja_R_norm = Adja_R/fro_R
    L_adja_L = laplacian(Adja_L_norm, normed=True); L_adja_R = laplacian(Adja_R_norm, normed=True)
    roughness_L = np.linalg.norm(L_adja_L @ SC_U_consensus[LH, :], axis=0)  # Norm for columns
    roughness_R = np.linalg.norm(L_adja_R @ SC_U_consensus[RH, :], axis=0)  # Norm for columns
    SC_ev_roughness = np.mean([roughness_L, roughness_R], axis=0)
    thresh = 1e-3
    SC_ev_zeroX = np.zeros(n_nodes)
    SC_ev_sparsity = np.zeros(n_nodes)
    SC_ev_mincut = ev_zeroXings(SC_consensus, SC_U_consensus, thresh)
    #print("SC_ev_mincut:", SC_ev_mincut)

    for i in np.arange(n_nodes):
        evec = SC_U_consensus[:,i]
        evec_thresh = evec
        evec_thresh[np.where(np.abs(evec_thresh)<=thresh)]=0
        SC_ev_zeroX[i] = zerocrossrate(evec_thresh)
        evec_thresh[np.where(np.abs(evec_thresh)>thresh)]=1
        SC_ev_sparsity[i] = (n_nodes - np.sum(evec_thresh))/n_nodes
    

    axes2[0,0].scatter(SC_ev_consensus, SC_ev_sparsity )
    axes2[0,0].set_ylabel('Sparsity'); axes2[0,0].set_xlabel('Eigenvalue');
    axes2[0,1].scatter(SC_ev_consensus, SC_ev_zeroX)
    axes2[0,1].set_ylabel('Zero-X Rate'); axes2[0,1].set_xlabel('Eigenvalue');
    axes2[1,0].scatter(SC_ev_consensus, SC_ev_mincut)
    axes2[1,0].set_ylabel('Net Zero-X'); axes2[1,0].set_xlabel('Eigenvalue');
    axes2[1,1].scatter(SC_ev_consensus, SC_ev_roughness)
    axes2[1,1].set_ylabel('Roughness'); axes2[1,1].set_xlabel('Eigenvalue');


    ### FIGURE 2
    SC_consensus = np.mean(matMetric, axis=2)
    SC_L = laplacian(SC_consensus, normed=True)
    SC_ev_consensus, SC_U_consensus = np.linalg.eigh(SC_L)
    
    U_all = np.zeros((n_nodes, n_nodes, n_subjects))
    U_all_matched = np.zeros((n_nodes, n_nodes, n_subjects))
    matched_order_all = np.zeros((n_nodes, n_subjects))
    ev_all_matched = np.zeros((n_nodes, n_subjects))
    UU_diag_matched2tmp = np.zeros((n_subjects,n_nodes))
    kendall_tau_corr = np.zeros((n_subjects,1))
    permutation_len = np.zeros((n_nodes,n_subjects))
    L = np.zeros((n_subjects,1))
    
    for n in np.arange(n_subjects):
        matMetric = np.load('./data/%s'%metric)
        SC = matMetric[:,:,n]
        SC_L = laplacian(SC, normed=True)
        ev_sub, U_all[:,:,n] = np.linalg.eigh(SC_L)
        U_all_matched[:,:,n], matched_order_all[:,n] = match_eigs(U_all[:,:,n], SC_U_consensus)
        #U_all_matched[:,:,n], matched_order_all[:,n] = matchpairs(U_all[:,:,n], SC_U_consensus, unmatched_cost)
        matched_order_all = np.array(matched_order_all, dtype=int)
        ev_all_matched[:,n] = ev_sub[matched_order_all[:,n]]
        result = np.dot(U_all_matched[:,:,n].T, SC_U_consensus)
        UU_diag_matched2tmp[n,:] = np.abs(np.diagonal(result))
        tau, _ = kendalltau(matched_order_all[:, n], np.arange(1, n_nodes + 1)); kendall_tau_corr[n] = tau
        print(kendalltau(matched_order_all[:, n], np.arange(1, n_nodes + 1)))
        permutation_len[:,n] , L[n] = perm_len(matched_order_all[:,n])
        
    combos = np.array(list(itertools.combinations(range(1, n_subjects+1), 2)))
    combos = np.transpose(combos)
    
    all_combo_diagonals = np.zeros((len(combos), n_nodes))
    all_combo_diagonals_matched2TMP =np.zeros((len(combos), n_nodes))
    all_combo_diagonals_matched2IND =np.zeros((len(combos), n_nodes))
    all_combo_kendall_tau_corr =np.zeros((len(combos),1))
    matched2IND_permutation_len =np.zeros((n_nodes, len(combos)))
    matched2IND_L =np.zeros((len(combos),1))

    for i in np.arange(len(combos)):
        U1 = U_all[:,:,combos[i,0]]
        U2 = U_all[:,:,combos[i,1]]
        all_combo_diagonals[i,:] = np.abs(np.diag(U1.T@U2))
        U1 = U_all_matched[:,:,combos[i,0]]
        U2 = U_all_matched[:,:,combos[i,1]]
        all_combo_diagonals_matched2TMP[i,:] = np.abs(np.diag(U1.T@U2))
        U1_matched2U2, matched2IND_order_all = match_eigs(U1, U2)
        all_combo_diagonals_matched2IND[i,:] = np.abs(np.diag(U1_matched2U2.T @ U2))
        matched2IND_permutation_len[:,i], matched2IND_L[i] = perm_len(matched2IND_order_all)
        col1 = matched_order_all[:, combos[i, 0]]  # Adjust for 0-indexing in Python
        col2 = matched_order_all[:, combos[i, 1]]  # Adjust for 0-indexing in Python
        tau, _ = kendalltau(col1, col2)
        all_combo_kendall_tau_corr[i] = tau
    
    
    sub = 0
    axes3[0,m].imshow(np.abs(U_all[:,:,sub].T @ SC_U_consensus)); axes3[0,m].set_title('%s'% labels[m])
    axes3[0,m].set_ylabel('Harmonics Before Matching'); 
    axes3[1,m].imshow(np.abs(U_all_matched[:,:,sub].T @ SC_U_consensus)); axes3[1,m].set_title('%s'%labels[m])
    axes3[1,m].set_ylabel('Harmonics Matched To Consensus'); 
    

    X = np.arange(0,n_nodes)
    plot_iqr(axs4[0,m], X, all_combo_diagonals, metric='median', fill=True, alpha=0.8)
    axs4[0,m].set_title('How similar are Individuals to each other? (Unmatched To consensus)')
    axs4[0,m].set_xlabel('Eigenvector Index')
    axs4[0,m].set_ylabel('UU Diagonal')
    axs4[0,m].set_xlim([0, n_nodes ])


    plot_iqr(axs4[1,m], X, all_combo_diagonals_matched2TMP, metric='median', fill=True, alpha=0.8)
    axs4[1,m].set_title('How similar are Individuals to each other? (Matched To consensus)')
    axs4[1,m].set_xlabel('Eigenvector Index')
    axs4[1,m].set_ylabel('UU Diagonal')
    axs4[1,m].set_xlim([0, n_nodes])


    plot_iqr(axs4[2,m], X, all_combo_diagonals_matched2IND, metric='median', fill=True, alpha=0.8)
    axs4[2,m].set_title('How similar are Individuals to each other? (Subs Matched 1-to-1)')
    axs4[2,m].set_xlabel('Eigenvector Index')
    axs4[2,m].set_ylabel('UU Diagonal')
    axs4[2,m].set_xlim([0, n_nodes])


    ### FIGURE 5
    

axes2[0,0].grid('on'); axes2[1,0].grid('on'); axes2[1,1].grid('on'); axes2[0,1].grid('on')
axes2[0,0].legend(labels); axes2[0,1].legend(labels)
axes2[1,0].legend(labels); axes2[1,1].legend(labels)

#print(U_all_matched[0:10,0:10,0])

plt.show(block=True)