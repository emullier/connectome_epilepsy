
import os
import pandas as pd
import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt

def compare_pdkeys_list(dataframe, list_keys):
    '''  Check with elements of the list are actual keys in the dataframe '''
    keys = dataframe.keys()
    avail_keys = [element for element in list_keys if element in keys]
    return avail_keys


def concatenate_processings(MatMat, EucMat, dict_df, config):
    MatConc = []; EucConc = []; df_conc = {}
    for p,proc in enumerate(config["Parameters"]["processing"]):
        if p==0:
            MatConc = MatMat[proc]
            EucConc = EucMat[proc]
            dict_df[proc]['proc'] = np.ones(np.shape(MatMat[proc])[2])*(p+1)
            df_conc = dict_df[proc]
            
            #df_conc['proc'] = np.ones(np.shape(dict_df[proc])[0])*p
        else:
            MatConc = np.concatenate((MatConc, MatMat[proc]),axis=2)
            EucConc = np.concatenate((EucConc, EucMat[proc]),axis=2)
            dict_df[proc]['proc'] = np.ones(np.shape(MatMat[proc])[2])*(p+1)
            df_conc = pd.concat([df_conc, dict_df[proc]], axis=0)
            #df_conc['proc'] = np.ones(len(dict_df[proc]))*p
    return MatConc, EucConc, df_conc




def compare_large_rotations(R_all):
    # Compute the Frobenius norm of the difference
    diff_norm = np.zeros(np.shape(R_all)[2])
    for v in np.arange(np.shape(R_all)[2]):
        diff_norm[v] = np.linalg.norm(np.mean(R_all,axis=2) - R_all[:,:,v])
    
    fig, axs = plt.subplots(1, 1, figsize=(9, 4))
    axs.plot(diff_norm, '*');  axs.set_title('Euclidean distance between mean rotation and individual participant')
    axs.set_xlabel('Participant number'); axs.set_ylabel('Euclidean distance')
    plt.show(block=False) 
    
    return diff_norm

def compare_harmonicwise_rotations(R_all):
    # Compute the Frobenius norfim of the difference
    diff_norm = np.zeros((np.shape(R_all)[1],np.shape(R_all)[2]))
    for s in np.arange(np.shape(R_all)[1]):
        mean_rotation = np.mean(R_all[:,s,:], axis=1) 
        for v in np.arange(np.shape(R_all)[2]):
            diff = mean_rotation - R_all[:, s, v]
            diff_norm[s,v] = np.linalg.norm(diff)
    mean_diff_norm = np.mean(diff_norm, axis=1)
    fig, axs = plt.subplots(1, 1, figsize=(9, 4))
    axs.plot(mean_diff_norm, '*');  axs.set_title('Average Euclidean distance between mean rotation and individual participant')
    #axs.boxplot(diff_norm);
    axs.set_xlabel('Eigenvectors'); axs.set_ylabel('Euclidean distance')
    plt.show(block=False) 
    
    return diff_norm

def diff_before_after_rotation(Q_ind, Q_all_rotated, df_conc):
    colors = ['r', 'b', 'g', 'y', 'm']
    diff_eig = np.zeros((np.shape(Q_ind)[0], np.shape(Q_ind)[2]))
    fig, axs = plt.subplots(2, 1, figsize=(9, 4))
    for s in np.arange(np.shape(Q_ind)[2]):
        for v in np.arange(np.shape(Q_ind)[0]):
            diff_eig[v,s] = np.linalg.norm(Q_ind[:,v,s] - Q_all_rotated[:,v,s])
        axs[0].plot(diff_eig[:,s], '.', c = colors[int(df_conc['proc'].iloc[s])]);  
    for v in np.arange(np.shape(Q_ind)[0]):
        #axs[1].plot(diff_eig[v,:], '.', c=np.array(df_conc['proc']), cmap='viridis');  
        scatter = axs[1].scatter(np.arange(np.shape(diff_eig)[1]), diff_eig[v, :], c=df_conc['proc'], cmap='viridis')

    axs[0].set_title('Difference Q - Qrotated (all eigenvectors, all subjects)')
    axs[0].set_ylabel('Euclidean distance'); axs[0].set_ylim([.90,1.1]); axs[0].set_xlabel('Eigenvectors')
    axs[1].set_ylabel('Euclidean distance'); axs[1].set_ylim([.90,1.1]); axs[1].set_xlabel('Subjects')
    plt.show(block=False) 
    
    return diff_eig


    

        