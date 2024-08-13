
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from lib import fcn_groups_bin
import itertools
import seaborn as sns
import pygsp
import random

def bias_PCAplot(MatMat, df_dict, colors_key, processings):
    fig, axs = plt.subplots(len(processings), len(colors_key)+1, figsize=(5 * (len(colors_key)+1), 5))
    
    ls_proc = []
    for p,proc in enumerate(processings):
        Mat = MatMat[proc]
        print(np.shape(Mat))
        ls_proc.extend(np.zeros(np.shape(Mat)[2])*p)
    ls_proc = np.array(ls_proc)
        
    for p,proc in enumerate(processings):
        Mat = MatMat[proc]
        df = df_dict[proc]
    
        X_vec = np.zeros((np.shape(Mat)[0]*np.shape(Mat)[0], np.shape(Mat)[2]))
        for s in np.arange(np.shape(MatMat[proc])[2]):
            X_vec[:,s] = Mat[:,:,s].flatten()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(np.transpose(X_vec))
    
        for c, color in enumerate(colors_key):
            colors = np.array(df[color])
            if isinstance(colors[0], str):
                uniq = np.unique(colors)
                color_mapping = {name: idx for idx, name in enumerate(uniq)}
                colors = np.array([color_mapping[col] for col in colors])
            
            if len(processings)==1:
                ax = axs[c]
            else:
                ax = axs[p,c]
                
            scatter = ax.scatter(X_pca[:,0], X_pca[:,1], 50, c=colors)
            ax.set_title('%s'% color), ax.set_xlabel('PCA Component 1'), ax.set_ylabel('PCA Component 2') 
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

        
        #    if c==len(colors_key)-1:
        #        ax = axs[p, c+1]
        #        ax.scatter(X_pca[:,0], X_pca[:,1], 50, c=ls_proc)
        #        ax.set_title('%s'% color), ax.set_xlabel('PCA Component 1'), ax.set_ylabel('PCA Component 2') 
        
        #fig.suptitle('Processing %s'% proc, fontsize=16)
        ax.annotate('Processing %s' % proc, xy=(0, 1), xytext=(0, 10), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline', fontsize=16)
        #fig.text(0.5, 0.98 - p*0.1, 'Processing %s' % proc, ha='center', va='top', fontsize=16)

    plt.tight_layout()
    plt.show(block=False)

    return X_pca, fig, axs


def bias_PCAplot_concat(MatConc, df_conc, colors_key):
    fig, axs = plt.subplots(1, len(colors_key)+1, figsize=(5 * (len(colors_key)+1), 5))
    Mat = MatConc
    df = df_conc
    X_vec = np.zeros((np.shape(Mat)[0]*np.shape(Mat)[0], np.shape(Mat)[2]))
    for s in np.arange(np.shape(Mat)[2]):
        X_vec[:,s] = Mat[:,:,s].flatten()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.transpose(X_vec))
    
    for c, color in enumerate(colors_key):
        colors = np.array(df[color])
        if isinstance(colors[0], str):
            uniq = np.unique(colors)
            color_mapping = {name: idx for idx, name in enumerate(uniq)}
            colors = np.array([color_mapping[col] for col in colors])
            
        scatter = axs[c].scatter(X_pca[:,0], X_pca[:,1], 50, c=colors)
        axs[c].set_title('%s'% color), axs[c].set_xlabel('PCA Component 1'), axs[c].set_ylabel('PCA Component 2') 
        fig.colorbar(scatter, ax=axs[c], fraction=0.046, pad=0.04)
        
        if c==len(colors_key)-1:
            axs[c+1].scatter(X_pca[:,0], X_pca[:,1], 50, c=list(df['proc']))
            axs[c+1].set_title('%s'% 'Processing'), axs[c+1].set_xlabel('PCA Component 1'), axs[c+1].set_ylabel('PCA Component 2') 

    plt.savefig('./public/images/BiasPCA.png')
    plt.tight_layout()
    plt.show(block=False)

    return X_pca, fig, axs


def ROIs_euclidean_distance(scale):
    roi_info_path = 'data/label/roi_info_l2018.xlsx'
    roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE {scale}')
    cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
    x = np.asarray(roi_info['x-pos'])[cort_rois]
    y = np.asarray(roi_info['y-pos'])[cort_rois]
    z = np.asarray(roi_info['z-pos'])[cort_rois]
    coordMat = np.concatenate((x[:,None],y[:,None],z[:,None]),1)
    EucDist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))
    hemii = np.ones(len(EucDist))
    hemii[int(len(hemii)/2):] = 2   
    return EucDist, cort_rois, hemii


def consensus_old(MatMat, processings, cort_rois, df_dict, EucDist, hemii, nbins):
    fig, axs = plt.subplots(len(processings),3)
    G_dist = {}; G_unif = {}
    for p,proc in enumerate(processings):
        Mat = MatMat[proc]
        Mat = Mat[cort_rois, :, :]
        Mat = Mat[:, cort_rois, :]
        df = df_dict[proc]
        [G, Gc] = fcn_groups_bin.fcn_groups_bin(Mat, EucDist, hemii, nbins) 
        G_dist[proc] = G
        G_unif[proc] = Gc
        print(G)
        
        if len(processings)==1:
            ims = axs[0].imshow(G); axs[0].set_title('Distance-based Consensus'); 
            ims = axs[1].imshow(Gc); axs[1].set_title('Uniform-based Consensus'); 
            fcn_groups_bin.plot_dist_distribution(axs[2], Mat, EucDist, nbins, G, Gc), axs[2].set_aspect('equal')           
        else:
            ims = axs[p,0].imshow(G); axs[p,0].set_title('Distance-based Consensus'); 
            ims = axs[p,1].imshow(Gc); axs[p,1].set_title('Uniform-based Consensus'); 
            fcn_groups_bin.plot_dist_distribution(axs[p,2], Mat, EucDist, nbins, G, Gc), axs[p,2].set_aspect('equal')
    return G_dist, G_unif


def consensus(MatMat, processings,  df_dict, EucMat, nbins):
    fig, axs = plt.subplots(len(processings),3)
    G_dist = {}; G_unif = {}
    for p,proc in enumerate(processings):
        Mat = MatMat[proc]
        df = df_dict[proc]
        EucDist = np.average(EucMat[proc], axis=2)
        hemii = np.ones(np.shape(EucDist)[0])
        #if np.shape(EucDist)[0]%2==0:
        hemii[int(len(hemii)/2):] = 2
        [G, Gc] = fcn_groups_bin.fcn_groups_bin(Mat, EucDist, hemii, nbins) 
        G_dist[proc] = G
        G_unif[proc] = Gc
        
        if len(processings)==1:
            ims = axs[0].imshow(G); axs[0].set_title('Distance-based Consensus'); axs[0].set_ylabel('%s'% proc)
            ims = axs[1].imshow(Gc); axs[1].set_title('Uniform-based Consensus'); 
            fcn_groups_bin.plot_dist_distribution(axs[2], Mat, EucDist, nbins, G, Gc), axs[2].set_aspect('equal')           
        else:
            ims = axs[p,0].imshow(G); axs[p,0].set_title('Distance-based Consensus'); axs[p,0].set_ylabel('%s'% proc)
            ims = axs[p,1].imshow(Gc); axs[p,1].set_title('Uniform-based Consensus'); 
            fcn_groups_bin.plot_dist_distribution(axs[p,2], Mat, EucDist, nbins, G, Gc);
            axs[p, 2].set_title('')
    
    plt.show(block=False)
    return G_dist, G_unif


def compare_matrices(Mat1, Mat2, df_mat1, df_mat2, proc1, proc2, plot=False):
    if (Mat1.ndim==2) and (Mat2.ndim==2):
        [r,p] = scipy.stats.pearsonr(Mat1.flatten(), Mat2.flatten())
        e = np.linalg.norm(Mat1 - Mat2) / np.sqrt(Mat1.size)
        MatDist_normError = e
        MatDist_corr = r
        if plot==True:
            fig, ax = plt.subplots(1,3, figsize=(20,5))
            nROIs = np.shape(Mat1)[0]
            half_nROIs = (nROIs -1 ) // 2
            hemisphere_array = np.array([1] * half_nROIs + [2] * half_nROIs + [3])
            connection_matrix = np.zeros((nROIs, nROIs), dtype=int)
            for i in range(nROIs):
                for j in range(nROIs):
                    if i == j:  # Skip self-loops if any
                        continue
                    if hemisphere_array[i] == hemisphere_array[j]:
                        connection_matrix[i, j] = 1  # Intra-hemispheric connection
                    else:
                        connection_matrix[i, j] = 2  # Inter-hemispheric connection
            ax[1].scatter(Mat1.flatten(), Mat2.flatten(),  50, c=connection_matrix.flatten(), cmap='tab10')
            ax[1].set_title('(r=%.3g, p=%.3g)'%(r,p)), ax[1].set_xlabel('%s'%proc1); ax[1].set_ylabel('%s'%proc2), ax[1].grid('on')
            ax[0].imshow(Mat1); ax[0].set_title('%s'%proc1); ax[2].imshow(Mat2); ax[2].set_title('%s'%proc2);
    else:
        MatDist_normError = np.zeros((np.shape(Mat1)[2], np.shape(Mat2)[2]))
        MatDist_corr = np.zeros((np.shape(Mat1)[2], np.shape(Mat2)[2]))
        for s1 in np.arange(np.shape(Mat1)[2]):
            for s2 in np.arange(np.shape(Mat2)[2]):
                ### pearson correlation
                [r,p] = scipy.stats.pearsonr(Mat1[:,:,s1].flatten(), Mat2[:,:,s2].flatten())
                ### normalized error
                e = np.linalg.norm(Mat1[:,:,s1] - Mat2[:,:,s2]) / np.sqrt(Mat1.size)
                MatDist_normError[s1,s2] = e
                MatDist_corr[s1,s2] = r
        if plot==True:
            fig, ax = plt.subplots(1,2)
            cax0 = ax[0].imshow(MatDist_normError, aspect='auto') 
            cax1 = ax[1].imshow(MatDist_corr, aspect='auto')
            ax[0].set_title('Normalized Error'); #fig.colorbar(cax0, ax=ax[0]),
            ax[1].set_title('Pearson Correlation'); #fig.colorbar(cax1, ax=ax[1])
            for t in [0,1]:
                axt = ax[t]
                axt.set_yticks(np.arange(0,len(list(df_mat1['sub'])))), axt.set_yticklabels(list(df_mat1['sub']), fontsize=6), axt.set_ylabel('%s'%proc1, fontsize=15)
                axt.set_xticks(np.arange(len(list(df_mat2['sub'])))), axt.set_xticklabels(list(df_mat2['sub']), rotation=45, ha='right', va='top', fontsize=6), axt.set_xlabel('%s'%proc2, fontsize=15)
        #fig.colorbar(caxt, ax=ax[axt])
            if proc1==proc2:
                fig.suptitle('Intra-processing, inter-subjects')
            else:
                fig.suptitle('Inter-processing')
    if plot==True:
        plt.show(block=False)
    return MatDist_normError, MatDist_corr


def inter_vs_intra_compare_matrices(MatMat, dict_df, processings):
    plt.rcParams['ytick.labelsize'] = 6
    nbperm = int((len(processings)*len(processings))/2)
    fig, ax = plt.subplots(2,nbperm+2, figsize=(10,30))
    k = 0
    ls_normError = []; ls_corr = []; ls_proc = []; legends = []
    for (p1, p2) in itertools.combinations_with_replacement(range(len(processings)), 2):
            MatDist_normError, MatDist_corr = compare_matrices(MatMat[processings[p1]], MatMat[processings[p2]], dict_df[processings[p1]], dict_df[processings[p2]], processings[p1], processings[p1], plot=False)
            cax0 = ax[0,k].imshow(MatDist_normError, aspect='auto')
            cax1 = ax[1,k].imshow(MatDist_corr, aspect='auto')
            ax[0,k].set_title('Normalized Error'); fig.colorbar(cax0, ax=ax[0,k])
            ax[1,k].set_title('Pearson Correlation'); fig.colorbar(cax1, ax=ax[1,k])
            df_mat1 = dict_df[processings[p1]]; df_mat2 = dict_df[processings[p2]]
            for t in [0,1]:
                axt = ax[t,k]
                axt.set_yticks(np.arange(0,len(list(df_mat1['sub'])))), axt.set_yticklabels(list(df_mat1['sub']), fontsize=4), axt.set_ylabel('%s'%processings[p1], fontsize=10)
                axt.set_xticks(np.arange(len(list(df_mat2['sub'])))), axt.set_xticklabels(list(df_mat2['sub']), rotation=45, ha='right', va='top', fontsize=4), axt.set_xlabel('%s'%processings[p2], fontsize=10)
            ls_normError.extend(MatDist_normError.flatten())
            ls_corr.extend(MatDist_corr.flatten())
            ls_proc.extend([k for _ in range(len(MatDist_corr.flatten()))])
            legends.append('%s-%s'%(processings[p1], processings[p2]))
            k = k +1
    ls_normError = np.array(ls_normError); ls_corr = np.array(ls_corr); ls_proc = np.array(ls_proc)
    
    ls_proc_norm = ls_proc[ls_normError != 0]
    ls_proc_corr = ls_proc[ls_corr != 1]
    ls_normError = ls_normError[ls_normError != 0]
    ls_corr = ls_corr[ls_corr != 1]
    
    normError_data = [ls_normError[ls_proc_norm == i] for i in np.unique(ls_proc_norm)]
    corr_data = [ls_corr[ls_proc_corr == i] for i in np.unique(ls_proc_corr)]
    
    # Boxplot for Normalized Error
    sns.stripplot(data=normError_data, ax=ax[0, nbperm + 1], color='k', size=2, jitter=True, alpha=.2)
    sns.boxplot(data=normError_data, ax=ax[0, nbperm+1],  boxprops=dict(alpha=.5), color='white',  width=.5, showfliers=False)
    if len(processings)==1:
        ax[0, nbperm + 1].set_title(f'Normalized Error', fontsize=10)
    else:
        kw_normError_stat, kw_normError_p = scipy.stats.kruskal(*normError_data)
        ax[0, nbperm + 1].set_title(f'Normalized Error\nKruskal-Wallis p={kw_normError_p:.1e}', fontsize=10)
    ax[0, nbperm + 1].grid('on'); ax[0, nbperm+1].set_xticklabels(legends,fontsize=7)
    
    # Boxplot for Pearson Correlation
    sns.stripplot(data=corr_data, ax=ax[1, nbperm + 1], color='k', size=2, jitter=True, alpha=.2)
    sns.boxplot(data=corr_data, ax=ax[1, nbperm+1], boxprops=dict(alpha=.5), color='white',  width=.5, showfliers=False)
    if len(processings)==1:
        ax[1, nbperm + 1].set_title(f'Pearson Correlation', fontsize=10)
    else:
        kw_corr_stat, kw_corr_p = scipy.stats.kruskal(*corr_data)
        ax[1, nbperm + 1].set_title(f'Pearson Correlation\nKruskal-Wallis p={kw_corr_p:.1e}', fontsize=10)
    ax[1, nbperm + 1].grid('on'); ax[1, nbperm+1].set_xticklabels(legends, fontsize=7)
    
    plt.show(block=False) 
    

def group_normalized_lap(MatMat, EucMat, procs, plot=False):
    P = {}; Q={}
    if plot==True:
        fig, axs = plt.subplots(len(procs),2, figsize=(10, 3))
    for p,proc in enumerate(procs):
        tmp = np.mean(MatMat[proc],axis=2)
        diag_zeros = np.diag(np.diag(tmp))
        tmp = tmp - diag_zeros
        sc = pygsp.graphs.Graph(tmp, lap_type='normalized', coords=np.mean(EucMat[proc],axis=2))
        sc.compute_fourier_basis()
        P[proc] = sc.e
        Q[proc] = sc.U
        if plot==True:
            im1 = axs[p,0].plot(P[proc])
            im1 = axs[p,0].set_title('%s \n Eigenvalues'%proc)
            im2 = axs[p,1].imshow(Q[proc], extent = [0,len(Q[proc]),0,len(Q[proc])], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            im2 = axs[p,1].set_title('%s \n Eigenvectors'%proc)
            im2 = axs[p,1].set_xlabel('eigenvalue index')
    if plot==True:
        plt.show(block=False) 
    return P,Q
    
def normalize_Lap(A):
    ''' Takes the adjacency matrix as input and returns the corresponding symmetric normalized Laplacian matrix'''
    indices_diag = np.diag_indices(len(A))
    A[indices_diag] = 0
    D = np.sum(A,axis=1)
    epsilon = 1e-10
    D = np.where(D == 0, epsilon, D)
    D = np.diag(D)
    Dn = np.power(D, -0.5)
    Dn = np.diag(np.diag(Dn))


    # symmetric normalize Adjacency
    An = Dn@A@Dn
    Ln = np.diag(np.full(len(An),1)) - An
    # Ln = np.diag(np.sum(An,axis=1)) - An
    return Ln, An
   
def ind_normalized_lap(MatMat, EucMat, df, plot=False):
    P = np.zeros((np.shape(MatMat)[0], len(df['sub']))); Q = np.zeros((np.shape(MatMat)[0], np.shape(MatMat)[0], len(df['sub'])))
    Ln = np.zeros((np.shape(MatMat)[0], np.shape(MatMat)[0], len(df['sub']))); An = np.zeros((np.shape(MatMat)[0], np.shape(MatMat)[0], len(df['sub'])));
      
    for s,sub in enumerate(list(df['sub'])):
        print(sub)
        tmp = MatMat[:,:,s]
        diag_zeros = np.diag(np.diag(tmp))
        tmp = tmp - diag_zeros
        Ln[:,:,s], An[:,:,s]  = normalize_Lap(tmp)
        #P[:,s], Q[:,:, s] = np.linalg.eigh(Ln[:,:,s]) 
        sc = pygsp.graphs.Graph(tmp, lap_type='normalized', coords=EucMat[:,:,s])
        sc.compute_fourier_basis()
        P[:,s] = sc.e
        Q[:,:,s] = sc.U
    return P, Q, Ln, An


def rotation_procrustes(Q_all, P_all,  plot=False, title=''):
    if np.shape(Q_all)[2]>1:
        Q_all_rotated = np.zeros(np.shape(Q_all))
        Q_all_rotated[:,:,0] = Q_all[:,:,0]
        R_all = np.zeros(np.shape(Q_all))
        scale_R = np.zeros(np.shape(Q_all)[2])
        Q_all[np.isnan(Q_all)]=0; Q_all[np.isinf(Q_all)]=0

        for i in range(1, np.shape(Q_all)[2]):
            _, Q_all_rotated[:,:,i], disparity = scipy.spatial.procrustes(Q_all[:,:,0], Q_all[:,:,i])
            R_all[:,:,i], _ = scipy.linalg.orthogonal_procrustes(Q_all[:,:,0], Q_all[:,:,i])    
        ### take the average of the rotated eigenvectors
        Q_all_rotated[np.isnan(Q_all_rotated)] = 1e-20
        Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
        ###second round of Procrustes transformation
        P_all_rotated = np.zeros((np.shape(Q_all)[0], np.shape(Q_all)[2]))
        for i in range(1, np.shape(Q_all)[2]):
            _, Q_all_rotated[:,:,i], disparity = scipy.spatial.procrustes(Q_mean_rotated, Q_all[:,:,i])
            R, scale_R[i] = scipy.linalg.orthogonal_procrustes(Q_mean_rotated, Q_all[:,:,i])    
            eig_rotated = R@Q_all[:,:,i]@np.diag(P_all[:,i])
            P_all_rotated[:,i] = np.sqrt(np.sum(np.multiply(eig_rotated,eig_rotated),axis=0))
        
            Q_mean_rotated = np.mean(Q_all_rotated,axis=2)
            P_mean = np.mean(P_all,axis=1); P_mean_rotated = np.mean(P_all_rotated, axis=1)


        if plot==True:
            fig, ax = plt.subplots(1,5, figsize=(20,3))            
            ax[0].imshow(Q_mean_rotated,  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[0].set_title('Average of rotated eigenvectors');  ax[0].set_aspect('equal')
            cax1 = ax[1].imshow(np.mean(Q_all, axis=2),  extent = [0,np.shape(Q_all)[2],0,np.shape(Q_all)[2]], aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
            ax[1].set_title('Average of original eigenvectors'); ax[1].set_aspect('equal')
            ax[2].plot(range(np.shape(Q_all)[0]), P_mean, range(np.shape(Q_all)[0]), P_mean_rotated)
            ax[2].set_title('Original and Rotated Eigenvectors '); ax[2].set_xlabel('eigenvalue index'); ax[2].set_ylabel('eigenvalues'); ax[2].legend(['Original Eigenvalues', 'Rotated Eigenvalues'])

        A = Q_all[:,:,0].T; B = Q_all[:,:,1].T; A_cos = np.dot(A, B.T)
        if plot==True:
            ax[3].imshow(A_cos,cmap = 'seismic',vmin = -1,vmax=1)
            ax[3].set_title('Cosine Similarity Before Rotation'); ax[3].set_xlabel('Subject 1 eigenvectors'), ax[3].set_ylabel('Subject 2 eigenvectors')
        
        A = Q_all_rotated[:,:,0].T; B = Q_all_rotated[:,:,1].T; A_cos = np.dot(A,B.T)
        if plot==True:
            ax[4].imshow(A_cos,cmap = 'seismic', vmin = -1,vmax=1); ax[4].set_title('Cosine Similarity After Rotation'); ax[4].set_xlabel('Subject 1 eigenvectors'); ax[4].set_ylabel('Subject 2 eigenvectors')        
            fig.suptitle('%s'%title); plt.show(block=False)
        
    return Q_all_rotated, P_all_rotated, R_all, scale_R


def reconstruct_SC(MatMat, df, P, Q, k=None, plot=False, title=''):
    Ln_group_recon = np.zeros(np.shape(MatMat))
    MatMat_recon = np.zeros(np.shape(MatMat))
    
    for s, sub in enumerate(list(df['sub'])):
        Qs = Q[:,:,s]
        Ps = P[:,s]
        
        if np.any(np.isnan(Qs)) or np.any(np.isinf(Qs)):
            print(f"NaN or Inf found in Q matrix slice {s}. Replacing with zeros.")
            Qs = np.nan_to_num(Qs)
        
        if k is not None:
            Qs = Qs[:, :k]  # Select the first k eigenvectors
            Ps = Ps[:k]     # Select the first k eigenvalues
        
        try:
            Q_pinv = np.linalg.pinv(Qs)
        except np.linalg.LinAlgError:
            print(f"SVD did not converge for slice {s}. Applying stronger regularization.")
            Q_pinv = np.linalg.pinv(Qs + np.eye(Qs.shape[0]) * 1e-8)
            if np.linalg.cond(Qs) > 1e10:  # Check the condition number
                print(f"Condition number is too high for slice {s}. Further regularization.")
                Q_pinv = np.linalg.pinv(Qs + np.eye(Qs.shape[0]) * 1e-6)
        
        Ln_group_recon[:,:,s] = Qs @ np.diag(Ps) @ Q_pinv
        MatMat_recon[:,:,s] = np.diag(np.full(len(MatMat[:,:,s]), 1)) - Ln_group_recon[:,:,s]
        MatMat_recon[:,:,s] = np.diag(np.diag(Ln_group_recon[:,:,s])) - Ln_group_recon[:,:,s]
    
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(9, 4))
        im1 = axs[0].imshow(MatMat_recon[:,:,1])
        axs[0].set_title('Reconstructed Normalized SC - Subject 1')
        im2 = axs[1].imshow(MatMat[:,:,1])
        axs[1].set_title('Raw Normalized SC - Subject 1')
        im3 = axs[2].scatter(MatMat[:,:,1], MatMat_recon[:,:,1])
        axs[2].set_title('Pearson correlation - Subject 1'); axs[2].set_xlabel('SC'); axs[2].set_ylabel('Reconstructed SC');
        fig.suptitle('%s' % title)
        plt.show(block=False)

    return MatMat_recon

    return MatMat_recon

def reconstruct_SC_part(MatMat, nbEig, df, P, Q, plot=False):
    Q = Q[:, :nbEig, :]; P = P[:nbEig,:]
    Ln_group_recon = np.zeros(np.shape(MatMat)); MatMat_recon = np.zeros(np.shape(MatMat))
    for s,sub in enumerate(list(df['sub'])):
        #Ln_group_recon[:,:,s] = Q[:,:,s]@np.diag(P[:,s])@np.linalg.inv(Q[:,:,s])
        Ln_group_recon[:,:,s] = Q[:,:,s]@np.diag(P[:,s])@np.linalg.pinv(Q[:,:,s]) ## use pseudo inverse because the matrix is not square anymore
        MatMat_recon[:,:,s] = np.diag(np.full(len(MatMat[:,:,s]),1)) - Ln_group_recon[:,:,s]
        MatMat_recon[:,:,s] = np.diag(np.diag(Ln_group_recon[:,:,s])) - Ln_group_recon[:,:,s]
    if plot==True:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        im1 = axs[0].imshow(MatMat_recon[:,:,1]); axs[0].set_title(' Reconstructed Normalized SC - Subject 1')
        im2 = axs[1].imshow(MatMat[:,:,1]); axs[1].set_title('Raw Normalized SC - Subject 1')
    plt.show(block=False)       

    return MatMat_recon


def generate_randomized_part_consensus(MatMat, nbPerm, EucDist, ls_bins):
    df_random = pd.DataFrame()
    ls_id = []
    hemii = np.ones(np.shape(EucDist)[0])
    hemii[int(len(hemii)/2):] = 2
    nSubs = np.shape(MatMat)[2]
    idxs = list(range(nSubs))
    RandCons = np.zeros((np.shape(MatMat)[0], np.shape(MatMat)[0], nbPerm, len(ls_bins)))
    ShuffIdxs = np.zeros((len(idxs), nbPerm, len(ls_bins)))
    for b,bi in enumerate(ls_bins):
        for p in np.arange(nbPerm):
            print('bin %d, perm %d' %(bi,p))
            random.shuffle(idxs)
            ShuffIdxs[:,p,b] = idxs
            idxs_tmp = idxs[0:bi]
            [G, Gc] = fcn_groups_bin.fcn_groups_bin(MatMat[:,:, idxs_tmp], EucDist, hemii, 40) 
            avg = np.mean(MatMat[:,:, idxs_tmp], 2) 
            RandCons[:,:,p,b] = G*avg
            ls_id.append('Bin_%d_Perm%d'%(bi,p))
    df_random['sub'] = ls_id   
    print('nROIs=%d, number of bins=%d, number of randomization=%d'%(np.shape(RandCons)[0], np.shape(RandCons)[3], np.shape(RandCons)[2]))
    return RandCons, df_random, ShuffIdxs
    
    
def harmonics_randomized_part_consensus(MatMat, RandCons, nbPerm, EucDist, df_random, ls_bins):
    nSubs = np.shape(MatMat)[2]; nROIs = np.shape(RandCons)[0]; labels_perm = []
    eigenvectors_perm = np.zeros((nROIs, nROIs, len(ls_bins)*nbPerm)); eigenvalues_perm = np.zeros((nROIs, len(ls_bins)*nbPerm))
    eigenvectors_perm_mat = np.zeros((nROIs, nROIs, len(ls_bins), nbPerm)); eigenvalues_perm_mat = np.zeros((nROIs, len(ls_bins), nbPerm))
    k = 0
    for b,bi in enumerate(ls_bins):
        for p in np.arange(nbPerm):
            tmp = RandCons[:,:,p,b]
            diag_zeros = np.diag(np.diag(tmp)); tmp = tmp - diag_zeros
            Ln, An  = normalize_Lap(tmp)
            sc = pygsp.graphs.Graph(tmp, lap_type='normalized', coords=EucDist); sc.compute_fourier_basis()
            eigenvalues_perm_mat[:, b, p] = sc.e;  eigenvalues_perm[:, k] = sc.e;  
            eigenvectors_perm_mat[:, :, b, p] = sc.U; eigenvectors_perm[:, :, k] = sc.U;
            labels_perm.append('Bin%d'%(bi))
            k = k+1  
    return eigenvalues_perm, eigenvalues_perm_mat, eigenvectors_perm, eigenvectors_perm_mat, labels_perm
        
def plot_randomized_part_consensus(MatMat, eigenvectors_perm, nbPerm, labels_perm, ls_bins, title='', plot=False):
    [nROIs, nROIs, nSubs] = np.shape(MatMat)
    ### Generate the corresponding labels
    labels_perm_bin = []
    for i in np.arange(len(labels_perm)):
        for j in np.arange(len(labels_perm)):  
            if labels_perm[i]==labels_perm[j]:
                labels_perm_bin.append('%s'%(labels_perm[i]))
            else:
                labels_perm_bin.append('Different bins');
    labels_perm_bin = np.array(labels_perm_bin)
    
    Dist_eigvec_perm = np.zeros((len(ls_bins)*nbPerm, len(ls_bins)*nbPerm, nROIs))
    for eigvec_nb in np.arange(nROIs):
        MatDist = 1 - scipy.spatial.distance.pdist(np.transpose(eigenvectors_perm[:, eigvec_nb,:]), metric='euclidean')
        Dist_eigvec_perm[:,:,eigvec_nb] = scipy.spatial.distance.squareform(MatDist)

    #### Remove the 0 values corresponding to the similarity between identical vectors
    Dist_eigvec_perm_vec = np.reshape(Dist_eigvec_perm, (len(ls_bins)*nbPerm*len(ls_bins)*nbPerm, nROIs))
    Dist_eigvec_perm_vec = np.abs(Dist_eigvec_perm_vec) ### Take absolute values for compensating for sign change 
    ### Remove the 0 values corresponding here to the diagonal
    for i in np.arange(nROIs):
        tmp = Dist_eigvec_perm_vec[:,i]; tmp = tmp[np.where(tmp>0)]
        if i==0:
            Dist_eigvec_perm_vec_nz = np.zeros((len(tmp), nROIs))
        Dist_eigvec_perm_vec_nz[:,i] = tmp 
    labels_perm_bin = labels_perm_bin[np.where(tmp>0)] 
    
    bin_variability = np.zeros((len(ls_bins), nROIs, 2))
    for b,bi in enumerate(ls_bins):
        idxs = np.where(labels_perm_bin=='Bin%d'%bi)[0]
        for i in np.arange(nROIs):
            bin_variability[b,i,0] = np.median(Dist_eigvec_perm_vec_nz[idxs,i])
            bin_variability[b,i,1] = np.std(Dist_eigvec_perm_vec_nz[idxs,i])
    
    if plot==True:
        fig, ax = plt.subplots(1,1,figsize=(15,5))
        handles = []  # To store the handles for lines in the plot
        for b,bi in enumerate(ls_bins):
            line, = ax.plot(bin_variability[b,:,0]); handles.append(line)
            upper_bound = bin_variability[b, :, 0] + bin_variability[b, :, 1]
            lower_bound = bin_variability[b, :, 0] - bin_variability[b, :, 1]
            ax.fill_between(range(nROIs), lower_bound, upper_bound, alpha=0.1)
        ax.set_xlabel('Eigenmode'); ax.set_xticks(range(0, nROIs, 10)); ax.set_ylim([0,1.1])
        ax.grid('on'); ax.legend(handles=handles, labels=ls_bins, loc='lower left', title='Bin');
        ax.set_ylabel('Correlation'); ax.set_title('Similarity between network harmonics', fontsize=15); 
        fig.suptitle('%s'%title)
    return bin_variability