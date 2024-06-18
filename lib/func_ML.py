
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from lib import fcn_groups_bin

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
    plt.show()

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

    plt.tight_layout()
    plt.show()

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
            ims = axs[0].imshow(G); axs[0].set_title('Distance-based Consensus'); 
            ims = axs[1].imshow(Gc); axs[1].set_title('Uniform-based Consensus'); 
            fcn_groups_bin.plot_dist_distribution(axs[2], Mat, EucDist, nbins, G, Gc), axs[2].set_aspect('equal')           
        else:
            ims = axs[p,0].imshow(G); axs[p,0].set_title('Distance-based Consensus'); 
            ims = axs[p,1].imshow(Gc); axs[p,1].set_title('Uniform-based Consensus'); 
            fcn_groups_bin.plot_dist_distribution(axs[p,2], Mat, EucDist, nbins, G, Gc);
            axs[p, 2].set_title('')
    
    plt.show()
    return G_dist, G_unif