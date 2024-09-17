
import os
import numpy as np
import nibabel as nib
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import nimare
from nimare.dataset import Dataset
from nimare.decode.continuous import CorrelationDecoder
import pandas as pd
import seaborn as sns
from nimare.extract import fetch_neurosynth
import joblib
from scipy.stats import zscore



def create_nii_activation_maps(eigenvectors, MNI_parcellation, rois, output_path):
    nii_L2018 = nib.load(MNI_parcellation)
    image_data = nii_L2018.get_fdata()
    for s in np.arange(np.shape(eigenvectors)[2]):
        for v in np.arange(len(eigenvectors)):
            print('Eigenvector %d - sub %d'% (v, s))
            tmp = np.zeros(np.shape(image_data))
            fname = '%s/eigv%d_sub%d.nii.gz' % (output_path, v, s)
            for r,roi in enumerate(rois):
                tmp[np.where(image_data==roi)] = eigenvectors[r,v,s]
            tmp[np.where(image_data)==0] = 0
            new_nifti_image = nib.Nifti1Image(tmp, nii_L2018.affine, header=nii_L2018.header)
            nib.save(new_nifti_image, fname)
            

def binarize_map(map_path, th):
    tmp = os.listdir(map_path)
    ls_maps = []
    for m, map in enumerate(tmp):
        if 'bin' in map:
            pass
        elif map.endswith('.nii.gz'):
            ls_maps.append('%s/%s'%(map_path,map))
        else: 
            pass
    for f,file in enumerate(ls_maps):
        print('binarization of %s'%file)
        map = nib.load(file)
        img_map = map.get_fdata()
        zscored_image_flattened = zscore(img_map)
        img_map = zscored_image_flattened.reshape(img_map.shape)
        img_map = np.abs(img_map)
        img_map[np.where(img_map<1e-5)] = 0
        img_bin = np.zeros(np.shape(img_map))
        #img_bin[np.where(img_map<th)] = 0
        img_bin[np.where(img_map>=th)] = 1
        new_nifti_image = nib.Nifti1Image(img_bin, map.affine, header=map.header)
        output = '%s_bin.nii.gz'%(file[0:-7])
        nib.save(new_nifti_image, output)
    

def comparison_neurosynthmaps_old(eigmaps_path, neurosynth_maps_path):
    ### requires the binary maps
    tmp = os.listdir(eigmaps_path)
    ls_eigmaps = []; ls_neuro = []
    for e,eig in enumerate(tmp):
        if eig.endswith('_bin.nii.gz'):
            ls_eigmaps.append('%s/%s'%(eigmaps_path,eig))
    tmp = os.listdir(neurosynth_maps_path)
    for n,neuro in enumerate(tmp):
        if neuro.endswith('_bin.nii.gz'):
            ls_neuro.append('%s/%s'%(neurosynth_maps_path, neuro))
    index = np.zeros((len(ls_eigmaps), len(ls_neuro)))
    for e,eig in enumerate(ls_eigmaps):
        eigmap = nib.load(eig); im_eig = eigmap.get_fdata(); im_eig = (im_eig > 0).astype(int)
        #print(np.unique(im_eig))
        #print(np.shape(eigmap))
        for n,neuro in enumerate(ls_neuro):
            neuromap = nib.load(neuro); im_neuro = neuromap.get_fdata(); im_neuro = (im_neuro > 0).astype(int)
            assert np.all(np.isin(im_eig.flatten(), [0, 1])), "matrix1 must contain only 0 and 1"
            assert np.all(np.isin(im_neuro.flatten(), [0, 1])), "matrix2 must contain only 0 and 1"
            #print(np.shape(neuromap))
            index[e,n] = jaccard_score(im_eig.flatten(), im_neuro.flatten(), average='binary')
            #print(index[e,n])
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    cax = ax.imshow(index, aspect='auto', cmap='jet', vmin = -0.1,vmax=0.1)
    ax.set_title('Jaccard index');  ax.set_aspect('equal')
    #ax.set_yticks(np.arange(len(ls_eigmaps)), labels=ls_eigmaps); ax.set_xticks(np.arange(len(ls_neuro)), labels=ls_neuro)
    ax.set_yticks(np.arange(len(ls_eigmaps))); ax.set_xticks(np.arange(len(ls_neuro)))
    plt.colorbar(cax, ax=ax)
    return index

def getOrder(d, thr):
    dh = []
    for i in range(0,len(d)):
        di = d[i]
        dh.append(np.average(np.array(range(0,len(d[i]))) + 1, weights=di))
    heatmapOrder = np.argsort(dh)
    return heatmapOrder 
            
def comparison_neurosynthmaps(eigmaps_path, neurosynth_maps_path, p=''):
    maps = []
    tmp = os.listdir(eigmaps_path)
    for f,file in enumerate(tmp):
        if file.endswith('_sub0_bin.nii.gz'):
        #if file.endswith('_sub0.nii.gz'):
            maps.append('%s/%s'%(eigmaps_path,file))
    maps = np.array(maps)
    
    #maps = [str(eigmaps_path + f'/eigv{i}_sub0_bin.nii.gz') for i in range(0, 114)]
    nimare_dset_dir = 'data/neurosynth_dataset.pkl'
    
    if os.path.exists(nimare_dset_dir):
        dset = Dataset.load(nimare_dset_dir)
    else:
        dset_dir = 'data/neurosynth_topic50'  
        files = fetch_neurosynth(data_dir=dset_dir,  version='7',overwrite=False,  vocab="LDA50" )
        neurosynth_db = files[0]
        dset = nimare.io.convert_neurosynth_to_dataset(coordinates_file=neurosynth_db["coordinates"],metadata_file=neurosynth_db["metadata"],annotations_files=neurosynth_db["features"],)
        dset.save(nimare_dset_dir)

        
    # Decoding using NiMARE
    df_results = {}
    #for m, map in enumerate(maps):
    for m,map in enumerate(maps):
        map_name = 'output/neurosynth_results/df_results_map%d.csv'%m
        if os.path.exists(map_name):
            print('Map=%d exists'%m)
        else: 
            print('Processing map=%d'%m)
            decoder = nimare.decode.discrete.ROIAssociationDecoder(masker=maps[m])
            #decoder = CorrelationDecoder(target_image=maps[m])
            decoder.fit(dset)
            decoding_results = decoder.transform()
            df_results[map] = decoding_results.copy() 
            df_results[map].to_csv('output/neurosynth_results/df_results_map%d.csv'%m)
    
    tmp = os.listdir('output/neurosynth_results')
    results_maps = []
    for f,file in enumerate(tmp):
        if file.startswith('df_results_map'):
            results_maps.append('output/neurosynth_results/%s'%(file))
    results_maps = np.array(results_maps)

    thr=-1; names_maps = []; 
    for re,result in enumerate(results_maps):
        df_tmp = pd.read_csv(result)
        if re==0:
            prefix = "LDA50_abstract_weight__"
            data = np.zeros((len(df_tmp), len(results_maps)))
        df_tmp['feature'] = df_tmp['feature'].str.replace(f'^{prefix}', '', regex=True)
        name_neurosynth = np.array(df_tmp['feature'].copy())
        df_tmp[df_tmp['r']<thr] = 0 
        data[:,re] = np.array(df_tmp['r'])
        names_maps.append('eigenvector%d'%re) 
    names_maps = np.array(names_maps)
   
    df_plot = pd.DataFrame(data)
    #topics_to_keep = [0,2,3,5,6,8,9,15,16,17,18,19,20,23,24,26,28,30,32,33,37,38,41,42,43,44,47,48]
    sns.set(context="paper", font="sans-serif", font_scale=2)
    f, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)
    sns.heatmap(data,  cbar=True, yticklabels=name_neurosynth, xticklabels=names_maps, ax=ax1, linewidths=1, square=True, cmap='Greys', robust=False)
    plt.savefig('./public/static/images/ComparisonNeurosynth%d.png'%p)
    
    return results_maps    
    
    
    
            
     