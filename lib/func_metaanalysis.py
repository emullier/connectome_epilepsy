
import os
import numpy as np
import nibabel as nib
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

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
        img_map = np.abs(img_map)
        img_map[np.where(img_map<1e-5)] = 0
        img_bin = np.zeros(np.shape(img_map))
        #img_bin[np.where(img_map<th)] = 0
        img_bin[np.where(img_map>=th)] = 1
        new_nifti_image = nib.Nifti1Image(img_bin, map.affine, header=map.header)
        output = '%s_bin.nii.gz'%(file[0:-7])
        nib.save(new_nifti_image, output)
    

def comparison_neurosynthmaps(eigmaps_path, neurosynth_maps_path):
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
             
            
    
    
    
    
            