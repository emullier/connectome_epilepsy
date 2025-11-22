
''' This script loads the data from the respectives data directories of Geneva and Bonn datasets 
to save them as 3D structures to be called later by the different codes for SDI analysis and
graph network analysis and else. 

Last modified: EM, 11.03.2025 
Created: 11.03.2025, Emeline Mullier
University of Geneva & Lausanne University Hospital '''


import os
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import h5py


file = h5py.File('./DATA/27_SCHZ_CTRL_dataset.mat', 'r')
Connectomes = file['SC_FC_Connectomes']
ref = Connectomes['SC_number_of_fibers']
ref_array = np.array(ref['ctrl'][:])
data_list = []
ref = ref_array[1]
obj = file[ref[0]]  # ref[0] because each 'ref' is an array of shape (1,)
data = obj[()]      # obj[()] reads the data
data_list.append(data)
SC = np.array(data_list)
SC = np.squeeze(SC)
cort_rois = np.concatenate((np.arange(0,57), np.arange(59,116)))
cort_rois = np.concatenate((cort_rois, [62,63,126,127]))
SC= SC[:,cort_rois,:]; SC = SC[:, :, cort_rois]
print(np.shape(SC))
np.save('DATA/matMetric_SCHZ_CTRL.npy', SC)


### Parameters
HD_path = r"E:\\"
scale = 2
metric = 'fiber_length_mean' #"number_of_fibers", "normalized_fiber_density" # 'shore_gfa_mean' 

### Geneva datasets & demographics
DSI_HC_data_path = os.path.join(HD_path, "PROJECT_CONNECTOME_EPILEPSY/DATA/DSI_EEG_HC_BIDS") 
DSI_EP_data_path = os.path.join(HD_path, "PROJECT_CONNECTOME_EPILEPSY/DATA/DIFFUSION_GVA_BIDS_BVALMAX5000")
multishell_HC_data_path = os.path.join(HD_path, "PROJECT_CONNECTOME_EPILEPSY/DATA/Data_jon")
multishell_EP_data_path = os.path.join(HD_path, "PROJECT_CONNECTOME_EPILEPSY/DATA/dwi_tle_multish_jon")
infoGVA_path = './DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv'

### Bonn dataset
Bonn_data_path = "F:\Bonn_dataset_testCMP"

### Open datasets
Ale_data_path = os.path.join(HD_path, "PROJECT_CONNECTOME_EPILEPSY/DATA/data_ale/Individual_Connectomes.mat")
Sipes_data_path = os.path.join(HD_path, "PROJECT_CONNECTOME_EPILEPSY/DATA/data_Sipes/MICA_schaefer200_SCFC_struct.mat")


#### LOAD GENEVA DATASETS & GENERATE 3D STRUCTURES
#####################################################
df_info_orig = pd.read_csv(infoGVA_path)
idxs2keep = np.where((df_info_orig['Inclusion']==1))[0]
df_info_orig = df_info_orig.iloc[idxs2keep]

GVA_datasets = [DSI_HC_data_path, DSI_EP_data_path, multishell_HC_data_path, multishell_EP_data_path]
ls_groups = ['HC', 'EP', 'HC', 'EP']
ls_dwi = ['dsi', 'dsi', 'multishell', 'multishell']

#### NEED TO BE SAVING ALSO THE EUCLIDEAN DISTANCES MATRICES. 

roi_info_path = 'DATA/label/roi_info_l2018.xlsx'
roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
for d,dataset in enumerate(GVA_datasets):
    cmp_dir = os.path.join(dataset, 'derivatives', 'cmp-v3.1.0')
    idxs2keep = np.where((df_info_orig['group']==ls_groups[d])*(df_info_orig['dwi']==ls_dwi[d]))[0]
    df_info = df_info_orig.iloc[idxs2keep]
    if not os.path.exists(cmp_dir):
        print('%s does not exist'%(cmp_dir))
    k=0
    print(dataset)
    for s,sub in enumerate(df_info['sub']):
        print(sub)
        mat_path = os.path.join(cmp_dir, sub,  'dwi','%s_ses-SPUM_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub, scale)) 
        if not os.path.exists(mat_path):
            mat_path = os.path.join(cmp_dir, sub, 'dwi', '%s_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub, scale))  
            if not os.path.exists(mat_path):
                mat_path = os.path.join(cmp_dir, sub, 'ses-SPUM', 'dwi','%s_ses-SPUM_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub,scale)) 
        mat = sio.loadmat(mat_path)

        matMetric = mat['sc'][metric][0][0]
        print(np.shape(matMetric))
        Euc = mat['nodes']['dn_position'][0][0]
        ### TO remove subctx regions
        #matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
        #Euc = Euc[cort_rois,:]; 
        ### TO remove specific rois (Rigoni2023)
        df_118 = pd.read_csv('DATA/label/labels_rois_118.csv')
        rois_118 = df_118['ID Lausanne2008']
        rois_118 = np.array(rois_118)
        matMetric = matMetric[rois_118,:]; matMetric = matMetric[:, rois_118]
        Euc = Euc[rois_118,:];         
        nROIs = np.shape(matMetric)[0]
        shape = np.shape(matMetric) 
        if len(shape)>2:
            MatMat = matMetric
            EucMat = np.repeat(Euc[:, :, np.newaxis], np.shape(MatMat)[2], axis=2)
        else:
            if k==0:
                MatMat = np.zeros((len(matMetric), len(matMetric),len(df_info['sub'])))
                EucMat = np.zeros((len(matMetric), len(matMetric) ,len(df_info['sub'])))
                MatMat[:,:,k] = matMetric; EucMat[:,:,k] = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Euc, metric='euclidean'))
                k=k+1
            else:
                MatMat[:,:,k] = matMetric; EucMat[:,:,k] = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Euc, metric='euclidean'))
                k=k+1
    np.save('DATA/matMetric_%s_%s_%s.npy'%(ls_groups[d], ls_dwi[d], metric), MatMat)
    np.save('DATA/EucMat_%s_%s_%s.npy'%(ls_groups[d], ls_dwi[d], metric), EucMat)
    
