
import os
import numpy as np
import json
import pandas as pd
import scipy.io as sio 
import scipy


def check_config_file(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config

def read_info(info_path):
    df_info = pd.read_csv(info_path)
    idxs2keep = np.where(df_info['Inclusion']==1)[0]
    df_info = df_info.iloc[idxs2keep]
    return df_info

def load_matrices(df_info, data_dir, scale, metric):
    k = 0
    for s,sub in enumerate(df_info['sub']):
        if 'L2008' in data_dir:
            mat_path = os.path.join(data_dir, sub, 'ses-SPUM','dwi','%s_ses-SPUM_atlas-L2008_res-scale%d_conndata-network_connectivity.mat'%(sub,scale)) 
            if not os.path.exists(mat_path):
                mat_path = os.path.join(data_dir, sub, 'dwi', '%s_atlas-L2008_res-scale%d_conndata-network_connectivity.mat'%(sub,scale))   
        else:
            mat_path = os.path.join(data_dir, sub, 'dwi','%s_ses-SPUM_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub,scale)) 
            if not os.path.exists(mat_path):
                mat_path = os.path.join(data_dir, sub, 'dwi', '%s_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub,scale))                 
        mat = sio.loadmat(mat_path)
        matMetric = mat['sc'][metric][0][0]
        Euc = mat['nodes']['dn_position'][0][0]
        nROIs = len(matMetric)
        if k==0:
            MatMat = np.zeros((len(matMetric), len(matMetric),len(df_info['sub'])))
            EucMat = np.zeros((len(matMetric), len(matMetric) ,len(df_info['sub'])))
            MatMat[:,:,k] = matMetric; EucMat[:,:,k] = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Euc, metric='euclidean'))
            k=k+1
        else:
            MatMat[:,:,k] = matMetric; EucMat[:,:,k] = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Euc, metric='euclidean'))
            k=k+1
    return MatMat, EucMat



def filtered_dataframe(dataframe_orig, filters, config):
    ''' Filter the subjects according the defaults parameters set in the configuration file for the existing filters'''
    dataframe = dataframe_orig
    for f,filter in enumerate(filters):
        val_filter = config["Parameters"][filter]
        if 'age' in filter:
            dataframe = dataframe.iloc[np.where((dataframe[filter]>val_filter[0])*(dataframe[filter]<val_filter[1]))[0]]
        elif isinstance(val_filter[0], str):
            dataframe = dataframe[dataframe[filter].isin(val_filter)]
    ls_subs = dataframe['sub']
    return dataframe, ls_subs
            