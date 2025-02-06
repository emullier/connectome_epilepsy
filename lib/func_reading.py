
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
        if 'struct_data' in sub:
           mat_path =  ".\\data\\struct_data.mat"
        elif 'L2008' in data_dir:
            mat_path = os.path.join(data_dir, sub, 'ses-SPUM','dwi','%s_ses-SPUM_atlas-L2008_res-scale%d_conndata-network_connectivity.mat'%(sub,scale)) 
            if not os.path.exists(mat_path):
                mat_path = os.path.join(data_dir, sub, 'dwi', '%s_atlas-L2008_res-scale%d_conndata-network_connectivity.mat'%(sub,scale))   
        else:
            mat_path = os.path.join(data_dir, sub,  'dwi','%s_ses-SPUM_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub,scale)) 
            if not os.path.exists(mat_path):
                mat_path = os.path.join(data_dir, sub, 'dwi', '%s_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub,scale))  
                if not os.path.exists(mat_path):
                    mat_path = os.path.join(data_dir, sub, 'ses-SPUM', 'dwi','%s_ses-SPUM_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub,scale)) 
        if 'struct_data' in sub:
            SC  = sio.loadmat('./data/Connectome_scale-%d.mat'%scale)['num']
            print(np.shape(SC))
            roi_info_path = 'data/label/roi_info.xlsx'
            roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
            cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
            #print(len(cort_rois))
            #print((cort_rois))           
            #cort_rois = np.concatenate((np.arange(0,57), np.arange(59,116)))
            #print(len(cort_rois))
            #print((cort_rois)) 
            SC = SC[cort_rois,:]; SC = SC[:, cort_rois]
            matMetric = SC
            x = np.asarray(roi_info['x-pos'])[cort_rois]
            y = np.asarray(roi_info['y-pos'])[cort_rois]
            z = np.asarray(roi_info['z-pos'])[cort_rois]
            coordMat = np.concatenate((x[:,None],y[:,None],z[:,None]),1)
            Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))  
        elif 'Individual_Connectomes' in sub:
            SC = sio.loadmat('./data/Individual_Connectomes.mat')
            SC = SC['connMatrices']['SC'][0][0][scale-1][0]
            roi_info_path = 'data/label/roi_info.xlsx'
            roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
            cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
            matMetric = SC
            x = np.asarray(roi_info['x-pos'])[cort_rois]
            y = np.asarray(roi_info['y-pos'])[cort_rois]
            z = np.asarray(roi_info['z-pos'])[cort_rois]
            coordMat = np.concatenate((x[:,None],y[:,None],z[:,None]),1)
            Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))  
        elif 'L2008' in data_dir:
            mat = sio.loadmat(mat_path)
            matMetric = mat['sc'][metric][0][0]
            Euc = mat['nodes']['dn_position'][0][0]         
        else:
            mat = sio.loadmat(mat_path)
            roi_info_path = 'data/label/roi_info_l2018.xlsx'
            roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
            cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
            matMetric = mat['sc'][metric][0][0]
            Euc = mat['nodes']['dn_position'][0][0]
            matMetric = matMetric[cort_rois,:]; matMetric = matMetric[:, cort_rois]
            Euc = Euc[cort_rois,:]; 
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
    return MatMat, EucMat, df_info



def filtered_dataframe(dataframe_orig, filters, config):
    ''' Filter the subjects according the defaults parameters set in the configuration file for the existing filters'''
    dataframe = dataframe_orig
    for f,filter in enumerate(filters):
        val_filter = config["Parameters"][filter]
        print(val_filter)
        if 'age' in filter:
            dataframe = dataframe.iloc[np.where((dataframe[filter]>val_filter[0])*(dataframe[filter]<val_filter[1]))[0]]
        elif isinstance(val_filter[0], str):
            dataframe = dataframe[dataframe[filter].isin(val_filter)]
    ls_subs = dataframe['sub']
    return dataframe, ls_subs
            
def save_consensus(MatMat, metric, G_dist, G_unif, out_dir, processings):
    G_dist_wei = {}; G_unif_wei = {}
    for p,proc in enumerate(processings):
        dist_bin_path = os.path.join(out_dir, 'Consensus_binary_%dsubs_%s_dist'%(np.shape(MatMat[proc])[2], proc))
        unif_bin_path = os.path.join(out_dir, 'Consensus_binary_%dsubs_%s_unif'%(np.shape(MatMat[proc])[2], proc))
        dist_wei_path = os.path.join(out_dir, 'Consensus_W%s_%dsubs_%s_dist'%(metric, np.shape(MatMat[proc])[2], proc))
        unif_wei_path = os.path.join(out_dir, 'Consensus_W%s_%dsubs_%s_unif'%(metric, np.shape(MatMat[proc])[2], proc))
        G_dist_wei[proc] = G_dist[proc]*np.mean(MatMat[proc], axis=2)
        G_unif_wei[proc] = G_unif[proc]*np.mean(MatMat[proc], axis=2)
        np.save('%s.npy'%dist_bin_path, G_dist[proc]); np.save('%s.npy'%dist_wei_path, G_dist_wei[proc])
        np.save('%s.npy'%unif_bin_path, G_unif[proc]); np.save('%s.npy'%unif_wei_path, G_unif_wei[proc])
        sio.savemat('%s.mat'%dist_bin_path, {'SC_cons': G_dist[proc]}); sio.savemat('%s.mat'%dist_wei_path, {'SC_cons':G_dist_wei[proc]})
        sio.savemat('%s.mat'%unif_bin_path, {'SC_cons':G_unif[proc]}); sio.savemat('%s.mat'%unif_wei_path, {'SC_cons':G_unif_wei[proc]})
    return G_dist_wei, G_unif_wei


def reading_consensus(procs, metric, outputdir, dict_df):
    for p, proc in enumerate(procs):
        nSubs = len(dict_df[proc])
        if p==0:
            tmp = np.load(os.path.join(outputdir, 'Consensus_W%s_%dsubs_%s_dist.npy'%(metric, nSubs, proc)))
            ConsConc = np.zeros((np.shape(tmp)[0],np.shape(tmp)[0], len(procs)))
            ConsConc[:,:,p] = tmp
        else:
            ConsConc[:,:,p] = np.load(os.path.join(outputdir, 'Consensus_W%s_%dsubs_%s_dist.npy'%(metric, nSubs, proc)))
    return ConsConc
    
    