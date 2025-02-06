

import os
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
from lib.func_harmonics import ev_zeroXings, zerocrossrate
import lib.func_reading as reading
import lib.func_utils as utils


matMetric = sio.loadmat('data/matMetric_EP_DSI.npy')
print(np.shape(matMetric))
sio.savemat('data/matMetric_EP_DSI.mat', {"matMetric":matMetric})
#np.save('data/matMetric_EP_DSI.npy', matMetric)




scale = 200
filename = './data/MICA_schaefer200_SCFC_struct.mat'
SC = sio.loadmat(filename)['MICA']
n_nodes = np.shape(SC[0][1][2])[0]
n_subjects = np.shape(SC)[1]
matMetric = np.zeros((n_nodes, n_nodes, n_subjects))
for k in np.arange(n_subjects):
    matMetric[:,:,k] = SC[0][k][2]
np.save('./data/matMetric_Sipes.npy', matMetric)

config_path = 'config.json'; config = reading.check_config_file(config_path)
df_info = reading.read_info(config['Parameters']['info_path'])
filters = utils.compare_pdkeys_list(df_info, config['Parameters']['filters'])
df, ls_subs = reading.filtered_dataframe(df_info, filters, config)
procs = list(config["Parameters"]["processing"].keys())
MatMat = {}; EucMat = {};
for p, proc in enumerate(procs):
    idxs_tmp = np.where((df[proc] == 1) | (df[proc] == '1'))[0]
    df_tmp = df.iloc[idxs_tmp]
    tmp_path = os.path.join(config["Parameters"]["data_dir"], config["Parameters"]["processing"][proc])
    MatMat[proc], EucMat[proc], df_info = reading.load_matrices(df_tmp, tmp_path, config['Parameters']['scale'], config['Parameters']['metric'])

matMetric = MatMat[proc]
np.save('data/matMetric_HC_multishell.npy', matMetric)


### Reading the data Ale
SC = sio.loadmat('./data/Individual_Connectomes.mat')   
SC = SC['connMatrices']['SC'][0][0][1][0]
roi_info_path = 'data/label/roi_info.xlsx'
roi_info = pd.read_excel(roi_info_path, sheet_name=f'SCALE 2')
cort_rois = np.where(roi_info['Structure'] == 'cort')[0]
matMetric = SC
x = np.asarray(roi_info['x-pos'])[cort_rois]
y = np.asarray(roi_info['y-pos'])[cort_rois]
z = np.asarray(roi_info['z-pos'])[cort_rois]
coordMat = np.concatenate((x[:,None],y[:,None],z[:,None]),1)
Euc = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordMat, metric='euclidean'))  

np.save('data/matMetric_Ale.npy', matMetric)
