

import os
import numpy as np   
import scipy.io as sio 

HC_multi = np.load(r"./data/matMetric_HC_multishell.npy", allow_pickle=True)
HC_DSI = np.load(r"./data/matMetric_HC_DSI.npy", allow_pickle=True)
EP_multi = np.load(r"./data/matMetric_EP_multishell.npy", allow_pickle=True)
#EP_DSI = np.load(r"./data/matMetric_EP_DSI.npy", allow_pickle=True)
#EP_DSI = sio.loadmat(r"./data/matMetric_EP_DSI.mat")


SC_NewSubs = np.concatenate((HC_multi, HC_DSI, EP_multi), axis=2)
print(np.shape(SC_NewSubs))
