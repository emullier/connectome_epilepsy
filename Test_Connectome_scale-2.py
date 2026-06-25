

import numpy as np
import scipy.io as sio

SC_path = 'data/Connectome_scale-2.mat'

SC = sio.loadmat(SC_path)['num']

CTRL = np.load('DATA/SC/matMetric_HC_dsi_number_of_fibers.npy')

print(CTRL.shape)


