

## Check dimensiosn SDI_surr

import os
import numpy as np
import scipy.io as sio

SDI_surr_path = r'C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\DATA\\data_isopubli\\data\\results\\data_GSP2_surr.mat'

mat = sio.loadmat(SDI_surr_path)
mat = mat['data_GSP2_surr'][0]

for s in np.arange(len(mat)):
    sub = mat[s][0][0]
    lat = mat[s][1][0]
    SDI_surr_sub = mat[s][2][0][0][0]


