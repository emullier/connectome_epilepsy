import matplotlib.pyplot as plt
import numpy as np
import lib.func_ML as ML
import scipy.io as sio

mat_path = r'E:\PROJECT_CONNECTOME_EPILEPSY\cmp_harmonization_merged\cmp-3.1.0_shore\sub-01\dwi\sub-01_atlas-L2018_res-scale2_conndata-network_connectivity.mat'

mat = sio.loadmat(mat_path)
print(mat['nodes']['dn_position'][0][0])