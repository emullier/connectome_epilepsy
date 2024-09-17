import matplotlib.pyplot as plt
import numpy as np
import lib.func_ML as ML
import scipy.io as sio
from lib.func_plot import plot_rois, plot_rois_pyvista
import lib.func_reading as reading 

#mat_path = r'E:\PROJECT_CONNECTOME_EPILEPSY\cmp_harmonization_merged\cmp-3.1.0_shore\sub-01\dwi\sub-01_atlas-L2018_res-scale2_conndata-network_connectivity.mat'
#mat = sio.loadmat(mat_path)
#print(mat['nodes']['dn_position'][0][0])

config_path = 'config.json' #'config_PEP3.json'
config_defaults = reading.check_config_file(config_path)

tmp = np.random.uniform(low=-1.0, high=1.0, size=(114,))
#plot_rois(tmp, config_defaults["Parameters"]["scale"], config_defaults, vmin=-1, vmax=1, label='test')
plot_rois_pyvista(tmp, 2, config_defaults, vmin=-1, vmax=1, label='test')
