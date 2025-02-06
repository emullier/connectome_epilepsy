

import os
import numpy as np 
import matplotlib.pyplot as plt

path_main = 'th3_Ltle_main.npy'
path_rigo = 'th3_Ltle_main_rigoni.npy'

path_scmain = 'Gdist_main.npy'
path_scrigo = 'matMetric_mainrigoni.npy'

path_sdimain = 'SDI_main.npy'
path_sdirigo = 'SDI_mainrigoni.npy'

rigo = np.load(path_rigo)
main = np.load(path_main)
scrigo = np.load(path_scrigo)
scmain = np.load(path_scmain)
sdirigo = np.load(path_sdirigo)
sdimain = np.load(path_sdimain)
MatMat = np.load('MatMat_main.npy', allow_pickle=True)
idxs_tmp = np.concatenate((np.arange(0,57), np.arange(59, 116)))
#rigo = rigo[idxs_tmp]

print(rigo[0])
print(main[0])

print(np.shape(rigo))
print(np.shape(main))

print(np.mean(rigo))
print(np.mean(main))

print(np.mean(scrigo))
print(np.mean(scmain))
print(np.mean(MatMat))

fig, axs = plt.subplots(1,3)
axs[0].scatter(rigo, main)
axs[1].scatter(scrigo.flatten(), scmain.flatten())
axs[2].scatter(sdirigo.flatten(), sdimain.flatten())
plt.show()