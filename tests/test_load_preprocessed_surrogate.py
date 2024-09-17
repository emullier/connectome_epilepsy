import scipy.io as sio
import numpy as np
import h5py

# Open the .mat file
with h5py.File('/home/localadmin/Documents/CODES/data/PHI.mat', 'r') as f:
    #print(list(f.keys()))
    PHI = f['PHI'][()]
    
#print((PHI))


GSP2_surr = sio.loadmat('/home/localadmin/Documents/CODES/data/results/data_GSP2_surr.mat')
#print(GSP2_surr)
GSP2_surr = GSP2_surr['data_GSP2_surr'][0]

for s in np.arange(np.shape(GSP2_surr)[0]):
    sub = GSP2_surr[s][0]
    lat = GSP2_surr[s][1]
    surr = GSP2_surr[s][2][0][0][0]
    if s==0:
        surr1 = surr
    #print(surr)
    #print(np.shape(surr[0][0][0]))
    #print(np.shape(surr[0][0]))


idxs_ctxs = np.concatenate((np.arange(0,58), np.arange(60,115)))
print(idxs_ctxs)
print(len(np.arange(0,57)))
print(len(np.arange(60,117)))
print(len(idxs_ctxs))
