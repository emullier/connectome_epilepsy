
import numpy as np
import scipy.io as sio
import os

example_dir = r"C:\\Users\\emeli\\Documents\\CHUV\\TEST_RETEST_DSI_microstructure\\SFcoupling_IED_GSP\\data\\data"

GSP2_surr = sio.loadmat(os.path.join(example_dir,'data_GSP2_surr.mat'))
GSP2_surr = GSP2_surr['data_GSP2_surr'][0]

print(GSP2_surr.shape)

#for s in np.arange(np.shape(GSP2_surr)[0]):
#    sub = GSP2_surr[s][0]
#    lat = GSP2_surr[s][1]
#    surr = GSP2_surr[s][2][0][0][0]
#    print(surr.shape)
#    #print(surr[0:10,0])

s = 0
sub = GSP2_surr[s][0]
lat = GSP2_surr[s][1]
surr = GSP2_surr[s][2][0][0][0]
print(GSP2_surr[s][2][0][0].shape)
print(surr[0:10,1])