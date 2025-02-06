

#### Prepare data for degenerative scripts
import numpy as np
import scipy.io as sio

SC = sio.loadmat('./data/Individual_Connectomes.mat')   
SC = SC['connMatrices']['SC'][0][0][1][0]

sio.savemat('data/datasetAle_DegenerativeSipes.mat', {'SC': SC})