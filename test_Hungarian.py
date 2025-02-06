
from lib.func_harmonics import match_eigs
import scipy.io as sio 
import numpy as np

mat = sio.loadmat('./data/randomMat12x12x8.mat')['mat']

cons = np.mean(mat,2)

m_matched = np.zeros(np.shape(mat))
matched_order_all = np.zeros((np.shape(mat)[0], np.shape(mat)[2]))
for m in np.arange(np.shape(mat)[2]):
    m_matched[:,:,m], matched_order_all[:,m] = match_eigs(mat[:,:,m], cons)
    
print(matched_order_all)