
'''TO BE RUN AFTER main_comparison_SDI_4consensus.py'''
''' This script compare the results before and after rotation for the different consensus from the harmonics to the SDI results'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import scipy
from lib.func_plot import plot_rois, plot_rois_pyvista
import lib.func_reading as reading
from main import load_data
import lib.func_ML as ML
import lib.func_reading as reading
from main import load_data
import lib.func_ML as ML
from sklearn.metrics.pairwise import cosine_similarity

### Consensus 0: DSI HC / Consensus 1: multishell HC / Consensus 2: DSI EP / Consensus 3: multishell EP 

### Initialize 2 matrices to concatenate the results of before and after rotation
### (SDI results) for the 17 subjects (EEG signal) using the 4 different consensus (structural connectomes)
### (DSI patients, DSI controls, multishell patients, multishell controls).
bRot = np.zeros((114,17,4))
aRot = np.zeros((114,17,4))

### Load the list of lateratization of the subjects
ls_lat = np.load(os.path.join(os.getcwd(),'output', 'epilepsy', 'ls_lat.npy'))

### Load the SDI results for the 4 consensus and stored them in the initialized matrices
for k in np.arange(4):
    bRot[:,:,k] = np.load(os.path.join(os.getcwd(),'output', 'epilepsy', 'SDI_%d_beforeRot.npy'%k))
    aRot[:,:,k] = np.load(os.path.join(os.getcwd(),'output', 'epilepsy', 'SDI_%d_afterRot.npy'%k))

### Load the structural harmonics of 4 consensus, before and after rotation.
Q_ind_wei = np.load(os.path.join(os.getcwd(),'output', 'epilepsy', 'Qind_%d_beforeRot_newProcu.npy'%4))
Q_ind_wei_rot = np.load(os.path.join(os.getcwd(),'output', 'epilepsy', 'Qind_rot_%d_afterRot_wei_newProcu.npy'%4))
### Load the P values of consensus 4
P_ind_wei = np.load(os.path.join(os.getcwd(),'output', 'epilepsy', 'Pind_%d_beforeRot_wei_newProcu.npy'%4))

## LOAD THE ORIGINAL CONNECTOME
### Load the configuration file 
config_path = 'config.json' #'config_PEP3.json'
config_defaults = reading.check_config_file(config_path)
### Load the original consensus used in (Rigoni,2023) publication
MatMat, EucMat, dict_df, compatibility, nbProcs = load_data(config_defaults)
MatMat = np.squeeze(MatMat['orig']); EucMat = np.squeeze(EucMat['orig']); G = MatMat
### Estimate the harmonics of the original consensus connectome
P_orig, Q_orig, Ln_ind, An_ind = ML.cons_normalized_lap(G, EucMat,  plot=False)

## CREATE RANDOM MATRIX
### Check the effects on a random matrix
G = np.random.rand(114, 114); G = (G + G.T) / 2
EucMat = np.random.rand(114, 114); EucMat = (EucMat + EucMat.T) / 2
P_shu, Q_shu, Ln_ind, An_ind = ML.cons_normalized_lap(G, EucMat,  plot=False)

### RESHAPE AND CONCATENATE TO THE 4 ORIGINAL CONSENSUS
Q_orig = np.reshape(Q_orig, (114,114,1)); P_orig = np.reshape(P_orig, (114,1))
Q_shu = np.reshape(Q_shu, (114,114,1)); P_shu = np.reshape(P_shu, (114,1))
Qind_concat = np.concatenate((Q_ind_wei, Q_orig, Q_shu),axis=2)
Pind_concat = np.concatenate((P_ind_wei, P_orig, P_shu),axis=1)


'''' Compare the results obtained with the 4 consensus to a random matrix and to the original consensus structural connectome '''

### CHOOSE WHICH CONSENSUS TO COMPARE
### 0 = HC DSI / 1 = HC MUL / 2 = EP DSI / 3 = EP MUL / 4 = ORIG CONS / 5 = RAND
idxs = [1,3]

Pind_concat = Pind_concat[:,idxs]; Qind_concat = Qind_concat[:,:,idxs]
Qind_concat = np.double(Qind_concat)
Q_ind_rot, P_ind_rot,  R_all, scale_R = ML.rotation_procrustes(Qind_concat, Pind_concat, plot=True, p='k%d'%k)
### Check the similarity of the diagonale
metric = 'cosine' # 'cosine'
if metric=='cosine':
    cos_sim_rot = np.diag(cosine_similarity(Q_ind_rot[:,:,0],Q_ind_rot[:,:,1]))
    cos_sim = np.diag(cosine_similarity(Qind_concat[:,:,0],Qind_concat[:,:,1]))
elif metric=='correlation':
    cos_sim_rot = np.diag(np.corrcoef(Q_ind_rot[:,:,0],Q_ind_rot[:,:,1]))
    cos_sim = np.diag(np.corrcoef(Qind_concat[:,:,0],Qind_concat[:,:,1]))

    
    
### Plot the similarity of the diagonal
plt.figure()
plt.scatter(Qind_concat[:,1,0], Qind_concat[:,1,1])
plt.title('Similarity between second eigenvectors');
plt.xlabel('Consensus %d'%idxs[0]); plt.ylabel('Consensus %d'%idxs[1]);
plt.grid()


### Check the diagonal of the cosine similarity matrix before and after rotation for consensus 2 and 3
cos_sim_rot = np.diag(cosine_similarity(Q_ind_wei_rot[:,:,0],Q_ind_wei_rot[:,:,1]))
cos_sim = np.diag(cosine_similarity(Q_ind_wei[:,:,0],Q_ind_wei[:,:,1]))
plt.figure()
#plt.scatter(cos_sim, cos_sim_rot)
plt.plot(np.abs(cos_sim)); plt.plot(cos_sim_rot)
plt.legend(['Before alignment', 'After alignment']); plt.title('Pairwise Cosine Similarity')
plt.grid('on'); plt.xlabel('Eigenvector'); plt.ylabel('Cosine Similarity \n Consensus %d and %d'%(idxs[0], idxs[1]))

plt.figure()
plt.subplot(2,2,1)
plt.scatter(np.mean(bRot[:,:,0], axis=1), np.mean(bRot[:,:,1], axis=1))
r,p = scipy.stats.pearsonr(np.mean(bRot[:,:,0], axis=1), np.mean(bRot[:,:,1], axis=1))
plt.scatter(np.mean(aRot[:,:,0], axis=1), np.mean(aRot[:,:,1], axis=1))
r2,p2 = scipy.stats.pearsonr(np.mean(aRot[:,:,0], axis=1), np.mean(aRot[:,:,1], axis=1))
plt.title('HC  ( r=%.2g, p=%.2g , alignement r=%.2g, p=%.2g) '%(r, p, r2, p2)); plt.xlabel('DSI'); plt.ylabel('multishell');
plt.ylim([0,3]); plt.xlim([0,3]); plt.legend(['no alignement', 'with alignment'])
plt.subplot(2,2,2)
plt.scatter(np.mean(bRot[:,:,2], axis=1), np.mean(bRot[:,:,3], axis=1))
r,p = scipy.stats.pearsonr(np.mean(bRot[:,:,2], axis=1), np.mean(bRot[:,:,3], axis=1))
plt.scatter(np.mean(aRot[:,:,2], axis=1), np.mean(aRot[:,:,3], axis=1))
r2,p2 = scipy.stats.pearsonr(np.mean(aRot[:,:,2], axis=1), np.mean(aRot[:,:,3], axis=1))
plt.title('EP  ( r=%.2g, p=%.2g , alignement r=%.2g, p=%.2g) '%(r, p, r2, p2)); plt.xlabel('DSI'); plt.ylabel('multishell');
plt.ylim([0,3]); plt.xlim([0,3]); plt.legend(['no alignement', 'with alignment'])
plt.subplot(2,2,3)
plt.scatter(np.mean(bRot[:,:,0], axis=1), np.mean(bRot[:,:,2], axis=1))
r,p = scipy.stats.pearsonr(np.mean(bRot[:,:,0], axis=1), np.mean(bRot[:,:,2], axis=1))
plt.scatter(np.mean(aRot[:,:,0], axis=1), np.mean(aRot[:,:,2], axis=1))
r2,p2 = scipy.stats.pearsonr(np.mean(aRot[:,:,0], axis=1), np.mean(aRot[:,:,2], axis=1))
plt.title('DSI  ( r=%.2g, p=%.2g , alignement r=%.2g, p=%.2g) '%(r, p, r2, p2)); plt.xlabel('HC'); plt.ylabel('EP');
plt.ylim([0,3]); plt.xlim([0,3]); plt.legend(['no alignement', 'with alignment'])
plt.subplot(2,2,4)
plt.scatter(np.mean(bRot[:,:,1], axis=1), np.mean(bRot[:,:,3], axis=1))
r,p = scipy.stats.pearsonr(np.mean(bRot[:,:,1], axis=1), np.mean(bRot[:,:,3], axis=1))
plt.scatter(np.mean(aRot[:,:,1], axis=1), np.mean(aRot[:,:,3], axis=1))
r2,p2 = scipy.stats.pearsonr(np.mean(aRot[:,:,1], axis=1), np.mean(aRot[:,:,3], axis=1))
plt.title('multishell  ( r=%.2g, p=%.2g , alignement r=%.2g, p=%.2g) '%(r, p, r2, p2)); plt.xlabel('HC'); plt.ylabel('EP');
plt.ylim([0,3]); plt.xlim([0,3]); plt.legend(['no alignement', 'with alignment'])
plt.suptitle('Correlation between mean SDI')





#plt.figure()
#for k in np.arange(4):
#    plt.subplot(4,1,k+1)
#    for i in np.arange(np.shape(bRot)[1]):
#        plt.scatter(bRot[:,i,k], aRot[:,i,k])
#        plt.title('SDI before/after rotation k=%d'%k)
    
#sub = 1
#plt.figure()
#plt.scatter(aRot[:,sub,2], aRot[:,sub,3])
#plt.title('SDI After rotation k=2 vs k=3 subd %d'%(sub))
#print(1 - spatial.distance.cosine(aRot[:,0], aRot[:,1]))

#plt.figure()
#eig=30
#plt.scatter(Q_ind_rot[:,eig,1], Q_ind_rot[:,eig,2])

#plt.figure()
#for eig in np.arange(np.shape(Q_ind)[1]):
#    plt.scatter(Q_ind_rot[:,eig,2], Q_ind_rot[:,eig,3])
#    #print(1 - spatial.distance.cosine(Q_ind[:,eig,i], Q_ind_rot[:,eig,i]))
#plt.title('Each eigenvectors k=2 and k=3 after rotation')    
    
#plt.figure()
#eig=30
##for i in np.arange(np.shape(Q_ind)[2]):
#plt.scatter(Q_ind[:,eig,0], Q_ind_rot[:,eig,2])
#print(1 - spatial.distance.cosine(Q_ind[:,eig,0], Q_ind_rot[:,eig,2]))    
    

    
plt.show()