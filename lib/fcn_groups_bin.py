

import numpy as np
import matplotlib.pyplot as plt

def fcn_groups_bin(A,dist,hemiid,nbins):
    
    """ fcn_distance_dependent_threshold      generate group matrix

        G = fcn_distance_dependent_threshold(A,dist,hemiid,frac) generates a
        group-representative structural connectivity matrix by preserving
        within-/between-hemisphere connection length distributions.
        
       Inputs:
               A,      [node x node x subject] structural connectivity
                       matrices.
               dist,   [node x node] distance matrix
               hemiid, indicator matrix for left (1) and right (2)
                       hemispheres
               nbins,  number of distance bins

       Outputs:
               G,      group matrix (binary) with distance-based consensus
               Gc,     group matrix (binary) with traditional
                       consistency-based thresholding.

   From matlab code from Richard Betzel, Indiana University, 2018 
   Adapted for Python by Emeline Mullier, Lausanne University Hospital, 2023"""
    
    distbins = np.linspace(np.min(np.nonzero(dist)),np.max(np.nonzero(dist)), nbins+1)
    distbins[-1] = distbins[-1] + 1    

    
    n, _, nsub = A.shape  # number of nodes (n) and subjects (nsub)
    C = np.sum(A > 0, axis=2)  # consistency   ### C = np.sum(np.where(A>.3))  ?
    W = np.sum(A, axis=2) / C  # average weight
    W[np.isnan(W)] = 0  # remove NaNs
    
    Grp = np.zeros((n, n, 2))  # for storing inter/intra hemispheric connections (we do these separately)
    Gc = Grp.copy()

    for j in range(2):
        if j == 0:  # make inter- or intra-hemispheric edge mask
            d = (hemiid == 1)[:, np.newaxis] * (hemiid.T == 2)
            d = d | d.T
        else:
            d = (hemiid == 1)[:, np.newaxis] * (hemiid.T == 1) | (hemiid == 2)[:, np.newaxis] * (hemiid.T == 2)
            d = d | d.T

        m = dist * d
        D = np.zeros(np.shape(A))
        for x in np.arange(np.shape(A)[2]):
            D[:,:,x] = ((A[:,:,x] > 0) * dist * np.triu(d))
        #D = np.nonzero(D)
        D = D[np.where(D>0)]
        #D = np.nonzero((A > 0) * (dist[:,:,np.newaxis] * np.triu(d)))
        #tgt = len(D[0]) / nsub
        tgt = len(D) / nsub
        G = np.zeros((n, n))

        for ibin in range(nbins):
            mask = np.where(np.triu(m >= distbins[ibin]) & (m < distbins[ibin + 1]))
            #frac = round(tgt * len(np.where((D >= distbins[ibin]) & (D < distbins[ibin + 1]))[0]) / len(D[0]))
            frac = round(tgt * len(np.where((D >= distbins[ibin]) & (D < distbins[ibin + 1]))[0]) / len(D))
            c = C[mask]
            idx = np.argsort(c)[::-1]
            G[mask[0][idx[:frac]], mask[1][idx[:frac]]] = 1
            
        Grp[:, :, j] = G
        I = np.where(np.triu(d, 1))
        w = W[I]
        #idx = np.argsort(w)[::-1]
        idx = np.argsort(-w)
        w = np.zeros((n, n))
        sumG = np.count_nonzero(G)
        idx_x = I[0][idx[0:int(np.sum(G))]]
        idx_y = I[1][idx[0:int(np.sum(G))]]
        w[idx_x, idx_y] = 1
        Gc[:, :, j] = w
    

    G = np.sum(Grp, axis=2)
    G = G + G.T
    G[G > 0] = 1 
    
    initial_weights = np.mean(A, axis=2)  # shape (118, 118)
    final_weights  = initial_weights
    #rows, cols = np.where(G == 1)
    #pooled_weights = initial_weights[rows, cols]
    #pooled_weights_flat = pooled_weights.flatten()
    #pooled_weights_flat_sorted = np.sort(pooled_weights_flat)
    #min_w, max_w = pooled_weights_flat_sorted.min(), pooled_weights_flat_sorted.max()
    #normalized_initial = (pooled_weights_flat_sorted - min_w) / (max_w - min_w)
    #reassigned_weights = np.interp(normalized_initial,np.linspace(0, 1, len(pooled_weights_flat_sorted)), pooled_weights_flat_sorted)
    #final_weights = np.zeros_like(initial_weights)
    #final_weights[rows, cols] = reassigned_weights
    
    #print('Number of nonzero final weights:', np.count_nonzero(final_weights))  # Should be 2680
    #print('Expected retained connections:', np.sum(G))  # Should be 2680
    G = G* final_weights   
    
    Gc = np.sum(Gc, axis=2)
    Gc = Gc + Gc.T
    Gc[Gc > 0] = 1 
    Gc = Gc * final_weights
    
    return G, Gc

    
    
    
def plot_dist_distribution(ax, A, dist, nbins, G, Gc):
    
    N, _, NSub = A.shape
    distbins = np.linspace(np.min(np.nonzero(dist)), np.max(np.nonzero(dist)), nbins + 1)

    # Calculate histograms
    h = np.zeros((NSub + 2, nbins))
    for iSub in range(NSub):
        h[iSub, :] = np.histogram(dist[np.triu(A[:, :, iSub] > 0)], bins=distbins)[0]
        h[iSub, :] = h[iSub, :] / np.sum(h[iSub, :])

    h[NSub, :] = np.histogram(dist[np.triu(Gc > 0)], bins=distbins)[0]
    h[NSub, :] = h[NSub, :] / np.sum(h[NSub, :])

    h[NSub + 1, :] = np.histogram(dist[np.triu(G > 0)], bins=distbins)[0]
    h[NSub + 1, :] = h[NSub + 1, :] / np.sum(h[NSub + 1, :])


    # Make figure
    ax.plot(distbins[:-1], h.T, color=np.ones(3) * 0.75) 
    ax.plot(distbins[:-1], h[NSub, :], color='b', linewidth=2, label='uniform') 
    ax.plot(distbins[:-1], h[NSub + 1, :], color='r', linewidth=2, label='distance')  
    # Add labels and change some axis properties
    ax.set(xlabel='distance (mm)', ylabel='probability', xlim=[np.min(distbins) - 0.1 * np.ptp(distbins), np.max(distbins) + 0.1 * np.ptp(distbins)])
    ax.legend()
    #plt.show()