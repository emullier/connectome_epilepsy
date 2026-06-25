"""
Export eigenvector matrices (Q_ref, Q_ind) and aligned versions to a .mat file
for use in Fig3_MATLAB_reproduction.m

Q_ref: harmonics from HC consensus (SC HC reference)
Q_ind: harmonics from IND consensus (SC IND - 27 independent controls)
Also exports Qind_rotated, Qind_ortho_rotated, Qind_matched (pre-aligned versions)

Run this script once; it saves DATA/eigendata_for_matlab.mat
"""

import os
import sys
import numpy as np
import scipy
import scipy.io as sio

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
import lib.func_GSP as gsp

print("Loading SC data...")
# --- HC reference consensus ---
consensus_HC_DSI = np.load(os.path.join(project_root, "DATA/SC/matMetric_HC_dsi_number_of_fibers.npy"))
EucDist = np.load(os.path.join(project_root, "DATA/EucMat/EucMat_HC_dsi_number_of_fibers.npy"))
consensus_HC_ref = np.mean(consensus_HC_DSI, axis=2)

# --- IND consensus (first 10 subjects of SCHZ CTRL) ---
consensus_schz = np.load(os.path.join(project_root, "DATA/SC/matMetric_SCHZ_CTRL.npy"))
consensus_schz = consensus_schz[:10, :, :]
consensus_schz_mean = np.mean(consensus_schz, axis=0)

print(f"HC consensus shape: {consensus_HC_ref.shape}")
print(f"IND consensus shape (10 subj avg): {consensus_schz_mean.shape}")

print("Computing graph Laplacian harmonics...")
P_ref, Q_ref, Ln_ref, An_ref = gsp.cons_normalized_lap(consensus_HC_ref, EucDist, plot=False)
P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus_schz_mean, EucDist, plot=False)

print(f"Q_ref shape: {Q_ref.shape}, Q_ind shape: {Q_ind.shape}")

print("Applying alignment methods...")

# Generalized Procrustes
Qind_rotated_raw, Qind_HC_centered, disparity = scipy.spatial.procrustes(Q_ref, Q_ind)
print(f"  Gen. Procrustes disparity: {disparity:.6f}")
# Re-orthonormalize: scipy.spatial.procrustes centers + scales, breaking
# orthonormality needed by compute_SDI. Polar factor restores an orthonormal
# basis with the same column directions -> matches MATLAB procrustes behaviour.
_U, _, _Vt = scipy.linalg.svd(Qind_rotated_raw, full_matrices=False)
Qind_rotated = _U @ _Vt

# Orthogonal Procrustes
R, _ = scipy.linalg.orthogonal_procrustes(Q_ref, Q_ind)
Qind_ortho_rotated = Q_ind @ R

# Hungarian matching
perm, total_cost = gsp.match_eigenvectors(Q_ref, Q_ind)
Qind_matched = Q_ind[:, perm]
print(f"  Hungarian total cost: {total_cost:.6f}")

# Save to .mat file
out_path = os.path.join(project_root, "DATA/eigendata_for_matlab.mat")
sio.savemat(out_path, {
    'Q_ref':             Q_ref,            # HC reference eigenvectors (n_rois x n_harmonics)
    'Q_ind':             Q_ind,            # IND eigenvectors, unaligned
    'Q_ind_rotated':     Qind_rotated,     # Generalized Procrustes aligned
    'Q_ind_ortho':       Qind_ortho_rotated,  # Orthogonal Procrustes aligned
    'Q_ind_matched':     Qind_matched,     # Hungarian matching aligned
    'P_ref':             P_ref,            # HC eigenvalues
    'P_ind':             P_ind,            # IND eigenvalues
    'EucDist':           EucDist,
})
print(f"\nSaved to: {out_path}")
print("Keys: Q_ref, Q_ind, Q_ind_rotated, Q_ind_ortho, Q_ind_matched, P_ref, P_ind, EucDist")

# -------------------------------------------------------------------------
# Export EEG inputs so SDI is computed directly in MATLAB (not loaded)
# -------------------------------------------------------------------------
print("\nExporting EEG inputs for MATLAB SDI computation...")
X_RS_allPat = gsp.load_EEG_example(os.path.join(project_root, "DATA", "EEG"))

X_RS_cell = np.empty((len(X_RS_allPat), 1), dtype=object)
for i, pat in enumerate(X_RS_allPat):
    X_RS_cell[i, 0] = pat['X_RS']

lat_labels = np.array([str(pat['lat'][0]) for pat in X_RS_allPat], dtype=object).reshape(-1, 1)
subject_names = np.array([str(pat['name']) for pat in X_RS_allPat], dtype=object).reshape(-1, 1)
subject_shapes = np.array([pat['X_RS'].shape for pat in X_RS_allPat], dtype=object).reshape(-1, 1)

eeg_out_path = os.path.join(project_root, "DATA/eeg_for_matlab.mat")
sio.savemat(eeg_out_path, {
    'X_RS_cell': X_RS_cell,
    'lat_labels': lat_labels,
    'subject_names': subject_names,
    'subject_shapes': subject_shapes,
})

print(f"EEG data saved to: {eeg_out_path}")
print(f"Exported {len(X_RS_allPat)} subjects as variable-size cell array (X_RS_cell)")
