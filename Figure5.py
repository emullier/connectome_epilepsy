# ================================================
# FULL FIG5 PIPELINE — CACHE-AWARE VERSION 
# ================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import scipy.linalg
from scipy.stats import linregress
from tqdm import tqdm
import lib.func_GSP as gsp
from scipy.interpolate import make_interp_spline

# ==================================================
# PATHS
# ==================================================
project_root = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(project_root, "DATA")
FIG_DIR = os.path.join(project_root, "FIGURES")
OUTPUT_DIR = os.path.join(project_root, "OUTPUT")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EEG_DIR = os.path.join(DATA_DIR, "EEG")

# =====================================================
# GLOBAL CACHE HELPERS
# =====================================================
def load_or_compute(path, compute_fn, compress=False):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    result = compute_fn()
    if compress:
        np.savez_compressed(path, data=result)
    else:
        np.save(path, result)
    return result

def load_npz_dict(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k].tolist() for k in data}

# ============================================================================
# HELPERS
# ============================================================================
def align_all(Q_ref, Q):
    Q_rot, _, _ = procrustes(Q_ref, Q)
    U, _, Vt = scipy.linalg.svd(Q_rot, full_matrices=False)
    Q_rot = U @ Vt

    perm, _ = gsp.match_eigenvectors(Q_ref, Q)
    Q_match = Q[:, perm]

    return {"raw": Q, "rotated": Q_rot, "matched": Q_match}

def harmonic_similarity(Q_ref, Q):
    return np.abs(np.diag(Q_ref.T @ Q))

# ============================================================================
# CACHE WRAPPERS
# ============================================================================
def get_permutations(label, idxs, max_bin, nbPerm):
    path = os.path.join(OUTPUT_DIR, f"perms_{label}.npy")

    def compute():
        return np.array([
            np.random.choice(idxs, max_bin, replace=False)
            for _ in range(nbPerm)
        ])
    return load_or_compute(path, compute)

def get_Q(label, bi, p, SC, perm_idxs, Euc):
    path = os.path.join(OUTPUT_DIR, f"Q_{label}_bin{bi}_perm{p}.npy")

    def compute():
        SC_sub = np.mean(SC[:, :, perm_idxs], axis=2)
        _, Q, _, _ = gsp.cons_normalized_lap(SC_sub, Euc, plot=False)
        return Q

    return load_or_compute(path, compute)

def get_alignments(label, bi, p, Q_ref, Q):
    path = os.path.join(OUTPUT_DIR, f"ALIGN_{label}_bin{bi}_perm{p}.npz")

    if os.path.exists(path):
        data = np.load(path)
        return {"raw": data["raw"], "rotated": data["rotated"],"matched": data["matched"]}

    Qs = align_all(Q_ref, Q)
    np.savez(path,raw=Qs["raw"],rotated=Qs["rotated"],matched=Qs["matched"])

    return Qs

def get_SDI(label, bi, p, method_key, Qm, X_RS_allPat):
    fname = f"SDI_{label}_bin{bi}_perm{p}_{method_key}.npy"
    path = os.path.join(OUTPUT_DIR, fname)

    def compute():
        return np.column_stack([gsp.compute_SDI(pat['X_RS'], Qm)[0] for pat in X_RS_allPat])

    return load_or_compute(path, compute)

# ============================================================================
# DATA
# ============================================================================
print("\nLoading data...")

Euc = np.load(os.path.join(DATA_DIR, "EucMat", "EucMat_HC_dsi_number_of_fibers.npy"))
X_RS_allPat = gsp.load_EEG_example(EEG_DIR)

roi_labels = np.array(np.loadtxt(os.path.join(DATA_DIR, "label", "labels_rois_118.csv"), delimiter=",", dtype=str, skiprows=1, usecols=0))

sc_configs = [
    ('SC-TLE', os.path.join(DATA_DIR, "SC", "matMetric_HC_dsi_number_of_fibers.npy")),
    ('SC-HC',  os.path.join(DATA_DIR, "SC", "matMetric_EP_dsi_number_of_fibers.npy")),
    ('SC-IND', os.path.join(DATA_DIR, "SC", "matMetric_SCHZ_CTRL.npy")),]

# ============================================================================
# PARAMETERS
# ============================================================================
nbPerm = 100
nbSurr = 100
ls_bins = [2,4,6,8,10,12]

# ============================================================================
# MAIN COMPUTATION (CACHED)
# ============================================================================
print("\nRunning analysis...")

results = {}
sdi_results = {}

for label, path in tqdm(sc_configs, desc="SC datasets"):

    res_path = os.path.join(OUTPUT_DIR, f"results_{label}.npz")

    if os.path.exists(res_path):
        print(f"Loading cached results: {label}")

        data = np.load(res_path, allow_pickle=True)

        results[label] = {"kw": data["kw"].tolist(),"mean_sim": data["mean_sim"].tolist(),"std_sim": data["std_sim"].tolist(),"bins": data["bins"].tolist() }

        # restore SDI correlations
        if "sdi_corr" in data:
            sdi_results[label] = data["sdi_corr"].tolist()

        continue

    SC = np.load(path)

    if SC.ndim == 3 and SC.shape[0] != SC.shape[1]:
        SC = np.transpose(SC, (1, 2, 0))

    n_subj = SC.shape[2]
    idxs = np.arange(n_subj)
    bins = [b for b in ls_bins if b <= n_subj]
    max_bin = max(bins)

    SC_ref = np.mean(SC, axis=2)
    _, Q_ref, _, _ = gsp.cons_normalized_lap(SC_ref, Euc, plot=False)

    perms = get_permutations(label, idxs, max_bin, nbPerm)

    kw = {m: [] for m in ["raw", "rotated", "matched"]}
    mean_sim = {m: [] for m in kw}
    std_sim = {m: [] for m in kw}
    sdi_corr = {m: {} for m in ["Before alignment", "Procrustes", "Hungarian"]}

    for bi in tqdm(bins, desc=f"{label} bins", leave=False):

        sim_tmp = {m: [] for m in kw}
        sdi_vectors = {m: [] for m in sdi_corr}

        for p in tqdm(range(nbPerm), desc=f"{label} | n={bi}", leave=False):

            perm_idxs = perms[p][:bi]

            Q = get_Q(label, bi, p, SC, perm_idxs, Euc)
            Qs = get_alignments(label, bi, p, Q_ref, Q)

            for method_key, key in zip(["Before alignment", "Procrustes", "Hungarian"],["raw", "rotated", "matched"]):
                SDI = get_SDI(label, bi, p, method_key, Qs[key], X_RS_allPat)
                sdi_vectors[method_key].append(np.mean(SDI, axis=1))

            for m, Qm in Qs.items():
                sim_tmp[m].extend(harmonic_similarity(Q_ref, Qm))

        for m in kw:
            vals = np.array(sim_tmp[m])
            kw[m].append(vals)
            mean_sim[m].append(vals.mean())
            std_sim[m].append(vals.std())

        # SDI correlations
        for m in sdi_vectors:
            vecs = sdi_vectors[m]
            if len(vecs) < 2:
                continue

            corr = np.corrcoef(vecs)
            vals = corr[np.triu_indices(len(vecs), k=1)]
            sdi_corr[m][bi] = vals

    results[label] = {"kw": kw, "mean_sim": mean_sim, "std_sim": std_sim,"bins": bins}
    sdi_results[label] = sdi_corr

    # SAVE FINAL RESULTS
    np.savez(res_path,kw=kw,mean_sim=mean_sim,std_sim=std_sim,bins=bins,sdi_corr=sdi_corr)

print("\n Computation finished (with caching)")

    
# ============================================================================
# FIG5A — HARMONIC SIMILARITY
# ============================================================================

methods = ["raw", "rotated", "matched"]
colors = {"raw": "#1f77b4",  "rotated": "#2ca02c", "matched": "#d62728"}    
labels_plot = {"raw": "Before alignment", "rotated": "Procrustes", "matched": "Hungarian"}

fig, ax = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

for i, (label, res) in enumerate(results.items()):

    bins = res["bins"]

    for m in methods:

        x_all, y_all = [], []

        # ---- SCATTER CLOUD ----
        for b_idx, b in enumerate(bins):

            vals = res["kw"][m][b_idx]
            jitter = np.random.normal(0, 0.12, len(vals))
            ax[i].scatter(b + jitter,vals,s=8,alpha=0.08, color=colors[m])
            x_all.extend([b]*len(vals))
            y_all.extend(vals)

        x_all = np.array(x_all)
        y_all = np.array(y_all)

        # ---- MEAN ± STD ----
        means = [np.mean(res["kw"][m][k]) for k in range(len(bins))]
        stds  = [np.std(res["kw"][m][k]) for k in range(len(bins))]
        ax[i].errorbar(bins,means,yerr=stds,fmt='o',color=colors[m],markeredgecolor='black',markersize=7,linewidth=2,capsize=4,zorder=5)

        # ---- REGRESSION ----
        if len(x_all) > 0:
            slope, intercept, r, p, _ = linregress(x_all, y_all)
            x_line = np.linspace(min(bins)-1, max(bins)+1, 100)
            ax[i].plot(x_line,slope*x_line + intercept,linestyle='--',color=colors[m],linewidth=2,alpha=0.8,label=f"{labels_plot[m]} (ρ={r:.3f})")

    # ---- PROCRUSTES REFERENCE LINE ----
    ax[i].axhline(1.0, linestyle='--', color="#2ca02c", alpha=0.7)
    ax[i].set_title(label)
    ax[i].set_xlabel("Consensus size (n)")
    ax[i].grid(alpha=0.3)

    if i == 0:
        ax[i].set_ylabel("Per-harmonic similarity")

    ax[i].legend(fontsize=8)

plt.suptitle("Fig5a. Consensus size effect on harmonic similarity", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig5a_final.png"), dpi=600)
plt.close()

# ============================================================================
# FIG5B — SDI STABILITY
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import make_interp_spline

fig, ax = plt.subplots(figsize=(8, 6))

label = "SC-IND"

if label not in sdi_results:
    raise RuntimeError(
        f"{label} not found in sdi_results. "
        f"Available keys: {list(sdi_results.keys())}")

sdi_corr_sc = sdi_results[label]

bins = results[label]["bins"]
x_all, y_all = [], []

# ---- SCATTER (all points) ----
for b in bins:

    if b not in sdi_corr_sc["Before alignment"]:
        continue

    vals = sdi_corr_sc["Before alignment"][b]
    jitter = np.random.normal(0, 0.1, len(vals))

    ax.scatter(
        b + jitter,
        vals,
        alpha=0.12,
        s=8,
        color="#7f7ab8"
    )

    x_all.extend([b] * len(vals))
    y_all.extend(vals)

# ---- MEAN + STD ----
means = [
    np.mean(sdi_corr_sc["Before alignment"][b])
    if b in sdi_corr_sc["Before alignment"] else np.nan
    for b in bins
]

stds = [
    np.std(sdi_corr_sc["Before alignment"][b])
    if b in sdi_corr_sc["Before alignment"] else np.nan
    for b in bins
]

ax.errorbar(
    bins,
    means,
    yerr=stds,
    fmt='o',
    color="#5e5ac5",
    markeredgecolor="black",
    markersize=7,
    linewidth=2,
    capsize=4,
    zorder=5,
    label="Before alignment"
)

# ---- SMOOTH CURVE ----
valid = ~np.isnan(means)
x_valid = np.array(bins)[valid]
y_valid = np.array(means)[valid]

if len(x_valid) > 3:
    x_smooth = np.linspace(min(x_valid), max(x_valid), 200)
    spline = make_interp_spline(x_valid, y_valid, k=3)
    y_smooth = spline(x_smooth)

    ax.plot(
        x_smooth,
        y_smooth,
        linestyle='--',
        color="#5e5ac5",
        linewidth=2
    )

# ---- PEARSON CORRELATION ----

# ✅ Recommended: correlation on mean values vs bins
r, p = pearsonr(x_valid, y_valid)

# 🔁 Alternative: correlation on ALL raw points
# x_arr = np.array(x_all)
# y_arr = np.array(y_all)
# r, p = pearsonr(x_arr, y_arr)

# ---- ANNOTATION ----
ax.text(
    0.05, 0.95,
    f"r = {r:.2f}\np = {p:.2e}",
    transform=ax.transAxes,
    verticalalignment='top',
    fontsize=11,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.7,
        edgecolor="none"
    )
)

# ---- FINAL FORMATTING ----
# ax.set_title(label)
# ax.set_xlabel("Consensus size")
# ax.set_ylabel("Pairwise SDI correlation")

ax.set_ylim(0.3, 1.02)
ax.grid(alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig5b_final.png"), dpi=600)
plt.close()

# ============================================================================
# FIG5C — ROI CONSISTENCY (SC‑IND, surrogate‑based, LEFT vs RIGHT)
# ============================================================================

print("Generating Fig5c (SC‑IND, LEFT vs RIGHT, no sorting)...")

# ------------------------------------------------------------
# Load SC‑IND
# ------------------------------------------------------------
SC_ind = np.load(os.path.join(DATA_DIR, "SC", "matMetric_SCHZ_CTRL.npy"))

# Ensure shape = (subjects, ROI, ROI)
if SC_ind.ndim == 3:
    if SC_ind.shape[1] != SC_ind.shape[2]:
        SC_ind = np.transpose(SC_ind, (2, 0, 1))
n_subj = SC_ind.shape[0]
print(f"SC‑IND subjects: {n_subj}")

# ------------------------------------------------------------
# Lateralization split
# ------------------------------------------------------------
lat_labels = np.array([str(p['lat'][0]) for p in X_RS_allPat])

LT_idx = np.where(np.char.find(lat_labels, 'L') >= 0)[0]
RT_idx = np.where(np.char.find(lat_labels, 'R') >= 0)[0]

print(f"LEFT EEG: {len(LT_idx)} | RIGHT EEG: {len(RT_idx)}")

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------
sig_maps_LT = []
sig_maps_RT = []
ls_lateralization = ["RT", "LT"]

# ------------------------------------------------------------
# Loop over SC‑IND
# ------------------------------------------------------------
matMetric = np.load(os.path.join(project_root, "DATA/SC/matMetric_SCHZ_CTRL.npy"))
EucDist = np.load(os.path.join(project_root, "DATA/EucMat/EucMat_HC_DSI_number_of_fibers.npy"))


for subj in tqdm(range(n_subj), desc="SC‑IND subjects"):

    consensus = matMetric[subj, :, :]

    for lateralization in ls_lateralization:
    
        # Generate harmonics
        P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist, plot=False)
    
        # Project functional signals
        X_RS_allPat = gsp.load_EEG_example(EEG_DIR)
    
        # Estimate SDI
        ls_cutoff = []
        SDI_tmp = np.zeros((118, len(X_RS_allPat)))
        ls_lat = []
        SDI = {}
    
        for p in np.arange(len(X_RS_allPat)):
            X_RS = X_RS_allPat[p]['X_RS']
            ls_lat.append(X_RS_allPat[p]['lat'][0])
            PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, X_RS)
            ls_cutoff.append(NN)
            SDI_tmp[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind)
    
        np.save(os.path.join(OUTPUT_DIR, f'cutoff_IND_{lateralization}_mat{subj}.npy'), ls_cutoff)
        ls_lat = np.array(ls_lat)
        SDI = SDI_tmp
    
        if lateralization == 'RT':
            idxs_lat = np.where(ls_lat=='Rtle')[0]
        elif lateralization == 'LT':
            idxs_lat = np.where(ls_lat=='Ltle')[0]
    
        SDI = SDI[:, idxs_lat]
        np.save(os.path.join(OUTPUT_DIR, f'SDI_IND_{lateralization}_mat{subj}.npy'), SDI)
    
    
        # Surrogate part
        surr_path = os.path.join(OUTPUT_DIR, f'SDI_surr_IND_{lateralization}_mat{subj}_nbSurr{nbSurr}.npy')
        if not os.path.exists(surr_path):
            SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, EEG_DIR, nbSurr=nbSurr, example=False)
            np.save(surr_path, SDI_surr)
        else:
            SDI_surr = np.load(surr_path)
            print('Surrogate SDI already generated')
    
        # Select significant SDI
        surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
        surr_thresh_path = os.path.join(OUTPUT_DIR, f'SDI_surr_thresh_IND_{lateralization}_mat{subj}.npy')
        np.save(surr_thresh_path, surr_thresh, allow_pickle=True)
    
        if lateralization == 'LT':
            sig_maps_LT.append(np.abs(surr_thresh[5]['SDI_sig']))
        elif lateralization == 'RT':
            sig_maps_RT.append(np.abs(surr_thresh[5]['SDI_sig']))
        

# ------------------------------------------------------------
# ROI consistency counts
# ------------------------------------------------------------
sig_maps_LT = np.array(sig_maps_LT)
sig_maps_RT = np.array(sig_maps_RT)

roi_counts_LT = np.sum(sig_maps_LT, axis=0)
roi_counts_RT = np.sum(sig_maps_RT, axis=0)

# ------------------------------------------------------------
# Bar plot 
# ------------------------------------------------------------

plt.rcParams.update({    "font.size": 18,    "axes.titlesize": 20,    "axes.labelsize": 16,    "xtick.labelsize": 1,    "ytick.labelsize": 12,})

#fig, axes = plt.subplots(1, 2, figsize=(25, 10), constrained_layout=True)

# Convert to arrays
roi_counts_LT = np.array(roi_counts_LT)
roi_counts_RT = np.array(roi_counts_RT)

# -------- FILTER zero values (removes empty spaces) --------
threshold = 5
idx_LT = np.where(roi_counts_LT > threshold)[0]
idx_RT = np.where(roi_counts_RT > threshold)[0]

roi_LT = roi_counts_LT[idx_LT]
roi_RT = roi_counts_RT[idx_RT]
labels_LT = [roi_labels[i] for i in idx_LT]
labels_RT = [roi_labels[i] for i in idx_RT]
# -------- SORT values (descending) --------
order_LT = np.argsort(roi_LT)[::-1]
order_RT = np.argsort(roi_RT)[::-1]
roi_LT = roi_LT[order_LT]
roi_RT = roi_RT[order_RT]
labels_LT = [labels_LT[i] for i in order_LT]
labels_RT = [labels_RT[i] for i in order_RT]

max_len = max(len(roi_LT), len(roi_RT))
x_LT = np.arange(len(roi_LT))
x_RT = np.arange(len(roi_RT))

fig, axes = plt.subplots(1, 2,figsize=(25, 10),constrained_layout=True,gridspec_kw={"width_ratios": [len(roi_LT), len(roi_RT)]})

# -------- LEFT --------
axes[0].bar(x_LT, roi_LT, color="#156082", edgecolor='black', linewidth=.7, width=.8)
#axes[0].set_title("Left IED", fontsize=18)
#axes[0].set_ylabel("Number of SC", fontsize=16)
axes[0].grid(axis='y', alpha=0.3)

# -------- RIGHT --------
axes[1].bar(x_RT, roi_RT, color="#196B24", edgecolor='black', linewidth=.7, width=.8)
#axes[1].set_title("Right IED", fontsize=18)
#axes[1].set_ylabel("Number of SC", fontsize=16)
#axes[1].grid(axis='y', alpha=0.3)


# -------- X ticks (no gaps now) --------
axes[0].set_xticks(range(len(labels_LT)))
axes[0].set_xticklabels(labels_LT, fontsize=18, rotation=45, ha='right')
axes[1].set_xticks(range(len(labels_RT)))
axes[1].set_xticklabels(labels_RT,  fontsize=18, rotation=45, ha='right')

max_y = max(max(roi_LT), max(roi_RT))
#axes[0].set_xlim(-0.5, len(x_LT) - 0.5)
#axes[1].set_xlim(-0.5, len(x_RT) - 0.5)
axes[0].set_ylim(0, max_y + 1)
axes[1].set_ylim(0, max_y + 1)


# -------- Title --------
#plt.suptitle("ROI consistency across individual connectomes", fontsize=20, y=1.1)

plt.savefig(os.path.join(FIG_DIR, "Fig5c_ROIs_consistency_indSC.png"), dpi=600, bbox_inches='tight')
plt.close()

print("Figure 5e done, saved as Fig5c_ROIs_consistency_indSC.png")