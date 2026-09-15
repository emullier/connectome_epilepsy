"""
SDI consensus-generation, group-comparison, and consensus-size analysis (MAIN FIGURES)
=========================================================================================

Consolidates the essential steps to generate all main-text figures related to
the Structural Decoupling Index (SDI) computed on three structural-connectivity
(SC) consensus matrices:
  - HC  : Healthy controls (Geneva dataset)
  - EP  : Epilepsy patients (Geneva dataset); 7 left TLE and 9 right TLE
  - IND : Independent healthy-control consensus

Outputs
-------
  - SDI_comparison_table_Manuscript.xlsx          : significant-ROI comparison table
  - FIGURES/Fig3_summary_activation_{LT,RT}.png   : brain plots of consensus activation
  - FIGURES/Fig2_heatmap_SDI_correlation_SC.png   : inter-group SDI correlation heatmap
  - FIGURES/Fig5_SDI_stability.png                : SDI stability vs. consensus size
  - FIGURES/Fig5_ROIs_consistency_indSC.png      : ROI-level consistency (surrogate-based)

Also writes intermediate arrays to OUTPUT/ (cutoff frequencies, nbROIs-significant
counts, SDI surrogate thresholds, cached Fig5 permutation results) that are reused
by Script_Supplementary_Analyses.py — run this script first.

Expects the companion data (DATA/, DEMOGRAPHIC/) to be available locally, as
archived on Zenodo (see manuscript).

Emeline Mullier, University of Geneva & Lausanne University Hospital
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

import lib.func_GSP as gsp
from lib.func_plot import plot_rois_pyvista_noaxes

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Aptos', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 10

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(project_root, "DATA")
EEG_DIR = os.path.join(DATA_DIR, "EEG")
output_dir = os.path.join(project_root, "OUTPUT")
figures_dir = os.path.join(project_root, "FIGURES")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

scale = 2
nbSurr = 100
infoGVA_path = os.path.join(project_root, "DEMOGRAPHIC/info_sc_participants_connectome_epilepsy.csv")

# ============================================================================
# PART 1 - Build HC, EP, IND consensus matrices and compute group-wise SDI
# ============================================================================
print("=" * 80)
print("PART 1: Computing group SDI for HC, EP, IND")
print("=" * 80)

ls_groups_geneva = ["HC", "EP"]
ls_lateralization = ["RT", "LT"]

surr_thresh_storage = {}

for group in ls_groups_geneva:
    df_info = pd.read_csv(infoGVA_path)
    data_path = os.path.join(project_root, f"DATA/SC/matMetric_{group}_dsi_number_of_fibers.npy")
    matMetric_full = np.load(data_path)
    EucDist = np.load(os.path.join(project_root, f"DATA/EucMat/EucMat_{group}_dsi_number_of_fibers.npy"))

    for lateralization in ls_lateralization:
        matMetric = matMetric_full
        if group == "EP":
            idxs = np.where(df_info["Lateralization"] == lateralization)[0]
            matMetric = matMetric_full[:, :, idxs]
        consensus = np.mean(matMetric, axis=2)

        surr_thresh_storage[f"{group}_{lateralization}"] = gsp.compute_group_sdi(label=group, consensus=consensus, EucDist=EucDist, lateralization=lateralization,
            example_dir=EEG_DIR, output_dir=output_dir, figures_dir=figures_dir, nbSurr=nbSurr, surr_key=f"SDI_surr_number_of_fibers_{group}_dsi_{lateralization}",)

matMetric_ind = np.load(os.path.join(project_root, "DATA/SC/matMetric_IND_10CTRL.npy"))
EucDist_ind = np.load(os.path.join(project_root, "DATA/EucMat/EucMat_HC_DSI_number_of_fibers.npy"))

for lateralization in ls_lateralization:
    consensus_ind = np.mean(matMetric_ind, axis=0)
    surr_thresh_storage[f"IND_{lateralization}"] = gsp.compute_group_sdi(label="IND", consensus=consensus_ind, EucDist=EucDist_ind, lateralization=lateralization,
        example_dir=EEG_DIR, output_dir=output_dir, figures_dir=figures_dir, nbSurr=nbSurr,)


# ============================================================================
# PART 2 - Comparison table of significant SDI ROIs (threshold = 5)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: Building SDI comparison table")
print("=" * 80)

df_roi = pd.read_csv(os.path.join(project_root, "DATA/label/labels_rois_118.csv"))
labels_118 = np.array(df_roi["Label Lausanne2008"])

groups_table = {
    "HC_LT": surr_thresh_storage["HC_LT"], "HC_RT": surr_thresh_storage["HC_RT"],
    "EP_LT": surr_thresh_storage["EP_LT"], "EP_RT": surr_thresh_storage["EP_RT"],
    "IND_LT": surr_thresh_storage["IND_LT"], "IND_RT": surr_thresh_storage["IND_RT"],}

all_idx = set()
for surr in groups_table.values():
    all_idx.update(np.where(surr[5]["SDI_sig"] != 0)[0])
all_idx = sorted(all_idx)

data = {("ROI", ""): [labels_118[idx] for idx in all_idx]}
for side in ["LT", "RT"]:
    for group in ["EP", "HC", "IND"]:
        values = []
        for idx in all_idx:
            surr = groups_table[f"{group}_{side}"]
            values.append(round(surr[5]["mean_SDI"][idx], 2) if surr[5]["SDI_sig"][idx] != 0 else np.nan)
        data[(side, group)] = values

df_comparison = pd.DataFrame(data)
df_comparison.columns = pd.MultiIndex.from_tuples(df_comparison.columns)
df_comparison.to_excel("SDI_comparison_table_Manuscript.xlsx", index=True)
print("Table saved to: SDI_comparison_table_Manuscript.xlsx")


# ============================================================================
# PART 3 - Fig3: brain plots of consensus activation across HC/EP/IND
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: Fig3 - summary brain plots")
print("=" * 80)

for lateralization in ["LT", "RT"]:
    summary_vector = np.zeros(118)
    for roi_idx in range(118):
        count = sum(surr_thresh_storage[f"{group}_{lateralization}"][5]["SDI_sig"][roi_idx] != 0 for group in ["HC", "EP", "IND"])
        summary_vector[roi_idx] = count
    summary_vector_masked = np.where(summary_vector > 0, summary_vector, np.nan)
    plot_rois_pyvista_noaxes(summary_vector_masked, scale, figures_dir, vmin=1, vmax=3, cmap="YlOrRd", label=f"Fig3_summary_activation_{lateralization}",)
    print(f"  {lateralization}: 1 SC={np.sum(summary_vector==1)}  2 SC={np.sum(summary_vector==2)}  3 SC={np.sum(summary_vector==3)}")

# ============================================================================
# PART 4 - Fig2: inter-group correlation of mean SDI values
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: Fig2 - SDI correlation heatmap across SC methods")
print("=" * 80)

def get_all_mean(group, side, thr=5):
    return surr_thresh_storage[f"{group}_{side}"][thr]["mean_SDI"]

groups_heatmap = ["HC", "EP", "IND"]
group_labels_heatmap = [r"SC$_{HC}$", r"SC$_{TLE}$", r"SC$_{IND}$"]

matrices_by_side = {}
for side in ["LT", "RT"]:
    sdi_vectors = [get_all_mean(group, side) for group in groups_heatmap]
    n_groups = len(sdi_vectors)
    r_mat, p_mat = np.eye(n_groups), np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(n_groups):
            if i != j:
                r_mat[i, j], p_mat[i, j] = pearsonr(sdi_vectors[i], sdi_vectors[j])
    matrices_by_side[side] = {
        "r_df": pd.DataFrame(r_mat, index=group_labels_heatmap, columns=group_labels_heatmap),
        "p_df": pd.DataFrame(p_mat, index=group_labels_heatmap, columns=group_labels_heatmap),
        "r_mat": r_mat, "p_mat": p_mat,}

purple_cmap = sns.light_palette("purple", as_cmap=True)
fig_hm, axes_hm = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
fig_hm.suptitle("SDI relationships across structural connectomes", fontsize=14, fontweight="bold")

mask_lower = np.triu(np.ones_like(matrices_by_side["LT"]["r_df"], dtype=bool), k=0)
mask_upper = np.tril(np.ones_like(matrices_by_side["LT"]["r_df"], dtype=bool), k=0)

sns.heatmap(matrices_by_side["LT"]["r_df"], mask=mask_lower, ax=axes_hm[0], cmap="Blues", vmin=0, vmax=1,square=True, linewidths=0.5, linecolor="white", cbar=False, annot=False)
sns.heatmap(matrices_by_side["RT"]["r_df"], mask=mask_upper, ax=axes_hm[0], cmap="Greens", vmin=0, vmax=1,square=True, linewidths=0.5, linecolor="white", cbar=False, annot=False)
sns.heatmap(matrices_by_side["LT"]["p_df"], mask=mask_lower, ax=axes_hm[1], cmap=purple_cmap, vmin=0, vmax=1, square=True, linewidths=0.5, linecolor="white", cbar=False, annot=False)
sns.heatmap(matrices_by_side["RT"]["p_df"], mask=mask_upper, ax=axes_hm[1], cmap=purple_cmap, vmin=0, vmax=1, square=True, linewidths=0.5, linecolor="white", cbar=False, annot=False)

for ax in axes_hm:
    for d in range(n_groups):
        ax.add_patch(plt.Rectangle((d, d), 1, 1, facecolor="lightgray", edgecolor="white", linewidth=0.5, zorder=3))
        ax.text(d + 0.5, d + 0.5, "\u2014", ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)

for i in range(n_groups):
    for j in range(n_groups):
        if i > j:
            p_lt = matrices_by_side["LT"]["p_mat"][i, j]
            stars_lt = "***" if p_lt < 0.001 else "**" if p_lt < 0.01 else "*" if p_lt < 0.05 else ""
            axes_hm[0].text(j + 0.5, i + 0.5, f"{matrices_by_side['LT']['r_mat'][i, j]:.2f}{stars_lt}", ha="center", va="center", fontsize=14, fontweight="bold" if p_lt < 0.05 else "normal")
            axes_hm[1].text(j + 0.5, i + 0.5, f"{p_lt:.1e}{stars_lt}", ha="center", va="center", fontsize=14, fontweight="bold" if p_lt < 0.05 else "normal")
        elif i < j:
            p_rt = matrices_by_side["RT"]["p_mat"][i, j]
            stars_rt = "***" if p_rt < 0.001 else "**" if p_rt < 0.01 else "*" if p_rt < 0.05 else ""
            axes_hm[0].text(j + 0.5, i + 0.5, f"{matrices_by_side['RT']['r_mat'][i, j]:.2f}{stars_rt}", ha="center", va="center", fontsize=14, fontweight="bold" if p_rt < 0.05 else "normal")
            axes_hm[1].text(j + 0.5, i + 0.5, f"{p_rt:.1e}{stars_rt}", ha="center", va="center", fontsize=14, fontweight="bold" if p_rt < 0.05 else "normal")

axes_hm[0].set_title("Pearson r: lower Left IED, upper Right IED", fontsize=12, fontweight="bold")
axes_hm[1].set_title("p-value: lower Left IED, upper Right IED", fontsize=12, fontweight="bold")
for ax in axes_hm:
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

sm_corr = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=1)); sm_corr.set_array([])
sm_pval = plt.cm.ScalarMappable(cmap=purple_cmap, norm=plt.Normalize(vmin=0, vmax=1)); sm_pval.set_array([])
cbar_corr = fig_hm.colorbar(sm_corr, ax=axes_hm[0], location="left", fraction=0.06, pad=0.04)
cbar_corr.set_label("Pearson r Left IED", fontsize=10, fontweight="bold")
cbar_pval = fig_hm.colorbar(sm_pval, ax=axes_hm[1], location="right", fraction=0.06, pad=0.04)
cbar_pval.set_label("p-value Left/Right IED", fontsize=10, fontweight="bold")

heatmap_path = os.path.join(figures_dir, "Fig2_heatmap_SDI_correlation_SC.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.basename(heatmap_path)}")


# ============================================================================
# PART 5 - Fig5: consensus-size effects on SDI stability & ROI consistency
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: Fig5 - consensus-size effects")
print("=" * 80)

Euc = np.load(os.path.join(DATA_DIR, "EucMat", "EucMat_HC_dsi_number_of_fibers.npy"))
X_RS_allPat = gsp.load_EEG_example(EEG_DIR)
roi_labels = np.array(np.loadtxt(os.path.join(DATA_DIR, "label", "labels_rois_118.csv"), delimiter=",", dtype=str, skiprows=1, usecols=0))

sc_configs = [
    ("SC-TLE", os.path.join(DATA_DIR, "SC", "matMetric_HC_dsi_number_of_fibers.npy")),
    ("SC-HC", os.path.join(DATA_DIR, "SC", "matMetric_EP_dsi_number_of_fibers.npy")),
        ("SC-IND", os.path.join(DATA_DIR, "SC", "matMetric_IND_CTRL_FULL.npy")),]

nbPerm = 10
nbSurr_fig5 = 10
ls_bins = [2, 4, 6, 8, 10, 12]
results, sdi_results = {}, {}

for label, path in tqdm(sc_configs, desc="SC datasets"):
    res_path = os.path.join(output_dir, f"results_{label}.npz")

    if os.path.exists(res_path):
        print(f"Loading cached results: {label}")
        data = np.load(res_path, allow_pickle=True)
        results[label] = {"kw": data["kw"].tolist(), "mean_sim": data["mean_sim"].tolist(),
                           "std_sim": data["std_sim"].tolist(), "bins": data["bins"].tolist()}
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
    perms = gsp.get_permutations(label, idxs, max_bin, nbPerm, output_dir)

    kw = {m: [] for m in ["raw", "rotated", "matched"]}
    mean_sim = {m: [] for m in kw}
    std_sim = {m: [] for m in kw}
    sdi_corr = {m: {} for m in ["Before alignment", "Procrustes", "Hungarian"]}

    for bi in tqdm(bins, desc=f"{label} bins", leave=False):
        sim_tmp = {m: [] for m in kw}
        sdi_vectors = {m: [] for m in sdi_corr}

        for p in tqdm(range(nbPerm), desc=f"{label} | n={bi}", leave=False):
            perm_idxs = perms[p][:bi]
            Q = gsp.get_Q(label, bi, p, SC, perm_idxs, Euc, output_dir)
            Qs = gsp.get_alignments(label, bi, p, Q_ref, Q, output_dir)

            for method_key, key in zip(["Before alignment", "Procrustes", "Hungarian"], ["raw", "rotated", "matched"]):
                SDI = gsp.get_SDI(label, bi, p, method_key, Qs[key], X_RS_allPat, output_dir)
                sdi_vectors[method_key].append(np.mean(SDI, axis=1))

            for m, Qm in Qs.items():
                sim_tmp[m].extend(gsp.harmonic_similarity(Q_ref, Qm))

        for m in kw:
            vals = np.array(sim_tmp[m])
            kw[m].append(vals)
            mean_sim[m].append(vals.mean())
            std_sim[m].append(vals.std())

        for m in sdi_vectors:
            vecs = sdi_vectors[m]
            if len(vecs) < 2:
                continue
            corr = np.corrcoef(vecs)
            sdi_corr[m][bi] = corr[np.triu_indices(len(vecs), k=1)]

    results[label] = {"kw": kw, "mean_sim": mean_sim, "std_sim": std_sim, "bins": bins}
    sdi_results[label] = sdi_corr
    np.savez(res_path, kw=kw, mean_sim=mean_sim, std_sim=std_sim, bins=bins, sdi_corr=sdi_corr)

print("Computation finished (with caching)")

# ---- Fig5: SDI stability ----
fig, ax = plt.subplots(figsize=(8, 6))
label = "SC-IND"
if label not in sdi_results:
    raise RuntimeError(f"{label} not found in sdi_results. Available keys: {list(sdi_results.keys())}")

sdi_corr_sc = sdi_results[label]
bins = results[label]["bins"]

for b in bins:
    if b not in sdi_corr_sc["Before alignment"]:
        continue
    vals = sdi_corr_sc["Before alignment"][b]
    jitter = np.random.normal(0, 0.1, len(vals))
    ax.scatter(b + jitter, vals, alpha=0.12, s=8, color="#7f7ab8")

means = [np.mean(sdi_corr_sc["Before alignment"][b]) if b in sdi_corr_sc["Before alignment"] else np.nan for b in bins]
stds = [np.std(sdi_corr_sc["Before alignment"][b]) if b in sdi_corr_sc["Before alignment"] else np.nan for b in bins]
ax.errorbar(bins, means, yerr=stds, fmt="o", color="#5e5ac5", markeredgecolor="black", markersize=7, linewidth=2, capsize=4, zorder=5, label="Before alignment")

valid = ~np.isnan(means)
x_valid, y_valid = np.array(bins)[valid], np.array(means)[valid]
if len(x_valid) > 3:
    x_smooth = np.linspace(min(x_valid), max(x_valid), 200)
    y_smooth = make_interp_spline(x_valid, y_valid, k=3)(x_smooth)
    ax.plot(x_smooth, y_smooth, linestyle="--", color="#5e5ac5", linewidth=2)

r, p = pearsonr(x_valid, y_valid)
ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.2e}", transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))
ax.set_ylim(0.3, 1.02)
ax.grid(alpha=0.1)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "Fig5_SDI_stability.png"), dpi=600)
plt.close()
print("Saved: Fig5_SDI_stability.png")

# ---- Fig5: ROI consistency (surrogate-based, LT vs RT) ----
print("Generating Fig5c (SC-IND, LEFT vs RIGHT)...")

SC_ind = np.load(os.path.join(DATA_DIR, "SC", "matMetric_IND_CTRL_FULL.npy"))
if SC_ind.ndim == 3 and SC_ind.shape[1] != SC_ind.shape[2]:
    SC_ind = np.transpose(SC_ind, (2, 0, 1))
n_subj = SC_ind.shape[0]
print(f"SC-IND subjects: {n_subj}")

lat_labels = np.array([str(p["lat"][0]) for p in X_RS_allPat])
LT_idx = np.where(np.char.find(lat_labels, "L") >= 0)[0]
RT_idx = np.where(np.char.find(lat_labels, "R") >= 0)[0]
print(f"LEFT EEG: {len(LT_idx)} | RIGHT EEG: {len(RT_idx)}")

sig_maps_LT, sig_maps_RT = [], []
ls_lateralization_fig5 = ["RT", "LT"]

for subj in tqdm(range(n_subj), desc="SC-IND subjects"):
    consensus = SC_ind[subj, :, :]

    for lateralization in ls_lateralization_fig5:
        P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, Euc, plot=False)
        X_RS_allPat_loop = gsp.load_EEG_example(EEG_DIR)

        SDI_tmp = np.zeros((118, len(X_RS_allPat_loop)))
        ls_cutoff, ls_lat = [], []

        for p in np.arange(len(X_RS_allPat_loop)):
            X_RS = X_RS_allPat_loop[p]["X_RS"]
            ls_lat.append(X_RS_allPat_loop[p]["lat"][0])
            PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_ind, X_RS)
            ls_cutoff.append(NN)
            SDI_tmp[:, p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_ind)

        np.save(os.path.join(output_dir, f"cutoff_IND_{lateralization}_mat{subj}.npy"), ls_cutoff)
        ls_lat = np.array(ls_lat)
        SDI = SDI_tmp

        idxs_lat = np.where(ls_lat == ("Rtle" if lateralization == "RT" else "Ltle"))[0]
        SDI = SDI[:, idxs_lat]
        np.save(os.path.join(output_dir, f"SDI_IND_{lateralization}_mat{subj}.npy"), SDI)

        surr_path = os.path.join(output_dir, f"SDI_surr_IND_{lateralization}_mat{subj}_nbSurr{nbSurr_fig5}.npy")
        if not os.path.exists(surr_path):
            SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, EEG_DIR, nbSurr=nbSurr_fig5, example=False)
            np.save(surr_path, SDI_surr)
        else:
            SDI_surr = np.load(surr_path)
            print("Surrogate SDI already generated")

        surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:, :, idxs_lat])
        np.save(os.path.join(output_dir, f"SDI_surr_thresh_IND_{lateralization}_mat{subj}.npy"), surr_thresh, allow_pickle=True)

        if lateralization == "LT":
            sig_maps_LT.append(np.abs(surr_thresh[5]["SDI_sig"]))
        else:
            sig_maps_RT.append(np.abs(surr_thresh[5]["SDI_sig"]))

sig_maps_LT = np.array(sig_maps_LT)
sig_maps_RT = np.array(sig_maps_RT)
roi_counts_LT = np.sum(sig_maps_LT, axis=0)
roi_counts_RT = np.sum(sig_maps_RT, axis=0)

plt.rcParams.update({"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 16, "xtick.labelsize": 1, "ytick.labelsize": 12})

threshold = 5
idx_LT = np.where(roi_counts_LT > threshold)[0]
idx_RT = np.where(roi_counts_RT > threshold)[0]
roi_LT, roi_RT = roi_counts_LT[idx_LT], roi_counts_RT[idx_RT]
labels_LT = [roi_labels[i] for i in idx_LT]
labels_RT = [roi_labels[i] for i in idx_RT]

order_LT = np.argsort(roi_LT)[::-1]
order_RT = np.argsort(roi_RT)[::-1]
roi_LT, roi_RT = roi_LT[order_LT], roi_RT[order_RT]
labels_LT = [labels_LT[i] for i in order_LT]
labels_RT = [labels_RT[i] for i in order_RT]

x_LT, x_RT = np.arange(len(roi_LT)), np.arange(len(roi_RT))
fig, axes = plt.subplots(1, 2, figsize=(25, 10), constrained_layout=True,gridspec_kw={"width_ratios": [len(roi_LT), len(roi_RT)]})
axes[0].bar(x_LT, roi_LT, color="#156082", edgecolor="black", linewidth=.7, width=.8)
axes[0].grid(axis="y", alpha=0.3)
axes[1].bar(x_RT, roi_RT, color="#196B24", edgecolor="black", linewidth=.7, width=.8)
axes[0].set_xticks(range(len(labels_LT))); axes[0].set_xticklabels(labels_LT, fontsize=18, rotation=45, ha="right")
axes[1].set_xticks(range(len(labels_RT))); axes[1].set_xticklabels(labels_RT, fontsize=18, rotation=45, ha="right")
max_y = max(max(roi_LT), max(roi_RT))
axes[0].set_ylim(0, max_y + 1)
axes[1].set_ylim(0, max_y + 1)

plt.savefig(os.path.join(figures_dir, "Fig5_ROIs_consistency_indSC.png"), dpi=600, bbox_inches="tight")
plt.close()
print("Saved: Fig5_ROIs_consistency_indSC.png")

print("\n" + "=" * 80)
print("Main-figure pipeline complete: Fig2, Fig3 (LT/RT), Fig5")
print("=" * 80)