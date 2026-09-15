"""
Supplementary analyses and figures (FigSX)
============================================

Consolidates the alignment-method comparison (SC-IND vs SC-HC) and the
descriptive supplementary figures on significant-ROI counts and cutoff
frequencies.

Run Script_SDI_3consensus.py first: FigS1 and FigS2 reuse the cutoff-frequency
arrays (cutoff_HC_RT.npy, cutoff_EP_RT.npy) and the significant-ROI-count
arrays (nbROIs_sig_*.npy) written to OUTPUT/ by that script's PART 1.

Outputs
-------
  - FIGURES/FigS3_harmonic_similarity_SC_IND_vs_SC_HC.png
  - FIGURES/FigS4_inter_method_correlation_LT_RT.png
  - FIGURES/FigS1_nbROIs_comparison.png
  - FIGURES/FigS2_cutoff_comparison.png

Emeline Mullier, University of Geneva & Lausanne University Hospital
"""

import os
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, mannwhitneyu, wilcoxon as _wilcoxon
import lib.func_GSP as gsp

# ----------------------------------------------------------------------------
# Style & paths
# ----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Aptos', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 10

project_root = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(project_root, "DATA/EEG")
output_dir = os.path.join(project_root, "OUTPUT")
figures_dir = os.path.join(project_root, "FIGURES")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
nbSurr = 100

# ============================================================================
# PART 1 - FigS3: alignment methods, SC-IND vs SC-HC (Procrustes / Hungarian)
# ============================================================================
print("=" * 80)
print("PART 1: FigS3 - alignment methods, SC-IND vs SC-HC")
print("=" * 80)

consensus_HC_DSI = np.load(os.path.join(project_root, "DATA", "SC", "matMetric_HC_dsi_number_of_fibers.npy"))
consensus_IND = np.load(os.path.join(project_root, "DATA", "SC", "matMetric_IND_CTRL_FULL.npy"))
EucDist_ref = np.load(os.path.join(project_root, "DATA", "EucMat", "EucMat_HC_dsi_number_of_fibers.npy"))

consensus_HC_ref = np.mean(consensus_HC_DSI, axis=2)
consensus_IND_mean = np.mean(consensus_IND, axis=0)

P_ref, Q_ref, Ln_ref, An_ref = gsp.cons_normalized_lap(consensus_HC_ref, EucDist_ref, plot=False)
P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus_IND_mean, EucDist_ref, plot=False)

Qind_rotated_raw, Qind_HC_centered, disparity = scipy.spatial.procrustes(Q_ref, Q_ind)
_Uind, _, _Vtind = scipy.linalg.svd(Qind_rotated_raw, full_matrices=False)
Qind_rotated = _Uind @ _Vtind
perm_ind, total_cost_ind = gsp.match_eigenvectors(Q_ref, Q_ind)
Qind_matched = Q_ind[:, perm_ind]

nb_eig = Q_ref.shape[1]
similarity_ind = np.zeros(nb_eig)
similarity_rotated = np.zeros(nb_eig)
similarity_matched = np.zeros(nb_eig)
for eigvec_nb in range(nb_eig):
    similarity_ind[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Q_ind[:, eigvec_nb])
    similarity_rotated[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_rotated[:, eigvec_nb])
    similarity_matched[eigvec_nb] = 1 - scipy.spatial.distance.correlation(Q_ref[:, eigvec_nb], Qind_matched[:, eigvec_nb])
similarity_ind = np.abs(similarity_ind)
similarity_rotated = np.abs(similarity_rotated)
similarity_matched = np.abs(similarity_matched)

X_RS_allPat = gsp.load_EEG_example(example_dir)

methods = {"SC_HC_ref": Q_ref, "SC_IND": Q_ind, "Gen_Procrustes": Qind_rotated, "Hungarian": Qind_matched}
method_file_prefixes = {"SC_HC_ref": "SC_HC_ref", "SC_IND": "SC_IND", "Gen_Procrustes": "Gen_Procrustes", "Hungarian": "Hungarian"}
cutoff_file_prefixes = {"SC_HC_ref": "HC", "SC_IND": "IND", "Gen_Procrustes": "Gen_Procrustes", "Hungarian": "Hungarian"}
sdi_results = {}

for method_name, Q_method in methods.items():
    for lateralization in ["LT", "RT"]:
        SDI = np.zeros((118, len(X_RS_allPat)))
        cutoff_values = []
        for p, patient in enumerate(X_RS_allPat):
            X_RS = X_RS_allPat[p]["X_RS"]
            PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q_method, patient["X_RS"])
            cutoff_values.append(NN)
            SDI[:, p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q_method)
        cutoff_values = np.array(cutoff_values)
        np.save(os.path.join(output_dir, f"cutoff_{cutoff_file_prefixes[method_name]}_{lateralization}.npy"), cutoff_values)

        surr_thresh_path = os.path.join(output_dir, f"SDI_surr_thresh_{method_file_prefixes[method_name]}_{lateralization}.npy")
        if os.path.exists(surr_thresh_path):
            surr_thresh = np.load(surr_thresh_path, allow_pickle=True)
        else:
            SDI_surr = gsp.surrogate_sdi(Q_method, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False)
            surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr)
            np.save(surr_thresh_path, surr_thresh)

        sdi_results[f"{method_name}_{lateralization}"] = {"surr_thresh": surr_thresh, "cutoff_frequencies": cutoff_values}


def _wilcoxon_vs_raw(aligned, raw):
    diff = aligned - raw
    if np.all(diff == 0):
        return np.nan, np.nan, "n.s."
    W, p = _wilcoxon(aligned, raw, alternative="greater")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    return W, p, sig

W_rotated, p_rotated, sig_rotated = _wilcoxon_vs_raw(similarity_rotated, similarity_ind)
W_matched, p_matched, sig_matched = _wilcoxon_vs_raw(similarity_matched, similarity_ind)

print("FigS3 - Wilcoxon signed-rank, aligned vs no-alignment (alternative='greater')")
for lbl, W, p, sig, arr in [("Procrustes", W_rotated, p_rotated, sig_rotated, similarity_rotated),
                              ("Hungarian", W_matched, p_matched, sig_matched, similarity_matched)]:
    delta = np.mean(arr) - np.mean(similarity_ind)
    print(f"  {lbl:<25} W={W:>10.1f}  p={p:>12.4e}  delta={delta:>+8.4f}  {sig:>5}")

fig2, ax2 = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
n_harmonics = len(similarity_ind)
harmonic_indices = np.arange(n_harmonics)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
similarity_ylim = (0, 1.02)

ax2[0].plot(harmonic_indices, similarity_ind, linewidth=2, color=colors[0])
mean_ind, std_ind = np.mean(similarity_ind), np.std(similarity_ind)
ax2[0].axhline(y=mean_ind, color=colors[0], linestyle=":", linewidth=1.5, alpha=0.7)
ax2[0].text(n_harmonics + 1, mean_ind, f"{mean_ind:.3f}\u00b1{std_ind:.3f}", color=colors[0], fontsize=8, va="center", ha="left", fontweight="bold")
ax2[0].fill_between(harmonic_indices, similarity_ind - std_ind, similarity_ind + std_ind, alpha=0.15, color=colors[0])
ax2[0].set_title(f"A. Before alignment (mean r={mean_ind:.3f}\u00b1{std_ind:.3f})", fontsize=13, fontweight="bold", loc="left", pad=10)
ax2[0].set_xlabel("Eigenmode", fontsize=12, fontweight="bold"); ax2[0].set_ylabel("Correlation", fontsize=12, fontweight="bold")

ax2[1].plot(harmonic_indices, similarity_rotated, linewidth=2, color=colors[2])
mean_rotated, std_rotated = np.mean(similarity_rotated), np.std(similarity_rotated)
ax2[1].axhline(y=mean_rotated, color=colors[2], linestyle=":", linewidth=1.5, alpha=0.7)
ax2[1].text(n_harmonics + 1, mean_rotated, f"{mean_rotated:.3f}\u00b1{std_rotated:.3f}", color=colors[2], fontsize=8, va="center", ha="left", fontweight="bold")
ax2[1].fill_between(harmonic_indices, similarity_rotated - std_rotated, similarity_rotated + std_rotated, alpha=0.15, color=colors[2])
ax2[1].set_title(f"B. Procrustes (mean r={mean_rotated:.3f}\u00b1{std_rotated:.3f})\nvs before alignment, p={p_rotated:.3e} {sig_rotated} (Wilcoxon signed-rank)", fontsize=11, fontweight="bold", loc="left", pad=10)
ax2[1].set_xlabel("Eigenmode", fontsize=12, fontweight="bold"); ax2[1].set_ylabel("Correlation", fontsize=12, fontweight="bold")

ax2[2].plot(harmonic_indices, similarity_matched, linewidth=2, color=colors[3])
mean_matched, std_matched = np.mean(similarity_matched), np.std(similarity_matched)
ax2[2].axhline(y=mean_matched, color=colors[3], linestyle=":", linewidth=1.5, alpha=0.7)
ax2[2].text(n_harmonics + 1, mean_matched, f"{mean_matched:.3f}\u00b1{std_matched:.3f}", color=colors[3], fontsize=8, va="center", ha="left", fontweight="bold")
ax2[2].fill_between(harmonic_indices, similarity_matched - std_matched, similarity_matched + std_matched, alpha=0.15, color=colors[3])
ax2[2].set_title(f"C. Hungarian matching (mean r={mean_matched:.3f}\u00b1{std_matched:.3f})\nvs before alignment, p={p_matched:.3e} {sig_matched} (Wilcoxon signed-rank)", fontsize=11, fontweight="bold", loc="left", pad=10)
ax2[2].set_xlabel("Eigenmode", fontsize=12, fontweight="bold"); ax2[2].set_ylabel("Correlation", fontsize=12, fontweight="bold")

for i in range(3):
    ax2[i].grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    ax2[i].set_ylim(similarity_ylim)
    ax2[i].set_xticks(range(0, n_harmonics, 20))
    ax2[i].spines["top"].set_visible(False); ax2[i].spines["right"].set_visible(False)
    ax2[i].spines["left"].set_linewidth(1.5); ax2[i].spines["bottom"].set_linewidth(1.5)
    ax2[i].tick_params(labelsize=11)

figS3_path = os.path.join(figures_dir, "FigS3_harmonic_similarity_SC_IND_vs_SC_HC.png")
plt.savefig(figS3_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {os.path.basename(figS3_path)}")


# ============================================================================
# PART 2 - FigS4: inter-method correlation heatmaps (LT/RT)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: FigS4 - inter-method correlation heatmaps")
print("=" * 80)

groups_heatmap_s4 = [
    ("SC_HC_ref", r"SC$_{HC}$ reference"), ("SC_IND", r"Before alignment"),
    ("Gen_Procrustes", r"Procrustes"), ("Hungarian", r"Hungarian"),]

thr = 0
matrices_by_side_s4 = {}
for lateralization in ["LT", "RT"]:
    sdi_vectors = [sdi_results[f"{group_name}_{lateralization}"]["surr_thresh"][thr]["mean_SDI"] for group_name, _ in groups_heatmap_s4]
    n_groups = len(sdi_vectors)
    r_mat, p_mat = np.eye(n_groups), np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                continue
            r_mat[i, j], p_mat[i, j] = pearsonr(sdi_vectors[i], sdi_vectors[j])
    labels = [label for _, label in groups_heatmap_s4]
    matrices_by_side_s4[lateralization] = {
        "r_df": pd.DataFrame(r_mat, index=labels, columns=labels),
        "p_df": pd.DataFrame(p_mat, index=labels, columns=labels),
        "r_mat": r_mat, "p_mat": p_mat,}

purple_cmap_s4 = sns.light_palette("purple", as_cmap=True)
mask_lower_s4 = np.triu(np.ones_like(matrices_by_side_s4["LT"]["r_df"], dtype=bool), k=0)
mask_upper_s4 = np.tril(np.ones_like(matrices_by_side_s4["LT"]["r_df"], dtype=bool), k=0)

figS4, axS4 = plt.subplots(1, 2, figsize=(6.5, 5.8), constrained_layout=True)
sns.heatmap(matrices_by_side_s4["LT"]["r_df"], mask=mask_lower_s4, ax=axS4[0], cmap="Blues", vmin=0, vmax=1, square=True, linewidths=0.5, linecolor="white", cbar=False, annot=False)
sns.heatmap(matrices_by_side_s4["RT"]["r_df"], mask=mask_upper_s4, ax=axS4[0], cmap="Greens", vmin=0, vmax=1, square=True, linewidths=0.5, linecolor="white", cbar=False, annot=False)
for d in range(n_groups):
    axS4[0].add_patch(plt.Rectangle((d, d), 1, 1, facecolor="lightgray", edgecolor="white", linewidth=0.5, zorder=3))
    axS4[0].text(d + 0.5, d + 0.5, "\u2014", ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
for i in range(n_groups):
    for j in range(n_groups):
        if i > j:
            p_lt = matrices_by_side_s4["LT"]["p_mat"][i, j]
            stars_lt = "***" if p_lt < 0.001 else "**" if p_lt < 0.01 else "*" if p_lt < 0.05 else ""
            axS4[0].text(j + 0.5, i + 0.5, f"{matrices_by_side_s4['LT']['r_mat'][i, j]:.2f}{stars_lt}", ha="center", va="center", fontsize=8, fontweight="bold" if p_lt < 0.05 else "normal")
        elif i < j:
            p_rt = matrices_by_side_s4["RT"]["p_mat"][i, j]
            stars_rt = "***" if p_rt < 0.001 else "**" if p_rt < 0.01 else "*" if p_rt < 0.05 else ""
            axS4[0].text(j + 0.5, i + 0.5, f"{matrices_by_side_s4['RT']['r_mat'][i, j]:.2f}{stars_rt}", ha="center", va="center", fontsize=8, fontweight="bold" if p_rt < 0.05 else "normal")
axS4[0].tick_params(axis="x", rotation=20, labelsize=10); axS4[0].tick_params(axis="y", rotation=0, labelsize=10)

sns.heatmap(matrices_by_side_s4["LT"]["p_df"], mask=mask_lower_s4, ax=axS4[1], cmap=purple_cmap_s4, vmin=0, vmax=1, square=True, linewidths=0.5, linecolor="white", cbar=False, annot=False)
sns.heatmap(matrices_by_side_s4["RT"]["p_df"], mask=mask_upper_s4, ax=axS4[1], cmap=purple_cmap_s4, vmin=0, vmax=1, square=True, linewidths=0.5, linecolor="white", cbar=False, annot=False)
for d in range(n_groups):
    axS4[1].add_patch(plt.Rectangle((d, d), 1, 1, facecolor="lightgray", edgecolor="white", linewidth=0.5, zorder=3))
    axS4[1].text(d + 0.5, d + 0.5, "\u2014", ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
for i in range(n_groups):
    for j in range(n_groups):
        if i > j:
            p_lt = matrices_by_side_s4["LT"]["p_mat"][i, j]
            stars_lt = "***" if p_lt < 0.001 else "**" if p_lt < 0.01 else "*" if p_lt < 0.05 else ""
            axS4[1].text(j + 0.5, i + 0.5, f"{p_lt:.1e}{stars_lt}", ha="center", va="center", fontsize=8, fontweight="bold" if p_lt < 0.05 else "normal", color="black")
        elif i < j:
            p_rt = matrices_by_side_s4["RT"]["p_mat"][i, j]
            stars_rt = "***" if p_rt < 0.001 else "**" if p_rt < 0.01 else "*" if p_rt < 0.05 else ""
            axS4[1].text(j + 0.5, i + 0.5, f"{p_rt:.1e}{stars_rt}", ha="center", va="center", fontsize=8, fontweight="bold" if p_rt < 0.05 else "normal", color="black")
axS4[1].set_title("p-value: lower Left IED, upper Right IED", fontsize=8, fontweight="bold")
axS4[1].tick_params(axis="x", rotation=20, labelsize=10); axS4[1].tick_params(axis="y", rotation=0, labelsize=10)

figS4_path = os.path.join(figures_dir, "FigS4_inter_method_correlation_LT_RT.png")
plt.savefig(figS4_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {os.path.basename(figS4_path)}")


# ============================================================================
# PART 3 - FigS1: number of significant ROIs per threshold
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: FigS1 - number of significant ROIs per threshold")
print("=" * 80)

nbROIs_HC_RT = np.load(os.path.join(output_dir, "nbROIs_sig_HC_RT.npy"))
nbROIs_HC_LT = np.load(os.path.join(output_dir, "nbROIs_sig_HC_LT.npy"))
nbROIs_EP_RT = np.load(os.path.join(output_dir, "nbROIs_sig_EP_RT.npy"))
nbROIs_EP_LT = np.load(os.path.join(output_dir, "nbROIs_sig_EP_LT.npy"))
nbROIs_IND_RT = np.load(os.path.join(output_dir, "nbROIs_sig_IND_RT.npy"))
nbROIs_IND_LT = np.load(os.path.join(output_dir, "nbROIs_sig_IND_LT.npy"))

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
ls_nbROIs_LT = [nbROIs_HC_LT, nbROIs_EP_LT, nbROIs_IND_LT]
ls_nbROIs_RT = [nbROIs_HC_RT, nbROIs_EP_RT, nbROIs_IND_RT]
ls_labels = ["HC", "TLE", "IND"]
ls_colors = ["#1f77b4", "#2ca02c", "#9467bd"]
ls_markers = ["o", "s", "^"]

for nbROIs, label, color, marker in zip(ls_nbROIs_LT, ls_labels, ls_colors, ls_markers):
    axes[0].plot(np.arange(len(nbROIs)), np.array(nbROIs), marker=marker, linewidth=2.5, markersize=6, color=color, label=label)
axes[0].set_xlabel("Threshold", fontsize=12, fontweight="bold"); axes[0].set_ylabel("# ROIs with significant SDI", fontsize=12, fontweight="bold")
axes[0].set_xticks(np.arange(0, len(nbROIs_HC_LT))); axes[0].grid(True, alpha=0.3, linestyle="--")
axes[0].set_title("Number of Significant SDI ROIs per Threshold (Left IED)", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=11, loc="upper right", framealpha=0.9); axes[0].tick_params(labelsize=10)

for nbROIs, label, color, marker in zip(ls_nbROIs_RT, ls_labels, ls_colors, ls_markers):
    axes[1].plot(np.arange(len(nbROIs)), np.array(nbROIs), marker=marker, linewidth=2.5, markersize=6, color=color, label=label)
axes[1].set_xlabel("Threshold", fontsize=12, fontweight="bold"); axes[1].set_ylabel("# ROIs with significant SDI", fontsize=12, fontweight="bold")
axes[1].set_xticks(np.arange(0, len(nbROIs_HC_RT))); axes[1].grid(True, alpha=0.3, linestyle="--")
axes[1].set_title("Number of Significant SDI ROIs per Threshold (Right IED)", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=11, loc="upper right", framealpha=0.9); axes[1].tick_params(labelsize=10)

figS1_path = os.path.join(figures_dir, "FigS1_nbROIs_comparison.png")
plt.savefig(figS1_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.basename(figS1_path)}")


# ============================================================================
# PART 4 - FigS2: cutoff-frequency comparison (HC, EP, IND)
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: FigS2 - cutoff frequency comparison")
print("=" * 80)

cutoff_HC = np.load(os.path.join(output_dir, "cutoff_HC_RT.npy"))
cutoff_EP = np.load(os.path.join(output_dir, "cutoff_EP_RT.npy"))
cutoff_IND = np.load(os.path.join(output_dir, "cutoff_IND_RT.npy"))

fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

r_ep, p_ep = pearsonr(cutoff_HC, cutoff_EP)
r_ind, p_ind = pearsonr(cutoff_HC, cutoff_IND)
ax[0].scatter(cutoff_HC, cutoff_EP, c="#ff7f0e", alpha=0.6, s=80, label=f"HC vs EP (r={r_ep:.2f}, p={p_ep:.3f})", edgecolors="darkgray")
ax[0].scatter(cutoff_HC, cutoff_IND, c="#2ca02c", alpha=0.6, s=80, label=f"HC vs IND (r={r_ind:.2f}, p={p_ind:.3f})", edgecolors="darkgray")
ax[0].set_xlabel("Cutoff frequency HC", fontsize=11, fontweight="bold"); ax[0].set_ylabel("Cutoff frequency (EP/IND)", fontsize=11, fontweight="bold")
ax[0].set_title(f"Cutoff Frequency Comparison\nHC\u2013EP: r={r_ep:.2f}, p={p_ep:.3f} | HC\u2013IND: r={r_ind:.2f}, p={p_ind:.3f}", fontsize=11, fontweight="bold")
ax[0].legend(fontsize=9); ax[0].grid(True, alpha=0.3, linestyle="--"); ax[0].tick_params(labelsize=10)

box_data = [cutoff_HC, cutoff_EP, cutoff_IND]
box_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
sns.boxplot(data=box_data, ax=ax[1], width=0.5, palette=box_palette)
for patch in ax[1].patches:
    patch.set_alpha(0.8)
sns.stripplot(data=box_data, ax=ax[1], color="black", size=5, jitter=True, alpha=0.6)
ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(["SC HC", "SC EP", "SC IND"], fontsize=11, fontweight="bold")
ax[1].set_ylabel("Cutoff frequency", fontsize=11, fontweight="bold")
ax[1].set_title("Cutoff Frequency Distribution", fontsize=12, fontweight="bold")
ax[1].grid(True, axis="y", linestyle="--", alpha=0.3); ax[1].tick_params(labelsize=10)

combined = np.concatenate(box_data)
y_base = combined.max() if combined.size else 1.0
y_step = 0.08 * y_base
pairs_stats = [("HC", "EP", cutoff_HC, cutoff_EP, 0, 1), ("HC", "IND", cutoff_HC, cutoff_IND, 0, 2), ("EP", "IND", cutoff_EP, cutoff_IND, 1, 2)]
for i, (g1, g2, v1, v2, x1, x2) in enumerate(pairs_stats):
    stat, pval = mannwhitneyu(v1, v2, alternative="two-sided")
    y = y_base + (i + 1) * y_step
    ax[1].plot([x1, x1, x2, x2], [y, y + 0.02 * y_base, y + 0.02 * y_base, y], color="k", linewidth=1)
    sig_text = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ax[1].text((x1 + x2) / 2, y + 0.025 * y_base, f"p = {pval:.3f} {sig_text}", ha="center", va="bottom", fontsize=9)
    print(f"Cutoff comparison {g1} vs {g2}: U={stat:.2f}, p={pval:.4f}")

figS2_path = os.path.join(figures_dir, "FigS2_cutoff_comparison.png")
plt.savefig(figS2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.basename(figS2_path)}")

print("\n" + "=" * 80)
print("Supplementary-figure pipeline complete: FigS1, FigS2, FigS3, FigS4")
print("=" * 80)