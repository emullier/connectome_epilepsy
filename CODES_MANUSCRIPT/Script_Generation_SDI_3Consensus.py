

''' Manuscript comparison script merging codes 02 and 07
This script generates a comprehensive comparison table of SDI values for:
- HC (Healthy Controls - Geneva dataset)
- EP (Epilepsy patients - Geneva dataset) 
- IND (Independent 27 healthy controls consensus)

The script computes consensus matrices, calculates SDI values, and generates comparison tables and figures.

Fig 1: Comparison of cutoff frequencies between HC, EP, and IND
Fig 2: Correlation between mean SDI values across the three groups
Fig 3: Number of significant SDI ROIs for different thresholds
Fig 4: Brain plots showing mean SDI distributions

Saved output: 
- Comparison table: SDI_comparison_table_Manuscript.xlsx
- Figures: ./FIGURES/Manuscript/

Last modified: EM, 05.12.2025
Created: 05.12.2025, Emeline Mullier
University of Geneva & Lausanne University Hospital '''

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import lib modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lib.func_GSP as gsp
from lib.func_plot import plot_rois, plot_rois_pyvista, plot_rois_pyvista_noaxes
import scipy
import scipy.io as sio
from scipy.stats import pearsonr, mannwhitneyu
import seaborn as sns
from tabulate import tabulate

# Set matplotlib font to Aptos Body (with fallback to sans-serif)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Aptos', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 10

# Define consistent color palette (matching code 02 style)
COLORS_COMPARISON = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']  # HC, EP, IND
MARKERS_COMPARISON = ['o', 's', '^']  # HC, EP, IND

# Get the directory paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

scale = 2
example_dir = os.path.join(project_root, "DATA/EEG")
infoGVA_path = os.path.join(project_root, 'DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv')

# Create output directories
output_dir = os.path.join(project_root, 'OUTPUT/Manuscript/')
figures_dir = os.path.join(project_root, 'FIGURES/Manuscript/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# ============================================================================
# PART 1: Load HC and EP data from Geneva datasets (from code 02)
# ============================================================================
print("\n" + "="*80)
print("PART 1: Processing HC and EP consensus matrices (Geneva dataset)")
print("="*80)

ls_groups_geneva = ["HC", "EP"]
ls_lateralization = ["RT", "LT"]
metric = "number_of_fibers"
dwi = "dsi"

surr_thresh_storage = {}  # Store all surrogate thresholds

for group in ls_groups_geneva:
    for lateralization in ls_lateralization:
        print(f"\nProcessing {group} {lateralization}...")
        
        # Load info
        df_info_orig = pd.read_csv(infoGVA_path)
        idxs2keep = np.where((df_info_orig['Inclusion']==1)*(df_info_orig['group']==group)*(df_info_orig['dwi']==dwi))[0]
        df_info = df_info_orig.iloc[idxs2keep]
        
        # Load the data
        data_path = os.path.join(project_root, f"DATA/SC/matMetric_{group}_{dwi}_{metric}.npy")
        matMetric = np.load(data_path)
        
        if group == 'EP':
            idxs = np.where(df_info['Lateralization']==lateralization)[0]
            matMetric = matMetric[:,:,idxs]
        
        # Generate consensus
        consensus = np.mean(matMetric, axis=2)
        EucDist = np.load(os.path.join(project_root, f"DATA/EucMat/EucMat_{group}_{dwi}_{metric}.npy"))
        
        # Generate harmonics
        P, Q, Ln, An = gsp.cons_normalized_lap(consensus, EucDist, plot=False)
        
        # Project functional signals
        X_RS_allPat = gsp.load_EEG_example(example_dir)
        
        # Estimate SDI
        ls_cutoff = []
        SDI_tmp = np.zeros((118, len(X_RS_allPat)))
        ls_lat = []
        SDI = {}
        
        for p in np.arange(len(X_RS_allPat)):
            X_RS = X_RS_allPat[p]['X_RS']
            ls_lat.append(X_RS_allPat[p]['lat'][0])
            PSD, NN, Vlow, Vhigh = gsp.get_cutoff_freq(Q, X_RS)
            ls_cutoff.append(NN)
            SDI_tmp[:,p], X_c_norm, X_d_norm, SD_hat = gsp.compute_SDI(X_RS, Q)
        
        np.save(os.path.join(output_dir, f'cutoff_{group}_{lateralization}.npy'), ls_cutoff)
        ls_lat = np.array(ls_lat)
        SDI = SDI_tmp
        
        if lateralization == 'RT':
            idxs_lat = np.where(ls_lat=='Rtle')[0]
        elif lateralization == 'LT':
            idxs_lat = np.where(ls_lat=='Ltle')[0]
        
        SDI = SDI[:, idxs_lat]
        np.save(os.path.join(output_dir, f'SDI_{group}_{lateralization}.npy'), SDI)
        
        # Plot mean SDI
        plot_rois_pyvista_noaxes(np.mean(SDI, axis=1), scale, figures_dir, 
                                vmin=-2, vmax=2, label=f'SDImean_{group}_{lateralization}')
        
        # Surrogate part - use same strategy as script 02: load from EPvsCTRL folder with standard path
        nbSurr = 100
        surr_path = os.path.join(project_root, f'OUTPUT/EPvsCTRL/SDI_surr_{metric}_{group}_{dwi}_{lateralization}.npy')
        if not os.path.exists(surr_path):
            SDI_surr = gsp.surrogate_sdi(Q, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False)
            np.save(surr_path, SDI_surr)
        else:
            SDI_surr = np.load(surr_path)
            print('Surrogate SDI already generated')
        
        # Select significant SDI
        surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
        surr_thresh_path = os.path.join(output_dir, f'SDI_surr_thresh_{group}_{lateralization}.npy')
        np.save(surr_thresh_path, surr_thresh, allow_pickle=True)
        
        # Store for later use
        surr_thresh_storage[f"{group}_{lateralization}"] = surr_thresh
        
        # Count significant ROIs
        nbROIs_sig = []
        for p in np.arange(np.shape(surr_thresh)[0]):
            nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
        np.save(os.path.join(output_dir, f'nbROIs_sig_{group}_{lateralization}.npy'), nbROIs_sig)
        
        # Plot thresholds
        for thr in [2, 5]:
            plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), 
                                    scale, figures_dir, vmin=-1, vmax=1, 
                                    label=f'SDImean_thr{thr}_{group}_{lateralization}')

# ============================================================================
# PART 2: Load IND (27 healthy controls) data from code 07
# ============================================================================
print("\n" + "="*80)
print("PART 2: Processing IND consensus matrix (Independent 27 HC dataset)")
print("="*80)

for lateralization in ls_lateralization:
    print(f"\nProcessing IND {lateralization}...")
    
    # Load the data from matMetric_SCHZ_CTRL.npy
    matMetric = np.load(os.path.join(project_root, "DATA/SC/matMetric_SCHZ_CTRL.npy"))
    consensus = np.mean(matMetric, axis=0)
    EucDist = np.load(os.path.join(project_root, "DATA/EucMat/EucMat_HC_DSI_number_of_fibers.npy"))
    
    # Generate harmonics
    P_ind, Q_ind, Ln_ind, An_ind = gsp.cons_normalized_lap(consensus, EucDist, plot=False)
    
    # Project functional signals
    X_RS_allPat = gsp.load_EEG_example(example_dir)
    
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
    
    np.save(os.path.join(output_dir, f'cutoff_IND_{lateralization}.npy'), ls_cutoff)
    ls_lat = np.array(ls_lat)
    SDI = SDI_tmp
    
    if lateralization == 'RT':
        idxs_lat = np.where(ls_lat=='Rtle')[0]
    elif lateralization == 'LT':
        idxs_lat = np.where(ls_lat=='Ltle')[0]
    
    SDI = SDI[:, idxs_lat]
    np.save(os.path.join(output_dir, f'SDI_IND_{lateralization}.npy'), SDI)
    
    # Plot mean SDI
    plot_rois_pyvista_noaxes(np.mean(SDI, axis=1), scale, figures_dir, 
                            vmin=-2, vmax=2, label=f'SDImean_IND_{lateralization}')
    
    # Surrogate part
    nbSurr = 100
    surr_path = os.path.join(output_dir, f'SDI_surr_IND_{lateralization}.npy')
    if not os.path.exists(surr_path):
        SDI_surr = gsp.surrogate_sdi(Q_ind, Vlow, Vhigh, example_dir, nbSurr=nbSurr, example=False)
        np.save(surr_path, SDI_surr)
    else:
        SDI_surr = np.load(surr_path)
        print('Surrogate SDI already generated')
    
    # Select significant SDI
    surr_thresh, SDI_sig_subjectwise = gsp.select_significant_sdi(SDI, SDI_surr[:,:,idxs_lat])
    surr_thresh_path = os.path.join(output_dir, f'SDI_surr_thresh_IND_{lateralization}.npy')
    np.save(surr_thresh_path, surr_thresh, allow_pickle=True)
    
    # Store for later use
    surr_thresh_storage[f"IND_{lateralization}"] = surr_thresh
    
    # Count significant ROIs
    nbROIs_sig = []
    for p in np.arange(np.shape(surr_thresh)[0]):
        nbROIs_sig.append(len(np.where(np.abs(surr_thresh[p]['SDI_sig']))[0]))
    np.save(os.path.join(output_dir, f'nbROIs_sig_IND_{lateralization}.npy'), nbROIs_sig)
    
    # Plot thresholds
    for thr in [2, 5]:
        plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), 
                                scale, figures_dir, vmin=-1, vmax=1, 
                                label=f'SDImean_thr{thr}_IND_{lateralization}')

# ============================================================================
# PART 3: Create comprehensive comparison table
# ============================================================================
print("\n" + "="*80)
print("PART 3: Creating comprehensive comparison table")
print("="*80)

# Load ROI labels
df_roi = pd.read_csv(os.path.join(project_root, 'DATA/label/labels_rois_118.csv'))
labels_118 = np.array(df_roi['Label Lausanne2008'])

# Collect all groups for table
groups_table = {
    "HC_LT": surr_thresh_storage["HC_LT"],
    "HC_RT": surr_thresh_storage["HC_RT"],
    "EP_LT": surr_thresh_storage["EP_LT"],
    "EP_RT": surr_thresh_storage["EP_RT"],
    "IND_LT": surr_thresh_storage["IND_LT"],
    "IND_RT": surr_thresh_storage["IND_RT"],
}

# Collect all indices that are significant in any group (threshold = 5)
all_idx = set()
for surr in groups_table.values():
    sig_idx = np.where(surr[5]['SDI_sig'] != 0)[0]
    all_idx.update(sig_idx)
all_idx = sorted(list(all_idx))

# Build a dictionary for DataFrame with MultiIndex columns
data = {("ROI", ""): [labels_118[idx] for idx in all_idx]}

for side in ["LT", "RT"]:
    for group in ["EP", "HC", "IND"]:
        col_name = (side, group)
        values = []
        for idx in all_idx:
            key = f"{group}_{side}"
            if groups_table[key][5]['SDI_sig'][idx] != 0:
                values.append(round(groups_table[key][5]['mean_SDI'][idx], 2))
            else:
                values.append(np.nan)
        data[col_name] = values

# Create DataFrame with MultiIndex columns
df_comparison = pd.DataFrame(data)
df_comparison.columns = pd.MultiIndex.from_tuples(df_comparison.columns)

# Print and save table
print("\n" + "="*80)
print("Significant ROIs for HC, EP, and IND (threshold=5)")
print("="*80)
print(df_comparison)

# Export to Excel
df_comparison.to_excel("SDI_comparison_table_Manuscript.xlsx", index=True)
print("\nTable saved to: SDI_comparison_table_Manuscript.xlsx")

# ============================================================================
# PART 4: Correlation between mean SDI values (threshold=5)
# ============================================================================
print("\n" + "="*80)
print("PART 4: Correlation between mean SDI values (threshold=5)")
print("="*80)

# Helper to get all mean SDI values for a given group/side (not masked)
def get_all_mean(group, side, thr=5):
    surr = surr_thresh_storage[f"{group}_{side}"][thr]
    return surr['mean_SDI']

pairs = [("HC", "EP", '#1f77b4', '#ff7f0e'), ("HC", "IND", '#1f77b4', '#2ca02c')]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle("Mean SDI correlations (LT & RT)", fontsize=13, fontweight='bold')

for row_idx, side in enumerate(["LT", "RT"]):
    for ax, (g1, g2, c1, c2) in zip(axes[row_idx], pairs):
        x = get_all_mean(g1, side)
        y = get_all_mean(g2, side)
        if x.size == 0 or y.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=11)
            ax.axis('off')
            continue
        ax.scatter(x, y, color=c2, edgecolor='k', alpha=0.7, s=60)
        r, p = pearsonr(x, y)
        #ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        #ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_title(f"{g1} vs {g2} ({side}, r={r:.2f}, p={p:.3f})", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"Mean SDI {g1}", fontsize=10)
        ax.set_ylabel(f"Mean SDI {g2}", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.text(0.05, 0.92, f"r = {r:.2f}\np = {p:.3f}", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

corr_path = os.path.join(figures_dir, 'corr_mean_SDI_all.png')
plt.savefig(corr_path, dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.basename(corr_path)}")

# ============================================================================
# PART 5: Create comparison plots
# ============================================================================
print("\n" + "="*80)
print("PART 5: Creating comparison plots")
print("="*80)

# Plot cutoff frequencies comparison
cutoff_HC = np.load(os.path.join(output_dir, 'cutoff_HC_RT.npy'))
cutoff_EP = np.load(os.path.join(output_dir, 'cutoff_EP_RT.npy'))
cutoff_IND = np.load(os.path.join(output_dir, 'cutoff_IND_RT.npy'))

fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Scatter plot with correlations
r_ep, p_ep = pearsonr(cutoff_HC, cutoff_EP)
r_ind, p_ind = pearsonr(cutoff_HC, cutoff_IND)
ax[0].scatter(cutoff_HC, cutoff_EP, c='#ff7f0e', alpha=0.6, s=80, label=f'HC vs EP (r={r_ep:.2f}, p={p_ep:.3f})', edgecolors='darkgray')
ax[0].scatter(cutoff_HC, cutoff_IND, c='#2ca02c', alpha=0.6, s=80, label=f'HC vs IND (r={r_ind:.2f}, p={p_ind:.3f})', edgecolors='darkgray')
ax[0].set_xlabel('Cutoff frequency HC', fontsize=11, fontweight='bold')
ax[0].set_ylabel('Cutoff frequency (EP/IND)', fontsize=11, fontweight='bold')
ax[0].set_title(f'Cutoff Frequency Comparison\nHC–EP: r={r_ep:.2f}, p={p_ep:.3f} | HC–IND: r={r_ind:.2f}, p={p_ind:.3f}', fontsize=11, fontweight='bold')
ax[0].legend(fontsize=9)
ax[0].grid(True, alpha=0.3, linestyle='--')
ax[0].tick_params(labelsize=10)

# Boxplot
box_data = [cutoff_HC, cutoff_EP, cutoff_IND]
box_palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
sns.boxplot(data=box_data, ax=ax[1], width=0.5, palette=box_palette)
for patch in ax[1].patches:
    patch.set_alpha(0.8)
sns.stripplot(data=box_data, ax=ax[1], color='black', size=5, jitter=True, alpha=0.6)
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(['HC', 'EP', 'IND'], fontsize=11, fontweight='bold')
ax[1].set_ylabel('Cutoff frequency', fontsize=11, fontweight='bold')
ax[1].set_title('Cutoff Frequency Distribution', fontsize=12, fontweight='bold')
ax[1].grid(True, axis='y', linestyle='--', alpha=0.3)
ax[1].tick_params(labelsize=10)

# Statistical tests on cutoff distributions (Mann-Whitney U)
combined = np.concatenate(box_data)
y_base = combined.max() if combined.size else 1.0
y_step = 0.08 * y_base
pairs_stats = [
    ("HC", "EP", cutoff_HC, cutoff_EP, 0, 1),
    ("HC", "IND", cutoff_HC, cutoff_IND, 0, 2),
    ("EP", "IND", cutoff_EP, cutoff_IND, 1, 2),
]
for i, (g1, g2, v1, v2, x1, x2) in enumerate(pairs_stats):
    stat, pval = mannwhitneyu(v1, v2, alternative='two-sided')
    y = y_base + (i + 1) * y_step
    ax[1].plot([x1, x1, x2, x2], [y, y + 0.02 * y_base, y + 0.02 * y_base, y], color='k', linewidth=1)
    ax[1].text((x1 + x2) / 2, y + 0.025 * y_base, f"p = {pval:.3f}", ha='center', va='bottom', fontsize=9)
    print(f"Cutoff comparison {g1} vs {g2}: U={stat:.2f}, p={pval:.4f}")

plt.savefig(os.path.join(figures_dir, 'cutoff_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved: cutoff_comparison.png")
#plt.close()

# Plot number of significant ROIs
nbROIs_HC_RT = np.load(os.path.join(output_dir, 'nbROIs_sig_HC_RT.npy'))
nbROIs_HC_LT = np.load(os.path.join(output_dir, 'nbROIs_sig_HC_LT.npy'))
nbROIs_EP_RT = np.load(os.path.join(output_dir, 'nbROIs_sig_EP_RT.npy'))
nbROIs_EP_LT = np.load(os.path.join(output_dir, 'nbROIs_sig_EP_LT.npy'))
nbROIs_IND_RT = np.load(os.path.join(output_dir, 'nbROIs_sig_IND_RT.npy'))
nbROIs_IND_LT = np.load(os.path.join(output_dir, 'nbROIs_sig_IND_LT.npy'))

fig, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)

ls_nbROIs = [nbROIs_HC_RT, nbROIs_HC_LT, nbROIs_EP_RT, nbROIs_EP_LT, nbROIs_IND_RT, nbROIs_IND_LT]
ls_labels = ["HC RT", "HC LT", "EP RT", "EP LT", "IND RT", "IND LT"]
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']
ls_markers = ['o', 'o', 's', 's', '^', '^']

for i, (nbROIs, label, color, marker) in enumerate(zip(ls_nbROIs, ls_labels, ls_colors, ls_markers)):
    ax.plot(np.arange(len(nbROIs)), np.array(nbROIs), marker=marker, linewidth=2.5, markersize=6, 
           color=color, label=label)
    # Add value labels on markers
    for j, y_value in enumerate(nbROIs):
        ax.text(j, y_value + 0.3, f'{int(y_value)}', fontsize=7, ha='center', va='bottom', color=color, fontweight='bold')

ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('# ROIs with significant SDI', fontsize=12, fontweight='bold')
ax.set_xticks(np.arange(0, len(nbROIs_HC_RT)))
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_title('Number of Significant SDI ROIs per Threshold', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='best', ncol=3, framealpha=0.9)
ax.tick_params(labelsize=10)

plt.savefig(os.path.join(figures_dir, 'nbROIs_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved: nbROIs_comparison.png")
#plt.close()

print("\n" + "="*80)
print("Manuscript comparison analysis complete!")
print("="*80)
print("\nGenerated outputs:")
print("  - Comparison table: SDI_comparison_table_Manuscript.xlsx")
print(f"  - Figures: {figures_dir}")
print(f"  - Data: {output_dir}")

plt.show()