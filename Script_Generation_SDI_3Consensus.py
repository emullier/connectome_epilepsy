

''' Manuscript comparison script merging codes 02 and 07
This script generates a comprehensive comparison table of SDI values for:
- HC (Healthy Controls - Geneva dataset)
- EP (Epilepsy patients - Geneva dataset) 
- IND (Independent 27 healthy controls consensus)
7 left TLE (age 34.7 ± 10.1 years, 3 females) and 9 right TLE (age 34.8 ± 8.4 years, 6 females)

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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
project_root = os.path.dirname(os.path.abspath(__file__))

scale = 2
example_dir = os.path.join(project_root, "DATA/EEG")
infoGVA_path = os.path.join(project_root, 'DEMOGRAPHIC/info_dsi_multishell_merged_csv.csv')

# Create output directories
output_dir = os.path.join(project_root, 'OUTPUT/')
figures_dir = os.path.join(project_root, 'FIGURES/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# =====================================================
# PART 1: Load HC and EP data from Geneva datasets 
# =====================================================
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
        print(f"\n Processing {group} {lateralization}...")
        
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
        
      
        # Surrogate part 
        nbSurr = 100
        surr_path = os.path.join(project_root, f'OUTPUT/SDI_surr_{metric}_{group}_{dwi}_{lateralization}.npy')
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
        thr = 0
        plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI'], 
                                    scale, figures_dir, vmin=-2, vmax=2, 
                                    label=f'Fig2_SDImean_{group}_{lateralization}')
        thr = 5
        plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), 
                                    scale, figures_dir, vmin=-2, vmax=2, 
                                    label=f'Fig2_SDImean_thr{thr}_{group}_{lateralization}')
        
        # Store surr_thresh for later use
        surr_thresh_storage[f"{group}_{lateralization}"] = surr_thresh


# ===============================================
# PART 2: Load IND (27 healthy controls) data 
# ===============================================
print("\n" + "="*80)
print("PART 2: Processing IND consensus matrix (Independent 27 HC dataset)")
print("="*80)

for lateralization in ls_lateralization:
    print(f"\nProcessing IND {lateralization}...")
    
    # Load the data from matMetric_SCHZ_CTRL.npy
    matMetric = np.load(os.path.join(project_root, "DATA/SC/matMetric_SCHZ_CTRL.npy"))
    
    # Use only first 10 subjects (instead of all 27)
    matMetric = matMetric[:10, :, :]
    print(f"Using first 10 subjects from IND dataset (shape: {matMetric.shape})")
    
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
    thr = 0
    plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI'], 
                                scale, figures_dir, vmin=-2, vmax=2, 
                                label=f'Fig2_SDImean_{group}_IND_{lateralization}')
    thr = 5
    plot_rois_pyvista_noaxes(surr_thresh[thr]['mean_SDI']*np.abs(surr_thresh[thr]['SDI_sig']), 
                                scale, figures_dir, vmin=-2, vmax=2, 
                                label=f'Fig2_SDImean_thr{thr}_IND_{lateralization}')




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
    "IND_RT": surr_thresh_storage["IND_RT"]
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
# PART 3b: Fig2bis - Summary brain plot (significant ROIs across 3 SC)
# ============================================================================
print("\n" + "="*80)
print("PART 3b: Creating Fig2bis - Summary brain plot with consensus activation")
print("="*80)

# Create summary values for each ROI indicating how many SC methods have it as significant
# For each lateralization, count activations across HC, EP, and IND
for lateralization in ["LT", "RT"]:
    print(f"\nGenerating Fig2bis for {lateralization}...")
    
    # Initialize summary vector (118 ROIs)
    summary_vector = np.zeros(118)
    
    # For each ROI, count how many of the three SC methods (HC, EP, IND) have it as significant at thr=5
    for roi_idx in range(118):
        count = 0
        for group in ["HC", "EP", "IND"]:
            key = f"{group}_{lateralization}"
            # Check if this ROI is significant in this group at threshold 5
            if surr_thresh_storage[key][5]['SDI_sig'][roi_idx] != 0:
                count += 1
        summary_vector[roi_idx] = count
    
    # Plot the summary brain with values 0-3 indicating number of active SC
    # Only show ROIs that are active in at least one SC (values 1, 2, or 3)
    summary_vector_masked = np.where(summary_vector > 0, summary_vector, np.nan)
    
    # Create brain plot
    plot_rois_pyvista_noaxes(summary_vector_masked, 
                            scale, figures_dir, vmin=1, vmax=3, 
                            cmap='YlOrRd',
                            label=f'Fig2bis_summary_activation_{lateralization}')
    
    print(f"  - ROIs active in 1 SC: {np.sum(summary_vector == 1)}")
    print(f"  - ROIs active in 2 SCs: {np.sum(summary_vector == 2)}")
    print(f"  - ROIs active in all 3 SCs: {np.sum(summary_vector == 3)}")

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

pairs = [
    ("HC", "EP", "HC", "TLE"),
    ("HC", "IND", "HC", "IND"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle("Mean SDI correlations (Left & Right IEDs)", fontsize=15, fontweight='bold', y=1.02)

for row_idx, side in enumerate(["LT", "RT"]):
    side_label = "Left IED" if side == "LT" else "Right IED"
    side_color = '#1f77b4' if side == "LT" else '#2ca02c'
    for ax, (g1_key, g2_key, g1_label, g2_label) in zip(axes[row_idx], pairs):
        x = get_all_mean(g1_key, side)
        y = get_all_mean(g2_key, side)
        if x.size == 0 or y.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=11)
            ax.axis('off')
            continue
        ax.scatter(x, y, color=side_color, edgecolor='k', alpha=0.7, s=60)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([np.min(x), np.max(x)])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, linestyle='--', linewidth=1.8, color=side_color, alpha=0.9)
        r, p = pearsonr(x, y)
        ax.set_title(f"{g1_label} vs {g2_label} ({side_label}, r={r:.2f}, p={p:.3f})", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"Mean SDI {g1_label}", fontsize=10)
        ax.set_ylabel(f"Mean SDI {g2_label}", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.text(0.05, 0.92, f"r = {r:.2f}\np = {p:.3f}", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

corr_path = os.path.join(figures_dir, 'FigS3_corr_mean_SDI_all.png')
plt.savefig(corr_path, dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.basename(corr_path)}")

# Figure 1: mixed LT/RT triangular heatmaps
groups_heatmap = ["HC", "EP", "IND"]
group_labels_heatmap = [r"SC$_{HC}$", r"SC$_{TLE}$", r"SC$_{IND}$"]

matrices_by_side = {}
for side in ["LT", "RT"]:
    sdi_vectors = [get_all_mean(group, side) for group in groups_heatmap]
    n_groups = len(sdi_vectors)
    r_mat = np.eye(n_groups)
    p_mat = np.zeros((n_groups, n_groups))

    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                r_mat[i, j] = 1.0
                p_mat[i, j] = 0.0
            else:
                r_tmp, p_tmp = pearsonr(sdi_vectors[i], sdi_vectors[j])
                r_mat[i, j] = r_tmp
                p_mat[i, j] = p_tmp

    matrices_by_side[side] = {
        'r_df': pd.DataFrame(r_mat, index=group_labels_heatmap, columns=group_labels_heatmap),
        'p_df': pd.DataFrame(p_mat, index=group_labels_heatmap, columns=group_labels_heatmap),
        'r_mat': r_mat,
        'p_mat': p_mat,
    }

purple_cmap = sns.light_palette("purple", as_cmap=True)
fig_hm, axes_hm = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
fig_hm.suptitle("SDI relationships across structural connectomes", fontsize=14, fontweight='bold')

mask_lower = np.triu(np.ones_like(matrices_by_side['LT']['r_df'], dtype=bool), k=0)
mask_upper = np.tril(np.ones_like(matrices_by_side['LT']['r_df'], dtype=bool), k=0)

# Left subplot: correlations
sns.heatmap(
    matrices_by_side['LT']['r_df'],
    mask=mask_lower,
    ax=axes_hm[0],
    cmap='Blues',
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar=False,
    annot=False,
)
sns.heatmap(
    matrices_by_side['RT']['r_df'],
    mask=mask_upper,
    ax=axes_hm[0],
    cmap='Greens',
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar=False,
    annot=False,
)

# Right subplot: p-values
sns.heatmap(
    matrices_by_side['LT']['p_df'],
    mask=mask_lower,
    ax=axes_hm[1],
    cmap=purple_cmap,
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar=False,
    annot=False,
)
sns.heatmap(
    matrices_by_side['RT']['p_df'],
    mask=mask_upper,
    ax=axes_hm[1],
    cmap=purple_cmap,
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar=False,
    annot=False,
)

for ax in axes_hm:
    for d in range(n_groups):
        ax.add_patch(
            plt.Rectangle((d, d), 1, 1, facecolor='lightgray', edgecolor='white', linewidth=0.5, zorder=3)
        )
        ax.text(d + 0.5, d + 0.5, '—', ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

for i in range(n_groups):
    for j in range(n_groups):
        if i > j:
            p_lt = matrices_by_side['LT']['p_mat'][i, j]
            stars_lt = '***' if p_lt < 0.001 else '**' if p_lt < 0.01 else '*' if p_lt < 0.05 else ''
            axes_hm[0].text(j + 0.5, i + 0.5, f"{matrices_by_side['LT']['r_mat'][i, j]:.2f}{stars_lt}",
                            ha='center', va='center', fontsize=14,
                            fontweight='bold' if p_lt < 0.05 else 'normal')
            axes_hm[1].text(j + 0.5, i + 0.5, f"{p_lt:.1e}{stars_lt}",
                            ha='center', va='center', fontsize=14,
                            fontweight='bold' if p_lt < 0.05 else 'normal', color='black')
        elif i < j:
            p_rt = matrices_by_side['RT']['p_mat'][i, j]
            stars_rt = '***' if p_rt < 0.001 else '**' if p_rt < 0.01 else '*' if p_rt < 0.05 else ''
            axes_hm[0].text(j + 0.5, i + 0.5, f"{matrices_by_side['RT']['r_mat'][i, j]:.2f}{stars_rt}",
                            ha='center', va='center', fontsize=14,
                            fontweight='bold' if p_rt < 0.05 else 'normal')
            axes_hm[1].text(j + 0.5, i + 0.5, f"{p_rt:.1e}{stars_rt}",
                            ha='center', va='center', fontsize=14,
                            fontweight='bold' if p_rt < 0.05 else 'normal', color='black')

axes_hm[0].set_title('Pearson r: lower Left IED, upper Right IED', fontsize=12, fontweight='bold')
axes_hm[1].set_title('p-value: lower Left IED, upper Right IED', fontsize=12, fontweight='bold')

for ax in axes_hm:
    ax.tick_params(axis='x', rotation=30, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)

sm_corr_lt = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
sm_corr_lt.set_array([])
sm_pval = plt.cm.ScalarMappable(cmap=purple_cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm_pval.set_array([])

cbar_corr_lt = fig_hm.colorbar(sm_corr_lt, ax=axes_hm[0], location='left', fraction=0.06, pad=0.04)
cbar_corr_lt.set_label('Pearson r Left IED', fontsize=10, fontweight='bold')
cbar_corr_lt.ax.tick_params(labelsize=9)

cbar_pval = fig_hm.colorbar(sm_pval, ax=axes_hm[1], location='right', fraction=0.06, pad=0.04)
cbar_pval.set_label('p-value Left/Right IED', fontsize=10, fontweight='bold')
cbar_pval.ax.tick_params(labelsize=9)

heatmap_path = os.path.join(figures_dir, 'Fig1_heatmap_SDI_correlation_SC.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.basename(heatmap_path)}")

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
ax[1].set_xticklabels(['SC HC', 'SC EP', 'SC IND'], fontsize=11, fontweight='bold')
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
    
    # Add asterisks for significant p-values
    if pval < 0.001:
        sig_text = "***"
    elif pval < 0.01:
        sig_text = "**"
    elif pval < 0.05:
        sig_text = "*"
    else:
        sig_text = "ns"
    
    ax[1].text((x1 + x2) / 2, y + 0.025 * y_base, f"p = {pval:.3f} {sig_text}", ha='center', va='bottom', fontsize=9)
    print(f"Cutoff comparison {g1} vs {g2}: U={stat:.2f}, p={pval:.4f}")

plt.savefig(os.path.join(figures_dir, 'FigS2_cutoff_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved: cutoff_comparison.png")
#plt.close()

# Plot number of significant ROIs
nbROIs_HC_RT = np.load(os.path.join(output_dir, 'nbROIs_sig_HC_RT.npy'))
nbROIs_HC_LT = np.load(os.path.join(output_dir, 'nbROIs_sig_HC_LT.npy'))
nbROIs_EP_RT = np.load(os.path.join(output_dir, 'nbROIs_sig_EP_RT.npy'))
nbROIs_EP_LT = np.load(os.path.join(output_dir, 'nbROIs_sig_EP_LT.npy'))
nbROIs_IND_RT = np.load(os.path.join(output_dir, 'nbROIs_sig_IND_RT.npy'))
nbROIs_IND_LT = np.load(os.path.join(output_dir, 'nbROIs_sig_IND_LT.npy'))

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

# LT subplot
ls_nbROIs_LT = [nbROIs_HC_LT, nbROIs_EP_LT, nbROIs_IND_LT]
ls_labels_LT = ["HC", "TLE", "IND"]
ls_colors_LT = ['#1f77b4', '#2ca02c', '#9467bd']
ls_markers_LT = ['o', 's', '^']

for i, (nbROIs, label, color, marker) in enumerate(zip(ls_nbROIs_LT, ls_labels_LT, ls_colors_LT, ls_markers_LT)):
    axes[0].plot(np.arange(len(nbROIs)), np.array(nbROIs), marker=marker, linewidth=2.5, markersize=6, 
           color=color, label=label)

axes[0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
axes[0].set_ylabel('# ROIs with significant SDI', fontsize=12, fontweight='bold')
axes[0].set_xticks(np.arange(0, len(nbROIs_HC_LT)))
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_title('Number of Significant SDI ROIs per Threshold (Left IED)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11, loc='upper right', framealpha=0.9)
axes[0].tick_params(labelsize=10)

# RT subplot
ls_nbROIs_RT = [nbROIs_HC_RT, nbROIs_EP_RT, nbROIs_IND_RT]
ls_labels_RT = ["HC", "TLE", "IND"]
ls_colors_RT = ['#1f77b4', '#2ca02c', '#9467bd']
ls_markers_RT = ['o', 's', '^']

for i, (nbROIs, label, color, marker) in enumerate(zip(ls_nbROIs_RT, ls_labels_RT, ls_colors_RT, ls_markers_RT)):
    axes[1].plot(np.arange(len(nbROIs)), np.array(nbROIs), marker=marker, linewidth=2.5, markersize=6, 
           color=color, label=label)

axes[1].set_xlabel('Threshold', fontsize=12, fontweight='bold')
axes[1].set_ylabel('# ROIs with significant SDI', fontsize=12, fontweight='bold')
axes[1].set_xticks(np.arange(0, len(nbROIs_HC_RT)))
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_title('Number of Significant SDI ROIs per Threshold (Right IED)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11, loc='upper right', framealpha=0.9)
axes[1].tick_params(labelsize=10)

plt.savefig(os.path.join(figures_dir, 'FigS1_nbROIs_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved: nbROIs_comparison.png")
#plt.close()

print("\n" + "="*80)
print("Manuscript comparison analysis complete!")
print("="*80)
print("\nGenerated outputs:")
print("  - Comparison table: SDI_comparison_table_Manuscript.xlsx")
print(f"  - Figures: {figures_dir}")
print(f"  - Data: {output_dir}")

# plt.show()  # Removed to prevent live plotting