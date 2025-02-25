
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
import scipy
import community as community_louvain
import lib.func_ML as ML
from sklearn.decomposition import PCA
import lib.func_reading as reading
import lib.func_utils as utils
from neuroHarmonize import harmonizationLearn
import matplotlib.lines as mlines

bids_dir = r"F:\Bonn_dataset_testCMP"
cmp_dir = os.path.join(bids_dir, 'derivatives/cmp-v3.2.0')
info_path = r"C:\Users\emeli\Documents\CHUV\TEST_RETEST_DSI_microstructure\connectome_epilepsy\demographic\BonnDTIPre-OpData_Imaging.csv"

df_info_Bonn = pd.read_csv(info_path)
scale = 2
metric = "number_of_fibers"
mask = 0

for s,sub in enumerate(df_info_Bonn['SubjID']):
    tmp = os.listdir(os.path.join(bids_dir, sub))
    ls_ses = []
    for t,tm in enumerate(tmp):
        if tm.startswith('ses-'):
            ls_ses.append(tm)
    ses = ls_ses[0]
    SC_path = os.path.join(cmp_dir, sub, ses, 'dwi', '%s_%s_atlas-L2018_res-scale%d_conndata-network_connectivity.mat'%(sub, ses, scale))
    SC = sio.loadmat(SC_path)['sc'][metric][0][0]
    if s==0:
        SC_subs = np.zeros((len(SC), len(SC), len(df_info_Bonn['SubjID'])))
    SC_subs[:,:,s] = SC
    
    ### Create mask to keep right hemisphere, left hemisphere and interhemispheric connections only
    if s==0:
        maskRH = np.zeros(np.shape(SC)); maskLH = np.zeros(np.shape(SC)); maskInter = np.zeros(np.shape(SC))
        right_hemisphere_indices = np.arange(0, len(SC) // 2) 
        maskRH[np.ix_(right_hemisphere_indices, right_hemisphere_indices)] = 1
        left_hemisphere_indices = np.arange(len(SC) // 2, len(SC))
        maskLH[np.ix_(left_hemisphere_indices, left_hemisphere_indices)] = 1   
        maskInter[np.ix_(left_hemisphere_indices, right_hemisphere_indices)]=1
        maskInter[np.ix_(right_hemisphere_indices, left_hemisphere_indices)]=1
    
    if mask==1:
        maskChosen = maskInter; masklabel = 'maskInter'
        SC_subs[:,:,s] = SC*maskChosen
        
    
### Graph measures - Change here is mask has been used
if mask==1:
    graph_metrics_path = './output/bonn_dataset/graph_metrics_%s.csv'% masklabel
else:
    graph_metrics_path = './output/bonn_dataset/graph_metrics.csv'
if os.path.exists(graph_metrics_path):
    df_metrics = pd.read_csv(graph_metrics_path)
else:    
    graph_metrics = []
    for s,sub in enumerate(df_info_Bonn['SubjID']):
        matrix = SC_subs[:,:,s]
        np.fill_diagonal(matrix, 0)
        [r,p] = scipy.stats.pearsonr(matrix[np.where(maskRH==1)], matrix[np.where(maskLH==1)])
        G = nx.from_numpy_array(matrix)
        partition = community_louvain.best_partition(G)
        # Compute Graph Measures
        metrics = {
            "Density": nx.density(G),
            "Clustering Coefficient": nx.average_clustering(G),
            "Global Efficiency": nx.global_efficiency(G),
            "Degree Centrality": nx.degree_centrality(G),
            "Betweenness Centrality": nx.betweenness_centrality(G),
            "Correlation RH-LH": r,
            "Modularity": community_louvain.modularity(partition, G)
            #"Eigenvector Centrality": nx.eigenvector_centrality(G)
        }
        graph_metrics.append(metrics)

    df_metrics = pd.DataFrame(graph_metrics)
    df_metrics.to_csv(graph_metrics_path, index=False)
    

# Select only the desired columns
df_filtered = df_metrics[["Density", "Clustering Coefficient", "Global Efficiency",  "Modularity",  "Correlation RH-LH"]]
# Add the secondary diagnostic column
df_filtered['SDx'] = df_info_Bonn['SDx']
label_mapping = {3: "LTLE - lesion", 4: "RTLE - lesion", 5: "LTLE - MRI-", 9: "BILATERAL"}
df_filtered["SDxMapped"] = df_filtered["SDx"].map(label_mapping).fillna(df_filtered["SDx"])
# Convert DataFrame to long format
df_long = df_filtered.melt(id_vars=["SDx", "SDxMapped"], var_name="Metric", value_name="Value")

# Plot boxplot according to secondary Dx
plt.figure(figsize=(10, 6))
sns.boxplot(x="Metric", y="Value", hue="SDxMapped", data=df_long, palette="deep", boxprops=dict(alpha=0.6), fliersize=0)
sns.stripplot(x="Metric", y="Value", hue="SDxMapped", data=df_long, palette="deep", jitter=True, dodge=True, marker="o")
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handle for i, handle in enumerate(handles) if 'Line2D' in str(type(handle))]
labels = [labels[i] for i, handle in enumerate(handles) if 'Line2D' in str(type(handle))]
plt.legend(handles=handles, labels=labels, title="Diagnosis", fontsize=10)
plt.xlabel("Graph Metric", fontsize=12)
plt.ylabel("Value", fontsize=12); plt.xticks(rotation=0)
plt.title("Graph Network Metrics Across Secondary Diagnoses", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
#plt.show(block=True)


### Statistical Testing
metrics = df_long["Metric"].unique()
# Loop through each metric to apply the statistical test
for metric in metrics:
    print(f"\n--- SDx - Testing for {metric} ---")
    metric_data = df_long[df_long["Metric"] == metric]
    # Kruskal-Wallis H Test for non-parametric data (if you assume non-normal distributions)
    grouped_data = [metric_data[metric_data["SDxMapped"] == diagnosis]["Value"] for diagnosis in metric_data["SDxMapped"].unique()]
    stat, p_value = kruskal(*grouped_data)
    print(f"Kruskal-Wallis p-value: {p_value}")
    # If the p-value from Kruskal-Wallis is significant, perform pairwise Mann-Whitney U tests
    if p_value < 0.05:
        print(f"Pairwise comparisons for {metric}:")
        for i in range(len(grouped_data)):
            for j in range(i+1, len(grouped_data)):
                stat, p_value = mannwhitneyu(grouped_data[i], grouped_data[j])
                print(f"Comparison between {metric_data['SDxMapped'].unique()[i]} and {metric_data['SDxMapped'].unique()[j]}: p-value = {p_value}")

# Boxplot according to Engel classification
df_filtered = df_filtered.drop(columns=["SDx","SDxMapped"])
df_filtered['ENGEL'] = df_info_Bonn['ENGEL']
df_long = df_filtered.melt(id_vars=["ENGEL"], var_name="Metric", value_name="Value")
plt.figure(figsize=(10, 6))
sns.boxplot(x="Metric", y="Value", hue="ENGEL", data=df_long, palette="flare", boxprops=dict(alpha=0.6), fliersize=0)
sns.stripplot(x="Metric", y="Value", hue="ENGEL", data=df_long, palette="flare", jitter=True, dodge=True, marker="o")
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handle for i, handle in enumerate(handles) if 'Line2D' in str(type(handle))]
labels = [labels[i] for i, handle in enumerate(handles) if 'Line2D' in str(type(handle))]
plt.legend(handles=handles, labels=labels, title="Engel Class", fontsize=10)
plt.xlabel("Graph Metric", fontsize=12)
plt.ylabel("Value", fontsize=12); plt.xticks(rotation=0)
plt.title("Graph Network Metrics Across Engel Classes", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
#plt.show(block=True)

### Statistical Testing
metrics = df_long["Metric"].unique()
# Loop through each metric to apply the statistical test
for metric in metrics:
    print(f"\n--- ENGEL - Testing for {metric} ---")
    metric_data = df_long[df_long["Metric"] == metric]
    # Kruskal-Wallis H Test for non-parametric data (if you assume non-normal distributions)
    grouped_data = [metric_data[metric_data["ENGEL"] == diagnosis]["Value"] for diagnosis in metric_data["ENGEL"].unique()]
    stat, p_value = kruskal(*grouped_data)
    print(f"Kruskal-Wallis p-value: {p_value}")
    # If the p-value from Kruskal-Wallis is significant, perform pairwise Mann-Whitney U tests
    if p_value < 0.05:
        print(f"Pairwise comparisons for {metric}:")
        for i in range(len(grouped_data)):
            for j in range(i+1, len(grouped_data)):
                stat, p_value = mannwhitneyu(grouped_data[i], grouped_data[j])
                print(f"Comparison between {metric_data['ENGEL'].unique()[i]} and {metric_data['ENGEL'].unique()[j]}: p-value = {p_value}")


HC_multi = np.load(r"./data/matMetric_HC_multishell.npy", allow_pickle=True)
HC_DSI = np.load(r"./data/matMetric_HC_DSI.npy", allow_pickle=True)
EP_multi = np.load(r"./data/matMetric_EP_multishell.npy", allow_pickle=True)
#EP_DSI = np.load(r"./data/matMetric_EP_DSI.npy", allow_pickle=True)
datasets = [SC_subs, HC_multi, HC_DSI, EP_multi]
datasets_labels = ["SC_subs", "HC_multi", "HC_DSI", "EP_multi"]
nbCombSubs = np.shape(SC_subs)[2] + np.shape(HC_multi)[2] + np.shape(HC_DSI)[2] + np.shape(EP_multi)[2] 
idxs2keep = np.concatenate((np.arange(0,57), np.arange(85,142)))
SC_subs_cut = np.copy(SC_subs)
SC_subs_cut = SC_subs_cut[idxs2keep,:,:]; SC_subs_cut = SC_subs_cut[:,idxs2keep,:]
SC_CombSubs = np.concatenate((SC_subs_cut, HC_multi, HC_DSI, EP_multi), axis=2)

df_CombSubs = pd.DataFrame()
ls_dat = []
for d,dat in enumerate(datasets):
    nb = np.shape(dat)[2]
    for n in np.arange(nb):
        ls_dat.append(datasets_labels[d])
df_CombSubs['dataset'] = ls_dat

graph_metrics_full_path = './output/bonn_dataset/graph_metrics_full.csv'
if os.path.exists(graph_metrics_full_path):
    df_metrics_full = pd.read_csv(graph_metrics_full_path)
else:    
    graph_metrics_full = []
    for s in np.arange(len(ls_dat)):
        matrix = SC_CombSubs[:,:,s]
        np.fill_diagonal(matrix, 0)
        G = nx.from_numpy_array(matrix)
        partition = community_louvain.best_partition(G)
        # Compute Graph Measures
        metrics = {
            "Density": nx.density(G),
            "Clustering Coefficient": nx.average_clustering(G),
            "Global Efficiency": nx.global_efficiency(G),
            "Degree Centrality": nx.degree_centrality(G),
            "Betweenness Centrality": nx.betweenness_centrality(G),
            "Modularity": community_louvain.modularity(partition, G)
        }
        graph_metrics_full.append(metrics)
    df_metrics_full = pd.DataFrame(graph_metrics_full)
    df_metrics_full.to_csv(graph_metrics_full_path, index=False)
    
df_filtered = df_metrics_full[["Density", "Clustering Coefficient", "Global Efficiency",  "Modularity"]]
df_filtered['dataset'] = df_CombSubs['dataset']
df_long = df_filtered.melt(id_vars=["dataset"], var_name="Metric", value_name="Value")
plt.figure(figsize=(10, 6))
sns.boxplot(x="Metric", y="Value", hue="dataset", data=df_long, palette="deep", boxprops=dict(alpha=0.6), fliersize=0)
sns.stripplot(x="Metric", y="Value", hue="dataset", data=df_long, palette="deep", jitter=True, dodge=True, marker="o")
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handle for i, handle in enumerate(handles) if 'Line2D' in str(type(handle))]
labels = [labels[i] for i, handle in enumerate(handles) if 'Line2D' in str(type(handle))]
plt.legend(handles=handles, labels=labels, title="Dataset", fontsize=10)
plt.xlabel("Graph Metric", fontsize=12)
plt.ylabel("Value", fontsize=12); plt.xticks(rotation=0)
plt.title("Graph Network Metrics Across Datasets", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
#plt.show(block=True)

### Statistical Testing
metrics = df_long["Metric"].unique()
# Loop through each metric to apply the statistical test
for metric in metrics:
    print(f"\n--- DATASET - Testing for {metric} ---")
    metric_data = df_long[df_long["Metric"] == metric]
    # Kruskal-Wallis H Test for non-parametric data (if you assume non-normal distributions)
    grouped_data = [metric_data[metric_data["dataset"] == diagnosis]["Value"] for diagnosis in metric_data["dataset"].unique()]
    stat, p_value = kruskal(*grouped_data)
    print(f"Kruskal-Wallis p-value: {p_value}")
    # If the p-value from Kruskal-Wallis is significant, perform pairwise Mann-Whitney U tests
    if p_value < 0.05:
        print(f"Pairwise comparisons for {metric}:")
        for i in range(len(grouped_data)):
            for j in range(i+1, len(grouped_data)):
                stat, p_value = mannwhitneyu(grouped_data[i], grouped_data[j])
                print(f"Comparison between {metric_data['dataset'].unique()[i]} and {metric_data['dataset'].unique()[j]}: p-value = {p_value}")


### PCA
#fig, axs = plt.subplots()
#X_vec = np.zeros((np.shape(SC_CombSubs)[0]*np.shape(SC_CombSubs)[0], np.shape(SC_CombSubs)[2]))
#for s in np.arange(np.shape(SC_CombSubs)[2]):
#    X_vec[:,s] = SC_CombSubs[:,:,s].flatten()
#pca = PCA(n_components=2)
#X_pca = pca.fit_transform(np.transpose(X_vec))
#unique_datasets = df_CombSubs['dataset'].unique()
#palette = sns.color_palette("husl", len(unique_datasets))
#dataset_colors = {dataset: color for dataset, color in zip(unique_datasets, palette)}
#df_CombSubs['color'] = df_CombSubs['dataset'].map(dataset_colors)
#scatter = axs.scatter(X_pca[:,0], X_pca[:,1], 50, c=df_CombSubs['color'])
#axs.set_title('Dataset'), axs.set_xlabel('PCA Component 1'), axs.set_ylabel('PCA Component 2') 
#plt.show()

config_path = 'config.json'; config = reading.check_config_file(config_path)
df_info = reading.read_info(config['Parameters']['info_path'])
filters = utils.compare_pdkeys_list(df_info, config['Parameters']['filters'])
df, ls_subs = reading.filtered_dataframe(df_info, filters, config)
df_HC_multi = df.iloc[np.where((df['dwi']=='multishell')*(df['group']=='HC'))[0]]
df_HC_dsi = df.iloc[np.where((df['dwi']=='dsi')*(df['group']=='HC'))[0]]
df_EP_multi = df.iloc[np.where((df['dwi']=='multishell')*(df['group']=='EP'))[0]]

df_Bonn_Comb = pd.DataFrame()
age = np.concatenate((df_info_Bonn['Age'], df_HC_multi['age'], df_HC_dsi['age'], df_EP_multi['age']))
dx = np.concatenate((df_info_Bonn['SDx'], df_HC_multi['group'], df_HC_dsi['group'], df_EP_multi['Lateralization']))
subs = np.concatenate((df_info_Bonn['SubjID'], df_HC_multi['sub'], df_HC_dsi['sub'], df_EP_multi['sub']))
dx[np.where(dx==3)] = 'LTLE'; dx[np.where(dx==5)] = 'LTLE';
dx[np.where(dx==4)] = 'RTLE'; dx[np.where(dx==6)] = 'RTLE';
dx[np.where(dx==9)] = 'BIL'; 
site = df_CombSubs['dataset'] 
df_Bonn_Comb['AGE'] = age
df_Bonn_Comb['DX'] = dx
df_Bonn_Comb['SUB'] = subs
df_Bonn_Comb['SITE'] = site
df_Bonn_Comb = df_Bonn_Comb.loc[df_Bonn_Comb["SUB"] != "struct_data"]
df_Bonn_Comb = df_Bonn_Comb.loc[df_Bonn_Comb["SUB"] != "Individual_Connectomes"]

### NEUROCOMBAT
### Create a dataframe for input of NeuroCombat - One column = lower triangle matrix of 1 subject
df_NeuroCombat = pd.DataFrame()
for s,sub in enumerate(df_Bonn_Comb['SUB']):
    print(s)
    Mat = SC_CombSubs[:,:,s]
    lower_triangle_indices = np.tril_indices(Mat.shape[0], k=-1)
    lower_triangle_values = Mat[lower_triangle_indices]
    df_NeuroCombat[s] = lower_triangle_values

### Create table of all numeric covariates
covars = df_Bonn_Comb[["SITE", "AGE"]]
    
### Convert the dataframe into an array
my_data = np.array(df_NeuroCombat)   
my_data = np.transpose(my_data) 
my_data[np.where(my_data==0)]=1e-8

print(np.shape(my_data))
print(np.shape(covars))   

# run harmonization and store the adjusted data
my_model, my_data_adj = harmonizationLearn(my_data, covars)



 # Boxplot values before/after NeuroCombat
boxplot_data = []; labels = []; colors = []  
for s,sub in enumerate(df_Bonn_Comb['SUB']):
    boxplot_data.append(my_data[:, s])        # Original
    boxplot_data.append(my_data_adj[:, s])    # Adjusted
    labels.append('%s'%sub); labels.append('%s'%sub)
    colors.append('#333333'); colors.append('#D4AF37')  ### charcoal grey, crisp gold   
plt.figure(figsize=(12, 6))
bp = plt.boxplot(boxplot_data, patch_artist=True)
for i, box in enumerate(bp['boxes']):
    box.set_facecolor(colors[i])  
group_labels = [labels[i] for i in range(0, len(labels), 2)]
xticks = np.arange(1.5, len(group_labels) * 2 + 1, 2)
plt.xticks(ticks=xticks, labels=group_labels, rotation=90, fontsize=8)
plt.ylabel("Number of fibers"); plt.title("Matrices before/after NeuroCombat")
plt.grid(axis='y', linestyle='--', alpha=0.7)
orig_legend = mlines.Line2D([], [], color='#333333', label='Original', linewidth=6)
harm_legend = mlines.Line2D([], [], color='#D4AF37', label='Harmonized', linewidth=6)
plt.legend(handles=[orig_legend, harm_legend], loc='upper right')


## Building back SC matrices after harmonization
reconstructed_Mat = np.zeros_like(SC_CombSubs)
#print(np.shape(reconstructed_Mat))
for s,sub in enumerate(df_Bonn_Comb['SUB']):
#    print(np.shape(np.array(df_Comb[sub])))
#    print(np.shape(reconstructed_Mat[lower_triangle_indices,s]))
    full_matrix = np.zeros((np.shape(reconstructed_Mat)[0], np.shape(reconstructed_Mat)[0]))
    lower_triangle_indices = np.tril_indices(np.shape(reconstructed_Mat)[0], k=-1)
    full_matrix[lower_triangle_indices] = my_data_adj[s,:]
    full_matrix[np.where(full_matrix<1e-7)] = 0
    #avg = np.mean(SC_CombSubs, axis=2)
    #binMask = np.copy(avg)
    #binMask[np.where(avg<1e-7)]=0; binMask[np.where(binMask>0)] = 1
    reconstructed_Mat[:,:,s] = (full_matrix + full_matrix.T)*binMask
    #fig,axs = plt.subplots()
    #plt.imshow(reconstructed_Mat[:,:,s]); plt.colorbar()
    #plt.show()
    

graph_metrics_full_rec_path = './output/bonn_dataset/graph_metrics_full_reconstructed.csv'
if os.path.exists(graph_metrics_full_rec_path):
    df_metrics_full_rec = pd.read_csv(graph_metrics_full_rec_path)
else:    
    graph_metrics_full_rec = []
    for s in np.arange(len(ls_dat)):
        matrix = reconstructed_Mat[:,:,s]
        np.fill_diagonal(matrix, 0)
        G = nx.from_numpy_array(matrix)
        partition = community_louvain.best_partition(G)
        # Compute Graph Measures
        metrics = {
            "Density": nx.density(G),
            "Clustering Coefficient": nx.average_clustering(G),
            "Global Efficiency": nx.global_efficiency(G),
            "Degree Centrality": nx.degree_centrality(G),
            "Betweenness Centrality": nx.betweenness_centrality(G),
            #"Modularity": community_louvain.modularity(partition, G)
        }
        graph_metrics_full_rec.append(metrics)
    df_metrics_full_rec = pd.DataFrame(graph_metrics_full_rec)
    df_metrics_full_rec.to_csv(graph_metrics_full_rec_path, index=False)
    
df_filtered = df_metrics_full_rec[["Density", "Clustering Coefficient", "Global Efficiency"]]
df_filtered['dataset'] = df_CombSubs['dataset']
df_long = df_filtered.melt(id_vars=["dataset"], var_name="Metric", value_name="Value")
plt.figure(figsize=(10, 6))
sns.boxplot(x="Metric", y="Value", hue="dataset", data=df_long, palette="deep", boxprops=dict(alpha=0.6), fliersize=0)
sns.stripplot(x="Metric", y="Value", hue="dataset", data=df_long, palette="deep", jitter=True, dodge=True, marker="o")
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handle for i, handle in enumerate(handles) if 'Line2D' in str(type(handle))]
labels = [labels[i] for i, handle in enumerate(handles) if 'Line2D' in str(type(handle))]
plt.legend(handles=handles, labels=labels, title="Dataset", fontsize=10)
plt.xlabel("Graph Metric", fontsize=12)
plt.ylabel("Value", fontsize=12); plt.xticks(rotation=0)
plt.title("Graph Network Metrics Across Harmonized Datasets", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()