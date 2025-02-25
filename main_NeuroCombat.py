
'''' Harmonization of Structural Connectivity matrices with NeuroCombat
----------------------------------------------------------------------------
February 2025 
Emeline Mullier 
Geneva University Hospital'''

# import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import lib.func_reading as reading
import lib.func_utils as utils
from neuroHarmonize import harmonizationLearn

# Load the matrices to harmonize and create the dataframe
# Data of interest to be chosen in the config_path (processing field): "shore":"cmp-3.1.0_shore"
config_path = 'config.json'; config = reading.check_config_file(config_path)
df_info = reading.read_info(config['Parameters']['info_path'])
filters = utils.compare_pdkeys_list(df_info, config['Parameters']['filters'])
df, ls_subs = reading.filtered_dataframe(df_info, filters, config)
procs = list(config["Parameters"]["processing"].keys())
MatMat = {}; EucMat = {};
for p, proc in enumerate(procs):
    idxs_tmp = np.where((df[proc] == 1) | (df[proc] == '1'))[0]
    df_tmp = df.iloc[idxs_tmp]
    tmp_path = os.path.join(config["Parameters"]["data_dir"], config["Parameters"]["processing"][proc])
    MatMat[proc], EucMat[proc], df_info = reading.load_matrices(df_tmp, tmp_path, config['Parameters']['scale'], config['Parameters']['metric'])

### Create a dataframe for input of NeuroCombat - One column = lower triangle matrix of 1 subject
proc = procs[0]
df_Comb = pd.DataFrame()
for s,sub in enumerate(df_info['sub']):
    Mat = MatMat[proc][:,:,s]
    #tri = np.tril(Mat, k=-1)
    lower_triangle_indices = np.tril_indices(Mat.shape[0], k=-1)
    lower_triangle_values = Mat[lower_triangle_indices]
    df_Comb[sub] = lower_triangle_values

### Create table of all numeric covariates
covars = pd.DataFrame()
site = np.array(df_info['dwi'])
site[np.where(site=='dsi')] = 'SITE_A'
site[np.where(site=='multishell')] = 'SITE_B'
age = np.array(df_info['age'])
sex_m = np.array(df_info['gender'])
sex_m[np.where(sex_m=='M')] = 1
sex_m[np.where(sex_m=='F')] = 0

covars['SITE'] = site; covars['AGE'] = age; covars['SEX_M'] = sex_m

### Convert the dataframe into an array
my_data = np.array(df_Comb)   
my_data = np.transpose(my_data) 
my_data[np.where(my_data==0)]=1e-8

# run harmonization and store the adjusted data
my_model, my_data_adj = harmonizationLearn(my_data, covars)

 # Boxplot values before/after NeuroCombat
boxplot_data = []; labels = []; colors = []  
for s,sub in enumerate(df_info['sub']):
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
#plt.show()

# Building back SC matrices after harmonization
reconstructed_Mat = np.zeros_like(MatMat[proc])
for s,sub in enumerate(df_info['sub']):
    tmpMat = np.zeros(np.shape(Mat))
    lower_triangle_indices = np.tril_indices(Mat.shape[0], k=-1)
    tmpMat[lower_triangle_indices] = df_Comb[sub]
    reconstructed_Mat[:,:,s] = tmpMat + tmpMat.T 

plt.figure()
plt.imshow(reconstructed_Mat[:,:,0])
plt.show()
    