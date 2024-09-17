import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


tmp = os.listdir('output/neurosynth_results')
results_maps = []
for f,file in enumerate(tmp):
    if file.startswith('df_results_map'):
        results_maps.append('output/neurosynth_results/%s'%(file))
results_maps = np.array(results_maps)

thr=0; names_maps = []; 
for re,result in enumerate(results_maps):
    df_tmp = pd.read_csv(result)
    if re==0:
        prefix = "LDA50_abstract_weight__"
        data = np.zeros((len(df_tmp), len(results_maps)))
    df_tmp['feature'] = df_tmp['feature'].str.replace(f'^{prefix}', '', regex=True)
    name_neurosynth = np.array(df_tmp['feature'].copy())
    df_tmp[df_tmp['r']<thr] = 0 
    data[:,re] = np.array(df_tmp['r'])
    names_maps.append('eigenvector%d'%re) 
names_maps = np.array(names_maps)
   
df_plot = pd.DataFrame(data)
topics_to_keep = [0,2,3,5,6,8,9,15,16,17,18,19,20,23,24,26,28,30,32,33,37,38,41,42,43,44,47,48]
sns.set(context="paper", font="sans-serif", font_scale=2)
f, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)
sns.heatmap(data[topics_to_keep,:],  cbar=True, yticklabels=name_neurosynth[topics_to_keep], xticklabels=names_maps, ax=ax1, linewidths=1, square=True, cmap='Greys', robust=False)


#cax = sns.heatmap(df_test, linewidths=1, square=True, cmap='Greys', robust=False, ax=ax1)
#cbar = cax.collections[0].colorbar
#cbar.set_label('r', rotation=270)
##cbar.set_ticks([thr, vmax])
##cbar.set_ticklabels([thr, vmax])
#cbar.outline.set_edgecolor('black')
#cbar.outline.set_linewidth(0.5)
#plt.draw()
#plt.show(block=True)

plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=10)
#plt.title('Heatmap of map0 and map1 values')
plt.show()


