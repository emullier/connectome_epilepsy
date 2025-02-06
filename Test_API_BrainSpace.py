
import numpy as np
import matplotlib.pyplot as plt
from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels


##########################
### BUILDING GRADIENTS ###
##########################
### https://brainspace.readthedocs.io/en/latest/python_doc/auto_examples/plot_tutorial1.html

# First load mean connectivity matrix and Schaefer parcellation
conn_matrix = load_group_fc('schaefer', scale=400)
labeling = load_parcellation('schaefer', scale=400, join=True)

# and load the conte69 surfaces
surf_lh, surf_rh = load_conte69()

#plot_hemispheres(surf_lh, surf_rh, array_name=labeling, size=(1200, 200),cmap='tab20', zoom=1.85)

# Ask for 10 gradients (default)
gm = GradientMaps(n_components=10, random_state=0)
gm.fit(conn_matrix)

mask = labeling != 0

grad = [None] * 2
for i in range(2):
    # map the gradient to the parcels
    grad[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask, fill=np.nan)

#plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap='viridis_r',color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1.55)

fig, ax = plt.subplots(1, figsize=(5, 4))
ax.scatter(range(gm.lambdas_.size), gm.lambdas_)
ax.set_xlabel('Component Nb')
ax.set_ylabel('Eigenvalue')
#plt.show()


##########################################
### CUSTOMIZING AND ALIGNING GRADIENTS ###
##########################################
### https://brainspace.readthedocs.io/en/latest/python_doc/auto_examples/plot_tutorial2.html 

# First load mean connectivity matrix and Schaefer parcellation
conn_matrix = load_group_fc('schaefer', scale=400)
labeling = load_parcellation('schaefer', scale=400, join=True)

mask = labeling != 0

# and load the conte69 hemisphere surfaces
surf_lh, surf_rh = load_conte69()

### Different gradient kernel
kernels = ['pearson', 'spearman', 'normalized_angle']

gradients_kernel = [None] * len(kernels)
for i, k in enumerate(kernels):
    gm = GradientMaps(kernel=k, approach='dm', random_state=0)
    gm.fit(conn_matrix)
    gradients_kernel[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask, fill=np.nan)

label_text = ['Pearson', 'Spearman', 'Normalized\nAngle']
#plot_hemispheres(surf_lh, surf_rh, array_name=gradients_kernel, size=(1200, 600), cmap='viridis_r', color_bar=True, label_text=label_text, zoom=1.45)

### Different gradient dimensionality reduction
# PCA, Laplacian eigenmaps and diffusion mapping
embeddings = ['pca', 'le', 'dm']

gradients_embedding = [None] * len(embeddings)
for i, emb in enumerate(embeddings):
    gm = GradientMaps(kernel='normalized_angle', approach=emb, random_state=0)
    gm.fit(conn_matrix)

    gradients_embedding[i] = map_to_labels(gm.gradients_[:, 0], labeling, mask=mask, fill=np.nan)

label_text = ['PCA', 'LE', 'DM']
#plot_hemispheres(surf_lh, surf_rh, array_name=gradients_embedding, size=(1200, 600), cmap='viridis_r', color_bar=True, label_text=label_text, zoom=1.45)


##########################
### GRADIENT AGLINMENT ###
##########################

conn_matrix2 = load_group_fc('schaefer', scale=400, group='holdout')
print( 'Dimensions conn_matrix: (%d, %d)' %np.shape(conn_matrix))
print('Dimensions conn_matrix2: (%d,%d)' % np.shape(conn_matrix2))

gp = GradientMaps(kernel='normalized_angle', alignment='procrustes')
gj = GradientMaps(kernel='normalized_angle', alignment='joint')

gp.fit([conn_matrix, conn_matrix2])
gj.fit([conn_matrix, conn_matrix2])

## gp contains the Procrustes aligned data
## gj contains the joint aligned data

# First gradient from original and holdout data, without alignment
gradients_unaligned = [None] * 2
for i in range(2):
    gradients_unaligned[i] = map_to_labels(gp.gradients_[i][:, 0], labeling,mask=mask, fill=np.nan)

label_text = ['Unaligned\nGroup 1', 'Unaligned\nGroup 2']
#plot_hemispheres(surf_lh, surf_rh, array_name=gradients_unaligned, size=(1200, 400),cmap='viridis_r', color_bar=True, label_text=label_text, zoom=1.5)

# With procrustes alignment
gradients_procrustes = [None] * 2
for i in range(2):
    gradients_procrustes[i] = map_to_labels(gp.aligned_[i][:, 0], labeling, mask=mask,fill=np.nan)

label_text = ['Procrustes\nGroup 1', 'Procrustes\nGroup 2']
#plot_hemispheres(surf_lh, surf_rh, array_name=gradients_procrustes, size=(1200, 400), cmap='viridis_r', color_bar=True, label_text=label_text, zoom=1.5)

# With joint alignment
gradients_joint = [None] * 2
for i in range(2):
    gradients_joint[i] = map_to_labels(gj.aligned_[i][:, 0], labeling, mask=mask,
                                       fill=np.nan)

label_text = ['Joint\nGroup 1', 'Joint\nGroup 2']
#plot_hemispheres(surf_lh, surf_rh, array_name=gradients_joint, size=(1200, 400), cmap='viridis_r', color_bar=True, label_text=label_text, zoom=1.5)


### Alignment to a reference gradient
gref = GradientMaps(kernel='normalized_angle', approach='le')
gref.fit(conn_matrix2)

galign = GradientMaps(kernel='normalized_angle', approach='le', alignment='procrustes')
galign.fit(conn_matrix, reference=gref.gradients_)