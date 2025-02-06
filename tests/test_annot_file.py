
import numpy as np
import nibabel as nib

# Path to your .annot file
annot_file = '.\\data\\label\\lh.lausanne2018.scale2.annot'

# Load the .annot file
labels, ctab, names = nib.freesurfer.read_annot(annot_file)
print(np.shape(ctab))

# labels: A 1D array with the label assigned to each vertex
# ctab: The color table, mapping labels to RGBA values
# names: List of region names corresponding to labels
