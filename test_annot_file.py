

import nibabel as nib

# Path to your .annot file
annot_file = 'C:\\Users\\emeli\\Downloads\\lh.lausanne2018.scale1.annot'

# Load the .annot file
labels, ctab, names = nib.freesurfer.read_annot(annot_file)
print(names)

# labels: A 1D array with the label assigned to each vertex
# ctab: The color table, mapping labels to RGBA values
# names: List of region names corresponding to labels
