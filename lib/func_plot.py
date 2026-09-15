import os
import numpy as np
import nibabel as nb
import pandas as pd
import pyvista as pv
from sklearn.utils import Bunch
from nilearn import datasets
from PIL import Image


def plot_rois_pyvista_noaxes(roi_values, scale, out_dir, center_at_zero=False, label='brain', cmap='coolwarm', vmin=None, vmax=None, fmt='png'):
    
    if vmin==None:
        vmin = np.min(roi_values)
        vmax = np.max(roi_values)
    
    # If roi_values has 118 ROIs, extract cortical ones; if it has 114, use as is
    if len(roi_values) == 118:
        cort_rois = np.concatenate((np.arange(0,57), np.arange(59,116)))
        roi_values = roi_values[cort_rois]
    elif len(roi_values) != 114:
        raise ValueError(f"Expected 114 or 118 ROIs, got {len(roi_values)}")
    
    annots = [
        os.path.join('DATA', 'label', f'rh.lausanne2008.scale{scale}.annot'),
        os.path.join('DATA', 'label', f'lh.lausanne2008.scale{scale}.annot')]

    annot_right = nb.freesurfer.read_annot(annots[0])
    annot_left = nb.freesurfer.read_annot(annots[1])

    labels_right = [elem.decode('utf-8') for elem in annot_right[2]]
    labels_left = [elem.decode('utf-8') for elem in annot_left[2]]


    desikan_atlas = Bunch(map_left=annot_left[0], map_right=annot_right[0])

    roi_vect_right = np.full_like(desikan_atlas['map_right'], np.nan, dtype=float)
    roi_vect_left = np.full_like(desikan_atlas['map_left'], np.nan, dtype=float)

    roifname = os.path.join('data', 'label', 'roi_info.xlsx')
    roidata = pd.read_excel(roifname, sheet_name=f'SCALE {scale}')
    
    right_rois = roidata[(roidata['Hemisphere'] == 'rh') & (roidata['Structure'] == 'cort')]['Label Lausanne2008']
    left_rois = roidata[(roidata['Hemisphere'] == 'lh') & (roidata['Structure'] == 'cort')]['Label Lausanne2008']

    for i, roi in enumerate(right_rois):
        label_id = labels_right.index(roi)
        ids_roi = np.where(desikan_atlas['map_right'] == label_id)[0]
        roi_vect_right[ids_roi] = roi_values[i]

    for i, roi in enumerate(left_rois):
        label_id = labels_left.index(roi)
        ids_roi = np.where(desikan_atlas['map_left'] == label_id)[0]
        roi_vect_left[ids_roi] = roi_values[len(right_rois) + i]


    # Load GIFTI files using nibabel
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    right_surf = nb.load(fsaverage['pial_right'])  # GIFTI file for the right hemisphere
    left_surf = nb.load(fsaverage['pial_left'])    # GIFTI file for the left hemisphere
    
    # Extract vertices and faces from the GIFTI file
    right_verts = right_surf.darrays[0].data  # vertices
    right_faces = right_surf.darrays[1].data  # faces
    left_verts = left_surf.darrays[0].data   # vertices
    left_faces = left_surf.darrays[1].data   # faces
    
    # Check the format of right_faces and left_faces
    if right_faces.shape[1] == 3:
        right_faces_pv = np.hstack([np.full((right_faces.shape[0], 1), 3), right_faces])
        left_faces_pv = np.hstack([np.full((left_faces.shape[0], 1), 3), left_faces])
    else:
        raise ValueError('Unexpected face format. Faces should have 3 vertices per face.')

    # Create PolyData objects for right and left hemispheres
    surf_right = pv.PolyData(right_verts, right_faces_pv)
    surf_left = pv.PolyData(left_verts, left_faces_pv)

    # Ensure that the number of vertices matches the length of ROI values
    if len(roi_vect_right) != surf_right.n_points:
        raise ValueError(f"Mismatch: {len(roi_vect_right)} ROI values for {surf_right.n_points} points on the right hemisphere.")
    if len(roi_vect_left) != surf_left.n_points:
        raise ValueError(f"Mismatch: {len(roi_vect_left)} ROI values for {surf_left.n_points} points on the left hemisphere.")

    # Assign ROI values to point arrays
    surf_right['roi_map'] = roi_vect_right
    surf_left['roi_map'] = roi_vect_left

    # Create PyVista plotter with 2x2 subplots for different views
    plotter = pv.Plotter(shape=(2, 2), off_screen=True)
    
    # Try to reduce padding/borders between subplots
    try:
        plotter.window_size = (800, 800)  # Set a reasonable window size
    except Exception:
        pass

    # Define camera positions for different views
    views = {
         'Lateral Right': [(100, 0, 0), (0, 0, 0), (0, 0, 1)],
         'Medial Right': [(-100, 0, 0), (0, 0, 0), (0, 0, 1)],
         'Lateral Left': [(-100, 0, 0), (0, 0, 0), (0, 0, 1)],
         'Medial Left': [(100, 0, 0), (0, 0, 0), (0, 0, 1)],}


    # Create individual plotters for each view and stitch them together
    images = []
    img_size = 400
    
    for idx, (view_name, view_pos) in enumerate(views.items()):
        # Create a separate plotter for each view
        single_plotter = pv.Plotter(shape=(1, 1), off_screen=True, window_size=(img_size, img_size))
        single_plotter.subplot(0, 0)
        
        if 'Right' in view_name:
            actor = single_plotter.add_mesh(surf_right, scalars="roi_map", cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False)
        else:
            actor = single_plotter.add_mesh(surf_left, scalars="roi_map", cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False)
        
        single_plotter.remove_bounds_axes()
        single_plotter.camera_position = view_pos
        single_plotter.reset_camera()
        
        # Capture screenshot to bytes
        import io
        img_bytes = single_plotter.screenshot(return_img=True)
        img = Image.fromarray(img_bytes)
        images.append(img)
        
        single_plotter.close()
    
    # Stitch images together in a 2x2 grid with minimal gaps
    # Crop each image to remove excess whitespace from top/bottom
    crop_pixels = 50  # Adjust this value to control vertical spacing
    
    cropped_images = []
    for idx, img in enumerate(images):
        # Convert to RGBA for transparency support
        img_rgba = img.convert('RGBA')
        
        # Get pixel data and make white background transparent
        data = img_rgba.getdata()
        new_data = []
        for item in data:
            # If pixel is close to white (R>240, G>240, B>240), make it transparent
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                new_data.append((255, 255, 255, 0))  # Transparent white
            else:
                new_data.append(item)
        
        img_rgba.putdata(new_data)
        
        # Crop top and bottom to reduce vertical space
        width, height = img_rgba.size
        cropped = img_rgba.crop((0, crop_pixels, width, height - crop_pixels))
        cropped_images.append(cropped)
    
    # Create colorbar image
    colorbar_plotter = pv.Plotter(shape=(1, 1), off_screen=True, window_size=(150, 600))
    colorbar_plotter.subplot(0, 0)
    
    # Create a dummy mesh just to generate the colorbar
    if 'Right' in list(views.keys())[0]:
        colorbar_plotter.add_mesh(surf_right, scalars="roi_map", cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False, opacity=0)
    else:
        colorbar_plotter.add_mesh(surf_left, scalars="roi_map", cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False, opacity=0)
    
    # Add a large, centered colorbar
    colorbar_plotter.add_scalar_bar(title='', vertical=True, title_font_size=16, label_font_size=14,position_x=0.2, position_y=0.15, width=0.6, height=0.7, bold=True)
    
    # Get colorbar as image
    colorbar_img_array = colorbar_plotter.screenshot(return_img=True)
    colorbar_img = Image.fromarray(colorbar_img_array).convert('RGBA')
    colorbar_plotter.close()
    
    # Calculate new grid dimensions after cropping - add space for colorbar
    cropped_height = cropped_images[0].size[1]
    colorbar_width = 150
    grid_width = img_size * 2 + colorbar_width
    grid_height = cropped_height * 2
    
    # Create transparent background
    grid_img = Image.new('RGBA', (grid_width, grid_height), (255, 255, 255, 0))
    
    # Paste brain images in 2x2 grid
    for idx, img in enumerate(cropped_images):
        row = idx // 2
        col = idx % 2
        # Paste each cropped image at the correct position
        grid_img.paste(img, (col * img_size, row * cropped_height), img)
    
    # Paste colorbar on the right side, vertically centered
    colorbar_resized = colorbar_img.resize((colorbar_width, grid_height), Image.Resampling.LANCZOS)
    # Don't use the image as its own mask - paste without mask parameter
    grid_img.paste(colorbar_resized, (img_size * 2, 0))
    
    # Save the combined image
    save_fname = os.path.join(out_dir, f'{label}.{fmt}')
    print(f"Saving figure to: {save_fname}")
    grid_img.save(save_fname)

    return save_fname

