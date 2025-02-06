
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import nilearn
from nilearn import plotting
import nibabel as nb
from sklearn.utils import Bunch
import pandas as pd
import pygsp
import tqdm
import pyvista as pv
from nilearn import datasets



def plot_shaded(x, y, axis, color, label, ax, ls='-'):
    ax.plot(x, np.median(y, axis=axis), color=color, lw=2, label=label, ls=ls)
    ax.fill_between(x, *np.percentile(y, (5, 95), axis=axis), color=color,
                    alpha=.4)
    return ax


def plot_shaded_norm(x, y, axis, color, label, ax, ls='-'):
    ax.plot(x, np.mean(y, axis=axis), color=color, lw=2, label=label, ls=ls)
    ax.fill_between(x, *[[1], [-1]]*np.std(y, axis=axis) +
                    np.mean(y, axis=axis), color=color, alpha=.4)
    return ax


def my_box_plot(data, ax, colors, labels):
    bplot = ax.boxplot(data, notch=False, vert=True, patch_artist=True,
                       labels=labels, showfliers=False)

    for i, color in enumerate(colors):
        plt.setp(bplot['boxes'][i],color=color, facecolor=color, alpha=.5,
                 lw=2)
        plt.setp(bplot['whiskers'][i * 2:i * 2 + 2], color=color,
                 alpha=1, lw=2)
        plt.setp(bplot['caps'][i * 2:i * 2 + 2], color=color, alpha=1, lw=2)
        plt.setp(bplot['medians'][i], color=color, alpha=1, lw=2)
        ymin, ymax = ax.get_ylim()

        x = np.random.normal(i + 1, 0.1, size=np.size(data[i]))
        if labels[i] == 'NH-sur.':
            alpha = .005
        else:
            alpha = .3

        ax.scatter(x, data[i], 40, np.tile(np.array(color).reshape(-1, 1),
                   len(x))[:3].T, alpha=alpha, edgecolors='gray')
        ax.set_ylim([ymin, ymax])
    return ax


def plot_rois(roi_values, scale,  config, center_at_zero=False, label='brain',
                  cmap='coolwarm', vmin=None, vmax=None, fmt='png'):

    annots = [os.path.join('data','label','rh.lausanne2008.scale{}.annot'.format(scale)),
              os.path.join('data','label','lh.lausanne2008.scale{}.annot'.format(scale))]

    annot_right = nb.freesurfer.read_annot(annots[0])
    annot_left = nb.freesurfer.read_annot(annots[1])

    labels_right = [elem.decode('utf-8') for elem in annot_right[2]]
    labels_left = [elem.decode('utf-8') for elem in annot_left[2]]

    desikan_atlas = Bunch(map_left=annot_left[0],
                          map_right=annot_right[0])

    parcellation_right = desikan_atlas['map_right']
    roi_vect_right = np.zeros_like(parcellation_right, dtype=float) * np.nan

    parcellation_left = desikan_atlas['map_left']
    roi_vect_left = np.zeros_like(parcellation_left, dtype=float) * np.nan

    roifname = os.path.join('data','label','roi_info.xlsx')
    roidata = pd.read_excel(roifname, sheet_name='SCALE {}'.format(scale))
    cort = np.where(roidata['Structure'] == 'cort')[0]

    right_rois = ([roidata['Label Lausanne2008'][i] for i in
                   range(len(roidata)) if ((roidata['Hemisphere'][i] == 'rh') &
                   (roidata['Structure'][i] == 'cort'))])
    left_rois = ([roidata['Label Lausanne2008'][i] for i in
                  range(len(roidata)) if ((roidata['Hemisphere'][i] == 'lh') &
                  (roidata['Structure'][i] == 'cort'))])
    
    for i in range(len(right_rois)):
        label_id = labels_right.index(right_rois[i])
        ids_roi = np.where(parcellation_right == label_id)[0]
        roi_vect_right[ids_roi] = roi_values[i]

    for i in range(len(left_rois)):
        label_id = labels_left.index(left_rois[i])
        ids_roi = np.where(parcellation_left == label_id)[0]
        roi_vect_left[ids_roi] = roi_values[len(right_rois) + i]

    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

    if vmin is None:
        vmin = min([0, min(roi_values)])
    if vmax is None:
        vmax = max(roi_values)

    if center_at_zero:
        max_val = max([abs(vmin), vmax])
        vmax = max_val
        vmin = -max_val

    
    plt.ioff()
    fig, axs = plt.subplots(1, 3, figsize=(6, 2), subplot_kw={'projection': '3d'})
    #fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'})

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='lateral',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[0])

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='dorsal',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[1])

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='dorsal',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[1])

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='lateral',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[2])


    axs[1].view_init(elev=90, azim=270)
    
    #for i in range(3):
    #    if i in [1]:
    #        axs[i].dist = 5.7
    #    else:
    #        axs[i].dist = 6

    fig.tight_layout()
    
    save_fname = f'%s/%s.%s'%(config["Parameters"]["fig_dir"], label, fmt)
    fig.savefig(save_fname)
    plt.close(fig)
    plt.show(block=False)
    
    
    return save_fname


def plot_rois_pyvista(roi_values, scale, config, center_at_zero=False, label='brain', cmap='coolwarm', vmin=None, vmax=None, fmt='png'):
    
    annots = [
        os.path.join('data', 'label', f'rh.lausanne2008.scale{scale}.annot'),
        os.path.join('data', 'label', f'lh.lausanne2008.scale{scale}.annot')
    ]

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

    # Create PyVista plotter with 2x3 subplots for different views
    plotter = pv.Plotter(shape=(2, 3), off_screen=True)

    # Define camera positions for different views
    views = {
        'Lateral Right': [(100, 0, 0), (0, 0, 0), (0, 0, 1)],
        'Medial Right': [(-100, 0, 0), (0, 0, 0), (0, 0, 1)],
        'Superior Right': [(0, 0, 100), (0, 0, 0), (0, 1, 0)],
        'Lateral Left': [(-100, 0, 0), (0, 0, 0), (0, 0, 1)],
        'Medial Left': [(100, 0, 0), (0, 0, 0), (0, 0, 1)],
        'Superior Left': [(0, 0, 100), (0, 0, 0), (0, 1, 0)]
    }

    # Add the right hemisphere with different views
    for idx, (view_name, view_pos) in enumerate(views.items()):
        row = idx // 3
        col = idx % 3
        plotter.subplot(row, col)
        if 'Right' in view_name:
            plotter.add_mesh(surf_right, scalars="roi_map", cmap=cmap, clim=(vmin, vmax), show_scalar_bar=True)
        else:
            plotter.add_mesh(surf_left, scalars="roi_map", cmap=cmap, clim=(vmin, vmax), show_scalar_bar=True)
        plotter.add_text(view_name, font_size=10)

        # Set the camera position and zoom level for each subplot
        plotter.camera_position = view_pos
        plotter.reset_camera()  # Adjust the camera to fit the view

    # Adjust layout and link views
    #plotter.link_views()
    
    plotter.screenshot()
    # Show the plot (to initialize properly)
    plotter.show(interactive=False)  # This initializes the plotter and prepares for screenshots
    

    # Save the entire subplot layout as a single image
    save_fname = os.path.join(config["Parameters"]["fig_dir"], f'{label}.{fmt}')
    print(f"Saving figure to: {save_fname}")
    plotter.screenshot(save_fname)

    return save_fname
