import glob
from os import path as osp

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

BASE_DIR = osp.dirname(osp.abspath(__file__))

class_names = [
    x.rstrip() for x in open(osp.join(BASE_DIR, 'meta_data/class_names.txt'))
]
class2label = {one_class: i for i, one_class in enumerate(class_names)}

# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA TO POINTS, SEM_LABEL AND INS_LABEL FILES
# -----------------------------------------------------------------------------

import laspy
import numpy as np
import os.path as osp
import pandas as pd

def export(anno_path, out_filename):
    """Convert original dataset files to points, instance mask, and semantic
    mask files. We aggregated all the points from the annotated LAS file.

    Args:
        anno_path (str): path to annotated las file
        out_filename (str): path to save collected points and labels

    Note:
        The points are shifted before save, the most negative point is now
        at origin.
    """
    points_list = []

    # Read the LAS file
    las_file = laspy.read(anno_path)

    # Extract point data
    points = np.vstack((las_file.x, las_file.y, las_file.z, 
                        las_file.red, las_file.green, las_file.blue)).transpose()

    # Extract semantic labels
    sem_labels = las_file.segmentation_labels #classification
    ins_labels = las_file.object_labels

    # Combine points, semantic labels, and instance labels
    data = np.column_stack((points, sem_labels, ins_labels))

    # Adjust the points to move the most negative point to origin
    xyz_min = np.amin(data[:, :3], axis=0)
    data[:, :3] -= xyz_min

    # Save the data
    np.save(f'{out_filename}_point.npy', data[:, :6].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', data[:, 6].astype(np.int64))
    np.save(f'{out_filename}_ins_label.npy', data[:, 7].astype(np.int64))

