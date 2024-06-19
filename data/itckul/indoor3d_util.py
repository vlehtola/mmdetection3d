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

    # Map from 200 semantic label format of itckul dataset to 14 labels of s3dis format
    # Also, add stair class and railing class.

    # orig_classes = {
    #     0: "Structural",
    #     1: "Columns",
    #     2: "Beams",
    #     3: "Floors",
    #     4: "Walls",
    #     5: "Stairs",
    #     6: "Roofs",
    #     7: "CurtainWalls",
    #     11: "Ceilings",
    #     12: "Doors",
    #     13: "Windows",
    #     14: "Railings"}
    
    # class_names = (
    #     'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    #     'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter', 'stair'
    #     )  
    mapping = {
    0: 12,  # 'Structural' -> 'clutter'
    1: 4,     # 'Columns' -> 'column'
    2: 3,     # 'Beams' -> 'beam'
    3: 1,     # 'Floors' -> 'floor'
    4: 2,     # 'Walls' -> 'wall'
    5: 13,    # 'Stairs' -> 'stair'
    6: 0,     # 'Roofs' -> 'ceiling'
    7: 2,     # 'CurtainWalls' -> 'wall'
    11: 0,    # 'Ceilings' -> 'ceiling'
    12: 6,    # 'Doors' -> 'door'
    13: 5,    # 'Windows' -> 'window'
    14: 3     # 'Railings' -> 'beam'
    }
    # Define the default label for 'clutter'
    default_label = 12

    # Remap labels. Anything undefined in the mapping is given the default_label
    sem_labels = map_labels(sem_labels, mapping, default_label)

    # Combine points, semantic labels, and instance labels
    data = np.column_stack((points, sem_labels, ins_labels))

    # Adjust the points to move the most negative point to origin
    xyz_min = np.amin(data[:, :3], axis=0)
    data[:, :3] -= xyz_min

    # Save the data
    np.save(f'{out_filename}_point.npy', data[:, :6].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', data[:, 6].astype(np.int64))
    np.save(f'{out_filename}_ins_label.npy', data[:, 7].astype(np.int64))

# Function to map original labels to new labels
def map_labels(orig_labels, mapping, default_label):
    new_labels = np.copy(orig_labels)
    unique_labels = np.unique(orig_labels)
    for label in unique_labels:
        if label in mapping and mapping[label] is not None:
            new_labels[orig_labels == label] = mapping[label]
        else:
            new_labels[orig_labels == label] = default_label
    return new_labels

