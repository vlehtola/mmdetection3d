import pickle
import numpy as np
import os

def check_pkl_file(pkl_file_path):
    """Check if the .pkl file contains bounding boxes and other annotations."""
    if not os.path.exists(pkl_file_path):
        print(f"File {pkl_file_path} does not exist.")
        return None

    # Load the .pkl file
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check for annotations (bounding boxes)
    if 'data_list' in data:
        for i, entry in enumerate(data['data_list']):
            print(f"Sample {i+1}:")
            if 'annos' in entry:
                annos = entry['annos']
                if 'gt_boxes_upright_depth' in annos:
                    bboxes = annos['gt_boxes_upright_depth']
                    print(f"  Bounding Boxes: {bboxes.shape[0]} boxes")
                    print(f"  First bounding box (center, dimensions): {bboxes[0] if len(bboxes) > 0 else 'None'}")
                else:
                    print("  No bounding boxes found in this sample.")
                if 'class' in annos:
                    labels = annos['class']
                    print(f"  Object Classes: {len(labels)} labels")
                    print(f"  First class: {labels[0] if len(labels) > 0 else 'None'}")
            else:
                print("  No annotations found.")
    else:
        print(f"No 'data_list' found in {pkl_file_path}.")

if __name__ == "__main__":
    # Path to your .pkl file
    pkl_file_path = 'itckul_infos_train.pkl'  # Change this to your actual .pkl file

    # Check the file
    check_pkl_file(pkl_file_path)
