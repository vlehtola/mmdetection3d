import numpy as np
import os
import random

def check_bin_file(file_path, dtype=np.float32):
    """Read a .bin file and display its basic information."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    
    # Load the binary data
    data = np.fromfile(file_path, dtype=dtype)
    
    # Print basic information about the data
    print(f"File: {file_path}")
    print(f"Data shape: {data.shape}")
    
    # For point cloud data (e.g., XYZ + RGB or XYZ + intensity), reshape if necessary
    if "points" in file_path:
        # Assume 6 features (e.g., XYZ + RGB or XYZ + intensity)
        data = data.reshape(-1, 6)
        print(f"Reshaped points data: {data.shape}")
        print(f"Some 5 points:\n{data[10000:]}")
        # Check RGB values (columns 4, 5, 6) for being within [0, 255]
        rgb_values = data[:, 2:5]  # RGB values are assumed to be in columns 3,4,5 (XYZ in 0,1,2)
        if np.any((rgb_values < 0) | (rgb_values > 255)):
            print(f"Warning: Found RGB values out of range [0, 255] in {file_path}")
            invalid_rgb = rgb_values[(rgb_values < 0) | (rgb_values > 255)]
            print(f"Invalid RGB values: {invalid_rgb}")
        else:
            print(f"All RGB values are within the valid range [0, 255].")
    else:
        # For instance or semantic masks, display the unique values (e.g., instance IDs or class labels)
        print(f"Unique values in {file_path}: {np.unique(data)}")
    
    return data

def check_sample_files(data_dir, file_name):
    """Check point cloud, instance mask, and semantic mask files for a specific sample."""
    
    # Define file paths for the point cloud, instance mask, and semantic mask
    point_cloud_file = os.path.join(data_dir, 'points', f'{file_name}.bin')
    instance_mask_file = os.path.join(data_dir, 'instance_mask', f'{file_name}.bin')
    semantic_mask_file = os.path.join(data_dir, 'semantic_mask', f'{file_name}.bin')

    # Check the point cloud file (assuming float32 for point clouds)
    check_bin_file(point_cloud_file, dtype=np.float32)

    # Check the instance mask file (assuming int64 for instance masks)
    check_bin_file(instance_mask_file, dtype=np.int64)

    # Check the semantic mask file (assuming int64 for semantic masks)
    check_bin_file(semantic_mask_file, dtype=np.int64)


if __name__ == "__main__":
    # Set the root directory where the .bin files are stored
    root_dir = './'  # Update this path based on your setup
    file_list_path = './meta_data/itckul_val.txt'

    split = 'train'
    if 'val' in file_list_path:
        split = 'val'
    if 'test' in file_list_path:
        split = 'test'
    # Read the list of files from the text file
    with open(file_list_path, 'r') as file_list:
        files = file_list.readlines()

    # Remove any extra whitespace or newlines from each file entry
    files = [file.strip() for file in files]

    random_file = random.choice(files)
    print(f"Selected random file for validation: {random_file}")
    if(True): 
        check_sample_files(root_dir, f'{split}_{random_file}')
    else: 
        # Process each file in the list
        for file_name in files:
            print(f"Validating file: {file_name}")
            check_sample_files(root_dir, f'{split}_{file_name}')
            
            print("\n")

