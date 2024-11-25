import numpy as np
import os
import argparse

# Define expected class labels for semantic masks
EXPECTED_SEMANTIC_LABELS = set(range(15))  # Update this range based on your expected class IDs

def check_bin_file(file_path, dtype=np.float32, verbose=True):
    """Read a .bin file and display its basic information."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    
    # Load the binary data
    data = np.fromfile(file_path, dtype=dtype)
    
    if verbose:
        print(f"Data shape: {data.shape} File: {file_path}")
    
    # For point cloud data (e.g., XYZ + RGB or XYZ + intensity), reshape if necessary
    if "points" in file_path:
        # Assume 6 features (e.g., XYZ + RGB or XYZ + intensity)
        data = data.reshape(-1, 6)
        if verbose:
            print(f"Reshaped data: {data.shape}")
            print(f"Some points:\n{data[10000:10003]}")
        # Check RGB values (columns 3, 4, 5) for being within [0, 255]
        rgb_values = data[:, 3:6]  # RGB values are assumed to be in columns 3, 4, 5
        if np.any((rgb_values < 0) | (rgb_values > 255)):
            print(f"Warning: Found RGB values out of range [0, 255] in {file_path}")
            invalid_rgb = rgb_values[(rgb_values < 0) | (rgb_values > 255)]
            print(f"Invalid RGB values: {invalid_rgb}")
        elif verbose:
            print(f"All RGB values are within the valid range [0, 255].")
        
        rgb_dtype = data[:, 3:6].dtype
        if rgb_dtype != np.float32:
            print(f"Error: RGB dtype in {file_path} is {rgb_dtype}, expected float32.")

    elif "semantic_mask" in file_path:
        # For semantic masks, display the unique values and check for unexpected values
        unique_values = np.unique(data)
        if verbose:
            print(f"Unique values in {file_path}: {unique_values}")
        invalid_labels = [label for label in unique_values if label not in EXPECTED_SEMANTIC_LABELS]
        if invalid_labels:
            print(f"Warning: Found unexpected semantic labels in {file_path}: {invalid_labels}")
        elif verbose:
            print(f"All semantic mask values are within the expected range.")
    elif "instance_mask" in file_path:
        # Check for negative values
        if np.any(data < 0):
            print(f"Warning: Found negative values in {file_path}: {data[data < 0]}")
        
    # Print "ok" for non-verbose mode if there are no warnings
    if not verbose:
        print(f"checked {file_path}")

    return data

def check_sample_files(data_dir, file_name, verbose=True):
    """Check point cloud, instance mask, and semantic mask files for a specific sample."""
    
    # Define file paths for the point cloud, instance mask, and semantic mask
    point_cloud_file = os.path.join(data_dir, 'points', f'{file_name}.bin')
    instance_mask_file = os.path.join(data_dir, 'instance_mask', f'{file_name}.bin')
    semantic_mask_file = os.path.join(data_dir, 'semantic_mask', f'{file_name}.bin')

    # Check the point cloud file (assuming float32 for point clouds)
    check_bin_file(point_cloud_file, dtype=np.float32, verbose=verbose)

    # Check the instance mask file (assuming int64 for instance masks)
    check_bin_file(instance_mask_file, dtype=np.int64, verbose=verbose)

    # Check the semantic mask file (assuming int64 for semantic masks)
    check_bin_file(semantic_mask_file, dtype=np.int64, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate multiple point cloud files listed in text files.")
    parser.add_argument('--root_dir', type=str, default='./', help="Root directory where the .bin files are stored")
    parser.add_argument('--file_list_paths', nargs='+', default=['./meta_data/itckul_val.txt', './meta_data/itckul_train.txt', './meta_data/itckul_test.txt'], help="List of paths to the text files containing file names")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")

    args = parser.parse_args()

    # Process each file list (val, train, test)
    for file_list_path in args.file_list_paths:
        # Determine the split based on the filename
        if 'val' in file_list_path:
            split = 'val'
        elif 'train' in file_list_path:
            split = 'train'
        else:
            continue
        
        # Read the list of files from the text file
        with open(file_list_path, 'r') as file_list:
            files = file_list.readlines()

        # Remove any extra whitespace or newlines from each file entry
        files = [file.strip() for file in files]

        # Process each file in the list
        for file_name in files:
            if args.verbose:
                print(f"Validating file: {file_name}")
            check_sample_files(args.root_dir, f'{split}_{file_name}', verbose=args.verbose)
            print("\n")
