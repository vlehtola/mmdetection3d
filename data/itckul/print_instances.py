import numpy as np
import os

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
    
    # For multidimensional data (like point clouds), reshape if necessary
    if "points" in file_path:
        # Assume 6 features (e.g., XYZ + RGB or XYZ + intensity)
        data = data.reshape(-1, 6)
        print(f"Reshaped points data: {data.shape}")
        print(f"First 5 points:\n{data[:5]}")
    else:
        # For instance/semantic masks, just display the unique values (classes or instance IDs)
        print(f"Unique values in {file_path}: {np.unique(data)}")
    
    return data

def check_files(root_dir, sample_idx, split):
    """Check points, instance mask, and semantic mask files for a specific sample."""
    point_file = os.path.join(root_dir, 'points', f'{split}_{sample_idx}.bin')
    instance_mask_file = os.path.join(root_dir, 'instance_mask', f'{split}_{sample_idx}.bin')
    semantic_mask_file = os.path.join(root_dir, 'semantic_mask', f'{split}_{sample_idx}.bin')

    # Check the points file (assuming float32 for point clouds)
    check_bin_file(point_file, dtype=np.float32)

    # Check the instance mask file (assuming int64 for instance masks)
    check_bin_file(instance_mask_file, dtype=np.int64)

    # Check the semantic mask file (assuming int64 for semantic masks)
    check_bin_file(semantic_mask_file, dtype=np.int64)

if __name__ == "__main__":
    # Root directory where the .bin files are stored
    root_dir = ''  # Update this path based on your setup

    # Sample index and split (e.g., 'train', 'val', or 'test')
    sample_idx = '2021_ITC_TLS'  # Update this based on your sample index
    split = 'train'  # Change this if checking validation or test sets

    # Check the files for the given sample
    check_files(root_dir, sample_idx, split)
