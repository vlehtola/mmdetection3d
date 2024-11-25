import numpy as np
import argparse
import os

def load_bin_file(file_path, dtype=np.float32, expected_features=6):
    """
    Load a .bin file and reshape it based on expected features.

    Args:
        file_path (str): Path to the .bin file.
        dtype (data-type): Data type of the binary data.
        expected_features (int): Number of features per point.

    Returns:
        np.ndarray: Loaded and reshaped data array.
    """
    try:
        data = np.fromfile(file_path, dtype=dtype)
        print(f"Loaded data from {file_path} with dtype={dtype}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if data.size % expected_features != 0:
        print(f"Warning: Data size {data.size} is not a multiple of expected features {expected_features}.")

    num_points = data.size // expected_features
    data = data[:num_points * expected_features]  # Truncate excess data
    data = data.reshape((num_points, expected_features))
    print(f"Reshaped data to {data.shape} (num_points, features)")

    return data

def analyze_data(data):
    """
    Analyze the loaded data and print data type information and statistics.

    Args:
        data (np.ndarray): The data array to analyze.
    """
    if data is None:
        print("No data to analyze.")
        return

    print("\nData Analysis:")
    print("-" * 60)
    print(f"Overall data type: {data.dtype}")
    print(f"Number of points: {data.shape[0]}")
    print(f"Number of features per point: {data.shape[1]}\n")

    for i in range(data.shape[1]):
        feature = data[:, i]
        feature_dtype = feature.dtype
        print(f"Feature {i}:")
        print(f"  Data type: {feature_dtype}")
        print(f"  Min value: {feature.min()}")
        print(f"  Max value: {feature.max()}")
        print(f"  Mean value: {feature.mean():.2f}")
        print(f"  Std deviation: {feature.std():.2f}\n")

    # Additional checks for RGB values if there are 6 features
    if data.shape[1] == 6:
        r, g, b = data[:, 3], data[:, 4], data[:, 5]
        # Check if RGB values are within [0, 255]
        if np.issubdtype(r.dtype, np.integer) and np.issubdtype(g.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
            rgb_in_range = np.all((r >= 0) & (r <= 255)) and np.all((g >= 0) & (g <= 255)) and np.all((b >= 0) & (b <= 255))
            print(f"RGB Values within [0, 255]: {rgb_in_range}")
        elif np.issubdtype(r.dtype, np.floating) and np.issubdtype(g.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
            rgb_in_range = np.all((r >= 0.0) & (r <= 255.0)) and np.all((g >= 0.0) & (g <= 255.0)) and np.all((b >= 0.0) & (b <= 255.0))
            print(f"RGB Values within [0.0, 255.0]: {rgb_in_range}")
        else:
            print("RGB Values have mixed or unexpected data types.")

def main():
    parser = argparse.ArgumentParser(description="Analyze a .bin file from the S3DIS dataset prepped with mmdetection3d.")
    parser.add_argument('--bin_path', type=str, required=True, help="Path to the .bin file (e.g., ./points.bin).")
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'int64', 'float64'], help="Data type to interpret the .bin file.")
    parser.add_argument('--expected_features', type=int, default=6, help="Number of features per point (e.g., 6 for x,y,z,r,g,b).")
    args = parser.parse_args()

    # Convert dtype string to numpy dtype
    dtype_mapping = {
        'float32': np.float32,
        'float64': np.float64,
        'uint8': np.uint8,
        'int64': np.int64,
        'int32': np.int32
    }
    dtype = dtype_mapping.get(args.dtype, np.float32)

    # Check if the bin file exists
    if not os.path.isfile(args.bin_path):
        print(f"Error: The file {args.bin_path} does not exist.")
        return

    # Load the .bin file
    data = load_bin_file(args.bin_path, dtype=dtype, expected_features=args.expected_features)

    # Analyze the data
    analyze_data(data)

if __name__ == "__main__":
    main()
