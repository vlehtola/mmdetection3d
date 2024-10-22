import laspy
from os import path as osp
import numpy as np
import argparse

# Split a LAS file into blocks that are 2 meters wide, sorted using x axis, and
# divide the blocks into training, validation, and test sets.
# Usage:
# python split_data.py path_to_input_las_file.las 

def split_las_file(input_path, las_filename, block_size=500.0, train_ratio=0.7, val_ratio=0.15):
    """
    Split a LAS file into blocks along the x-axis and divide the blocks into
    training, validation, and test sets.

    Args:
        input_path (str): Path to the input LAS file.
        las_filename (str): Filename of the LAS file.
        block_size (float): Size of each block along the x-axis (default is 500.0 centimeters).
        train_ratio (float): Proportion of blocks to be used for training (default is 0.7).
        val_ratio (float): Proportion of blocks to be used for validation (default is 0.15).
    """
    
    # Read the LAS file
    las_file = laspy.read('/'.join([input_path, las_filename]))
    print(f'Splitting LAS file, {las_filename}: {las_file}')
    # Create output directory and base file names
    output_las_path_template = input_path + "/{}_block_{}_{}.las"
    
    # Extract X coordinates directly without needing to extract all points first
    x_coords = las_file.X  # This avoids creating a large points array unnecessarily

    # Sort indices based on x-coordinates
    sorted_indices = np.argsort(x_coords)

    # Apply the sorted indices to all point data in a memory-efficient way
    sorted_points = las_file.points[sorted_indices]  # Sort points in one go

    # Define min and max x coordinates to know how many blocks are needed
    min_x, max_x = sorted_points.X[0], sorted_points.X[-1]

    x_coords_sorted = sorted_points.X
    block_start = min_x
    block_num = 0
    block_paths = []
    print(block_start, ' cm <', max_x, ' cm', 'Block size', block_size)

    # List to store the blocks
    all_blocks = []

    # Split data into 2-meter blocks
    while block_start < max_x:
        block_end = block_start + block_size
        
        # Select points that fall within this block
        mask = (x_coords_sorted >= block_start) & (x_coords_sorted < block_end)
        block_points = sorted_points[mask]

        if len(block_points) > 0:
            # Store the block in the list
            all_blocks.append((block_start, block_end, block_points))

        # Move to the next block
        block_start += block_size
        block_num += 1

    # Shuffle the blocks
    #np.random.shuffle(all_blocks)

    # Calculate the split indices for train, val, and test
    num_blocks = len(all_blocks)
    train_idx = int(train_ratio * num_blocks)
    val_idx = int((train_ratio + val_ratio) * num_blocks)

    # Split the blocks into train, val, and test
    train_blocks = all_blocks[:train_idx]
    val_blocks = all_blocks[train_idx:val_idx]
    test_blocks = all_blocks[val_idx:]

    # Function to write blocks to LAS files
    def write_blocks(blocks, split_name):
        output_paths = []
        for (start, end, points) in blocks:
            header = las_file.header
            las_block = laspy.create(point_format=header.point_format, file_version=header.version)
            las_block.points = points

            # Define output path for this block
            output_path = output_las_path_template.format(str(las_filename), int(start), int(end))
            output_paths.append(output_path)

            # Write the block LAS file
            if osp.isfile(output_path):
                print(f'File {output_path} already exists. skipping.')
            else:
                las_block.write(output_path)

        return output_paths

    # Write training, validation, and test blocks to files
    print("Writing training blocks...")
    train_paths = write_blocks(train_blocks, "train")

    print("Writing validation blocks...")
    val_paths = write_blocks(val_blocks, "val")

    print("Writing testing blocks...")
    test_paths = write_blocks(test_blocks, "test")

    return train_paths, val_paths, test_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a LAS file into blocks along the x-axis and divide into train/val/test sets.')
    parser.add_argument('input_las_path', type=str, help='Path to the input LAS file.')

    args = parser.parse_args()

    split_las_file(args.input_las_path)
