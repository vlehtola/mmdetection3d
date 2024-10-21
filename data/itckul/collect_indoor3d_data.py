import argparse
from os import path as osp, remove
import glob
import mmengine
from indoor3d_util import export
from split_data import split_las_file

# example filename: ITC-KUL_constructor/ITC_BUILDING/2021/PCD/pcd-tls/2021_labels.las

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output-folder',
    default='./itckul_data',
    help='output folder of the result.')
parser.add_argument(
    '--data-dir',
    default='./',
    help='itckul data directory.')
parser.add_argument(
    '--ann-file',
    default='meta_data/anno_paths.txt',
    help='The path of the file that stores the annotation names.')
args = parser.parse_args()

# Delete the existing meta_data files if they exist
files_to_delete = ['meta_data/itckul_train.txt', 'meta_data/itckul_val.txt', 'meta_data/itckul_test.txt']
for file in files_to_delete:
    if osp.exists(file):
        remove(file)
        print(f'Deleted existing file: {file}')

# here we open() the meta_data files to read las filenames
anno_paths = [line.rstrip() for line in open(args.ann_file)]
anno_paths = [osp.join(args.data_dir, p) for p in anno_paths]

output_folder = args.output_folder
mmengine.mkdir_or_exist(output_folder)

def process_paths(dataset_type, paths, output_folder):
    """
    Processes a list of LAS file paths for a given dataset type (train, val, test).
    
    :param dataset_type: The type of dataset being processed (e.g., 'train', 'val', 'test').
    :param paths: A list of file paths to process.
    :param output_folder: The folder where the output files will be saved.
    """
    assert(dataset_type in ['train', 'val', 'test'])
    # Output file path
    output_file = f'meta_data/itckul_{dataset_type}.txt'
    # Write the filenames to the text file
    with open(output_file, 'a') as file:
        for las_filepath in paths:        
            elems = las_filepath.split('/')
            out_filename = elems[-1]
            out_filename = out_filename[:-4]  # Omit the .las extension
            file.write(f"{out_filename}\n")
            print(f'Exporting data from annotation file: {las_filepath}')
            elements = las_filepath.split('/')
            out_filename = '_'.join(elements[-1:])  # Get the last part of the file path
            out_filename = out_filename[:-4]  # Omit the .las extension
            out_filename = f'{dataset_type}_{out_filename}'
            out_filename = osp.join(output_folder, out_filename)
        
            # Check if the file already exists
            if osp.isfile(f'{out_filename}_point.npy'):
                print(f'File {out_filename}_point.npy already exists. Skipping.')
                continue
            
            # Export data
            export(las_filepath, out_filename)

for anno_path in anno_paths:    
    las_directory, las_filename = osp.split(anno_path)
    train_paths, val_paths, test_paths = split_las_file(las_directory, las_filename, block_size=500.0, train_ratio=0.7, val_ratio=0.15)
    elems = anno_path.split('/')
    anno_path = '/'.join(elems[0:-1])
    
    # Example usage for train, val, and test sets:
    process_paths('train', train_paths, output_folder)
    process_paths('val', val_paths, output_folder)
    process_paths('test', test_paths, output_folder)

