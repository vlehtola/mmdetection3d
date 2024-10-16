import argparse
from os import path as osp
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

# here we open() the meta_data files to read las filenames
anno_paths = [line.rstrip() for line in open(args.ann_file)]
anno_paths = [osp.join(args.data_dir, p) for p in anno_paths]

output_folder = args.output_folder
mmengine.mkdir_or_exist(output_folder)

for anno_path in anno_paths:    
    print(f'Splitting data file: {anno_path}')
    las_directory, las_filename = osp.split(anno_path)
    train_paths, val_paths, test_paths = split_las_file(las_directory, las_filename)    
    #las_files = [anno_path]  # skip splitting the las files for debug
    elems = anno_path.split('/')
    anno_path = '/'.join(elems[0:-1])
    #las_files = glob.glob(osp.join(anno_path, '/*.las'))
    las_files = train_paths + val_paths + test_paths
    print(anno_path, las_files)
    for las_filepath in las_files:        
        print(f'Exporting data from annotation file: {las_filepath}')
        elements = las_filepath.split('/')
        out_filename = \
            '_'.join(elements[-1:])
        out_filename = out_filename[:-4]  # omit .las
        out_filename = osp.join(output_folder, out_filename)
        if osp.isfile(f'{out_filename}_point.npy'):
            print('File already exists. skipping.')
            continue
        export(las_filepath, out_filename)

