import os
from concurrent import futures as futures
from os import path as osp

import mmengine
import numpy as np


class ITCKULData(object):
    """ITCKUL data.

    Generate itckul infos for itckul_converter.

    Args:
        root_path (str): Root path of the raw data.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.data_dir = osp.join(root_path,
                                 'itckul')

        # Following `GSDN <https://arxiv.org/abs/2006.12356>`_, use 5 furniture
        # classes for detection: table, chair, sofa, bookcase, board.
        #self.cat_ids = np.array([7, 8, 9, 10, 11])
        self.cat_ids = np.array(range(15)) # use instances for all classes
        self.cat_ids2class = {
            cat_id: i
            for i, cat_id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'itckul_{split}.txt')
        mmengine.check_file_exist(split_file)
        self.sample_id_list = mmengine.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {
                'num_features': 6,
                'lidar_idx': f'{self.split}_{sample_idx}'
            }
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'itckul_data',
                                    f'{self.split}_{sample_idx}_point.npy')
            pts_instance_mask_path = osp.join(
                self.root_dir, 'itckul_data',
                f'{self.split}_{sample_idx}_ins_label.npy')
            pts_semantic_mask_path = osp.join(
                self.root_dir, 'itckul_data',
                f'{self.split}_{sample_idx}_sem_label.npy')

            points = np.load(pts_filename).astype(np.float32)
            pts_instance_mask = np.load(pts_instance_mask_path).astype(
                np.int64)
            pts_semantic_mask = np.load(pts_semantic_mask_path).astype(
                np.int64)

            mmengine.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            mmengine.mkdir_or_exist(osp.join(self.root_dir, 'instance_mask'))
            mmengine.mkdir_or_exist(osp.join(self.root_dir, 'semantic_mask'))

            points.tofile(
                osp.join(self.root_dir, 'points',
                         f'{self.split}_{sample_idx}.bin'))
            pts_instance_mask.tofile(
                osp.join(self.root_dir, 'instance_mask',
                         f'{self.split}_{sample_idx}.bin'))
            pts_semantic_mask.tofile(
                osp.join(self.root_dir, 'semantic_mask',
                         f'{self.split}_{sample_idx}.bin'))

            info['pts_path'] = osp.join('points',
                                        f'{self.split}_{sample_idx}.bin')
            info['pts_instance_mask_path'] = osp.join(
                'instance_mask', f'{self.split}_{sample_idx}.bin')
            info['pts_semantic_mask_path'] = osp.join(
                'semantic_mask', f'{self.split}_{sample_idx}.bin')
            info['annos'] = self.get_bboxes(points, pts_instance_mask,
                                            pts_semantic_mask)

            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def get_bboxes(self, points, pts_instance_mask, pts_semantic_mask):
        """Convert instance masks to axis-aligned bounding boxes.

        Args:
            points (np.array): Scene points of shape (n, 6).
            pts_instance_mask (np.ndarray): Instance labels of shape (n,).
            pts_semantic_mask (np.ndarray): Semantic labels of shape (n,).

        Returns:
            dict: A dict containing detection infos with following keys:

                - gt_boxes_upright_depth (np.ndarray): Bounding boxes
                    of shape (n, 6)
                - class (np.ndarray): Box labels of shape (n,)
                - gt_num (int): Number of boxes.
        """
       
        bboxes, labels = [], []
        
        # Check instance mask and semantic mask validity
        print(f"Points shape: {points.shape}")
        print(f"Instance mask shape: {pts_instance_mask.shape}")
        print(f"Max instance ID: {pts_instance_mask.max()}")
        print(f"Semantic mask shape: {pts_semantic_mask.shape}")
        print(f"Unique instance IDs: {np.unique(pts_instance_mask)}")
        print(f"Unique semantic IDs: {np.unique(pts_semantic_mask)}")

        for i in range(1, pts_instance_mask.max() + 1):
            ids = pts_instance_mask == i
            if ids.sum() == 0:
                print(f"Empty instance {i}", end='')
                continue
            
            mask = pts_semantic_mask[ids]
            if mask.min() != mask.max():
                print(f"Inconsistent semantic labels for instance {i}")
                continue
            
            label = mask[0]
            # Keep only furniture objects
            if label in self.cat_ids2class:
                labels.append(self.cat_ids2class[label])
                pts = points[:, :3][ids]
                min_pts = pts.min(axis=0)
                max_pts = pts.max(axis=0)
                locations = (min_pts + max_pts) / 2
                dimensions = max_pts - min_pts
                bbox = np.concatenate((locations, dimensions))
                bboxes.append(bbox)
                print(f"Instance {i} bounding box: {bbox}")

        annotation = dict()
        annotation['gt_boxes_upright_depth'] = np.array(bboxes)
        annotation['class'] = np.array(labels)
        annotation['gt_num'] = len(labels)
        
        # Check final output
        print(f"Generated {len(bboxes)} bounding boxes and {len(labels)} labels")
        
        return annotation



class ITCKULSegData(object):
    """itckul dataset used to generate infos for semantic segmentation task.

    Args:
        data_root (str): Root path of the raw data.
        ann_file (str): The generated scannet infos.
        split (str, optional): Set split type of the data. Default: 'train'.
        num_points (int, optional): Number of points in each data input.
            Default: 8192.
        label_weight_func (function, optional): Function to compute the
            label weight. Default: None.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split='train',
                 num_points=4096,
                 label_weight_func=None):
        self.data_root = data_root
        self.split = split
        assert split in ['train', 'val', 'test']
        self.data_infos = mmengine.load(ann_file)
        self.num_points = num_points

        self.all_ids = np.arange(15)  # all possible ids
        self.cat_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                 12,13,14])  # used for seg task
        self.ignore_index = len(self.cat_ids)

        self.cat_id2class = np.ones(
            (self.all_ids.shape[0], ), dtype=np.int64) * self.ignore_index
        for i, cat_id in enumerate(self.cat_ids):
            self.cat_id2class[cat_id] = i

        # label weighting function is taken from
        # https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py#L24
        self.label_weight_func = (lambda x: 1.0 / np.log(1.2 + x)) if \
            label_weight_func is None else label_weight_func

    def get_seg_infos(self):
        scene_idxs, label_weight = self.get_scene_idxs_and_label_weight()
        save_folder = osp.join(self.data_root, 'seg_info')
        mmengine.mkdir_or_exist(save_folder)
        np.save(
            osp.join(save_folder, f'{self.split}_resampled_scene_idxs.npy'),
            scene_idxs)
        np.save(
            osp.join(save_folder, f'{self.split}_label_weight.npy'),
            label_weight)
        print(f'{self.split} resampled scene index and label weight saved')

    def _convert_to_label(self, mask):
        """Convert class_id in loaded segmentation mask to label."""
        if isinstance(mask, str):
            if mask.endswith('npy'):
                mask = np.load(mask)
            else:
                mask = np.fromfile(mask, dtype=np.int64)
        label = self.cat_id2class[mask]
        return label

    def get_scene_idxs_and_label_weight(self):
        """Compute scene_idxs for data sampling and label weight for loss
        calculation.

        We sample more times for scenes with more points. Label_weight is
        inversely proportional to number of class points.
        """
        num_classes = len(self.cat_ids)
        num_point_all = []
        label_weight = np.zeros((num_classes + 1, ))  # ignore_index
        for data_info in self.data_infos:
            label = self._convert_to_label(
                osp.join(self.data_root, data_info['pts_semantic_mask_path']))
            num_point_all.append(label.shape[0])
            class_count, _ = np.histogram(label, range(num_classes + 2))
            label_weight += class_count

        # repeat scene_idx for num_scene_point // num_sample_point times
        sample_prob = np.array(num_point_all) / float(np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) / float(self.num_points))
        scene_idxs = []
        for idx in range(len(self.data_infos)):
            scene_idxs.extend([idx] * int(round(sample_prob[idx] * num_iter)))
        scene_idxs = np.array(scene_idxs).astype(np.int32)

        # calculate label weight, adopted from PointNet++
        label_weight = label_weight[:-1].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = self.label_weight_func(label_weight).astype(np.float32)

        return scene_idxs, label_weight
