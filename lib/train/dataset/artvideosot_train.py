# 文件路径: lib/train/dataset/artvideosot_train.py

import os
import torch
import pandas
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import numpy as np

class ArtVideoSOTTrain(BaseVideoDataset):
    """ ArtVideoSOT dataset for training.
    The dataset is assumed to be organized as follows by your script:
    - root_path
        - train
            - list.txt
            - sequence_name_1
                - groundtruth.txt
                - 00000001.jpg
                - ...
        - val
            - list.txt
            - ...
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train'):
        """
        args:
            root - path to the artvideo_sot dataset.
            image_loader (jpeg4py_loader) - The function to read the images.
            split - 'train' or 'val'.
        """
        root = env_settings().artvideo_sot_train_dir if root is None else root
        super().__init__('ArtVideoSOT', root, image_loader)

        self.split = split
        self.root = os.path.join(self.root, split)
        self.sequence_list = self._get_sequence_list()

    def get_name(self):
        return 'artvideo_sot_train'

    def _get_sequence_list(self):
        list_file_path = os.path.join(self.root, 'list.txt')
        with open(list_file_path) as f:
            sequence_list = f.read().splitlines()
        return sequence_list

    def _read_bb_anno(self, seq_path):
        anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_name):
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = self._get_sequence_path(seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(self.sequence_list[seq_id])
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = {}
        return frame_list, anno_frames, object_meta