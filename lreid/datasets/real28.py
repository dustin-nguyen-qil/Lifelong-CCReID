from __future__ import division, print_function, absolute_import
import os
import copy
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
import re
import glob
import os.path as osp
import warnings
import pickle
import numpy as np
import random

class IncrementalSamples4Real28(IncrementalPersonReIDSamples):
    """
        Real28 dataset, only used for testing
    """
    dataset_dir = 'Real28'
    def __init__(self, datasets_root, **kwargs):
        self.dataset_dir = osp.join(datasets_root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        
        self._check_before_run()

        query, gallery = self._process_dir_test()

        self.query = query
        self.gallery = gallery

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_test(self):
        query_img_paths = glob.glob(osp.join(self.query_dir, '*.jpeg'))
        gallery_img_paths = glob.glob(osp.join(self.gallery_dir, '*.jpeg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern = re.compile(r'(\d+)_(\d+)_(\d+)_(\d+)')

        pid_container = set()
        clothes_container = set()
        for img_path in query_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            query_dataset.append([img_path, pid, camid, 'real28', pid])

        for img_path in gallery_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            gallery_dataset.append([img_path, pid, camid, 'real28', pid])
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset

