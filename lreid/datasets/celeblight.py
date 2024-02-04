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

class IncrementalSamples4CelebLight(IncrementalPersonReIDSamples):
    """
        Celebrities-ReID-Light dataset
    """
    dataset_dir = 'Celeb-reID'
    def __init__(self, datasets_root, **kwargs):
        self.dataset_dir = osp.join(datasets_root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        
        self._check_before_run()

        train = self._process_dir_train()
        query, gallery = self._process_dir_test()
        
        self.train = train
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

    def _process_dir_train(self):
        img_paths = glob.glob(osp.join(self.train_dir, '*.jpg'))
        img_paths.sort()
        pattern = re.compile(r'(\d+)_(\d+)_(\d+)')

        pid_container = set()
        
        for img_path in img_paths:
            pid, camid, _ = pattern.search(img_path).groups()
            pid, camid = int(pid), int(camid)
            pid_container.add(pid)
        
        pid_container = sorted(pid_container)
        
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        

        num_pids = len(pid_container)
        

        dataset = []
        
        for img_path in img_paths:
            pid, camid = pattern.search(img_path).groups()
            
            pid, camid = int(pid), int(camid)
            camid -= 1 # index starts from 0
            pid = pid2label[pid]
            
            dataset.append([img_path, pid, camid, 'celeb-light', pid])
            
        
        num_imgs = len(dataset)

        return dataset

    def _process_dir_test(self):
        query_img_paths = glob.glob(osp.join(self.query_dir, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(self.gallery_dir, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern = re.compile(r'(\d+)_(\d+)_(\d+)')

        pid_container = set()
        clothes_container = set()
        for img_path in query_img_paths:
            pid, camid, _ = pattern.search(img_path).groups()
            
            pid, camid = int(pid), int(camid)
            pid_container.add(pid)
            
        for img_path in gallery_img_paths:
            pid, camid, _ = pattern.search(img_path).groups()
            
            pid, camid = int(pid), int(camid)
            pid_container.add(pid)
            
        pid_container = sorted(pid_container)

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            pid, camid, _ = pattern.search(img_path).groups()
            
            pid, camid = int(pid), int(camid)
            camid -= 1 # index starts from 0
            
            query_dataset.append([img_path, pid, camid, 'celeb-light', pid])

        for img_path in gallery_img_paths:
            pid, camid, _ = pattern.search(img_path).groups()
            
            pid, camid = int(pid), int(camid)
            camid -= 1 # index starts from 0
            
            gallery_dataset.append([img_path, pid, camid, 'celeb-light', pid])

        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset
