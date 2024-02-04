from __future__ import division, print_function, absolute_import
import os
import copy
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
import re
import glob
import os.path as osp
import warnings
import pickle

class IncrementalSamples4LTCC(IncrementalPersonReIDSamples):
    """
        LTCC dataset
    """
    dataset_dir = 'ltcc'
    def __init__(self, datasets_root, relabel=True, combineall = False):
        self.relabel = relabel
        self.combineall = combineall
        root = osp.join(datasets_root, self.dataset_dir)
        self.train_dir = osp.join(root, 'train.pkl')
        self.query_dir = osp.join(root, 'query.pkl')
        self.gallery_dir = osp.join(root, 'gallery.pkl')
        self.train = self.process_dir(self.train_dir, relabel=True)
        self.query = self.process_dir(self.query_dir, relabel=False)
        self.gallery = self.process_dir(self.gallery_dir, relabel=False)
    
    def process_dir(self, dir_path, relabel=False):
        with open(dir_path, 'rb') as f:
            data = pickle.load(f)
        pid_container = set()
        for sample in data:
            p_id = sample['p_id']
            pid_container.add(p_id)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data_ = []
        for sample in data:
            img_path = sample['img_path']
            p_id = sample['p_id']
            camid = sample['cam_id']
            if relabel: 
                pid = pid2label[p_id]
            data_.append([img_path, pid, camid, 'ltcc', pid])
        
        return data_


