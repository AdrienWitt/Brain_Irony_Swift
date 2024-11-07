import os
import torch
from torch.utils.data import Dataset, IterableDataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

import logging

class BaseDataset(Dataset):
    def __init__(self,  **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.data = self._set_data(self.subject_dict, self.max_length)
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
        
    def load_random_sequence(self, data, file, subject, target):
        if self.use_contrastive:    
            same_subject_target_files = [t for t in data if t[1] == subject and t[6] == target and t[2] != file]
            if same_subject_target_files:
                rand_file = random.choice(same_subject_target_files)
                return (rand_file[2], rand_file[4])
    
    def pad_back_forward(self, length, desired_length):
        pad_back = (desired_length-length)//2
        pad_forward = desired_length-length-pad_back
        return (pad_back, pad_forward)
    
    def pad_to_max(self, file):

        y = torch.load(file).unsqueeze(0)
        y_x, y_y, y_z, y_t = y.size(1), y.size(2), y.size(3), y.size(4)
        x_back, x_for = self.pad_back_forward(y_x, self.img_size[0])
        y_back, y_for = self.pad_back_forward(y_y, self.img_size[1])
        z_back, z_for = self.pad_back_forward(y_z, self.img_size[2])
        t_back, t_for = self.pad_back_forward(y_t, self.max_length)
        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        y = torch.nn.functional.pad(y, (z_back, z_for, y_back, y_for, x_back, x_for, t_back, t_for), value=background_value)
        y = y.permute(0,2,3,4,1)
        return y
                    
    def _set_data(self, subject_dict, max_length):
        data = []
        for i, subject in enumerate(subject_dict):
            file_dicts = subject_dict[subject]
            for file_dict in file_dicts:
                subject = file_dict["subject"]
                file = file_dict["file"]
                file_type = file_dict["file_type"]
                sequence_length = file_dict["sequence_length"]
                target = file_dict["target"]
                label = file_dict["label"]
                                
                data_tuple = (i, subject, file, file_type, sequence_length, max_length, target, label)
                data.append(data_tuple)
                                    
        if self.train : 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject, file, file_type, sequence_length, max_length, target, label = self.data[index]
        if self.use_contrastive:
            y = self.pad_to_max(file)
            rand_file, rand_sequence_length = self.load_random_sequence(self.data, file, subject, target)
            rand_y = self.pad_to_max(rand_file)
            return {
            "fmri_sequence": (y, rand_y),
            "subject_name": subject,
            "target": target,
            "label": label}
        else:
            y = self.pad_to_max(file)
            return {
            "fmri_sequence": y,
            "subject_name": subject,
            "target": target,
            "label": label}
            
    def __len__(self):
        return  len(self.data)
    
    

